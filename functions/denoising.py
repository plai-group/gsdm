import torch
from .utils import DiffusionProcess, gaussian_analytical_kl, gaussian_log_prob


def compute_alpha(beta, t, dims_like=torch.tensor([])):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)  # I believe this line could be removed if the "t+1" below is changed to "t"
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1)
    while a.ndim < dims_like.ndim:
        a = a.unsqueeze(-1)
    return a

def ddim_steps(x, seq, model, b, predict, obs=None, obs_mask=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long(), x)
            at_next = compute_alpha(b, next_t.long(), x)
            xt = xs[-1].to(x.device)
            output = model(xt, t, obs=obs, obs_mask=obs_mask)
            if predict == 'eps':
                et = output
                x0_pred = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * et
            elif predict == 'x0':
                x0_pred = output
                et = (xt - x0_pred * at.sqrt()) / (1 - at).sqrt()
            x0_preds.append(x0_pred.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_pred + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def ddpm_steps(x, seq, model, b, config, obs=None, obs_mask=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long(), x)
            atm1 = compute_alpha(betas, next_t.long(), x)
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)
            output = model(x, t.float(), obs_mask=obs_mask, obs=obs)

            if config.model.predict == 'eps':   # TODO replace with process.get_x0
                x0_pred = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * output
            elif config.model.predict == 'x0':
                x0_pred = output

            x0_preds.append(x0_pred.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_pred + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1)
            while mask.ndim < x.ndim:
                mask = mask.unsqueeze(-1)
            logvar = beta_t.log()
            # print('sampling logvar')
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


@torch.no_grad()
def ddpm_elbo(model,
              x0: torch.Tensor,
              b: torch.Tensor, T,
              n_cont, config,
              obs_mask=None):
    """
    b is a vector of beta_1 to beta_T
    Indices are shifted relative to paper. We sum the:
    - likelihood for x_{-1} given x_0
    - KLs for each x_{t-1} given x_t for t in [1, T-2] inclusive
    - KL between Gaussian and x_{T-1}
    """
    device = b.device
    process = DiffusionProcess(b, T)
    obs_mask_dict = {'xt': None, 'emb': None} if obs_mask is None else obs_mask
    def apply_mask(x, mask, slice_mask_to=None, slice_mask_from=None):
        if mask is None:
            return x
        if slice_mask_to is not None:
            mask = mask[:, :slice_mask_to]
        if slice_mask_from is not None:
            mask = mask[:, slice_mask_from:]
        return x * (1-mask.unsqueeze(1))

    B, *data_dims = x0.shape
    x0 = x0.view(B, 1, *data_dims).to(device)
    kwargs = dict(leading_dims=1, trailing_dims=len(data_dims))

    # compute various factors and sample xt -----------------------------------------------
    forward_scaling = process.get('forward_q_scaling', **kwargs)
    forward_logvar = process.get('forward_q_logvar', **kwargs)
    scaling_p1, logvar_p1 = forward_scaling[:, 1:], forward_logvar[:, 1:]   # q(x_t|x0) for t=1,...,T
    xtp1s = scaling_p1 * x0 + (0.5*logvar_p1).exp() * torch.randn(size=(B, T, *data_dims), device=device)
    tp1s = torch.arange(1, T+1).view(1, T).expand(B, T).to(device)
    output = []
    for i in range(T): # tp1, xtp1 in zip(tp1s, xtp1s):
        output.append(model(xtp1s[:, i], tp1s[:, i], obs=x0[:, 0], obs_mask=obs_mask))
    # stack again
    output = torch.stack(output, dim=1)
    assert output.shape == (B, T, *data_dims)
    if config.model.predict == 'eps':
        x0_from_tp1 = (1./scaling_p1) * (xtp1s - output*(0.5*logvar_p1).exp())
    elif config.model.predict == 'x0':
        x0_from_tp1 = output

    # means for predicted x_0,...,x_{T-1}
    reverse_mean_x0_scaling = process.get('reverse_mean_x0_scaling', **kwargs)
    reverse_mean_xtp1_scaling = process.get('reverse_mean_xtp1_scaling', **kwargs)
    p_mean = x0_from_tp1 * reverse_mean_x0_scaling + xtp1s * reverse_mean_xtp1_scaling
    q_mean = x0 * reverse_mean_x0_scaling + xtp1s * reverse_mean_xtp1_scaling
    p_logvar = process.get('reverse_p_logvar', **kwargs)
    q_logvar = process.get('reverse_q_logvar', **kwargs)

    # evaluate sum of KL losses for predicting x_1,...,x_{T-1}
    kls = gaussian_analytical_kl(mu1=q_mean[:, 1:], mu2=p_mean[:, 1:],
                                 logsigma1=0.5*q_logvar[:, 1:], logsigma2=0.5*p_logvar[:, 1:])
    assert kls.shape == (B, T-1, *data_dims)
    kl_sum = apply_mask(kls, obs_mask_dict["xt"]).flatten(start_dim=1).sum(dim=1)

    # evaluate likelihood for x0
    # TODO tidy
    assert len(data_dims) == 1
    D = data_dims[0]
    likelihood = gaussian_log_prob(x=x0[:, :, :n_cont], mu=p_mean[:, :1, :n_cont], logsigma=0.5*p_logvar[:, :1, :n_cont])
    likelihood = apply_mask(likelihood, obs_mask_dict["xt"], slice_mask_to=n_cont).flatten(start_dim=1).sum(dim=1)
    x0_onehot = x0[:, :, n_cont:]
    x0_probs = p_mean[:, :1, n_cont:]
    disc_likelihood = apply_mask(x0_onehot * x0_probs.log().clamp(min=-1e10), obs_mask_dict["xt"], slice_mask_from=n_cont).flatten(start_dim=1).sum(dim=1)
    likelihood = likelihood + disc_likelihood
    # likelihood for discrete variables

    # compute KL at T
    q_mean = forward_scaling[:, -1:] * x0
    q_logvar = forward_logvar[:, -1:]
    p_mean = torch.zeros_like(q_mean)
    p_logvar = torch.zeros_like(q_logvar)
    kl_T = gaussian_analytical_kl(mu1=q_mean, mu2=p_mean, logsigma1=0.5*q_logvar, logsigma2=0.5*p_logvar)
    kl_T = apply_mask(kl_T, obs_mask_dict["xt"]).flatten(start_dim=1).sum(dim=1)

    elbo = likelihood - kl_sum - kl_T
    return {'elbo': elbo.mean().item(), 'likelihood': likelihood.mean().item(), 'kl_sum': kl_sum.mean().item(), 'kl_T': kl_T.mean().item(), 'disc_likelihood': disc_likelihood.mean().item()}
