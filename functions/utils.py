import numpy as np
import torch


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def gaussian_log_prob(x, mu, logsigma):
    return -logsigma - 0.5*torch.tensor(2*np.pi).log() - 0.5 * ( (x - mu) / logsigma.exp() ) ** 2


class DiffusionProcess():
    def __init__(self, b: torch.Tensor, T):
        """
        b: length T tensor of beta_1,...,beta_T
        """
        assert len(b) == T
        t = torch.arange(0, T+1).view(T+1).to(b.device)
        self.beta = torch.cat([torch.zeros(1).to(b.device), b], dim=0)
        self.alpha = (1 - self.beta).cumprod(dim=0).index_select(0, t).view(-1)  # this is alpha_bar in DDPM notation

        at = self.alpha[:-1]
        atp1 = self.alpha[1:]
        btp1 = self.beta[1:]

        # logvars for each x_t given x_{t+1}
        self.reverse_p_logvar = (1 - atp1/at).log()               # the DDIM paper calls this noisy DDPM, denoted with \hat{\sigma}
        self.reverse_q_logvar = (btp1 * (1-at) / (1-atp1)).log()  # from DDPM's equation 6

        # such that index t corresponds to scaling of x0 (or x_{t+1}) in q(x_t|x_{t+1},x_0)
        self.reverse_mean_x0_scaling = (at.sqrt() * btp1) / (1 - atp1)               # DDPM equation 7
        self.reverse_mean_xtp1_scaling = (1 - btp1).sqrt() * (1 - at) / (1 - atp1)   # DDPM equation 7

        # compute logvars and mean scalings such that index t corresponds to processes q(x_t|x_0)
        self.forward_q_scaling = self.alpha.sqrt()    # DPPM equation 4
        self.forward_q_logvar = (1-self.alpha).log()  # DPPM equation 4

    def add_trailing_dims(self, params, dims_like):
        assert params.ndim == 1
        return params.view(-1, *(1,)*(dims_like.ndim-1))

    def get(self, name, t=None, leading_dims=0, trailing_dims=0):
        attr = getattr(self, name)
        if t is not None:
            attr = attr.index_select(dim=0, index=t)
        attr = attr.view(*(1,)*leading_dims, -1, *(1,)*trailing_dims)
        return attr

    def get_x0(self, network_output, t, xt, predict):
        if predict == 'x0':
            return network_output
        elif predict == 'eps':
            at = self.at.index_select(t)
            return (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * network_output
        elif predict == 'discrete':
            return network_output

    @property
    def elbo_weighting(self):   # TODO generalise for predicting things other than e
        """
        Return weights for MSE loss (on predicted epsilon) at each time t so that optimising the loss
        corresponds to optimising the ELBO.
        """
        sigma_p = (0.5*self.reverse_p_logvar).exp()
        w1 = (1 - self.alpha[1]) / (2*self.alpha[1]*sigma_p[0]**2)

        atm1 = self.alpha[1:-1]
        at = self.alpha[2:]
        bt = self.beta[2:]
        # index i of wt corresponds to loss term with x_{i+2}
        wt = (atm1*bt**2) / (2*sigma_p[1:]**2*(1-at)*at)
        return torch.cat([w1.view(1), wt])
