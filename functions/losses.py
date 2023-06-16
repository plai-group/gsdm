import torch
from .utils import DiffusionProcess


def noise_estimation_loss(model,
                        x0: torch.Tensor,
                        t: torch.LongTensor,
                        e: torch.Tensor,
                        b: torch.Tensor,
                        w: torch.Tensor,
                        predict,
                        obs_mask,
                        keepdim=False,
                        log_attn=False,
                        mean_over_latents=False,
                        loss_mask=None,
                        regression=False):
    a = (1-b).cumprod(dim=0).index_select(0, t)
    while a.ndim < x0.ndim:
        a = a.unsqueeze(-1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    if regression:
        x = torch.zeros_like(x0)
        t = torch.zeros_like(t)

    kwargs = {}
    if model.module.config.model.conditional != 'not':
        kwargs = {**kwargs, 'obs_mask': obs_mask, 'obs': obs_mask["xt"]*x0}
    output = model(x, t.float(), log_attn=log_attn, **kwargs)

    target = {'eps': e, 'x0': x0}[predict]
    loss_per_item = (target - output).square()
    if loss_mask is not None:
        loss_per_item = loss_per_item * loss_mask
    if mean_over_latents:
        loss_per_item = loss_per_item.flatten(start_dim=1).mean(dim=1)
    else:
        loss_per_item = loss_per_item.flatten(start_dim=1).sum(dim=1)
    if w is not None:
        loss_per_item = loss_per_item * w.index_select(0, t) * len(t)
    if keepdim:
        return loss_per_item
    else:
        return loss_per_item.mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
