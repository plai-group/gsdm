import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time

def get_timestep_embedding(timesteps, embedding_dim, max_timesteps=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(max_timesteps) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, kernel_size=3, var_emb_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        if var_emb_channels is not None:
            self.var_proj = torch.nn.Conv2d(var_emb_channels,
                                            out_channels,
                                            kernel_size=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=kernel_size//2)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, var_emb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        if var_emb is not None:
            h = h + self.var_proj(var_emb)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=1, attn_dim_reduce=1):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads

        self.norm = Normalize(in_channels)
        # self.norm_out = Normalize(in_channels//attn_dim_reduce)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels//attn_dim_reduce,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def sparse_forward(self, x, sparse_attention_mask_and_indices):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, h*w)
        # beta = (int(c//heads)**(-0.5)) # standard attention scaling
        # we used unnormalized attention, it should not matter
        beta = 1
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)


        valid_indices_mask, attendable_indices = sparse_attention_mask_and_indices
        nq, max_attendable_keys = valid_indices_mask.shape
        attendable_indices = attendable_indices.view(1, 1, nq, max_attendable_keys)\
                                               .expand(b, heads, nq, max_attendable_keys)
        def get_keys_or_values(t, indices):
            *batch_shape, nd, nv = t.shape
            t = t.transpose(-1, -2)\
                .view(*batch_shape, nv, 1, nd)\
                .expand(*batch_shape, nv, max_attendable_keys, nd)
            index = indices.view(*batch_shape, nv, max_attendable_keys, 1)\
                .expand(-1, -1, -1, -1, c//heads)
            return t.gather(dim=2, index=index)

        attended_keys = get_keys_or_values(k, indices=attendable_indices)   # b x heads x h*w x max_attendable_keys x c
        attended_values = get_keys_or_values(v, indices=attendable_indices)

        weights = beta * torch.einsum('bhqkc,bhcq->bhqk', attended_keys, q)
        inf_matrix = torch.zeros_like(valid_indices_mask)
        inf_matrix[valid_indices_mask==0] = torch.inf
        weights = weights - inf_matrix.view(1, 1, nq, max_attendable_keys)
        weights = weights.softmax(dim=-1)

        h_ = torch.einsum('bhqk,bhqkc->bhqc', weights, attended_values)
        h_ = h_.permute(0, 3, 1, 2).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        out = x+h_
        return out, None

    def forward(self, x, sparsity_matrix=None, sparse_attention_mask_and_indices=None, return_w=False):

        if sparse_attention_mask_and_indices is not None:
            out, w_ = self.sparse_forward(x, sparse_attention_mask_and_indices)
            return out, w_ if return_w else out

        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, h*w)
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)

        w_ = torch.einsum('bhdk,bhdq->bhqk', k, q)
        w_ = w_ * (int(c//heads)**(-0.5))
        if sparsity_matrix is not None:
            inf_matrix = torch.zeros_like(sparsity_matrix)
            inf_matrix[sparsity_matrix==0] = torch.inf
            w_ = w_ - inf_matrix.view(-1, 1, h*w, h*w)
        w_ = torch.nn.functional.softmax(w_, dim=3)
        h_ = torch.einsum('bhdk,bhqk->bhdq', v, w_)
        h_ = h_.view(b, c, h, w)

        h_ = self.proj_out(h_)

        out = x+h_
        return out, w_ if return_w else out

class TransformerModel(nn.Module):
    def __init__(self, config, dataset, faithful_inversion_edges=None, sparse_attention_mask_and_indices=None):
        super().__init__()
        self.config = config
        # self.input_dim = config.data.dim
        self.dataset = dataset
        self.n_cont = dataset.n_cont
        self.n_discrete_options = dataset.n_discrete_options
        if self.config.data.vary_dimensions:
            self.dataset.set_dims_to_max()
        self.shared_var_embeds = dataset.shared_var_embeds if config.model.use_shared_var_embeds else slice(0, self.n_variables)
        self.emb_dim = self.config.model.emb_dim
        self.temb_dim = self.emb_dim
        self.num_transformers = config.model.num_transformers

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.emb_dim,
                            self.temb_dim),
            torch.nn.Linear(self.temb_dim,
                            self.temb_dim),
        ])

        self.cont_in_proj = nn.Conv1d(1, self.emb_dim, kernel_size=1)
        self.cont_out_proj = nn.Conv1d(self.emb_dim, 1, kernel_size=1)
        disc_in_projs = {}
        disc_out_projs = {}
        for n_options in set(self.n_discrete_options):
            disc_in_projs[str(n_options)] = nn.Conv1d(n_options, self.emb_dim, kernel_size=1)
            disc_out_projs[str(n_options)] = nn.Conv1d(self.emb_dim, n_options, kernel_size=1)
        self.disc_in_projs = nn.ModuleDict(disc_in_projs)
        self.disc_out_projs = nn.ModuleDict(disc_out_projs)

        if config.model.var_embedding:
            if config.model.use_shared_var_embeds:
                num_embeds = len(set(self.shared_var_embeds))
                self.var_embs = nn.Parameter(torch.randn(1,
                                                         self.emb_dim,
                                                         num_embeds,
                                                         1),
                                             requires_grad=True)
            else:
                if config.data.vary_dimensions:
                    raise Exception("Cannot vary dimensions with fixed variable embedding.")
                else:
                    self.var_embs = nn.Parameter(torch.randn(1, self.emb_dim,
                                                             self.n_variables, 1),
                                                 requires_grad=True)


        else:
            self.var_embs = None

        # dimension embedding, use small std, since we multiply by #n_vars
        self.demb = nn.Parameter(torch.randn(1, 1, self.emb_dim), requires_grad=False)

        if self.config.model.conditional == 'fixed':
            self.cond_embs = nn.Parameter(
                torch.randn(1, 1, self.emb_dim), requires_grad=True)

        transformers = []
        for i in range(self.num_transformers):
            transformers.append(
                AttnBlock(self.emb_dim, n_heads=self.config.model.n_heads,
                          attn_dim_reduce=self.config.model.attn_dim_reduce)
            )
        self.transformers = nn.ModuleList(transformers)

        self.res_blocks = nn.Sequential(*[ResnetBlock(in_channels=self.emb_dim, out_channels=self.emb_dim,
                                                      temb_channels=self.temb_dim, dropout=False,
                                                      var_emb_channels=self.emb_dim if self.config.model.var_embedding else None,
                                                      kernel_size=1)
                                          for _ in range(self.num_transformers)])

        assert self.config.model.impose_sparsity in ['sparse', 'not']
        if self.config.model.impose_sparsity == 'not':
            pass
        elif sparse_attention_mask_and_indices is None:
            self.faithful_inversion_matrix = self.make_faithful_inversion_matrix(faithful_inversion_edges)
            if self.config.data.save_sparsity_mask:
                np.savez(f"{self.config.data.dataset}_sparsity_mask.npz",
                         mask=self.faithful_inversion_matrix)
            wandb.log({'attn/sparsity': wandb.Image(self.faithful_inversion_matrix.cpu().numpy())})
            max_attendable_keys = self.faithful_inversion_matrix.sum(dim=1).max().int().item()
            self.valid_indices_mask, self.attendable_indices = (nn.Parameter(t, requires_grad=False) for t in self.faithful_inversion_matrix.topk(k=max_attendable_keys, dim=1))
        else:
            self.valid_indices_mask, self.attendable_indices = (nn.Parameter(t, requires_grad=False) for t in sparse_attention_mask_and_indices)

    def reset_dimensions(self, model_dims, data_dims, plot_mask=False, log_stats=False):
        n_cont, n_discrete_options, shared_var_embeds = model_dims
        self.n_cont = n_cont[0].item()
        self.n_discrete_options = [n[0].item() for n in n_discrete_options]
        self.shared_var_embeds = [so[0].item() for so in shared_var_embeds]

        start_t = time.time()
        if self.config.model.impose_sparsity == 'not':
            pass
        else:
            model_device = self.temb.dense[0].weight.device
            self.faithful_inversion_matrix = self.make_faithful_inversion_matrix(self.dataset.faithful_inversion_edges()).to(device=model_device)
            if plot_mask:
                wandb.log({'attn/sparsity': wandb.Image(self.faithful_inversion_matrix.cpu().numpy())})
            max_attendable_keys = self.faithful_inversion_matrix.sum(dim=1).max().int().item()
            self.valid_indices_mask, self.attendable_indices = (nn.Parameter(t.to(device=model_device), requires_grad=False) for t in self.faithful_inversion_matrix.topk(k=max_attendable_keys, dim=1))
        end_t = time.time()
        if log_stats:
            wandb.log({'faithful_mask_creation_time':  end_t - start_t})
            if self.config.model.impose_sparsity != 'not':
                wandb.log({'max_attendable_keys': max_attendable_keys})

    def reinit(self):
        sparsity_things = (self.valid_indices_mask, self.attendable_indices) if self.config.model.impose_sparsity == 'sparse' else None
        return TransformerModel(self.config, self.dataset, faithful_inversion_edges=None,
                                sparse_attention_mask_and_indices=sparsity_things)

    @property
    def n_variables(self):
        return self.n_cont + len(self.n_discrete_options)

    @property
    def x_dim(self):
        return self.n_cont + sum(self.n_discrete_options)

    def variable_emb_begin(self, var_index):
        if var_index < self.n_cont:
            return var_index
        else:
            return self.n_cont+sum(self.n_discrete_options[:var_index-self.n_cont])

    def variable_emb_dim(self, var_index):
        if var_index < self.n_cont:
            return 1
        else:
            return self.n_discrete_options[var_index-self.n_cont]

    def project_x_to_emb(self, x, cont_in_proj, disc_in_projs):
        B, _ = x.shape
        NC = self.n_cont
        if NC > 0:
            cont = x[:, :NC].view(B, 1, NC)
            cont_emb = cont_in_proj(cont)    # B x emb_dim x NC
            embs = [cont_emb]
        else:
            embs = []
        index = NC
        for n_options in set(self.n_discrete_options):
            n_var = sum(el == n_options for el in self.n_discrete_options)
            index_step = n_var*n_options
            values = x[:, index:index+index_step].view(B, n_var, n_options).permute(0, 2, 1)
            emb = disc_in_projs[str(n_options)](values)    # B x emb_dim x n_var
            embs.append(emb)
            index += index_step
        return torch.cat(embs, dim=2).permute(0, 2, 1) # B x n_var x emb_dim

    def project_emb_to_x(self, emb):
        B, *_ = emb.shape
        NC = self.n_cont
        if NC > 0:
            cont_emb = emb[:, :NC]
            cont_x = self.cont_out_proj(cont_emb.permute(0, 2, 1)).view(B, NC)
            xs = [cont_x]
        else:
            xs = []
        index = NC
        for n_options in set(self.n_discrete_options):
            n_var = sum(el == n_options for el in self.n_discrete_options)
            emb_bit = emb[:, index:index+n_var].permute(0, 2, 1)   # B x emb_dim x n_var
            x_bit = self.disc_out_projs[str(n_options)](emb_bit)
            if self.config.model.softmax:
                x_bit = torch.softmax(x_bit, dim=1)
            x_bit = x_bit.permute(0, 2, 1)  # B x n_var x n_options
            x_bit = x_bit.reshape(B, -1)
            xs.append(x_bit)
            index += n_var
        return torch.cat(xs, dim=1)


    def forward(self, x, t, obs_mask=None, obs=None, log_attn=False):
        B, N = x.shape
        NV = self.n_variables
        D = self.emb_dim

        # timestep embedding
        temb = get_timestep_embedding(t, self.emb_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # dimension embedding
        demb = self.demb.reshape(1, 1, D).expand(B, NV, D) * np.log(NV)

        if obs_mask is not None:
            x = x*(1-obs_mask["xt"]) + obs*obs_mask["xt"]

        emb = self.project_x_to_emb(x, self.cont_in_proj, self.disc_in_projs) # + demb
        assert emb.shape == (B, NV, D)

        if self.config.model.conditional == 'fixed':
            emb = emb + self.cond_embs * obs_mask["emb"]

        compute_attn_reward = self.config.model.attn_reg_lambda != 0
        if compute_attn_reward or log_attn:
            cumulative_w = torch.eye(NV, device=emb.device).view(1, NV, NV)
            attn_mu = self.config.model.attn_reg_mu

        if log_attn:
            self.logged_weight_matrices = {}

        if not self.config.model.resnet:
            emb = emb + temb.view(B, 1, D)
        for l, (res_block, transformer) in enumerate(zip(self.res_blocks, self.transformers)):
            emb = emb.permute(0, 2, 1).reshape(B, D, NV, 1)
            if self.config.model.resnet:
                # TODO reparametrize res_block for demb
                var_embs = self.var_embs[:, :, self.shared_var_embeds, :] if self.var_embs is not None else None
                if self.config.model.use_shared_var_positions:
                    positions = torch.tensor(self.dataset.shared_var_positions).to(emb.device)
                    n_dims = positions.shape[-1]
                    embedding_dims = [self.emb_dim//n_dims for dim in range(n_dims)]
                    embedding_dims[-1] = self.emb_dim - sum(embedding_dims[:-1])
                    position_embedding = []
                    for dim in range(n_dims):
                        position_embedding.append(get_timestep_embedding(timesteps=positions[:, dim], embedding_dim=embedding_dims[dim], max_timesteps=1000))
                    position_embedding = torch.cat(position_embedding, dim=1).unsqueeze(0).unsqueeze(-1).permute(0, 2, 1, 3)  # concatenate along channel dimension and add batch dimension plus redundant last dim
                    var_embs = var_embs + position_embedding
                emb = res_block(emb, temb, var_embs)
            if self.config.model.impose_sparsity == 'dense':
                attn_sparsity_matrix = torch.ones((B, NV, NV)).to(emb.device)
                attn_sparsity_matrix = attn_sparsity_matrix * self.faithful_inversion_matrix.unsqueeze(0).to(emb.device)
                emb, w = transformer(emb, return_w=True, sparsity_matrix=attn_sparsity_matrix)
            elif self.config.model.impose_sparsity == 'sparse':
                sparsity_things = (self.valid_indices_mask, self.attendable_indices)
                emb, w = transformer(emb, return_w=True,
                                     sparse_attention_mask_and_indices=sparsity_things)
            else:
                emb, w = transformer(emb, return_w=True)
            emb = emb.view(B, D, NV).permute(0, 2, 1)
            if (compute_attn_reward or log_attn) and w is not None:
                w = w.mean(dim=1)
                cumulative_w = attn_mu*torch.einsum('bxy,byz->bxz', cumulative_w, w) + (1-attn_mu)*cumulative_w
                if log_attn:
                    self.logged_weight_matrices[f'w_{l}'] = w.detach().mean(dim=0)

        if compute_attn_reward:
            self.attn_reward = self.config.model.attn_reg_lambda * (
                 self.faithful_inversion_matrix.unsqueeze(0) * cumulative_w.log()
            ).flatten(start_dim=1).mean(dim=1).mean(dim=0)
        if log_attn:
            self.logged_weight_matrices['w_cumulative'] = cumulative_w.detach().mean(dim=0)

        assert emb.shape == (B, NV, D)
        emb = self.project_emb_to_x(emb[:, :NV])

        assert emb.shape == (B, N)
        if obs_mask is not None:
            emb = emb * (1 - obs_mask["xt"]) + obs * obs_mask["xt"]

        return emb

    def make_faithful_inversion_matrix(self, edges):
        NV = self.n_variables
        matrix = torch.zeros((NV, NV))
        if edges is None:
            return matrix*0 + 1.
        for i, j in edges:
            matrix[i, j] = 1.
        for i in range(NV):
            matrix[i, i] = 1.
        return matrix

    def sample_xT(self, B):
        device = next(self.parameters()).device
        return torch.randn((B, self.x_dim), device=device)

    def dequantize(self, x):
        x_cont, x_disc = x

        B, cont_dim = x_cont.shape
        assert x_disc.shape == (B, len(self.n_discrete_options))

        log_prob = torch.zeros(B)
        dequants = []
        for dim, n_options in enumerate(self.n_discrete_options):
            val = x_disc[:, dim]
            dequants.append(F.one_hot(val, num_classes=n_options).float())
        if len(dequants) == 0:
            dequants = torch.zeros((B, 0))
        else:
            dequants = torch.cat(dequants, dim=1)

        return torch.cat([x_cont, dequants], dim=1), log_prob

    def requantize(self, x):
        n_discrete_options = self.n_discrete_options
        n_discrete_dims = sum(n_discrete_options)
        if n_discrete_dims == 0:
            return x, torch.zeros((x.shape[0], 0))
        x_cont, x_dequant = x[:, :-n_discrete_dims], x[:, -n_discrete_dims:]

        B, cont_dim = x_cont.shape
        assert x_dequant.shape == (B, n_discrete_dims)

        x_disc = []
        for i, n_options in enumerate(n_discrete_options):
            start, end = sum(n_discrete_options[:i]), sum(n_discrete_options[:i+1])
            dequant_chunk = x_dequant[:, start:end]
            x_disc.append(torch.argmax(dequant_chunk, dim=1))
        x_disc = torch.stack(x_disc, dim=1)

        return x_cont, x_disc
