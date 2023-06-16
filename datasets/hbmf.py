import itertools as it
import torch
import numpy as np
from numpy.random import randint
from .graphical_dataset import GraphicalDataset
import matplotlib.pyplot as plt
import functools
from .utils import RNG

@functools.lru_cache(maxsize=1000, typed=False)
def bmf_shared_var_positions(t, m, n, fit_intermediate):
    def make_indices(r, c, extra_dim=1):
        arr = np.array([[[[i, j, k] for k in range(extra_dim)] for j in range(c)] for i in range(r)])
        return [list(ind) for ind in arr.reshape(-1, 3)]
    return make_indices(t, m) + make_indices(n, m) + (make_indices(t, n, m) if fit_intermediate else []) + make_indices(t, n) + make_indices(1, 1) + (make_indices(t, 1) if fit_intermediate else [])


class HBMF(GraphicalDataset):

    def __init__(self, config, is_test):
        super().__init__(config, is_test)
        self.n = self.config.data.n
        self.m = self.config.data.m
        self.t = self.config.data.t
        if config.data.vary_dimensions:
            self.n_max = self.config.data.max_n
            self.m_max = self.config.data.max_m
            self.t_max = self.config.data.max_t

        self.generated_items = 0

    @property
    def n_cont(self):
        K_dim = self.n*self.m*self.t if self.fit_intermediate else 0
        return self.n*self.m + self.m*self.t + K_dim

    @property
    def shared_var_embeds(self):
        R_dim = self.t*self.n
        A_dim = self.t*self.m
        K_dim = self.t*self.m*self.n
        E_dim = self.n*self.m

        if self.fit_intermediate:
            nonzero_row_indicators = [5]*self.t if self.config.model.use_shared_var_positions else list(range(5, 5+self.t))
        else:
            nonzero_row_indicators = []

        return [0]*A_dim + \
               [1]*E_dim + \
               ([2]*K_dim if self.fit_intermediate else []) + \
               [3 if self.fit_intermediate else 2]*R_dim + \
               [4 if self.fit_intermediate else 3] + \
               nonzero_row_indicators

    @property
    def shared_var_positions(self):
        return bmf_shared_var_positions(t=self.t, m=self.m, n=self.n, fit_intermediate=self.fit_intermediate)

    def vary_dimensions(self):
        if self.config.data.vary_dimensions:
            too_large = int(1e16)
            n = too_large
            m = too_large
            t = too_large
            i = 0 # rejection counts
            K_dims = n*m*t if self.fit_intermediate else 0
            while (K_dims + m*n + m*t + n*t)*max(m, n) > self.config.model.max_attn_matrix_size:
                if i == 100:
                    print("Warning: rejected over 100 dimension combinations.")
                    print("m, n, t: ", m, n, t)
                i += 1
                t = randint(1, self.t_max)
                n = randint(t, self.n_max)
                m = randint(t, self.m_max)
                K_dims = n*m*t
            self.n = n
            self.m = m
            self.t = t

    def A_E_K_R(self):
        rank_minus_1 = torch.randint(self.t, size=())
        row_nonzero = (rank_minus_1 >= torch.arange(self.t))
        def generate_nonzero_row():
            row = (torch.rand(1, self.n) < 0.3).float()
            return row if row.sum() > 0 else generate_nonzero_row()
        def generate_zero_row():
            return torch.zeros(1, self.n)
        R = torch.cat([generate_nonzero_row() if row_nonzero[i] else generate_zero_row() for i in range(self.t)], dim=0)
        A = torch.rand(self.t, self.m) * row_nonzero.float().unsqueeze(1)
        if self.fit_intermediate:
            K = torch.einsum('tn,tm->tnm', R, A)
            E = torch.einsum('tnm->nm', K)
            return A, E, K, R, rank_minus_1, row_nonzero
        else:
            E = torch.einsum('tn,tm->nm', R, A)
            return A, E, R, rank_minus_1

    def set_dims_to_max(self):
        self.n = self.n_max
        self.m = self.m_max
        self.t = self.t_max

    def __getitem__(self, index):
        if self.finite_length:
            seed = index + 2**31 if self.is_test else index
            with RNG(seed):
                return self.__unseeded_getitem__(index)
        else:
            return self.__unseeded_getitem__(index)

    def __unseeded_getitem__(self, index):
        # first batch is always at maximum initial size
        if self.config.data.vary_dimensions and index % self.config.training.batch_size == 0 and index > 0:
            self.vary_dimensions()
        self.generated_items += 1


        if self.fit_intermediate:
            A, E, K, R, rank_minus_1, row_nonzero = self.A_E_K_R()
            cont = torch.cat([A.flatten(), E.flatten(), K.flatten()], dim=0)
            disc = torch.cat([R.flatten().long(), rank_minus_1.long().unsqueeze(0), row_nonzero.long()], dim=0)
        else:
            A, E, R, rank_minus_1 = self.A_E_K_R()
            cont = torch.cat([A.flatten(), E.flatten()], dim=0)
            disc = torch.cat([R.flatten().long(), rank_minus_1.long().unsqueeze(0)], dim=0)
        return cont, disc, (self.n_cont, self.n_discrete_options, self.shared_var_embeds), \
            (self.m, self.n, self.t)


    @property
    def n_discrete_options(self):
        return [2]*self.t*self.n + [self.t] + ([2]*self.t if self.fit_intermediate else [])

    def faithful_inversion_edges(self):
        M, N, T = self.m, self.n, self.t
        R_dim = T*N
        A_dim = T*M
        K_dim = T*M*N
        E_dim = N*M

        edges = set()

        if not self.fit_intermediate:
            rank_index = A_dim + E_dim + R_dim
            for t in range(T):
                for n in range(N):
                    R_index = A_dim + E_dim + t*N + n
                    for edge in [(R_index, rank_index), (rank_index, R_index)]:
                        edges.add(edge)
                for m in range(M):
                    A_index = t*M + m
                    for edge in [(A_index, rank_index), (rank_index, A_index)]:
                        edges.add(edge)

            for t, n, m in it.product(range(T), range(N), range(M)):
                A_index = t*M + m
                R_index = A_dim + E_dim + t*N + n
                E_index = A_dim + n*M + m
                for edge in [(E_index, A_index), (A_index, E_index),
                             (E_index, R_index), (R_index, E_index)]:
                    edges.add(edge)

            return edges

        # now for the fit_intermediate case
        rank_index = A_dim + E_dim + K_dim + R_dim
        row_nonzero_index = lambda i: rank_index + 1 + i
        for t in range(T):
            for edge in [(rank_index, row_nonzero_index(t)), (row_nonzero_index(t), rank_index)]:
                edges.add(edge)
            for n in range(N):
                R_index = A_dim + E_dim + K_dim + t*N + n
                for edge in [(R_index, row_nonzero_index(t)), (row_nonzero_index(t), R_index)]:
                    edges.add(edge)
            for m in range(M):
                A_index = t*M + m
                for edge in [(A_index, row_nonzero_index(t)), (row_nonzero_index(t), A_index)]:
                    edges.add(edge)

        for t, n, m in it.product(range(T), range(N), range(M)):
            K_index = A_dim + E_dim + t*N*M + n*M + m
            A_index = t*M + m
            R_index = A_dim + E_dim + K_dim + t*N + n
            for edge in [(K_index, A_index), (A_index, K_index),
                         (K_index, R_index), (R_index, K_index)]:
                edges.add(edge)

            E_index = A_dim + n*M + m
            K_indices = [A_dim + E_dim + t_*N*M + n*M + m for t_ in range(T)]
            for index in K_indices:
                edges.add((E_index, index))
                edges.add((index, E_index))

        return edges

    def A_E_R_Ehat_from_samples(self, samples_disc, samples_cont):
        dim_R = self.t*self.n
        R = samples_disc[:, :dim_R].reshape(-1, self.t, self.n).float()
        dim_A = self.t*self.m
        dim_E = self.n*self.m
        A = samples_cont[:, :dim_A].reshape(-1, self.t, self.m)
        E = samples_cont[:, dim_A:dim_A+dim_E].reshape(-1, self.n, self.m)
        E_hat = torch.einsum('btn,btm->bnm', R, A)
        return A, E, R, E_hat

    def validation_metrics(self, samples_disc, samples_cont, gt_cont=None, gt_disc=None):
        m, n, t = self.m, self.n, self.t
        R_dim = t*n
        A_dim = t*m
        K_dim = t*m*n if self.fit_intermediate else 0
        E_dim = n*m
        gt_rank_minus_1 = gt_disc[:, R_dim]
        pred_rank_minus_1 = samples_disc[:, R_dim]
        rank_pred_acc = (gt_rank_minus_1 == pred_rank_minus_1).float().mean()
        A, E, R, E_hat = self.A_E_R_Ehat_from_samples(samples_disc, samples_cont)
        nonzero_rows = (R != 0).any(dim=2).sum(dim=1)
        rank_pred_consistency = (nonzero_rows == (gt_rank_minus_1 + 1)).float().mean()
        rmse = ((E - E_hat)**2).mean().sqrt()
        max_norm = (E - E_hat).abs().max()
        Es = []
        rmses = []
        max_norms = []
        for i in range(50):
            E1 = self.A_E_K_R()[1]
            E2 = self.A_E_K_R()[1]
            Es += [E1, E2]
            rmses.append(((E1 - E2)**2).mean().sqrt())
            max_norms.append((E1 - E2).abs().max())


        return {'rmse': rmse,
           'max_norm': max_norm,
           'avg_rmse': np.mean(rmses),
           'std_rmse': np.std(rmses),
           'avg_max_norm': np.mean(max_norms),
           'std_max_norm': np.std(max_norms),
           'E_std': torch.std(torch.stack(Es), dim=0).mean().item(),
           'rank_pred_acc': rank_pred_acc,
           'rank_pred_consistency': rank_pred_consistency,
        }

    def set_dims(self, dims):
        self.m, self.n, self.t = dims

    def sample_obs_mask(self, B, device, obs_prop=None, exclude_mask=None):
        m, n, t = self.m, self.n, self.t

        R_dim = t*n
        A_dim = t*m
        K_dim = t*m*n if self.fit_intermediate else 0
        E_dim = n*m
        row_nonzero_dim = self.t if self.fit_intermediate else 0
        emb = torch.zeros((B, A_dim+E_dim+K_dim+R_dim+1+row_nonzero_dim, 1), device=device)   #  A, E, R
        emb[:, A_dim:(A_dim+E_dim)] = 1.
        cont_dim = A_dim + E_dim + K_dim
        start_rank = cont_dim + R_dim
        start_row_nonzero = start_rank + 1
        xt = torch.cat([emb[:, :cont_dim, 0], emb[:, cont_dim:, 0], emb[:, cont_dim:, 0]], dim=1)
        xt = [emb[:, :cont_dim, 0],] + \
             [emb[:, cont_dim:start_rank, 0],]*2 + \
             [emb[:, start_rank:start_row_nonzero, 0],]*self.t + \
             [emb[:, start_row_nonzero:, 0],]*2
        xt = torch.cat(xt, dim=1)
        return {"emb": emb, "xt": xt}

    def plot(self, samples_cont, samples_disc, obs_mask, **kwargs):
        A, E, R, E_hat = self.A_E_R_Ehat_from_samples(samples_disc, samples_cont)
        A = A.detach().cpu().numpy()
        E = E.detach().cpu().numpy()
        R = R.detach().cpu().numpy()
        E_hat = E_hat.detach().cpu().numpy()
        B = len(samples_cont)
        fig, axes = plt.subplots(nrows=B, ncols=4)
        axes[0, 0].set_title('A')
        axes[0, 1].set_title('R')
        axes[0, 2].set_title('Ê')
        axes[0, 3].set_title('E')
        for ax_row, a, e, r, e_hat in zip(axes, A, E, R, E_hat):
            # delete zero rows from A and R
            a = a[r.sum(axis=1) > 0]
            r = r[r.sum(axis=1) > 0]
            kwargs = {'vmin': 0., 'vmax': E.max(), 'cmap': 'binary'}
            ax_row[0].imshow(a, **kwargs)
            ax_row[1].imshow(r, **kwargs)
            ax_row[2].imshow(e_hat, **kwargs)
            ax_row[3].imshow(e, **kwargs)
        for ax in np.array(axes).flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, ax_row[0]
