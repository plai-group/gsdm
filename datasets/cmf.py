import itertools as it
import torch
import numpy as np
from numpy.random import randint
from .graphical_dataset import GraphicalDataset
import matplotlib.pyplot as plt
import functools
from .utils import RNG

@functools.lru_cache(maxsize=1000, typed=False)
def cmf_shared_var_positions(t, m, n, fit_intermediate):
    def make_indices(r, c, extra_dim=1):
        arr = np.array([[[[i, j, k] for k in range(extra_dim)] for j in range(c)] for i in range(r)])
        return [list(ind) for ind in arr.reshape(-1, 3)]
    return make_indices(t, m) + make_indices(n, m) + (make_indices(t, n, m) if fit_intermediate else []) + make_indices(t, n)


class CMF(GraphicalDataset):

    def __init__(self, config, is_test):
        super().__init__(config, is_test)
        self.n = self.config.data.n
        self.m = self.config.data.m
        self.t = self.config.data.t
        self.valid_n = self.n
        self.valid_m = self.m
        self.valid_t = self.t
        if config.data.vary_dimensions:
            self.n_max = self.config.data.max_n
            self.m_max = self.config.data.max_m
            self.t_max = self.config.data.max_t

        self.generated_items = 0

    @property
    def n_cont(self):
        K_dim = self.n*self.m*self.t if self.fit_intermediate else 0
        return self.n*self.m + self.m*self.t + K_dim + self.n*self.t

    @property
    def shared_var_embeds(self):
        R_dim = self.t*self.n
        A_dim = self.t*self.m
        K_dim = self.t*self.m*self.n
        E_dim = self.n*self.m
        return [0]*A_dim + [1]*E_dim + ([2]*K_dim if self.fit_intermediate else []) + [3 if self.fit_intermediate else 2]*R_dim

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
        if self.fit_intermediate:
            R = torch.randn(self.t, self.n)
            A = torch.randn(self.t, self.m)
            K = torch.einsum('tn,tm->tnm', R, A)
            E = torch.einsum('tnm->nm', K)
            return A, E, K, R
        else:
            R = torch.randn(self.t, self.n)
            A = torch.randn(self.t, self.m)
            E = torch.einsum('tn,tm->nm', R, A)
            return A, E, R

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
            A, E, K, R = self.A_E_K_R()
            cont = torch.cat([A.flatten(), E.flatten(), K.flatten(), R.flatten()], dim=0)
        else:
            A, E, R = self.A_E_K_R()
            cont = torch.cat([A.flatten(), E.flatten(), R.flatten()], dim=0)
        disc = torch.tensor([])
        return cont, disc, (self.n_cont, self.n_discrete_options, self.shared_var_embeds), \
            (self.m, self.n, self.t)


    @property
    def n_discrete_options(self):
        return []

    def faithful_inversion_edges(self):
        M, N, T = self.m, self.n, self.t
        R_dim = T*N
        A_dim = T*M
        K_dim = T*M*N
        E_dim = N*M
        edges = set()

        if not self.fit_intermediate:
            for t, n, m in it.product(range(T), range(N), range(M)):
                A_index = t*M + m
                R_index = A_dim + E_dim + t*N + n
                E_index = A_dim + n*M + m
                for edge in [(E_index, A_index), (A_index, E_index),
                             (E_index, R_index), (R_index, E_index)]:
                    edges.add(edge)

            return edges

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
        A_dim = self.t*self.m
        E_dim = self.n*self.m
        K_dim = self.t*self.m*self.n if self.fit_intermediate else 0
        R_dim = self.t*self.n

        A = samples_cont[:, :A_dim].reshape(-1, self.t, self.m)
        E = samples_cont[:, A_dim:A_dim+E_dim].reshape(-1, self.n, self.m)
        R = samples_cont[:, A_dim+E_dim+K_dim:A_dim+E_dim+K_dim+R_dim].reshape(-1, self.t, self.n)
        E_hat = torch.einsum('btn,btm->bnm', R, A)
        return A, E, R, E_hat

    def validation_metrics(self, samples_disc, samples_cont, **kwargs):
        _, E, _, E_hat = self.A_E_R_Ehat_from_samples(samples_disc, samples_cont)
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
           'E_std': torch.std(torch.stack(Es), dim=0).mean().item()}

    @property
    def shared_var_positions(self):
        return cmf_shared_var_positions(t=self.t, m=self.m, n=self.n, fit_intermediate=self.fit_intermediate)


    def set_dims(self, dims):
        self.m, self.n, self.t = dims

    def set_dims_to_max(self):
        self.n = self.n_max
        self.m = self.m_max
        self.t = self.t_max

    def sample_obs_mask(self, B, device, obs_prop=None, exclude_mask=None):
        m, n, t = self.m, self.n, self.t

        R_dim = t*n
        A_dim = t*m
        K_dim = t*m*n if self.fit_intermediate else 0
        E_dim = n*m
        emb = torch.zeros((B, A_dim+E_dim+K_dim+R_dim, 1), device=device)   #  A, E, R
        emb[:, :(A_dim+E_dim)] = 1.
        cont_dim = A_dim + E_dim + K_dim + R_dim
        xt = emb[:, :, 0]
        return {"emb": emb, "xt": xt}

    def condition_batch(self, batch, obs_mask):
        # set E
        B = batch[0].shape[0]
        m, n, t = self.valid_m, self.valid_n, self.valid_t
        A_dim = m*t
        E_dim = m*n
        NI = 5
        Im = 1.0*torch.eye(m, n).reshape(1, E_dim).expand(2*NI, E_dim)
        batch[0][1::2] = batch[0][::2]
        batch[0][:2*NI, A_dim:A_dim+E_dim] = Im
        return batch, obs_mask

    def plot(self, samples_cont, samples_disc, obs_mask, **kwargs):
        A, E, R, E_hat = self.A_E_R_Ehat_from_samples(samples_disc, samples_cont)
        B = len(samples_cont)
        fig, axes = plt.subplots(nrows=B, ncols=4)
        axes[0, 0].set_title('A')
        axes[0, 1].set_title('R')
        axes[0, 2].set_title('Ê')
        axes[0, 3].set_title('E')
        for ax_row, a, e, r, e_hat in zip(axes, A, E, R, E_hat):
            kwargs = {'vmin': 0., 'vmax': E.max(), 'cmap': 'binary'}
            ax_row[0].imshow(a, **kwargs)
            ax_row[1].imshow(r, **kwargs)
            ax_row[2].imshow(e_hat, **kwargs)
            ax_row[3].imshow(e, **kwargs)
        for ax in np.array(axes).flatten():
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)
        return fig, ax_row[0]
