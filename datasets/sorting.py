import itertools as it
import torch
import numpy as np
from numpy.random import randint
from .graphical_dataset import GraphicalDataset
import matplotlib.pyplot as plt
import functools
from .utils import RNG


@functools.lru_cache(maxsize=1000, typed=False)
def sorting_faithful_inversion_edges(n, nc, fit_intermediate, sparsity_mask_index):
    """
    sparsity_mask_index:
        - 0 for connecting to above and below
        - 1 for connecting to everything
        - 2 for connecting to nothing
        - 3 for random
        - 4 for not symmetrizing
    """

    s = lambda i: i + n
    u = lambda i: i
    P = lambda i, j: nc + i*n + j
    K = lambda i, j: 2*n + i*n + j

    if sparsity_mask_index == 3:
        n_rows = 2*n + (1 + fit_intermediate) * n**2
        edges = []
        for r in range(n_rows):
            edges.extend([(r, c) for c in np.random.choice(n_rows, n//2, replace=False)])
        # symmetrize
        edges += [(a, b) for (b, a) in edges]
        return  set(edges)

    edges = []
    # connect all sorted elements
    condition = {
        0: (lambda i, j: (i == j + 1 or i == j - 1 or i == j)),
        1: (lambda i, j: True),
        2: (lambda i, j: (i == j)),
        4: (lambda i, j: (i == j + 1 or i == j - 1 or i == j))
    }[sparsity_mask_index]

    edges += [(s(i), s(j))
              for i in range(n)
              for j in range(n)
              if condition(i, j)]
    # connect all unsorted elements to themselves
    edges += [(u(i), u(i))
              for i in range(n)]
    # connect all rows/columns in P
    edges += [(P(i, j), P(i, k)) for i in range(n) for j in range(n) for k in range(n)]
    edges += [(P(i, j), P(k, j)) for i in range(n) for j in range(n) for k in range(n)]

    if fit_intermediate:
        # connect each element of u to column of K
        edges += [(u(i), K(j, i)) for i in range(n) for j in range(n)]
        # and each element of s to column of K
        edges += [(K(i, j), s(i)) for i in range(n) for j in range(n)]
        # connect each element in K to the respective element in P
        edges += [(P(i, j), K(i, j)) for i in range(n) for j in range(n)]
    else:
        # and each element of s to column of P
        edges += [(P(i, j), s(i)) for i in range(n) for j in range(n)]
        # and connect all of u to all of s
        edges += [(u(i), s(j)) for i in range(n) for j in range(n)]

    # symmetrize
    if sparsity_mask_index == 4:
        edges = [(a, b) for (b, a) in edges]
    else:
        edges += [(a, b) for (b, a) in edges]
    return  set(edges)


class Sorting(GraphicalDataset):

    def __init__(self, config, is_test):
        super().__init__(config, is_test)
        self.n = self.config.data.n
        self.fit_intermediate = self.config.data.fit_intermediate
        if config.data.vary_dimensions:
            self.n_max = self.config.data.max_n

    @property
    def n_cont(self):
        n = self.n
        return n*2 + (n**2 if self.fit_intermediate else 0)

    @property
    def shared_var_embeds(self):
        n = self.n
        if self.config.model.use_shared_var_positions:
            return [0]*n + [1]*n + ([2]*(n**2) if self.fit_intermediate else []) + [3 if self.fit_intermediate else 2]*(n**2)
        else:
            return [n+0]*n + list(range(n)) + ([n+1]*(n**2) if self.fit_intermediate else []) + [n+2 if self.fit_intermediate else n+1]*(n**2)

    @property
    def shared_var_positions(self):
        n = self.n
        n2_indices = [[i, j] for i in range(n) for j in range(n)]
        n1_indices = [[i, 0] for i in range(n)]
        if self.config.model.use_shared_var_positions:
            return n1_indices + n1_indices + (n2_indices if self.fit_intermediate else []) + n2_indices
        else:
            raise Exception

    def vary_dimensions(self):
        if self.config.data.vary_dimensions:
            self.n = randint(2, self.n_max)

    def set_dims_to_max(self):
        self.n = self.n_max

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

        n = self.n
        u = torch.randn(n)
        u_sort = torch.sort(u)
        P = torch.zeros(n, n)
        K = torch.zeros(n, n) if self.fit_intermediate else torch.tensor([])
        for (i, j) in enumerate(u_sort.indices):
            P[i, j] = 1
            if self.fit_intermediate:
                K[i, j] = u[j]

        cont = torch.cat([u, u_sort.values, K.flatten()], dim=0)
        disc = P.flatten().long()

        return cont, disc, (self.n_cont, self.n_discrete_options, self.shared_var_embeds), \
            (self.n, )

    @property
    def n_discrete_options(self):
        n = self.n
        return [2]*(n**2)

    def faithful_inversion_edges(self):
        return sorting_faithful_inversion_edges(n=int(self.n), nc=int(self.n_cont), fit_intermediate=bool(self.fit_intermediate), 
                                                sparsity_mask_index=self.config.data.sparsity_mask_index)

    def validation_metrics(self, samples_disc, samples_cont, **kwargs):
        B, n2 = samples_cont.shape
        n = self.n
        nc = self.n_cont
        u = samples_cont[:, :n]
        ground = torch.sort(u)
        P = samples_disc.reshape(B, n, n)
        approx = torch.argmax(P, dim=2)
        sort_matches = (ground.indices == approx).sum()/(self.n*B)
        ddpm_sort = torch.gather(u, 1, approx)
        return {'sort_matches': sort_matches,
           'sort_completed': (ground.indices == approx).all().item(),
           'rmse': ((ground.values - ddpm_sort)**2).mean().sqrt()}

    def set_dims(self, dims):
        self.n = dims[0]

    def sample_obs_mask(self, B, device, obs_prop=None, exclude_mask=None):
        n = self.n

        cont_dim = self.n_cont
        emb = torch.zeros((B, cont_dim+n**2, 1), device=device)
        emb[:, :n] = 1. # unsorted part
        xt = torch.cat([emb[:, :cont_dim, 0], emb[:, cont_dim:, 0], emb[:, cont_dim:, 0]], dim=1)
        return {"emb": emb, "xt": xt}

    def plot(self, samples_cont, samples_disc, obs_mask):
        B = len(samples_cont)
        n = self.n
        fig, axes = plt.subplots(nrows=3, ncols=1)
        x_cont = samples_cont[0]
        x_disc = samples_disc[0]
        u_sort = torch.sort(x_cont[:n])
        axes[0].set_title("Sorted (ground truth)")
        axes[0].imshow(u_sort.values.reshape(1, -1), vmin=-3.0, vmax=3.0, cmap='gray')
        P_ = x_disc.reshape(n, n).float()
        x_sort = P_ @ x_cont[:n]
        axes[1].set_title("Sorting process")
        axes[1].imshow(x_sort.reshape(1, -1), vmin=-3.0, vmax=3.0,
                       cmap='gray')
        axes[2].set_title("")
        axes[2].imshow((x_sort != u_sort.values).reshape(1, -1), vmin=0.0, vmax=5.0,
                       cmap='hot')

        for ax in np.array(axes).flatten():
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)
        return fig, axes[0]
