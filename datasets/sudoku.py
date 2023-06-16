import itertools as it
import torch
import numpy as np
#from torch.utils.data import Dataset
from .graphical_dataset import GraphicalDataset
import matplotlib.pyplot as plt
try:
    from .sudoku_helpers import has_at_least_one_solution
except ModuleNotFoundError:
    print("Can't import .sudoku_helpers, likely because sudoku-solver not installed. Oh well.")
from .utils import RNG
import functools



def sudoku_generate(gridTotalCount, gridBlockSize1, gridBlockSize2, gridBlockCount):
    gridBlockSize = [gridBlockSize1, gridBlockSize2]
    grid = torch.zeros(gridTotalCount).long()
    options = np.random.permutation(list(range(gridBlockCount)))

    # generate [1,2,3]. is used to offset a row of block in other blocks
    shiftsY = np.random.permutation(range(gridBlockSize[0]))

    # generate [1,2,3], [4,5,6], [7,8,9] and shuffles per block. is used to offset columns
    shiftsX = []
    for shift2 in range(gridBlockSize[1]):
        values = np.random.permutation(range(gridBlockSize[0]))
        for v in values:
            shiftsX.append(shift2 * gridBlockSize[0] + v)

    # generate [1,2,3], [4,5,6], [7,8,9] and shuffles per block. is used to offset rows
    shiftsXY = []
    for shift2 in range(gridBlockSize[0]):
        values = np.random.permutation(range(gridBlockSize[1]))
        for v in values:
            shiftsXY.append(shift2 * gridBlockSize[1] + v)

    # generate [1,2,3], [4,5,6], [7,8,9] and shuffles per block. is used to offset rows
    shiftsXY2 = []
    for shift2 in range(gridBlockSize[0]):
        values = np.random.permutation(range(gridBlockSize[1]))
        for v in values:
            shiftsXY2.append(shift2 * gridBlockSize[1] + v)

    for x in range(gridBlockCount):
        row = 0
        for shift1 in range(gridBlockSize[0]):
            for shift2 in range(gridBlockSize[1]):
                foo = (shiftsXY[row]) * gridBlockCount + shiftsX[x]
                grid[foo] \
                    = options[(x + shiftsY[shift1] + shift2*gridBlockSize[0]) % gridBlockCount]
                row = row + 1
    return grid


class Sudoku(GraphicalDataset):
    n_cont = 0

    def __init__(self, config, is_test):
        super().__init__(config, is_test)
        seed = 1
        self.seed = seed
        self.type = config.data.type
        types = [[2,2],[2,3],[2,4],[2,5],[2,6],[3,2],[3,3],[3,4],[3,5],[3,6],
                 [4,2],[4,3],[4,4],[4,5],[4,6],[5,2],[5,3],[5,4],[5,5],[5,6],
                 [6,2],[6,3],[6,4],[6,5],[6,6]]
        self.gridBlockSize = [types[self.type][0], types[self.type][1]]
        self.gridBlockCount = self.gridBlockSize[0] * self.gridBlockSize[1]
        self.gridTotalCount = self.gridBlockCount * self.gridBlockCount

        n_elems_hori, n_elems_vert = self.gridBlockSize
        n_rows = n_cols = n_elems_hori * n_elems_vert
        #difficulty = 0.8 # min=0.6, max=0.95, step=0.05
        #self.difficulty = difficulty
        showSolution = 0 # min=0, max=1, step=1
        self.showSolution = showSolution

        # 0...gridBlockCount
        self.checks = list(range(self.gridBlockCount))

    @property
    def shared_var_embeds(self):
        if self.config.model.use_shared_var_positions:
            return [0] * self.gridTotalCount
        else:
            assert self.gridBlockSize[0] == self.gridBlockSize[1]
            three = self.gridBlockSize[0]
            return sum([[i]*three for i in range(self.gridTotalCount//three)], [])

    @property
    def shared_var_positions(self):
        return [[i, j] for i in range(self.gridBlockCount) for j in range(self.gridBlockCount)]

    # generate a complete sudoku grid
    # https://turtletoy.net/turtle/5098380d82
    # based on <https:#gamedev.stackexchange.com/a/138228>
    def __getitem__(self, index):
        if self.finite_length:
            seed = index + 2**31 if self.is_test else index
            with RNG(seed):
                s = sudoku_generate(self.gridTotalCount, *self.gridBlockSize, self.gridBlockCount)
        else:
            s = sudoku_generate(self.gridTotalCount, *self.gridBlockSize, self.gridBlockCount)
        return torch.zeros(size=(0,)), s

    def avg_log_prob(self, N=None):
        # https://en.wikipedia.org/wiki/Mathematics_of_Sudoku
        sudoku_log_prob = -torch.tensor(6670903752021072936960.).log()
        return sudoku_log_prob

    @property
    def n_discrete_options(self):
        return [self.gridBlockCount] * self.gridTotalCount

    def constraint_edges(self):
        edges = []
        n_elems_hori, n_elems_vert = self.gridBlockSize
        n_rows = n_cols = n_elems_hori * n_elems_vert
        for row_start in range(0, n_rows*n_cols, n_cols):
            row_end = row_start + n_cols
            edges.extend([(i, j)
                          for i in range(row_start, row_end)
                          for j in range(row_start, row_end)
                          if i != j])
        for col_start in range(0, n_cols):
            col_end = col_start + n_cols * (n_rows-1) + 1
            edges.extend([(i, j)
                          for i in range(col_start, col_end, n_cols)
                          for j in range(col_start, col_end, n_cols)
                          if i != j])
        n_blocks_vert, n_blocks_hori = self.gridBlockSize
        for block_i, block_j in it.product(range(n_blocks_vert), range(n_blocks_hori)):
            r_start = block_i * n_elems_vert
            c_start = block_j * n_elems_hori
            block_elem_coords = [(i, j)
                                 for i in range(r_start, r_start+n_elems_vert)
                                 for j in range(c_start, c_start+n_elems_hori)]
            block_indices = [i*n_cols+j for i, j in block_elem_coords]
            edges.extend([(i, j)
                          for i in block_indices
                          for j in block_indices
                          if i != j])
        return set(edges)

    def faithful_inversion_edges(self):
        edges = []
        n_elems_hori, n_elems_vert = self.gridBlockSize
        n_rows = n_cols = n_elems_hori * n_elems_vert
        for row_start in range(0, n_rows*n_cols, n_cols):
            row_end = row_start + n_cols
            edges.extend([(i, j)
                          for i in range(row_start, row_end)
                          for j in range(row_start, row_end)
                          if i != j])
        for col_start in range(0, n_cols):
            col_end = col_start + n_cols * (n_rows-1) + 1
            edges.extend([(i, j)
                          for i in range(col_start, col_end, n_cols)
                          for j in range(col_start, col_end, n_cols)
                          if i != j])
        n_blocks_vert, n_blocks_hori = self.gridBlockSize
        for block_i, block_j in it.product(range(n_blocks_vert), range(n_blocks_hori)):
            r_start = block_i * n_elems_vert
            c_start = block_j * n_elems_hori
            block_elem_coords = [(i, j)
                                 for i in range(r_start, r_start+n_elems_vert)
                                 for j in range(c_start, c_start+n_elems_hori)]
            block_indices = [i*n_cols+j for i, j in block_elem_coords]
            edges.extend([(i, j)
                          for i in block_indices
                          for j in block_indices
                          if i != j])
        sym_edges = [(j, i) for i, j in edges]
        edges.extend(sym_edges)
        return set(edges)

    def condition_batch(self, batch, obs_mask):
        shape = batch[1].shape
        batch[1][:] = batch[1][0].expand(*shape)
        return batch, obs_mask

    def validation_metrics(self, samples_disc, samples_cont, **kwargs):
        HW = samples_disc.shape[1]
        H = int(HW**0.5)
        sudokus = samples_disc[:, :].reshape(-1, H, H)
        rows = sudokus.reshape(-1, H)
        cols = sudokus.permute(0, 2, 1).reshape(-1, H)
        def prop_correct(blocks):
            prop = 0
            for block in blocks:
                nz = [i for i in block if i != 0]
                contains = set(int(i) for i in nz)
                is_correct = float(len(contains) == len(nz))
                prop += is_correct/len(blocks)
            return prop
        constraints_violated = 0.
        for v1, v2 in self.faithful_inversion_edges():
            equal = samples_disc[:, v1] == samples_disc[:, v2]
            constraints_violated += equal
        sudokus_correct = 0
        sudokus_timed_out = 0
        total_solve_time = 0
        return {'rows_correct': prop_correct(rows),
                'cols_correct': prop_correct(cols),
                'constraints_violated': constraints_violated.mean(),
                'sudokus_without_clashes': (constraints_violated==0.).float().mean(),
                'sudokus_correct': sudokus_correct/len(sudokus),
                'sudokus_timed_out': sudokus_timed_out/len(sudokus),
                'avg_solve_time': total_solve_time /len(sudokus)}

    def sample_obs_mask(self, B, device, obs_prop=None, exclude_mask=None,
                      batch_dims = None):
        emb = torch.zeros((B, self.gridTotalCount, 1), device=device)
        for row in emb:
            n_obs = np.random.randint(self.gridTotalCount) if obs_prop is None else int(self.gridTotalCount*obs_prop)
            row[np.random.choice(self.gridTotalCount, n_obs, replace=False)] = 1.
        if exclude_mask is not None:
            emb = emb * (1-exclude_mask["emb"])
        xt = emb[:, :, 0].repeat_interleave(self.gridBlockCount, dim=1)
        return {"emb": emb, "xt": xt}

    def plot(self, samples_cont, samples_disc, obs_mask, **kwargs):
        if self.type != 6:
            fig, ax = plt.subplots()
            return fig, ax
        sample_none_mask = lambda x: self.sample_obs_mask(x.shape[0], x.device, 0.)
        if obs_mask is None:
            obs_mask = sample_none_mask(samples_cont)
        B = len(samples_disc)
        fig, axes = plt.subplots(ncols=B, figsize=(B*self.gridBlockCount, self.gridBlockCount))
        for ax, samples_disc_i, obs_mask_emb in zip(axes, samples_disc, obs_mask['emb'].squeeze(-1)):
            grid = 1 + samples_disc_i.cpu().numpy().reshape(self.gridBlockCount, self.gridBlockCount)
            colors = np.ones(tuple(samples_disc_i.shape) + (3,))
            MANUAL_MAX = 15
            max_block_violations = np.zeros((3,3)) + MANUAL_MAX
            max_row_violations = np.zeros(9) + MANUAL_MAX
            max_col_violations = np.zeros(9) + MANUAL_MAX
            block_violations = np.zeros((3,3))
            row_violations = np.zeros(9)
            col_violations = np.zeros(9)
            for v1, v2 in self.constraint_edges():
                if samples_disc_i[v1] == samples_disc_i[v2]:
                    if v1 % 9 == v2 % 9: # same col
                        colors[v1] += (1, 0.5, 0.5)
                        colors[v2] += (1, 0.5, 0.5)
                        col_violations[v1 % 9] += 1
                    if v1//9 == v2//9: # same row
                        colors[v1] += (0.5, 1, 0.5)
                        colors[v2] += (0.5, 1, 0.5)
                        row_violations[v1//9] += 1
                    if v1//(9*3) == v2//(9*3) and (v1//3) % 3 == (v2//3) % 3: # same block
                        colors[v1] += (0.5, 0.5, 1.)
                        colors[v2] += (0.5, 0.5, 1.)
                        block_violations[v1//(9*3), (v1//3) % 3] += 1
                    # renormalize
                    colors[v1] *= 2/np.sum(colors[v1])
                    colors[v2] *= 2/np.sum(colors[v2])
            table = ax.table(grid, loc='center', cellLoc='center')
            for is_obs, color, ((row, col), cell) in zip(obs_mask_emb.cpu(), colors, table.get_celld().items()):
                cell.set_height(.1)
                cell.set_width(.1)
                cell.set_fontsize(22)
                cell.set_color((0., 0., 0., 0.1)) # light grey
                cell.set_edgecolor('k')
                if is_obs:
                    cell.set_text_props(fontweight=1000)
                else:
                    cell.set_text_props(fontweight=100)
            ax.axis('off')
            ax.axis('tight')
            ax.set_aspect(1)

            ax_pos = ax.get_position()
            margin_x, margin_y = 0.018, 0.035 # ax.margins()
            x0 = ax_pos.x0 + margin_x
            x1 = ax_pos.x1 - margin_x
            y0 = ax_pos.y0 + margin_y
            y1 = ax_pos.y1 - margin_y
            table_width = (x1 - x0)
            table_height = (y1 - y0)
            col_width = table_width/9
            row_height = table_height/9
            block_width = table_width/3
            block_height = table_height/3
            for i in range(3):
                for j in range(3):
                    if block_violations[j,i] > 0:
                        y_pos = lambda j: (2 - j)
                        alpha = min(block_violations[j,i]/max_block_violations[j,i], 1) * 0.8
                        rect = plt.Rectangle(
                            (x0 + i * block_width,
                             y0 + y_pos(j) * block_height), block_width, block_height, fill=True, color=(1, 0, 0, alpha), lw=1,
                            zorder=-500, transform=fig.transFigure, figure=fig)
                        fig.patches.extend([rect])
            for i in range(9):
                alpha = min(col_violations[i]/max_col_violations[i], 1) * 0.8
                if col_violations[i] > 0:
                    rect = plt.Rectangle(
                        (x0 + i * col_width, y0), col_width, table_height, fill=True, color=(1, 0, 0, alpha), lw=1,
                        zorder=-500, transform=fig.transFigure, figure=fig)
                    fig.patches.extend([rect])
            for i in range(9):
                alpha = min(row_violations[i]/max_row_violations[i], 1) * 0.8
                if row_violations[i] > 0:
                    y_pos = lambda i: (8 - i)
                    rect = plt.Rectangle(
                        (x0, y0 + y_pos(i) * row_height), table_width, row_height, fill=True, color=(1, 0, 0, alpha), lw=1,
                        zorder=-500, transform=fig.transFigure, figure=fig)
                    fig.patches.extend([rect])

            # put 3x3 block grid on top
            for i in range(3):
                for j in range(3):
                    y_pos = lambda j: (2 - j)
                    rect = plt.Rectangle(
                        (x0 + i * block_width,
                         y0 + y_pos(j) * block_height), block_width, block_height, fill=False, color="k", lw=3,
                        zorder=1000, transform=fig.transFigure, figure=fig)
                    fig.patches.extend([rect])

        return fig, ax

