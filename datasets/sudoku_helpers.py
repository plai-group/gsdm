import sudokulib
from sudokulib.constants import GRID_TOTAL
from sudokulib.layer import Layer
from sudokulib.backtracking import BacktrackingSolver
from sudokulib.grid import StringGrid, BLOCK_WIDTH, GRID_TOTAL, GRID_WIDTH, INVALID_GRID_SIZE, INVALID_GRID_PUZZLE, InvalidGrid
from sudokulib.solver import SudokuSolver

from time import time

VOID_SOLUTION = ' ' * GRID_TOTAL


class MultiSolutionGrid(StringGrid):
    def validate(self):
        """
        Taken from https://github.com/Fantomas42/sudoku-solver/blob/develop/sudokulib/grid.py
        but we don't check for there being enough clues to allow a unique solution.
        """
        if len(self) != GRID_TOTAL:
            raise InvalidGrid(INVALID_GRID_SIZE)

        v = []
        for y in range(GRID_WIDTH):
            for x, c in enumerate(
                    self.data[y * GRID_WIDTH:(y + 1) * GRID_WIDTH]):
                for k in x, y + GRID_WIDTH, (x / BLOCK_WIDTH, y / BLOCK_WIDTH):
                    if c != self.mystery_char:
                        v.append((k, c))
        if len(v) > len(set(v)):
            raise InvalidGrid(INVALID_GRID_PUZZLE)

        return True

class SingleSolutionBacktrackingSolver(BacktrackingSolver):
    def solve(self, layer):
        """
        Taken from solve definition at
        https://github.com/Fantomas42/sudoku-solver/blob/8d5e55984480abc30874ad763106d2bff41d9ee2/sudokulib/backtracking.py
        and added a single line. """
        layer = self.preprocess(layer)

        solutions = []
        missings_str = ''.join(layer.table)
        missings = missings_str.count(layer.mystery_char)

        if missings == 1:
            missing_index = missings_str.index(layer.mystery_char)
            missing_candidates = layer._candidates[missing_index]
            return [[missing_index, missing_candidates.pop()]] if len(missing_candidates) > 0 else []

        candidate_indexes = []
        for i in range(GRID_TOTAL):
            candidates = layer._candidates[i]
            if candidates:
                candidate_indexes.append((len(candidates), i))

        if not candidate_indexes:
            return None

        best_index = sorted(candidate_indexes)[0][1]

        for candidate in layer._candidates[best_index]:
            if hasattr(self, 'timeout_at') and time() > self.timeout_at:
                break
            layer_table = layer.table[:]
            layer_table[best_index] = str(candidate)
            solution_str = ''.join(layer_table)
            new_layer = Layer(solution_str, VOID_SOLUTION)
            new_solution = self.solve(new_layer)
            if not new_solution:
                continue
            else:
                solutions = [[best_index, candidate]]
                solutions.extend(new_solution)
                return solutions  # ADDED THIS - return a single solution

        return solutions


def make_timed_solver_class(timeout_at):
    class TimedSolver(SingleSolutionBacktrackingSolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.timeout_at = timeout_at
    return TimedSolver


def has_at_least_one_solution(grid, max_runtime=10):
    start = time()
    timeout_at = start + max_runtime
    try:
        solver = SudokuSolver(grid, grid_class=MultiSolutionGrid,
                              backtracking_solver_class=make_timed_solver_class(timeout_at))
        solver.run()
        if time() < timeout_at:
            has_solutions = solver.grid.completed
        else:
            has_solutions = None
    except sudokulib.grid.InvalidGrid:
        has_solutions = False
    return has_solutions, time()-start
