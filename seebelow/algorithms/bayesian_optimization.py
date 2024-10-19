import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from seebelow.algorithms.gp import gp_posterior, SquaredExpKernel
from seebelow.algorithms.grid import Grid, GridMap2D
from seebelow.algorithms.gui import HeatmapAnimation
from seebelow.utils.constants import HISTORY_DTYPE


class BayesianOptimization:
    def __init__(self, grid: Grid, kernel: SquaredExpKernel):
        self.grid = grid
        self.grid_mean = np.zeros(grid.shape)
        self.kernel = kernel

    def get_optimal_state(self):
        X_visited = self.grid.X_visited
        y_visited = self.grid.grid[X_visited[:, 0], X_visited[:, 1]]
        new_states = self.grid.unvisited_states()

        mean_s, var_s = gp_posterior(new_states, X_visited, y_visited, self.kernel)
        EPS = 0.01
        sigma = np.sqrt(var_s)
        z = (mean_s - y_visited.max() - EPS) / sigma
        ei_term = (mean_s - y_visited.max()) * norm.cdf(z) + sigma * norm.pdf(z)

        assert np.all(np.sqrt(var_s) >= 0)

        ei_term[np.sqrt(var_s) == 0] = 0

        # print("Y_MAX", y_visited.max())
        # print("EI_STD", ei_term.std())
        # print("EI_MEAN", ei_term.mean())

        # print("num of maxes", len(ei_term[ei_term >= ei_term.max()]))
        idx = np.argmax(ei_term)
        # print(idx)

        # update grid_mean

        # print("mean_s", mean_s.shape)
        new_states = np.einsum("ijk->ik", new_states)
        self.grid_mean[new_states[:, 0], new_states[:, 1]] = mean_s.flatten()
        self.grid_mean[X_visited[:, 0], X_visited[:, 1]] = y_visited

        return tuple(new_states[idx].flatten())


def add_spots(grid_size, num_spots, spot_intensity, variance):
    data = np.zeros(grid_size)  # Start with a grid of zeros

    # Add random high expectation spots
    for _ in range(num_spots):
        # Choose a random location for the spot
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        data[x, y] = spot_intensity  # Set the spot intensity

    # Apply Gaussian filter for smoothing
    smoothed_data = gaussian_filter(data, sigma=variance)
    # print(smoothed_data.max())
    return smoothed_data


if __name__ == "__main__":
    grid_size = (20, 20)
    gt_grid = add_spots(grid_size, 1, 10, 3.0)
    # gt_grid += np.random.normal(0, 0.01, grid_size)
    gt_grid[gt_grid < 0] = 0
    gt_grid[gt_grid > 10] = 10
    gt_grid = gt_grid / gt_grid.max()  # Normalize
    grid = GridMap2D(*grid_size)
    kernel = SquaredExpKernel(scale=1.162)
    bo = BayesianOptimization(grid, kernel)

    saved_posterior_means = []

    # random sample from grid
    x_next = grid.sample_uniform()
    y = gt_grid[x_next]
    grid.update(x_next, y)
    for i in range(100):
        x_next = bo.get_optimal_state()
        saved_posterior_means.append(
            np.array(
                (np.array(x_next), bo.grid_mean.copy()), dtype=HISTORY_DTYPE(grid_size)
            )
        )
        grid.update(x_next, y)
        y = gt_grid[x_next]
    ani = HeatmapAnimation(np.array(saved_posterior_means), ground_truth=gt_grid)
    ani.visualize()
