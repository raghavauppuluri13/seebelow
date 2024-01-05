from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from rpal.algorithms.gp import GP, SquaredExpKernel
from rpal.algorithms.grid import Grid, GridMap2D, SurfaceGridMap
from rpal.algorithms.gui import HeatmapAnimation
from rpal.utils.constants import HISTORY_DTYPE


class BayesianOptimization:
    def __init__(self, grid: Grid, kernel):
        self.gp = GP(kernel)
        self.grid = grid
        self.grid_mean = np.zeros(grid.shape)
        self.y_max = None

    def get_optimal_state(self, prev_x_hat: Tuple[float, float], prev_y: float):
        self.y_max = max(self.y_max, prev_y) if self.y_max is not None else prev_y
        print(prev_y)
        self.gp.add_sample(prev_x_hat, prev_y)
        self.grid.grid[prev_x_hat[0], prev_x_hat[1]] = prev_y

        X_visited = np.array(self.gp.X)
        new_states = self.grid.unvisited_states(X_visited)
        new_states = new_states[:, np.newaxis, :]

        print("states", new_states.shape)

        mean_s, var_s = self.gp.posterior(new_states)
        EPS = 0.01
        sigma = np.sqrt(var_s)
        z = (mean_s - self.y_max - EPS) / sigma
        ei_term = (mean_s - self.y_max) * norm.cdf(z) + sigma * norm.pdf(z)

        assert np.all(np.sqrt(var_s) >= 0)

        ei_term[np.sqrt(var_s) == 0] = 0

        print("Y_MAX", self.y_max)
        print("EI_STD", ei_term.std())
        print("EI_MEAN", ei_term.mean())

        print("num of maxes", len(ei_term[ei_term >= ei_term.max()]))
        idx = np.argmax(ei_term)
        print(idx)

        # update grid_mean

        print("mean_s", mean_s.shape)
        new_states = np.einsum("ijk->ik", new_states)
        self.grid_mean[new_states[:, 0], new_states[:, 1]] = mean_s.flatten()
        self.grid_mean[X_visited[:, 0], X_visited[:, 1]] = np.array(self.gp.y)

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
    print(smoothed_data.max())
    return smoothed_data


if __name__ == "__main__":
    grid_size = (20, 20)
    grid = GridMap2D(*grid_size)
    spots = add_spots(grid_size, 1, 10, 3.0)
    grid.grid = spots
    grid.grid += np.random.normal(0, 0.01, grid_size)
    grid.grid[grid.grid < 0] = 0
    grid.grid[grid.grid > 10] = 10
    grid.grid = grid.grid / grid.grid.max()  # Normalize
    kernel = SquaredExpKernel(scale=2)
    bo = BayesianOptimization(grid, kernel)

    saved_posterior_means = []

    # random sample from grid
    x_next = grid.sample_uniform()
    y = grid[x_next]
    for i in range(100):
        x_next = bo.get_optimal_state(x_next, y)
        saved_posterior_means.append(
            np.array(
                (np.array(x_next), bo.grid_mean.copy()), dtype=HISTORY_DTYPE(grid_size)
            )
        )
        y = grid[x_next]
    ani = HeatmapAnimation(np.array(saved_posterior_means), ground_truth=grid.grid)
    ani.visualize()
