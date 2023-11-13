import numpy as np
from scipy.stats import norm
from rpal.algorithms.gp import GP, SquaredExpKernel
from rpal.algorithms.grid import SurfaceGridMap, GridMap2D
from rpal.algorithms.gui import HeatmapAnimation

from scipy.ndimage import gaussian_filter


class BayesianOptimization:
    def __init__(self, grid: SurfaceGridMap, gp: GP):
        self.gp = gp
        self.grid = grid

        self.y_max = None

    def get_optimal_state(
        self, prev_x_hat: np.ndarray, prev_y: float, normalized=False
    ):

        self.y_max = max(self.y_max, prev_y) if self.y_max else prev_y

        self.gp.add_sample(prev_x_hat, prev_y)
        all_states = self.grid.vectorized_states
        mean_s, var_s = self.gp.posterior(all_states)
        z = (mean_s - self.y_max) / np.sqrt(var_s)
        ei_term = (mean_s - self.y_max) * norm.cdf(z) + np.sqrt(var_s) * norm.pdf(z)

        assert np.all(np.sqrt(var_s) >= 0)

        ei_term[np.sqrt(var_s) == 0] = 0

        idx = np.argmax(ei_term)
        return list(self.grid.vectorized_states[idx].flatten()), mean_s


def add_spots(grid_size, num_spots, spot_intensity, variance):
    data = np.zeros(grid_size)  # Start with a grid of zeros

    # Add random high expectation spots
    for _ in range(num_spots):
        # Choose a random location for the spot
        x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
        data[y, x] = spot_intensity  # Set the spot intensity

    # Apply Gaussian filter for smoothing
    smoothed_data = gaussian_filter(data, sigma=variance)
    return smoothed_data


if __name__ == "__main__":

    grid_size = (100, 100)
    grid = GridMap2D(*grid_size)

    grid._grid = add_spots(grid_size, 5, 2, 10)

    gp = GP(SquaredExpKernel(scale=0.5))
    bo = BayesianOptimization(grid, gp)

    saved_posterior_means = []

    # random sample from grid
    x_next = grid.sample_states_uniform()
    y = grid[x_next]
    gp.add_sample(x_next, y)
    x_next = grid.sample_states_uniform()
    y = grid[x_next]
    for i in range(100):
        x_next, mean_s = bo.get_optimal_state(x_next, y, normalized=False)
        saved_posterior_means.append((x_next, mean_s))
        y = grid[x_next]

    ani = HeatmapAnimation(saved_posterior_means)

    ani.save_animation()
