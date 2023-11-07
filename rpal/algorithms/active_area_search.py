"""
@brief
Implementation of Active Area Search from Ma, Y., Garnett, R. &amp; Schneider, J.. (2014). Active Area Search via Bayesian Quadrature. <i>Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 33:595-603 Available from https://proceedings.mlr.press/v33/ma14.html.
@author Raghava Uppuluri
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm
from rpal.algorithms.gp import GP
from rpal.algorithms.gp import SquaredExpKernel
from rpal.algorithms.cluster_store import SurfaceGridMap
from collections import defaultdict
from rpal.algorithms.quadtree import QuadTree


class ActiveAreaSearch:
    state_dim = 2

    def __init__(
        self,
        surface_grid: SurfaceGridMap,
        group_quadtree: QuadTree,
        kernel: SquaredExpKernel,
        threshold=7,
        confidence=0.6,
        noise_var=0.01,
    ):
        self.kernel = kernel
        self.X = []  # shape: (len(X), state_dim)
        self.Y = []  # shape: (len(Y), 1)
        self.grid = surface_grid
        self.group_quadtree = group_quadtree
        self.noise_var = noise_var
        self.threshold = threshold
        self.confidence = confidence

    def get_optimal_state(
        self, prev_x_hat: np.ndarray, prev_y: float, normalized=False
    ):
        """Gets optimal state given previous states and observed f(x) values
        prev_sample_x_hat: previous state shape=(state_dim,1)
        prev_sample_y: previous observed f(x)
        """
        assert prev_x_hat.shape == (self.state_dim,), print(prev_x_hat.shape)

        idx = self.surface_grid.normalize(x_hat) if not normalized else prev_x_hat

        self.X.append(idx)
        self.Y.append(prev_y)

        self.group_quadtree.insert(idx, len(self.X) - 1)

        X = np.array(self.X)
        y = np.array(self.Y)

        nx, ny = self.grid.shape
        x = np.arange(0, self.grid.shape[0])
        y = np.arange(0, self.grid.shape[1])
        X_grid = np.meshgrid(x, y)

        # vectorize reward computation over all states (S) in grid -> (S, 2)
        X_s = np.concatenate(
            [X_grid[0].reshape(-1, 1), X_grid[1].reshape(-1, 1)], axis=1
        )
        print(X_s.shape)
        X_hat = np.zeros((X_s.shape[0], X.shape[0] + 1, self.state_dim))
        X_hat[:, :-1, :] = X
        X_hat[:, -1, :] = X_s
        print(X_hat.shape)

        reward = np.zeros(X_hat.shape[0])
        w_g_s_hat = np.zeros_like(X_s)

        for group, group_X_idxs in self.group_quadtree.get_group_dict().items():
            X_idxs = np.asarray(group_X_idxs)

            print(X_idxs)

            V_hat = self.kernel.cov(X_hat)

            # assume p_g(x) pdf is uniform
            # omega shape: (len(X), 1), eqn 11
            V_sum = V_hat[:, group_X_idxs, group_X_idxs]
            print("V_sum", V_sum.shape)
            kern_sum = V_sum.sum(axis=-1, keepdims=True)
            w_g = kern_sum / self.group_quadtree.group_area

            # Z, eqn 12
            # times 2 was added as V is symmetric
            Zg = kern_sum * 2 / self.group_quadtree.group_area**2

            V_sum = V_hat[:, -1, group_X_idxs]
            print("V_sum", V_sum.shape)
            kern_sum = V_hat[:, -1, group_X_idxs].sum(axis=-1, keepdims=True)
            w_g_s = kern_sum / self.group_quadtree.group_area
            print("w_g_s", w_g_s.shape)

            print("w_g", w_g.shape)

            w_g_s_hat[:, :-1] = w_g
            w_g_s_hat[:, -1:] = w_g_s
            print("w_g_s_hat", w_g_s_hat.shape)

            # beta_g, eqn 20
            V_hat_inv = np.linalg.inv(V_hat)
            w_g_s_hat_T = np.swapaxes(w_g_s_hat, -1, -2)
            print("V_hat_inv", V_hat_inv.shape)
            print("wgs", w_g_s_hat.shape)
            beta_g_hat = Zg - w_g_s_hat_T @ V_hat_inv @ w_g_s_hat

            # vg^2_hat

            # V_s|D: unsure why this is a scalar if its a capital letter
            x_s = X_hat[:, -1, :]
            X = X_s[:, :-1, :]
            V_inv = V_hat[:-1, :-1]
            k_ss = self.kernel(x_s, x_s)

            # alpha_g, below eqn 11
            wg_T = np.swapaxes(w_g, -1, -2)
            alpha_g = w_g_T @ V_inv @ y

            # above eqn 19
            v_sD = (
                k_ss
                - self.kernel(x_s, X) * V_inv @ self.kernel(X, x_s)
                + self.noise_var
            )

            K_s = V_hat[:, -1]
            K_s_T = np.swapaxes(K_s, -1, -2)
            v_g_2_hat_term = w_g_s_hat - K_s.T @ V_inv @ w_g
            v_g_2_hat_term_T = np.swapaxes(v_g_2_hat_term, -1, -2)
            v_g_2_hat = v_g_2_hat_term.T * 1 / V_sD * v_g_2_hat_term

            reward += norm.cdf(
                (
                    alpha_g
                    - self.threshold
                    - beta_g_hat * norm.ppf(self.confidence) / v_g_2_hat
                )
            )

        optimal_state = np.argmax(reward)

        return self.grid.unnormalize(optimal_state)


if __name__ == "__main__":
    kernel = SquaredExpKernel(scale=2)
    grid_map = SurfaceGridMap(phantom_pcd)
    qt_dim = max(grid_map.shape)
    qt_dim += 10
    qt_dim = (qt_dim // 10) * 10
    group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)

    aas = ActiveAreaSearch(grid_map, group_quadtree, kernel)

    samples = np.array(
        [
            [0, 0, 1],
            [0, 8, 1],
            [8, 0, 2],
            [8, 8, 1],
        ]
    )

    for sample in samples:
        next_state = aas.get_optimal_state(sample[:3], sample[-1])
