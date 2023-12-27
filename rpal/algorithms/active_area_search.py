"""
@brief
Implementation of Active Area Search from Ma, Y., Garnett, R. &amp; Schneider, J.. (2014). Active Area Search via Bayesian Quadrature. <i>Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 33:595-603 Available from https://proceedings.mlr.press/v33/ma14.html.
@author Raghava Uppuluri
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm

from rpal.algorithms.gp import GP, SquaredExpKernel
from rpal.algorithms.grid import SurfaceGridMap
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
        self.X = []
        self.Y = []
        self.grid = surface_grid
        self.group_quadtree = group_quadtree
        self.noise_var = noise_var
        self.threshold = threshold
        self.confidence = confidence

    def get_optimal_state(self, prev_x_hat: np.ndarray, prev_y: float):
        """Gets optimal state given previous states and observed f(x) values
        prev_sample_x_hat: previous state shape=(state_dim,1)
        prev_sample_y: previous observed f(x)
        """
        assert prev_x_hat.shape == (self.state_dim,), print(prev_x_hat.shape)

        self.X.append(prev_x_hat)
        self.Y.append(prev_y)

        self.group_quadtree.insert(idx, len(self.X) - 1)

        X = np.array(self.X)  # shape: (len(X), state_dim)
        y = np.array(self.Y)  # shape: (len(X), 1)
        y = y[:, np.newaxis]

        X_s = self.grid.vectorized_states
        print("X_s", X_s.shape)
        X_hat = np.zeros((X_s.shape[0], X.shape[0] + 1, self.state_dim))
        X_hat[:, :-1, :] = X
        X_hat[:, -1:, :] = X_s
        print("X_hat", X_hat.shape)

        reward = np.zeros(X_hat.shape[0])

        for group, group_X_idxs in self.group_quadtree.get_group_dict().items():
            group_X_idxs = np.asarray(group_X_idxs)
            group_X_idxs = group_X_idxs[:, np.newaxis]

            V_hat = self.kernel.cov(X_hat)
            print("V_hat", V_hat.mean(axis=0))

            # assume p_g(x) pdf is uniform
            # omega shape: (len(X), 1), eqn 11
            V_sum = V_hat[:, group_X_idxs, group_X_idxs]
            print("V_sum", V_sum.shape)
            kern_sum = V_sum.sum(axis=-1, keepdims=True)
            w_g = kern_sum / self.group_quadtree.group_area
            w_g_T = np.einsum("ijk->ikj", w_g)
            print("w_g_T", w_g_T.shape)

            # Z, eqn 12
            # times 2 was added as V is symmetric
            Zg = kern_sum * 2 / self.group_quadtree.group_area**2
            print("Zg", Zg.shape)

            V_sum = V_hat[:, -1, group_X_idxs]
            print("V_sum", V_sum.shape)
            kern_sum = V_hat[:, -1, group_X_idxs].sum(axis=-1, keepdims=True)
            w_g_s = kern_sum / self.group_quadtree.group_area
            print("w_g_s", w_g_s.shape)
            print("w_g", w_g.shape)

            # beta2_g, eqn 12
            V_inv = np.linalg.inv(V_hat[:, :-1, :-1])
            print("V_inv", V_inv.shape)
            beta2_g = (
                Zg - np.einsum("ikj,ijj,ijk->ik", w_g_T, V_inv, w_g)[:, :, np.newaxis]
            )
            print("beta2_g", beta2_g.shape)

            # vg^2_hat
            # V_s|D: unsure why this is a scalar if its a capital letter
            print("X_hat", X_hat.shape)
            x_s = X_hat[:, -1:, :]
            print("x_s", x_s.shape)
            X = X_hat[:, :-1, :]
            print("X", X.shape)
            k_ss = self.kernel(x_s, x_s)
            print("k_ss", k_ss.shape)

            # alpha_g, below eqn 11
            print("y", y.shape)
            alpha_g = np.einsum("ikj,ijj,jk->ik", w_g_T, V_inv, y)
            print("alpha_g", alpha_g.shape)

            # v_sD above eqn 19
            k_s_X = self.kernel(x_s, X, keepdims=True)
            k_X_s = self.kernel(X, x_s, keepdims=True)
            print("k_s_X", k_s_X.shape)
            print("k_X_s", k_X_s.shape)
            v_sD = (
                k_ss
                - np.einsum("ikj,ijj,ijk->ik", k_s_X, V_inv, k_X_s)
                + self.noise_var
            )

            print("v_sD", v_sD.shape)

            # v_g_tilde
            v_g_tilde_term = w_g_s - k_s_X @ V_inv @ w_g
            print("v_g_tilde_term", v_g_tilde_term.shape)
            v_g_tilde = v_g_tilde_term / (v_sD[:, :, np.newaxis] + 1e-8)
            beta2_g_tilde = beta2_g - v_g_tilde**2
            print("v_g_tilde", v_g_tilde.shape)
            print("beta2_g_tilde", beta2_g_tilde.shape)
            assert np.all(beta2_g_tilde >= 0), print(beta2_g_tilde.min(axis=0))
            # assert not np.any(v_g_tilde == 0), print(v_g_tilde)

            reward_g = norm.cdf(
                (
                    alpha_g[:, :, np.newaxis]
                    - self.threshold
                    - np.sqrt(beta2_g_tilde) * norm.ppf(self.confidence),
                )
                / (v_g_tilde + 1e-8)
            )

            print(reward_g.max())

        optimal_state = X_s[np.argmax(reward)]
        print(optimal_state)
        return optimal_state


if __name__ == "__main__":
    kernel = SquaredExpKernel(scale=2)
    grid_map = SurfaceGridMap(phantom_pcd)
    qt_dim = max(grid_map.shape)
    qt_dim += 10
    qt_dim = (qt_dim // 10) * 10
    group_quadtree = QuadTree(qt_dim, qt_dim, group_dim, group_dim)

    aas = ActiveAreaSearch(grid_map, group_quadtree, kernel)

    samples = np.array([[0, 0, 1], [0, 8, 1], [8, 0, 2], [8, 8, 1]])

    for sample in samples:
        next_state = aas.get_optimal_state(sample[:3], sample[-1])
