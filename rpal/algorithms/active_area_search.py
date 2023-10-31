"""
@brief
Implementation of Active Area Search from Ma, Y., Garnett, R. &amp; Schneider, J.. (2014). Active Area Search via Bayesian Quadrature. <i>Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 33:595-603 Available from https://proceedings.mlr.press/v33/ma14.html.
@author Raghava Uppuluri
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm
from gp import GP
from gp import SquaredExpKernel
from cluster_store import ClusterStore
from collections import defaultdict


class ActiveAreaSearch:
    noise_var = 0.01
    state_dim = 2
    cluster_area = 0.1
    threshold = 7
    confidence = 0.6

    def __init__(self, grid_dims, region_area, cluster_store):
        self.kernel = SquaredExpKernel(scale=2)
        self.grid_dims = grid_dims
        self.region_area = region_area
        self.X = []  # shape: (len(X), state_dim)
        self.Y = []  # shape: (len(Y), 1)

        self.group_store = defaultdict(list)
        self.cluster_store: ClusterStore = cluster_store

    def get_optimal_state(self, prev_sample: np.ndarray):
        """Gets optimal state given previous states and observed f(x) values
        prev_sample: previous state and observed f(x), shape=(state_dim + 1,1)
        """
        assert prev_sample.shape == (self.state_dim + 1,), print(prev_sample.shape)

        y = prev_sample[self.state_dim :]
        x_hat = prev_sample[: self.state_dim]

        self.X.append(x_hat)
        self.Y.append(y)

        cluster_key = self.cluster_store.get_cluster_from_pt(x_hat)
        self.group_store[cluster_key].append(len(self.X) - 1)

        X = np.array(self.X)
        y = np.array(self.Y)

        nx, ny = (100, 100)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X_grid = np.meshgrid(x, y)

        # vectorize reward computation over all states (S) in grid -> (S, 2)
        X_s = np.concatenate(
            [X_grid[0].reshape(-1, 1), X_grid[1].reshape(-1, 1)], axis=1
        )
        X_hat = np.zeros((X_s.shape[0], X.shape[1] + 1))
        X_hat[:, :-1, :] = X
        X_hat[:, -1, :] = X_s

        for group, group_X_idxs in self.group_store.items():

            X_idxs = np.asarray(group_X_idxs)

            V_hat = self.kernel.cov(X_hat)

            # assume p_g(x) pdf is uniform
            # omega shape: (len(X), 1), eqn 11
            kern_sum = V_hat[:, group_X_idxs][group_X_idxs].sum(axis=1)
            w_g = kern_sum / self.cluster_area

            kern_sum = V_hat[:, -1][group_X_idxs].sum(axis=1)
            w_g_s_hat = kern_sum / self.cluster_area

            # Z, eqn 12
            # times 2 was added as V is symmetric
            Zg = kern_sum * 2 / self.cluster_area**2

            # beta_g, eqn 20
            V_hat_inv = V_hat.inv()
            beta_g_hat = Zg - w_g_hat.T @ V_hat_inv @ w_g_hat

            # vg^2_hat

            # V_s|D: unsure why this is a scalar if its a capital letter
            x_s = X_hat[:, -1, :]
            X = X_s[:, :-1, :]
            V_inv = V_hat[:-1, :-1]
            k_ss = self.kernel(x_s, x_s)

            # alpha_g, below eqn 11
            alpha_g = w_g.T @ V_inv @ y

            # above eqn 19
            v_sD = (
                k_ss
                - self.kernel(x_s, X) * V_inv @ self.kernel(X, x_s)
                + self.noise_var
            )

            K_s = V_hat[:, -1]
            v_g_2_hat_term = w_g_s_hat - K_s.T @ V_inv @ w_g
            v_g_2_hat = v_g_2_hat_term.T * 1 / V_sD * v_g_2_hat_term

            reward = norm.cdf(
                (
                    alpha_g
                    - self.threshold
                    - beta_g_hat * norm.ppf(self.confidence) / v_g_2_hat
                )
            )

            optimal_state = Xs[np.argmax(reward)]
            print(optimal_state)


if __name__ == "__main__":

    store = ClusterStore(phantom_pcd)

    aas = ActiveAreaSearch((10, 10), 1, cluster_store)

    samples = np.array(
        [
            [0, 0, 1],
            [0, 8, 1],
            [8, 0, 2],
            [8, 8, 1],
        ]
    )

    for sample in samples:
        next_state = aas.get_optimal_state(sample)
