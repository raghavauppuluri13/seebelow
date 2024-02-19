from typing import Tuple

import numpy as np

from rpal.algorithms.grid import GridMap2D


class SquaredExpKernel:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: np.ndarray, x_prime: np.ndarray, keepdims=False):
        distance = np.linalg.norm(x - x_prime, axis=-1, keepdims=keepdims)
        return np.exp(-(distance**2) / (2 * self.scale**2))

    def cov(self, X: np.ndarray, X_prime: np.ndarray = None, noise_var=0.01):
        """using the kernel, generates a covariance matrix"""
        if X_prime is None:
            X_prime = X
        assert X.shape[-1] == 2
        X_reshaped = np.expand_dims(X, axis=-2)
        # print("X_reshaped", X_reshaped.shape)
        X_prime_reshaped = np.expand_dims(X_prime, axis=-3)
        # print("X_prime_reshaped", X_prime_reshaped.shape)
        K = self(X_reshaped, X_prime_reshaped)
        # print("K", K.shape)
        N = K.shape[-1]
        K += np.eye(N) * noise_var  # obs noise
        return K


def gp_posterior(X_s: np.ndarray, X, y, kernel, noise_var=0.01):
    """
    Computes posterior of p(f(X_s) | f(self.X))
    X_star: np.ndarray (N, 2)
    X: np.ndarray (N, 2)
    y: np.ndarray (N, 1)
    kernel: SquaredExpKernel
    noise_var: float
    """
    y = y[:, np.newaxis]

    # print("X", X.shape)
    # print("X_s", X_s.shape)
    # print("y", y.shape)

    K = kernel.cov(X)
    # print("K", K)

    # print("X_s", X_s.shape)
    K_s_x = kernel(X_s, X)
    K_s_x = K_s_x[..., np.newaxis]
    # print("K_s_x", K_s_x.shape)

    beta = np.einsum("ijk,jj->ijk", K_s_x, np.linalg.inv(K))
    # print("beta", beta.shape)

    k_ss = kernel(X_s, X_s)
    # print("k_ss", k_ss.shape)

    # these are dot products
    posterior_mean = np.einsum("ijk,jk->ik", beta, y)
    posterior_std = k_ss + np.einsum("ijk,ijk->ik", beta, K_s_x)

    # print("posterior_mean", posterior_mean.shape)
    # print("posterior_std", posterior_std.shape)
    return posterior_mean, posterior_std


if __name__ == "__main__":
    kernel = SquaredExpKernel(scale=0.5)

    obs = np.array(
        [
            (0, 0, 1),
            (0, 8, 1),
            (8, 0, 2),
            (8, 8, 1),
        ]
    )

    print(obs.shape)

    import matplotlib.pyplot as plt

    grid = GridMap2D(10, 10)
    X_s = grid.vectorized_states
    # print("X_s", X_s.shape)
    # permuations via numpy between 0,0 and 10,10
    mu, std = gp_posterior(X_s, obs[:, :2], obs[:, -1], kernel)
    # print("Mu_s", Mu_s.shape)

    plt.imshow(mu.reshape(grid.shape), cmap="hot", interpolation="bilinear")
    plt.colorbar()
    plt.show()
