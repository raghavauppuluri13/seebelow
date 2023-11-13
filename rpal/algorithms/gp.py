from typing import Tuple
from rpal.algorithms.grid import GridMap2D
import numpy as np


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
        print("X_reshaped", X_reshaped.shape)
        X_prime_reshaped = np.expand_dims(X_prime, axis=-3)
        print("X_prime_reshaped", X_prime_reshaped.shape)
        K = self(X_reshaped, X_prime_reshaped)
        print("K", K.shape)
        N = K.shape[-1]
        K += np.eye(N) * noise_var  # obs noise
        return K


class GP:
    def __init__(self, kernel, noise_var=0.01):
        self.kernel = kernel
        self.noise_var = noise_var
        self.X = []
        self.y = []

    def add_sample(self, x: Tuple[float, float], y: float):
        self.X.append(x)
        self.y.append(y)

    def posterior(self, X_s: np.ndarray):
        """
        Computes posterior of p(f(X_s) | f(self.X))
        X_star: np.ndarray (N, 2)
        """

        X = np.array(self.X)
        y = np.array(self.y)
        y = y[:, np.newaxis]

        print("X", X.shape)
        print("y", y.shape)

        K = self.kernel.cov(X)
        print("K", K)

        print("X_s", X_s.shape)
        K_s_x = self.kernel(X_s, X)
        K_s_x = K_s_x[..., np.newaxis]
        print("K_s_x", K_s_x.shape)

        beta = np.einsum("ijk,jj->ijk", K_s_x, np.linalg.inv(K))
        print("beta", beta.shape)

        k_ss = self.kernel(X_s, X_s)
        print("k_ss", k_ss.shape)

        # these are dot products
        posterior_mean = np.einsum("ijk,jk->ik", beta, y)
        posterior_std = k_ss + np.einsum("ijk,ijk->ik", beta, K_s_x)

        print("posterior_mean", posterior_mean.shape)
        print("posterior_std", posterior_std.shape)
        return posterior_mean, posterior_std


if __name__ == "__main__":
    kernel = SquaredExpKernel(scale=0.5)

    gp = GP(kernel, noise_var=0)

    obs = [
        ((0, 0), 1),
        ((0, 8), 1),
        ((8, 0), 2),
        ((8, 8), 1),
    ]

    for x, y in obs:
        gp.add_sample(x, y)

    import matplotlib.pyplot as plt

    grid = GridMap2D(10, 10)
    X_s = grid.vectorized_states
    print("X_s", X_s.shape)
    # permuations via numpy between 0,0 and 10,10
    Mu_s = gp.posterior_mean(X_s)
    print("Mu_s", Mu_s.shape)

    plt.imshow(Mu_s.reshape(grid_shape), cmap="hot", interpolation="bilinear")
    plt.colorbar()
    plt.show()
