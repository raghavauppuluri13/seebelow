from typing import Tuple
import numpy as np


class SquaredExpKernel:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: np.ndarray, x_prime: np.ndarray, keepdims=False):
        distance = np.linalg.norm(x - x_prime, axis=-1, keepdims=keepdims)
        # print("distance", distance)
        return np.exp(-(distance**2) / (2 * self.scale**2))

    def cov(self, X: np.ndarray, X_prime: np.ndarray = None, noise_var=0.01):
        """using the kernel, generates a covariance matrix"""
        if X_prime is None:
            X_prime = X
        assert X.shape[-1] == 2
        X_reshaped = np.expand_dims(X, axis=-2)
        print("X_reshaped", X_reshaped.shape)
        X_prime_reshaped = np.expand_dims(X_prime, axis=1)
        print("X_prime_reshaped", X_prime_reshaped.shape)
        K = self(X_reshaped, X_prime_reshaped)
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

    def posterior_mean(self, X_s: np.ndarray):
        """
        Expectation of p(f(x_star) | f(obs))

        x_star: np.ndarray X: shapex,y)
        """

        X = np.array(self.X)[np.newaxis, :, :]
        y = np.array(self.y)[np.newaxis, :]

        print("X", X.shape)
        print("y", y.shape)

        K = self.kernel.cov(X) + np.eye(len(X)) * self.noise_var
        print("K", K.shape)

        print("X_s", X_s.shape)
        K_s_x = self.kernel(X_s, X)
        print("K_s_x", K_s_x.shape)

        beta = K_x_X.T @ np.linalg.inv(K)
        print("beta", beta)
        print("k_star", K_star)

        assert beta.shape == (N,)

        return np.dot(beta, obs[:, -1])


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

    grid_shape = (10, 10)
    grid = np.zeros(grid_shape)

    nx, ny = (10, 10)
    gx = np.arange(0, grid.shape[0])
    gy = np.arange(0, grid.shape[1])
    X_grid = np.meshgrid(gx, gy)
    X_s = np.concatenate([X_grid[0].reshape(-1, 1), X_grid[1].reshape(-1, 1)], axis=1)

    # permuations via numpy between 0,0 and 10,10
    Mu_s = gp.posterior_mean(X_s)
    print("Mu_s", Mu_s.shape)

    plt.imshow(grid, cmap="hot", interpolation="bilinear")
    plt.colorbar()
    # plt.show()
