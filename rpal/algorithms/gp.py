import numpy as np


class SquaredExpKernel:
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: np.ndarray, x_prime: np.ndarray):
        distance = np.linalg.norm(x - x_prime, axis=-1)
        # print("distance", distance)
        return np.exp(-(distance**2) / (2 * self.scale**2))

    def cov(self, X: np.ndarray, X_prime: np.ndarray = None, noise_var=0.01):
        """using the kernel, generates a covariance matrix"""
        if X_prime is None:
            X_prime = X
        assert X.shape[-1] == 2
        X_reshaped = np.expand_dims(X, axis=-2)
        X_prime_reshaped = np.expand_dims(X_prime, axis=1)
        K = self(X_reshaped, X_prime_reshaped)
        N = K.shape[-1]
        K += np.eye(N) * noise_var  # obs noise
        return K


class GP:
    noise_var = 0.01

    def __init__(self, length_scale, input_space_dim):
        self.kernel = SquaredExpKernel(scale=2)

    def _posterior_mean(self, x_star: np.ndarray, obs: np.ndarray):
        """
        Expectation of p(f(x_star) | f(obs))

        x_star: np.ndarray of structure X: (x,y)
        obs: observations of structure (n, (X,f(X)))
        """

        K = self.kernel.cov(obs[:, :2])
        K_star = np.zeros(N)
        for i in range(len(obs)):
            K_star[i] = self.kernel(obs[i, :2], x_star)

        assert K_star.T.shape == (N,), print("inv", K_star.T @ np.linalg.inv(K))

        beta = K_star.T @ np.linalg.inv(K)
        # print("beta", beta)
        # print("k_star", K_star)

        assert beta.shape == (N,)

        return np.dot(beta, obs[:, -1])


if __name__ == "__main__":
    gp = GP(1, 2)

    obs = np.array(
        [
            [0, 0, 1],
            [0, 8, 1],
            [8, 0, 2],
            [8, 8, 1],
        ]
    )

    x_star_tests = np.array([[3, 3], [5, 5], [3, 5], [5, 3]])

    means = []
    for x_star in x_star_tests:
        means.append(gp._posterior_mean(x_star, obs))

    import matplotlib.pyplot as plt

    grid_shape = (10, 10)
    grid = np.zeros(grid_shape)

    # permuations via numpy between 0,0 and 10,10
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            grid[i, j] = gp.posterior_mean(np.array([i, j]), obs)

    plt.imshow(grid, cmap="hot", interpolation="bilinear")

    plt.colorbar()

    plt.show()
