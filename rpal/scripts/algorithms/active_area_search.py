import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor


class ActiveAreaSearch(ActiveGP):  # Assuming ActiveGP is another class you have
    def __init__(self, gp_model, gp_para, pool_locs, regions, level, side, highprob):
        super().__init__(
            gp_model, gp_para
        )  # Assuming a similar constructor for ActiveGP
        self.regions = regions
        self.level = level
        self.side = side
        self.highprob = highprob
        self.cumfound = np.zeros(regions.shape[0])
        self.pool_locs = pool_locs  # This will call the setter method

    @property
    def pool_locs(self):
        return self._pool_locs

    @pool_locs.setter
    def pool_locs(self, pool_locs):
        omegas, tV = self.pool_locs_compute_stats(pool_locs)
        self._pool_locs = pool_locs
        self.omegas = omegas
        self.tV = tV

    def update(self, new_locs, new_vals):
        super().update(
            new_locs, new_vals
        )  # Assuming a similar update method in ActiveGP
        new_found, Tg = self.update_regions()
        return new_found, Tg

    def update_regions(self):
        # (Place the implementation here, similar to the MATLAB code)
        pass

    def region_rewards(self):
        Tg = norm.cdf(self.side * (self.alpha - self.level) / np.sqrt(self.beta2))
        new_found = (Tg > self.highprob).astype(int)
        return new_found, Tg

    def pool_locs_compute_stats(self, pool_locs):
        omegas = covSEregion(self.gp_para["cov"], self.regions, pool_locs)
        _, tV = self.predict_points(pool_locs)  # Assuming a similar method in ActiveGP
        return omegas, tV

    def utility(self, pool_locs=None):
        if pool_locs is None:
            pool_locs = self.pool_locs
            omegas = self.omegas
            tV = self.tV
        else:
            omegas, tV = self.pool_locs_compute_stats(pool_locs)

        # (Place the remaining implementation here, similar to the MATLAB code)
