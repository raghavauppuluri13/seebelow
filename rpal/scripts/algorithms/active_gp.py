import numpy as np


class ActiveGP(object):
    def __init__(self, gp_model, gp_para):
        assert (
            np.exp(2 * gp_para["lik"]) > 1e-6
        )  # gpml uses different representations otherwise
        assert gp_model.mean.__name__ == "meanZero"  # simplify model

        self.gp_model = gp_model
        self.gp_para = gp_para if "mean" in gp_para else {**gp_para, "mean": []}
        self.collected_locs = np.array([])  # Initialize as empty arrays
        self.collected_vals = np.array([])
        self.gp_post = None
        self.R = None

    def predict_points(self, new_x):
        if self.collected_locs.size == 0:
            fmu = self.gp_model.mean(self.gp_para["mean"], new_x)
            fs2 = self.gp_model.cov(self.gp_para["cov"], new_x, diag=True)
            ymu = fmu
            ys2 = fs2 + np.exp(2 * self.gp_para["lik"])
        else:
            ymu, ys2, fmu, fs2 = gp(
                self.gp_para,
                self.gp_model.inf,
                self.gp_model.mean,
                self.gp_model.cov,
                self.gp_model.lik,
                self.collected_locs,
                self.gp_post,
                new_x,
            )
        return ymu, ys2, fmu, fs2

    def update(self, locations, values):
        values = np.array(values).flatten()
        if self.collected_locs.size == 0:
            _, _, self.gp_post = gp(
                self.gp_para,
                self.gp_model.inf,
                self.gp_model.mean,
                self.gp_model.cov,
                self.gp_model.lik,
                locations,
                values,
            )
            self.R = self.gp_post["L"] * np.exp(self.gp_para["lik"])
        else:
            self.gp_post = update_posterior(
                self.gp_para,
                self.gp_model.mean,
                [self.gp_model.cov],
                self.collected_locs,
                self.gp_post,
                locations,
                values,
            )
            self.R = self.gp_post["L"] * np.exp(self.gp_para["lik"])

        self.collected_locs = (
            np.vstack([self.collected_locs, locations])
            if self.collected_locs.size
            else locations
        )
        self.collected_vals = (
            np.hstack([self.collected_vals, values])
            if self.collected_vals.size
            else values
        )
