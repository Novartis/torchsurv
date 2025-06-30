# global modules
import json
import unittest

import numpy as np
import torch
from lifelines import WeibullAFTFitter
from lifelines.datasets import load_gbsg2, load_lung

# Local modules
from torchsurv.loss.weibull import neg_log_likelihood as weibull

# Load the benchmark cox log likelihoods from R
with open("tests/benchmark_data/benchmark_weibull.json") as file:
    benchmark_weibull_logliks = json.load(file)

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestWeibullSurvivalLoss(unittest.TestCase):
    """
    List of packages compared
        - survival (R)
        - lifelines (python)
    """

    # random data and parameters
    N = 32
    log_params = torch.randn(N, 2)
    y = torch.randint(low=0, high=2, size=(N, 1)).bool()
    t = torch.randint(low=1, high=100, size=(N, 1))

    # prepare data
    lung = load_lung()
    lung["sex"] = (lung["sex"] == 1).astype(float)
    lung["age"] = (lung["age"] - lung["age"].mean()) / lung["age"].std()

    gbsg = load_gbsg2()
    gbsg["age"] = (gbsg["age"] - gbsg["age"].mean()) / gbsg["age"].std()
    gbsg["tsize"] = (gbsg["tsize"] - gbsg["tsize"].mean()) / gbsg["tsize"].std()

    def test_y_tensor(self):
        y_np_array = np.random.randint(0, 1 + 1, size=(self.N, 1), dtype="bool")
        self.assertRaises(TypeError, weibull, self.log_params, y_np_array, self.t)

    def test_t_tensor(self):
        t_np_array = np.random.randint(0, 100, size=(self.N, 1))
        self.assertRaises(TypeError, weibull, self.log_params, self.y, t_np_array)

    def test_log_params_tensor(self):
        log_params_np_array = np.random.randn(self.N, 2)
        self.assertRaises(TypeError, weibull, log_params_np_array, self.y, self.t)

    def test_nrow_log_params(self):
        log_params_wrong_nrow = torch.randn(self.N + 1, 2)
        self.assertRaises(ValueError, weibull, log_params_wrong_nrow, self.y, self.t)

    def test_len_data(self):
        t_wrong_len = torch.randint(low=1, high=100, size=(self.N + 1, 1))
        self.assertRaises(ValueError, weibull, self.log_params, self.y, t_wrong_len)

    def test_positive_t(self):
        t_negative = torch.randint(low=-100, high=100, size=(self.N, 1))
        self.assertRaises(ValueError, weibull, self.log_params, self.y, t_negative)

    def test_boolean_y(self):
        y_non_boolean = torch.randint(low=0, high=3, size=(self.N, 1))
        self.assertRaises(ValueError, weibull, self.log_params, y_non_boolean, self.t)

    def test_log_likelihood_1_param(self):
        """test weibull log likelihood with only 1 param (log scale) on lung and gbsg data"""
        for benchmark_weibull_loglik in benchmark_weibull_logliks:
            log_lik = -weibull(
                torch.tensor(
                    benchmark_weibull_loglik["log_shape"], dtype=torch.float32
                ).squeeze(0),
                torch.tensor(benchmark_weibull_loglik["status"]).bool(),
                torch.tensor(benchmark_weibull_loglik["time"], dtype=torch.float32),
                reduction="sum",
            )

            log_lik_survival = benchmark_weibull_loglik["log_likelihood"]

            self.assertTrue(
                np.allclose(
                    log_lik.numpy(),
                    np.array(log_lik_survival),
                    rtol=1e-5,
                    atol=1e-8,
                )
            )

    def test_log_likelihood_2_params_lung(self):
        """test value of weibull log likelihood with 2 params (log scale and log shape) on lung data"""

        event = torch.tensor(self.lung["status"]).bool()
        time = torch.tensor(self.lung["time"], dtype=torch.float32)

        # intercept only
        wbf = WeibullAFTFitter()
        wbf.fit(self.lung[["time", "status"]], duration_col="time", event_col="status")
        log_params = np.ones((len(event), 1)) * wbf.summary.loc[:, "coef"].values
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        # one covariate
        wbf = WeibullAFTFitter()
        wbf.fit(
            self.lung[["time", "status", "age"]],
            duration_col="time",
            event_col="status",
        )
        log_scale = (
            wbf.summary.loc[:, "coef"].lambda_.Intercept
            + np.array(self.lung["age"]) * wbf.summary.loc[:, "coef"].lambda_.age
        )
        log_shape = (
            np.ones((self.lung.shape[0],)) * wbf.summary.loc[:, "coef"].rho_.Intercept
        )
        log_params = np.column_stack((log_scale, log_shape))
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        # two covariates
        wbf = WeibullAFTFitter()
        wbf.fit(
            self.lung[["time", "status", "age", "sex"]],
            duration_col="time",
            event_col="status",
        )
        log_scale = (
            wbf.summary.loc[:, "coef"].lambda_.Intercept
            + np.array(self.lung["age"]) * wbf.summary.loc[:, "coef"].lambda_.age
            + np.array(self.lung["sex"]) * wbf.summary.loc[:, "coef"].lambda_.sex
        )
        log_shape = (
            np.ones((self.lung.shape[0],)) * wbf.summary.loc[:, "coef"].rho_.Intercept
        )
        log_params = np.column_stack((log_scale, log_shape))
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_log_likelihood_2_params(self):
        """test value of weibull log likelihood with 2 params (log scale and log shape) on gbsg data"""

        event = torch.tensor(self.gbsg["cens"]).bool()
        time = torch.tensor(self.gbsg["time"], dtype=torch.float32)

        # intercept only
        wbf = WeibullAFTFitter()
        wbf.fit(self.gbsg[["time", "cens"]], duration_col="time", event_col="cens")
        log_params = np.ones((len(event), 1)) * wbf.summary.loc[:, "coef"].values
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        # one covariate
        wbf = WeibullAFTFitter()
        wbf.fit(
            self.gbsg[["time", "cens", "age"]],
            duration_col="time",
            event_col="cens",
        )
        log_scale = (
            wbf.summary.loc[:, "coef"].lambda_.Intercept
            + np.array(self.gbsg["age"]) * wbf.summary.loc[:, "coef"].lambda_.age
        )
        log_shape = (
            np.ones((self.gbsg.shape[0],)) * wbf.summary.loc[:, "coef"].rho_.Intercept
        )
        log_params = np.column_stack((log_scale, log_shape))
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        # two covariates
        wbf = WeibullAFTFitter()
        wbf.fit(
            self.gbsg[["time", "cens", "age", "tsize"]],
            duration_col="time",
            event_col="cens",
        )
        log_scale = (
            wbf.summary.loc[:, "coef"].lambda_.Intercept
            + np.array(self.gbsg["age"]) * wbf.summary.loc[:, "coef"].lambda_.age
            + np.array(self.gbsg["tsize"]) * wbf.summary.loc[:, "coef"].lambda_.tsize
        )
        log_shape = (
            np.ones((self.gbsg.shape[0],)) * wbf.summary.loc[:, "coef"].rho_.Intercept
        )
        log_params = np.column_stack((log_scale, log_shape))
        log_likelihood_lifelines = wbf.log_likelihood_
        log_likelihood = -weibull(
            torch.tensor(log_params, dtype=torch.float32),
            event,
            time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()
