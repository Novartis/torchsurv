# global modules
import json
import unittest

import numpy as np
import torch
from lifelines import WeibullAFTFitter
from lifelines.datasets import load_gbsg2, load_lung

# Local modules
from torchsurv.loss.survival import neg_log_likelihood_survival as survival
from torchsurv.loss.survival import survival_function
from torchsurv.loss.weibull import survival_function as weibull_sf

# Load the benchmark cox log likelihoods from R
with open("tests/benchmark_data/benchmark_weibull.json") as file:
    benchmark_weibull_logliks = json.load(file)

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestSurvivalLoss(unittest.TestCase):
    """
    List of packages compared
        - survival (R)
        - lifelines (python)
    """

    # random data and parameters
    N = 32
    log_params = torch.randn((N, 2), dtype=torch.float)
    y = torch.randint(low=0, high=2, size=(N, 1), dtype=torch.bool)
    t = torch.randint(low=1, high=100, size=(N, 1), dtype=torch.float)

    # prepare data
    lung = load_lung()
    lung["sex"] = (lung["sex"] == 1).astype(float)
    lung["age"] = (lung["age"] - lung["age"].mean()) / lung["age"].std()

    gbsg = load_gbsg2()
    gbsg["age"] = (gbsg["age"] - gbsg["age"].mean()) / gbsg["age"].std()
    gbsg["tsize"] = (gbsg["tsize"] - gbsg["tsize"].mean()) / gbsg["tsize"].std()

    def test_log_likelihood_1_param(self):
        """test survival log likelihood with only 1 param (log scale) on lung and gbsg data"""
        for benchmark_weibull_loglik in benchmark_weibull_logliks:
            time = torch.tensor(benchmark_weibull_loglik["time"], dtype=torch.float32)
            event = torch.tensor(benchmark_weibull_loglik["status"]).bool()
            log_scale = torch.tensor(
                benchmark_weibull_loglik["log_scale"], dtype=torch.float32
            ).squeeze(0)

            eval_time = (
                torch.cat([torch.linspace(0, time.max(), steps=100), time])
                .unique()
                .sort()
                .values
            )
            log_hz = -log_scale.unsqueeze(1).expand(-1, len(eval_time))

            log_lik = -survival(
                log_hz,
                event,
                time,
                eval_time,
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
        """test value of survival log likelihood with 2 params (log scale and log shape) on lung data"""

        event = torch.tensor(self.lung["status"]).bool()
        time = torch.tensor(self.lung["time"], dtype=torch.float32)

        # compute eval_time that includes all event times
        time_sorted, _ = torch.sort(time)
        deltas = torch.diff(torch.cat([torch.tensor([0.0]), time_sorted]).unique())
        min_delta = deltas[deltas > 0].min()
        eval_time = torch.arange(0, time.max() + min_delta, step=min_delta)

        # intercept only
        wbf = WeibullAFTFitter()
        wbf.fit(
            self.lung[["time", "status"]],
            duration_col="time",
            event_col="status",
        )
        log_params = torch.tensor(
            np.ones((len(event), 1)) * wbf.summary.loc[:, "coef"].values,
            dtype=torch.float32,
        )
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-4,
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
        log_params = torch.tensor(np.column_stack((log_scale, log_shape)))
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-4,
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
        log_params = torch.tensor(np.column_stack((log_scale, log_shape)))
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
            reduction="sum",
        )
        self.assertTrue(
            np.allclose(
                log_likelihood.numpy(),
                np.array(log_likelihood_lifelines),
                rtol=1e-4,
                atol=1e-8,
            )
        )

    def test_log_likelihood_2_params_gbsg(self):
        """test value of survival log likelihood with 2 params (log scale and log shape) on gbsg data"""

        event = torch.tensor(self.gbsg["cens"]).bool()
        time = torch.tensor(self.gbsg["time"], dtype=torch.float32)

        # compute eval_time that includes all event times
        time_sorted, _ = torch.sort(time)
        deltas = torch.diff(torch.cat([torch.tensor([0.0]), time_sorted]).unique())
        min_delta = deltas[deltas > 0].min()
        eval_time = torch.arange(0, time.max() + min_delta, step=min_delta)

        # intercept only
        wbf = WeibullAFTFitter()
        wbf.fit(self.gbsg[["time", "cens"]], duration_col="time", event_col="cens")
        log_params = torch.tensor(
            np.ones((len(event), 1)) * wbf.summary.loc[:, "coef"].values
        )
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
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
        log_params = torch.tensor(np.column_stack((log_scale, log_shape)))
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
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
        log_params = torch.tensor(np.column_stack((log_scale, log_shape)))
        log_likelihood_lifelines = wbf.log_likelihood_

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # compute log likelihood
        log_likelihood = -survival(
            log_hz,
            event,
            time,
            eval_time,
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

    def test_survival_function_2_params_lung(self):
        """test value of survival function with 2 params (log scale and log shape) on lung data"""

        time = torch.tensor(self.lung["time"], dtype=torch.float32)

        # compute eval_time that includes all event times
        time_sorted, _ = torch.sort(time)
        deltas = torch.diff(torch.cat([torch.tensor([0.0]), time_sorted]).unique())
        min_delta = deltas[deltas > 0].min()
        eval_time = torch.arange(0, time.max() + min_delta, step=min_delta)

        # intercept only
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
        log_params = torch.tensor(np.column_stack((log_scale, log_shape)))

        # compute log_hz
        log_scale, log_shape = log_params[:, 0].unsqueeze(1), log_params[
            :, 1
        ].unsqueeze(1)
        log_hz = (
            log_shape
            - log_scale
            + (torch.exp(log_shape) - 1)
            * (torch.log(eval_time).unsqueeze(0) - log_scale)
        )

        # pick new time
        new_time = time[0:10]

        # compute survival function
        survival = survival_function(
            log_hz,
            new_time,
            eval_time,
        )

        # compute survival function using weibull_sf
        survival_weibull = weibull_sf(
            torch.tensor(log_params, dtype=torch.float32),
            new_time,
        )

        self.assertTrue(
            np.allclose(
                survival.numpy(),
                np.array(survival_weibull),
                rtol=1e-3,
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()
