import json
import unittest

import numpy as np
import pandas as pd
import torch

from lifelines import CoxPHFitter
from lifelines.datasets import load_gbsg2, load_lung

from torchsurv.loss.cox import neg_partial_log_likelihood as cox
from torchsurv.loss.cox import baseline_survival_function, survival_function
from torchsurv.tools.validate_data import validate_survival_data

# Load the benchmark cox log likelihoods from R
with open("tests/benchmark_data/benchmark_cox.json") as file:
    benchmark_cox_logliks = json.load(file)

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestCoxSurvivalLoss(unittest.TestCase):
    """
    List of packages compared
        - survival (R)
        - lifelines (python)
    """

    # random data and parameters
    N = 32
    log_hz = torch.randn(N, dtype=torch.float)
    event = torch.randint(low=0, high=2, size=(N,), dtype=torch.bool)
    time = torch.randint(low=1, high=100, size=(N,), dtype=torch.float)

    # prepare data
    lung = load_lung().dropna()
    lung["sex"] = (lung["sex"] == 1).astype(float)
    lung["age"] = (lung["age"] - lung["age"].mean()) / lung["age"].std()

    gbsg = load_gbsg2().dropna()
    gbsg["age"] = (gbsg["age"] - gbsg["age"].mean()) / gbsg["age"].std()
    gbsg["tsize"] = (gbsg["tsize"] - gbsg["tsize"].mean()) / gbsg["tsize"].std()
    gbsg_strata = gbsg.copy()
    gbsg_strata.drop(columns=["horTh"], inplace=True)
    gbsg.drop(columns=["horTh", "menostat", "tgrade"], inplace=True)

    def test_t_tensor(self):
        time_np_array = np.random.randint(0, 100, size=(self.N,))
        with self.assertRaises((RuntimeError, TypeError)):
            cox(self.log_hz, self.event, time_np_array)

    def test_log_hz_tensor(self):
        log_hz_np_array = np.random.randn(
            self.N,
        )
        with self.assertRaises((RuntimeError, TypeError)):
            cox(log_hz_np_array, self.event, self.time)

    def test_len_data(self):
        event_wrong_length = torch.randint(
            low=0, high=2, size=(self.N + 1,), dtype=torch.bool
        )
        with self.assertRaises(ValueError):
            validate_survival_data(event_wrong_length, self.time)

    def test_positive_t(self):
        time_negative = torch.randint(
            low=-100, high=100, size=(self.N,), dtype=torch.float
        )
        with self.assertRaises(ValueError):
            validate_survival_data(self.event, time_negative)

    def test_boolean_y(self):
        event_non_boolean = torch.randint(low=0, high=3, size=(self.N,))
        with self.assertRaises(ValueError):
            validate_survival_data(event_non_boolean, self.time)

    def test_log_likelihood_without_ties_survival(self):
        """test cox partial log likelihood without ties on lung and gbsg datasets compared to R survival"""
        for benchmark_cox_loglik in benchmark_cox_logliks:
            if benchmark_cox_loglik["no_ties"][0]:
                log_lik = -cox(
                    torch.tensor(benchmark_cox_loglik["log_hazard"], dtype=torch.float32).squeeze(0),
                    torch.tensor(benchmark_cox_loglik["status"]).bool(),
                    torch.tensor(benchmark_cox_loglik["time"], dtype=torch.float32),
                    reduction="sum",
                )
                log_lik_survival = benchmark_cox_loglik["log_likelihood"]

                self.assertTrue(
                    np.allclose(
                        log_lik.numpy(),
                        np.array(log_lik_survival),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                )

    def test_log_likelihood_with_ties_survival(self):
        """test Efron and Breslow's approximation of cox partial log likelihood with ties on lung and gbsg data compared to R survival"""
        for benchmark_cox_loglik in benchmark_cox_logliks:
            if benchmark_cox_loglik["no_ties"][0] is False:
                # efron approximation of partial log likelihood
                log_lik_efron = -cox(
                    torch.tensor(
                        benchmark_cox_loglik["log_hazard_efron"],
                        dtype=torch.float32,
                    ).squeeze(0),
                    torch.tensor(benchmark_cox_loglik["status"]).bool(),
                    torch.tensor(benchmark_cox_loglik["time"], dtype=torch.float32),
                    ties_method="efron",
                    reduction="sum",
                )
                log_lik_efron_survival = benchmark_cox_loglik["log_likelihood_efron"]

                # breslow approximation of partial log likelihood
                log_lik_breslow = -cox(
                    torch.tensor(
                        benchmark_cox_loglik["log_hazard_breslow"],
                        dtype=torch.float32,
                    ).squeeze(0),
                    torch.tensor(benchmark_cox_loglik["status"]).bool(),
                    torch.tensor(benchmark_cox_loglik["time"], dtype=torch.float32),
                    ties_method="breslow",
                    reduction="sum",
                )
                log_lik_breslow_survival = benchmark_cox_loglik["log_likelihood_breslow"]

                self.assertTrue(
                    np.allclose(
                        log_lik_efron.numpy(),
                        np.array(log_lik_efron_survival),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                )

                self.assertTrue(
                    np.allclose(
                        log_lik_breslow.numpy(),
                        np.array(log_lik_breslow_survival),
                        rtol=1e-5,
                        atol=1e-8,
                    )
                )

    def test_log_likelihood_lung_lifelines(self):
        """test cox partial log likelihood on lung datasets compared to lifelines"""

        time = torch.tensor(self.lung["time"].values, dtype=torch.float32)
        event = torch.tensor(self.lung["status"].values, dtype=torch.float32).bool()

        # fit (Ties are handled using Efron’s method on lifelines)
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.lung,
            duration_col="time",
            event_col="status",
        )

        # relative hazard
        hz = coxphf.predict_partial_hazard(self.lung).values
        log_hz = torch.log(torch.tensor(hz))

        # loss function torchsurv
        log_lik = -cox(log_hz, event, time, reduction="sum", ties_method="efron")
        log_lik_lifelines = coxphf.log_likelihood_

        self.assertTrue(
            np.allclose(
                log_lik.numpy(),
                np.array(log_lik_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_log_likelihood_gbsg_lifelines(self):
        """test cox partial log likelihood on gbsg datasets compared to lifelines"""

        time = torch.tensor(self.gbsg["time"].values, dtype=torch.float32)
        event = torch.tensor(self.gbsg["cens"].values, dtype=torch.float32).bool()

        # fit (Ties are handled using Efron’s method on lifelines)
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.gbsg,
            duration_col="time",
            event_col="cens",
        )

        # relative hazard
        hz = coxphf.predict_partial_hazard(self.gbsg).values
        log_hz = torch.log(torch.tensor(hz))

        # compute log likelihood with lifelines
        log_lik_lifelines = coxphf.log_likelihood_

        # compute log_likelihood likelihood with torchsurv
        log_lik = -cox(log_hz, event, time, reduction="sum", ties_method="efron")

        self.assertTrue(
            np.allclose(
                log_lik.numpy(),
                np.array(log_lik_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_log_likelihood_strata_lifelines(self):
        """test cox partial log likelihood with strata on gbsg datasets compared to lifelines"""

        time = torch.tensor(self.gbsg_strata["time"].values, dtype=torch.float32)
        event = torch.tensor(
            self.gbsg_strata["cens"].values, dtype=torch.float32
        ).bool()

        # fit (Ties are handled using Efron’s method on lifelines)
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.gbsg_strata,
            strata=["tgrade", "menostat"],
            duration_col="time",
            event_col="cens",
        )

        # get relative log hazard
        hz = coxphf.predict_partial_hazard(self.gbsg_strata).values
        log_hz = torch.log(torch.tensor(hz))

        # Create a torch tensor from the codes
        strata_comb = (
            self.gbsg_strata["tgrade"].astype(str)
            + "_"
            + self.gbsg_strata["menostat"].astype(str)
        )
        strata_code = strata_comb.astype("category").cat.codes
        strata = torch.tensor(strata_code.values, dtype=torch.long)

        # compute log_likelihood likelihood with torchsurv
        log_lik = -cox(log_hz, event, time, strata=strata, reduction="sum")

        # compute log likelihood with lifelines
        log_lik_lifelines = coxphf.log_likelihood_

        self.assertTrue(
            np.allclose(
                log_lik.numpy(),
                np.array(log_lik_lifelines),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_survival_function_lung_lifelines(self):
        """test value of estimated survival function with Breslow's method on lung data compared to lifelines"""

        time = torch.tensor(self.lung["time"].values, dtype=torch.float32)
        event = torch.tensor(self.lung["status"].values, dtype=torch.float32).bool()

        # fit
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.lung,
            duration_col="time",
            event_col="status",
        )

        # get relative log hazard
        hz = coxphf.predict_partial_hazard(self.lung).values
        log_hz = torch.tensor(np.log(hz), dtype=torch.float32)

        # compute baseline survival with lifelines
        bsf_coxphfitter = coxphf.baseline_survival_.values[:, 0]

        # compute survival function with lifelines
        surv_lifelines = coxphf.predict_survival_function(self.lung)
        surv_lifelines_numpy = surv_lifelines.T.to_numpy()
        new_time = torch.tensor(surv_lifelines.index.values)

        # compute baseline cumulative hazard and survival with torchsurv
        baseline_survival = baseline_survival_function(log_hz, event, time)
        bsf = baseline_survival["baseline_survival"]

        # compute survival function with torchsurv
        sf = survival_function(baseline_survival, log_hz, new_time)

        self.assertTrue(
            np.allclose(
                bsf_coxphfitter,
                np.array(bsf),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                surv_lifelines_numpy,
                np.array(sf),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_survival_function_gbsg_lifelines(self):
        """test value of estimated survival function with Breslow's method on gbsg data compared to lifelines"""

        event = torch.tensor(self.gbsg["cens"]).bool()
        time = torch.tensor(self.gbsg["time"], dtype=torch.float32)

        # fit
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.gbsg,
            duration_col="time",
            event_col="cens",
        )

        # get relative log hazard
        hz = coxphf.predict_partial_hazard(self.gbsg).values
        log_hz = torch.tensor(np.log(hz), dtype=torch.float32)

        # compute baseline survival with lifelines
        bsf_coxphfitter = coxphf.baseline_survival_.values[:, 0]

        # compute survival function with lifelines
        surv_lifelines = coxphf.predict_survival_function(self.gbsg_strata)
        surv_lifelines_numpy = surv_lifelines.T.to_numpy()
        new_time = torch.tensor(surv_lifelines.index.values)

        # compute baseline cumulative hazard and survival with torchsurv
        baseline_survival = baseline_survival_function(log_hz, event, time)
        bsf = baseline_survival["baseline_survival"]

        # compute survival function with torchsurv
        sf = survival_function(baseline_survival, log_hz, new_time)

        self.assertTrue(
            np.allclose(
                bsf_coxphfitter,
                np.array(bsf),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                surv_lifelines_numpy,
                np.array(sf),
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_survival_function_strata_lifelines(self):
        """test survival function with strata on gbsg datasets compared to lifelines"""

        time = torch.tensor(self.gbsg_strata["time"].values, dtype=torch.float32)
        event = torch.tensor(
            self.gbsg_strata["cens"].values, dtype=torch.float32
        ).bool()

        # fit (Ties are handled using Efron’s method on lifelines)
        coxphf = CoxPHFitter()
        coxphf.fit(
            self.gbsg_strata,
            strata=["tgrade", "menostat"],
            duration_col="time",
            event_col="cens",
        )

        # get relative log hazard
        hz = coxphf.predict_partial_hazard(self.gbsg_strata).values
        log_hz = torch.log(torch.tensor(hz))

        # compute baseline survival with lifelines
        bsf_coxphfitter = coxphf.baseline_survival_.values

        # predicted survival function lifelines
        surv_lifelines = coxphf.predict_survival_function(self.gbsg_strata)
        surv_lifelines_numpy = surv_lifelines.T.to_numpy()
        new_time = torch.tensor(surv_lifelines.index.values)

        # Create strata
        strata_comb = (
            self.gbsg_strata["tgrade"].astype(str)
            + "_"
            + self.gbsg_strata["menostat"].astype(str)
        )
        strata_code = strata_comb.astype("category").cat.codes
        strata = torch.tensor(strata_code.values, dtype=torch.long)

        # compute baseline survival with torchsurv
        baseline_survival = baseline_survival_function(
            log_hz, event, time, strata=strata
        )

        # predict survival function torchsurv
        surv = survival_function(baseline_survival, log_hz, new_time, strata)

        # the baseline survival function correspond but not the individual survival function
        # the formula S0 ** h is respected on torchsurv. Not sure what lifelines does.
        # also the values are more reasonable in torchsurv compared to no strata.

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 4]),
                np.unique(baseline_survival[0]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 5]),
                np.unique(baseline_survival[1]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 2]),
                np.unique(baseline_survival[2]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 3]),
                np.unique(baseline_survival[3]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 0]),
                np.unique(baseline_survival[4]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )

        self.assertTrue(
            np.allclose(
                np.unique(bsf_coxphfitter[:, 1]),
                np.unique(baseline_survival[5]["baseline_survival"].numpy()),
                rtol=1e-5,
                atol=1e-8,
            )
        )


if __name__ == "__main__":
    unittest.main()
