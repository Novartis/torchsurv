import json
import os
import unittest

import numpy as np
import torch

from torchsurv.loss.cox import neg_partial_log_likelihood as cox
from torchsurv.tools.validate_data import validate_loss

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
    """

    # random data and parameters
    N = 32
    log_hz = torch.randn(N)
    event = torch.randint(low=0, high=2, size=(N,)).bool()
    time = torch.randint(low=1, high=100, size=(N,))

    def test_y_tensor(self):
        event_np_array = np.random.randint(0, 1 + 1, size=(self.N,), dtype="bool")
        with self.assertRaises((RuntimeError, TypeError)):
            cox(self.log_hz, event_np_array, self.time)

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
        time_wrong_len = torch.randint(low=1, high=100, size=(self.N + 1,))
        with self.assertRaises(TypeError):
            validate_loss(self.log_hz, self.event, time_wrong_len)

    def test_positive_t(self):
        time_negative = torch.randint(low=-100, high=100, size=(self.N,))
        with self.assertRaises(TypeError):
            validate_loss(self.log_hz, self.event, time_negative)

    def test_boolean_y(self):
        event_non_boolean = torch.randint(low=0, high=3, size=(self.N,))
        with self.assertRaises(TypeError):
            validate_loss(self.log_hz, event_non_boolean, self.time)

    def test_log_likelihood_without_ties(self):
        """test cox partial log likelihood without ties on lung and gbsg datasets"""
        for benchmark_cox_loglik in benchmark_cox_logliks:
            if benchmark_cox_loglik["no_ties"][0] == True:
                log_lik = -cox(
                    torch.tensor(
                        benchmark_cox_loglik["log_hazard"], dtype=torch.float32
                    ).squeeze(0),
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

    def test_log_likelihood_with_ties(self):
        """test Efron and Breslow's approximation of cox partial log likelihood with ties on lung and gbsg data"""
        for benchmark_cox_loglik in benchmark_cox_logliks:
            if benchmark_cox_loglik["no_ties"][0] == False:
                # efron approximation of partial log likelihood
                log_lik_efron = -cox(
                    torch.tensor(
                        benchmark_cox_loglik["log_hazard_efron"], dtype=torch.float32
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
                        benchmark_cox_loglik["log_hazard_breslow"], dtype=torch.float32
                    ).squeeze(0),
                    torch.tensor(benchmark_cox_loglik["status"]).bool(),
                    torch.tensor(benchmark_cox_loglik["time"], dtype=torch.float32),
                    ties_method="breslow",
                    reduction="sum",
                )
                log_lik_breslow_survival = benchmark_cox_loglik[
                    "log_likelihood_breslow"
                ]

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


if __name__ == "__main__":
    unittest.main()
