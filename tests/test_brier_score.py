# global modules
import json
import unittest

import numpy as np
import torch
from sksurv.metrics import brier_score as brier_score_sksurv
from sksurv.metrics import integrated_brier_score as integrated_brier_score_sksurv
from utils import DataBatchContainer, conditions_ci, conditions_p_value

# Local modules
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.stats.ipcw import get_ipcw

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the benchmark metrics from R
with open("tests/benchmark_data/benchmark_brier_score.json") as file:
    benchmark_brier_scores = json.load(file)

brier_score = BrierScore()


class TestBrierScore(unittest.TestCase):
    """
    List of packages compared
        - survMetrics (R)
        - survAUC (R)
        - sksurv (Python)
    """

    def test_brier_score_real_data(self):
        """test point estimate of brier score and integrated brier score on lung and gbsg datasets"""
        for benchmark_brier_score in benchmark_brier_scores:
            train_event = torch.tensor(benchmark_brier_score["train_status"]).bool()
            train_time = torch.tensor(
                benchmark_brier_score["train_time"], dtype=torch.float32
            )
            test_event = torch.tensor(benchmark_brier_score["test_status"]).bool()
            test_time = torch.tensor(
                benchmark_brier_score["test_time"], dtype=torch.float32
            )
            estimate = torch.tensor(
                benchmark_brier_score["estimate"], dtype=torch.float32
            )
            new_time = torch.tensor(benchmark_brier_score["times"], dtype=torch.float32)

            # ipcw obtained using censoring distribution estimated on train data
            ipcw = get_ipcw(train_event, train_time, test_time)
            ipcw_new_time = get_ipcw(train_event, train_time, new_time)

            bs = brier_score(
                estimate,
                test_event,
                test_time,
                new_time=new_time,
                weight=ipcw,
                weight_new_time=ipcw_new_time,
            )
            ibs = brier_score.integral()

            bs_survAUC = np.array(benchmark_brier_score["brier_score_survAUC"])
            ibs_survAUC = np.array(benchmark_brier_score["ibrier_score_survAUC"])

            # commented out: values far off compared to other packages
            # self.assertTrue(np.allclose(bs.numpy(), bs_survAUC, rtol=1e-5, atol=1e-8))
            # self.assertTrue(np.allclose(ibs.numpy(), ibs_survAUC, rtol=1e-5, atol=1e-8))

            # ipcw obtained using censoring distribution estimated on test data
            ipcw = get_ipcw(test_event, test_time)
            ipcw_new_time = get_ipcw(test_event, test_time, new_time)

            bs = brier_score(
                estimate,
                test_event,
                test_time,
                new_time=new_time,
                weight=ipcw,
                weight_new_time=ipcw_new_time,
            )
            ibs = brier_score.integral()

            bs_survMetrics = np.array(benchmark_brier_score["brier_score_survMetrics"])
            ibs_survMetrics = np.array(
                benchmark_brier_score["ibrier_score_survMetrics"]
            )

            self.assertTrue(
                np.allclose(bs.numpy(), bs_survMetrics, rtol=1e-2, atol=1e-3)
            )
            self.assertTrue(
                np.allclose(ibs.numpy(), ibs_survMetrics, rtol=1e-2, atol=1e-3)
            )

    def test_brier_score_simulated_data(self):
        """test point estimate of brier score and integrated brier score on simulated batches including edge cases"""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=20,
            flags_to_set=[
                "train_ties_time_event",
                "test_ties_time_event",
                "train_ties_time_censoring",
                "test_ties_time_censoring",
                "train_ties_time_event_censoring",
                "test_ties_time_event_censoring",
                "test_no_censoring",
                # "train_no_censoring", sksurv fails
                "test_event_at_last_time",
                "ties_score_events",
                "ties_score_event_censoring",
                "ties_score_censoring",
            ],
        )
        for batch in batch_container.batches:
            (
                train_time,
                train_event,
                test_time,
                test_event,
                _,
                new_time,
                y_train_array,
                y_test_array,
                _,
                new_time_array,
            ) = batch

            estimate = torch.rand((len(test_time), len(new_time)))

            ipcw = get_ipcw(train_event, train_time, test_time)
            ipcw_new_time = get_ipcw(train_event, train_time, new_time)
            bs = brier_score(
                estimate,
                test_event,
                test_time,
                new_time=new_time,
                weight=ipcw,
                weight_new_time=ipcw_new_time,
            )

            _, bs_sksurv = brier_score_sksurv(
                y_train_array, y_test_array, estimate.numpy(), new_time_array
            )

            self.assertTrue(np.allclose(bs.numpy(), bs_sksurv, rtol=1e-5, atol=1e-8))

            if len(new_time) > 2:
                ibs = brier_score.integral()
                ibs_sksurv = integrated_brier_score_sksurv(
                    y_train_array, y_test_array, estimate.numpy(), new_time_array
                )
                self.assertTrue(
                    np.allclose(ibs.numpy(), ibs_sksurv, rtol=1e-5, atol=1e-8)
                )

    def test_brier_score_confidence_interval_pvalue(self):
        """test brier score confidence interval and p value are as expected"""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=20,
            flags_to_set=[
                "test_ties_time_event",
                "test_ties_time_censoring",
                "test_ties_time_event_censoring",
                #'test_no_censoring', concordance_index_censored fails
                "test_event_at_last_time",
                "ties_score_events",
                "ties_score_event_censoring",
                "ties_score_censoring",
            ],
        )
        for batch in batch_container.batches:
            (_, _, time, event, _, new_time, *_) = batch

            estimate = torch.rand((len(time), len(new_time)))

            # Run c-index
            brier_score(estimate, event, time, new_time)

            n_bootstraps = 9

            for method in ["parametric", "bootstrap"]:
                for alternative in ["two_sided", "less", "greater"]:
                    brier_score_ci = brier_score.confidence_interval(
                        method=method,
                        alternative=alternative,
                        n_bootstraps=n_bootstraps,
                    )
                    self.assertTrue(
                        all(
                            [
                                conditions_ci(brier_score_ci[:, i])
                                for i in range(len(brier_score.brier_score))
                            ]
                        )
                    )

            for alternative in ["two_sided", "less", "greater"]:
                brier_score_pvalue = brier_score.p_value(
                    method="bootstrap",
                    alternative=alternative,
                    n_bootstraps=n_bootstraps,
                )
                self.assertTrue(
                    all(
                        [
                            conditions_p_value(brier_score_pvalue[i])
                            for i in range(len(brier_score.brier_score))
                        ]
                    )
                )
                brier_score_pvalue = brier_score.p_value(
                    method="parametric",
                    alternative=alternative,
                    n_bootstraps=n_bootstraps,
                    null_value=0.3,
                )
                self.assertTrue(
                    all(
                        [
                            conditions_p_value(brier_score_pvalue[i])
                            for i in range(len(brier_score.brier_score))
                        ]
                    )
                )

    def test_brier_score_compare(self):
        """test compare function of brier score behaves as expected"""

        _ = torch.manual_seed(42)
        n = 128
        m = 100
        estimate_informative = torch.rand((n,))  # estimate used to define time-to-event
        estimate_non_informative = (
            torch.rand((n,)).unsqueeze(1).expand(n, n)
        )  # random estimate
        event = torch.randint(low=0, high=2, size=(n,)).bool()
        time = torch.randn(size=(n,)) + estimate_informative * 20.0 + 200

        estimate_informative = estimate_informative.unsqueeze(1).expand(n, n)

        mask = event & (time < torch.max(time))
        new_time, inverse_indices, counts = torch.unique(
            time[mask], sorted=True, return_inverse=True, return_counts=True
        )
        sorted_unique_indices = BrierScore._find_torch_unique_indices(
            inverse_indices, counts
        )
        estimate_informative = (estimate_informative[:, mask])[:, sorted_unique_indices]
        estimate_non_informative = (estimate_non_informative[:, mask])[
            :, sorted_unique_indices
        ]

        ipcw = get_ipcw(event, time)  # ipcw weights at subject event time
        ipcw_new_time = get_ipcw(event, time, new_time)  # ipcw weights at new_time

        brier_score_informative = BrierScore()
        bs_informative = brier_score_informative(
            estimate_informative, event, time, new_time, ipcw, ipcw_new_time
        )[0:-1]
        ibs_informative = brier_score_informative.integral()

        brier_score_non_informative = BrierScore()
        bs_non_informative = brier_score_non_informative(
            estimate_non_informative, event, time, new_time, ipcw, ipcw_new_time
        )[0:-1]
        ibs_non_informative = brier_score_non_informative.integral()

        p_value_compare_informative = brier_score_informative.compare(
            brier_score_non_informative
        )
        p_value_compare_non_informative = brier_score_non_informative.compare(
            brier_score_informative
        )

        self.assertTrue(np.all(bs_informative.numpy() < bs_non_informative.numpy()))
        self.assertTrue(np.all(ibs_informative.numpy() < ibs_non_informative.numpy()))
        self.assertTrue(np.any(p_value_compare_informative.numpy() < 0.05))
        self.assertTrue(np.all(p_value_compare_non_informative.numpy() > 0.05))


if __name__ == "__main__":
    unittest.main()
