# global modules
import json
import unittest

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from utils import DataBatchContainer, conditions_ci, conditions_p_value

# Local modules
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the benchmark metrics from R
with open("tests/benchmark_data/benchmark_cindex.json") as file:
    benchmark_cindexs = json.load(file)


cindex = ConcordanceIndex()


class TestCIndex(unittest.TestCase):
    """
    List of packages compared
        - survival (R): Harrell's C index
        - survcomp (R): Harrell's C index
        - survAUC (R): Uno's C-index
        - survC1 (R): Uno's C-index
        - sksurv (Python): Harrell's C index and Uno's C-index
    """

    def test_cindex_real_data(self):
        """test point estimate concordance index on lung and gbsg datasets"""
        for benchmark_cindex in benchmark_cindexs:
            train_event = torch.tensor(benchmark_cindex["train_status"]).bool()
            train_time = torch.tensor(
                benchmark_cindex["train_time"], dtype=torch.float32
            )
            test_event = torch.tensor(benchmark_cindex["test_status"]).bool()
            test_time = torch.tensor(benchmark_cindex["test_time"], dtype=torch.float32)
            estimate = torch.tensor(benchmark_cindex["estimate"], dtype=torch.float32)
            new_time = torch.tensor(benchmark_cindex["times"], dtype=torch.float32)

            # harrell's c-index
            c_harrell = cindex(
                estimate,
                test_event,
                test_time,
            ).numpy()

            c_harrell_survival = np.array(
                benchmark_cindex["c_Harrell_survival"]
            )  # survival
            c_harrell_survcomp = np.array(
                benchmark_cindex["c_Harrell_survcomp"]
            )  # survcomp

            # uno's c-index
            ipcw = get_ipcw(train_event, train_time, test_time)
            c_uno = cindex(
                estimate, test_event, test_time, weight=ipcw, tmax=new_time[-1]
            )

            c_uno_survAUC = benchmark_cindex["c_Uno_survAUC"]  # survAUC
            c_uno_survC1 = benchmark_cindex["c_Uno_survC1"]  # survC1

            self.assertTrue(
                np.isclose(
                    c_harrell,
                    c_harrell_survival,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )

            self.assertTrue(
                np.isclose(
                    c_harrell,
                    c_harrell_survcomp,
                    rtol=1e-2,
                    atol=1e-8,
                )
            )

            self.assertTrue(
                np.isclose(c_uno.numpy(), np.array(c_uno_survAUC), rtol=1e-1, atol=1e-8)
            )

            self.assertTrue(
                np.isclose(c_uno.numpy(), np.array(c_uno_survC1), rtol=1e-1, atol=1e-8)
            )

    def test_cindex_se_real_data(self):
        """test standard error of concordance index on lung and gbsg datasets"""
        for benchmark_cindex in benchmark_cindexs:
            concordant = torch.tensor(benchmark_cindex["ch_survcomp"], dtype=torch.int)
            discordant = torch.tensor(benchmark_cindex["dh_survcomp"], dtype=torch.int)
            weight = torch.tensor(
                benchmark_cindex["weights_survcomp"], dtype=torch.float32
            )
            c_harrell_survcomp = torch.tensor(
                benchmark_cindex["c_Harrell_survcomp"], dtype=torch.float32
            )

            # overwrite objects used to calculate standard error
            cindex.concordant = concordant
            cindex.discordant = discordant
            cindex.weight = weight
            cindex.cindex = c_harrell_survcomp

            # Noether standard error
            cindex_se = cindex._concordance_index_se()
            cindex_se_survcomp = np.array(
                benchmark_cindex["c_se_noether_survcomp"]
            )  # survcomp

            # conservative confidence interval
            cindex_lower = cindex._confidence_interval_conservative(
                alpha=0.05, alternative="two_sided"
            )[0]
            cindex_lower_survcomp = np.array(
                benchmark_cindex["c_lower_conservative_survcomp"]
            )  # survcomp

            self.assertTrue(
                np.isclose(
                    cindex_se.numpy(),
                    np.array(cindex_se_survcomp),
                    rtol=1e-1,
                    atol=1e-8,
                )
            )

            self.assertTrue(
                np.isclose(
                    cindex_lower.numpy(),
                    np.array(cindex_lower_survcomp),
                    rtol=1e-1,
                    atol=1e-8,
                )
            )

    def test_cindex_simulated_data(self):
        """test estimate of concordance index on simulated batches including edge cases"""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=20,
            flags_to_set=[
                "train_ties_time_event",
                "test_ties_time_event",
                "train_ties_time_censoring",
                # "test_ties_time_censoring",
                "train_ties_time_event_censoring",
                "test_ties_time_event_censoring",
                "test_no_censoring",
                #'train_no_censoring', concordance_index_ipcw from sksurv fails
                "test_event_at_last_time",
                "ties_score_events",
                "ties_score_event_censoring",
                "ties_score_censoring",
            ],
        )
        for _, batch in enumerate(batch_container.batches):
            (
                train_time,
                train_event,
                test_time,
                test_event,
                estimate,
                new_time,
                y_train_array,
                y_test_array,
                estimate_array,
                new_time_array,
            ) = batch

            # harrell's c index
            c_harrell = cindex(
                estimate,
                test_event,
                test_time,
            )
            c_harrell_sksurv = concordance_index_censored(
                y_test_array["survival"], y_test_array["futime"], estimate_array
            )[0]

            # uno's c-index
            ipcw = get_ipcw(train_event, train_time, test_time)
            c_uno = cindex(
                estimate, test_event, test_time, weight=ipcw, tmax=new_time[-1]
            ).numpy()

            c_uno_sksurv = concordance_index_ipcw(
                y_train_array, y_test_array, estimate_array, tau=new_time_array[-1]
            )[0]

            self.assertTrue(
                np.isclose(c_harrell.numpy(), c_harrell_sksurv, rtol=1e-2, atol=1e-8)
            )
            self.assertTrue(np.isclose(c_uno, c_uno_sksurv, rtol=1e-2, atol=1e-8))

    def test_cindex_confidence_interval_pvalue(self):
        """test concordance index confidence interval and p value are as expected"""
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
            (_, _, time, event, estimate, *_) = batch

            # Run c-index
            cindex(
                estimate,
                event,
                time,
            )

            n_bootstraps = 9

            for method in ["noether", "conservative", "bootstrap"]:
                for alternative in ["two_sided", "less", "greater"]:
                    self.assertTrue(
                        conditions_ci(
                            cindex.confidence_interval(
                                method=method,
                                alternative=alternative,
                                n_bootstraps=n_bootstraps,
                            )
                        )
                    )

            for method in ["noether", "bootstrap"]:
                for alternative in ["two_sided", "less", "greater"]:
                    self.assertTrue(
                        conditions_p_value(
                            cindex.p_value(
                                method=method,
                                alternative=alternative,
                                n_bootstraps=n_bootstraps,
                            )
                        )
                    )

    def test_cindex_compare(self):
        "test compare function of cindex behaves as expected"

        _ = torch.manual_seed(42)
        n = 128
        estimate_informative = torch.randn(
            (n,)
        )  # estimate used to define time-to-event
        estimate_non_informative = torch.randn((n,))  # random estimate
        event = torch.randint(low=0, high=2, size=(n,)).bool()
        time = (
            torch.randn(size=(n,)) * 10 - estimate_informative * 5.0 + 200
        )  # + estimate for cindex < 0.5 and - for cindex > 0.5

        cindex_informative = ConcordanceIndex()
        c1 = cindex_informative(estimate_informative, event, time)

        cindex_non_informative = ConcordanceIndex()
        c2 = cindex_non_informative(estimate_non_informative, event, time)

        p_value_compare = cindex_informative.compare(cindex_non_informative)

        self.assertTrue(c1.numpy() > c2.numpy())
        self.assertTrue(p_value_compare < 0.05)

    def test_cindex_error_raised(self):
        """test that error are raised in not-accepted edge cases."""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=1,
            flags_to_set=["test_all_censored"],
        )
        for batch in batch_container.batches:
            (_, _, test_time, test_event, estimate, *_) = batch

            self.assertRaises(ValueError, cindex, estimate, test_event, test_time)


if __name__ == "__main__":
    unittest.main()
