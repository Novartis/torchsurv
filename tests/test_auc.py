# global modules
import json
import unittest

import numpy as np
import torch
from sksurv.metrics import cumulative_dynamic_auc
from utils import DataBatchContainer, conditions_ci, conditions_p_value

# Local modules
from torchsurv.metrics.auc import Auc
from torchsurv.stats.ipcw import get_ipcw

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the benchmark metrics from R
with open("tests/benchmark_data/benchmark_auc.json") as file:
    benchmark_aucs = json.load(file)

auc = Auc()


class TestAUC(unittest.TestCase):
    """_summary_
    List of packages compared
        - survAUC (R): AUC C/D with IPCW
                       Integral of AUC C/D
                       Integral of AUC I/D
        - RiskRegression (R): AUC C/D with IPCW
                              Standard error of AUC C/D with IPCW
        - timeROC (R): AUC C/D with IPCW
                       Standard error of AUC C/D with IPCW
        - sksurv (Python): AUC C/D with IPCW
    """

    def test_auc_cd_real_data(self):
        """test point estimate of auc cumulative/dynamic with ipcw on lung and gbsg datasets"""
        for benchmark_auc in benchmark_aucs:
            train_event = torch.tensor(benchmark_auc["train_status"]).bool()
            train_time = torch.tensor(benchmark_auc["train_time"], dtype=torch.float32)
            test_event = torch.tensor(benchmark_auc["test_status"]).bool()
            test_time = torch.tensor(benchmark_auc["test_time"], dtype=torch.float32)
            estimate = torch.tensor(benchmark_auc["estimate"], dtype=torch.float32)
            new_time = torch.tensor(benchmark_auc["times"], dtype=torch.float32)

            # ipcw obtained using censoring distribution estimated on train data
            ipcw = get_ipcw(train_event, train_time, test_time)
            ipcw_new_time = get_ipcw(train_event, train_time, new_time)
            auc_cd = auc(
                estimate,
                test_event,
                test_time,
                auc_type="cumulative",
                weight=ipcw,
                weight_new_time=ipcw_new_time,
                new_time=new_time,
            )

            auc_cd_survAUC = benchmark_auc["auc_cd_survAUC"]  # survAUC
            auc_cd_Uno_riskRegression = benchmark_auc[
                "auc_cd_Uno_riskRegression"
            ]  # riskRegression

            self.assertTrue(
                np.allclose(
                    auc_cd.numpy(), np.array(auc_cd_survAUC), rtol=1e-1, atol=1e-8
                )
            )
            self.assertTrue(
                np.allclose(
                    auc_cd.numpy(),
                    np.array(auc_cd_Uno_riskRegression),
                    rtol=1e-1,
                    atol=1e-2,
                )
            )

            # commented out: riskRegression does not match other R packages
            # auc_cd_se = auc._auc_se()

            # auc_cd_Uno_se_riskRegression = benchmark_auc[
            #     "auc_cd_Uno_se_riskRegression"
            # ]  # riskRegression

            # self.assertTrue(
            #     np.allclose(
            #         auc_cd_se.numpy(),
            #         np.array(auc_cd_Uno_se_riskRegression),
            #         rtol=1e-1,
            #         atol=1e-1,
            #     )
            # )

    def test_auc_cd_se_real_data(self):
        """test standard error of auc cumulative/dynamic with ipcw on lung and gbsg datasets"""
        for benchmark_auc in benchmark_aucs:
            test_event = torch.tensor(benchmark_auc["test_status"]).bool()
            test_time = torch.tensor(benchmark_auc["test_time"], dtype=torch.float32)
            estimate = torch.tensor(benchmark_auc["estimate"], dtype=torch.float32)
            new_time = torch.tensor(benchmark_auc["times_se"], dtype=torch.float32)

            # ipcw obtained using censoring distribution estimated on test data
            ipcw = get_ipcw(test_event, test_time, test_time)
            ipcw_new_time = get_ipcw(test_event, test_time, new_time)
            auc_cd = auc(
                estimate,
                test_event,
                test_time,
                auc_type="cumulative",
                weight=ipcw,
                weight_new_time=ipcw_new_time,
                new_time=new_time,
            )  # point estimate
            auc_cd_se = auc._auc_se()  # standard error

            # timeROC
            auc_cd_Uno_timeROC = benchmark_auc["auc_cd_Uno_timeROC"]  # point estimate
            auc_cd_Uno_se_timeROC = benchmark_auc[
                "auc_cd_Uno_se_timeROC"
            ]  # standard error

            self.assertTrue(
                np.allclose(
                    auc_cd.numpy(),
                    np.array(auc_cd_Uno_timeROC),
                    rtol=1e-1,
                    atol=1e-8,
                )
            )

            self.assertTrue(
                np.allclose(
                    auc_cd_se.numpy(),
                    np.array(auc_cd_Uno_se_timeROC),
                    rtol=1e-1,
                    atol=1e-8,
                )
            )

    def test_i_auc_real_data(self):
        """test integral of auc with survAUC on lung and gbsg datasets"""
        for benchmark_auc in benchmark_aucs:
            times = torch.tensor(benchmark_auc["times"], dtype=torch.float64)
            S = torch.tensor(benchmark_auc["surv.prob"], dtype=torch.float64)

            # integral of auc cumulative/dynamic
            auc_cd_survAUC = torch.tensor(
                benchmark_auc["auc_cd_survAUC"], dtype=torch.float64
            )
            auc_ne = Auc()
            auc_ne.auc = auc_cd_survAUC
            auc_ne.new_time = times
            i_auc_cd = auc_ne._integrate_cumulative(S, times.max())

            i_auc_cd_survAUC = benchmark_auc["iauc_cd_survAUC"]  # survAUC

            self.assertTrue(
                np.isclose(
                    i_auc_cd.numpy(), np.array(i_auc_cd_survAUC), rtol=1e-3, atol=1e-8
                )
            )

            # integral of auc incident/dynamic
            auc_id_sz_survAUC = torch.tensor(
                benchmark_auc["auc_id_sz_survAUC"], dtype=torch.float64
            )
            auc_ne = Auc()
            auc_ne.auc = auc_id_sz_survAUC
            auc_ne.new_time = times
            i_auc_id_sz = auc_ne._integrate_incident(S, times.max())
            i_auc_id_sz_survAUC = benchmark_auc["i_auc_id_sz_survAUC"]  # survAUC

            self.assertTrue(
                np.isclose(
                    i_auc_id_sz.numpy(),
                    np.array(i_auc_id_sz_survAUC),
                    rtol=1e-3,
                    atol=1e-8,
                )
            )

    def test_auc_cd_simulated_data(self):
        """test auc cumulative/dynamic on simulated batches including edge cases"""
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
                #'train_no_censoring', cumulative_dynamic_auc from sksurv fails
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
                estimate,
                new_time,
                y_train_array,
                y_test_array,
                _,
                new_time_array,
            ) = batch

            ipcw = get_ipcw(train_event, train_time, test_time)
            ipcw_new_time = get_ipcw(train_event, train_time, new_time)

            # expand in 2D
            estimate = estimate.unsqueeze(1).expand((len(test_time), len(new_time)))
            estimate = estimate + torch.randn_like(estimate) * 0.1

            auc_cd = auc(
                estimate,
                test_event,
                test_time,
                auc_type="cumulative",
                weight=ipcw,
                weight_new_time=ipcw_new_time,
                new_time=new_time,
            )

            auc_cd_sksurv, _ = cumulative_dynamic_auc(
                y_train_array, y_test_array, estimate.numpy(), new_time_array
            )  # sksurv

            self.assertTrue(
                np.allclose(auc_cd.numpy(), auc_cd_sksurv, rtol=1e-5, atol=1e-8)
            )

    def test_auc_confidence_interval_pvalue(self):
        """test auc confidence interval and p-value are as expected"""
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
            auc(
                estimate,
                event,
                time,
            )

            n_bootstraps = 9

            for method in ["blanche", "bootstrap"]:
                for alternative in ["two_sided", "less", "greater"]:
                    auc_ci = auc.confidence_interval(
                        method=method,
                        alternative=alternative,
                        n_bootstraps=n_bootstraps,
                    )
                    self.assertTrue(
                        all([conditions_ci(auc_ci[:, i]) for i in range(len(auc.auc))])
                    )

            for method in ["blanche", "bootstrap"]:
                for alternative in ["two_sided", "less", "greater"]:
                    auc_p_value = auc.p_value(
                        method=method,
                        alternative=alternative,
                        n_bootstraps=n_bootstraps,
                    )

                    self.assertTrue(
                        all(
                            [
                                conditions_p_value(auc_p_value[i])
                                for i in range(len(auc.auc))
                            ]
                        )
                    )

    def test_auc_compare(self):
        "test compare function of auc behavesas expected."
        _ = torch.manual_seed(42)
        n = 128
        estimate_informative = torch.randn(
            (n,)
        )  # estimate used to define time-to-event
        estimate_non_informative = torch.randn((n,))  # random estimate
        event = torch.randint(low=0, high=2, size=(n,)).bool()
        time = (
            torch.randn(size=(n,)) * 10 - estimate_informative * 5.0 + 200
        )  # + estimate for auc < 0.5 and - for auc > 0.5

        Auc_informative = Auc()
        auc_informative = Auc_informative(estimate_informative, event, time)

        Auc_non_informative = Auc()
        auc_non_informative = Auc_non_informative(estimate_non_informative, event, time)

        p_value_compare_informative = Auc_informative.compare(Auc_non_informative)
        p_value_compare_non_informative = Auc_non_informative.compare(Auc_informative)

        self.assertTrue(np.all(auc_informative.numpy() > auc_non_informative.numpy()))
        self.assertTrue(np.any(p_value_compare_informative.numpy() < 0.05))
        self.assertTrue(np.all(p_value_compare_non_informative.numpy() > 0.05))

    def test_auc_error_raised(self):
        """test that errors are raised in not-accepted edge cases."""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=2,
            flags_to_set=[
                "test_all_censored",
                "test_max_time_in_new_time",
            ],
        )
        for batch in batch_container.batches:
            _, _, test_time, test_event, estimate, new_time, *_ = batch

            self.assertRaises(
                ValueError,
                auc,
                estimate,
                test_event,
                test_time,
                new_time=new_time,
            )


if __name__ == "__main__":
    unittest.main()
