import json
import unittest

import numpy as np
import torch
from sksurv.metrics import CensoringDistributionEstimator, SurvivalFunctionEstimator
from utils import DataBatchContainer

# local
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator

# Load the benchmark cox log likelihoods from R
with open("tests/benchmark_data/benchmark_kaplan_meier.json", "r") as file:
    benchmark_kaplan_meiers = json.load(file)

torch.manual_seed(23)


class TestNonParametric(unittest.TestCase):
    """
    List of packages compared
        - survival (R)
        - sksurv (Python)
    """

    def test_kaplan_meier_survival_distribution_real_data(self):
        """test Kaplan Meier survival distribution estimate on lung and gbsg datasets"""

        for benchmark_kaplan_meier in benchmark_kaplan_meiers:
            event = torch.tensor(benchmark_kaplan_meier["status"]).bool()
            time = torch.tensor(benchmark_kaplan_meier["time"], dtype=torch.float32)
            new_time = torch.tensor(
                benchmark_kaplan_meier["times"], dtype=torch.float32
            )

            km = KaplanMeierEstimator()
            km(event, time, censoring_dist=False)
            st = km.predict(new_time)

            st_survival = np.array(benchmark_kaplan_meier["surv_prob_survival"])

            self.assertTrue(
                np.allclose(
                    st.numpy(),
                    st_survival,
                    rtol=1e-3,
                    atol=1e-8,
                )
            )

    def test_kaplan_meier_censoring_distribution_simulated_data(self):
        """test Kaplan Meier estimate of censoring distribution on simulated batches including edge cases"""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=20,
            flags_to_set=[
                "train_ties_time_event",
                "train_ties_time_censoring",
                "train_ties_time_event_censoring",
                "train_no_censoring",
            ],
        )
        for batch in batch_container.batches:
            (time, event, _, _, _, _, y_array, *_) = batch

            km = KaplanMeierEstimator()
            km(event, time, censoring_dist=True)
            ct = km.km_est

            cens = CensoringDistributionEstimator()
            cens.fit(y_array)
            ct_sksurv = cens.prob_
            if not event.all():
                ct_sksurv = ct_sksurv[1:]

            self.assertTrue(np.allclose(ct.numpy(), ct_sksurv, rtol=1e-5, atol=1e-8))

    def test_kaplan_meier_survival_distribution_simulated_data(self):
        """test Kaplan Meier estimate of survival distribution on simulated batches including edge cases"""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=20,
            flags_to_set=[
                "train_ties_time_event",
                "train_ties_time_censoring",
                "train_ties_time_event_censoring",
                "train_no_censoring",
            ],
        )
        for batch in batch_container.batches:
            (time, event, _, _, _, _, y_array, *_) = batch

            km = KaplanMeierEstimator()
            km(event, time, censoring_dist=False)
            st = km.km_est

            surv = SurvivalFunctionEstimator()
            surv.fit(y_array)
            st_sksurv = surv.prob_[1:]

            self.assertTrue(np.allclose(st.numpy(), st_sksurv, rtol=1e-5, atol=1e-8))

    def test_kaplan_meier_predict_censoring_distribution_simulated_data(self):
        """test Kaplan Meier prediction of censoring distribution on simulated batches including edge cases"""
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
                #'train_no_censoring', CensoringDistributionEstimator from sksurv fails
                "test_event_at_last_time",
            ],
        )
        for batch in batch_container.batches:
            (
                train_time,
                train_event,
                test_time,
                _,
                _,
                _,
                y_train_array,
                y_test_array,
                *_,
            ) = batch

            km = KaplanMeierEstimator()
            km(train_event, train_time, censoring_dist=True)
            ct_pred = km.predict(test_time)

            cens = CensoringDistributionEstimator()
            cens.fit(y_train_array)
            ct_pred_sksurv = cens.predict_proba(y_test_array["futime"])

            self.assertTrue(
                np.allclose(ct_pred.numpy(), ct_pred_sksurv, rtol=1e-5, atol=1e-8)
            )

    def test_kaplan_meier_predict_survival_distribution_simulated_data(self):
        """test Kaplan Meier prediction of survival distribution on simulated batches including edge cases"""
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
                #'train_no_censoring', CensoringDistributionEstimator from sksurv fails
                "test_event_at_last_time",
            ],
        )
        for batch in batch_container.batches:
            (
                train_time,
                train_event,
                test_time,
                _,
                _,
                _,
                y_train_array,
                y_test_array,
                *_,
            ) = batch

            km = KaplanMeierEstimator()
            km(train_event, train_time, censoring_dist=False)
            st_pred = km.predict(test_time)

            surv = SurvivalFunctionEstimator()
            surv.fit(y_train_array)
            st_pred_sksurv = surv.predict_proba(y_test_array["futime"])

            self.assertTrue(
                np.allclose(st_pred.numpy(), st_pred_sksurv, rtol=1e-5, atol=1e-8)
            )

    def test_kaplan_meier_estimate_error_raised(self):
        """test that errors are raised for estimation in not-accepted edge cases."""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=1,
            flags_to_set=["train_all_censored"],
        )
        for batch in batch_container.batches:
            (train_time, train_event, *_) = batch

            self.assertRaises(
                ValueError, KaplanMeierEstimator(), train_event, train_time
            )

    def test_kaplan_meier_prediction_error_raised(self):
        """test that errors are raised for prediction in not-accepted edge cases."""
        batch_container = DataBatchContainer()
        batch_container.generate_batches(
            n_batch=1,
            flags_to_set=["test_max_time_gt_train_max_time"],
        )
        for batch in batch_container.batches:
            (train_time, train_event, test_time, *_) = batch

            train_event[-1] = (
                False  # if last event is censoring, the last KM is > 0 and it cannot predict beyond this time
            )
            km = KaplanMeierEstimator()
            km(train_event, train_time, censoring_dist=False)

            self.assertRaises(ValueError, km.predict, test_time)


if __name__ == "__main__":
    unittest.main()
