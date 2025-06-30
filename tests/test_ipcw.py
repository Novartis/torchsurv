# global modules
import json
import unittest

import numpy as np
import torch
from sksurv.nonparametric import CensoringDistributionEstimator
from utils import DataBatchContainer

# Local modules
from torchsurv.stats.ipcw import get_ipcw

# Load the benchmark cox log likelihoods from R
with open("tests/benchmark_data/benchmark_ipcw.json") as file:
    benchmark_ipcws = json.load(file)

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TestIPCW(unittest.TestCase):
    """
    List of packages compared
        - pec (R)
        - sksurv (Python)
    """

    def test_ipcw_real_data(self):
        """test ipcw values on lung and gbsg data"""

        for benchmark_ipcw in benchmark_ipcws:
            event = torch.tensor(benchmark_ipcw["status"]).bool()
            time = torch.tensor(benchmark_ipcw["time"], dtype=torch.float32)
            new_time = torch.tensor(benchmark_ipcw["times"], dtype=torch.float32)

            # ipcw evaluated at subject time
            ipcw_subject_time = get_ipcw(event, time)
            ipcw_subject_time_pec = np.array(benchmark_ipcw["ipcw_subjectimes"])

            # ipcw evaluated at new time
            ipcw_new_time = get_ipcw(event, time, new_time)
            ipcw_new_time_pec = np.array(benchmark_ipcw["ipcw_times"])

            self.assertTrue(
                np.allclose(
                    ipcw_subject_time.numpy(),
                    ipcw_subject_time_pec,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )

            self.assertTrue(
                np.allclose(
                    ipcw_new_time.numpy(), ipcw_new_time_pec, rtol=1e-4, atol=1e-8
                )
            )

    def test_ipcw_simulated_data(self):
        """test ipcw on simulated batches including edge cases"""
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
                _,
                _,
                y_train_array,
                y_test_array,
                _,
                _,
            ) = batch

            # ipcw
            ipcw = get_ipcw(train_event, train_time, test_time)

            # sksurv imposes survival data (event and time) for ipcw prediction
            # instead of just time. And then force icpw to be 0 if event == False
            ipcw[test_event == False] = 0.0

            # ipcw with sksurv
            cens = CensoringDistributionEstimator()
            cens.fit(y_train_array)
            ipcw_sksurv = cens.predict_ipcw(y_test_array)

            self.assertTrue(
                np.all(np.isclose(ipcw.numpy(), ipcw_sksurv, rtol=1e-4, atol=1e-8))
            )


if __name__ == "__main__":
    unittest.main()
