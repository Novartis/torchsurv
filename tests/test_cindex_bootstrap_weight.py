from unittest import mock

import torch

from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw


def _simulated_data(n: int = 100):
    torch.manual_seed(1)
    estimate = torch.randn(n)
    event_time = torch.distributions.Exponential(torch.exp(estimate)).sample()
    censoring_time = torch.distributions.Exponential(torch.ones(n)).sample()
    time = torch.minimum(event_time, censoring_time)
    event = event_time <= censoring_time
    return estimate, event, time


def _permutation(low, high, size):
    """Stand-in for torch.randint: a resample that draws every subject once."""
    return torch.randperm(high)


def test_uno_bootstrap_resample_keeps_weight_with_subject():
    """A bootstrap replicate that contains every subject once must give back the
    point estimate. That only holds if each subject keeps its own IPCW weight."""
    estimate, event, time = _simulated_data()
    weight = get_ipcw(event, time)
    assert weight[event].max() > 2 * weight[event].min()  # weights differ enough to matter

    for tmax in (None, torch.quantile(time, 0.8)):
        cindex = ConcordanceIndex()
        uno = cindex(estimate, event, time, weight=weight, tmax=tmax)

        torch.manual_seed(0)
        with mock.patch("torch.randint", side_effect=_permutation):
            replicates = cindex._bootstrap_cindex(metric="confidence_interval", n_bootstraps=5)

        torch.testing.assert_close(replicates, uno.expand(5))
