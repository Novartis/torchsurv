import pytest
import torch

from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw


@pytest.mark.parametrize(
    ("time_dtype", "weight_dtype"),
    [
        (torch.float64, None),  # weight from get_ipcw, which returns float32
        (torch.int64, None),  # integer follow-up time, e.g. days
        (torch.float32, torch.float64),  # weights computed elsewhere in float64
    ],
)
@pytest.mark.parametrize("tmax", [None, 150.0])
def test_uno_cindex_accepts_time_and_weight_of_different_dtypes(time_dtype, weight_dtype, tmax):
    torch.manual_seed(42)
    n = 64
    time = torch.randint(low=5, high=250, size=(n,))
    event = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
    estimate = torch.randn((n,), dtype=torch.float)

    ipcw = get_ipcw(event, time.float())
    expected = ConcordanceIndex()(estimate, event, time.float(), weight=ipcw, tmax=tmax)

    weight = ipcw if weight_dtype is None else ipcw.to(weight_dtype)
    tmax_ = None if tmax is None else torch.tensor(tmax, dtype=time_dtype)
    result = ConcordanceIndex()(estimate, event, time.to(time_dtype), weight=weight, tmax=tmax_)

    torch.testing.assert_close(result.float(), expected)
