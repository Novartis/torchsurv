import pytest_benchmark as benchmark
import torch
from torchsurv.loss.cox import neg_partial_log_likelihood

# set seed for reproducibility
torch.manual_seed(42)


log_hz = torch.randn(1000, dtype=torch.float32)  # log hazards
event = torch.ones(1000, dtype=torch.float32)  # all events observed
time = torch.arange(1, 1001, dtype=torch.float32)  # survival times


def my_function(log_hz, event, time) -> torch.Tensor:
    return neg_partial_log_likelihood(log_hz, event, time)


def test_benchmark(benchmark: benchmark) -> None:
    """Benchmark the neg_partial_log_likelihood function."""
    # Benchmark the function
    benchmark(my_function, log_hz, event, time)
    # assert result == 123
