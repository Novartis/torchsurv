from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def n_samples() -> int:
    return 64


@pytest.fixture(scope="session")
def seed() -> int:
    return 42


@pytest.fixture
def survival_data(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (event: bool Tensor, time: float Tensor)."""
    torch.manual_seed(42)
    event = torch.randint(0, 2, (n_samples,), dtype=torch.bool)
    time = torch.randint(1, 100, (n_samples,), dtype=torch.float)
    return event, time


@pytest.fixture
def survival_data_with_strata(n_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (event: bool Tensor, time: float Tensor, strata: int64 Tensor)."""
    torch.manual_seed(42)
    event = torch.randint(0, 2, (n_samples,), dtype=torch.bool)
    time = torch.randint(1, 100, (n_samples,), dtype=torch.float)
    strata = torch.randint(0, 3, (n_samples,), dtype=torch.long)
    return event, time, strata


@pytest.fixture
def cox_log_hz(n_samples: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(n_samples, dtype=torch.float)


@pytest.fixture
def weibull_log_params(n_samples: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(n_samples, 2, dtype=torch.float)


@pytest.fixture
def survival_log_hz(n_samples: int) -> torch.Tensor:
    torch.manual_seed(42)
    n_times = 10
    return torch.randn(n_samples, n_times, dtype=torch.float)


@pytest.fixture(scope="session")
def benchmark_loader() -> Callable[[str], dict]:
    """Returns a loader for benchmark JSON files from tests/benchmark_data/."""
    benchmark_dir = Path(__file__).parent / "benchmark_data"

    def _load(filename: str) -> dict:
        with open(benchmark_dir / filename) as f:
            return json.load(f)

    return _load
