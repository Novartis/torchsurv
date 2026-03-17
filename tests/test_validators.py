from __future__ import annotations

import pytest
import torch

from torchsurv.tools.validators import (
    SurvivalInputs,
    impute_missing_log_shape,
)


N = 10


def _make_event(n: int = N) -> torch.Tensor:
    t = torch.zeros(n, dtype=torch.bool)
    t[0] = True
    return t


# ---------------------------------------------------------------------------
# SurvivalInputs — dtype coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "time_dtype,expected_dtype",
    [
        (torch.int32, torch.float32),
        (torch.int64, torch.float32),
        (torch.float64, torch.float32),
        (torch.float32, torch.float32),
    ],
    ids=["int32→float32", "int64→float32", "float64→float32", "float32→noop"],
)
def test_time_dtype_coercion(time_dtype: torch.dtype, expected_dtype: torch.dtype) -> None:
    """Integer and double time tensors are silently coerced to float32."""
    event = _make_event()
    time = torch.ones(N, dtype=time_dtype)
    inp = SurvivalInputs(event=event, time=time)
    assert inp.time.dtype == expected_dtype


def test_float32_time_passes_unchanged() -> None:
    """float32 time tensor passes through without modification."""
    event = _make_event()
    time = torch.ones(N, dtype=torch.float32) * 5.0
    inp = SurvivalInputs(event=event, time=time)
    assert torch.equal(inp.time, time)


# ---------------------------------------------------------------------------
# SurvivalInputs — ValueError cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "time_factory",
    [
        pytest.param(lambda: torch.full((N,), -1.0), id="all_negative"),
        pytest.param(lambda: torch.cat([torch.ones(N - 1), torch.tensor([-0.1])]), id="one_negative"),
    ],
)
def test_negative_time_raises(time_factory) -> None:  # type: ignore[no-untyped-def]
    """Negative time value raises ValueError naming the time field."""
    event = _make_event()
    with pytest.raises(ValueError, match="time"):
        SurvivalInputs(event=event, time=time_factory())


@pytest.mark.parametrize(
    "time_factory",
    [
        pytest.param(lambda: torch.full((N,), float("nan")), id="all_nan"),
        pytest.param(lambda: torch.cat([torch.ones(N - 1), torch.tensor([float("nan")])]), id="one_nan"),
    ],
)
def test_nan_in_time_raises(time_factory) -> None:  # type: ignore[no-untyped-def]
    """NaN in time tensor raises ValueError."""
    event = _make_event()
    with pytest.raises(ValueError, match="NaN"):
        SurvivalInputs(event=event, time=time_factory())


@pytest.mark.parametrize(
    "time_factory",
    [
        pytest.param(lambda: torch.full((N,), float("inf")), id="all_inf"),
        pytest.param(lambda: torch.cat([torch.ones(N - 1), torch.tensor([float("inf")])]), id="one_inf"),
    ],
)
def test_inf_in_time_raises(time_factory) -> None:  # type: ignore[no-untyped-def]
    """Inf in time tensor raises ValueError."""
    event = _make_event()
    with pytest.raises(ValueError, match="Inf|inf"):
        SurvivalInputs(event=event, time=time_factory())


def test_mismatched_event_time_lengths_raises() -> None:
    """Mismatched event/time lengths raise ValueError with 'mismatch' in message."""
    event = _make_event(N)
    time = torch.ones(N + 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="[Dd]imension|mismatch|length"):
        SurvivalInputs(event=event, time=time)


# ---------------------------------------------------------------------------
# Device preservation (CPU only; CUDA tested if available)
# ---------------------------------------------------------------------------


def test_cpu_device_preserved_after_coercion() -> None:
    """CPU int tensor coerced to float remains on CPU."""
    event = _make_event()
    time_int = torch.ones(N, dtype=torch.int32)
    inp = SurvivalInputs(event=event, time=time_int)
    assert inp.time.device.type == "cpu"
    assert inp.time.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preserved_after_coercion() -> None:
    """CUDA tensor coerced to float remains on the same CUDA device."""
    event = _make_event().cuda()
    time_int = torch.ones(N, dtype=torch.int32).cuda()
    inp = SurvivalInputs(event=event, time=time_int)
    assert inp.time.device.type == "cuda"
    assert inp.time.dtype == torch.float32


# ---------------------------------------------------------------------------
# impute_missing_log_shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,expected_shape",
    [
        ((N,), (N, 2)),
        ((N, 1), (N, 2)),
        ((N, 2), (N, 2)),
    ],
    ids=["1D→(n,2)", "(n,1)→(n,2)", "(n,2)→noop"],
)
def test_impute_missing_log_shape(input_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> None:
    """impute_missing_log_shape always returns shape (n, 2)."""
    x = torch.randn(*input_shape)
    out = impute_missing_log_shape(x)
    assert out.shape == torch.Size(expected_shape), f"Expected {expected_shape}, got {out.shape}"
