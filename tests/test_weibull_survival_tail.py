import math

import pytest
import torch

from torchsurv.loss.weibull import survival_function_weibull


@pytest.fixture(autouse=True)
def distribution_validation():
    previous = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(True)
    yield
    torch.distributions.Distribution.set_default_validate_args(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("scalar", [False, True])
def test_survival_tail(dtype, shape, scalar):
    scale = 2.0
    params = torch.tensor([[math.log(scale), math.log(shape)]], dtype=dtype, requires_grad=True)
    time = torch.tensor(scale * 40 ** (1 / shape), dtype=dtype, requires_grad=True)
    if not scalar:
        time = time.unsqueeze(0)
    actual = survival_function_weibull(params, time)
    expected = math.exp(-40)
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=2e-5, atol=0)
    grad_params, grad_time = torch.autograd.grad(actual.sum(), (params, time))
    torch.testing.assert_close(
        grad_params,
        torch.tensor([[expected * 40 * shape, -expected * 40 * math.log(40)]], dtype=dtype),
        rtol=3e-5,
        atol=0,
    )
    torch.testing.assert_close(grad_time, -shape * 40 * expected / time.detach(), rtol=3e-5, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exponential_missing_shape_and_grid(dtype):
    params = torch.tensor([[0.0], [math.log(2)]], dtype=dtype)
    times = torch.tensor([0.01, 1, 10, 20, 40], dtype=dtype)
    expected = torch.tensor([[math.exp(-t / s) for t in times.tolist()] for s in [1, 2]], dtype=dtype)
    torch.testing.assert_close(survival_function_weibull(params, times), expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("time", [-1.0, float("nan")])
def test_survival_preserves_invalid_time_validation(time):
    with pytest.raises(ValueError):
        survival_function_weibull(torch.zeros((1, 2)), torch.tensor([time]))


@pytest.mark.parametrize("params", [[[float("-inf"), 0.0]], [[0.0, float("-inf")]], [[float("nan"), 0.0]]])
def test_survival_preserves_invalid_parameter_validation(params):
    with pytest.raises(ValueError):
        survival_function_weibull(torch.tensor(params), torch.tensor([1.0]))


def test_survival_gradcheck():
    params = torch.tensor([[0.2, -0.3], [-0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0.2, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(survival_function_weibull, (params, times))
    assert torch.autograd.gradgradcheck(survival_function_weibull, (params, times))


def test_survival_zero_and_infinity():
    params = torch.zeros((1, 2))
    times = torch.tensor([0.0, float("inf")])
    torch.testing.assert_close(survival_function_weibull(params, times), torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("dtype, log_shape", [(torch.float32, 100.0), (torch.float64, 1000.0)])
@pytest.mark.parametrize("validate", [True, False])
def test_survival_preserves_overflowed_shape(dtype, log_shape, validate):
    old_validate = torch.distributions.Distribution._validate_args
    try:
        torch.distributions.Distribution.set_default_validate_args(validate)
        params = torch.tensor([[0.0, log_shape], [1.0, log_shape]], dtype=dtype)
        times = torch.tensor(1.0, dtype=dtype)
        expected = 1 - torch.distributions.Weibull(params[:, 0].exp(), params[:, 1].exp()).cdf(times)
        torch.testing.assert_close(survival_function_weibull(params, times), expected)
    finally:
        torch.distributions.Distribution.set_default_validate_args(old_validate)


def test_survival_preserves_nan_shape_without_validation():
    torch.distributions.Distribution.set_default_validate_args(False)
    params = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)
    times = torch.tensor(1.0, dtype=torch.float64)
    expected = 1 - torch.distributions.Weibull(params[:, 0].exp(), params[:, 1].exp()).cdf(times)
    torch.testing.assert_close(survival_function_weibull(params, times), expected)


def test_survival_preserves_zero_scale_without_validation():
    torch.distributions.Distribution.set_default_validate_args(False)
    params = torch.tensor([[float("-inf"), 0.0]], dtype=torch.float64)
    times = torch.tensor(1.0, dtype=torch.float64)
    expected = 1 - torch.distributions.Weibull(params[:, 0].exp(), params[:, 1].exp()).cdf(times)
    torch.testing.assert_close(survival_function_weibull(params, times), expected)
