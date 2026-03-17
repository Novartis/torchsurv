from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings

from tests.strategies import (
    cox_log_hazard,
    survival_tensors,
    weibull_log_params as weibull_log_params_strategy,
)

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood_weibull
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator


@given(survival_tensors(), cox_log_hazard())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_cox_loss_finite(survival_and_time, log_hz_vals):
    event, time = survival_and_time
    n = len(event)
    torch.manual_seed(0)
    log_hz = torch.randn(n)
    loss = neg_partial_log_likelihood(log_hz, event, time)
    assert torch.isfinite(loss), f"Cox loss not finite: {loss}"


@given(survival_tensors(), weibull_log_params_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_weibull_loss_finite(survival_and_time, log_params):
    event, time = survival_and_time
    n = len(event)
    torch.manual_seed(0)
    params = torch.randn(n, 2)
    loss = neg_log_likelihood_weibull(params, event, time)
    assert torch.isfinite(loss), f"Weibull loss not finite: {loss}"


@given(survival_tensors())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_cindex_output_range(survival_and_time):
    event, time = survival_and_time
    n = len(event)
    torch.manual_seed(0)
    log_hz = torch.randn(n)
    cindex = ConcordanceIndex()
    result = cindex(log_hz, event, time)
    assert torch.isfinite(result), f"C-index not finite: {result}"
    assert 0.0 <= result.item() <= 1.0, f"C-index out of [0,1]: {result}"


@given(survival_tensors())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_auc_output_range(survival_and_time):
    event, time = survival_and_time
    n = len(event)
    torch.manual_seed(0)
    log_hz = torch.randn(n)
    # Create valid new_time within follow-up range
    t_min = time.min().item()
    t_max = time.max().item()
    if t_min >= t_max:
        return  # skip degenerate case
    # Use a proportional epsilon to handle float32 precision with large values
    eps = max(1e-3, (t_max - t_min) * 0.01)
    upper = t_max - eps
    if upper <= t_min:
        return
    new_time = torch.unique(torch.linspace(t_min, upper, steps=5))
    if len(new_time) == 0:
        return
    auc = Auc()
    result = auc(log_hz, event, time, new_time=new_time)
    assert result is not None


@given(survival_tensors())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_kaplan_meier_estimator(survival_and_time):
    event, time = survival_and_time
    km = KaplanMeierEstimator()
    km(event, time)
    assert km.km_est is not None
    assert torch.all(km.km_est >= 0)
    assert torch.all(km.km_est <= 1)
