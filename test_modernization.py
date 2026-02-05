#!/usr/bin/env python
"""Test script to verify modernization changes."""

from __future__ import annotations

import sys

import torch

# Test 1: Import new validation module
print("=" * 60)
print("TEST 1: Import new validation module")
print("=" * 60)
try:
    from torchsurv.tools.validation import (
        ModelParameters,
        NewTimeData,
        SurvivalData,
        impute_missing_log_shape,
    )

    print("✓ Successfully imported validation module")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Validate survival data
print("\n" + "=" * 60)
print("TEST 2: Validate survival data")
print("=" * 60)
try:
    n = 100
    event = torch.randint(0, 2, (n,), dtype=torch.bool)
    time = torch.rand(n) * 100
    data = SurvivalData(event=event, time=time)
    print(f"✓ Valid data accepted, event sum: {event.sum().item()}")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)

# Test 3: Catch validation error (all censored)
print("\n" + "=" * 60)
print("TEST 3: Catch validation error (all censored)")
print("=" * 60)
try:
    bad_event = torch.zeros(n, dtype=torch.bool)
    bad_time = torch.rand(n) * 100
    data = SurvivalData(event=bad_event, time=bad_time)
    print("✗ Should have raised validation error")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Validation error caught: All samples are censored")

# Test 4: Cox loss function
print("\n" + "=" * 60)
print("TEST 4: Cox loss function")
print("=" * 60)
try:
    from torchsurv.loss.cox import neg_partial_log_likelihood

    log_hz = torch.randn(n)
    loss = neg_partial_log_likelihood(log_hz, event, time, checks=True)
    print(f"✓ Cox loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Cox loss failed: {e}")
    sys.exit(1)

# Test 5: Weibull loss function
print("\n" + "=" * 60)
print("TEST 5: Weibull loss function")
print("=" * 60)
try:
    from torchsurv.loss.weibull import neg_log_likelihood_weibull

    log_params = torch.randn(n, 2)
    loss = neg_log_likelihood_weibull(log_params, event, time, checks=True)
    print(f"✓ Weibull loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Weibull loss failed: {e}")
    sys.exit(1)

# Test 6: Survival loss function
print("\n" + "=" * 60)
print("TEST 6: Survival loss function")
print("=" * 60)
try:
    from torchsurv.loss.survival import neg_log_likelihood

    n_eval_times = 50
    log_hz = torch.randn(n, n_eval_times)
    eval_time = torch.linspace(time.min().item(), time.max().item() - 1, n_eval_times)
    loss = neg_log_likelihood(log_hz, event, time, eval_time, checks=True)
    print(f"✓ Survival loss computed: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Survival loss failed: {e}")
    sys.exit(1)

# Test 7: C-index metric
print("\n" + "=" * 60)
print("TEST 7: C-index metric")
print("=" * 60)
try:
    from torchsurv.metrics.cindex import ConcordanceIndex

    estimate = torch.randn(n)
    cindex = ConcordanceIndex(checks=True)
    result = cindex(estimate, event, time)
    print(f"✓ C-index computed: {result.item():.4f}")
except Exception as e:
    print(f"✗ C-index failed: {e}")
    sys.exit(1)

# Test 8: AUC metric
print("\n" + "=" * 60)
print("TEST 8: AUC metric")
print("=" * 60)
try:
    from torchsurv.metrics.auc import Auc

    n_times = 10
    estimate = torch.rand(n, n_times)
    new_time = torch.linspace(time.min().item() + 1, time.max().item() - 1, n_times)
    auc = Auc(checks=True)
    result = auc(estimate, event, time, auc_type="cumulative", new_time=new_time)
    print(f"✓ AUC computed: {result.mean().item():.4f}")
except Exception as e:
    print(f"✗ AUC failed: {e}")
    sys.exit(1)

# Test 9: Brier score metric
print("\n" + "=" * 60)
print("TEST 9: Brier score metric")
print("=" * 60)
try:
    from torchsurv.metrics.brier_score import BrierScore

    estimate = torch.rand(n, n_times)
    bs = BrierScore(checks=True)
    result = bs(estimate, event, time, new_time)
    print(f"✓ Brier score computed: {result.mean().item():.4f}")
except Exception as e:
    print(f"✗ Brier score failed: {e}")
    sys.exit(1)

# Test 10: Kaplan-Meier estimator
print("\n" + "=" * 60)
print("TEST 10: Kaplan-Meier estimator")
print("=" * 60)
try:
    from torchsurv.stats.kaplan_meier import KaplanMeierEstimator

    km = KaplanMeierEstimator()
    km(event, time, check=True)
    print(f"✓ Kaplan-Meier computed, n_unique_times: {len(km.km_est)}")
except Exception as e:
    print(f"✗ Kaplan-Meier failed: {e}")
    sys.exit(1)

# Test 11: IPCW
print("\n" + "=" * 60)
print("TEST 11: IPCW")
print("=" * 60)
try:
    from torchsurv.stats.ipcw import get_ipcw

    weights = get_ipcw(event, time, checks=True)
    print(f"✓ IPCW computed: mean={weights.mean().item():.4f}")
except Exception as e:
    print(f"✗ IPCW failed: {e}")
    sys.exit(1)

# Test 12: Import types module
print("\n" + "=" * 60)
print("TEST 12: Import types module")
print("=" * 60)
try:
    from torchsurv.types import (
        Alternative,
        ConfidenceMethod,
        Reduction,
        TiesMethod,
    )

    print(f"✓ Types module imported")
    print(f"  TiesMethod.EFRON = {TiesMethod.EFRON.value}")
    print(f"  Reduction.MEAN = {Reduction.MEAN.value}")
except Exception as e:
    print(f"✗ Types import failed: {e}")
    sys.exit(1)

# Test 13: Test impute_missing_log_shape
print("\n" + "=" * 60)
print("TEST 13: Test impute_missing_log_shape")
print("=" * 60)
try:
    log_scale = torch.randn(10, 1)
    log_params = impute_missing_log_shape(log_scale)
    assert log_params.shape == (10, 2), f"Expected (10, 2), got {log_params.shape}"
    print(f"✓ impute_missing_log_shape works correctly")
except Exception as e:
    print(f"✗ impute_missing_log_shape failed: {e}")
    sys.exit(1)

# Test 14: Test NewTimeData validation
print("\n" + "=" * 60)
print("TEST 14: Test NewTimeData validation")
print("=" * 60)
try:
    new_time = torch.tensor([1.0, 2.0, 3.0])
    time = torch.tensor([0.5, 1.5, 2.5, 3.5])
    data = NewTimeData(new_time=new_time, time=time)
    print(f"✓ NewTimeData validation passed")
except Exception as e:
    print(f"✗ NewTimeData validation failed: {e}")
    sys.exit(1)

# Test 15: Catch NewTimeData validation error (not sorted)
print("\n" + "=" * 60)
print("TEST 15: Catch NewTimeData error (not sorted)")
print("=" * 60)
try:
    new_time = torch.tensor([3.0, 1.0, 2.0])  # Not sorted
    time = torch.tensor([0.5, 1.5, 2.5, 3.5])
    data = NewTimeData(new_time=new_time, time=time)
    print("✗ Should have raised validation error")
    sys.exit(1)
except ValueError:
    print(f"✓ Validation error caught: new_time not sorted")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
