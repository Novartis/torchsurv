# Migration Guide: TorchSurv v0.1.6 → v0.2.0

## Overview

Version 0.2.0 introduces breaking changes as part of a comprehensive modernization effort. The main changes are:

1. **Python 3.9+ required** (was 3.8+)
2. **Pydantic v2 for validation** (replaced procedural validation functions)
3. **New `pydantic>=2.0` dependency**
4. **Modern type hints** using Python 3.9+ syntax

## Breaking Changes

### 1. Python Version Requirement

**Before (v0.1.6):**
```python
requires-python = ">=3.8"
```

**After (v0.2.0):**
```python
requires-python = ">=3.9"
```

**Action Required:**
- Upgrade to Python 3.9 or later
- Update CI/CD pipelines
- Update environment files

### 2. Validation API

The internal validation functions have been replaced with Pydantic models. **Most users will not be affected** as validation is used internally by loss functions and metrics.

#### If You Imported Validation Functions Directly

**Before (v0.1.6):**
```python
from torchsurv.tools.validate_data import validate_survival_data, validate_model

# Procedural validation
validate_survival_data(event, time, strata)
validate_model(log_hz, event, model_type="cox")
```

**After (v0.2.0):**
```python
from torchsurv.tools.validation import SurvivalData, ModelParameters

# Pydantic model validation
SurvivalData(event=event, time=time, strata=strata)
ModelParameters(log_params=log_hz, event=event, model_type="cox")
```

#### Public API (No Changes Required)

If you only use the public API (loss functions and metrics), **no code changes are needed**:

```python
# This code works in both v0.1.6 and v0.2.0
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex

loss = neg_partial_log_likelihood(log_hz, event, time, checks=True)

cindex = ConcordanceIndex()
result = cindex(estimate, event, time)
```

### 3. Dependencies

**Before (v0.1.6):**
```toml
dependencies = ["torch", "scipy", "numpy", "torchmetrics"]
```

**After (v0.2.0):**
```toml
dependencies = ["torch", "scipy", "numpy", "torchmetrics", "pydantic>=2.0"]
```

**Action Required:**
```bash
pip install --upgrade torchsurv
# or
conda install torchsurv
```

### 4. Type Hints (Internal Changes)

Type hints have been modernized to use Python 3.9+ syntax. This is mostly internal and should not affect user code.

**Before:**
```python
from typing import Optional

def func(x: Optional[torch.Tensor] = None) -> None:
    pass
```

**After:**
```python
from __future__ import annotations

def func(x: torch.Tensor | None = None) -> None:
    pass
```

### 5. TorchScript Compatibility

**⚠️ Known Limitation:** `torch.jit.script` is no longer compatible with TorchSurv loss functions in v0.2.0+ due to Pydantic validation models. TorchScript performs static analysis of the entire function body, including imports, which conflicts with Pydantic's Python-only features.

**What Still Works:**
- ✅ `torch.compile` - Fully supported and tested
- ✅ Regular eager mode execution
- ✅ Autograd and gradients

**What Doesn't Work:**
- ❌ `torch.jit.script` - Will raise compilation errors

**Workaround:** Use `torch.compile` instead of `torch.jit.script`. `torch.compile` provides better performance than TorchScript in most cases and is the recommended approach for PyTorch 2.0+.

**Example:**
```python
import torch
from torchsurv.loss.cox import neg_partial_log_likelihood

# ✅ This works (and is recommended):
compiled_loss = torch.compile(neg_partial_log_likelihood)

# ❌ This will fail in v0.2.0+:
# scripted_loss = torch.jit.script(neg_partial_log_likelihood)
```

## New Features

### 1. Enum Types for Parameters

New enum types are available for better type safety (optional to use):

```python
from torchsurv.types import TiesMethod, Reduction

# Using enums (new way)
loss = neg_partial_log_likelihood(
    log_hz, event, time,
    ties_method=TiesMethod.EFRON,
    reduction=Reduction.MEAN
)

# Using strings still works (backward compatible)
loss = neg_partial_log_likelihood(
    log_hz, event, time,
    ties_method="efron",
    reduction="mean"
)
```

Available enums:
- `TiesMethod`: `EFRON`, `BRESLOW`
- `Reduction`: `MEAN`, `SUM`
- `ConfidenceMethod`: `NOETHER`, `BOOTSTRAP`, `BLANCHE`
- `Alternative`: `TWO_SIDED`, `LESS`, `GREATER`

### 2. Better Error Messages

Pydantic provides more detailed, structured error messages:

**Before (v0.1.6):**
```
ValueError: All samples are censored
```

**After (v0.2.0):**
```
1 validation error for SurvivalData
event
  Value error, All samples are censored [type=value_error, input_value=tensor([False, False, ...]), input_type=Tensor]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error
```

Error messages now include:
- Field-level details
- Input values that failed
- Links to documentation

## Testing Your Migration

### 1. Basic Import Test

```python
# Test that all imports work
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood_weibull
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
print("✓ All imports successful")
```

### 2. Basic Functionality Test

```python
import torch
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex

# Create sample data
n = 100
log_hz = torch.randn(n)
event = torch.randint(0, 2, (n,), dtype=torch.bool)
time = torch.rand(n) * 100

# Test loss
loss = neg_partial_log_likelihood(log_hz, event, time)
print(f"✓ Loss computed: {loss.item():.4f}")

# Test metric
cindex = ConcordanceIndex()
result = cindex(log_hz, event, time)
print(f"✓ C-index computed: {result.item():.4f}")
```

### 3. Validation Test

```python
from torchsurv.tools.validation import SurvivalData

# Test validation error handling
try:
    bad_event = torch.zeros(100, dtype=torch.bool)  # All censored
    bad_time = torch.rand(100) * 100
    SurvivalData(event=bad_event, time=bad_time)
except ValueError as e:
    print(f"✓ Validation error caught correctly")
```

## Deprecation Timeline

### Removed in v0.2.0

- `torchsurv.tools.validate_data` module (replaced by `torchsurv.tools.validation`)
  - `validate_survival_data()` → `SurvivalData` model
  - `validate_model()` → `ModelParameters` model
  - `validate_new_time()` → `NewTimeData` model
  - `_impute_missing_log_shape()` → `impute_missing_log_shape()` (moved to validation module)

## Support

If you encounter issues during migration:

1. Check this guide
2. Review the [CHANGELOG.md](docs/CHANGELOG.md)
3. Open an issue: https://github.com/Novartis/torchsurv/issues

## Rollback

If you need to rollback to v0.1.6:

```bash
pip install torchsurv==0.1.6
# or
conda install torchsurv=0.1.6
```

Note: v0.1.6 will continue to receive critical bug fixes for a limited time.
