from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "CompetingRisksInputs",
    "CompetingRisksModelInputs",
    "EvalTimeInputs",
    "ModelInputs",
    "NewTimeInputs",
    "SurvivalInputs",
    "TimeVaryingCoxInputs",
    "impute_missing_log_shape",
]

# ---------------------------------------------------------------------------
# Private helpers — shared coercion logic used by all field validators
# ---------------------------------------------------------------------------


def _to_float_tensor(name: str, v: object, *, allow_inf: bool = False) -> torch.Tensor:
    """Coerce *v* to a float32 tensor on its original device.

    Args:
        name: Field name used in error messages.
        v: Value to coerce; must be a :class:`torch.Tensor`.
        allow_inf: When ``False`` (default) Inf values raise ``ValueError``.

    Raises:
        ValueError: If *v* is not a tensor, or contains NaN / Inf (when
            *allow_inf* is ``False``).
    """
    if not isinstance(v, torch.Tensor):
        raise ValueError(f"Input '{name}' must be a torch.Tensor.")
    coerced = v.float().to(v.device)
    if torch.any(torch.isnan(coerced)):
        raise ValueError(f"Input '{name}' contains NaN values, which are not allowed.")
    if not allow_inf and torch.any(torch.isinf(coerced)):
        raise ValueError(f"Input '{name}' contains Inf values, which are not allowed.")
    return coerced


def _to_bool_tensor(name: str, v: object) -> torch.Tensor:
    """Coerce *v* to a bool tensor on its original device.

    Raises:
        ValueError: If *v* is not a :class:`torch.Tensor`.
    """
    if not isinstance(v, torch.Tensor):
        raise ValueError(f"Input '{name}' must be a torch.Tensor.")
    return v.bool().to(v.device)


# ---------------------------------------------------------------------------
# Shared base — enables torch.Tensor fields in every model
# ---------------------------------------------------------------------------


class _TorchModel(BaseModel):
    """Base model that allows arbitrary types (required for :class:`torch.Tensor` fields)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


class SurvivalInputs(_TorchModel):
    """Validates and coerces survival analysis inputs.

    Coerces ``event`` to ``bool``, ``time`` to ``float``, and ``strata`` to
    ``long``.  When ``strata`` is omitted it defaults to a tensor of ones
    (single stratum).

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import SurvivalInputs
        >>> event = torch.tensor([True, False, True])
        >>> time = torch.tensor([1.0, 2.0, 3.0])
        >>> inp = SurvivalInputs(event=event, time=time)
        >>> inp.event
        tensor([ True, False,  True])
        >>> inp.time
        tensor([1., 2., 3.])
        >>> inp.strata  # auto-filled with ones when omitted
        tensor([1, 1, 1])

        Integer ``event`` tensors are silently coerced to ``bool``:

        >>> SurvivalInputs(event=torch.tensor([1, 0, 1]), time=time).event
        tensor([ True, False,  True])

        An explicit ``strata`` tensor is cast to ``long``:

        >>> SurvivalInputs(event=event, time=time, strata=torch.tensor([0, 0, 1])).strata
        tensor([0, 0, 1])

        Fully-censored data raises a ``ValidationError``:

        >>> from pydantic import ValidationError
        >>> try:
        ...     SurvivalInputs(event=torch.zeros(3, dtype=torch.bool), time=time)
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, All samples are censored. At least one event=True is required.

        Mismatched lengths between ``event`` and ``time`` also raise:

        >>> try:
        ...     SurvivalInputs(event=event, time=torch.tensor([1.0, 2.0]))
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Dimension mismatch: 'event' has 3 samples but 'time' has 2.
    """

    event: torch.Tensor
    time: torch.Tensor
    strata: torch.Tensor | None = None

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        coerced = _to_bool_tensor("event", v)
        if not coerced.any():
            raise ValueError("All samples are censored. At least one event=True is required.")
        return coerced

    @field_validator("time", mode="before")
    @classmethod
    def coerce_time(cls, v: object) -> torch.Tensor:
        coerced = _to_float_tensor("time", v)
        if torch.any(coerced < 0.0):
            raise ValueError("Input 'time' must be non-negative.")
        return coerced

    @field_validator("strata", mode="before")
    @classmethod
    def coerce_strata(cls, v: object) -> torch.Tensor | None:
        if v is None:
            return None
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'strata' must be a torch.Tensor or None.")
        return v.long().to(v.device)

    @model_validator(mode="after")
    def check_dimensions(self) -> SurvivalInputs:
        n = len(self.event)
        if len(self.time) != n:
            raise ValueError(f"Dimension mismatch: 'event' has {n} samples but 'time' has {len(self.time)}.")
        if self.strata is not None and len(self.strata) != n:
            raise ValueError(f"Dimension mismatch: 'event' has {n} samples but 'strata' has {len(self.strata)}.")
        if self.strata is None:
            self.strata = torch.ones_like(self.event, dtype=torch.long)
        return self


class CompetingRisksInputs(_TorchModel):
    """Validates and coerces competing-risks survival inputs.

    ``event`` is encoded as integers with ``0`` for censoring and ``1..K`` for
    observed causes. Unlike :class:`SurvivalInputs`, fully-censored data is
    allowed so batch-level helpers can decide how to handle it.
    """

    event: torch.Tensor
    time: torch.Tensor
    strata: torch.Tensor | None = None

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'event' must be a torch.Tensor.")
        if torch.is_floating_point(v):
            if not torch.allclose(v, v.round()):
                raise ValueError("Input 'event' must contain integer-coded causes.")
        coerced = v.long().to(v.device)
        if torch.any(coerced < 0):
            raise ValueError("Input 'event' must contain non-negative cause labels.")
        return coerced

    @field_validator("time", mode="before")
    @classmethod
    def coerce_time(cls, v: object) -> torch.Tensor:
        coerced = _to_float_tensor("time", v)
        if torch.any(coerced < 0.0):
            raise ValueError("Input 'time' must be non-negative.")
        return coerced

    @field_validator("strata", mode="before")
    @classmethod
    def coerce_strata(cls, v: object) -> torch.Tensor | None:
        if v is None:
            return None
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'strata' must be a torch.Tensor or None.")
        return v.long().to(v.device)

    @model_validator(mode="after")
    def check_dimensions(self) -> CompetingRisksInputs:
        n = len(self.event)
        if len(self.time) != n:
            raise ValueError(f"Dimension mismatch: 'event' has {n} samples but 'time' has {len(self.time)}.")
        if self.strata is not None and len(self.strata) != n:
            raise ValueError(f"Dimension mismatch: 'event' has {n} samples but 'strata' has {len(self.strata)}.")
        if self.strata is None:
            self.strata = torch.ones_like(self.event, dtype=torch.long)
        return self


class ModelInputs(_TorchModel):
    """Validates and coerces model parameter inputs.

    ``model_type`` is normalised to lowercase.  ``log_params`` is cast to
    ``float`` and checked for NaN.  Shape constraints are enforced per model.
    For Weibull models, ``log_params`` is always normalised to shape ``(n, 2)``
    by :func:`impute_missing_log_shape` so callers receive a fully-specified
    tensor without needing to call the function separately.

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import ModelInputs
        >>> event = torch.tensor([True, False, True, True])
        >>> log_hz = torch.tensor([0.1, -0.2, 0.3, -0.4])
        >>> inp = ModelInputs(log_params=log_hz, event=event, model_type="cox")
        >>> inp.model_type
        'cox'
        >>> inp.log_params.shape
        torch.Size([4])

        ``model_type`` is case-insensitive and whitespace-stripped:

        >>> ModelInputs(log_params=log_hz, event=event, model_type="  COX  ").model_type
        'cox'

        1-D Weibull ``log_params`` are promoted to ``(n, 2)`` automatically:

        >>> ModelInputs(log_params=log_hz, event=event, model_type="weibull").log_params.shape
        torch.Size([4, 2])

        A length mismatch between ``log_params`` and ``event`` raises:

        >>> from pydantic import ValidationError
        >>> try:
        ...     ModelInputs(log_params=torch.randn(3), event=event, model_type="cox")
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Dimension mismatch: 'log_params' has 3 samples but 'event' has 4.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    log_params: torch.Tensor
    event: torch.Tensor
    model_type: Literal["cox", "weibull", "survival"]

    @field_validator("log_params", mode="before")
    @classmethod
    def coerce_log_params(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("log_params", v, allow_inf=True)

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        return _to_bool_tensor("event", v)

    @field_validator("model_type", mode="before")
    @classmethod
    def normalize_model_type(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("Input 'model_type' must be a string.")
        return v.lower().strip()

    @model_validator(mode="after")
    def check_shape(self) -> ModelInputs:
        n = len(self.event)
        if self.log_params.shape[0] != n:
            raise ValueError(
                f"Dimension mismatch: 'log_params' has {self.log_params.shape[0]} samples but 'event' has {n}."
            )
        if self.model_type == "weibull":
            if self.log_params.dim() not in (1, 2):
                raise ValueError(
                    f"For Weibull model, 'log_params' must have 1 or 2 dimensions. Found {self.log_params.dim()}."
                )
            if self.log_params.dim() == 2 and self.log_params.size(1) not in (1, 2):
                raise ValueError(
                    f"For Weibull model, 'log_params' must have 1 or 2 columns. Found {self.log_params.size(1)}."
                )
            # Normalise to (n, 2) so callers always receive a fully-specified tensor.
            self.log_params = impute_missing_log_shape(self.log_params)
        elif self.model_type == "cox":
            if self.log_params.dim() > 2 or (
                self.log_params.dim() == 2 and self.log_params.shape[0] != self.log_params.shape[1]
            ):
                raise ValueError("For Cox model, 'log_hz' must have shape (n_samples,) or (n_samples, n_samples).")
        elif self.model_type == "survival":
            if self.log_params.dim() != 2:
                raise ValueError("For Survival model, 'log_hz' must have shape (n_samples, n_eval_times).")
        return self


class CompetingRisksModelInputs(_TorchModel):
    """Validates multi-cause log-hazard tensors against competing-risks labels."""

    log_hz: torch.Tensor
    event: torch.Tensor

    @field_validator("log_hz", mode="before")
    @classmethod
    def coerce_log_hz(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("log_hz", v, allow_inf=True)

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'event' must be a torch.Tensor.")
        return v.long().to(v.device)

    @model_validator(mode="after")
    def check_shape(self) -> CompetingRisksModelInputs:
        n = len(self.event)
        if self.log_hz.dim() != 2:
            raise ValueError("For competing risks, 'log_hz' must have shape (n_samples, n_causes).")
        if self.log_hz.shape[0] != n:
            raise ValueError(f"Dimension mismatch: 'log_hz' has {self.log_hz.shape[0]} samples but 'event' has {n}.")
        if self.log_hz.shape[1] == 0:
            raise ValueError("For competing risks, 'log_hz' must contain at least one cause column.")
        if len(self.event) > 0 and int(self.event.max().item()) > self.log_hz.shape[1]:
            raise ValueError(
                f"Input 'event' contains cause label {int(self.event.max().item())}, "
                f"but 'log_hz' only has {self.log_hz.shape[1]} cause columns."
            )
        return self


class NewTimeInputs(_TorchModel):
    """Validates new evaluation times.

    ``new_time`` must be sorted, contain unique values, and (by default) lie
    strictly within the follow-up range ``[time.min(), time.max())``.

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import NewTimeInputs
        >>> time = torch.tensor([1.0, 2.0, 3.0, 5.0])
        >>> inp = NewTimeInputs(new_time=torch.tensor([1.5, 2.5]), time=time)
        >>> inp.new_time
        tensor([1.5000, 2.5000])

        Unsorted ``new_time`` raises a ``ValidationError``:

        >>> from pydantic import ValidationError
        >>> try:
        ...     NewTimeInputs(new_time=torch.tensor([2.5, 1.5]), time=time)
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Input 'new_time' must be sorted from smallest to largest.

        Times outside the follow-up window also raise:

        >>> try:
        ...     NewTimeInputs(new_time=torch.tensor([6.0]), time=time)
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, All 'new_time' values must be within follow-up range [1.0, 5.0).

        Pass ``within_follow_up=False`` to skip the range check:

        >>> NewTimeInputs(new_time=torch.tensor([6.0]), time=time, within_follow_up=False).new_time
        tensor([6.])
    """

    new_time: torch.Tensor
    time: torch.Tensor
    within_follow_up: bool = True

    @field_validator("new_time", mode="before")
    @classmethod
    def coerce_new_time(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("new_time", v)

    @field_validator("time", mode="before")
    @classmethod
    def coerce_time(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("time", v)

    @model_validator(mode="after")
    def check_new_time(self) -> NewTimeInputs:
        new_time_sorted, _ = torch.sort(self.new_time)
        if not torch.equal(new_time_sorted, self.new_time):
            raise ValueError("Input 'new_time' must be sorted from smallest to largest.")
        if len(new_time_sorted) != len(torch.unique(new_time_sorted)):
            raise ValueError("Input 'new_time' must contain unique values.")
        if self.within_follow_up:
            if self.new_time.max() >= self.time.max() or self.new_time.min() < self.time.min():
                min_time = self.time.min().item()
                max_time = self.time.max().item()
                raise ValueError(f"All 'new_time' values must be within follow-up range [{min_time}, {max_time}).")
        return self


class EvalTimeInputs(_TorchModel):
    """Validates log_hz shape against eval_times for the Survival model.

    ``log_hz`` must be 2-D with shape ``(n_samples, n_eval_times)``.
    ``eval_times`` must be strictly increasing.

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import EvalTimeInputs
        >>> log_hz = torch.zeros(4, 3)
        >>> eval_times = torch.tensor([1.0, 2.0, 3.0])
        >>> inp = EvalTimeInputs(log_hz=log_hz, eval_times=eval_times)
        >>> inp.log_hz.shape
        torch.Size([4, 3])
        >>> inp.eval_times
        tensor([1., 2., 3.])

        A column-count mismatch raises a ``ValidationError``:

        >>> from pydantic import ValidationError
        >>> try:
        ...     EvalTimeInputs(log_hz=log_hz, eval_times=torch.tensor([1.0, 2.0]))
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Shape mismatch: 'log_hz' has 3 time columns but 'eval_times' has 2 entries.
    """

    log_hz: torch.Tensor
    eval_times: torch.Tensor

    @field_validator("log_hz", mode="before")
    @classmethod
    def coerce_log_hz(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("log_hz", v, allow_inf=True)

    @field_validator("eval_times", mode="before")
    @classmethod
    def coerce_eval_times(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("eval_times", v)

    @model_validator(mode="after")
    def check_shapes(self) -> EvalTimeInputs:
        if self.log_hz.dim() != 2:
            raise ValueError("Input 'log_hz' must be a 2D tensor for the Survival model.")
        if self.log_hz.shape[1] != len(self.eval_times):
            raise ValueError(
                f"Shape mismatch: 'log_hz' has {self.log_hz.shape[1]} time columns "
                f"but 'eval_times' has {len(self.eval_times)} entries."
            )
        if torch.any(self.eval_times[:-1] >= self.eval_times[1:]):
            raise ValueError("Input 'eval_times' must be strictly increasing with no duplicate values.")
        return self


class TimeVaryingCoxInputs(_TorchModel):
    """Validates consistency of time-varying log hazard at repeated time points.

    For each pair of adjacent identical times in ``time_sorted``, the
    corresponding columns of ``log_hz_sorted`` must be equal across all
    subjects (checked with :func:`torch.allclose` to guard against
    floating-point rounding).

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import TimeVaryingCoxInputs
        >>> time = torch.tensor([1.0, 2.0, 2.0, 3.0])
        >>> log_hz = torch.tensor([[0.1, 0.2, 0.2, 0.3], [0.4, 0.5, 0.5, 0.6]])
        >>> TimeVaryingCoxInputs(time_sorted=time, log_hz_sorted=log_hz)  # no error
        TimeVaryingCoxInputs(time_sorted=tensor([1., 2., 2., 3.]), log_hz_sorted=tensor([[0.1000, 0.2000, 0.2000, 0.3000],
                [0.4000, 0.5000, 0.5000, 0.6000]]))

        Inconsistent columns at a repeated time raise a ``ValidationError``:

        >>> from pydantic import ValidationError
        >>> log_hz_bad = torch.tensor([[0.1, 0.2, 0.9, 0.3], [0.4, 0.5, 0.5, 0.6]])
        >>> try:
        ...     TimeVaryingCoxInputs(time_sorted=time, log_hz_sorted=log_hz_bad)
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Inconsistency found for repeated time 2 at columns 1 and 2. The columns must have identical values at the same time.
    """

    time_sorted: torch.Tensor
    log_hz_sorted: torch.Tensor

    @field_validator("time_sorted", mode="before")
    @classmethod
    def coerce_time_sorted(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("time_sorted", v)

    @field_validator("log_hz_sorted", mode="before")
    @classmethod
    def coerce_log_hz_sorted(cls, v: object) -> torch.Tensor:
        return _to_float_tensor("log_hz_sorted", v, allow_inf=True)

    @model_validator(mode="after")
    def check_consistency(self) -> TimeVaryingCoxInputs:
        time = self.time_sorted
        log_hz = self.log_hz_sorted
        if log_hz.dim() != 2:
            raise ValueError("Input 'log_hz_sorted' must be a 2D tensor.")
        if log_hz.shape[1] != len(time):
            raise ValueError(
                f"Shape mismatch: 'log_hz_sorted' has {log_hz.shape[1]} columns "
                f"but 'time_sorted' has {len(time)} entries."
            )
        # Vectorised: find all tie positions in one shot, then check only those columns.
        tie_indices = (time[:-1] == time[1:]).nonzero(as_tuple=False).squeeze(1)
        for i in tie_indices.tolist():
            if not torch.allclose(log_hz[:, i], log_hz[:, i + 1]):
                raise ValueError(
                    f"Inconsistency found for repeated time {time[i].item():.6g} "
                    f"at columns {i} and {i + 1}. "
                    "The columns must have identical values at the same time."
                )
        return self


@torch.jit.script  # type: ignore[untyped-decorator]
def impute_missing_log_shape(log_params: torch.Tensor) -> torch.Tensor:
    """
    Pure tensor function (torch.jit.script-compatible).
    Promotes log_params from shape (n,) or (n,1) to (n,2) by appending zeros.

    Args:
        log_params: Tensor of shape (n,), (n, 1), or (n, 2).

    Returns:
        Tensor of shape (n, 2).

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import impute_missing_log_shape
        >>> impute_missing_log_shape(torch.tensor([1.0, 2.0, 3.0]))
        tensor([[1., 0.],
                [2., 0.],
                [3., 0.]])
        >>> impute_missing_log_shape(torch.tensor([[1.0], [2.0], [3.0]]))
        tensor([[1., 0.],
                [2., 0.],
                [3., 0.]])
        >>> impute_missing_log_shape(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        tensor([[1., 2.],
                [3., 4.],
                [5., 6.]])
    """
    if log_params.dim() == 1:
        log_params = log_params.unsqueeze(1)
    if log_params.dim() == 2 and log_params.size(1) == 1:
        log_params = torch.hstack((log_params, torch.zeros_like(log_params)))
    return log_params
