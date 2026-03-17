from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SurvivalInputs(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event: torch.Tensor
    time: torch.Tensor
    strata: torch.Tensor | None = None

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'event' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.bool().to(original_device)
        if torch.sum(coerced) <= 0:
            raise ValueError("All samples are censored. At least one event=True is required.")
        return coerced

    @field_validator("time", mode="before")
    @classmethod
    def coerce_time(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'time' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)) or torch.any(torch.isinf(coerced)):
            raise ValueError("Input 'time' contains NaN or Inf values, which are not allowed.")
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
        original_device = v.device
        return v.long().to(original_device)

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


class ModelInputs(BaseModel):
    """Validates and coerces model parameter inputs.

    ``model_type`` is normalised to lowercase.  ``log_params`` is cast to
    ``float`` and checked for NaN.  Shape constraints are enforced per model.

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

        Weibull models accept a 2-column ``log_params`` tensor:

        >>> log_params2d = torch.zeros(4, 2)
        >>> ModelInputs(log_params=log_params2d, event=event, model_type="weibull").log_params.shape
        torch.Size([4, 2])

        A length mismatch between ``log_params`` and ``event`` raises:

        >>> from pydantic import ValidationError
        >>> try:
        ...     ModelInputs(log_params=torch.randn(3), event=event, model_type="cox")
        ... except ValidationError as e:
        ...     print(e.errors()[0]["msg"])
        Value error, Dimension mismatch: 'log_params' has 3 samples but 'event' has 4.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_params: torch.Tensor
    event: torch.Tensor
    model_type: Literal["cox", "weibull", "survival"]

    @field_validator("log_params", mode="before")
    @classmethod
    def coerce_log_params(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'log_params' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)):
            raise ValueError("Input 'log_params' contains NaN values, which are not allowed.")
        return coerced

    @field_validator("event", mode="before")
    @classmethod
    def coerce_event(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'event' must be a torch.Tensor.")
        original_device = v.device
        return v.bool().to(original_device)

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
        elif self.model_type == "cox":
            if self.log_params.dim() > 2 or (
                self.log_params.dim() == 2 and self.log_params.shape[0] != self.log_params.shape[1]
            ):
                raise ValueError("For Cox model, 'log_hz' must have shape (n_samples,) or (n_samples, n_samples).")
        elif self.model_type == "survival":
            if self.log_params.dim() != 2:
                raise ValueError("For Survival model, 'log_hz' must have shape (n_samples, n_eval_times).")
        return self


class NewTimeInputs(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_time: torch.Tensor
    time: torch.Tensor
    within_follow_up: bool = True

    @field_validator("new_time", mode="before")
    @classmethod
    def coerce_new_time(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'new_time' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)) or torch.any(torch.isinf(coerced)):
            raise ValueError("Input 'new_time' contains NaN or Inf values, which are not allowed.")
        return coerced

    @field_validator("time", mode="before")
    @classmethod
    def coerce_time(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'time' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)) or torch.any(torch.isinf(coerced)):
            raise ValueError("Input 'time' contains NaN or Inf values, which are not allowed.")
        return coerced

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


class EvalTimeInputs(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    log_hz: torch.Tensor
    eval_times: torch.Tensor

    @field_validator("log_hz", mode="before")
    @classmethod
    def coerce_log_hz(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'log_hz' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)):
            raise ValueError("Input 'log_hz' contains NaN values, which are not allowed.")
        return coerced

    @field_validator("eval_times", mode="before")
    @classmethod
    def coerce_eval_times(cls, v: object) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError("Input 'eval_times' must be a torch.Tensor.")
        original_device = v.device
        coerced = v.float().to(original_device)
        if torch.any(torch.isnan(coerced)) or torch.any(torch.isinf(coerced)):
            raise ValueError("Input 'eval_times' contains NaN or Inf values, which are not allowed.")
        return coerced

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


@torch.jit.script
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


def validate_time_varying_log_hz(time_sorted: torch.Tensor, log_hz_sorted: torch.Tensor) -> None:
    """Validate consistency of time-varying log hazard at repeated time points.

    For each pair of adjacent identical times, the corresponding columns of
    ``log_hz_sorted`` must be equal across all subjects.

    Args:
        time_sorted: 1-D tensor of times in ascending order.
        log_hz_sorted: 2-D tensor of shape ``(n_subjects, n_times)``.

    Raises:
        ValueError: If any repeated time point has inconsistent log-hazard
            columns.

    Examples:
        >>> import torch
        >>> from torchsurv.tools.validators import validate_time_varying_log_hz
        >>> time = torch.tensor([1.0, 2.0, 2.0, 3.0])
        >>> log_hz = torch.tensor([[0.1, 0.2, 0.2, 0.3], [0.4, 0.5, 0.5, 0.6]])
        >>> validate_time_varying_log_hz(time, log_hz)  # identical columns → no error

        Inconsistent columns at a repeated time raise ``ValueError``:

        >>> log_hz_bad = torch.tensor([[0.1, 0.2, 0.9, 0.3], [0.4, 0.5, 0.5, 0.6]])
        >>> try:
        ...     validate_time_varying_log_hz(time, log_hz_bad)
        ... except ValueError as e:
        ...     print("ValueError raised")
        ValueError raised
    """
    for i in range(len(time_sorted) - 1):
        if time_sorted[i] == time_sorted[i + 1]:
            if not torch.all(log_hz_sorted[:, i] == log_hz_sorted[:, i + 1]):
                raise ValueError(
                    f"Inconsistency found for repeated time {time_sorted[i]} at columns {i} and {i + 1}. "
                    "The columns must have identical values at the same time."
                )
