"""Pydantic-based validation for survival analysis data.

This module provides structured validation using Pydantic v2 models,
replacing the procedural validation functions in validate_data.py.
"""

from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator


class SurvivalData(BaseModel):
    """Validated survival analysis data.

    Validates event indicators, time-to-event data, and optional strata.
    Ensures proper types, dimensions, and value ranges.

    Attributes
    ----------
    event : torch.Tensor
        Boolean tensor indicating if event occurred (True) or censored (False).
    time : torch.Tensor
        Float tensor of time-to-event or censoring times (non-negative).
    strata : torch.Tensor, optional
        Integer tensor for stratification. Must match length of event/time.

    Raises
    ------
    TypeError
        If inputs are not tensors or have wrong types.
    ValueError
        If values are invalid (negative times, all censored, dimension mismatch).

    Examples
    --------
    >>> import torch
    >>> event = torch.tensor([True, False, True])
    >>> time = torch.tensor([1.0, 2.0, 3.0])
    >>> data = SurvivalData(event=event, time=time)
    >>> data.event
    tensor([ True, False,  True])
    """

    model_config = {"arbitrary_types_allowed": True}

    event: torch.Tensor
    time: torch.Tensor
    strata: torch.Tensor | None = None

    @field_validator("event", mode="before")
    @classmethod
    def validate_event(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate event tensor.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'event' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if v.ndim != 1:
            raise ValueError("Input 'event' should be 1-dimensional")

        if v.dtype != torch.bool:
            raise ValueError("Input 'event' should be of boolean type")

        if torch.sum(v) <= 0:
            raise ValueError("All samples are censored")

        return v

    @field_validator("time", mode="before")
    @classmethod
    def validate_time(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate time tensor.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'time' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if v.ndim != 1:
            raise ValueError("Input 'time' should be 1-dimensional")

        if not torch.is_floating_point(v):
            raise ValueError("Input 'time' should be of float type")

        if torch.any(v < 0.0):
            raise ValueError("Input 'time' should be non-negative")

        return v

    @field_validator("strata", mode="before")
    @classmethod
    def validate_strata(cls, v: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        """Validate strata tensor if provided.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        if v is None:
            return v

        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'strata' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if v.ndim != 1:
            raise ValueError("Input 'strata' should be 1-dimensional")

        if v.dtype not in (torch.int32, torch.int64):
            raise ValueError("Input 'strata' should be of integer type")

        return v

    @model_validator(mode="after")
    def validate_dimensions(self) -> SurvivalData:
        """Validate dimension consistency across tensors."""
        if len(self.event) != len(self.time):
            raise ValueError(
                f"Dimension mismatch: 'event' has length {len(self.event)}, but 'time' has length {len(self.time)}"
            )
        if self.strata is not None and len(self.event) != len(self.strata):
            raise ValueError(
                f"Dimension mismatch: 'event' has length {len(self.event)}, but 'strata' has length {len(self.strata)}"
            )
        return self


class ModelParameters(BaseModel):
    """Validated model parameters for survival models.

    Validates model outputs (log hazards, log parameters) against
    expected shapes for different model types (Cox, Weibull, Survival).

    Attributes
    ----------
    log_params : torch.Tensor
        Model output tensor (log hazards, log scale/shape, etc.).
    event : torch.Tensor
        Boolean event indicators (used for dimension checking).
    model_type : str
        Type of model: "cox", "weibull", or "survival".

    Raises
    ------
    TypeError
        If inputs are not tensors.
    ValueError
        If model_type is invalid or shapes are incompatible.

    Examples
    --------
    >>> import torch
    >>> log_hz = torch.randn(100, 1)
    >>> event = torch.randint(0, 2, (100,), dtype=torch.bool)
    >>> params = ModelParameters(log_params=log_hz, event=event, model_type="cox")
    >>> params.model_type
    'cox'
    """

    model_config = {"arbitrary_types_allowed": True}

    log_params: torch.Tensor
    event: torch.Tensor
    model_type: str

    @field_validator("log_params", mode="before")
    @classmethod
    def validate_log_params(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate log parameters tensor.

        Accepts numpy arrays and converts to tensors.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'log_params' should be a tensor")

        if not torch.is_floating_point(v):
            raise ValueError("Input 'log_params' should be of float type")

        return v

    @field_validator("event", mode="before")
    @classmethod
    def validate_event(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate event tensor.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'event' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if v.dtype != torch.bool:
            raise ValueError("Input 'event' should be of boolean type")

        return v

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate and normalize model type."""
        v_lower = v.lower()
        if v_lower not in ["cox", "weibull", "survival"]:
            raise ValueError(f"Invalid model type '{v}'. Must be 'cox', 'weibull', or 'survival'")
        return v_lower

    @model_validator(mode="after")
    def validate_shape(self) -> ModelParameters:
        """Validate tensor shapes based on model type."""
        n_samples = len(self.event)

        if self.log_params.shape[0] != n_samples:
            raise ValueError(
                f"Dimension mismatch: 'log_params' has {self.log_params.shape[0]} samples, "
                f"but 'event' has {n_samples} samples"
            )

        if self.model_type == "weibull":
            if self.log_params.dim() not in [1, 2]:
                raise ValueError(f"For Weibull model, 'log_params' must be 1D or 2D, got {self.log_params.dim()}D")
            if self.log_params.dim() == 2 and self.log_params.shape[1] not in [1, 2]:
                raise ValueError(
                    f"For Weibull model, 'log_params' must have shape "
                    f"(n_samples, 1) or (n_samples, 2), got {self.log_params.shape}"
                )
        elif self.model_type == "cox":
            if self.log_params.dim() == 1:
                pass  # (n_samples,) is valid
            elif self.log_params.dim() == 2:
                if self.log_params.shape[1] == 1:
                    pass  # (n_samples, 1) is valid
                elif self.log_params.shape != (n_samples, n_samples):
                    raise ValueError(
                        f"For Cox model with time-varying covariates, 'log_hz' must have shape "
                        f"(n_samples, n_samples), got {self.log_params.shape}"
                    )
            else:
                raise ValueError(f"For Cox model, 'log_hz' must be 1D or 2D, got {self.log_params.dim()}D")
        elif self.model_type == "survival":
            if self.log_params.dim() != 2:
                raise ValueError(
                    f"For Survival model, 'log_hz' must be 2D (n_samples, n_eval_times), got {self.log_params.dim()}D"
                )

        return self


class NewTimeData(BaseModel):
    """Validated new time data for evaluation metrics.

    Validates evaluation time points for metrics like AUC and Brier score.
    Ensures times are sorted, unique, and optionally within follow-up range.

    Attributes
    ----------
    new_time : torch.Tensor
        Sorted, unique evaluation time points.
    time : torch.Tensor
        Original time-to-event data (for range checking).
    within_follow_up : bool, default=True
        Whether to enforce new_time is within [min(time), max(time)).

    Raises
    ------
    TypeError
        If new_time is not a tensor.
    ValueError
        If new_time is not sorted, has duplicates, or is outside follow-up range.

    Examples
    --------
    >>> import torch
    >>> new_time = torch.tensor([1.0, 2.0, 3.0])
    >>> time = torch.tensor([0.5, 1.5, 2.5, 3.5])
    >>> data = NewTimeData(new_time=new_time, time=time)
    >>> data.new_time
    tensor([1., 2., 3.])
    """

    model_config = {"arbitrary_types_allowed": True}

    new_time: torch.Tensor
    time: torch.Tensor
    within_follow_up: bool = True

    @field_validator("new_time", mode="before")
    @classmethod
    def validate_new_time(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate new_time tensor.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Type error: Input 'new_time' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if not torch.is_floating_point(v):
            raise ValueError("Value error: Input 'new_time' should be of floating-point type")

        # Check sorted
        new_time_sorted, _ = torch.sort(v)
        if not torch.equal(new_time_sorted, v):
            raise ValueError("Value error: Input 'new_time' should be sorted")

        # Check unique
        if len(v) != len(torch.unique(v)):
            raise ValueError("Value error: Input 'new_time' should contain unique values")

        return v

    @field_validator("time", mode="before")
    @classmethod
    def validate_time(cls, v: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Validate time tensor.

        Accepts numpy arrays and converts to tensors.
        Accepts 2D tensors with shape [n, 1] and squeezes to 1D.
        """
        # Convert numpy array to tensor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not isinstance(v, torch.Tensor):
            raise TypeError("Input 'time' should be a tensor")

        # Squeeze 2D tensors with shape [n, 1] to 1D
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.squeeze(1)

        if not torch.is_floating_point(v):
            raise ValueError("Input 'time' should be of floating-point type")

        return v

    @model_validator(mode="after")
    def check_within_follow_up(self) -> NewTimeData:
        """Check if new_time is within follow-up range."""
        if self.within_follow_up:
            min_time = self.time.min().item()
            max_time = self.time.max().item()

            if self.new_time.max() >= max_time or self.new_time.min() < min_time:
                raise ValueError(f"All new_time must be within follow-up time: [{min_time}, {max_time})")
        return self


def impute_missing_log_shape(log_params: torch.Tensor) -> torch.Tensor:
    """Impute missing log_shape parameter for Weibull distribution.

    For exponential distribution (special case of Weibull with shape=1),
    add a zero log_shape column to make it compatible with Weibull loss.

    Parameters
    ----------
    log_params : torch.Tensor
        Log scale parameter(s), shape (n_samples,) or (n_samples, 1).

    Returns
    -------
    torch.Tensor
        Log parameters with shape (n_samples, 2), where second column is log_shape.
        If input already has 2 columns, returns unchanged.

    Examples
    --------
    >>> import torch
    >>> log_scale = torch.randn(10, 1)
    >>> log_params = impute_missing_log_shape(log_scale)
    >>> log_params.shape
    torch.Size([10, 2])
    """
    if log_params.dim() in (0, 1) or (log_params.dim() > 1 and log_params.size(1) == 1):
        # Reshape to (n_samples, 1) if needed
        if log_params.dim() == 1:
            log_params = log_params.unsqueeze(1)

        # Add zero log_shape column (shape=1 for exponential)
        log_params = torch.hstack((log_params, torch.zeros_like(log_params)))

    return log_params
