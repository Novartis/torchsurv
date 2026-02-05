"""Type definitions and enums for TorchSurv.

This module provides enum types for string parameters and type aliases
for improved type safety and IDE support.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class TiesMethod(str, Enum):
    """Method for handling ties in Cox proportional hazards model.

    Attributes
    ----------
    EFRON : str
        Efron's method (default) - more accurate for ties.
    BRESLOW : str
        Breslow's method - faster but less accurate for ties.
    """

    EFRON = "efron"
    BRESLOW = "breslow"


class Reduction(str, Enum):
    """Reduction method for loss aggregation.

    Attributes
    ----------
    MEAN : str
        Average loss over samples (default).
    SUM : str
        Sum of losses over samples.
    """

    MEAN = "mean"
    SUM = "sum"


class ConfidenceMethod(str, Enum):
    """Method for confidence interval computation.

    Attributes
    ----------
    NOETHER : str
        Noether's method - asymptotic normal approximation.
    BOOTSTRAP : str
        Bootstrap resampling method.
    BLANCHE : str
        Blanche's method - specific to time-dependent metrics.
    """

    NOETHER = "noether"
    BOOTSTRAP = "bootstrap"
    BLANCHE = "blanche"


class Alternative(str, Enum):
    """Alternative hypothesis for statistical tests.

    Attributes
    ----------
    TWO_SIDED : str
        Two-sided test (default).
    LESS : str
        One-sided test: metric is less than reference.
    GREATER : str
        One-sided test: metric is greater than reference.
    """

    TWO_SIDED = "two_sided"
    LESS = "less"
    GREATER = "greater"


# Type aliases for backward compatibility with string literals
TiesMethodType = Literal["efron", "breslow"]
ReductionType = Literal["mean", "sum"]
ConfidenceMethodType = Literal["noether", "bootstrap", "blanche"]
AlternativeType = Literal["two_sided", "less", "greater"]
