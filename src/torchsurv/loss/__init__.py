"""This module defines various loss functions for survival analysis."""

from __future__ import annotations

from torchsurv.loss import competing_risks
from torchsurv.loss.cox import baseline_survival_function, neg_partial_log_likelihood, survival_function_cox
from torchsurv.loss.momentum import Momentum
from torchsurv.loss.survival import neg_log_likelihood, survival_function
from torchsurv.loss.weibull import log_hazard, neg_log_likelihood_weibull, survival_function_weibull

__all__ = [
    "baseline_survival_function",
    "competing_risks",
    "log_hazard",
    "Momentum",
    "neg_log_likelihood",
    "neg_log_likelihood_weibull",
    "neg_partial_log_likelihood",
    "survival_function",
    "survival_function_cox",
    "survival_function_weibull",
]
