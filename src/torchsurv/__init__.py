"""This is the main module of the `TorchSurv` package.
It contains the main classes and functions to perform
survival analysis using PyTorch."""

from __future__ import annotations

from torchsurv.loss.cox import baseline_survival_function, neg_partial_log_likelihood, survival_function_cox
from torchsurv.loss.momentum import Momentum
from torchsurv.loss.survival import neg_log_likelihood, survival_function
from torchsurv.loss.weibull import log_hazard, neg_log_likelihood_weibull, survival_function_weibull
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Auc",
    "baseline_survival_function",
    "BrierScore",
    "ConcordanceIndex",
    "get_ipcw",
    "KaplanMeierEstimator",
    "log_hazard",
    "Momentum",
    "neg_log_likelihood",
    "neg_log_likelihood_weibull",
    "neg_partial_log_likelihood",
    "survival_function",
    "survival_function_cox",
    "survival_function_weibull",
]
