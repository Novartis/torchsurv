# pylint: disable=no-member

from typing import Union

import torch
import numpy as np


def _check_inputs(
    loc: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    tmax: float,
    var: Union[torch.Tensor, float],
) -> None:
    if not isinstance(loc, torch.Tensor):
        raise TypeError(f"Expected loc to be of type torch.Tensor but got {type(loc)}")
    if not isinstance(event, torch.Tensor):
        raise TypeError(
            f"Expected event to be of type torch.Tensor but got {type(event)}"
        )
    if not isinstance(time, torch.Tensor):
        raise TypeError(
            f"Expected time to be of type torch.Tensor but got {type(time)}"
        )
    if not isinstance(tmax, int):
        raise TypeError(f"Expected tmax to be of type float but got {type(tmax)}")
    if not isinstance(var, torch.Tensor):
        raise TypeError(
            f"Expected var to be of type torch.Tensor or float but got {type(var)}"
        )
    if loc.dim() != 2:
        raise ValueError(f"Expected loc to have 2 dimensions but got {loc.dim()}")
    if event.dim() != 1:
        raise ValueError(f"Expected event to have 1 dimension but got {event.dim()}")
    if time.dim() != 1:
        raise ValueError(f"Expected time to have 1 dimension but got {time.dim()}")
    if loc.shape[0] != event.shape[0]:
        raise ValueError(
            f"Expected loc and event to have the same number of samples but got {loc.shape[0]} and {event.shape[0]}"
        )
    if loc.shape[0] != time.shape[0]:
        raise ValueError(
            f"Expected loc and time to have the same number of samples but got {loc.shape[0]} and {time.shape[0]}"
        )
    if var.dim() != 2:
        raise ValueError(f"Expected var to have 2 dimensions but got {var.dim()}")
    if var.shape[0] != loc.shape[0]:
        raise ValueError(
            f"Expected loc and var to have the same number of samples but got {loc.shape[0]} and {var.shape[0]}"
        )
    if var.shape[1] != 1:
        raise ValueError(f"Expected var to have 1 column but got {var.shape[1]}")


def _calculate_distribution(
    loc: torch.Tensor, var: torch.Tensor, tmax: float, return_probs: bool = False
) -> torch.Tensor:
    """
    Helper function to calculate the discretized gaussian distribution.

    Args:
        loc: (torch.Tensor):
            The predicted event time. The first parameter of the chosen distribution. Shape: (n_samples, 1).
        var: (torch.Tensor):
            The predicted variance. The second parameter of the chosen distribution. Shape: (n_samples, 1).
        tmax: (float):
            The maximum possible survival time. Must be greater than all observed event times.
        return_probs: (bool):
            Whether to return the probabilities instead of the log probabilities. Defaults to False.

    Returns:
        torch.Tensor:
            The log probabilities (or probabilities if ``return_probs`` is True) of the discretized gaussian distribution.
    """
    trange = torch.arange(1, tmax + 1, device=loc.device).view(1, -1)
    log_probs = -((trange - loc) ** 2) / (2 * var + 1e-8)
    # log_probs -= torch.logsumexp(log_probs, dim=1, keepdim=True)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
    if return_probs:
        return log_probs.exp()
    return log_probs


def _get_censoring_mask(cens_time: torch.Tensor, tmax: float) -> torch.Tensor:
    r"""
    Helper function to get the mask to be used for the censored observations term in the likelihood. For each censored observation, the mask is 1 for all possible event times greater than the censoring time.

    Args:
        cens_time: (torch.Tensor):
            The censored times. Shape: (n_samples, 1).
        tmax: (float):
            The maximum possible survival time. Must be greater than all observed event times.

    Returns:
        torch.Tensor:
            The mask for the censored observations. Shape: (n_samples, tmax).
    """
    trange = torch.arange(1, tmax + 1, device=cens_time.device).view(1, -1)
    mask = trange > cens_time
    return mask


def get_survival_function(prob: torch.Tensor) -> np.ndarray:
    r"""
    Calculate the survival function from the event time distribution.

    Args:
        prob: (torch.Tensor):
            The probabilities of the event time distribution. Shape: (n_samples, tmax).

    Returns:
        np.ndarray:
            The survival function. Shape: (n_samples, tmax).
    """
    prob = prob.data.cpu().numpy()
    return np.cumsum(prob[:, ::-1], axis=1)[:, ::-1]


def get_mean_prediction(prob: torch.Tensor, tmax) -> torch.Tensor:
    r"""
    Calculate the mean prediction from the event time distribution.

    Args:
        prob: (torch.Tensor):
            The probabilities of the event time distribution. Shape: (n_samples, tmax).
        tmax: (int):
            The maximum possible survival time. Must be greater than all observed event times.

    Returns:
        torch.Tensor:
            The mean prediction. Shape: (n_samples, 1).
    """
    trange = torch.arange(1, tmax + 1, device=prob.device)
    return (prob * trange).sum(dim=1).data


def neg_log_likelihood_centime(
    loc: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    tmax: int,
    var: Union[torch.Tensor, float] = 1.0,
    reduction: str = "mean",
    checks: bool = True,
) -> torch.Tensor:
    r"""
    Negative of the log likelihood of according to the centime model.
    https://arxiv.org/abs/2309.03851

    Args:
        loc: (torch.Tensor, float):
            The predicted event time. The first parameter of the chosen distribution. Shape: (n_samples, 1).
        event: (torch.Tensor, bool):
            Event indicator of length n_samples. True if the event was observed, False if it was censored.
        time: (torch.Tensor, float):
            Time-to-event of censoring of length n_samples.
        tmax: (int):
            The maximum possible survival time. Must be greater than all observed event times.
        var: (torch.Tensor or float):
            The predicted variance. The second parameter of the chosen distribution. Shape: (n_samples, 1).
            If a float is given, the variance is assumed to be constant. Defaults to 1.0 -- i.e. a constant variance of 1.
        reduction: (str):
            Method to reduce the loss. Either "mean" or "sum". Defaults to "mean".
        checks: (bool):
            Whether to perform input checks.
            Enabling checks can help catch potential issues in the input data. Defaults to True.

    Returns:
        torch.Tensor:
            The negative log likelihood of the centime model.

    Note:

        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`t_i` as the survival time and :math:`c_i` as the censoring time. Survival data consist of the event indicator :math:`\delta_i` and the survival time :math:`t_i` if :math:`\delta_i = 1` or the censoring time :math:`c_i` if :math:`\delta_i = 0`. The input covariates are denoted by :math:`x_i`.

        CenTime :cite:p:`Shahin2024` defines the likelihood function as:

        .. math::

            \mathcal{L}(\theta) = \sum_{i \in N_{\text{uncens.}}} \log p_{\theta}(t_i | x_i) + \sum_{i\ in N_{\text{cens.}}} \log \sum_{t=c_i+1}^{T_{\text{max}}} \frac{1}{t-1} p_{\theta}(t | x_i)

        where :math:`p_{\theta}(t | x_i)` is the conditional density of the survival time given the covariates and the parameters :math:`\theta`, :math:`N_{\text{uncens.}}` is the set of uncensored observations, :math:`N_{\text{cens.}}` is the set of censored observations, and :math:`T_{\text{max}}` is the maximum possible survival time (argument ``tmax``).

        The implementation minimizes the negative log likelihood.

        :math:`p_{\theta}(t | x_i)` is currently a discretized gaussian:

        .. math::

            p_{\theta}(t | x) = \frac{1}{Z} \exp\left(-\frac{(t - \mu_\theta(x))^2}{2\sigma_\theta(x)^2}\right)

        where :math:`Z` is the normalization constant, :math:`\mu_\theta(x)` is the predicted mean (argument ``loc``), and :math:`\sigma_\theta(x)^2` is the predicted variance (argument ``var``).

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 4
            >>> tmax = 100
            >>> loc = torch.randn(n, 1) * tmax
            >>> var = 1.0
            >>> event = torch.randint(0, 2, (n,)).bool()
            >>> time = torch.randint(1, tmax, (n,)).float()
            >>> neg_log_likelihood_centime(loc, event, time, tmax, var, reduction="mean")
            tensor(1480.6102)
            >>> neg_log_likelihood_centime(loc, event, time, tmax, var, reduction="sum")
            tensor(5922.4409)

        References:
            .. bibliography::
                :filter: False

                Shahin2024
    """
    if isinstance(var, float):
        var = torch.full_like(loc, var)

    if checks:
        _check_inputs(loc, event, time, tmax, var)

    time = time.type(torch.int64)

    uncens_time = time[event].view(-1, 1)
    cens_time = time[~event].view(-1, 1)

    logp = _calculate_distribution(loc, var, tmax)
    logp_cens = logp[~event]
    logp_uncens = logp[event]

    # uncensored observations
    nll_uncens = -logp_uncens.gather(1, uncens_time - 1)  # adjust for 0-based indexing

    # censored observations
    cens_mask = _get_censoring_mask(cens_time, tmax)
    factor = (
        torch.arange(tmax, device=loc.device)
        .view(1, -1)
        .repeat(cens_time.shape[0], 1)
        .log()
    )

    # ignore values in logp that are less than cens_time when computing logsumexp
    logp_cens[~cens_mask] = -float("inf")
    factor[~cens_mask] = 0
    nll_cens = -torch.logsumexp(logp_cens - factor, dim=1, keepdim=True)

    nll = torch.cat([nll_uncens, nll_cens])
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        raise ValueError(
            f"Reduction method {reduction} not recognized. Please use 'mean' or 'sum'."
        )
