# pylint: disable=C0103
# pylint: disable=C0301

import sys
import warnings

import torch

from torchsurv.tools.validate_data import (
    validate_eval_time,
    validate_model,
    validate_survival_data,
)

__all__ = ["neg_log_likelihood", "survival_function"]


def _cumulative_hazard_trapezoid(
    new_log_hz: torch.Tensor,
    new_time: torch.Tensor,
    eval_time: torch.Tensor,
    respective_times: bool = False,
    clamp_value: float = 1e10,
):
    r"""
    Cumulative hazard for a survival model approximated with the trapezoid method.

    Args:
        new_log_hz (torch.Tensor, float):
            Log hazard rates of shape = (new_n_samples, n_eval_time).
        new_time (torch.Tensor, float):
            Event or censoring time of shape = (new_n_samples,).
        eval_time (torch.Tensor, float):
            Times at which ``log_hz`` is evaluated of shape: (n_eval_time,)
        respective_times (bool, optional):
            If True, ``new_time`` must have the same length as ``new_log_hz``.
            The subject-specific cumulative hazard is then evaluated at each corresponding value in ``new_time``.
            Defaults to False.
        clamp_value (float, optional):
            Maximum value to which the cumulative hazard is clipped.
            This prevents numerical overflow or instability by capping extremely large values of the cumulative hazard.
            Defaults to 1e10.

    Returns:
        torch.Tensor: Cumulative hazard of the log likelihood of survival model.
    """

    # empty tensor for cumulative hazard
    if respective_times and new_time.size(0) == new_log_hz.size(0):
        cum_hazard = torch.zeros_like(new_time)
    else:
        cum_hazard = torch.zeros(
            (len(new_log_hz), len(new_time)),
            device=new_log_hz.device,
            dtype=new_log_hz.dtype,
        )

    for t in range(len(new_time)):
        # Mask eval_time <= observed time for subject i
        mask_time = eval_time <= new_time[t]
        t_eval = eval_time[mask_time]
        if respective_times:
            hz_eval = torch.exp(new_log_hz[t, mask_time])
            cum_hazard[t] = torch.trapezoid(hz_eval, t_eval)
        else:
            hz_eval = torch.exp(new_log_hz[:, mask_time])
            cum_hazard[:, t] = torch.trapezoid(hz_eval, t_eval, dim=1)

    return torch.clamp(
        cum_hazard,
        min=0,
        max=clamp_value,
    )


def neg_log_likelihood(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    eval_time: torch.Tensor,
    reduction: str = "mean",
    checks: bool = True,
) -> torch.Tensor:
    r"""
    Negative log-likelihood for a survival model.

    Args:
        log_hz (torch.Tensor, float):
            Log hazard rates of shape = (n_samples, n_eval_time).
            The entry at row i and column j corresponds to the log relative hazard for subject i at the jth ``n_eval_time``.
        event (torch.Tensor, bool):
            Event indicator (= True if event occurred) of shape = (n_samples,).
        time (torch.Tensor, float):
            Event or censoring time of shape = (n_samples,).
        eval_time (torch.Tensor, float):
            Times at which ``log_hz`` is evaluated of shape: (n_eval_time,)
        reduction (str, optional):
            Method to reduce losses. Defaults to "mean".
            Must be one of the following: "sum", "mean".
        checks (bool, optional):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        torch.Tensor: Negative of the log likelihood of survival model.

    Note:
        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(X_i\leq D_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
        (argument ``time``).

        Further, let :math:`\tau_1 < \tau_2 < \cdots < \tau_M` be the evaluation times (argument ``eval_time``), and
        :math:`\lambda_i(\tau)` be the hazard function for subject :math:`i` at time :math:`\tau` (argument ``log_hz``).

        The (continuous) log-likelihood for the survival model is given by:

        .. math::

            \text{ll} = - \sum_{i=1}^N \left( \delta_i \log \lambda_i(T_i) - \int_0^{T_i} \lambda_i(u) du \right).

        We approximate the cumulative hazard term using the trapezoidal rule evaluated at discrete
        times :math:`\{\tau_1, \tau_2, \ldots, \tau_M\}`:

        .. math::

            \int_0^{T_i} \lambda_i(u)\,du
            \;\approx\;
            \sum_{k=2}^{K_i}
                \frac{\lambda_i(\tau_{k-1}) + \lambda_i(\tau_k)}{2}
                \, (\tau_k - \tau_{k-1}),

        where :math:`K_i = \max\{\,k : \tau_k \le T_i\,\}` is the index of the largest evaluation time not
        exceeding :math:`T_i`. The integration therefore begins at :math:`\tau_1`, which should represent the start of
        observation (often :math:`\tau_1 = 0`).

        If :math:`T_i` does not coincide exactly with any of the evaluation times, the log-hazard at :math:`T_i`
        is approximated by the value corresponding to the nearest evaluation time
        not exceeding :math:`T_i`, that is:

        .. math::

            \log \lambda_i(T_i)
            \;\approx\;
            \log \lambda_i(\tau_{K_i}),
            \quad
            \text{where }
            K_i = \max\{\,k : \tau_k \le T_i\,\}.

    Examples:
        >>> _ = torch.manual_seed(43)
        >>> n, M = 4, 5
        >>> eval_time = torch.linspace(0, 100, steps=M, dtype=torch.float)
        >>> log_hz = torch.randn((n, M), dtype=torch.float)
        >>> event = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
        >>> time = torch.randint(low=1, high=100, size=(n,), dtype=torch.float)
        >>> neg_log_likelihood(log_hz, event, time, eval_time)  # default, mean of log likelihoods across patients
        tensor(54.0886)
        >>> neg_log_likelihood(
        ...     log_hz, event, time, eval_time, reduction="sum"
        ... )  # sum of log likelihoods across patients
        tensor(216.3546)
    """

    # If not event, or only one sample, return zero loss
    if any([event.sum().item() == 0, len(log_hz.size()) == 0]):
        warnings.warn(
            "No events OR single sample. Returning zero loss for the batch",
            stacklevel=2,
        )
        return torch.tensor(0.0, requires_grad=True)

    # ensure log_hz, event, time, eval_time are squeezed
    log_hz = log_hz.squeeze()
    event = event.squeeze()
    time = time.squeeze()
    eval_time = eval_time.squeeze()

    if checks:
        validate_survival_data(event, time)
        validate_model(log_hz, event, model_type="survival")
        validate_eval_time(log_hz, eval_time)

    # Cumulative hazard
    cum_hazard = _cumulative_hazard_trapezoid(log_hz, time, eval_time, respective_times=True)

    # Log hazard at exact observed time (interpolate last point)
    log_hz_at_time = torch.zeros_like(time)
    for i in range(len(event)):
        # Find nearest index
        idx_last = torch.searchsorted(eval_time, time[i], right=True) - 1
        log_hz_at_time[i] = log_hz[i, idx_last.clamp(min=0)]

    # Negative log likelihood contribution
    nll_all = -(event * log_hz_at_time) + cum_hazard

    if reduction.lower() == "mean":
        return nll_all.mean()
    elif reduction.lower() == "sum":
        return nll_all.sum()
    else:
        raise ValueError(f"Reduction {reduction} not supported, use 'mean' or 'sum'.")


def survival_function(
    new_log_hz: torch.Tensor,
    new_time: torch.Tensor,
    eval_time: torch.Tensor,
    checks: bool = True,
) -> torch.Tensor:
    r"""
    Compute the individual survival function for new subjects for the survival model.

    Args:
        new_log_hz (torch.Tensor, float):
            Log hazard rates for new subjects of shape = (n_samples_new, n_eval_time).
        new_time (torch.Tensor, float):
            Time at which to evaluate the survival probability of shape = (n_times,).
        eval_time (torch.Tensor, float):
            Times at which ``new_log_hz`` is evaluated of shape = (n_eval_time,)
        checks (bool, optional):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        torch.Tensor:
            Individual survival probabilities for each new subject at ``new_time`` of shape = (n_samples_new, n_times).

    Note:
        Let let :math:`\tau_1 < \tau_2 < \cdots < \tau_M` be the evaluation times (argument ``eval_time``), and
        :math:`\lambda^{\star}_i(\tau)` be the hazard function for new subject :math:`i` at
        time :math:`\tau` (argument ``new_log_hz``).

        The estimated survival function for new subject $i$ under the survival model is given by:

        .. math::

            \hat{S}_i(t) = \exp\left(- \int_0^t \lambda_i^{\star}(u) du\right).


        The cumulative hazard term, i.e. :math:`\int_0^t \lambda_i^{\star}(u) du`, is approximated using the trapezoidal rule evaluated at discrete
        times :math:`\{\tau_1, \tau_2, \ldots, \tau_M\}`. The integration begins at :math:`\tau_1`,
        which should represent the start of observation (often :math:`\tau_1 = 0`).

    Examples:
        >>> eval_time = torch.linspace(0, 4.5, steps=3, dtype=torch.float)
        >>> new_log_hz = torch.tensor([[0.15, 0.175, 0.2], [0.25, 0.5, 0.75]])  # 2 new subjects
        >>> new_time = torch.tensor([2.5, 4.5])
        >>> survival_function(new_log_hz, new_time, eval_time)
        tensor([[0.0708, 0.0047],
                [0.0369, 0.0005]])
    """

    # ensure new_log_hz, new_time, eval_time are squeezed
    new_log_hz = new_log_hz.squeeze()
    new_time = new_time.squeeze()
    eval_time = eval_time.squeeze()

    if checks:
        validate_eval_time(new_log_hz, eval_time)

    return torch.exp(-_cumulative_hazard_trapezoid(new_log_hz, new_time, eval_time))


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
