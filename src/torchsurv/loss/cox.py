# pylint: disable=C0103
# pylint: disable=C0301

import sys
import warnings
from typing import Optional

import torch

from torchsurv.tools.validate_data import (
    validate_model,
    validate_survival_data,
    validate_time_varying_log_hz,
)

__all__ = [
    "_partial_likelihood_cox",
    "_partial_likelihood_efron",
    "_partial_likelihood_breslow",
    "neg_partial_log_likelihood",
    "baseline_survival_function",
    "survival_function_cox",
]


def _partial_likelihood_cox(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    is_time_varying_log_hz: bool,
) -> torch.Tensor:
    """
    Args:
        log_hz_sorted (torch.Tensor, float):
            Log hazard rates sorted by time.
        event_sorted (torch.Tensor, bool):
            Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
        is_time_varying_log_hz (bool):
            If True, log relative hazards are time-varying.
    Returns:
        torch.Tensor: partial log likelihood for the Cox proportional hazards model in the absence of ties in event time.
    """
    log_hz_flipped = log_hz_sorted.flip(0)
    log_nominator = log_hz_sorted
    log_denominator = torch.logcumsumexp(log_hz_flipped, dim=0).flip(0)

    if is_time_varying_log_hz:
        log_nominator = log_nominator.diagonal()
        log_denominator = log_denominator.diagonal()

    result: torch.Tensor = (log_nominator - log_denominator)[event_sorted]
    return result


def _partial_likelihood_efron(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    time_sorted: torch.Tensor,
    is_time_varying_log_hz: bool,
) -> torch.Tensor:
    """
    Args:
        log_hz_sorted (torch.Tensor, float):
            Log hazard rates sorted by time.
        event_sorted (torch.Tensor, bool):
            Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
        time_sorted (torch.Tensor, float):
            Event or censoring times sorted in ascending order.
        is_time_varying_log_hz (bool):
            If True, log relative hazards are time-varying.

    Returns:
        torch.Tensor: partial log likelihood for the Cox proportional hazards model using Efron's method to handle ties in event time.
    """

    # Event or censoring times sorted without ties.
    time_unique = torch.unique(time_sorted)

    # Number of unique time points
    K = len(time_unique)

    # H[k] = indices of subjects who experienced an event at time_unique[k]
    H = [
        torch.where((time_sorted == time_unique[k]) & (event_sorted))[0]
        for k in range(K)
    ]

    # R[k] = indices of subjects who are still at risk at time_unique[k]
    R = [torch.where(time_sorted >= time_unique[k])[0] for k in range(K)]

    # Length of each element in H
    m = torch.tensor([len(h) for h in H])

    # Boolean tensor indicating whether each element in H has a length greater than 0
    include = torch.tensor([len(h) > 0 for h in H])

    if is_time_varying_log_hz:
        # log nominator
        log_nominator = torch.stack([torch.sum(log_hz_sorted[h, h]) for h in H])

        # denominator
        denominator_naive = torch.stack(
            [
                torch.sum(torch.exp(log_hz_sorted[R[k]][:, H[k]][:, :1]))
                for k in range(K)
            ]
        )  # the columns are identifical for tied time points
        denominator_ties = torch.stack(
            [torch.sum(torch.exp(log_hz_sorted[h, h])) for h in H]
        )
    else:
        # log nominator
        log_nominator = torch.stack([torch.sum(log_hz_sorted[h]) for h in H])

        # denominator
        denominator_naive = torch.stack(
            [torch.sum(torch.exp(log_hz_sorted[r])) for r in R]
        )
        denominator_ties = torch.stack(
            [torch.sum(torch.exp(log_hz_sorted[h])) for h in H]
        )

    # log denominator
    log_denominator_efron = torch.zeros(K, device=log_hz_sorted.device)
    for k in range(K):
        mk = int(m[k].item())
        for r in range(1, mk + 1):
            log_denominator_efron[k] += torch.log(
                denominator_naive[k] - (r - 1) / float(m[k]) * denominator_ties[k]
            )

    # Define results
    results: torch.Tensor = (log_nominator - log_denominator_efron)[include]
    return results


def _partial_likelihood_breslow(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    time_sorted: torch.Tensor,
    is_time_varying_log_hz: bool,
) -> torch.Tensor:
    """
    Compute the partial likelihood using Breslow's method for Cox proportional hazards model.

    Args:
        log_hz_sorted (torch.Tensor, float):
            Log hazard rates sorted by time.
        event_sorted (torch.Tensor, bool):
            Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
        time_sorted (torch.Tensor, float):
            Event or censoring times sorted in ascending order.
        is_time_varying_log_hz (bool):
            If True, log relative hazards are time-varying.

    Returns:
        torch.Tensor: partial log likelihood for the Cox proportional hazards model using Breslow's method to handle ties in event time.
    """  # noqa: E501

    # Event or censoring times sorted without ties.
    time_unique = torch.unique(time_sorted)

    # Number of unique time points
    K = len(time_unique)

    # H[k] = indices of subjects who experienced an event at time_unique[k]
    H = [
        torch.where((time_sorted == time_unique[k]) & (event_sorted))[0]
        for k in range(K)
    ]

    # R[k] = indices of subjects who are still at risk"at time_unique[k]
    R = [torch.where(time_sorted >= time_unique[k])[0] for k in range(K)]

    # Length of each element in H
    m = torch.tensor([len(h) for h in H])

    # Boolean tensor indicating whether each element in H has a length greater than 0
    include = torch.tensor([len(h) > 0 for h in H])

    if is_time_varying_log_hz:
        # log nominator
        log_nominator = torch.stack([torch.sum(log_hz_sorted[h, h]) for h in H])

        # log denominator
        log_denominator = torch.stack(
            [
                torch.sum(torch.logsumexp(log_hz_sorted[R[k]][:, H[k]][:, :1], dim=0))
                for k in range(K)
            ]
        )  # the columns are identifical for tied time points; sum to handle empty tensor with 0.0
    else:
        log_nominator = torch.stack([torch.sum(log_hz_sorted[h]) for h in H])

        # log denominator
        log_denominator = torch.stack(
            [torch.logsumexp(log_hz_sorted[r], dim=0) for r in R]
        )

    # Define results
    results: torch.Tensor = (log_nominator - m * log_denominator)[include]
    return results


def _cumulative_baseline_hazard(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    time_sorted: torch.Tensor,
    is_time_varying_log_hz: bool,
) -> torch.Tensor:
    """
    Compute the log cumulative baseline hazard for the Cox proportional hazards model using the Breslow's method.

    Args:
        log_hz_sorted (torch.Tensor, float):
            Log hazard rates sorted by time.
        event_sorted (torch.Tensor, bool):
            Binary tensor indicating if the event occurred (True) or was censored (False), sorted by time.
        time_sorted (torch.Tensor, float):
            Event or censoring times sorted in ascending order.
        is_time_varying_log_hz (bool):
            If True, log relative hazards are time-varying.

    Returns:
        torch.Tensor: Log cumulative baseline hazard evaluated at each unique time point.
    """  # noqa: E501

    # Event or censoring times sorted without ties.
    time_unique = torch.unique(time_sorted)

    # Number of unique time points
    K = len(time_unique)

    # H[k] = indices of subjects who experienced an event at time_unique[k]
    H = [
        torch.where((time_sorted == time_unique[k]) & (event_sorted))[0]
        for k in range(K)
    ]

    # R[k] = indices of subjects who are still at risk"at time_unique[k]
    R = [torch.where(time_sorted >= time_unique[k])[0] for k in range(K)]

    # Length of each element in H
    m = torch.tensor([len(h) for h in H])

    # log denominator
    if is_time_varying_log_hz:
        log_denominator = torch.stack(
            [
                torch.sum(torch.logsumexp(log_hz_sorted[R[k]][:, H[k]][:, :1], dim=0))
                for k in range(K)
            ]
        )  # the columns are identifical for tied time points; sum to handle empty tensor with 0.0
    else:
        log_denominator = torch.stack(
            [torch.logsumexp(log_hz_sorted[r], dim=0) for r in R]
        )

    return torch.cumsum(m / torch.exp(log_denominator), dim=0)


def neg_partial_log_likelihood(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    ties_method: str = "efron",
    reduction: str = "mean",
    strata: Optional[torch.Tensor] = None,
    checks: bool = True,
) -> torch.Tensor:
    r"""Compute the negative of the partial log likelihood for the Cox proportional hazards model.

    Args:
        log_hz (torch.Tensor, float):
            Log relative hazard of shape = (n_samples,) or (n_samples, n_samples).
            If shape = (n_samples, n_samples), it represents time-varying log relative hazards for the extended Cox model.
            The entry at row i and column j corresponds to the log relative hazard for subject i at the ``time`` of subject j.
        event (torch.Tensor, bool):
            Event indicator (= True if event occurred) of shape = (n_samples,).
        time (torch.Tensor, float):
            Event or censoring time of shape = (n_samples,).
        ties_method (str):
            Method to handle ties in event time. Defaults to "efron".
            Must be one of the following: "efron", "breslow".
        reduction (str, optional):
            Method to reduce losses. Defaults to "mean".
            Must be one of the following: "sum", "mean".
        strata (torch.Tensor, int, optional):
            Integer tensor of shape = (n_samples,) representing stratum for each subject defined by combinations of categorical variables.
            This is useful if a categorical covariate does not obey the proportional hazard assumption.
            This is used similar to the strata expression in `R` and `lifelines`.
        checks (bool, optional):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        (torch.tensor, float):
            Negative of the Cox partial log likelihood.

    Note:
        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(X_i\leq D_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
        (argument ``time``).

        The log hazard function for the Cox proportional hazards model has the form:

        .. math::

            \log h_i (t) = \log \lambda_{0}(t) + \log \lambda_i

        where :math:`\log \lambda_i` is the log relative hazard (argument ``log_hz``).

        If the set :math:`\{T_i: \delta_i = 1\}_{i = 1, \cdots, N}` represent unique event times (i.e., no ties),
        the standard Cox partial likelihood can be used :cite:p:`Cox1972`. Let :math:`\tau_1 < \tau_2 < \cdots < \tau_N`
        be the ordered times and let  :math:`R(\tau_i) = \{ j: \tau_j \geq \tau_i\}`
        be the risk set at :math:`\tau_i`. The partial log likelihood is defined as:

        .. math::

            \text{pll} = \sum_{i: \: \delta_i = 1} \left(\log \lambda_i - \log\left(\sum_{j \in R(\tau_i)} \lambda_j \right) \right)

        **Ties in event time handled with Breslow's method.**
        Breslow's method :cite:p:`Breslow1975` describes the approach in which the procedure described above is used unmodified,
        even when ties are present. If two subjects A and B have the same event time, subject A will be at risk for the
        event that happened to B, and B will be at risk for the event that happened to A.
        Let :math:`\xi_1 < \xi_2 < \cdots` denote the unique ordered times (i.e., unique :math:`\tau_i`). Let :math:`H_k` be the set of
        subjects that have an event at time :math:`\xi_k` such that :math:`H_k = \{i: \tau_i = \xi_k, \delta_i = 1\}`, and let :math:`m_k`
        be the number of subjects that have an event at time :math:`\xi_k` such that :math:`m_k = |H_k|`.

        .. math::

            \text{pll} = \sum_{k} \left( {\sum_{i\in H_{k}}\log \lambda_i} - m_k \: \log\left(\sum_{j \in R(\xi_k)} \lambda_j \right) \right)


        **Ties in event time handled with Efron's method.**
        An alternative approach that is considered to give better results is the Efron's method :cite:p:`Efron1977`.
        As a compromise between the Cox's and Breslow's method, Efron suggested to use the average
        risk among the subjects that have an event at time :math:`\xi_k`:

        .. math::

            \bar{\lambda}_{k} = {\frac {1}{m_{k}}}\sum_{i\in H_{k}}\lambda_i

        Efron approximation of the partial log likelihood is defined by

        .. math::

            \text{pll} = \sum_{k} \left( {\sum_{i\in H_{k}}\log \lambda_i} - \sum_{r =0}^{m_{k}-1} \log\left(\sum_{j \in R(\xi_k)}\lambda_j-r\:\bar{\lambda}_{j}\right)\right)

        **Stratified Cox model.**
        When subjects come from different strata (argument ``strata``), each stratum has its own baseline hazard function :cite:p:`Therneau2000` (Chapter 3.2).
        Let :math:`\lambda_{0}^s(t)` be the baseline hazard for stratum :math:`s`.
        The hazard function for patient :math:`i` in stratum :math:`s` becomes:

        .. math::

            \log h_i^s(t) = \log \lambda_{0}^s(t) + \log \lambda_i

        The partial likelihood is computed separately within each stratum and then combined:

        .. math::

            \text{pll} = \sum_{s}  \text{pll}_{s}

        where :math:`\text{pll}_{s}` is the partial log likelihood contribution computed using only subjects in stratum :math:`s`

        **Extended Cox model.**
        When the log relative hazard :math:`\log \lambda_i(t)` is time-varying (i.e., a function of time), the extended Cox model can be used :cite:p:`OQuigley2021` (Chapter 6.5).
        The extended partial log likelihood is defined as:

        .. math::

            \text{Without ties: } \text{pll} &= \sum_{i: \: \delta_i = 1} \left(\log \lambda_i(\tau_i) - \log\left(\sum_{j \in R(\tau_i)} \lambda_j(\tau_i) \right) \right), \\
            \text{Breslow's correction: } \text{pll} &= \sum_{k} \left( {\sum_{i\in H_{k}}\log \lambda_i(\xi_k)} - m_k \: \log\left(\sum_{j \in R(\xi_k)} \lambda_j(\xi_k) \right) \right), \\
            \text{Efron's correction: } \text{pll} &= \sum_{k} \left( {\sum_{i\in H_{k}}\log \lambda_i(\xi_k)} - \sum_{r =0}^{m_{k}-1} \log\left(\sum_{j \in R(\xi_k)}\lambda_j(\xi_k)-r\:\bar{\lambda}_{j}(\xi_k)\right)\right), \: \bar{\lambda}_{k} = {\frac {1}{m_{k}}}\sum_{i\in H_{k}}\lambda_i(\xi_k).

    Examples:
        >>> _ = torch.manual_seed(43)
        >>> n = 4
        >>> log_hz = torch.randn((n, 1), dtype=torch.float)
        >>> event = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
        >>> time = torch.randint(low=1, high=100, size=(n,), dtype=torch.float)
        >>> neg_partial_log_likelihood(log_hz, event, time)  # default, mean of log likelihoods across patients
        tensor(1.9908)
        >>> neg_partial_log_likelihood(log_hz, event, time, reduction="sum")  # sum of log likelihoods across patients
        tensor(5.9724)
        >>> time[0] = time[1]  # Dealing with ties (default: Efron)
        >>> neg_partial_log_likelihood(log_hz, event, time, ties_method="efron")
        tensor(2.9877)
        >>> neg_partial_log_likelihood(log_hz, event, time, ties_method="breslow")  # Dealing with ties (Breslow)
        tensor(3.0371)

    References:

        .. bibliography::
            :filter: False

            Cox1972
            Breslow1975
            Efron1977
            Therneau2000
            OQuigley2021

    """  # noqa: E501

    # If not event, or only one sample, return zero loss
    if any([event.sum().item() == 0, len(log_hz.size()) == 0]):
        warnings.warn(
            "No events OR single sample. Returning zero loss for the batch",
            stacklevel=2,
        )
        return torch.tensor(0.0, requires_grad=True)

    # if no strata specified, every subject if in the same strata
    if strata is None:
        strata = torch.ones_like(event, dtype=torch.long)

    # ensure log_hz, event, time, strata are squeezed
    log_hz = log_hz.squeeze()
    event = event.squeeze()
    time = time.squeeze()
    strata = strata.squeeze()

    if checks:
        validate_survival_data(event, time, strata)
        validate_model(log_hz, event, model_type="cox")

    # Flag indicating whether `log_hz` is time-varying
    is_time_varying_log_hz = len(log_hz.size()) == 2

    # sort data by event or censoring time
    time_sorted, idx = torch.sort(time)
    event_sorted = event[idx]
    if is_time_varying_log_hz:
        log_hz_sorted = log_hz[idx[:, None], idx]
    else:
        log_hz_sorted = log_hz[idx]

    # if log_hz is time-varying, check the repetition of time points
    if checks and is_time_varying_log_hz:
        validate_time_varying_log_hz(time_sorted, log_hz_sorted)

    assert strata is not None  # for mypy
    strata_sorted = strata[idx]
    strata_unique = torch.unique(strata_sorted)

    pll = []

    for stratum in strata_unique:
        mask = strata_sorted == stratum
        event_stratum = event_sorted[mask]
        time_stratum = time_sorted[mask]
        log_hz_stratum = (
            log_hz_sorted[:, mask][mask, :]
            if is_time_varying_log_hz
            else log_hz_sorted[mask]
        )

        # Determine whether there are ties in event time
        has_ties = len(torch.unique(time_stratum[event_stratum])) != len(
            time_stratum[event_stratum]
        )

        if not has_ties:
            # No ties → use standard Cox
            pll.append(
                _partial_likelihood_cox(
                    log_hz_stratum, event_stratum, is_time_varying_log_hz
                )
            )
        else:
            # Warn about ties and select appropriate method
            warnings.warn(
                f"Ties in `time` detected; using {ties_method} method to handle ties.",
                stacklevel=2,
            )

            # if ties, use either efron or breslow approximation of partial likelihood
            if ties_method == "efron":
                pll.append(
                    _partial_likelihood_efron(
                        log_hz_stratum,
                        event_stratum,
                        time_stratum,
                        is_time_varying_log_hz,
                    )
                )
            elif ties_method == "breslow":
                pll.append(
                    _partial_likelihood_breslow(
                        log_hz_stratum,
                        event_stratum,
                        time_stratum,
                        is_time_varying_log_hz,
                    )
                )
            else:
                raise ValueError(
                    f'Ties method {ties_method} should be one of ["efron", "breslow"]'
                )

    # Negative partial log likelihood
    pll = torch.neg(torch.cat(pll))
    if reduction.lower() == "mean":
        mean_loss: torch.Tensor = pll.nanmean()
        return mean_loss
    elif reduction.lower() == "sum":
        sum_loss: torch.Tensor = pll.sum()
        return sum_loss
    else:
        raise (
            ValueError(
                f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."
            )
        )


def baseline_survival_function(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    strata: Optional[torch.Tensor] = None,
    checks: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    r"""Compute the baseline survival function for the Cox proportional hazards model with Breslow's method.

    Args:
        log_hz (torch.Tensor, float):
            Log relative hazard of shape = (n_samples,) or (n_samples, n_samples).
            If shape = (n_samples, n_samples), it represents time-varying log relative hazards for the extended Cox model.
            The entry at row i and column j corresponds to the log relative hazard for subject i at the ``time`` of subject j.
        event (torch.Tensor, bool):
            Event indicator (= True if event occurred) of shape = (n_samples,) used to fit the model.
        time (torch.Tensor, float):
            Event or censoring time of shape = (n_samples,) used to fit the model.
        strata (torch.Tensor, int, optional):
            Integer tensor of shape = (n_samples,) representing stratum for each subject defined by combinations of categorical variables.
        checks (bool, optional):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        (dict):
            Dictionary with two entries:
                - `"time"` (`torch.Tensor`): Sorted unique ``time``.
                - `"baseline_survival"` (`torch.Tensor`): Estimated baseline survival function evaluated at these times.

    Note:
        The baseline survival function, :math:`S_0(t)`, and the baseline cumulative hazard, :math:`H_0(t)`, under the Cox proportional hazards model are defined as:

        .. math::

            S_0(t) = \exp\Big(-H_0(u)\, du \Big), \quad H_0(t) = \int_{0}^{t} h_0(u)\, du.

        Using the Breslow's estimator :cite:p:`Breslow1972`, we estimate the baseline cumulative hazard as:

        .. math::

            \hat{H}_0(t) = \sum_{\xi_k \le t} \frac{m_k}{\sum_{j \in R(\xi_k)} \lambda_j}.

        The estimated baseline survival function is then given by:

        .. math::

            \hat{S}_0(t) = \exp\left(-\hat{H}_0(t)\right).

        **Stratified Cox model.**
        When ``strata`` are provided, the baseline cumulative hazard :math:`\hat{H}_{0}^s(t)` and baseline survival function
        :math:`\hat{S}_{0}^s(t)` are computed separately for each stratum :math:`s`, using only subjects from the same stratum.

        **Extended Cox model.**
        When the log relative hazard :math:`\log \lambda_i(t)` is time-varying (i.e., a function of time), the baseline cumulative hazard is estimated as:

        .. math::

            \hat{H}_0(t) = \sum_{\xi_k \le t} \frac{m_k}{\sum_{j \in R(\xi_k)} \lambda_j(\xi_k)}.
    Examples:
        >>> log_hz = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> event = torch.tensor([1, 0, 0, 1, 1], dtype=torch.bool)
        >>> time = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0])
        >>> baseline_survival_function(log_hz, event, time)
        {'time': tensor([1., 2., 3., 4.]), 'baseline_survival': tensor([0.8636, 0.8636, 0.8636, 0.4568])}

    References:

        .. bibliography::
            :filter: False

            Breslow1972
    """  # noqa: E501

    # if no strata specified, every subject if in the same strata
    if strata is None:
        strata = torch.ones_like(event, dtype=torch.long)

    # ensure log_hz, event, time, strata are squeezed
    log_hz = log_hz.squeeze()
    event = event.squeeze()
    time = time.squeeze()
    strata = strata.squeeze()

    if checks:
        validate_survival_data(event, time, strata)

    # Flag indicating whether `log_hz` is time-varying
    is_time_varying_log_hz = len(log_hz.size()) == 2

    # sort data by event or censoring time
    time_sorted, idx = torch.sort(time)
    event_sorted = event[idx]

    assert strata is not None  # for mypy
    strata_sorted = strata[idx]

    if is_time_varying_log_hz:
        log_hz_sorted = log_hz[idx[:, None], idx]
    else:
        log_hz_sorted = log_hz[idx]

    # if log_hz is time-varying, check the repetition of time points
    if checks and is_time_varying_log_hz:
        validate_time_varying_log_hz(time_sorted, log_hz_sorted)

    strata_unique = torch.unique(strata_sorted)

    strata_results = {}
    for str in strata_unique:
        mask = strata_sorted == str
        event_sorted_stratum = event_sorted[mask]
        time_sorted_stratum = time_sorted[mask]
        log_hz_stratum = (
            log_hz_sorted[:, mask][mask, :]
            if is_time_varying_log_hz
            else log_hz_sorted[mask]
        )

        # event or censoring time without ties
        time_unique_strata = torch.unique(time_sorted_stratum)

        # Compute baseline cumulative hazard
        cumulative_baseline_hazard_strata = _cumulative_baseline_hazard(
            log_hz_stratum,
            event_sorted_stratum,
            time_sorted_stratum,
            is_time_varying_log_hz,
        )

        # return baseline survival function
        if len(strata_unique) == 1:
            # unique strata
            strata_results = {
                "time": torch.unique(time_unique_strata),
                "baseline_survival": torch.exp(-cumulative_baseline_hazard_strata),
            }
        else:
            # multiple strata
            key = int(str.item())
            strata_results[key] = {  # type: ignore
                "time": torch.unique(time_unique_strata),
                "baseline_survival": torch.exp(-cumulative_baseline_hazard_strata),
            }

    return strata_results


def survival_function_cox(
    baseline_survival: torch.Tensor,
    new_log_hz: torch.Tensor,
    new_time: torch.Tensor,
    new_strata: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute the individual survival function for new subjects for the Cox proportional hazards model.

    Args:
        baseline_survival (dict):
            Output of ``baseline_survival_function``.
        new_log_hz (torch.Tensor, float):
            Log relative hazard for new subjects of shape = (n_samples_new,) or (n_samples_new, n_times).
            If shape = (n_samples_new, n_times), it represents time-varying log relative hazards for the extended Cox model.
            The entry at row i and column j corresponds to the log relative hazard for new subject i at the entry j of ``new_time``.
        new_time (torch.Tensor, float):
            Time at which to evaluate the survival probability of shape = (n_times,).
        new_strata (torch.Tensor, int, optional):
            Integer tensor of shape = (n_samples_new,) representing stratum for each new subject defined by combinations of categorical variables.

    Returns:
        torch.Tensor:
            Individual survival probabilities for each new subject at ``new_time`` of shape = (n_samples_new, n_times).

    Note:
        The estimated survival function for new subject $i$ under the Cox proportional hazards model is given by:

        .. math::

            \hat{S}_i(t) = \hat{S}_0(t)^{\lambda_i^{\star}},

        where :math:`\hat{S}_0(t)` is the estimated baseline survival function and
        :math:`\log \lambda_i^{\star}` is the log relative hazard of new subjects (argument ``new_log_hz``).

        **Stratified Cox model.**
        When strata are provided for both the original model fitting and new subject prediction (argument ``new_strata``),
        the survival function uses the baseline survival function specific to the subject's stratum :math:`\hat{S}_{0}^s(t)`.

        **Extended Cox model.**
        When the log relative hazard :math:`\log \lambda_i^{\star}(t)` is time-varying (i.e., a function of time), the estimated survival function is:

        .. math::

            \hat{S}_i(t) = \hat{S}_0(t)^{\lambda_i^{\star}(t)}.
    Examples:
        >>> event = torch.tensor([1, 0, 0, 1, 1], dtype=torch.bool)  # original subjects
        >>> time = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0])
        >>> log_hz = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> baseline_survival = baseline_survival_function(log_hz, event, time)
        >>> new_log_hz = torch.tensor([0.15, 0.25])  # 2 new subjects
        >>> new_time = torch.tensor([2.5, 4.5])
        >>> survival_function_cox(baseline_survival, new_log_hz, new_time)
        tensor([[0.8433, 0.4024],
                [0.8283, 0.3657]])
    """

    # if no strata specified, every new subject if in the same strata
    if new_strata is None:
        new_strata = torch.ones(
            len(new_log_hz), device=new_log_hz.device, dtype=torch.long
        )

    # ensure log_hz, new_time, new_strata are squeezed
    new_log_hz = new_log_hz.squeeze()
    new_time = new_time.squeeze()
    new_strata = new_strata.squeeze()

    # Flag indicating whether `log_hz` is time-varying
    is_time_varying_log_hz = len(new_log_hz.size()) == 2

    # unique new strata
    new_strata_unique = torch.unique(new_strata)

    # instantiate empty tensor to store individual survival
    individual_survival = torch.empty(
        (len(new_log_hz), len(new_time)),
        dtype=new_log_hz.dtype,
        device=new_log_hz.device,
    )

    # For each strata, compute individual survival function
    for str in new_strata_unique:
        mask = new_strata == str
        new_log_hz_stratum = new_log_hz[mask]

        # get baseline survival for the stratum
        if isinstance(baseline_survival, dict) and all(
            isinstance(v, dict) for v in baseline_survival.values()
        ):
            # multiple strata
            key = int(str.item())
            baseline_survival_strata = baseline_survival[key]
        else:
            # unique strata
            baseline_survival_strata = baseline_survival

        time_stratum = baseline_survival_strata["time"]
        bs_stratum = baseline_survival_strata["baseline_survival"]

        # new_time values may not exactly match any entry in time_stratum
        # Index of last time_stratum value <= new_time (floor index)
        time_index = torch.searchsorted(
            time_stratum, new_time, right=True
        ) - torch.tensor(1)

        # If new_time is smaller than the first element of time_stratum,
        # Clamp these cases to 0 so we use the earliest available time point.
        time_index = time_index.clamp(min=0)

        # baseline survival at new_time
        bs_stratum_new_time = bs_stratum[time_index]

        # survival at new_time
        if is_time_varying_log_hz:
            individual_survival[mask] = bs_stratum_new_time ** torch.exp(
                new_log_hz_stratum
            )
        else:
            individual_survival[mask] = bs_stratum_new_time.unsqueeze(0) ** torch.exp(
                new_log_hz_stratum
            ).unsqueeze(1)

    return individual_survival


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
