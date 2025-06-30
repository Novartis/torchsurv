# pylint: disable=C0103
# pylint: disable=C0301

import sys
import warnings

import torch

from torchsurv.tools.validate_data import validate_loss

__all__ = [
    "_partial_likelihood_cox",
    "_partial_likelihood_efron",
    "_partial_likelihood_breslow",
    "neg_partial_log_likelihood",
]


def _partial_likelihood_cox(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
) -> torch.Tensor:
    """Calculate the partial log likelihood for the Cox proportional hazards model
    in the absence of ties in event time.
    """
    log_hz_flipped = log_hz_sorted.flip(0)
    log_denominator = torch.logcumsumexp(log_hz_flipped, dim=0).flip(0)
    return (log_hz_sorted - log_denominator)[event_sorted]


def _partial_likelihood_efron(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    time_sorted: torch.Tensor,
    time_unique: torch.Tensor,
) -> torch.Tensor:
    """Calculate the partial log likelihood for the Cox proportional hazards model
    using Efron's method to handle ties in event time.
    """
    J = len(time_unique)

    H = [
        torch.where((time_sorted == time_unique[j]) & (event_sorted == 1))[0]
        for j in range(J)
    ]
    R = [torch.where(time_sorted >= time_unique[j])[0] for j in range(J)]

    # Calculate the length of each element in H and store it in a tensor
    m = torch.tensor([len(h) for h in H])

    # Create a boolean tensor indicating whether each element in H has a length greater than 0
    include = torch.tensor([len(h) > 0 for h in H])

    log_nominator = torch.stack([torch.sum(log_hz_sorted[h]) for h in H])

    denominator_naive = torch.stack([torch.sum(torch.exp(log_hz_sorted[r])) for r in R])
    denominator_ties = torch.stack([torch.sum(torch.exp(log_hz_sorted[h])) for h in H])

    log_denominator_efron = torch.zeros(J, device=log_hz_sorted.device)
    for j in range(J):
        mj = int(m[j].item())
        for l in range(1, mj + 1):
            log_denominator_efron[j] += torch.log(
                denominator_naive[j] - (l - 1) / float(m[j]) * denominator_ties[j]
            )
    return (log_nominator - log_denominator_efron)[include]


def _partial_likelihood_breslow(
    log_hz_sorted: torch.Tensor,
    event_sorted: torch.Tensor,
    time_sorted: torch.Tensor,
):
    """
    Compute the partial likelihood using Breslow's method for Cox proportional hazards model.

    Args:
        log_hz_sorted (torch.Tensor): Log hazard rates sorted by time.
        event_sorted (torch.Tensor): Binary tensor indicating if the event occurred (1) or was censored (0), sorted by time.
        time_sorted (torch.Tensor): Event or censoring times sorted in ascending order.

    Returns:
        torch.Tensor: The partial likelihood for the observed events.
    """
    N = len(time_sorted)
    R = [torch.where(time_sorted >= time_sorted[i])[0] for i in range(N)]
    log_denominator = torch.stack(
        [torch.logsumexp(log_hz_sorted[R[i]], dim=0) for i in range(N)]
    )

    return (log_hz_sorted - log_denominator)[event_sorted]


def neg_partial_log_likelihood(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    ties_method: str = "efron",
    reduction: str = "mean",
    checks: bool = True,
) -> torch.Tensor:
    r"""Compute the negative of the partial log likelihood for the Cox proportional hazards model.

    Args:
        log_hz (torch.Tensor, float):
            Log relative hazard of length n_samples.
        event (torch.Tensor, bool):
            Event indicator of length n_samples (= True if event occurred).
        time (torch.Tensor):
            Time-to-event or censoring of length n_samples.
        ties_method (str):
            Method to handle ties in event time. Defaults to "efron".
            Must be one of the following: "efron", "breslow".
        reduction (str):
            Method to reduce losses. Defaults to "mean".
            Must be one of the following: "sum", "mean".
        checks (bool):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        (torch.tensor, float):
            Negative of the partial log likelihood.

    Note:
        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(X_i\leq D_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
        (argument ``time``).

        The log hazard function for the Cox proportional hazards model has the form:

        .. math::

            \log \lambda_i (t) = \log \lambda_{0}(t) + \log \theta_i

        where :math:`\log \theta_i` is the log relative hazard (argument ``log_hz``).

        **No ties in event time.**
        If the set :math:`\{T_i: \delta_i = 1\}_{i = 1, \cdots, N}` represent unique event times (i.e., no ties),
        the standard Cox partial likelihood can be used :cite:p:`Cox1972`. Let :math:`\tau_1 < \tau_2 < \cdots < \tau_N`
        be the ordered times and let  :math:`R(\tau_i) = \{ j: \tau_j \geq \tau_i\}`
        be the risk set at :math:`\tau_i`. The partial log likelihood is defined as:

        .. math::

            pll = \sum_{i: \: \delta_i = 1} \left(\log \theta_i - \log\left(\sum_{j \in R(\tau_i)} \theta_j \right) \right)

        **Ties in event time handled with Breslow's method.**
        Breslow's method :cite:p:`Breslow1975` describes the approach in which the procedure described above is used unmodified,
        even when ties are present. If two subjects A and B have the same event time, subject A will be at risk for the
        event that happened to B, and B will be at risk for the event that happened to A.
        Let :math:`\xi_1 < \xi_2 < \cdots` denote the unique ordered times (i.e., unique :math:`\tau_i`). Let :math:`H_k` be the set of
        subjects that have an event at time :math:`\xi_k` such that :math:`H_k = \{i: \tau_i = \xi_k, \delta_i = 1\}`, and let :math:`m_k`
        be the number of subjects that have an event at time :math:`\xi_k` such that :math:`m_k = |H_k|`.

        .. math::

            pll = \sum_{k} \left( {\sum_{i\in H_{k}}\log \theta_i} - m_k \: \log\left(\sum_{j \in R(\tau_k)} \theta_j \right) \right)


        **Ties in event time handled with Efron's method.**
        An alternative approach that is considered to give better results is the Efron's method :cite:p:`Efron1977`.
        As a compromise between the Cox's and Breslow's method, Efron suggested to use the average
        risk among the subjects that have an event at time :math:`\xi_k`:

        .. math::

            \bar{\theta}_{k} = {\frac {1}{m_{k}}}\sum_{i\in H_{k}}\theta_i

        Efron approximation of the partial log likelihood is defined by

        .. math::

            pll = \sum_{k} \left( {\sum_{i\in H_{k}}\log \theta_i} - \sum_{r =0}^{m_{k}-1} \log\left(\sum_{j \in R(\xi_k)}\theta_j-r\:\bar{\theta}_{j}\right)\right)


    Examples:
        >>> log_hz = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> event = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
        >>> time = torch.tensor([1., 2., 3., 4., 5.])
        >>> neg_partial_log_likelihood(log_hz, event, time) # default, mean of log likelihoods across patients
        tensor(1.0071)
        >>> neg_partial_log_likelihood(log_hz, event, time, reduction = 'sum') # sum of log likelihoods across patients
        tensor(3.0214)
        >>> time = torch.tensor([1., 2., 2., 4., 5.])  # Dealing with ties (default: Efron)
        >>> neg_partial_log_likelihood(log_hz, event, time, ties_method = "efron")
        tensor(1.0873)
        >>> neg_partial_log_likelihood(log_hz, event, time, ties_method = "breslow")  # Dealing with ties (Breslow)
        tensor(1.0873)

    References:

        .. bibliography::
            :filter: False

            Cox1972
            Breslow1975
            Efron1977

    """

    if checks:
        validate_loss(log_hz, event, time, model_type="cox")

    if any([event.sum().item() == 0, len(log_hz.size()) == 0]):
        warnings.warn("No events OR single sample. Returning zero loss for the batch")
        return torch.tensor(0.0, requires_grad=True)

    # sort data by time-to-event or censoring
    time_sorted, idx = torch.sort(time)
    log_hz_sorted = log_hz[idx]
    event_sorted = event[idx]
    time_unique = torch.unique(time_sorted)  # time-to-event or censoring without ties

    if len(time_unique) == len(time_sorted):
        # if not ties, use traditional cox partial likelihood
        pll = _partial_likelihood_cox(log_hz_sorted, event_sorted)
    else:
        # add warning about ties
        warnings.warn(
            f"Ties in event time detected; using {ties_method}'s method to handle ties."
        )
        # if ties, use either efron or breslow approximation of partial likelihood
        if ties_method == "efron":
            pll = _partial_likelihood_efron(
                log_hz_sorted,
                event_sorted,
                time_sorted,
                time_unique,
            )
        elif ties_method == "breslow":
            pll = _partial_likelihood_breslow(log_hz_sorted, event_sorted, time_sorted)
        else:
            raise ValueError(
                f'Ties method {ties_method} should be one of ["efron", "breslow"]'
            )

    # Negative partial log likelihood
    pll = torch.neg(pll)
    if reduction.lower() == "mean":
        loss = pll.nanmean()
    elif reduction.lower() == "sum":
        loss = pll.sum()
    else:
        raise (
            ValueError(
                f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."
            )
        )
    return loss


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
