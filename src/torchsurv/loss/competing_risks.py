from __future__ import annotations

import sys
import warnings

import torch

from torchsurv.loss.cox import neg_partial_log_likelihood as cox_neg_partial_log_likelihood
from torchsurv.tools.validators import CompetingRisksInputs, CompetingRisksModelInputs

__all__ = [
    "baseline_cumulative_incidence_function",
    "cumulative_incidence_function",
    "neg_partial_log_likelihood",
    "survival_function",
]


def _searchsorted(sorted_seq: torch.Tensor, values: torch.Tensor, right: bool = False) -> torch.Tensor:
    """torch.searchsorted with CPU fallback for devices that don't support it."""
    return torch.searchsorted(sorted_seq.cpu(), values.cpu(), right=right).to(sorted_seq.device)


def _validate_new_prediction_inputs(
    new_log_hz: torch.Tensor, new_time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce prediction inputs and validate simple shape constraints."""
    if not isinstance(new_log_hz, torch.Tensor):
        raise ValueError("Input 'new_log_hz' must be a torch.Tensor.")
    if not isinstance(new_time, torch.Tensor):
        raise ValueError("Input 'new_time' must be a torch.Tensor.")

    new_log_hz = new_log_hz.float().to(new_log_hz.device).squeeze()
    new_time = new_time.float().to(new_time.device).squeeze()

    if new_log_hz.dim() == 1:
        new_log_hz = new_log_hz.unsqueeze(0)
    if new_log_hz.dim() != 2:
        raise ValueError("Input 'new_log_hz' must have shape (n_samples_new, n_causes).")
    if new_log_hz.shape[1] == 0:
        raise ValueError("Input 'new_log_hz' must contain at least one cause column.")
    if new_time.dim() == 0:
        new_time = new_time.unsqueeze(0)
    if new_time.dim() != 1:
        raise ValueError("Input 'new_time' must be one-dimensional.")
    if torch.any(torch.isnan(new_time)) or torch.any(torch.isinf(new_time)):
        raise ValueError("Input 'new_time' contains NaN or Inf values, which are not allowed.")
    if torch.any(new_time < 0):
        raise ValueError("Input 'new_time' must be non-negative.")
    if torch.any(new_time[:-1] > new_time[1:]):
        raise ValueError("Input 'new_time' must be sorted from smallest to largest.")
    if len(new_time) != len(torch.unique(new_time)):
        raise ValueError("Input 'new_time' must contain unique values.")
    return new_log_hz, new_time


def _compute_baseline_curves(log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute baseline cause-specific hazards, survival, and CIF on a stratum."""
    time_sorted, idx = torch.sort(time)
    event_sorted = event[idx]
    log_hz_sorted = log_hz[idx]

    time_unique = torch.unique(time_sorted)
    n_times = len(time_unique)
    n_causes = log_hz.shape[1]

    first_idx = _searchsorted(time_sorted, time_unique)
    baseline_hazard = torch.zeros((n_times, n_causes), dtype=log_hz.dtype, device=log_hz.device)

    for cause_idx in range(n_causes):
        cause = cause_idx + 1
        exp_hz = torch.exp(log_hz_sorted[:, cause_idx])
        reverse_cumsum = exp_hz.flip(0).cumsum(0).flip(0)
        denominator = reverse_cumsum[first_idx]

        event_idx = _searchsorted(time_unique, time_sorted[event_sorted == cause])
        event_count = torch.zeros(n_times, dtype=log_hz.dtype, device=log_hz.device)
        if len(event_idx) > 0:
            event_count.scatter_add_(0, event_idx, torch.ones_like(event_idx, dtype=log_hz.dtype))

        baseline_hazard[:, cause_idx] = torch.where(
            denominator > 0,
            event_count / denominator,
            torch.zeros_like(event_count),
        )

    baseline_cumulative_hazard = torch.cumsum(baseline_hazard, dim=0)
    baseline_survival = torch.ones(n_times, dtype=log_hz.dtype, device=log_hz.device)
    baseline_cif = torch.zeros((n_times, n_causes), dtype=log_hz.dtype, device=log_hz.device)

    survival_prev = torch.tensor(1.0, dtype=log_hz.dtype, device=log_hz.device)
    cif_prev = torch.zeros(n_causes, dtype=log_hz.dtype, device=log_hz.device)
    for t_idx in range(n_times):
        increments = baseline_hazard[t_idx]
        total_increment = increments.sum()
        if total_increment > 0:
            delta = 1 - torch.exp(-total_increment)
            cif_prev = cif_prev + survival_prev * delta * increments / total_increment
            survival_prev = survival_prev * torch.exp(-total_increment)
        baseline_survival[t_idx] = survival_prev
        baseline_cif[t_idx] = cif_prev

    return {
        "time": time_unique,
        "baseline_hazard": baseline_hazard,
        "baseline_cumulative_hazard": baseline_cumulative_hazard,
        "baseline_survival": baseline_survival,
        "baseline_cif": baseline_cif,
    }


def neg_partial_log_likelihood(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    ties_method: str = "efron",
    reduction: str = "mean",
    strata: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Negative partial log-likelihood for a cause-specific Cox competing-risks model.

    Args:
        log_hz (torch.Tensor, float):
            Cause-specific log relative hazards of shape ``(n_samples, n_causes)``.
        event (torch.Tensor, int):
            Integer event indicator of shape ``(n_samples,)`` with ``0`` for censoring
            and ``1..K`` for the observed cause.
        time (torch.Tensor, float):
            Event or censoring time of shape ``(n_samples,)``.
        ties_method (str):
            Method to handle ties in event time. Defaults to ``"efron"``.
        reduction (str, optional):
            Method to reduce losses. Defaults to ``"mean"``.
            Must be one of ``"sum"`` or ``"mean"``.
        strata (torch.Tensor, int, optional):
            Integer tensor of shape ``(n_samples,)`` representing strata.

    Returns:
        torch.Tensor: Negative cause-specific partial log-likelihood.

    Note:
        The loss is defined as the sum of binary Cox partial log-likelihoods, one
        per cause. For cause :math:`k`, subjects with ``event == k`` are treated as
        events and all other subjects are treated as censored.

    Examples:
        >>> log_hz = torch.tensor([[0.1, -0.2], [0.3, 0.1], [-0.4, 0.5], [0.0, -0.1]])
        >>> event = torch.tensor([1, 2, 0, 1])
        >>> time = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> neg_partial_log_likelihood(log_hz, event, time)
        tensor(0.8381)
    """

    if strata is None:
        strata = torch.ones_like(event, dtype=torch.long)

    log_hz = log_hz.squeeze()
    event = event.squeeze()
    time = time.squeeze()
    strata = strata.squeeze()

    if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
        _surv = CompetingRisksInputs(event=event, time=time, strata=strata)
        event, time = _surv.event, _surv.time
        strata = _surv.strata
        _model = CompetingRisksModelInputs(log_hz=log_hz, event=event)
        log_hz = _model.log_hz

    if torch.count_nonzero(event).item() == 0:
        warnings.warn(
            "No observed causes in the batch. Returning zero loss for the batch",
            stacklevel=2,
        )
        return torch.tensor(0.0, requires_grad=True, device=log_hz.device, dtype=log_hz.dtype)

    n_causes = log_hz.shape[1]
    total_loss = torch.tensor(0.0, dtype=log_hz.dtype, device=log_hz.device)
    total_events = 0

    assert strata is not None  # for mypy
    for cause_idx in range(n_causes):
        cause = cause_idx + 1
        cause_event = event == cause
        n_events = int(cause_event.sum().item())
        if n_events == 0:
            continue
        total_loss = total_loss + cox_neg_partial_log_likelihood(
            log_hz[:, cause_idx],
            cause_event,
            time,
            ties_method=ties_method,
            reduction="sum",
            strata=strata,
        )
        total_events += n_events

    if reduction.lower() == "sum":
        return total_loss
    if reduction.lower() == "mean":
        return total_loss / total_events
    raise ValueError(f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum'].")


def baseline_cumulative_incidence_function(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    strata: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | dict[int, dict[str, torch.Tensor]]:
    r"""Estimate baseline cause-specific hazards, survival, and CIF curves.

    Args:
        log_hz (torch.Tensor, float):
            Cause-specific log relative hazards of shape ``(n_samples, n_causes)``.
        event (torch.Tensor, int):
            Integer event indicator of shape ``(n_samples,)`` with ``0`` for censoring
            and ``1..K`` for the observed cause.
        time (torch.Tensor, float):
            Event or censoring time of shape ``(n_samples,)``.
        strata (torch.Tensor, int, optional):
            Integer tensor of shape ``(n_samples,)`` representing strata.

    Returns:
        dict:
            Baseline competing-risks curves. For a single stratum, the dictionary
            contains ``time``, ``baseline_hazard``, ``baseline_cumulative_hazard``,
            ``baseline_survival``, and ``baseline_cif``. With multiple strata, the
            result is keyed by the integer stratum values.

    Examples:
        >>> log_hz = torch.zeros((4, 2))
        >>> event = torch.tensor([1, 2, 0, 1])
        >>> time = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> baseline = baseline_cumulative_incidence_function(log_hz, event, time)
        >>> baseline["baseline_cif"].shape
        torch.Size([4, 2])
    """

    if strata is None:
        strata = torch.ones_like(event, dtype=torch.long)

    log_hz = log_hz.squeeze()
    event = event.squeeze()
    time = time.squeeze()
    strata = strata.squeeze()

    if not (torch.jit.is_scripting() or torch.jit.is_tracing()):
        _surv = CompetingRisksInputs(event=event, time=time, strata=strata)
        event, time = _surv.event, _surv.time
        strata = _surv.strata
        _model = CompetingRisksModelInputs(log_hz=log_hz, event=event)
        log_hz = _model.log_hz

    time_sorted, idx = torch.sort(time)
    event_sorted = event[idx]
    log_hz_sorted = log_hz[idx]
    assert strata is not None  # for mypy
    strata_sorted = strata[idx]

    strata_unique = torch.unique(strata_sorted)
    if len(strata_unique) == 1:
        mask = strata_sorted == strata_unique[0]
        return _compute_baseline_curves(
            log_hz_sorted[mask],
            event_sorted[mask],
            time_sorted[mask],
        )

    strata_results: dict[int, dict[str, torch.Tensor]] = {}
    for stratum in strata_unique:
        mask = strata_sorted == stratum
        strata_results[int(stratum.item())] = _compute_baseline_curves(
            log_hz_sorted[mask],
            event_sorted[mask],
            time_sorted[mask],
        )

    return strata_results


def cumulative_incidence_function(
    baseline: dict[str, torch.Tensor] | dict[int, dict[str, torch.Tensor]],
    new_log_hz: torch.Tensor,
    new_time: torch.Tensor,
    new_strata: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute subject-specific cumulative incidence functions.

    Args:
        baseline (dict):
            Output of :func:`baseline_cumulative_incidence_function`.
        new_log_hz (torch.Tensor, float):
            Cause-specific log relative hazards for new subjects of shape
            ``(n_samples_new, n_causes)``.
        new_time (torch.Tensor, float):
            Times at which to evaluate the CIF of shape ``(n_times,)``.
        new_strata (torch.Tensor, int, optional):
            Integer tensor of shape ``(n_samples_new,)`` representing strata.

    Returns:
        torch.Tensor:
            Subject-specific CIF values of shape ``(n_samples_new, n_times, n_causes)``.

    Examples:
        >>> baseline = baseline_cumulative_incidence_function(
        ...     torch.zeros((4, 2)),
        ...     torch.tensor([1, 2, 0, 1]),
        ...     torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ... )
        >>> cumulative_incidence_function(baseline, torch.tensor([[0.0, 0.0]]), torch.tensor([1.0, 4.0])).shape
        torch.Size([1, 2, 2])
    """

    new_log_hz, new_time = _validate_new_prediction_inputs(new_log_hz, new_time)

    if new_strata is None:
        new_strata = torch.ones(len(new_log_hz), device=new_log_hz.device, dtype=torch.long)
    elif not isinstance(new_strata, torch.Tensor):
        raise ValueError("Input 'new_strata' must be a torch.Tensor or None.")
    else:
        new_strata = new_strata.long().to(new_log_hz.device).squeeze()

    if new_strata.dim() == 0:
        new_strata = new_strata.unsqueeze(0)
    if len(new_strata) != len(new_log_hz):
        raise ValueError(
            f"Dimension mismatch: 'new_log_hz' has {len(new_log_hz)} samples but 'new_strata' has {len(new_strata)}."
        )

    n_samples = len(new_log_hz)
    n_times = len(new_time)
    n_causes = new_log_hz.shape[1]
    cif = torch.empty((n_samples, n_times, n_causes), dtype=new_log_hz.dtype, device=new_log_hz.device)

    for stratum in torch.unique(new_strata):
        mask = new_strata == stratum
        new_log_hz_stratum = new_log_hz[mask]

        if isinstance(baseline, dict) and all(isinstance(v, dict) for v in baseline.values()):
            baseline_stratum = baseline[int(stratum.item())]  # type: ignore[index]
        else:
            baseline_stratum = baseline

        baseline_time = baseline_stratum["time"]
        baseline_hazard = baseline_stratum["baseline_hazard"]
        if baseline_hazard.shape[1] != n_causes:
            raise ValueError(
                f"Input 'new_log_hz' has {n_causes} causes but baseline was fitted with {baseline_hazard.shape[1]} causes."
            )

        scales = torch.exp(new_log_hz_stratum)
        cif_path = torch.zeros(
            (len(new_log_hz_stratum), len(baseline_time), n_causes),
            dtype=new_log_hz.dtype,
            device=new_log_hz.device,
        )
        survival_prev = torch.ones(len(new_log_hz_stratum), dtype=new_log_hz.dtype, device=new_log_hz.device)
        cif_prev = torch.zeros((len(new_log_hz_stratum), n_causes), dtype=new_log_hz.dtype, device=new_log_hz.device)

        for t_idx in range(len(baseline_time)):
            increments = baseline_hazard[t_idx].unsqueeze(0) * scales
            total_increment = increments.sum(dim=1)
            has_increment = total_increment > 0
            delta = 1 - torch.exp(-total_increment)
            ratio = torch.zeros_like(increments)
            if has_increment.any():
                ratio[has_increment] = increments[has_increment] / total_increment[has_increment].unsqueeze(1)
                cif_prev = cif_prev + survival_prev.unsqueeze(1) * delta.unsqueeze(1) * ratio
                survival_prev = survival_prev * torch.exp(-total_increment)
            cif_path[:, t_idx, :] = cif_prev

        time_index = _searchsorted(baseline_time, new_time, right=True) - torch.tensor(1, device=baseline_time.device)
        time_index = time_index.clamp(min=0)
        cif[mask] = cif_path[:, time_index, :]

    return cif


def survival_function(
    baseline: dict[str, torch.Tensor] | dict[int, dict[str, torch.Tensor]],
    new_log_hz: torch.Tensor,
    new_time: torch.Tensor,
    new_strata: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute event-free survival under a cause-specific Cox competing-risks model.

    Args:
        baseline (dict):
            Output of :func:`baseline_cumulative_incidence_function`.
        new_log_hz (torch.Tensor, float):
            Cause-specific log relative hazards for new subjects of shape
            ``(n_samples_new, n_causes)``.
        new_time (torch.Tensor, float):
            Times at which to evaluate the event-free survival of shape ``(n_times,)``.
        new_strata (torch.Tensor, int, optional):
            Integer tensor of shape ``(n_samples_new,)`` representing strata.

    Returns:
        torch.Tensor:
            Event-free survival probabilities of shape ``(n_samples_new, n_times)``.
    """

    cif = cumulative_incidence_function(baseline, new_log_hz, new_time, new_strata=new_strata)
    return 1 - cif.sum(dim=2)


if __name__ == "__main__":
    import doctest

    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
