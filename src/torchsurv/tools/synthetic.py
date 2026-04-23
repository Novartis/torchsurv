from __future__ import annotations

import math

import torch

__all__ = ["make_synthetic_data"]


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise ValueError(f"Input '{name}' must be an integer.")
    if value <= 0:
        raise ValueError(f"Input '{name}' must be strictly positive.")


def _validate_probability(name: str, value: float, *, allow_one: bool) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Input '{name}' must be a number.")
    value = float(value)
    upper_bound = 1.0 if allow_one else 1.0 - 1e-12
    if value < 0.0 or value > upper_bound:
        comparator = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"Input '{name}' must be in {comparator}.")
    return value


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(torch.finfo(x.dtype).eps)


def _calibrate_censoring(
    event_time: torch.Tensor, raw_censor_time: torch.Tensor, censoring_rate: float
) -> torch.Tensor:
    """Scale raw censoring times to approximately match the requested censoring rate."""
    if censoring_rate == 0.0:
        return torch.full_like(event_time, torch.inf)

    lower = torch.tensor(0.0, dtype=event_time.dtype, device=event_time.device)
    upper = torch.tensor(1.0, dtype=event_time.dtype, device=event_time.device)

    def observed_censoring(scale: torch.Tensor) -> torch.Tensor:
        return (scale * raw_censor_time < event_time).float().mean()

    while observed_censoring(upper) > censoring_rate:
        upper = upper * 2.0

    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if observed_censoring(midpoint) > censoring_rate:
            lower = midpoint
        else:
            upper = midpoint

    return upper * raw_censor_time


def make_synthetic_data(
    n: int,
    m: int,
    rho: float,
    *,
    censoring_rate: float = 0.3,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a synthetic survival dataset for Cox-model benchmarking.

    The generated data follow a proportional-hazards construction:

    - features are IID Gaussian,
    - the true log-risk is a mixture of feature-derived signal and independent noise,
    - event times follow an exponential baseline hazard scaled by ``exp(log_risk)``,
    - censoring is independent and calibrated to an approximate target rate.

    Args:
        n: Number of samples.
        m: Number of features.
        rho: Signal strength in ``[0, 1]``. ``rho=1`` means the latent log-risk
            is fully determined by the features, while ``rho=0`` means the
            latent log-risk is independent of the features.
        censoring_rate: Approximate fraction of censored observations in ``[0, 1)``.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary containing:

        - ``x``: Covariate matrix with shape ``(n, m)``
        - ``event``: Event indicator with shape ``(n,)`` and dtype ``bool``
        - ``time``: Observed event/censoring time with shape ``(n,)``
        - ``log_risk``: Ground-truth latent Cox log-risk with shape ``(n,)``
        - ``beta``: Ground-truth normalized feature coefficients with shape ``(m,)``

    Examples:
        >>> batch = make_synthetic_data(n=64, m=8, rho=0.75, seed=7)
        >>> sorted(batch.keys())
        ['beta', 'event', 'log_risk', 'time', 'x']
        >>> batch["x"].shape
        torch.Size([64, 8])
    """
    _validate_positive_int("n", n)
    _validate_positive_int("m", m)
    rho = _validate_probability("rho", rho, allow_one=True)
    censoring_rate = _validate_probability("censoring_rate", censoring_rate, allow_one=False)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    x = torch.randn((n, m), generator=generator, dtype=torch.float32)

    beta = torch.randn((m,), generator=generator, dtype=torch.float32)
    beta = beta / beta.norm().clamp_min(torch.finfo(beta.dtype).eps)

    signal = _standardize(x @ beta)
    noise = _standardize(torch.randn((n,), generator=generator, dtype=torch.float32))

    log_risk = math.sqrt(rho) * signal + math.sqrt(1.0 - rho) * noise

    baseline_hazard = torch.tensor(0.1, dtype=torch.float32)
    uniforms = torch.rand((n,), generator=generator, dtype=torch.float32).clamp_min(torch.finfo(torch.float32).eps)
    event_time = -torch.log(uniforms) / (baseline_hazard * torch.exp(log_risk))

    censor_uniforms = torch.rand((n,), generator=generator, dtype=torch.float32).clamp_min(
        torch.finfo(torch.float32).eps
    )
    raw_censor_time = -torch.log(censor_uniforms)
    censor_time = _calibrate_censoring(event_time, raw_censor_time, censoring_rate)

    event = event_time <= censor_time
    time = torch.minimum(event_time, censor_time)

    if not event.any():
        first_event = torch.argmin(event_time)
        event[first_event] = True
        time[first_event] = event_time[first_event]

    return {
        "x": x,
        "event": event,
        "time": time,
        "log_risk": log_risk,
        "beta": beta,
    }


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
