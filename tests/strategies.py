from __future__ import annotations

import torch
from hypothesis import strategies as st


@st.composite
def survival_tensors(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Draw (event: bool Tensor, time: float Tensor) with:
    - n in [min_size, max_size]
    - at least one event=True
    - all time > 0, finite
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    # Draw event flags; ensure at least one True by forcing index 0
    event_list = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    event_list[0] = True  # guarantee at least one event
    event = torch.tensor(event_list, dtype=torch.bool)

    # Draw positive finite floats for time
    time_values = draw(
        st.lists(
            st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    time = torch.tensor(time_values, dtype=torch.float)

    return event, time


@st.composite
def cox_log_hazard(
    draw: st.DrawFn,
    n: int | None = None,
) -> torch.Tensor:
    """Draw a finite float Tensor of shape (n,)."""
    if n is None:
        n = draw(st.integers(min_value=10, max_value=100))

    values = draw(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return torch.tensor(values, dtype=torch.float)


@st.composite
def weibull_log_params(
    draw: st.DrawFn,
    n: int | None = None,
) -> torch.Tensor:
    """Draw a finite float Tensor of shape (n, 2) for Weibull parameters."""
    if n is None:
        n = draw(st.integers(min_value=10, max_value=100))

    rows = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=n,
            max_size=n,
        )
    )
    return torch.tensor(rows, dtype=torch.float)


@st.composite
def new_times(
    draw: st.DrawFn,
    time: torch.Tensor,
) -> torch.Tensor:
    """
    Draw sorted unique float Tensor within follow-up range of ``time``.
    """
    t_min = float(time.min().item())
    t_max = float(time.max().item())

    if t_min >= t_max:
        return time.sort().values[:1]

    m = draw(st.integers(min_value=1, max_value=10))
    values = draw(
        st.lists(
            st.floats(min_value=t_min, max_value=t_max - 1e-6, allow_nan=False, allow_infinity=False),
            min_size=m,
            max_size=m,
        )
    )
    t = torch.tensor(values, dtype=torch.float)
    t = torch.unique(t.sort().values)
    if len(t) == 0:
        t = torch.tensor([t_min], dtype=torch.float)
    return t
