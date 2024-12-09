import torch
import torch


@torch.jit.script
def validate_weibull(log_params: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
    """Private function, perform input format checks."""
    if not isinstance(log_params, torch.Tensor):
        raise TypeError("Input 'log_params' must be a tensor.")

    if not isinstance(event, torch.Tensor):
        raise TypeError("Input 'event' must be a tensor.")

    if not isinstance(time, torch.Tensor):
        raise TypeError("``Input 'time' must be a tensor.")

    if log_params.shape[0] != len(event):
        raise ValueError(
            "Length mismatch: The length of 'log_params' must match the length of 'event'."
        )

    if len(time) != len(event):
        raise ValueError(
            "Length mismatch: The length of 'time' must match the length of 'event'.`"
        )

    if torch.any(time < 0):
        raise ValueError("All elements in 'time' must be non-negative.")

    if torch.any((event != 0) & (event != 1)):
        raise ValueError(
            "Invalid values: 'event' must contain only boolean values (True/False or 1/0)"
        )


@torch.jit.script
def validate_cox(log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
    if not isinstance(log_hz, torch.Tensor):
        raise TypeError("Input 'log_hz' must be a tensor.")

    if not isinstance(event, torch.Tensor):
        raise TypeError("Input 'event' must be a tensor.")

    if not isinstance(time, torch.Tensor):
        raise TypeError("Input 'time' must be a tensor.")

    if len(log_hz) != len(event):
        raise ValueError(
            "Length mismatch: 'log_hz' and 'event' must have the same length."
        )

    if len(time) != len(event):
        raise ValueError(
            "Length mismatch: 'time' must have the same length as 'event'."
        )

    if torch.any(time < 0):
        raise ValueError("Invalid values: All elements in 'time' must be non-negative.")

    if torch.any((event != 0) & (event != 1)):
        raise ValueError(
            "Invalid values: 'event' must contain only boolean values (True/False or 1/0)"
        )


@torch.jit.script
def validate_inputs(event: torch.Tensor, time: torch.Tensor) -> None:
    """
    Validate the inputs for survival analysis functions.

    Args:
        event (torch.Tensor): Event indicator tensor.
        time (torch.Tensor): Time-to-event or censoring tensor.

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: If any ``time`` are negative.
    """
    if not isinstance(event, torch.Tensor) or not isinstance(time, torch.Tensor):
        raise TypeError("Inputs 'event' and 'time' should be tensors")

    if torch.any(time < 0):
        raise ValueError("All elements in 'time' must be non-negative")

    if torch.any((event != 0) & (event != 1)):
        raise ValueError(
            "Input 'event' must contain only boolean values (True/False or 1/0)"
        )


@torch.jit.script
def check_within_follow_up(
    new_time: torch.Tensor, time: torch.Tensor, within_follow_up: bool
) -> None:
    # Check if the within_follow_up flag is set to True
    if within_follow_up:
        # Check if any value in new_time is outside the range of time
        if new_time.max() >= time.max() or new_time.min() < time.min():
            # Get the minimum and maximum values of time
            min_time = time.min().item()
            max_time = time.max().item()
            # Raise a ValueError if new_time is not within the follow-up time range
            raise ValueError(
                "Value error: All new_time must be within follow-up time of test data: [{}; {}[".format(
                    min_time, max_time
                )
            )


@torch.jit.script
def validate_new_time(
    new_time: torch.Tensor, time: torch.Tensor, within_follow_up: bool = True
) -> None:
    """
    Validate the new_time tensor for survival analysis functions.

    Args:
        new_time (torch.Tensor, float): Time points for metric computation of size n_times.
        time (torch.Tensor, float): Event or censoring time of size n_samples.
        within_follow_up (bool, optional): Whether values of ``new_time`` must be within values in ``time``. Defaults to True.

    Raises:
        ValueError: If ``new_time`` contains duplicate values.
        ValueError: If ``new_time`` is not sorted.
        TypeError: If ``new_time`` is not a tensor.
        ValueError: If ``new_time`` is not of floating-point type.
        ValueError: If ``new_time`` is not within the range of follow-up in ``time``. Assessed only if ``within_follow_up`` is True.
    """
    if not isinstance(new_time, torch.Tensor):
        raise TypeError("Type error: Input 'new_time' should be a tensor.")

    if not torch.is_floating_point(new_time):
        raise ValueError(
            "Value error: Input 'new_time' should be of floating-point type."
        )

    new_time_sorted, _ = torch.sort(new_time)
    if not torch.equal(new_time_sorted, new_time):
        raise ValueError(
            "Value error: Input 'new_time' should be sorted from the smallest time to the largest."
        )

    if len(new_time_sorted) != len(torch.unique(new_time_sorted)):
        raise ValueError("Value error: Input 'new_time' should contain unique values.")

    check_within_follow_up(new_time, time, within_follow_up)
