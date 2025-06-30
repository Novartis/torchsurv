import torch


def validate_log_shape(log_params: torch.Tensor) -> torch.Tensor:
    """Private function, check if the log shape is missing and impute it with 0
    if needed.
    Used only for Weibull loss, as it can handle either log_scale alone or both
    log_scale and log_shape."""
    if any(
        [
            log_params.dim() == 0,
            log_params.dim() == 1,  # if shape = [n_samples]
            log_params.dim() > 1
            and log_params.size(1) == 1,  # if shape = [n_samples, 1]
        ]
    ):
        if log_params.dim() == 1:
            log_params = log_params.unsqueeze(1)

        # Missing log shape parameter. Creating zeros placeholder instead.
        log_params = torch.hstack((log_params, torch.zeros_like(log_params)))

    return log_params


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
                f"Value error: All new_time must be within follow-up time of test data: [{min_time}; {max_time}"
            )


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


def validate_survival_data(event: torch.Tensor, time: torch.Tensor):
    """Perform format and validity checks for survival data.

    Args:
        event (torch.Tensor, boolean):
            Event indicator of size n_samples (= True if event occurred).
        time (torch.Tensor, float):
            Event or censoring time of size n_samples.

    Raises:
        TypeError: If ``event`` or ``time`` are not tensors.
        ValueError: If ``event`` is not boolean.
        ValueError: If ``event`` and ``time`` are not of the same length.
        ValueError: If all ``event`` are False.
        ValueError: If any ``time`` are negative.
    """
    validate_tensor(event, "event")
    validate_tensor(time, "time")

    if not event.dtype == torch.bool:
        raise ValueError("Input 'event' should be of boolean type.")

    if not torch.is_floating_point(time):
        raise ValueError("Input 'time' should be of float type.")

    if len(event) != len(time):
        raise ValueError(
            "Dimension mismatch: Incompatible length between inputs 'time' and 'event'."
        )

    if torch.sum(event) <= 0:
        raise ValueError("All samples are censored.")

    if torch.any(time < 0.0):
        raise ValueError("Input 'time' should be non-negative.")


def validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input '{name}' should be a tensor")


def validate_event(event: torch.Tensor) -> torch.Tensor:
    if event.dtype == torch.bool:
        event = event.float()  # convert boolean to float for internal computation
    if not torch.all((event == 0) | (event == 1)):
        raise ValueError("Invalid values: 'event' must contain only [0, 1] values")
    return event


def validate_model_type(log_params: torch.Tensor, model_type: str) -> None:
    if model_type.lower() == "weibull":
        if log_params.dim() not in [1, 2]:
            raise ValueError(
                f"For Weibull model, 'log_params' must have shape (n_samples, 2) or (n_samples, 1). Found {log_params.dim()} dimensions."
            )
    elif model_type.lower() == "cox":
        if log_params.dim() != 1:
            raise ValueError(
                "For Cox model, 'log_params' must have shape (n_samples, 1)."
            )
    else:
        raise ValueError("Invalid model type. Must be 'weibull' or 'cox'.")


def validate_loss(
    log_params: torch.Tensor, event: torch.Tensor, time: torch.Tensor, model_type: str
) -> None:
    # sanity checks
    validate_tensor(log_params, "log_params")
    validate_tensor(event, "event")
    validate_tensor(time, "time")

    log_params = log_params.squeeze()
    event = event.squeeze()
    time = time.squeeze()

    if log_params.shape[0] != len(event):
        raise ValueError(
            "Dimension mismatch: 'log_params' and 'event' must have the same length"
        )

    if time.shape != event.shape:
        raise ValueError(
            "Dimension mismatch: 'time' and 'event' must have the same shape"
        )

    if torch.any(time < 0):
        raise ValueError("All elements in 'time' must be non-negative.")

    event = validate_event(event)
    validate_model_type(log_params, model_type)


if __name__ == "__main__":
    log_params_weibull = torch.randn((5, 2))
    log_params_cox = torch.randn((5, 1))
    event_data = torch.tensor([1, 0, 1, 1, 0])
    time_data = torch.tensor([10, 20, 30, 40, 50])

    # Validate Weibull model inputs
    validate_loss(log_params_weibull, event_data, time_data, model_type="weibull")

    # Validate Cox model inputs
    validate_loss(log_params_cox, event_data, time_data, model_type="cox")

    # Valid booleans values
    validate_loss(log_params_cox, event_data.bool(), time_data, model_type="cox")

    # check that the output is a ValueError
    try:
        validate_loss(
            log_params_weibull,
            torch.tensor([1, 0, 1]),
            time_data,
            model_type="weibull",
        )
    except ValueError as e:
        print(f"ValueError: {e}")
