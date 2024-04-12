import torch


def validate_survival_data(event, time):
    """Perform format and validity checks for survival data.

    Args:
        event (torch.Tensor, boolean):
            Event indicator of size n_samples (= True if event occured).
        time (torch.Tensor, float):
            Event or censoring time of size n_samples.

    Raises:
        TypeError: If ``event`` or ``time`` are not tensors.
        ValueError: If ``event`` is not boolean.
        ValueError: If ``event`` and ``time`` are not of the same length.
        ValueError: If all ``event`` are False.
        ValueError: If any ``time`` are negative.
    """
    if not torch.is_tensor(event) or not torch.is_tensor(time):
        raise TypeError("Inputs 'event' and 'time' should be tensors")

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


def validate_evaluation_time(new_time, time, within_follow_up=True):
    """Perform format and validity checks for evaluation time.

    Args:
        new_time (torch.Tensor, optional):
            Time points for metric computation of size n_times.
        time (torch.Tensor, float):
            Event or censoring time of size n_samples.
        within_follow_up (bool, optional):
            Whether values of ``new_time`` must be within values in ``time``.
            Defaults to True.

    Raises:
        ValueError: If ``new_time`` contains duplicate values.
        ValueError: If ``new_time`` is not sorted.
        TypeError: If ``new_time`` is not a tensor.
        ValueError: If ``new_time`` is not of floating-point type.
        ValueError:
            If ``new_time`` is not within the range of follow-up in ``time``.
            Assessed only if ``within_follow_up`` is True.
    """
    new_time_sorted, indices = torch.unique(new_time, return_inverse=True)

    if len(new_time_sorted) != len(new_time):
        raise ValueError("'Value error: Input 'new_time' should contain unique values.")

    if not torch.all(indices == torch.arange(len(new_time))):
        raise ValueError(
            "Value error: Input 'new_time' should be sorted from the smallest time to the largest."
        )

    if not torch.is_tensor(new_time):
        raise TypeError("Type error: Input 'new_time' should be a tensor.")

    if not torch.is_floating_point(new_time):
        raise ValueError(
            "Value error: Input 'new_time' should be of floating-point type."
        )

    if within_follow_up:
        if new_time.max() >= time.max() or new_time.min() < time.min():
            raise ValueError(
                "Value error: All new_time must be within follow-up time of test data: [{}; {}[".format(
                    time.min(), time.max()
                )
            )


def validate_estimate(estimate, time, new_time=None) -> None:
    """Perform format and validity checks for estimate.

    Args:
        estimate (torch.Tensor):
            Estimates of shape = (n_samples,) or (n_samples, n_times).
        time (torch.Tensor, float):
            Time of event or censoring of size n_samples.
        new_time (torch.Tensor, optional):
            Time points for metric computation of size n_times.

    Raises:
        TypeError: If ``estimate`` is not a tensor.
        ValueError: If ``estimate`` has more than 2 dimensions.
        ValueError: If number of rows of ``estimate`` is not n_samples.
        ValueError:
            If number of columns of ``estimate`` is not n_times.
            Assessed only if ``new_time`` is not None.
    """
    if not torch.is_tensor(estimate):
        raise TypeError("Type error: Input 'estimate' should be a tensor.")

    if estimate.ndim > 2:
        raise ValueError("Value error: Input 'estimate' should have 1 or 2 dimensions.")

    if estimate.shape[0] != time.shape[0]:
        raise ValueError(
            "Dimension mismatch: Inconsistent number of samples between inputs 'time' and 'estimate'."
        )

    if not new_time is None:
        if estimate.ndim == 2 and estimate.shape[1] != new_time.shape[0]:
            raise ValueError(
                "Dimension mismatch: Expected  inputs 'estimate' with {} columns, but got {}".format(
                    new_time.shape[0], estimate.shape[1]
                )
            )
