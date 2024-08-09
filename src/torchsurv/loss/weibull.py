import sys

import torch

TORCH_CLAMP_VALUE = 1e10


def neg_log_likelihood(
    log_params: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    reduction: str = "mean",
    checks: bool = True,
) -> torch.Tensor:
    r"""
    Negative of the log likelihood for the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        log_params (torch.Tensor, float):
            Parameters of the Weibull distribution of shape = (n_samples, 1) or (n_samples, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter.  If the log shape parameter is missing, it is
            imputed with 0.
        event (torch.Tensor, bool):
            Event indicator of length n_samples (= True if event occured).
        time (torch.Tensor, float):
            Time-to-event or censoring of length n_samples.
        reduction (str):
            Method to reduce losses. Defaults to "mean".
            Must be one of the following: "sum", "mean".
        checks (bool):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.

    Returns:
        (torch.Tensor, float): Negative of the log likelihood.

    Note:

        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(X_i\leq D_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
        (argument ``time``).

        The log hazard function for the Weibull AFT survival model :cite:p:`Carroll2003` of subject :math:`i` at time :math:`t` has the form:

        .. math::

            \log h_i(t) = \log{\rho_i} - \log{\lambda_i} + (\rho_i -1) \left( \log{t} - \log{\lambda_i}\right)

        where :math:`\log{\lambda_i}` is the log scale parameter (first column of argument ``log_params``)
        and :math:`\log{\rho_i}` is the log shape parameter (second column of argument ``log_params``).
        The cumulative hazard for the Weibull survival model of subject :math:`i` at time :math:`t` has the form:

        .. math::

            H_i(t) = \left(\frac{t}{\lambda_i}\right)^{\rho_i}

        The survival function for the Weibull survival model of subject :math:`i` at time :math:`t` has the form:

        .. math::

            S_i(t) = 1 - F(t | \lambda_i, \rho_i)

        where :math:`F(t | \lambda, \rho)` is the cumulative distribution function (CDF) of the Weibull distribution given
        scale parameter :math:`\lambda` and shape parameter :math:`\rho`.

        The log likelihood of the Weibull survival model is

        .. math::

            ll = \sum_{i: \delta_i = 1} \log h_i(T_i) - \sum_{i = 1}^N H_i(T_i)

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> n = 4
        >>> log_params = torch.randn((n, 2))
        >>> event = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
        >>> time = torch.randint(low=1, high=100, size=(n,))
        >>> neg_log_likelihood(log_params, event, time) # Default: mean of log likelihoods across subject
        tensor(47.5035)
        >>> neg_log_likelihood(log_params, event, time, reduction = 'sum') # Sum of log likelihoods across subject
        tensor(190.0141)
        >>> neg_log_likelihood(torch.randn((n, 1)), event, time)  # Missing shape: exponential decrease
        tensor(66.7203)

    References:

        .. bibliography::
            :filter: False

            Carroll2003

    """

    if checks:
        _check_inputs(log_params, event, time)

    # Negative log likelihood
    nll = torch.neg(
        event * log_hazard(log_params, time, all_times=False)
        - cumulative_hazard(log_params, time, all_times=False)  # Huge values here
    )

    if any(torch.isinf(nll)):
        # Remove any torch.inf values
        nll = nll[~torch.isinf(nll)]

    if reduction.lower() == "mean":
        loss = nll.nanmean()
    elif reduction.lower() == "sum":
        loss = nll.sum()
    else:
        raise (
            ValueError(
                f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."
            )
        )
    return loss


def survival_function(
    log_params: torch.Tensor, time: torch.Tensor, all_times: bool = True
) -> torch.Tensor:
    """Survival function for the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        log_params (torch.Tensor, float):
            Parameters of the Weibull distribution of shape = (n_samples, 1) or (n_samples, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        time (torch.Tensor, float):
            Time at which to evaluate the survival function.
            Should be of length n_samples to evaluate the survival function at observed time-to-event or censoring,
            or of length one to evaluate the survival function at a new time.
        all_times (bool):
            If True, subject-specific survival function is evaluated at all ``time`` (used for evaluation metrics).
            If False, subject-specific survival function is evaluated at respective ``time``.
            Defaults is True.
            Ignored if ``time`` is of length one.

    Returns:
        (torch.Tensor, float): Subject-specific survival function evaluated at ``time``.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> time = torch.randint(low=1, high=100, size=(4,))
        >>> log_params = torch.randn((4, 2))
        >>> survival_function(log_params, time, all_times = False)  # Survival at respective time
        tensor([0.0002, 0.0000, 0.0299, 0.0000])
        >>> survival_function(log_params, time, all_times = True)  # Default. Survival at all observed time
        tensor([[1.7941e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                [2.8610e-06, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                [4.1870e-01, 3.1040e-02, 2.9881e-02, 6.8224e-02],
                [9.5576e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
        >>> survival_function(log_params, time=torch.tensor(10.0))  # Survival at one new time (e.g., 10 years)
        tensor([1.3709e-06, 5.9605e-08, 3.4954e-01, 1.5438e-05])
        >>> for t in torch.tensor([100.0, 150.0]): survival_function(log_params, time=t)  # Subject-specific survival at multiple new times
        tensor([0.0000, 0.0000, 0.0288, 0.0000])
        tensor([0.0000, 0.0000, 0.0123, 0.0000])


    """
    log_scale, log_shape = _check_log_shape(log_params).unbind(1)

    if time.dim() == 0:
        # Use one time for each sample
        time = time.repeat(len(log_params))
    elif all([time.size(0) == log_params.size(0), all_times]):
        # Use all times for each sample
        time = time.unsqueeze(0).expand(len(time), len(time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns
    if time.size(0) != log_params.size(0):
        raise ValueError(
            f"Dimension mismatch: 'time' ({len(time)}) does not match the length of 'log_params' ({len(log_params)})."
        )
    return 1 - torch.distributions.weibull.Weibull(
        torch.exp(log_scale), torch.exp(log_shape)
    ).cdf(time)


def log_hazard(
    log_params: torch.Tensor, time: torch.Tensor, all_times: bool = True
) -> torch.Tensor:
    """Log hazard of the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        log_params (torch.Tensor, float):
            Parameters of the Weibull distribution of shape = (n_samples, 1) or (n_samples, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        time (torch.Tensor, float):
            Time at which to evaluate the log hazard.
            Should be of length n_samples to evaluate the log hazard at observed time-to-event or censoring,
            or of length one to evaluate the log hazard at a new time.
        all_times (bool):
            If True, subject-specific log hazard is evaluated at all ``time`` (used for evaluation metrics).
            If False, subject-specific log hazard is evaluated at respective ``time``.
            Defaults is True.
            Ignored if ``time`` is of length one.

    Returns:
        (torch.Tensor, float): Subject-specific log hazard evaluated at ``time``.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> time = torch.randint(low=1, high=100, size=(4,))
        >>> log_params = torch.randn((4, 2))
        >>> log_hazard(log_params, time, all_times = False)  # Log hazard at respective time
        tensor([ 0.4392, -0.0303, -3.9672,  0.9140])
        >>> log_hazard(log_params, time, all_times = True)  # Default. Log hazard at all time
        tensor([[ 0.4392,  1.1174,  1.1227,  0.9913],
                [ 0.4148, -0.0303, -0.0338,  0.0525],
                [-2.7225, -3.9575, -3.9672, -3.7279],
                [ 0.2606,  1.0632,  1.0695,  0.9140]])
        >>> log_hazard(log_params, time=torch.tensor(10.0))  # Log hazard at one new time (e.g., 10 years)
        tensor([ 0.5316,  0.3542, -2.8907,  0.3699])
        >>> for t in torch.tensor([100.0, 150.0]): log_hazard(log_params, time=t)  # Subject-specific log hazard at multiple new times
        tensor([ 1.1280, -0.0372, -3.9767,  1.0757])
        tensor([ 1.2330, -0.1062, -4.1680,  1.1999])
        >>> log_params  *= 1e2  # Increase scale
        >>> log_hazard(log_params, time, all_times = False)  # Check for Torch.Inf values
        tensor([-1.0000e+10, -2.3197e+01, -6.8385e+01, -1.0000e+10])
    """

    log_scale, log_shape = _check_log_shape(log_params).unbind(1)

    if time.dim() == 0:
        # Use fixed time for each sample
        time = time.repeat(len(log_params))
    elif all([time.size(0) == log_params.size(0), all_times]):
        # Use all times for each sample
        time = time.unsqueeze(0).expand(len(time), len(time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns
    if time.size(0) != log_params.size(0):
        raise ValueError(
            f"Dimension mismatch: 'time' ({len(time)}) does not match the length of 'log_params' ({len(log_params)})."
        )

    return torch.clamp(
        log_shape
        - log_scale
        + torch.expm1(log_shape)
        * (torch.log(torch.clip(time, 1e-100, torch.inf)) - log_scale),
        min=-TORCH_CLAMP_VALUE,
        max=TORCH_CLAMP_VALUE,
    )


def cumulative_hazard(
    log_params: torch.Tensor, time: torch.Tensor, all_times: bool = True
) -> torch.Tensor:
    """Cumulative hazard for the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        log_params (torch.Tensor, float):
            Parameters of the Weibull distribution of shape = (n_samples, 1) or (n_samples, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        time (torch.Tensor, float):
            Time-to-event or censoring of length n_samples.
        all_times (bool)
            If True, subject-specific cumulative hazard is evaluated at all ``time`` (used for evaluation metrics).
            If False, subject-specific cumulative hazard is evaluated at respective ``time``.
            Defaults is True.

    Returns:
        (torch.Tensor, float): Subject-specific cumulative hazard evaluated at ``time``.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> time = torch.randint(low=1, high=100, size=(4,))
        >>> log_params = torch.randn((4, 2))
        >>> cumulative_hazard(log_params, time, all_times=False) # Cumulative hazard at respective time
        tensor([  8.6257, 112.2115,   3.5105, 112.6339])
        >>> cumulative_hazard(log_params, time, all_times=True) # Default. Cumulative hazard at all time
        tensor([[  8.6257, 233.0865, 239.2167, 126.2805],
                [ 12.7698, 112.2115, 114.1484,  74.9134],
                [  0.8706,   3.4725,   3.5105,   2.6850],
                [  6.9530, 212.7592, 218.5687, 112.6339]])
    """
    log_scale, log_shape = _check_log_shape(log_params).unbind(1)

    if all_times:
        # Use all times for each sample
        time = time.unsqueeze(0).expand(len(time), len(time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(
            len(time), len(time)
        )  # expand across columns

    return torch.clamp(
        torch.exp(
            torch.exp(log_shape)
            * (torch.log(torch.clip(time, 1e-100, torch.inf)) - log_scale)
        ),
        min=0,
        max=TORCH_CLAMP_VALUE,
    )


def _check_log_shape(log_params: torch.Tensor) -> torch.Tensor:
    """Private function, check if the log shape is missing and impute it with 0
    if needed."""
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


def _check_inputs(log_params: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
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

    if any(val < 0 for val in time):
        raise ValueError("All elements in 'time' must be non-negative.")

    if any(val not in [True, False, 0, 1] for val in event):
        raise ValueError("All elements in 'event' must be boolean (True/False or 0/1).")


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
