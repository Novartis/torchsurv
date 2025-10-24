import sys

import torch

from torchsurv.tools.validate_data import (
    _impute_missing_log_shape,
    validate_model,
    validate_survival_data,
)

__all__ = [
    "neg_log_likelihood",
    "log_hazard",
    "survival_function",
]


def _cumulative_hazard(
    new_log_params: torch.Tensor,
    new_time: torch.Tensor,
    respective_times: bool = False,
    clamp_value: float = 1e10,
) -> torch.Tensor:
    """Cumulative hazard for the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        new_log_params (torch.Tensor, float):
            Parameters of the Weibull distribution for new subjects,
            of shape = (n_samples_new, 1) or (n_samples_new, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        new_time (torch.Tensor, float):
            Time at which to evaluate the cumulative hazard of length n_times.
        respective_times (bool, optional):
            If True, ``new_time`` must have the same length as ``new_log_params``.
            The subject-specific cumulative hazard is then evaluated at each corresponding value in ``new_time``.
            Defaults to False.
        clamp_value (float, optional):
            Maximum value to which the cumulative hazard is clipped.
            This prevents numerical overflow or instability by capping extremely large values of the cumulative hazard.
            Defaults to 1e10.

    Returns:
        (torch.Tensor, float): Subject-specific cumulative hazard evaluated at ``new_time``.
        Shape = (n_samples_new, n_times) if respective_times is False.
        Shape = (n_samples_new,) if respective_times is True.

    Examples:
        >>> new_log_params = torch.tensor([[0.15, 0.25], [0.1, 0.2]])  # 2 new subjects
        >>> new_time = torch.tensor([1.0, 2.0])
        >>> _cumulative_hazard(new_log_params, new_time)
        tensor([[0.8248, 2.0086],
                [0.8850, 2.0636]])
        >>> _cumulative_hazard(new_log_params, new_time, respective_times=True)
        tensor([0.8248, 2.0636])
    """
    log_scale, log_shape = _impute_missing_log_shape(new_log_params).unbind(1)

    if new_time.dim() == 0:
        # Use one time for each sample
        time = new_time.repeat(len(new_log_params))
    elif respective_times and new_time.size(0) == new_log_params.size(0):
        time = new_time
    else:
        # Use new time for each sample
        time = new_time.unsqueeze(0).expand(len(log_scale), len(new_time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(time.shape)  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(time.shape)  # expand across columns

    return torch.clamp(
        torch.exp(torch.exp(log_shape) * (torch.log(torch.clamp(time, min=1e-100, max=torch.inf)) - log_scale)),
        min=0,
        max=clamp_value,
    )


def log_hazard(
    new_log_params: torch.Tensor,
    new_time: torch.Tensor,
    respective_times: bool = False,
    clamp_value: float = 1e10,
) -> torch.Tensor:
    """Log hazard of the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        new_log_params (torch.Tensor, float):
            Parameters of the Weibull distribution for new subjects,
            of shape = (n_samples_new, 1) or (n_samples_new, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        new_time (torch.Tensor, float):
            Time at which to evaluate the log hazard of length n_times.
        respective_times (bool, optional):
            If True, ``new_time`` must have the same length as ``new_log_params``.
            The subject-specific log hazard is then evaluated at each respective index in ``new_time``.
            Defaults to False.
        clamp_value (float, optional):
            Maximum value to which the log hazard is clipped.
            This prevents numerical overflow or instability by capping extremely large values of the log hazard.
            Defaults to 1e10.

    Returns:
        (torch.Tensor, float): Subject-specific log hazard evaluated at ``new_time``.
        Shape = (n_samples_new, n_times) if ``respective_times`` is False.
        Shape = (n_samples_new,) if ``respective_times`` is True.

    Examples:
        >>> new_log_params = torch.tensor([[0.15, 0.25], [0.1, 0.2]])  # 2 new subjects
        >>> new_time = torch.tensor([1.0, 2.0])
        >>> log_hazard(new_log_params, new_time)
        tensor([[0.0574, 0.2543],
                [0.0779, 0.2313]])
        >>> log_hazard(new_log_params, new_time, respective_times=True)
        tensor([0.0574, 0.2313])
    """

    log_scale, log_shape = _impute_missing_log_shape(new_log_params).unbind(1)

    if new_time.dim() == 0:
        # Use one time for each sample
        time = new_time.repeat(len(new_log_params))
    elif respective_times and new_time.size(0) == new_log_params.size(0):
        time = new_time
    else:
        # Use new time for each sample
        time = new_time.unsqueeze(0).expand(len(log_scale), len(new_time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(time.shape)  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(time.shape)  # expand across columns

    return torch.clamp(
        log_shape
        - log_scale
        + torch.expm1(log_shape) * (torch.log(torch.clamp(time, min=1e-100, max=torch.inf)) - log_scale),
        min=-clamp_value,
        max=clamp_value,
    )


def survival_function(
    new_log_params: torch.Tensor,
    new_time: torch.Tensor,
) -> torch.Tensor:
    """Survival function for the Weibull Accelerated Time Failure (AFT) survival model.

    Args:
        new_log_params (torch.Tensor, float):
            Parameters of the Weibull distribution for new subjects,
            of shape = (n_samples_new, 1) or (n_samples_new, 2).
            The first column corresponds to the log scale parameter. The second column
            corresponds to the log shape parameter. If the log shape parameter is missing, it is
            imputed with 0.
        new_time (torch.Tensor, float):
            Time at which to evaluate the survival probability of length n_times.

    Returns:
        torch.Tensor:
            Individual survival probabilities for each new subject at ``new_time``. Shape = (n_samples_new, n_times).

    Examples:
        >>> new_log_params = torch.tensor([[0.15, 0.25], [0.1, 0.2]])  # 2 new subjects
        >>> new_time = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> survival_function(new_log_params, new_time)  #  Survival at new times
        tensor([[0.4383, 0.1342, 0.0340, 0.0075],
                [0.4127, 0.1270, 0.0338, 0.0081]])

    """
    log_scale, log_shape = _impute_missing_log_shape(new_log_params).unbind(1)

    if new_time.dim() == 0:
        # Use one time for each sample
        time = new_time.repeat(len(new_log_params))
    else:
        # Use new time for each sample
        time = new_time.unsqueeze(0).expand(len(log_scale), len(new_time))  # expand across rows
        log_scale = log_scale.unsqueeze(1).expand(time.shape)  # expand across columns
        log_shape = log_shape.unsqueeze(1).expand(time.shape)  # expand across columns

    return 1 - torch.distributions.weibull.Weibull(torch.exp(log_scale), torch.exp(log_shape)).cdf(time)


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
            Event indicator of length n_samples (= True if event occurred).
        time (torch.Tensor, float):
            Event or censoring time of length n_samples.
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
        (argument ``event``) and the event or censoring time, :math:`T_i = \min(\{ X_i,D_i \})`
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
        >>> _ = torch.manual_seed(43)
        >>> n = 4
        >>> log_params = torch.randn((n, 2), dtype=torch.float)
        >>> event = torch.randint(low=0, high=2, size=(n,), dtype=torch.bool)
        >>> time = torch.randint(low=1, high=100, size=(n,), dtype=torch.float)
        >>> neg_log_likelihood(log_params, event, time)  # Default: mean of log likelihoods across subject
        tensor(143039.2656)
        >>> neg_log_likelihood(log_params, event, time, reduction="sum")  # Sum of log likelihoods across subject
        tensor(572157.0625)
        >>> neg_log_likelihood(
        ...     torch.randn((n, 1), dtype=torch.float), event, time
        ... )  # Missing shape: exponential distribution
        tensor(67.4289)

    References:

        .. bibliography::
            :filter: False

            Carroll2003

    """

    if checks:
        validate_survival_data(event, time)
        validate_model(log_params, event, model_type="weibull")

    # ensure event and time are 1-dimensional
    event = event.squeeze()
    time = time.squeeze()

    # Negative log likelihood
    nll = torch.neg(
        event * log_hazard(log_params, time, True) - _cumulative_hazard(log_params, time, True)  # Huge values here
    )

    if any(torch.isinf(nll)):
        # Remove any torch.inf values
        nll = nll[~torch.isinf(nll)]

    if reduction.lower() == "mean":
        loss = nll.nanmean()
    elif reduction.lower() == "sum":
        loss = nll.sum()
    else:
        raise (ValueError(f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."))
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
