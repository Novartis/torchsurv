import torch


def neg_partial_time_log_likelihood(
    log_hz: torch.Tensor,
    time: torch.Tensor,
    events: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the negative partial log-likelihood for time-dependent covariates in a Cox proportional hazards model.
    Args:
        log_hz (torch.Tensor): A tensor of shape (T, n, p) where T is the number of time points, n is the batch size,
                               and p is the number of different covariates over time.
        time (torch.Tensor): A tensor of length n representing the time at which an event occurs for each individual.
        events (torch.Tensor): A boolean tensor of length n indicating whether an event occurred (True) or not (False) for each individual.
        reduction (str, optional): Specifies the reduction to apply to the output: 'mean' | 'sum'. Default is 'mean'.
    Returns:
        torch.Tensor: The computed negative partial log-likelihood. If reduction is 'mean', returns the mean value.
                      If reduction is 'sum', returns the sum of the values.
    Raises:
        ValueError: If the specified reduction method is not 'mean' or 'sum'.

    Examples:
        >>> _ = torch.manual_seed(52)
        >>> n = 10  # number of samples
        >>> t = 5  # time steps
        >>> k = 16  # # covariates
        >>> time = torch.randint(low=5, high=250, size=(n,)).float()
        >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
        >>> x = torch.rand((t, n, k))
        >>> h0 = torch.randn(t, n, 1)
        >>> rnn = torch.nn.RNN(k, 1, t)
        >>> estimates, _ = rnn(x, h0)
        >>> neg_partial_time_log_likelihood(estimates, time, event)
        tensor(0.9452, grad_fn=<DivBackward0>)
        >>> neg_partial_time_log_likelihood(estimates.squeeze(), time, event)  # Also works with 2D tensor
        tensor(0.9452, grad_fn=<DivBackward0>)
        >>> neg_partial_time_log_likelihood(estimates, time, event, reduction="sum")
        tensor(37.8082, grad_fn=<SumBackward0>)
        >>> from torchsurv.metrics.cindex import ConcordanceIndex
        >>> cindex = ConcordanceIndex()
        >>> cindex_t = torch.stack([cindex(log_hz_t, event, time) for log_hz_t in estimates.unbind(0)])
        >>> cindex_t  # Compute c-index for each time step t
        tensor([0.6061, 0.2424, 0.5758, 0.3333, 0.5152])
        >>> cindex_t.mean()  # Average over all time steps t
        tensor(0.4545)
    """

    # only consider theta at time of
    pll = _partial_likelihood_time_cox(log_hz, time, events)

    # Negative partial log likelihood
    pll = torch.neg(pll)
    if reduction.lower() == "mean":
        loss = pll.nanmean()
    elif reduction.lower() == "sum":
        loss = pll.sum()
    else:
        raise (ValueError(f"Reduction {reduction} is not implemented yet, should be one of ['mean', 'sum']."))
    return loss


def _partial_likelihood_time_cox(
    log_hz: torch.Tensor,
    time: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the partial log likelihood for the Cox proportional hazards model
    with time-varying covariates and in the absence of ties in event time.

    Args:
        log_hz (torch.Tensor, float):
            Log relative hazard of dimension T x n_samples x P.
            T is the time series dimension, P is the number of parameters observed over time.
        event (torch.Tensor, bool):
            Event indicator of length n_samples (= True if event occurred).
        time (torch.Tensor):
            Time-to-event or censoring of length n_samples.

    Returns:
        (torch.tensor, float):
            Vector of the partial log likelihood, length n_samples.

    Note:
        For each subject :math:`i \\in \\{1, \\cdots, N\\}`, denote :math:`\tau^*_i` as the survival time and :math:`C_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\\delta_i=1(\tau^*_i\\leq C_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`\tau_i = \\min(\\{ \tau^*_i,D_i \\})`
        (argument ``time``).

        Consider some covariate :math:`Z(t)` with covariate history denoted as :math:`H_Z` and a general form of the cox proportional hazards model:
            .. math::

            \\log \\lambda_i (t|H_Z) = lambda_0(t) \theta(Z(t))

        A network that maps the input covariates $Z(t)$ to the log relative hazards: :math:`\\log \theta(Z(t))`.
        The partial likelihood with respect to  :math:`\\log \theta(Z(t))` is written as:

            .. math::

             \\log L(\theta) = \\sum_j \\Big( \\log \theta(Z_i(\tau_j)) - \\log [\\sum_{j \\in R_i} \theta (Z_i(\tau_j))] \\Big)

        and it only considers the values of te covariate :math:`Z` at event time :math:`\tau_i`

    Remarks:
    - values inside the time vector must be strictly zero or positive as they are used to identify values of
        covariates at event time
    - the maximum value inside the vector time cannot exceed T-1 for indexing reasons
    - this function was not tested for P>1 but it should be possible for an extension
    - the values of Z at event time should not be null, a reasonable imputation method should be used,
        unless the network fulfills that role
    - future formatting: time vector must somehow correspond to the T dimension in the log_hz tensor, i.e. for those who experience an event,
        we want to identify the index of the covariate upon failure. We could either consider the last covariate before a series of zeros
        (requires special data formatting but could reduce issues as it automatically contains event time information).

    Examples:
        >>> _ = torch.manual_seed(52)
        >>> n = 3  # number of samples
        >>> t = 3  # time steps
        >>> time = torch.randint(low=5, high=250, size=(n,)).float()
        >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
        >>> log_hz = torch.rand((t, n, 1))
        >>> _partial_likelihood_time_cox(log_hz, time, event)
        tensor([-1.3772, -1.0683, -0.7879, -0.8220,  0.0000,  0.0000])
    """
    # Last dimension must be equal to 1 if shape == 3
    if len(log_hz.shape) == 3:
        assert log_hz.shape[-1] == 1, "Last dimension of log_hz must be equal to 1."
        log_hz = log_hz.squeeze(-1)

    # time cannot be smaller than zero, and maximum value cannot exceed the T dimension for this to work
    if time.min() < 0:
        raise ValueError("Time values must be greater or equal to zero.")

    # Sort the time vector and the output of the RNN by the subjects who have earlier event time
    _, idx = torch.sort(time)

    # Sort the output of the RNN by the subjects who have earlier event time
    log_hz_sorted = log_hz[:, idx]
    events_sorted = events[idx]

    # as an outcome we want an 1xn tensor aka. we only consider Z(tau_j), a covariate at time of event
    log_hz_sorted_tj = torch.gather(log_hz_sorted, 1, idx.expand(log_hz_sorted.size()))

    # same step as in normal cox loss, just again, we consider Z(tau_j) where tau_j denotes event time to subject j
    log_cumulative_hazard = torch.logcumsumexp(log_hz_sorted.flip(0), dim=0).flip(0)

    # Keep only patients with events
    include = events_sorted.expand(log_hz_sorted.size())

    # return the partial log likelihood
    return (log_hz_sorted_tj - log_cumulative_hazard)[include]


# Code below will be either deleted or moved to another file (e.g. stats)
def _time_varying_covariance(
    log_hz: torch.Tensor,  # nx1 vector
    event: torch.Tensor,  # n vector (i think)
    time: torch.Tensor,  # n vector (i think)
    covariates: torch.Tensor,  # nxp vector, p number of params
) -> torch.Tensor:
    """Calculate the covariance matrix for the outcome thetas from a network in
    in the case of time-varying covariates. Returns a nxn matrix with n being the batch size.
    """
    # sort data by time-to-event or censoring
    time_sorted, idx = torch.sort(time)
    log_hz_sorted = log_hz[idx]

    # keep log if we can
    exp_log_hz = torch.exp(log_hz_sorted)
    # remove mean over time from covariates
    # sort covariates so that the rows match the ordering
    covariates_sorted = covariates[idx, :] - covariates.mean(dim=0)

    # right hand size of the equation
    # formulate the brackets \sum exp(theta)Z_k
    bracket = torch.mul(exp_log_hz, covariates_sorted)
    covariance_matrix = torch.matmul(bracket, bracket.T)  # nxn matrix

    return covariance_matrix


if __name__ == "__main__":
    import doctest
    import sys

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
