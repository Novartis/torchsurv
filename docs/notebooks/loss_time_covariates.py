import sys
import warnings

import torch


def neg_partial_time_log_likelihood(
    log_hz: torch.Tensor, #Txnxp torch tensor, n is batch size, T number of time points, p is number of different covariates over time
    time: torch.Tensor, #n length vector, time at which someone experiences event
    events: torch.Tensor, #n length vector, boolean, true or false to determine if someone had an event
    reduction: str = "mean"
) -> torch.Tensor:
    '''
    needs further work
    '''
    # only consider theta at tiem of
    pll = _partial_likelihood_time_cox(log_hz, time, events)
   
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

def _partial_likelihood_time_cox(
    log_hz: torch.Tensor, #Txnxp torch tensor, n is batch size, T number of time points, p is number of different covariates over time
    time: torch.Tensor, #n length vector, time at which someone experiences event
    events: torch.Tensor, #n length vector, boolean, true or false to determine if someone had an event
    
) -> torch.Tensor:
    """
    Calculate the partial log likelihood for the Cox proportional hazards model
    with time-varying covariates and in the absence of ties in event time.

    Args:
        log_hz (torch.Tensor, float):
            Log relative hazard of dimension T x n_samples x P.
            T is the time series dimension, P is the number of parameters observed over time.
        event (torch.Tensor, bool):
            Event indicator of length n_samples (= True if event occured).
        time (torch.Tensor):
            Time-to-event or censoring of length n_samples.

    Returns:
        (torch.tensor, float):
            Vector of the partial log likelihood, length n_samples.

    Note:
        For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`\tau^*_i` as the survival time and :math:`C_i` as the
        censoring time. Survival data consist of the event indicator, :math:`\delta_i=1(\tau^*_i\leq C_i)`
        (argument ``event``) and the time-to-event or censoring, :math:`\tau_i = \min(\{ \tau^*_i,D_i \})`
        (argument ``time``).

        Consider some covariate :math:`Z(t)` with covariate history denoted as :math:`H_Z` and a general form of the cox proportional hazards model:
            .. math::

            \log \lambda_i (t|H_Z) = lambda_0(t) \theta(Z(t))
        
        A network that maps the input covariates $Z(t)$ to the log relative hazards: :math:`\log \theta(Z(t))`.
        The partial likelihood with repsect to  :math:`\log \theta(Z(t))` is written as:

            .. math::

             \log L(\theta) = \sum_j \Big( \log \theta(Z_i(\tau_j)) - \log [\sum_{j \in R_i} \theta (Z_i(\tau_j))] \Big)

        and it only considers the values of te covariate :math:`Z` at event time :math:`\tau_i`

    Remarks:
    - values inside the time vector must be strictly zero or positive as they are used to identify values of 
        covariates at event time
    - the maximum value inside the vector time cannt exceed T-1 for indexing reasons
    - this function was not tested for P>1 but it should be possile for an extension
    - the values of Z at event time should not be null, a reasonable imputation method should be used, 
        unless the network fullfills that role
    - future formatting: time vector must somehow correspond to the T dimension in the log_hz tensor, i.e. for those who experience an event,
        we want to identify the index of the covariate upon failure. We could either consider the last covariate before a series of zeros 
        (requires special data formatting but could reduce issues as it automatically contains event time information).


    """
    # time cannot be smaller than zero, and maximum value cannot exceed the T dimension for this to work
    # somehwere here it might be good to make sure maximum values in time do not exceed T and raise a warning
    time_sorted, idx = torch.sort(time)
    
    # sort the output of the RNN by the subjects who have earlier event time
    # we want a tensor out
    log_hz_sorted = log_hz[:,idx,:]
    events_sorted = events[idx]

    #format the time so we can use it to index
    #in the next step we want to pick out the covariate at event time for each subject for each covariate p
    time_sorted=time_sorted.type(torch.int64)

    #as an outcome we want an 1xnxp tensor aka. we only cosnider Z(tau_j), a covariate at time of event
    log_hz_sorted_tj = log_hz_sorted.gather(0, time_sorted.unsqueeze(0).unsqueeze(-1))

    #same step as in normal cox loss, just again, we consider Z(tau_j) where tau_j denotes event time to subject j
    log_denominator_tj = torch.logcumsumexp(log_hz_sorted_tj.flip(0), dim=0).flip(0)
    #give the mask the same dimensions as the log_hz and log_denominator vectors
    event_mask = events_sorted.unsqueeze(0).unsqueeze(-1)
    return (log_hz_sorted_tj - log_denominator_tj)[event_mask]


def _time_varying_covariance(
    log_hz: torch.Tensor, #nx1 vector
    event: torch.Tensor, #n vector (i think)
    time: torch.Tensor, #n vector (i think)
    covariates: torch.Tensor, #nxp vector, p number of params
) -> torch.Tensor:
    """ Calculate the covariance matrix for the outcome thetas from a network in
    in the case of time-varying covariates. Returns a nxn matrix with n being the batch size."""
    # sort data by time-to-event or censoring
    time_sorted, idx = torch.sort(time)
    log_hz_sorted = log_hz[idx]
    event_sorted = event[idx]

    #keep log if we can
    exp_log_hz = torch.exp(log_hz_sorted)
    #remove mean over time from covariates
    #sort covariates so that the rows match the ordering
    covariates_sorted = covariates[idx, :] - covariates.mean(dim=0)

    #the left hand side (HS) of the equation
    #below is Z_k Z_k^T - i think it should be a vector matrix dim nxn
    covariate_inner_product = torch.matmul(covariates_sorted, covariates_sorted.T)
    
    #pointwise multiplication of vectors to get the nominator of left HS
    #outcome in a vector of length n
    # Ends up being (1, n)
    log_nominator_left = torch.matmul(exp_log_hz.T, covariate_inner_product)

    #right hand size of the equation
    #formulate the brackets \sum exp(theta)Z_k
    bracket = torch.mul(exp_log_hz, covariates_sorted)
    covariance_matrix = torch.matmul(bracket, bracket.T) #nxn matrix
    # ###nbelow is commented out as it does not apply but I wanted to keep it for the functions
    # #log_nominator_right = torch.sum(nominator_right, dim=0).unsqueeze(0)
    # log_nominator_right = nominator_right[0,].unsqueeze(0)
    # log_denominator = torch.logcumsumexp(log_hz_sorted.flip(0), dim=0).flip(0) #dim=0 sums over the oth dimension
    # partial_log_likelihood = torch.div(log_nominator_left - log_nominator_right, log_denominator) # (n, n)
    
    return (covariance_matrix)