import sys
import warnings

import torch


def neg_partial_time_log_likelihood(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    ties_method: str = "efron",
    reduction: str = "mean",
    checks: bool = True,
) -> torch.Tensor:
    '''
    THIS FUNCTION IS NOT DONE, i HAVENT TESTED THE NEGATIVE PART YET
    '''
    if checks:
        _check_inputs(log_hz, event, time)

    if any([event.sum() == 0, len(log_hz.size()) == 0]):
        warnings.warn("No events OR single sample. Returning zero loss for the batch")
        return torch.tensor(0.0, requires_grad=True)

    # sort data by time-to-event or censoring
    time_sorted, idx = torch.sort(time)
    log_hz_sorted = log_hz[idx]
    event_sorted = event[idx]
    time_unique = torch.unique(time_sorted)  # time-to-event or censoring without ties

    # only consider theta at tiem of
    pll = _partial_likelihood_time_cox(log_hz_sorted, event_sorted)
   
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
    log_hz: torch.Tensor, #nxTxp torch tensor, n is batch size, T number of time points, p is number of different covariates over time
    event: torch.Tensor, #n length vector, boolean, true or false to determine if someone had an event
    time: torch.Tensor, #n length vector, time at which someone experiences event
) -> torch.Tensor:
    """Calculate the partial log likelihood for the Cox proportional hazards model
    with time-varying covariates and in the absence of ties in event time.

    For time-varying covariates, the haard ratio is no longer assumed to be constant, 
    but the partial log likelihood only cares about the covariate value at time of death.

    Hence, despite taking in a whole vector of stuff, we only take the last value 
    into consideration for the partial log likelihood.

    Requirements we want:
    - time vector must somehow correspond to the T dimension in the log_hz tensor, i.e. for those who experience an event,
        we want to identify the index of the covariate upon failure. We could either consider the last covariate before a series of zeros 
        (requires special data formatting but could reduce issues as it automatically contains event time information).
    - this version doesn't allow for P>1 but it can be considered as an additional dimension and then in the final
        step you can take the mean across p
    - we want values of the covariate at event time to not be null, maybe there could be some automation function that imputes the latest values if possible
    - maybe some guidance can go here on how to format the covariates, right now its just a tensor.
    """

    time_sorted, idx = torch.sort(time)
    #sort the output of the RNN by the subjects who have earlier event time
    #we want a tensor out
    log_hz_sorted = outputs[:,idx,:]
    event_sorted = events[idx]

    #format the time so we can use it to index
    #in the next step we want to pick out the covariate at event time for each subject for each covariate p
    #this line is just to be able to index - can be changed depending on how time is formatted
    time_sorted=time_sorted.type(torch.int64)
    # below is pseudocode of what to do to geth the log likelihood
    #as an outcome we want an nx1xp tensor aka. time is reduced and we only cosnider Z(tau_j)
    log_hz_sorted_tj = log_hz_sorted[time_sorted, :, :]

    #same step as in normal cox loss, just again, we consider Z(tau_j) where tau_j denotes event time to subject j
    log_denominator_tj = torch.logcumsumexp(log_hz_sorted_tj.flip(0), dim=0).flip(0)

    return (log_hz_sorted_tj - log_denominator_tj)[event_sorted]


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