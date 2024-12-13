import sys
import warnings

import torch

def time_partial_log_likelihood(
    log_hz: torch.Tensor, #nx1 vector
    event: torch.Tensor, #n vector (i think)
    time: torch.Tensor, #n vector (i think)
    covariates: torch.Tensor, #nxp vector, p number of params
) -> torch.Tensor:

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
    nominator_right = torch.matmul(bracket, bracket.T) #nxn matrix
    ###not sure if the next line is this
    #log_nominator_right = torch.sum(nominator_right, dim=0).unsqueeze(0)
    ### or this
    log_nominator_right = nominator_right[0,].unsqueeze(0)
    #the denominator is the same on both sides
    log_denominator = torch.logcumsumexp(log_hz_sorted.flip(0), dim=0).flip(0) #dim=0 sums over the oth dimension
    partial_log_likelihood = torch.div(log_nominator_left - log_nominator_right, log_denominator) # (n, n)
    
    return (partial_log_likelihood)[event_sorted]