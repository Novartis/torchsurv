import sys
import warnings
from typing import Optional

import torch

from ..tools import validate_inputs
from . import kaplan_meier


# pylint: disable=anomalous-backslash-in-string
def get_ipcw(
    event: torch.tensor,
    time: torch.tensor,
    new_time: Optional[torch.tensor] = None,
    checks: bool = True,
) -> torch.Tensor:
    """Calculate the inverse probability censoring weights (IPCW).

    Args:
        event (torch.Tensor, boolean):
            Event indicator of size n_samples (= True if event occured).
        time (torch.Tensor, float):
            Time-to-event or censoring of size n_samples.
        new_time (torch.Tensor, float, optional):
            New time at which to evaluate the IPCW.
            Defaults to ``time``.
        checks (bool):
            Whether to perform input format checks.
            Enabling checks can help catch potential issues in the input data.
            Defaults to True.
    Returns:
        torch.Tensor: IPCW evaluated at ``new_time``.

    Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 5
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> time = torch.randint(low=1, high=100, size=(n,)).float()
            >>> new_time = torch.randint(low=1, high=100, size=(n*2,))
            >>> get_ipcw(event, time) # ipcw evaluated at time
            tensor([1.8750, 1.2500, 3.7500, 0.0000, 1.2500])
            >>> get_ipcw(event, time, new_time) # ipcw evaluated at new_time
            tensor([1.8750, 1.8750, 3.7500, 3.7500, 0.0000, 1.2500, 0.0000, 1.2500, 1.2500,
                    1.2500])

    Note:
        The inverse probability of censoring weight at time :math:`t` is defined by

        .. math::
            \omega(t) = 1 / \hat{D}(t),

        where :math:`\hat{D}(t)` is the Kaplan-Meier estimate of the censoring distribution, :math:`P(D>t)`.


    """

    if checks:
        validate_inputs.validate_survival_data(event, time)

    # time on which to evaluate IPCW
    if new_time is None:  # if none, return ipcw of same size as time
        new_time = time

    # fit KM censoring estimator
    km = kaplan_meier.KaplanMeierEstimator()
    km(event, time, censoring_dist=True)

    # predict censoring distribution at time
    ct = km.predict(new_time)

    # caclulate ipcw
    mask = ct > 0
    ipcw = torch.zeros_like(new_time, dtype=time.dtype)
    ipcw[mask] = _inverse_censoring_dist(ct[mask])

    return ipcw


def _inverse_censoring_dist(ct: torch.Tensor) -> torch.Tensor:
    """Compute inverse of the censoring distribution.

    Args:
        ct (torch.Tensor):
            Estimated censoring distribution.
            Must be strictly positive.

    Returns:
        torch.Tensor: 1/ct if ct > 0, else 1.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> ct = torch.randn((4,))
        >>> _inverse_censoring_dist(ct)
        tensor([2.9701, 7.7634, 4.2651, 4.3415])

    """
    if torch.any(ct.eq(0.0)):
        warnings.warn(
            "Censoring distribution zero at one or more time points. Returning ones as weight"
        )
        return torch.ones_like(ct, dtype=ct.dtype)
    weight = torch.ones(1, dtype=ct.dtype) / ct
    return weight


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
