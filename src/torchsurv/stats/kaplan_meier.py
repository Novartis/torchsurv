import itertools
import sys
from typing import Tuple

import torch
import pandas as pd

from torchsurv.tools.validate_data import validate_survival_data

__all__ = [
    "KaplanMeierEstimator",
]


class KaplanMeierEstimator:
    """Kaplan-Meier estimate of survival or censoring distribution for right-censored data :cite:p:`Kaplan1958`."""

    def __init__(self, device: str = None):
        """
        Args:
            device (str, optional): Device to use for tensor computations (e.g., 'cpu', 'cuda'). Defaults to None (uses CPU).
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def __call__(
        self,
        event: torch.Tensor,
        time: torch.Tensor,
        censoring_dist: bool = False,
        check: bool = True,
    ):
        """Initialize Kaplan Meier estimator.

        Args:
            event (torch.tensor, bool):
                Event indicator of size n_samples (= True if event occurred).
            time (torch.tensor, float):
                Time-to-event or censoring of size n_samples.
            censoring_dist (bool, optional):
                If False, returns the Kaplan-Meier estimate of the survival distribution.
                If True, returns the Kaplan-Meier estimate of the censoring distribution.
                Defaults to False.
            check (bool):
                Whether to perform input format checks.
                Enabling checks can help catch potential issues in the input data.
                Defaults to True.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 32
            >>> time = torch.randint(low=0, high=8, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> s = KaplanMeierEstimator()  # estimate survival distribution
            >>> s(event, time)
            >>> s.km_est
            tensor([1.0000, 1.0000, 0.8214, 0.7143, 0.6391, 0.6391, 0.5113, 0.2556])
            >>> c = KaplanMeierEstimator()  # estimate censoring distribution
            >>> c(event, time, censoring_dist=True)
            >>> c.km_est
            tensor([0.9688, 0.8750, 0.8750, 0.8312, 0.6357, 0.4890, 0.3667, 0.0000])

        References:

        .. bibliography::
            :filter: False

            Kaplan1958
        """

        # create attribute state
        # pylint: disable=attribute-defined-outside-init
        self.event = event.to(self.device)
        self.time = time.to(self.device)

        # Check input validity if required
        if check:
            validate_survival_data(event, time)

        # Compute the counts of events, censorings, and the number at risk at each unique time
        uniq_times, n_events, n_at_risk, n_censored = self._compute_counts()

        # If 'censoring_dist' is True, estimate the censoring distribution instead of the survival distribution
        if censoring_dist:
            n_at_risk -= n_events
            n_events = n_censored

        # Compute the Kaplan-Meier estimator
        ratio = torch.where(
            n_events != 0,  # Check if the number of events is not equal to zero
            n_events / n_at_risk,  # Element-wise division when the number of events is not zero
            torch.zeros_like(
                n_events, dtype=torch.float
            ),  # Set to zero when the number of events is zero to avoid division by zero
        )
        values = 1.0 - ratio  # Compute the survival (or censoring) probabilities at each unique time
        y = torch.cumprod(values, dim=0)  # Cumulative product to get the Kaplan-Meier estimator

        # Keep track of the unique times and Kaplan-Meier estimator values
        self.time = uniq_times
        self.km_est = y

    def plot_km(self, ax=None, **kwargs):
        """Plot the Kaplan-Meier estimate of the survival distribution.

        Args:
            ax (matplotlib.axes.Axes, optional):
                The axes to plot the Kaplan-Meier estimate.
                If None, a new figure and axes are created.
                Defaults to None.
            **kwargs:
                Additional keyword arguments to pass to the plot function.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 32
            >>> time = torch.randint(low=0, high=8, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> km = KaplanMeierEstimator()
            >>> km(event, time)
            >>> km.plot_km()


        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        if ax is None:
            _, ax = plt.subplots()

        ax.step(self.time.cpu(), self.km_est.cpu(), where="post", **kwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier Estimate")

    def predict(self, new_time: torch.Tensor) -> torch.Tensor:
        """Predicts the Kaplan-Meier estimate on new time points.
        If the new time points do not match any times used to fit, the left-limit is used.

        Args:
            new_time (torch.tensor):
                New time points at which to predict the Kaplan-Meier estimate.

        Returns:
            Kaplan-Meier estimate evaluated at ``new_time``.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 8
            >>> time = torch.randint(low=1, high=10, size=(n * 4,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n * 4,)).bool()
            >>> km = KaplanMeierEstimator()
            >>> km(event, time)
            >>> km.predict(torch.randint(low=0, high=10, size=(n,)))  # predict survival distribution
            tensor([1.0000, 0.9062, 0.8700, 1.0000, 0.9062, 0.9062, 0.4386, 0.0000])

        """
        # Ensure new_time is on the correct device
        new_time = new_time.to(self.device)

        # Prepend baseline survival probability of 1 (at time 0 or before any event)
        # and extend the reference time vector with -∞ to align indexing
        ref_time = torch.cat((-torch.tensor([torch.inf], device=self.device, dtype=self.time.dtype), self.time), dim=0)
        km_est_ = torch.cat((torch.ones(1, device=self.device, dtype=self.km_est.dtype), self.km_est))

        # Identify time points in new_time that extend beyond the last train time in training
        extends = new_time > torch.max(ref_time)

        # Initialize predicted survival probabilities
        km_pred = torch.zeros_like(new_time, dtype=km_est_.dtype, device=self.device)

        # For times beyond the train range, use the survival probability at the last train time
        # (probability "holds" after the last event)
        km_pred[extends] = km_est_[torch.argmax(ref_time)]

        # For all other new_time points, locate their position relative to train times
        idx = torch.searchsorted(ref_time, new_time[~extends], side="left")

        # If new_time does not exactly match an observed time, adjust index to take left limit
        eps = torch.finfo(ref_time.dtype).eps
        idx[torch.abs(ref_time[idx] - new_time[~extends]) >= eps] -= 1

        # Assign survival estimates for in-range new_time points
        km_pred[~extends] = km_est_[idx]

        return km_pred

    def get_survival_table(self):
        """Prints the survival table with the unique times and Kaplan-Meier estimates.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 32
            >>> time = torch.randint(low=0, high=8, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> s = KaplanMeierEstimator()
        """
        # Print header
        print("Time\tSurvival")
        print("-" * 16)

        # Print unique times and Kaplan-Meier estimates
        for t, y in zip(self.time, self.km_est):
            print(f"{t:.2f}\t{y:.4f}")

        return pd.DataFrame({
                "Time": self.time.detach().cpu().numpy(),
                "Survival": self.km_est.detach().cpu().numpy()
            })

    def _compute_counts(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the counts of events, censorings and risk set at ``time``.

        Returns: Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]
            unique times
            number of events at unique times
            number at risk at unique times
            number of censored at unique times
        """
        # Get the number of samples
        n_samples = len(self.event)

        # Sort indices based on time
        order = torch.argsort(self.time, dim=0)

        # Initialize arrays to store unique times, event counts, and total counts
        uniq_times = torch.empty_like(self.time, device=self.device)
        uniq_events = torch.empty_like(self.time, dtype=torch.long, device=self.device)
        uniq_counts = torch.empty_like(self.time, dtype=torch.long, device=self.device)

        # Group indices by unique time values
        groups = itertools.groupby(range(len(self.time)), key=lambda i: self.time[order[i]])

        # Initialize index for storing unique values
        j = 0

        # Iterate through unique times
        for _, group_indices in groups:
            group_indices = list(group_indices)

            # Count events and total occurrences
            count_event = sum(self.event[order[i]].item() for i in group_indices)
            count = len(group_indices)

            # Store unique time, event count, and total count
            uniq_times[j] = self.time[order[group_indices[0]]].item()
            uniq_events[j] = count_event
            uniq_counts[j] = count
            j += 1

        # Extract valid values based on the index
        times = uniq_times[:j]
        n_events = uniq_events[:j]
        total_count = uniq_counts[:j]
        n_censored = total_count - n_events

        # Offset cumulative sum by one to get the number at risk
        n_at_risk = n_samples - torch.cumsum(
            torch.cat([torch.tensor([0], device=self.device, dtype=total_count.dtype), total_count], dim=0), dim=0
        )

        return times, n_events, n_at_risk[:-1], n_censored


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
