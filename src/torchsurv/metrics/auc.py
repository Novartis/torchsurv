import copy
import sys
from typing import Optional

import torch
from scipy import stats
from torchmetrics import regression

from torchsurv.stats import kaplan_meier
from torchsurv.tools.validate_data import (
    validate_log_shape,
    validate_new_time,
    validate_survival_data,
)

__all__ = ["Auc"]


class Auc:
    """Area Under the Curve class for survival models."""

    def __init__(self, checks: bool = True, tied_tol: float = 1e-8):
        """Initialize an Auc for survival class model evaluation.

        Args:
            tied_tol (float):
                Tolerance for tied risk scores.
                Defaults to 1e-8.
            checks (bool):
                Whether to perform input format checks.
                Enabling checks can help catch potential issues in the input data.
                Defaults to True.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> auc = Auc()
            >>> auc(estimate, event, time) # default: auc cumulative/dynamic
            tensor([0.7500, 0.4286, 0.3333])
            >>> auc.integral()
            tensor(0.5040)
            >>> auc.confidence_interval() # default: Blanche, two_sided
            tensor([[0.4213, 0.0000, 0.0000],
                    [1.0000, 0.9358, 0.7289]])
            >>> auc.p_value()
            tensor([0.1360, 0.7826, 0.4089])
        """
        self.tied_tol = tied_tol
        self.checks = checks

        # init instate variables
        self.order_time = None
        self.time = None
        self.event = None
        self.weight = None
        self.new_time = None
        self.weight_new_time = None
        self.estimate = None
        self.is_case = None
        self.is_control = None
        self.auc_type = None
        self.auc = None

    def __call__(
        self,
        estimate: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        auc_type: str = "cumulative",
        weight: Optional[torch.Tensor] = None,
        new_time: Optional[torch.Tensor] = None,
        weight_new_time: Optional[torch.Tensor] = None,
        instate: bool = True,
    ) -> torch.Tensor:
        r"""Compute the time-dependent Area Under the Receiver Operating Characteristic Curve (AUC).

        The AUC at time :math:`t` is the probability that a model correctly predicts
        which of two comparable samples will experience an event by time :math:`t` based on their
        estimated risk scores.
        The AUC is particularly useful for evaluating time-dependent predictions (e.g., 10-year mortality).
        It is recommended to use AUC instead of the concordance index for such time-dependent predictions, as
        AUC is proper in this context, while the concordance index is not :cite:p:`Blanche2018`.

        Args:
            estimate (torch.Tensor):
                Estimated risk of event occurrence (i.e., risk score).
                Can be of shape = (n_samples,) if subject-specific risk score is time-independent,
                of shape = (n_samples, n_samples) if subject-specific risk score is evaluated at ``time``,
                or of shape = (n_samples, n_times) if subject-specific risk score is evaluated at ``new_time``.
            event (torch.Tensor, boolean):
                Event indicator of size n_samples (= True if event occurred).
            time (torch.Tensor, float):
                Time-to-event or censoring of size n_samples.
            auc_type (str, optional):
                AUC type. Defaults to "cumulative".
                Must be one of the following: "cumulative" for cumulative/dynamic, "incident" for incident/dynamic.
            weight (torch.Tensor, optional):
                Optional sample weight evaluated at ``time`` of size n_samples.
                Defaults to 1.
            new_time (torch.Tensor, optional):
                Time points at which to evaluate the AUC of size n_times. Values must be within the range of follow-up in ``time``.
                Defaults to the event times excluding maximum (because number of controls for t > max(time) is 0).
            weight_new_time (torch.Tensor):
                Optional sample weight evaluated at ``new_time`` of size n_times.
                Defaults to 1.

        Returns:
            torch.Tensor: AUC evaluated at ``new_time``.

        Note:
            The function evaluates either the cumulative/dynamic (C/D) or the incident/dynamic (I/D) AUC (argument ``auc_type``)
            at time :math:`t \in \{t_1, \cdots, t_K\}` (argument ``new_time``).

            For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
            censoring time. Survival data consist of the event indicator, :math:`\delta_i=(X_i\leq D_i)`
            (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
            (argument ``time``).

            The risk score measures the risk (or a proxy thereof) that a subject has an event.
            The function accepts time-dependent risk score or time-independent risk score. The time-dependent risk score
            of subject :math:`i` is specified through a function :math:`q_i: [0, \infty) \rightarrow \mathbb{R}`.
            The time-independent risk score of subject :math:`i` is specified by a constant :math:`q_i`.
            The argument ``estimate`` is the estimated risk score.
            For time-dependent risk score: if ``new_time`` is specified, the argument ``estimate`` should be of shape = (N,K)
            (:math:`(i,k)` th element is :math:`\hat{q}_i(t_k)`);
            if ``new_time`` is not specified, the argument ``estimate`` should be of shape = (N,N)
            (:math:`(i,j)` th element is :math:`\hat{q}_i(T_j)`) .
            For time-independent risk score, the argument ``estimate`` should be of length
            N (:math:`i` th element is :math:`\hat{q}_i`).

            The AUC C/D and AUC I/D evaluated at time :math:`t` are defined by

            .. math::
                \text{AUC}^{C/D}(t) = p(q_i(t) > q_j(t) \: | \: X_i \leq t, X_j > t) \\
                \text{AUC}^{I/D}(t) = p(q_i(t) > q_j(t) \: | \: X_i = t, X_j > t).


            The default estimators of the AUC C/D and AUC I/D at time :math:`t` :cite:p:`Blanche2013` returned by the function are

            .. math::

                \hat{\text{AUC}}^{C/D}(t) = \frac{\sum_i \sum_j \delta_i \: I(T_i \leq t, T_j > t) I(\hat{q}_i(t) > \hat{q}_j(t))}{\sum_i \delta_i \: I(T_i \leq t) \sum_j I(T_j > t)} \\
                \hat{\text{AUC}}^{I/D}(t) = \frac{\sum_i \sum_j \delta_i \: I(T_i = t, T_j > t) I(\hat{q}_i(t) > \hat{q}_j(t))}{\sum_i \delta_i \: I(T_i = t) \sum_j I(T_j > t)}.

            These estimators are considered naive because, when the event times are censored, all subjects censored
            before time point :math:`t` are ignored. Additionally, the naive estimators
            converge to an association measure that involves the censoring distribution.
            To address this shortcoming, :cite:t:`Uno2007` proposed to employ the
            inverse probability weighting technique. In this context, each subject included at time
            :math:`t` is weighted by the inverse probability of censoring :math:`\omega(t) = 1 / \hat{D}(t)`, where
            :math:`\hat{D}(t)` is the Kaplan-Meier estimate of the censoring distribution, :math:`P(D>t)`.
            The censoring-adjusted AUC C/D estimate at time :math:`t` is

            .. math::

                \hat{\text{AUC}}^{C/D}(t) = \frac{\sum_i \sum_j \delta_i \: \omega(T_i) \: I(T_i \leq t, T_j > t) I(\hat{q}_i(t) > \hat{q}_j(t))}{\sum_i \delta_i \: \omega(T_i) \: I(T_i \leq t)  \sum_j I(T_j > t)} \\

            Note that the censoring-adjusted AUC I/D estimate is the same as the "naive" estimate because the weights are all equal to :math:`\omega(t)`.

            The censoring-adjusted AUC C/D estimate can be obtained by specifying the argument
            ``weight``, the weights evaluated at each ``time`` (:math:`\omega(T_1), \cdots, \omega(T_N)`).
            If ``new_time`` is specified, the argument  ``weight_new_time``
            should also be specified accordingly, the weights evaluated at each ``new_time``
            (:math:`\omega(t_1), \cdots, \omega(t_K)`). The latter is required to compute the standard error of the AUC.
            In the context of train/test split, the weights should be derived from the censoring distribution estimated in the training data.
            Specifically, the censoring distribution is estimated using the training set and then evaluated at the subject time within the test set.

        Examples:
            >>> from torchsurv.stats.ipcw import get_ipcw
            >>> _ = torch.manual_seed(42)
            >>> n = 20
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> auc = Auc()
            >>> auc(estimate, event, time) # default: naive auc c/d
            tensor([0.9474, 0.5556, 0.5294, 0.6429, 0.5846, 0.6389, 0.5844, 0.5139, 0.4028,
                    0.5400, 0.4545, 0.7500])
            >>> auc(estimate, event, time, auc_type = "incident") # naive auc i/d
            tensor([0.9474, 0.1667, 0.4706, 0.9286, 0.3846, 0.8333, 0.3636, 0.2222, 0.0000,
                    0.8000, 0.5000, 1.0000])
            >>> ipcw = get_ipcw(event, time) # ipcw weight at time
            >>> auc(estimate, event, time, weight = ipcw) # Uno's auc c/d
            tensor([0.9474, 0.5556, 0.5294, 0.6521, 0.5881, 0.6441, 0.5865, 0.5099, 0.3929,
                    0.5422, 0.4534, 0.7996])
            >>> new_time = torch.unique(torch.randint(low=100, high=150, size=(n,)).float()) # new time at which to evaluate auc
            >>> ipcw_new_time = get_ipcw(event, time, new_time) # ipcw at new_time
            >>> auc(estimate, event, time, new_time = new_time, weight = ipcw, weight_new_time = ipcw_new_time) # Uno's auc c/d at new_time
            tensor([0.5333, 0.5333, 0.5333, 0.5333, 0.6521, 0.6521, 0.5881, 0.5881, 0.5865,
                    0.5865, 0.5865, 0.5865, 0.5865, 0.6018, 0.6018, 0.5099])

        References:

            .. bibliography::
                :filter: False

                Blanche2018
                Blanche2013
                Uno2007
        """

        # mandatory input format checks
        self._validate_auc_inputs(
            estimate, time, auc_type, new_time, weight, weight_new_time
        )

        # update inputs as required
        estimate, new_time, weight, weight_new_time = self._update_auc_new_time(
            estimate, event, time, new_time, weight, weight_new_time
        )
        estimate = self._update_auc_estimate(estimate, new_time)
        weight, weight_new_time = self._update_auc_weight(
            time, new_time, weight, weight_new_time
        )

        # further input format checks
        if self.checks:
            validate_survival_data(event, time)
            validate_new_time(new_time, time)
            validate_log_shape(estimate)

        # sample size and length of time
        n_samples, n_times = estimate.shape[0], new_time.shape[0]

        # Expand arrays to (n_samples, n_times) shape
        time_long = time.unsqueeze(1).expand((n_samples, n_times))
        event_long = event.unsqueeze(1).expand((n_samples, n_times))
        weight_long = weight.unsqueeze(1).expand((n_samples, n_times))
        new_time_long = new_time.unsqueeze(0).expand(
            n_samples, n_times
        )  # Size n_times, unsqueeze(0) instead

        # sort each time point (columns) by risk score (descending)
        index = torch.argsort(-estimate, dim=0)
        time_long, event_long, estimate, weight_long = map(
            lambda x: torch.gather(x, 0, index),
            [time_long, event_long, estimate, weight_long],
        )

        # find case and control
        if auc_type == "incident":
            is_case = (time_long == new_time_long) & event_long
        elif auc_type == "cumulative":
            is_case = (time_long <= new_time_long) & event_long
        is_control = time_long > new_time_long
        n_controls = is_control.sum(axis=0)

        # estimate censoring adjusted true positive rate and false positive rate
        cumsum_tp = torch.cumsum(is_case * weight_long, axis=0)
        cumsum_fp = torch.cumsum(is_control, axis=0)
        true_pos = cumsum_tp / cumsum_tp[-1]
        false_pos = cumsum_fp / n_controls

        # prepend row of infinity values
        estimate_diff = torch.cat(
            (torch.full((1, n_times), float("inf")), estimate), dim=0
        )
        is_tied = torch.abs(torch.diff(estimate_diff, dim=0)) <= self.tied_tol

        # initialize empty tensor to store auc
        auc = torch.zeros_like(new_time)

        # iterate over time
        for i in range(n_times):
            # Extract relevant columns for the current time point
            tp, fp, mask = true_pos[:, i], false_pos[:, i], is_tied[:, i]

            # Create a boolean mask with False at the indices where ties occur
            mask_no_ties = torch.ones(tp.numel(), dtype=torch.bool)
            mask_no_ties[torch.nonzero(mask) - 1] = False

            # Extract values without ties using the boolean mask
            tp_no_ties = tp[mask_no_ties]
            fp_no_ties = fp[mask_no_ties]

            # Add an extra threshold position
            # to make sure that the curve starts at (0, 0)
            tp_no_ties = torch.cat([torch.tensor([0]), tp_no_ties])
            fp_no_ties = torch.cat([torch.tensor([0]), fp_no_ties])

            # Calculate AUC using trapezoidal rule for numerical integration
            auc[i] = torch.trapz(tp_no_ties, fp_no_ties)

        # Create/overwrite internal attributes states
        if instate:
            index_rev = torch.argsort(index, dim=0)
            estimate, is_case, is_control = map(
                lambda x: torch.gather(x, 0, index_rev),
                [estimate, is_case, is_control],
            )

            # sort all objects by time
            self.order_time = torch.argsort(time, dim=0)
            self.time = time[self.order_time]
            self.event = event[self.order_time]
            self.weight = weight[self.order_time]
            self.new_time = new_time
            self.weight_new_time = weight_new_time
            self.estimate = torch.index_select(estimate, 0, self.order_time)
            self.is_case = torch.index_select(is_case, 0, self.order_time)
            self.is_control = torch.index_select(is_control, 0, self.order_time)
            self.auc_type = auc_type
            self.auc = auc

        return auc

    def integral(self, tmax: Optional[torch.Tensor] = None):
        """Compute the integral of the time-dependent AUC.

        Args:
            tmax (torch.Tensor, optional):
                A number specifying the upper limit of the time range to compute the AUC integral.
                Defaults to ``new_time[-1]`` for cumulative/dynamic AUC and ``new_time[-1]-1`` for incident/dynamic AUC.

        Returns:
            torch.Tensor: Integral of AUC over [0-``tmax``].

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> auc = Auc()
            >>> auc(estimate, event, time, auc_type = "incident")
            tensor([0.7500, 0.1429, 0.1667])
            >>> auc.integral() # integral of the auc incident/dynamic
            tensor(0.4667)

        Notes:
            In case ``auc_type`` = "cumulative" (cumulative/dynamic AUC), the values of AUC are weighted by
            the estimated event density. In case ``auc_type`` = "incident"
            (incident/dynamic AUC), the values of AUC are weighted by 2 times the product of the estimated
            event density and the estimated survival function :cite:p:`Heagerty2005`.
            The estimated survival
            function is the Kaplan-Meier estimate. The estimated event density
            is obtained from the discrete incremental changes of the estimated survival function.

        References:

            .. bibliography::
                :filter: False

                Heagerty2005
        """
        # Only one time step available
        if len(self.new_time) == 1:
            auc = self.auc[0]
        else:
            # handle cases where tmax is not specified
            if tmax is None:
                tmax = (
                    self.new_time[-1]
                    if self.auc_type == "cumulative"
                    else self.new_time[-1] - 1
                )

            # estimate of survival distribution
            km = kaplan_meier.KaplanMeierEstimator()
            km(self.event, self.time)
            survival = km.predict(self.new_time)

            # integral of auc
            if self.auc_type == "cumulative":
                auc = self._integrate_cumulative(survival, tmax)
            else:
                auc = self._integrate_incident(survival, tmax)
        return auc

    def confidence_interval(
        self,
        method: str = "blanche",
        alpha: float = 0.05,
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
    ) -> torch.Tensor:
        """Compute the confidence interval of the AUC.

        This function calculates either the pointwise confidence interval or the bootstrap
        confidence interval for the AUC. The pointwise confidence interval is computed
        assuming that the AUC is normally distributed and using the standard error estimated with
        :cite:t:`Blanche2013b` method. The bootstrap confidence interval is constructed based on the
        distribution of bootstrap samples.

        Args:
            method (str):
                Method for computing confidence interval. Defaults to "blanche".
                Must be one of the following: "blanche", "bootstrap".
            alpha (float):
                Significance level. Defaults to 0.05.
            alternative (str):
                Alternative hypothesis. Defaults to "two_sided".
                Must be one of the following: "two_sided", "greater", "less".
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor([lower,upper]):
                Lower and upper bounds of the confidence interval.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> auc = Auc()
            >>> auc(estimate, event, time)
            tensor([0.7500, 0.4286, 0.3333])
            >>> auc.confidence_interval() # Default: Blanche, two_sided
            tensor([[0.4213, 0.0000, 0.0000],
                    [1.0000, 0.9358, 0.7289]])
            >>> auc.confidence_interval(method = "bootstrap", alternative = "greater")
            tensor([[0.3750, 0.1667, 0.0833],
                    [1.0000, 1.0000, 1.0000]])

        References:

            .. bibliography::
                :filter: False

                Blanche2013b
        """

        assert (
            hasattr(self, "auc") and self.auc is not None
        ), "Error: Please calculate AUC using `Auc()` before calling `confidence_interval()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                "'alternative' parameter must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "blanche":
            conf_int = self._confidence_interval_blanche(alpha, alternative)
        elif method == "bootstrap":
            conf_int = self._confidence_interval_bootstrap(
                alpha, alternative, n_bootstraps
            )
        else:
            raise ValueError(
                "Method not implemented. Please choose either 'blanche' or 'bootstrap'."
            )
        return conf_int

    def p_value(
        self,
        method: str = "blanche",
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
    ) -> torch.Tensor:
        """Perform a one-sample hypothesis test on the AUC.

        This function calculates either the pointwise p-value or the bootstrap p-value
        for testing the null hypothesis that the AUC is equal to 0.5.
        The pointwise p-value is computed assuming that the AUC is normally distributed
        and using the standard error estimated using :cite:t:`Blanche2013b` method.
        The bootstrap p-value is derived by permuting risk predictions to estimate
        the sampling distribution under the null hypothesis.

        Args:
            method (str):
                Method for computing p-value. Defaults to "blanche".
                Must be one of the following: "blanche", "bootstrap".
            alternative (str):
                Alternative hypothesis. Defaults to "two_sided".
                Must be one of the following: "two_sided" (AUC is not equal to 0.5),
                "greater" (AUC is greater than 0.5), "less" (AUC is less than 0.5).
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor: p-value of the statistical test.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> auc = Auc()
            >>> auc(estimate, event, time)
            tensor([0.7500, 0.4286, 0.3333])
            >>> auc.p_value() # Default: Blanche, two_sided
            tensor([0.1360, 0.7826, 0.4089])
            >>> auc.p_value(method = "bootstrap", alternative = "greater")
            tensor([0.2400, 0.5800, 0.7380])

        """

        assert (
            hasattr(self, "auc") and self.auc is not None
        ), "Error: Please calculate AUC using `Auc()` before calling `p_value()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                "'alternative' parameter must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "blanche":
            pvalue = self._p_value_blanche(alternative)
        elif method == "bootstrap":
            pvalue = self._p_value_bootstrap(alternative, n_bootstraps)
        else:
            raise ValueError(
                "Method not implemented. Please choose either 'blanche' or 'bootstrap'."
            )
        return pvalue

    def compare(
        self, other, method: str = "blanche", n_bootstraps: int = 999
    ) -> torch.Tensor:
        """Compare two AUCs.

        This function compares two AUCs computed on the same data with different
        risk predictions. The statistical hypotheses are
        formulated as follows, null hypothesis: auc1 = auc2 and alternative
        hypothesis: auc1 > auc2.
        The statistical test is either a Student t-test for dependent samples or
        a two-sample bootstrap test. The Student t-test assumes that the AUC is normally distributed
        and uses the standard errors estimated with :cite:t:`Blanche2013b` method.

        Args:
            other (Auc):
                Another instance of the Auc class representing auc2.
            method (str):
                Statistical test used to perform the hypothesis test. Defaults to "blanche".
                Must be one of the following: "blanche", "bootstrap".
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor: p-value of the statistical test.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> auc1 = Auc()
            >>> auc1(torch.randn((n,)), event, time)
            tensor([0.7500, 0.4286, 0.3333])
            >>> auc2 = Auc()
            >>> auc2(torch.randn((n,)), event, time)
            tensor([0.0000, 0.1429, 0.0556])
            >>> auc1.compare(auc2) # default: Blanche
            tensor([0.0008, 0.2007, 0.1358])
            >>> auc1.compare(auc2, method = "bootstrap")
            tensor([0.0220, 0.1970, 0.1650])

        """

        assert (
            hasattr(self, "auc") and self.auc is not None
        ), "Error: Please calculate AUC using `Auc()` before calling `compare()`."

        # assert that the same data were used to compute the two auc
        if torch.any(self.event != other.event) or torch.any(self.time != other.time):
            raise ValueError(
                "Mismatched survival data: 'time' and 'event' should be the same for both AUC computations."
            )
        if torch.any(self.new_time != other.new_time):
            raise ValueError(
                "Mismatched evaluation times: 'new_time' should be the same for both AUC computations."
            )
        if self.auc_type != other.auc_type:
            raise ValueError(
                "Mismatched AUC types: 'auc_type' should be the same for both AUC computations."
            )

        if method == "blanche":
            pvalue = self._compare_blanche(other)
        elif method == "bootstrap":
            pvalue = self._compare_bootstrap(other, n_bootstraps)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'blanche' or 'bootstrap'."
            )
        return pvalue

    # pylint: disable=invalid-name
    def _integrate_incident(self, S: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
        """Integrates the incident/dynamic AUC, int_t AUC(t) x w(t) dt
        where w(t) = 2*f(t)*S(t) and f(t) is the lifeline distribution,
        S(t) is the survival distribution estimated with the Kaplan
        Meier estimator.
        """

        # Find the index corresponding to tmax in times
        tmax_index = torch.sum(self.new_time <= tmax)

        # Initialize an array to store the density function f(t)
        f = torch.zeros_like(S)

        # Compute the density function f(t)
        f[0] = 1.0 - S[0]
        for i in range(1, len(S)):
            f[i] = S[i - 1] - S[i]

        # Initialize a variable to accumulate the weighted sum of f(t)*S(t)
        wT = 0.0

        # Accumulate the weighted sum up to maxI
        for i in range(tmax_index):
            wT += 2.0 * f[i] * S[i]

        # Initialize the integrated AUC
        i_auc = 0

        # Calculate the integrated AUC using the weight
        for i in range(tmax_index):
            if wT != 0.0:
                if f[i] > torch.finfo(torch.double).eps:
                    i_auc += self.auc[i] * (2.0 * f[i] * S[i]) / wT

        return i_auc

    # pylint: disable=invalid-name
    def _integrate_cumulative(
        self, S: torch.Tensor, tmax: torch.Tensor
    ) -> torch.Tensor:
        """Integrates the cumulative/dynamic AUC, int_t AUC(t) · f(t) dt
        where f(t) is the lifeline distribution estimated from the discrete
        incremental changes of the Kalpan-Meier estimate of the survival function.
        """

        # Find the index corresponding to tmax in times

        tmax_index = torch.sum(self.new_time <= tmax)

        # Initialize an array to store the density function f(t)
        f = torch.zeros_like(S)

        # Compute the density function f(t)
        f[0] = 1.0 - S[0]
        for i in range(1, len(S)):
            f[i] = S[i - 1] - S[i]

        # Initialize a variable to accumulate the weighted sum of f(t)
        wT = 0.0

        # Accumulate the weighted sum up to maxI
        for i in range(tmax_index):
            if f[i] > torch.finfo(torch.double).eps:
                wT += f[i]

        # Initialize the integrated AUC
        i_auc = 0

        # Calculate the integrated AUC using the weight
        for i in range(tmax_index):
            if wT != 0.0:
                if f[i] > torch.finfo(torch.double).eps:
                    i_auc += self.auc[i] * (f[i]) / wT

        return i_auc

    def _confidence_interval_blanche(
        self, alpha: float, alternative: str
    ) -> torch.Tensor:
        """Confidence interval of AUC assuming that the AUC
        is normally distributed and using
        standard error estimated with Blanche et al.'s method."""

        alpha = alpha / 2 if alternative == "two_sided" else alpha

        auc_se = self._auc_se()

        if torch.all(auc_se) >= 0:
            ci = (
                -torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(alpha))
                * auc_se
            )
            lower = torch.max(torch.tensor(0.0), self.auc - ci)
            upper = torch.min(torch.tensor(1.0), self.auc + ci)

            if alternative == "less":
                lower = torch.zeros_like(lower)
            elif alternative == "greater":
                upper = torch.ones_like(upper)
        else:
            raise ValueError("The standard error of AUC must be a positive value.")

        return torch.stack([lower, upper], dim=0).squeeze()

    def _confidence_interval_bootstrap(
        self, alpha: float, alternative: str, n_bootstraps: int
    ) -> torch.Tensor:
        """Bootstrap confidence interval of the AUC using Efron percentile method.

        References:
            Efron, Bradley; Tibshirani, Robert J. (1993).
                An introduction to the bootstrap, New York: Chapman & Hall, software.
        """

        # auc bootstraps given bootstrap distribution
        auc_bootstrap = self._bootstrap_auc(
            metric="confidence_interval", n_bootstraps=n_bootstraps
        )

        # initialize tensor to store confidence intervals
        lower = torch.zeros_like(self.auc)
        upper = torch.zeros_like(self.auc)

        # iterate over time
        for index_t in range(len(self.auc)):
            # obtain confidence interval
            if alternative == "two_sided":
                lower[index_t], upper[index_t] = torch.quantile(
                    auc_bootstrap[:, index_t],
                    torch.tensor([alpha / 2, 1 - alpha / 2], device=self.auc.device),
                )
            elif alternative == "less":
                upper[index_t] = torch.quantile(
                    auc_bootstrap[:, index_t],
                    torch.tensor(1 - alpha, device=self.auc.device),
                )
                lower[index_t] = torch.tensor(0.0, device=self.auc.device)
            elif alternative == "greater":
                lower[index_t] = torch.quantile(
                    auc_bootstrap[:, index_t],
                    torch.tensor(alpha, device=self.auc.device),
                )
                upper[index_t] = torch.tensor(1.0, device=self.auc.device)

        return torch.stack([lower, upper], dim=0).squeeze()

    def _p_value_blanche(
        self, alternative: str, null_value: float = 0.5
    ) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the AUC
        assuming that the AUC is normally distributed and using standard error estimated
        with Blanche et al's method.
        """

        auc_se = self._auc_se()

        # get p-value
        if torch.all(auc_se >= 0.0):
            p = torch.distributions.normal.Normal(0, 1).cdf(
                (self.auc - null_value) / auc_se
            )
            if alternative == "two_sided":
                mask = self.auc >= torch.tensor(0.5)
                p[mask] = torch.tensor(1.0) - p[mask]
                p *= torch.tensor(2.0)
            elif alternative == "greater":
                p = torch.tensor(1.0) - p
        else:
            raise ValueError("The standard error of AUC must be a positive value.")

        return p

    def _p_value_bootstrap(self, alternative, n_bootstraps) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the AUC using
        permutation of risk prediction to estimate sampling distribution under the null
        hypothesis.
        """

        # auc bootstraps given null distribution auc = 0.5
        auc0 = self._bootstrap_auc(metric="p_value", n_bootstraps=n_bootstraps)

        # initialize empty tensor to store p-values
        p_values = torch.zeros_like(self.auc)

        # iterate over time
        for index_t, _ in enumerate(self.auc):
            # Derive p-value
            p = (
                torch.tensor(1.0) + torch.sum(auc0[:, index_t] <= self.auc[index_t])
            ) / torch.tensor(n_bootstraps + 1.0)
            if alternative == "two_sided":
                if self.auc[index_t] >= torch.tensor(0.5):
                    p = torch.tensor(1.0) - p
                p *= torch.tensor(2.0)
                p = torch.min(
                    torch.tensor(1.0, device=self.auc.device), p
                )  # in case very small bootstrap sample size is used
            elif alternative == "greater":
                p = torch.tensor(1.0) - p

            p_values[index_t] = p

        return p_values

    def _compare_blanche(self, other) -> torch.Tensor:
        """Student t-test for dependent samples given Blanche's standard error to
        compare two AUCs.
        """

        # sample size
        N = self.estimate.shape[0]

        # compute noether standard error
        auc1_se = self._auc_se()
        # pylint: disable=protected-access
        auc2_se = other._auc_se()

        # initialize empty vector to store p_values
        p_values = torch.zeros_like(self.auc)

        # iterate over time
        for index_t, _ in enumerate(self.auc):
            # compute spearman correlation between risk prediction
            corr = regression.SpearmanCorrCoef()(
                self.estimate[:, index_t], other.estimate[:, index_t]
            )

            # check for perfect positive monotonic relationship between two variables
            if 1 - torch.abs(corr) < 1e-15:
                p_values[index_t] = 1.0
            else:
                # compute t-stat
                t_stat = (self.auc[index_t] - other.auc[index_t]) / torch.sqrt(
                    auc1_se[index_t] ** 2
                    + auc2_se[index_t] ** 2
                    - 2 * corr * auc1_se[index_t] * auc2_se[index_t]
                )

                # p-value
                p_values[index_t] = torch.tensor(
                    1
                    - stats.t.cdf(
                        t_stat, df=N - 1
                    ),  # student-t cdf not available on torch
                    dtype=self.auc.dtype,
                    device=self.auc.device,
                )

        return p_values

    def _compare_bootstrap(self, other, n_bootstraps) -> torch.Tensor:
        """Bootstrap two-sample test to compare two AUCs."""

        # auc bootstraps given null hypothesis that auc1 and
        # auc2 come from the same distribution
        auc1_null = self._bootstrap_auc(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )
        auc2_null = self._bootstrap_auc(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )

        # Bootstrapped test statistics
        t_boot = auc1_null - auc2_null

        # observed test statistics
        t_obs = self.auc - other.auc

        # initialize empty tensor to store p-values
        p_values = torch.zeros_like(self.auc)

        # iterate over time
        for index_t in range(len(self.auc)):
            p_values[index_t] = 1 - (
                1 + torch.sum(t_boot[:, index_t] <= t_obs[index_t])
            ) / (n_bootstraps + 1)

        return p_values

    def _auc_se(self) -> torch.Tensor:
        """AUC's standard error estimated using Blanche et al's method."""

        # sample size and length of time
        n_samples, n_times = self.estimate.shape[0], self.new_time.shape[0]

        # survival distribution estimated with KM
        km = kaplan_meier.KaplanMeierEstimator()
        km(self.event, self.time)
        S = km.predict(self.new_time)

        # Expand arrays to (n_samples, n_times) shape
        time_long = torch.unsqueeze(self.time, dim=1).expand((n_samples, n_times))
        new_time_long = self.new_time.unsqueeze(0).expand(n_samples, n_times)
        survival_long = S.unsqueeze(0).expand(n_samples, n_times)

        # element (i, t) = I(T_i > t) / S(t)
        ipsw = (time_long >= new_time_long) / survival_long

        # element (i,k) = int_0^{T_i} M_Ck(t) / S dt, where M_Ck(t) = I(delta_k = 0, T_k <= t) - int_0^t dLambda_c(u)
        integral_m_div_s = self._integral_censoring_martingale_divided_survival()

        # element (i, t) = f_i1t = I(T_i <= t, delta_i = 1) * W(T_i) (if incident, condition for case is T_i = t)
        f = self.is_case * self.weight.unsqueeze(1).expand(-1, n_times)

        # element (t) = F_1t = 1/n * sum_i f_i1t
        F = 1 / n_samples * torch.sum(f, dim=0)

        # initialize empty tensor to store standard error of auc
        auc_se = torch.zeros_like(self.new_time)

        # iterate over time
        for index_t in range(n_times):
            # element(i,j) given t = h*_tij = I(T_i <= t, delta_i = 1)  * I(U_j > t) * (I(M_i > M_j) + 0.5 * I(M_i > M_j)) * W(T_i) * W(t)
            h_t = (
                self.is_case[:, index_t].unsqueeze(1) * self.is_control[:, index_t]
            ).float()
            h_t *= (
                self.estimate[:, index_t].unsqueeze(1)
                > self.estimate[:, index_t].unsqueeze(0)
            ).float() + 0.5 * (
                self.estimate[:, index_t].unsqueeze(1)
                == self.estimate[:, index_t].unsqueeze(0)
            ).float()
            h_t *= self.weight.unsqueeze(1)
            h_t *= self.weight_new_time[index_t]

            # h*_t = 1 / n^2 * sum_i sum_j h_tij
            H_t = (1 / (n_samples**2)) * torch.sum(h_t)

            # phi*(t), eq.bottom page 3 of Supplementary Material of Blanche et al. (2013)
            phi = self._phi_blanche(
                h_t,
                H_t,
                f[:, index_t],
                F[index_t],
                S[index_t],
                integral_m_div_s,
                ipsw[:, index_t],
                n_samples,
            )

            # sum of phi
            sum_phi = (
                torch.sum(
                    phi, axis=(1, 2)
                )  # sum_j sum_k phi_ijk = Les_sum_jk_a_i_fixe * n_samples in Blanche's code
                + torch.sum(
                    phi, axis=(0, 2)
                )  # sum_i sum_k phi_ijk = Les_sum_ik_a_j_fixe * n_samples in Blanche's code
                + torch.sum(
                    phi, axis=(0, 1)
                )  # sum_i sum_j phi_ijk = Les_sum_ij_a_k_fixe * n_samples in Blanche's code
            )

            # estimate of influence function
            hat_IF = 1 / (n_samples**2) * sum_phi

            # estimate of auc standard error
            auc_se[index_t] = torch.std(hat_IF) / (n_samples ** (1 / 2))

        return auc_se

    def _integral_censoring_martingale_divided_survival(self) -> torch.Tensor:
        """Compute the integral of the censoring martingale divided by the survival distribution."""

        # Number of samples
        n_samples = len(self.time)

        # check that time is ordered
        if torch.any(self.time != self.time[torch.argsort(self.time, dim=0)]):
            raise ValueError(
                "The 'time' values in `self.time` should be ordered in ascending order."
            )

        # find censoring events
        censoring = self.event == 0.0

        # Compute censoring hazard, denoted lambda_C in in Blanche et al's paper
        censoring_hazard = censoring / torch.arange(n_samples, 0, -1)

        # cumulative censoring hazard, denoted Lambda_C in Blanche et al's paper
        cumsum_censoring_hazard = torch.cumsum(censoring_hazard, dim=0)

        # censoring martingale, denoted M_C in in Blanche et al's paper
        censoring_martingale = torch.zeros((n_samples, n_samples), dtype=torch.float32)
        for i in range(n_samples):
            censoring_martingale[:, i] = censoring[i].float() * (
                self.time[i] <= self.time
            ).float() - torch.cat(
                (
                    cumsum_censoring_hazard[0:i],
                    torch.full((n_samples - i,), cumsum_censoring_hazard[i]),
                )
            )

        # derivative of censoring martingal, denoted d M_C in Blanche et al's paper
        d_censoring_martingale = torch.cat(
            (
                censoring_martingale[0, :].view(1, -1),
                censoring_martingale[1:, :] - censoring_martingale[:-1, :],
            )
        )

        def divide_by_empirical_survival(v):
            return v / torch.cat(
                (torch.tensor([1.0]), 1 - torch.arange(1, len(v)) / len(v))
            )

        d_censoring_martingale_div_s = torch.stack(
            [
                divide_by_empirical_survival(d_censoring_martingale[:, i])
                for i in range(n_samples)
            ],
            dim=1,
        )

        return torch.cumsum(d_censoring_martingale_div_s, dim=0)

    # pylint: disable=invalid-name
    @staticmethod
    def _phi_blanche(
        h_t: torch.Tensor,
        H_t: torch.Tensor,
        f_t: torch.Tensor,
        F_t: torch.Tensor,
        S_t: torch.Tensor,
        integral_m_div_s: torch.Tensor,
        ipsw_t: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Compute the three components of the influence function."""

        phi = torch.zeros(
            (n_samples, n_samples, n_samples),
            dtype=h_t.dtype,
            device=h_t.device,
        )
        for i in range(n_samples):
            _phi = 1 + integral_m_div_s[:, i]
            _phi *= f_t
            _phi -= F_t
            _phi /= F_t
            _phi += ipsw_t
            _phi *= -H_t
            _phi = _phi.unsqueeze(1).expand(-1, n_samples)
            _phi = _phi + (
                h_t * ((1 + integral_m_div_s[:, i]).unsqueeze(1).expand(-1, n_samples))
            )
            _phi /= S_t * F_t

            phi[:, :, i] = _phi

        return phi

    def _bootstrap_auc(
        self, metric: str, n_bootstraps: int, other=None
    ) -> torch.Tensor:
        """Compute bootstrap samples of the AUC.

        Args:
            metric (str): Must be one of the following: "p_value", "confidence_interval", "compare".
                If "p_value", computes bootstrap samples of the AUC given the sampling distribution
                under the null hypothesis (AUC = 0.5). If "confidence_interval", computes bootstrap
                samples of the AUC given the data distribution. If "compare", computes
                bootstrap samples of the AUC given the sampling distribution under the comparison test
                null hypothesis (auc1 = auc2).
            n_bootstraps (int): Number of bootstrap samples.
            other (optional, Auc):
                Another instance of the Auc class representing auc2.
                Only required for ``metric`` is equal to "compare".


        Returns:
            torch.Tensor: Bootstrap samples of AUC.
        """
        # Initiate empty list to store auc
        aucs = []

        # Get the bootstrap samples of auc
        for _ in range(n_bootstraps):
            if (
                metric == "p_value"
            ):  # bootstrap samples given null distribution (auc = 0.5)
                estimate = copy.deepcopy(self.estimate)
                estimate = estimate[
                    torch.randperm(estimate.shape[0]), :
                ]  # Shuffle estimate
                aucs.append(
                    self(
                        estimate,
                        self.event,
                        self.time,
                        self.auc_type,
                        self.weight,
                        self.new_time,
                        self.weight_new_time,
                        instate=False,
                    )
                )  # Run without saving internal state
            elif (
                metric == "confidence_interval"
            ):  # bootstrap samples given data distribution
                index = torch.randint(
                    low=0,
                    high=self.estimate.shape[0],
                    size=(self.estimate.shape[0] - 2,),
                )
                # auc is evaluated at new_time:
                # so need to keep the min and max time to ensure that new_time is within follow-up time
                index_max_time = torch.argmax(self.time).unsqueeze(0)
                index_min_time = torch.nonzero(
                    self.time == self.time[self.event].min()
                ).squeeze(1)
                index = torch.cat(
                    (
                        index,
                        index_max_time,
                        index_min_time,
                    ),
                    dim=0,
                )

                aucs.append(
                    self(
                        self.estimate[index, :],
                        self.event[index],
                        self.time[index],
                        self.auc_type,
                        self.weight[index],
                        self.new_time,
                        self.weight_new_time,
                        instate=False,
                    )
                )  # Run without saving internal state

            elif (
                metric == "compare"
            ):  # bootstrap samples given null distribution (auc1 = auc2)
                # index included in bootstrap samples
                index = torch.randint(
                    low=0,
                    high=self.estimate.shape[0] * 2,
                    size=(self.estimate.shape[0] - 2,),
                )

                # auc is evaluated at new_time: need to keep the indices corresponding to min and max time
                # to ensure that new_time is within follow-up time
                # with prob 0.5, take the index corresponding to max time from self and with prob 0.5 from other
                # same for min time
                index_max_time = torch.argmax(self.time).unsqueeze(0)
                index_max_time += (torch.rand(1) < 0.5) * self.estimate.shape[0]
                index_min_time = torch.nonzero(
                    self.time == self.time[self.event].min()
                ).squeeze(1)
                index_min_time += (torch.rand(1) < 0.5) * self.estimate.shape[0]
                index = torch.cat(
                    (index, index_min_time, index_max_time),
                    dim=0,
                )

                # with prob 0.5, take the weight_new_time from self and with prob 0.5 from other
                weight_new_time = (
                    self.weight_new_time
                    if torch.rand(1) < 0.5
                    else other.weight_new_time
                )

                aucs.append(
                    self(  # sample with replacement from pooled sample
                        torch.cat((self.estimate, other.estimate))[index, :],
                        torch.cat((self.event, other.event))[index],
                        torch.cat((self.time, other.time))[index],
                        self.auc_type,
                        torch.cat((self.weight, other.weight))[index],
                        self.new_time,
                        weight_new_time,
                        instate=False,
                    )
                )

        aucs = torch.stack(aucs)

        if torch.any(torch.isnan(aucs)):
            raise ValueError("The AUC computed using bootstrap should not be NaN.")

        return aucs

    @staticmethod
    def _find_torch_unique_indices(
        inverse_indices: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """return unique_sorted_indices such that
        sorted_unique_tensor[inverse_indices] = original_tensor
        original_tensor[unique_sorted_indices] = sorted_unique_tensor

        Usage:
            _, inverse_indices, counts = torch.unique(
                    x, sorted=True, return_inverse=True, return_counts=True
            )
            sorted_unique_indices = Auc._find_torch_unique_indices(
                    inverse_indices, counts
            )
        """

        _, ind_sorted = torch.sort(inverse_indices, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        return ind_sorted[cum_sum]

    @staticmethod
    def _validate_auc_inputs(
        estimate, time, auc_type, new_time, weight, weight_new_time
    ):
        # check new_time and weight are provided, weight_new_time should be provided
        if all([new_time is not None, weight is not None, weight_new_time is None]):
            raise ValueError(
                "Please provide 'weight_new_time', the weight evaluated at 'new_time'."
            )

        # check if auc type is correct
        if auc_type not in ["cumulative", "incident"]:
            raise ValueError(
                "The 'auc_type' parameter must be either 'cumulative' or 'incident'."
            )

        # check if new_time are not specified and time-dependent estimate are not evaluated at time
        if new_time is None and estimate.ndim == 2 and estimate.shape[1] != 1:
            if len(time) != estimate.shape[1]:
                raise ValueError(
                    "Mismatched dimensions: The number of columns in 'estimate' does not match the length of 'time'. "
                    "Please provide the times at which 'estimate' is evaluated using the 'new_time' input."
                )

    @staticmethod
    def _update_auc_new_time(
        estimate: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        new_time: torch.Tensor,
        weight: torch.Tensor,
        weight_new_time: torch.Tensor,
    ) -> torch.Tensor:
        # update new time
        if (
            new_time is not None
        ):  # if new_time are specified: ensure it has the correct format
            # ensure that new_time are float
            if isinstance(new_time, int):
                new_time = torch.tensor([new_time]).float()

            # example if new_time is tensor(12), unsqueeze to tensor([12])
            if new_time.ndim == 0:
                new_time = new_time.unsqueeze(0)

        else:  # else: find new_time
            # if new_time are not specified, use unique event time
            mask = event & (time < torch.max(time))
            new_time, inverse_indices, counts = torch.unique(
                time[mask], sorted=True, return_inverse=True, return_counts=True
            )
            sorted_unique_indices = Auc._find_torch_unique_indices(
                inverse_indices, counts
            )

            # select weight corresponding at new time
            if weight is not None:
                weight_new_time = (weight[mask])[sorted_unique_indices]

            # for time-dependent estimate, select those corresponding to new time
            if estimate.ndim == 2 and estimate.size(1) > 1:
                estimate = estimate[:, sorted_unique_indices]

        return estimate, new_time, weight, weight_new_time

    @staticmethod
    def _update_auc_estimate(
        estimate: torch.Tensor, new_time: torch.Tensor
    ) -> torch.Tensor:
        # squeeze estimate if shape = (n_samples, 1)
        if estimate.ndim == 2 and estimate.shape[1] == 1:
            estimate = estimate.squeeze(1)

        # Ensure estimate is (n_samples, n_times) shape
        if estimate.ndim == 1:
            estimate = estimate.unsqueeze(1).expand(
                (estimate.shape[0], new_time.shape[0])
            )

        return estimate

    @staticmethod
    def _update_auc_weight(
        time: torch.Tensor,
        new_time: torch.Tensor,
        weight: torch.Tensor,
        weight_new_time: torch.Tensor,
    ) -> torch.Tensor:
        # if weight was not specified, weight of 1
        if weight is None:
            weight = torch.ones_like(time)
            weight_new_time = torch.ones_like(new_time)

        return weight, weight_new_time


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
