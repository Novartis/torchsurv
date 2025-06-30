import copy
import sys
from typing import Optional

import torch
from scipy import stats

from torchsurv.tools.validate_data import (
    validate_log_shape,
    validate_new_time,
    validate_survival_data,
)

__all__ = ["BrierScore"]


class BrierScore:
    r"""Compute the Brier Score for survival models."""

    def __init__(self, checks: bool = True):
        """Initialize a BrierScore for survival class model evaluation.

        Args:
            checks (bool):
                Whether to perform input format checks.
                Enabling checks can help catch potential issues in the input data.
                Defaults to True.

        Examples:
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.rand((n,len(time)))
            >>> brier_score = BrierScore()
            >>> brier_score(estimate, event, time)
            tensor([0.2463, 0.2740, 0.3899, 0.1964, 0.3608, 0.2821, 0.1932, 0.2978, 0.1950,
                    0.1668])
            >>> brier_score.integral() # integrated brier score
            tensor(0.2862)
            >>> brier_score.confidence_interval() # default: parametric, two-sided
            tensor([[0.1061, 0.0604, 0.2360, 0.0533, 0.1252, 0.0795, 0.0000, 0.1512, 0.0381,
                     0.0051],
                    [0.3866, 0.4876, 0.5437, 0.3394, 0.5965, 0.4847, 0.4137, 0.4443, 0.3520,
                     0.3285]])
            >>> brier_score.p_value() # default: bootstrap permutation test, two-sided
            tensor([1.0000, 0.7860, 1.0000, 0.3840, 1.0000, 1.0000, 0.3840, 1.0000, 0.7000,
                    0.2380])
        """
        self.checks = checks

        # init instate attributes
        self.order_time = None
        self.time = None
        self.event = None
        self.weight = None
        self.new_time = None
        self.weight_new_time = None
        self.estimate = None
        self.brier_score = None
        self.residuals = None

    def __call__(
        self,
        estimate: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        new_time: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        weight_new_time: Optional[torch.Tensor] = None,
        instate: bool = True,
    ) -> torch.Tensor:
        r"""Compute the Brier Score.

        Args:
            estimate (torch.Tensor):
                Estimated probability of remaining event-free (i.e., survival function).
                Can be of shape = (n_samples, n_samples) if subject-specific survival is evaluated at ``time``,
                or of shape = (n_samples, n_times) if subject-specific survival is evaluated at ``new_time``.
            event (torch.Tensor, boolean):
                Event indicator of size n_samples (= True if event occurred)
            time (torch.Tensor, float):
                Time-to-event or censoring of size n_samples.
            new_time (torch.Tensor, float, optional):
                Time points at which to evaluate the Brier score of size n_times.
                Defaults to unique ``time``.
            weight (torch.Tensor, optional):
                Optional sample weight evaluated at ``time`` of size n_samples.
                Defaults to 1.
            weight_new_time (torch.Tensor, optional):
                Optional sample weight evaluated at ``new_time`` of size n_times.
                Defaults to 1.

        Returns:
            torch.Tensor: Brier score evaluated at ``new_time``.

        Note:
            The function evaluates the time-dependent Brier score at time :math:`t \in \{t_1, \cdots, t_T\}` (argument ``new_time``).

            For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
            censoring time. Survival data consist of the event indicator, :math:`\delta_i=(X_i\leq D_i)`
            (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
            (argument ``time``).

            The survival function, of subject :math:`i`
            is specified through :math:`S_i: [0, \infty) \rightarrow [0,1]`.
            The argument ``estimate`` is the estimated survival function. If ``new_time`` is specified, it should be of
            shape = (N,T) (:math:`(i,k)` th element is :math:`\hat{S}_i(t_k)`); if ``new_time`` is not specified,
            it should be of shape = (N,N) (:math:`(i,j)` th element is :math:`\hat{S}_i(T_j)`).

            The time-dependent Brier score :cite:p:`Graf1999` at time :math:`t` is the mean squared error of the event status

            .. math::

                BS(t) = \mathbb{E}\left[\left(1\left(X > t\right) - \hat{S}(t)\right)^2\right]

            The default Brier score estimate is

            .. math::

                \hat{BS}(t) = \frac{1}{n}\sum_i 1(T_i \leq t, \delta_i = 1) (0 - \hat{S}_i(t))^2 + 1(T_1 > t) (1- \hat{S}_i(t))^2

            To account for the fact that the event time are censored, the
            inverse probability weighting technique can be used. In this context,
            each subject associated with time
            :math:`t` is weighted by the inverse probability of censoring :math:`\omega(t) = 1 / \hat{D}(t)`, where
            :math:`\hat{D}(t)` is the Kaplan-Meier estimate of the censoring distribution, :math:`P(D>t)`.
            The censoring-adjusted Brier score is

            .. math::

                \hat{BS}(t) = \frac{1}{n}\sum_i \omega(T_i) 1(T_i \leq t, \delta_i = 1) (0 - \hat{S}_i(t))^2 + \omega(t) 1(T_1 > t) (1- \hat{S}_i(t))^2

            The censoring-adjusted Brier score can be obtained by specifying the argument
            ``weight``, the weights evaluated at each ``time`` (:math:`\omega(T_1), \cdots, \omega(T_N)`).
            If ``new_time`` is specified, the argument  ``weight_new_time``
            should also be specified accordingly, the weights evaluated at each ``new_time``
            (:math:`\omega(t_1), \cdots, \omega(t_K)`).
            In the context of train/test split, the weights should be derived from the censoring distribution estimated in the training data.


        Examples:
            >>> from torchsurv.stats.ipcw import get_ipcw
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.rand((n,len(time)))
            >>> brier_score = BrierScore()
            >>> brier_score(estimate, event, time)
            tensor([0.2463, 0.2740, 0.3899, 0.1964, 0.3608, 0.2821, 0.1932, 0.2978, 0.1950,
                    0.1668])
            >>> ipcw = get_ipcw(event, time) # ipcw at time
            >>> brier_score(estimate, event, time, weight=ipcw) # censoring-adjusted brier-score
            tensor([0.2463, 0.2740, 0.4282, 0.2163, 0.4465, 0.3826, 0.2630, 0.3888, 0.2219,
                    0.1882])
            >>> new_time = torch.unique(torch.randint(low=5, high=time.max().int(), size=(n*2,)).float())
            >>> ipcw_new_time = get_ipcw(event, time, new_time) # ipcw at new_time
            >>> estimate = torch.rand((n,len(new_time)))
            >>> brier_score(estimate, event, time, new_time, ipcw, ipcw_new_time) # censoring-adjusted brier-score at new time
            tensor([0.4036, 0.3014, 0.2517, 0.3947, 0.4200, 0.3908, 0.3766, 0.3737, 0.3596,
                    0.2088, 0.4922, 0.3237, 0.2255, 0.1841, 0.3029, 0.6919, 0.2357, 0.3507,
                    0.4364, 0.3312])

        References:

            .. bibliography::
                :filter: False

                Graf1999

        """

        # mandatory input format checks
        BrierScore._validate_brier_score_inputs(
            estimate, time, new_time, weight, weight_new_time
        )

        # update inputs as required
        (
            estimate,
            new_time,
            weight,
            weight_new_time,
        ) = BrierScore._update_brier_score_new_time(
            estimate, time, new_time, weight, weight_new_time
        )
        weight, weight_new_time = BrierScore._update_brier_score_weight(
            time, new_time, weight, weight_new_time
        )

        # further input format checks
        if self.checks:
            validate_survival_data(event, time)
            validate_new_time(new_time, time, within_follow_up=False)
            validate_log_shape(estimate)

        # Calculating the residuals for each subject and time point
        residuals = torch.zeros_like(estimate)
        for index, new_time_i in enumerate(new_time):
            est = estimate[:, index]
            is_case = ((time <= new_time_i) & (event)).int()
            is_control = (time > new_time_i).int()

            residuals[:, index] = (
                torch.square(est) * is_case * weight
                + torch.square(1.0 - est) * is_control * weight_new_time[index]
            )

        # Calculating the brier scores at each time point
        brier_score = torch.mean(residuals, axis=0)

        # Create/overwrite internal attributes states
        if instate:
            # sort all objects by time
            self.order_time = torch.argsort(time, dim=0)
            self.time = time[self.order_time]
            self.event = event[self.order_time]
            self.weight = weight[self.order_time]
            self.new_time = new_time
            self.weight_new_time = weight_new_time
            self.estimate = torch.index_select(estimate, 0, self.order_time)
            self.brier_score = brier_score
            self.residuals = residuals

        return brier_score

    def integral(self):
        r"""Compute the integrated Brier Score.

        Returns:
            torch.Tensor: Integrated Brier Score.

        Examples:
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.rand((n,len(time)))
            >>> brier_score = BrierScore()
            >>> brier_score(estimate, event, time)
            tensor([0.2463, 0.2740, 0.3899, 0.1964, 0.3608, 0.2821, 0.1932, 0.2978, 0.1950,
                    0.1668])
            >>> brier_score.integral() # integrated brier score
            tensor(0.2862)

        Note:

            The integrated Brier score is the integral of the time-dependent Brier score over the interval
            :math:`[t_1, t_2]`, where :math:`t_1 = \min\left(\{T_i\}_{i = 1, \cdots, N}\right)` and :math:`t_2 = \max\left(\{T_i\}_{i = 1, \cdots, N}\right)`.
            It is defined by :cite:p:`Graf1999`

            .. math::

                    \hat{IBS} = \int_{t_1}^{t_2} \hat{BS}(t) dW(t)

            where :math:`W(t) = t / t_2`.

            The integral is estimated with the trapzoidal rule.

        """
        # Single time available
        if len(self.new_time) == 1:
            brier = self.brier_score[0]
        else:
            brier = torch.trapezoid(self.brier_score, self.new_time) / (
                self.new_time[-1] - self.new_time[0]
            )
        return brier

    def confidence_interval(
        self,
        method: str = "parametric",
        alpha: float = 0.05,
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
    ) -> torch.Tensor:
        """Compute the confidence interval of the Brier Score.

        This function calculates either the pointwise confidence interval or the bootstrap
        confidence interval for the Brier Score. The pointwise confidence interval is computed
        assuming that the Brier score is normally distributed and using empirical standard errors.
        The bootstrap confidence interval is constructed based on the distribution of bootstrap samples.

        Args:
            method (str):
                Method for computing confidence interval. Defaults to "parametric".
                Must be one of the following: "parametric", "bootstrap".
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
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.rand((n,len(time)))
            >>> brier_score = BrierScore()
            >>> brier_score(estimate, event, time)
            tensor([0.2463, 0.2740, 0.3899, 0.1964, 0.3608, 0.2821, 0.1932, 0.2978, 0.1950,
                    0.1668])
            >>> brier_score.confidence_interval() # default: parametric, two-sided
            tensor([[0.1061, 0.0604, 0.2360, 0.0533, 0.1252, 0.0795, 0.0000, 0.1512, 0.0381,
                     0.0051],
                    [0.3866, 0.4876, 0.5437, 0.3394, 0.5965, 0.4847, 0.4137, 0.4443, 0.3520,
                     0.3285]])
            >>> brier_score.confidence_interval(method = "bootstrap", alternative = "greater")
            tensor([[0.1455, 0.1155, 0.2741, 0.0903, 0.1985, 0.1323, 0.0245, 0.1938, 0.0788,
                     0.0440],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                     1.0000]])


        """

        assert (
            hasattr(self, "brier_score") and self.brier_score is not None
        ), "Error: Please calculate brier score using `BrierScore()` before calling `confidence_interval()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                "'alternative' parameter must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "bootstrap":
            conf_int = self._confidence_interval_bootstrap(
                alpha, alternative, n_bootstraps
            )
        elif method == "parametric":
            conf_int = self._confidence_interval_parametric(alpha, alternative)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'parametric' or 'bootstrap'."
            )
        return conf_int

    def p_value(
        self,
        method: str = "bootstrap",
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
        null_value: float = None,
    ) -> torch.Tensor:
        """Perform a one-sample hypothesis test on the Brier score.

        This function calculates either the pointwise p-value or the bootstrap p-value
        for testing the null hypothesis that the estimated brier score is equal to
        bs0, where bs0 is the brier score that would be expected if the survival model
        was not providing accurate predictions beyond
        random chance. The pointwise p-value is computed assuming that the
        Brier score is normally distributed and using the empirical standard errors.
        To obtain the pointwise p-value, the Brier score under the null, bs0, must
        be provided.
        The bootstrap p-value is derived by permuting survival function's predictions
        to estimate the the sampling distribution under the null hypothesis.

        Args:
            method (str):
                Method for computing p-value. Defaults to "bootstrap".
                Must be one of the following: "parametric", "bootstrap".
            alternative (str):
                Alternative hypothesis. Defaults to "two_sided".
                Must be one of the following: "two_sided" (Brier score is not equal to bs0),
                "greater" (Brier score is greater than bs0), "less" (Brier score is less than bs0).
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ```method``` is not "bootstrap".
            null_value (float):
                The Brier score expected if the survival model was not
                providing accurate predictions beyond what would be beyond
                by random chance alone, i.e., bs0.


        Returns:
            torch.Tensor: p-value of the statistical test.

        Examples:
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> new_time = torch.unique(time)
            >>> estimate = torch.rand((n,len(new_time)))
            >>> brier_score = BrierScore()
            >>> brier_score(estimate, event, time, new_time)
            tensor([0.3465, 0.5310, 0.4222, 0.4582, 0.3601, 0.3395, 0.2285, 0.1975, 0.3120,
                    0.3883])
            >>> brier_score.p_value() # Default: bootstrap, two_sided
            tensor([1.0000, 0.0560, 1.0000, 1.0000, 1.0000, 1.0000, 0.8320, 0.8620, 1.0000,
                    1.0000])
            >>> brier_score.p_value(method = "parametric", alternative = "less", null_value = 0.3) # H0: bs = 0.3, Ha: bs < 0.3
            tensor([0.7130, 0.9964, 0.8658, 0.8935, 0.6900, 0.6630, 0.1277, 0.1128, 0.5383,
                    0.8041])

        """

        assert (
            hasattr(self, "brier_score") and self.brier_score is not None
        ), "Error: Please calculate the brier score using `BrierScore()` before calling `p_value()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                "'alternative' parameter must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "parametric" and null_value is None:
            raise ValueError(
                "Error: If the method is 'parametric', you must provide the 'null_value'."
            )

        if method == "parametric":
            pvalue = self._p_value_parametric(alternative, null_value)
        elif method == "bootstrap":
            pvalue = self._p_value_bootstrap(alternative, n_bootstraps)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'parametric' or 'bootstrap'."
            )
        return pvalue

    def compare(
        self, other, method: str = "parametric", n_bootstraps: int = 999
    ) -> torch.Tensor:
        """Compare two Brier scores.

        This function compares two Brier scores computed on the
        same data with different risk scores. The statistical hypotheses are
        formulated as follows, null hypothesis: brierscore1 = brierscore2 and alternative
        hypothesis: brierscore1 < brierscore2.
        The statistical test is either a Student t-test for paired samples or a two-sample bootstrap test.
        The Student t-test for paired samples assumes that the Brier Scores are normally distributed
        and uses the Brier scores' empirical standard errors.

        Args:
            other (BrierScore):
                Another instance of the BrierScore class representing brierscore2.
            method (str):
                Statistical test used to perform the hypothesis test. Defaults to "parametric".
                Must be one of the following: "parametric", "bootstrap".
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor: p-value of the statistical test.

        Examples:
            >>> _ = torch.manual_seed(52)
            >>> n = 10
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> brier_score = BrierScore()
            >>> brier_score(torch.rand((n,len(time))), event, time)
            tensor([0.2463, 0.2740, 0.3899, 0.1964, 0.3608, 0.2821, 0.1932, 0.2978, 0.1950,
                    0.1668])
            >>> brier_score2 = BrierScore()
            >>> brier_score2(torch.rand((n,len(time))), event, time)
            tensor([0.4136, 0.2750, 0.3002, 0.2826, 0.2030, 0.2643, 0.2525, 0.2964, 0.1804,
                    0.3109])
            >>> brier_score.compare(brier_score2) # default: parametric
            tensor([0.1793, 0.4972, 0.7105, 0.1985, 0.9254, 0.5591, 0.3455, 0.5060, 0.5437,
                    0.0674])
            >>> brier_score.compare(brier_score2, method = "bootstrap")
            tensor([0.1360, 0.5030, 0.7310, 0.2090, 0.8630, 0.5490, 0.3120, 0.5110, 0.5460,
                    0.1030])

        """

        assert (
            hasattr(self, "brier_score") and self.brier_score is not None
        ), "Error: Please calculate the brier score using `BrierScore()` before calling `compare()`."

        # assert that the same data were used to compute the two brier score
        if torch.any(self.event != other.event) or torch.any(self.time != other.time):
            raise ValueError(
                "Mismatched survival data: 'time' and 'event' should be the same for both brier score computations."
            )
        if torch.any(self.new_time != other.new_time):
            raise ValueError(
                "Mismatched evaluation times: 'new_time' should be the same for both brier score computations."
            )

        if method == "parametric":
            pvalue = self._compare_parametric(other)
        elif method == "bootstrap":
            pvalue = self._compare_bootstrap(other, n_bootstraps)
        else:
            raise ValueError(
                "Method not implemented. Please choose either 'parametric' or 'bootstrap'."
            )
        return pvalue

    def _brier_score_se(self):
        """Brier Score's empirical standard errors."""

        return torch.std(self.residuals, axis=0) / (self.time.shape[0] ** (1 / 2))

    def _confidence_interval_parametric(
        self, alpha: float, alternative: str
    ) -> torch.Tensor:
        """Confidence interval of Brier score assuming that the Brier score
        is normally distributed and using empirical standard errors.
        """

        alpha = alpha / 2 if alternative == "two_sided" else alpha

        brier_score_se = self._brier_score_se()

        if torch.all(brier_score_se) > 0:
            ci = (
                -torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(alpha))
                * brier_score_se
            )
            lower = torch.max(torch.tensor(0.0), self.brier_score - ci)
            upper = torch.min(torch.tensor(1.0), self.brier_score + ci)

            if alternative == "less":
                lower = torch.zeros_like(lower)
            elif alternative == "greater":
                upper = torch.ones_like(upper)
        else:
            raise ValueError(
                "The standard error of the brier score must be a positive value."
            )

        return torch.stack([lower, upper], dim=0)

    def _confidence_interval_bootstrap(
        self, alpha: float, alternative: str, n_bootstraps: int
    ) -> torch.Tensor:
        """Bootstrap confidence interval of the Brier Score using Efron percentile method.

        References:
            Efron, Bradley; Tibshirani, Robert J. (1993).
                An introduction to the bootstrap, New York: Chapman & Hall, software.
        """

        # brier score given bootstrap distribution
        brier_score_bootstrap = self._bootstrap_brier_score(
            metric="confidence_interval", n_bootstraps=n_bootstraps
        )

        # initialize tensor to store confidence intervals
        lower = torch.zeros_like(self.brier_score)
        upper = torch.zeros_like(self.brier_score)

        # iterate over time
        for index_t in range(len(self.brier_score)):
            # obtain confidence interval
            if alternative == "two_sided":
                lower[index_t], upper[index_t] = torch.quantile(
                    brier_score_bootstrap[:, index_t],
                    torch.tensor(
                        [alpha / 2, 1 - alpha / 2], device=self.brier_score.device
                    ),
                )
            elif alternative == "less":
                upper[index_t] = torch.quantile(
                    brier_score_bootstrap[:, index_t],
                    torch.tensor(1 - alpha, device=self.brier_score.device),
                )
                lower[index_t] = torch.tensor(0.0, device=self.brier_score.device)
            elif alternative == "greater":
                lower[index_t] = torch.quantile(
                    brier_score_bootstrap[:, index_t],
                    torch.tensor(alpha, device=self.brier_score.device),
                )
                upper[index_t] = torch.tensor(1.0, device=self.brier_score.device)

        return torch.stack([lower, upper], dim=0)

    def _p_value_parametric(
        self, alternative: str, null_value: float = 0.5
    ) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the Brier score
        assuming that the Brier score is normally distributed and using empirical standard error.
        """

        brier_score_se = self._brier_score_se()

        # get p-value
        if torch.all(brier_score_se) > 0:
            p = torch.distributions.normal.Normal(0, 1).cdf(
                (self.brier_score - null_value) / brier_score_se
            )
            if alternative == "two_sided":
                mask = self.brier_score >= 0.5
                p[mask] = 1 - p[mask]
                p *= 2
                p = torch.min(
                    torch.tensor(1.0, device=self.brier_score.device), p
                )  # in case critical value is below 0.5
            elif alternative == "greater":
                p = 1 - p
        else:
            raise ValueError(
                "The standard error of the brier score must be a positive value."
            )

        return p

    def _p_value_bootstrap(self, alternative, n_bootstraps) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the Brier score using
        permutation of survival distribution prediction to estimate sampling distribution under the null
        hypothesis.
        """

        # brier score bootstraps given null distribution
        brierscore0 = self._bootstrap_brier_score(
            metric="p_value", n_bootstraps=n_bootstraps
        )

        # initialize empty tensor to store p-values
        p_values = torch.zeros_like(self.brier_score)

        # iterate over time
        for index_t, brier_score_t in enumerate(self.brier_score):
            # Derive p-value
            p = (1 + torch.sum(brierscore0[:, index_t] <= brier_score_t)) / (
                n_bootstraps + 1
            )
            if alternative == "two_sided":
                if brier_score_t >= 0.5:
                    p = 1 - p
                p *= 2
                p = torch.min(
                    torch.tensor(1.0, device=self.brier_score.device), p
                )  # in case very small bootstrap sample size is used
            elif alternative == "greater":
                p = 1 - p

            p_values[index_t] = p

        return p_values

    def _compare_parametric(self, other):
        """Student t-test for paired samples assuming that
        the Brier scores are normally distributed and using
        empirical standard errors."""

        # sample size
        n_samples = self.time.shape[0]

        # initialize empty vector to store p_values
        p_values = torch.zeros_like(self.brier_score)

        # iterate over time
        for index_t, brier_score_t in enumerate(self.brier_score):
            # compute standard error of the difference
            paired_se = torch.std(
                self.residuals[:, index_t] - other.residuals[:, index_t]
            ) / (n_samples ** (1 / 2))

            # compute t-stat
            t_stat = (brier_score_t - other.brier_score[index_t]) / paired_se

            # p-value
            p_values[index_t] = torch.tensor(
                stats.t.cdf(
                    t_stat, df=n_samples - 1
                ),  # student-t cdf not available on torch
                dtype=self.brier_score.dtype,
                device=self.brier_score.device,
            )

        return p_values

    def _compare_bootstrap(self, other, n_bootstraps) -> torch.Tensor:
        """Bootstrap two-sample test to compare two Brier scores."""

        # bootstrap brier scores given null hypothesis that brierscore1 and
        # brierscore2 come from the same distribution
        brier_score1_null = self._bootstrap_brier_score(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )
        brier_score2_null = self._bootstrap_brier_score(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )

        # bootstrapped test statistics
        t_boot = brier_score1_null - brier_score2_null

        # observed test statistics
        t_obs = self.brier_score - other.brier_score

        # initialize empty tensor to store p-values
        p_values = torch.zeros_like(self.brier_score)

        # iterate over time
        for index_t, _ in enumerate(self.brier_score):
            p_values[index_t] = (
                1 + torch.sum(t_boot[:, index_t] <= t_obs[index_t])
            ) / (n_bootstraps + 1)

        return p_values

    def _bootstrap_brier_score(
        self, metric: str, n_bootstraps: int, other=None
    ) -> torch.Tensor:
        """Compute bootstrap samples of the Brier Score.

        Args:
            metric (str): Must be one of the following: "confidence_interval", "compare".
                If "confidence_interval", computes bootstrap
                samples of the Brier score given the data distribution. If "compare", computes
                bootstrap samples of the Brier score given the sampling distribution under the comparison test
                null hypothesis (brierscore1 = brierscore2).
            n_bootstraps (int): Number of bootstrap samples.
            other (optional, BrierScore):
                Another instance of the BrierScore class representing brierscore2.
                Only required if ``metric`` is "compare".


        Returns:
            torch.Tensor: Bootstrap samples of Brier score.
        """

        # Initiate empty list to store brier score
        brier_scores = []

        # Get the bootstrap samples of brier score
        for _ in range(n_bootstraps):
            if (
                metric == "confidence_interval"
            ):  # bootstrap samples given data distribution
                index = torch.randint(
                    low=0,
                    high=self.estimate.shape[0],
                    size=(self.estimate.shape[0],),
                )
                brier_scores.append(
                    self(
                        self.estimate[index, :],
                        self.event[index],
                        self.time[index],
                        self.new_time,
                        self.weight[index],
                        self.weight_new_time,
                        instate=False,
                    )
                )  # Run without saving internal state
            elif (
                metric == "compare"
            ):  # bootstrap samples given null distribution (brierscore1 = brierscore2)
                index = torch.randint(
                    low=0,
                    high=self.estimate.shape[0] * 2,
                    size=(self.estimate.shape[0],),
                )

                # with prob 0.5, take the weight_new_time from self and with prob 0.5 from other
                weight_new_time = (
                    self.weight_new_time
                    if torch.rand(1) < 0.5
                    else other.weight_new_time
                )

                brier_scores.append(
                    self(  # sample with replacement from pooled sample
                        torch.cat((self.estimate, other.estimate))[index, :],
                        torch.cat((self.event, other.event))[index],
                        torch.cat((self.time, other.time))[index],
                        self.new_time,
                        torch.cat((self.weight, other.weight))[index],
                        weight_new_time,
                        instate=False,
                    )
                )
            elif (
                metric == "p_value"
            ):  # bootstrap samples given null distribution (estimate are not informative)
                estimate = copy.deepcopy(self.estimate)
                estimate = estimate[
                    torch.randperm(estimate.shape[0]), :
                ]  # Shuffle estimate
                brier_scores.append(
                    self(
                        estimate,
                        self.event,
                        self.time,
                        self.new_time,
                        self.weight,
                        self.weight_new_time,
                        instate=False,
                    )
                )  # Run without saving internal state

        brier_scores = torch.stack(brier_scores, dim=0)

        if torch.any(torch.isnan(brier_scores)):
            raise ValueError(
                "The brier score computed using bootstrap should not be NaN."
            )

        return brier_scores

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
    def _validate_brier_score_inputs(
        estimate: torch.Tensor,
        time: torch.Tensor,
        new_time: torch.Tensor,
        weight: torch.Tensor,
        weight_new_time: torch.Tensor,
    ) -> torch.Tensor:
        # check new_time and weight are provided, weight_new_time should be provided
        if all([new_time is not None, weight is not None, weight_new_time is None]):
            raise ValueError(
                "Please provide 'weight_new_time', the weight evaluated at 'new_time'."
            )

        # check that estimate has 2 dimensions estimate are probabilities
        if torch.any(estimate < 0) or torch.any(estimate > 1):
            raise ValueError(
                "The 'estimate' input should contain estimated survival probabilities between 0 and 1."
            )

        # check if estimate is of the correct dimension
        if estimate.ndim != 2:
            raise ValueError("The 'estimate' input should have two dimensions.")

        # check if new_time are not specified and estimate are not evaluated at time
        if new_time is None and len(time) != estimate.shape[1]:
            raise ValueError(
                "Mismatched dimensions: The number of columns in 'estimate' does not match the length of 'time'. "
                "Please provide the times at which 'estimate' is evaluated using the 'new_time' input."
            )

    @staticmethod
    def _update_brier_score_new_time(
        estimate: torch.Tensor,
        time: torch.Tensor,
        new_time: torch.Tensor,
        weight: torch.Tensor,
        weight_new_time: torch.Tensor,
    ) -> torch.Tensor:
        # check format of new_time
        if (
            new_time is not None
        ):  # if new_time are specified: ensure it has the correct format
            if isinstance(new_time, int):
                new_time = torch.tensor([new_time]).float()

            if new_time.ndim == 0:
                new_time = new_time.unsqueeze(0)

        else:  # else: find new_time
            # if new_time are not specified, use unique time
            new_time, inverse_indices, counts = torch.unique(
                time, sorted=True, return_inverse=True, return_counts=True
            )
            sorted_unique_indices = BrierScore._find_torch_unique_indices(
                inverse_indices, counts
            )

            # for time-dependent estimate, select those corresponding to new time
            estimate = estimate[:, sorted_unique_indices]

            if weight is not None:
                # select weight corresponding at new time
                weight_new_time = weight[sorted_unique_indices]

        return estimate, new_time, weight, weight_new_time

    @staticmethod
    def _update_brier_score_weight(
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
