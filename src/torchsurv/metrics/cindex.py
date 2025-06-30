import copy
import sys
import warnings
from typing import Optional, Tuple

import torch
from scipy import stats
from torchmetrics import regression

from torchsurv.tools.validate_data import validate_log_shape, validate_survival_data

__all__ = ["ConcordanceIndex"]


class ConcordanceIndex:
    """Compute the Concordance Index (C-index) for survival models."""

    def __init__(
        self,
        tied_tol: float = 1e-8,
        checks: bool = True,
    ) -> dict:
        """Initialize a ConcordanceIndex for survival class model evaluation.

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
            >>> n = 64
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> cindex = ConcordanceIndex()
            >>> cindex(estimate, event, time)
            tensor(0.5337)
            >>> cindex.confidence_interval() # default: Noether, two_sided
            tensor([0.3251, 0.7423])
            >>> cindex.p_value(method='bootstrap', alternative='greater')
            tensor(0.2620)
        """
        self.tied_tol = tied_tol
        self.checks = checks

        # init instate attributes
        self.time = None
        self.event = None
        self.estimate = None
        self.cindex = None
        self.concordant = None
        self.discordant = None
        self.tied_risk = None
        self.weight = None
        self.weight_squared = None
        self.tmax = None

    # disable long line check for this function due to docstring. Might be better to try and do multiline math in the TeX formula
    # pylint: disable=C0301
    def __call__(
        self,
        estimate: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        tmax: Optional[torch.Tensor] = None,
        instate: bool = True,
    ) -> torch.Tensor:
        r"""Compute the Concordance Index (C-index).

        The concordance index is the probability that a model correctly predicts
        which of two comparable samples will experience an event first based on their
        estimated risk scores.

        Args:
            estimate (torch.Tensor):
                Estimated risk of event occurrence (i.e., risk score).
                Can be of shape = (n_samples,) if subject-specific risk score is time-independent,
                or of shape = (n_samples, n_samples) if subject-specific risk score is evaluated at ``time``.
            event (torch.Tensor, boolean):
                Event indicator of size n_samples (= True if event occurred).
            time (torch.Tensor, float):
                Time-to-event or censoring of size n_samples.
            weight (torch.Tensor, optional):
                Optional sample weight of size n_samples. Defaults to 1.
            tmax (torch.Tensor, optional):
                Truncation time. Defaults to None.
                ``tmax`` should be chosen such that the probability of
                being censored after time ``tmax`` is non-zero. If ``tmax`` is None, no truncation
                is performed.
            instate (bool):
                Whether to create/overwrite internal attributes states.
                Defaults to True.

        Returns:
            torch.Tensor: Concordance-index

        Note:

            The concordance index provides a global assessment of a fitted survival model over the entire observational
            period rather than focussing on the prediction for a fixed time (e.g., 10-year mortality).
            It is recommended to use AUC instead of the concordance index for such time-dependent predictions, as
            AUC is proper in this context, while the concordance index is not :cite:p:`Blanche2018`.

            For each subject :math:`i \in \{1, \cdots, N\}`, denote :math:`X_i` as the survival time and :math:`D_i` as the
            censoring time. Survival data consist of the event indicator, :math:`\delta_i=(X_i\leq D_i)`
            (argument ``event``) and the time-to-event or censoring, :math:`T_i = \min(\{ X_i,D_i \})`
            (argument ``time``).

            The risk score measures the risk (or a proxy thereof) that a subject has an event.
            The function accepts time-dependent risk score and time-independent risk score. The time-dependent risk score
            of subject :math:`i` is specified through a function :math:`q_i: [0, \infty) \rightarrow \mathbb{R}`.
            The time-independent risk score of subject :math:`i` is specified by a constant :math:`q_i`.
            The argument ``estimate`` is the estimated risk score.
            For time-dependent risk score, the argument ``estimate`` should be a tensor of shape = (N,N)
            (:math:`(i,j)` th element is :math:`\hat{q}_i(T_j)`). For time-independent risk score, the argument ``estimate``
            should be a tensor of size N (:math:`i` th element is :math:`\hat{q}_i`).

            For a pair :math:`(i,j)`, we say that the pair is comparable if the event of :math:`i` has occurred before
            the event of :math:`j`, i.e., :math:`X_i < X_j`. Given that the pair is comparable, we say that the pair is
            concordant if :math:`q_i(X_i) > q_j(X_i)`.

            The concordance index measures the probability that, for a pair of randomly selected comparable samples,
            the one that experiences an event first has a higher risk. The concordance index is defined as

            .. math::

                C = p(q_i(X_i) > q_j(X_i) \: | \: X_i < X_j)

            The default concordance index estimate is the popular nonparametric estimation proposed by :cite:t:`Harrell1996`

            .. math::

                \hat{C} = \frac{\sum_i\sum_j \delta_i \: I(T_i < T_j)\left(I \left( \hat{q}_i(T_i) > \hat{q}_j(T_i) \right) + \frac{1}{2} I\left(\hat{q}_i(T_i) = \hat{q}_j(T_i)\right)\right)}{\sum_i\sum_j \delta_i\: I\left(T_i < T_j\right)}

            When the event time are censored, the Harrell's concordance index converges to an association measure that
            involves the censoring distribution.
            To address this shortcoming, :cite:t:`Uno2011`  proposed to employ the
            inverse probability weighting technique. In this context, each subject with event time at
            :math:`t` is weighted by the inverse probability of censoring :math:`\omega(t) = 1 / \hat{D}(t)`, where
            :math:`\hat{D}(t)` is the Kaplan-Meier estimate of the censoring distribution, :math:`P(D>t)`.
            Let :math:`\omega(T_i)` be the weight associated with subject time :math:`i` (argument ``weight``).
            The concordance index estimate with weight is,

            .. math::

                \hat{C} = \frac{\sum_i\sum_j \delta_i \: \omega(T_i)^2 \: I(T_i < T_j)\left(I \left( \hat{q}_i(T_i) > \hat{q}_j(T_i) \right) + \frac{1}{2} I\left(\hat{q}_i(T_i) = \hat{q}_j(T_i)\right)\right)}{\sum_i\sum_j \delta_i \: \omega(T_i)^2\: I\left(T_i < T_j\right)}

            In the context of train/test split, the weights should be derived from the censoring distribution estimated in the training data.
            Specifically, the censoring distribution is estimated using the training set and then evaluated at the subject time within the test set.

        Examples:
            >>> from torchsurv.stats.ipcw import get_ipcw
            >>> _ = torch.manual_seed(42)
            >>> n = 64
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> cindex = ConcordanceIndex()
            >>> cindex(estimate, event, time) # Harrell's c-index
            tensor(0.5337)
            >>> ipcw = get_ipcw(event, time) # ipcw at subject time
            >>> cindex(estimate, event, time, weight=ipcw) # Uno's c-index
            tensor(0.5453)

        References:

            .. bibliography::
                :filter: False

                Blanche2018
                Harrell1996
                Uno2011

        """
        # update inputs if necessary
        estimate = ConcordanceIndex._update_cindex_estimate(estimate)
        weight = ConcordanceIndex._update_weight(time, weight, tmax)

        # square the weight
        weight_squared = torch.square(weight)

        # Inputs checks
        if self.checks:
            validate_survival_data(event, time)
            validate_log_shape(estimate)

        # find comparable pairs
        comparable = self._get_comparable_and_tied_time(event, time)

        # get order index by time
        order = torch.argsort(time)

        # Initialize variables to count concordant, discordant, and tied risk pairs
        concordant, discordant, tied_risk = [], [], []

        # Initialize numerator and denominator for calculating the concordance index
        numerator, denominator = 0.0, 0.0

        # Iterate through the comparable items over time
        # (dictionary with indices that have an event occurred at time and boolean masks of comparable pair a time)
        for ind, mask in comparable.items():
            # Extract risk score, event indicator and weight for the current sample
            est_i, event_i, w_i = (
                estimate[order[ind], order[ind]],
                event[order[ind]],
                weight_squared[order[ind]],
            )

            # Extract risk score of comparable pairs and the number of comparable samples
            est_j, n = estimate[order[mask], order[ind]], mask.sum()

            # Check that the current sample is uncensored
            assert (
                event_i
            ), f"Got censored sample at index {order[ind]}, but expected uncensored"

            # Identify tied pairs based on a tolerance (ties)
            ties = (
                torch.absolute(est_j.float() - est_i.float()) <= self.tied_tol
            )  # ties
            n_ties = ties.sum()

            # Identify concordant pairs
            con = est_j < est_i
            n_con = con[~ties].sum()

            # Update numerator and denominator for concordance index calculation
            numerator += w_i * n_con + 0.5 * w_i * n_ties
            denominator += w_i * n

            # Update counts for tied, concordant, and discordant pairs
            tied_risk.append(w_i * n_ties)
            concordant.append(w_i * n_con)
            discordant.append(w_i * n - w_i * n_con - w_i * n_ties)

        # Create/overwrite internal attributes states
        if instate:
            self.time = time
            self.event = event
            self.estimate = estimate
            self.cindex = numerator / denominator
            self.concordant = torch.stack(concordant, dim=0)
            self.discordant = torch.stack(discordant, dim=0)
            self.tied_risk = torch.stack(tied_risk, dim=0)
            self.weight = weight
            self.weight_squared = weight_squared
            self.tmax = tmax

        return numerator / denominator  # cindex

    def confidence_interval(
        self,
        method: str = "noether",
        alpha: float = 0.05,
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
    ) -> torch.Tensor:
        """Compute the confidence interval of the Concordance index.

        This function calculates either the pointwise confidence interval or the bootstrap
        confidence interval for the concordance index. The pointwise confidence interval is computed
        assuming that the concordance index is normally distributed and using the standard error estimated with either the Noether or
        the conservative method :cite:p:`Pencina2004`.
        The bootstrap confidence interval is constructed based on the distribution of bootstrap samples.

        Args:
            method (str):
                Method for computing confidence interval. Defaults to "noether".
                Must be one of the following: "noether", "conservative", "bootstrap".
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
            >>> n = 64
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> cindex = ConcordanceIndex()
            >>> cindex(estimate, event, time)
            tensor(0.5337)
            >>> cindex.confidence_interval() # default: Noether, two_sided
            tensor([0.3251, 0.7423])
            >>> cindex.confidence_interval(method="bootstrap", alternative="greater")
            tensor([0.4459, 1.0000])
            >>> cindex.confidence_interval(method="conservative", alternative="less")
            tensor([0.0000, 0.7558])

        References:
            .. bibliography::
                :filter: False

                Pencina2004
        """

        assert isinstance(method, str)
        assert isinstance(alternative, str)
        assert (
            hasattr(self, "cindex") and self.cindex is not None
        ), "Error: Please calculate the concordance index using `ConcordanceIndex()` before calling `confidence_interval()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                f"'alternative' {alternative} must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "noether":
            conf_int = self._confidence_interval_noether(alpha, alternative)
        elif method == "bootstrap":
            conf_int = self._confidence_interval_bootstrap(
                alpha, alternative, n_bootstraps
            )
        elif method == "conservative":
            conf_int = self._confidence_interval_conservative(alpha, alternative)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'noether', 'conservative' or 'bootstrap'."
            )
        return conf_int

    def p_value(
        self,
        method: str = "noether",
        alternative: str = "two_sided",
        n_bootstraps: int = 999,
    ) -> torch.Tensor:
        """Perform one-sample hypothesis test on the Concordance index.

        This function calculates either the pointwise p-value or the bootstrap p-value
        for testing the null hypothesis that the concordance index is equal to 0.5.
        The pointwise p-value is computed assuming that the concordance index is normally distributed and using the
        standard error estimated using the Noether's method :cite:p:`Pencina2004`.
        The bootstrap p-value is derived by permuting risk predictions to estimate
        the sampling distribution under the null hypothesis.

        Args:
            method (str):
                Method for computing p-value. Defaults to "noether".
                Must be one of the following: "noether", "bootstrap".
            alternative (str):
                Alternative hypothesis. Defaults to "two_sided".
                Must be one of the following: "two_sided" (concordance index is not equal to 0.5),
                "greater" (concordance index is greater than 0.5), "less" (concordance index is less than 0.5).
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor: p-value of statistical test.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 64
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> estimate = torch.randn((n,))
            >>> cindex = ConcordanceIndex()
            >>> cindex(estimate, event, time)
            tensor(0.5337)
            >>> cindex.p_value() # default: Noether, two_sided
            tensor(0.7516)
            >>> cindex.p_value(method="bootstrap", alternative="greater")
            tensor(0.2620)

        """

        assert (
            hasattr(self, "cindex") and self.cindex is not None
        ), "Error: Please calculate the concordance index using `ConcordanceIndex()` before calling `p_value()`."

        if alternative not in ["less", "greater", "two_sided"]:
            raise ValueError(
                "'alternative' parameter must be one of ['less', 'greater', 'two_sided']."
            )

        if method == "noether":
            pvalue = self._p_value_noether(alternative)
        elif method == "bootstrap":
            pvalue = self._p_value_bootstrap(alternative, n_bootstraps)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'noether' or 'bootstrap'."
            )
        return pvalue

    def compare(
        self, other, method: str = "noether", n_bootstraps: int = 999
    ) -> torch.Tensor:
        """Compare two Concordance indices.

        This function compares two concordance indices computed on the
        same data with different risk scores. The statistical hypotheses are
        formulated as follows, null hypothesis: cindex1 = cindex2 and alternative
        hypothesis: cindex1 > cindex2.
        The statistical test is either a Student t-test for dependent samples or
        a two-sample bootstrap test. The Student t-test assumes that the concordance index is normally distributed
        and uses standard errors estimated with the Noether's method :cite:p:`Pencina2004`.

        Args:
            other (ConcordanceIndex):
                Another instance of the ConcordanceIndex class representing cindex2.
            method (str):
                Statistical test used to perform the hypothesis test. Defaults to "noether".
                Must be one of the following: "noether", "bootstrap".
            n_bootstraps (int):
                Number of bootstrap samples. Defaults to 999.
                Ignored if ``method`` is not "bootstrap".

        Returns:
            torch.Tensor: p-value of the statistical test.

        Examples:
            >>> _ = torch.manual_seed(42)
            >>> n = 64
            >>> time = torch.randint(low=5, high=250, size=(n,)).float()
            >>> event = torch.randint(low=0, high=2, size=(n,)).bool()
            >>> cindex1 = ConcordanceIndex()
            >>> cindex1(torch.randn((n,)), event, time)
            tensor(0.5337)
            >>> cindex2 = ConcordanceIndex()
            >>> cindex2(torch.randn((n,)), event, time)
            tensor(0.5047)
            >>> cindex1.compare(cindex2) # default: Noether
            tensor(0.4267)
            >>> cindex1.compare(cindex2, method = "bootstrap")
            tensor(0.3620)

        """

        assert isinstance(other, ConcordanceIndex)
        assert isinstance(method, str)

        assert (
            hasattr(self, "cindex") and self.cindex is not None
        ), "Error: Please calculate the concordance index using `ConcordanceIndex()` before calling `compare()`."

        # assert that the same data were used to compute the two c-index
        if torch.any(self.event != other.event) or torch.any(self.time != other.time):
            raise ValueError(
                "Mismatched survival data: 'time' and 'event' should be the same for both concordance index computations."
            )

        if self.tmax != other.tmax:
            raise ValueError(
                "Mismatched truncation time: 'tmax' should be the same for both concordance index computations."
            )

        if method == "noether":
            pvalue = self._compare_noether(other)
        elif method == "bootstrap":
            pvalue = self._compare_bootstrap(other, n_bootstraps)
        else:
            raise ValueError(
                f"Method {method} not implemented. Please choose either 'noether' or 'bootstrap'."
            )
        return pvalue

    def _confidence_interval_noether(
        self, alpha: float, alternative: str
    ) -> torch.Tensor:
        """Confidence interval of Concordance index assuming that the concordance index
        is normally distributed and using standard errors estimated using Noether's method.
        """

        alpha = alpha / 2 if alternative == "two_sided" else alpha

        cindex_se = self._concordance_index_se()

        if cindex_se > 0:
            ci = (
                -torch.distributions.normal.Normal(0, 1).icdf(
                    torch.tensor(alpha, device=self.cindex.device)
                )
                * cindex_se
            )
            lower = torch.max(
                torch.tensor(0.0, device=self.cindex.device), self.cindex - ci
            )
            upper = torch.min(
                torch.tensor(1.0, device=self.cindex.device), self.cindex + ci
            )

            if alternative == "less":
                lower = torch.tensor(0.0, device=self.cindex.device)
            elif alternative == "greater":
                upper = torch.tensor(1.0, device=self.cindex.device)
        else:
            raise ValueError(
                "The standard error of the concordance index must be a positive value."
            )

        return torch.stack([lower, upper], dim=0)

    # pylint: disable=invalid-name
    def _confidence_interval_conservative(
        self, alpha: float, alternative: str
    ) -> torch.Tensor:
        """Confidence interval of Concordance index assuming that the concordance index
        is normally distributed and using the conservative method.
        """
        alpha = alpha / 2 if alternative == "two_sided" else alpha

        N = torch.sum(self.weight)

        pc = (1 / (N * (N - 1))) * torch.sum(self.concordant)
        pd = (1 / (N * (N - 1))) * torch.sum(self.discordant)

        w = (
            (
                torch.distributions.normal.Normal(0, 1).icdf(
                    torch.tensor(alpha, device=self.cindex.device)
                )
                ** 2
            )
            * 2
        ) / (N * (pc + pd))

        ci = torch.sqrt(w**2 + 4 * w * self.cindex * (1 - self.cindex)) / (
            2 * (1 + w)
        )
        point = (w + 2 * self.cindex) / (2 * (1 + w))

        lower = point - ci
        upper = point + ci

        if alternative == "less":
            lower = torch.tensor(0.0, device=self.cindex.device)
        elif alternative == "greater":
            upper = torch.tensor(1.0, device=self.cindex.device)
        return torch.stack([lower, upper])

    def _confidence_interval_bootstrap(
        self, alpha: float, alternative: str, n_bootstraps: int
    ) -> torch.Tensor:
        """Bootstrap confidence interval of the Concordance index using
        Efron's percentile method.

        References:
            Efron, Bradley; Tibshirani, Robert J. (1993).
                An introduction to the bootstrap, New York: Chapman & Hall, software.
        """

        # c-index bootstraps given bootstrap distribution
        cindex_bootstrap = self._bootstrap_cindex(
            metric="confidence_interval", n_bootstraps=n_bootstraps
        )

        # obtain confidence interval
        if alternative == "two_sided":
            lower, upper = torch.quantile(
                cindex_bootstrap,
                torch.tensor([alpha / 2, 1 - alpha / 2], device=self.cindex.device),
            )
        elif alternative == "less":
            upper = torch.quantile(
                cindex_bootstrap, torch.tensor(1 - alpha, device=self.cindex.device)
            )
            lower = torch.tensor(0.0, device=self.cindex.device)
        elif alternative == "greater":
            lower = torch.quantile(
                cindex_bootstrap, torch.tensor(alpha, device=self.cindex.device)
            )
            upper = torch.tensor(1.0, device=self.cindex.device)

        return torch.stack([lower, upper])

    def _p_value_noether(self, alternative, null_value: float = 0.5) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the Concordance index
        assuming that the concordance index is normally distributed and using standard
        errors estimated with Noether's method.
        """

        cindex_se = self._concordance_index_se()

        # get p-value
        if cindex_se > 0:
            p = torch.distributions.normal.Normal(0, 1).cdf(
                (self.cindex - null_value) / cindex_se
            )
            if alternative == "two_sided":
                if self.cindex >= torch.tensor(0.5):
                    p = torch.tensor(1.0) - p
                p *= torch.tensor(2.0)
            elif alternative == "greater":
                p = torch.tensor(1.0) - p
        else:
            raise ValueError(
                "The standard error of concordance index must be a positive value."
            )

        return p

    def _p_value_bootstrap(self, alternative, n_bootstraps) -> torch.Tensor:
        """p-value for a one-sample hypothesis test of the Concordance Index using
        permutation of risk prediction to estimate sampling distribution under the null
        hypothesis.
        """

        # c-index bootstraps given null distribution cindex = 0.5
        cindex0 = self._bootstrap_cindex(metric="p_value", n_bootstraps=n_bootstraps)

        # Derive p-value
        p = (torch.tensor(1) + torch.sum(cindex0 <= self.cindex)) / torch.tensor(
            n_bootstraps + 1
        )
        if alternative == "two_sided":
            if self.cindex >= torch.tensor(0.5):
                p = torch.tensor(1.0) - p
            p *= torch.tensor(2.0)
            p = torch.min(
                torch.tensor(1.0, device=self.cindex.device), p
            )  # in case very small bootstrap sample size is used
        elif alternative == "greater":
            p = torch.tensor(1.0) - p

        return p

    # pylint: disable=W0212
    def _compare_noether(self, other):
        """Student t-test for dependent samples given Noether's standard error to
        compare two concordance indices.
        """

        # sample size
        N = sum(self.weight)

        # compute noether standard error
        cindex1_se = self._concordance_index_se()
        cindex2_se = other._concordance_index_se()

        # Suppress the specific warning
        warnings.filterwarnings(
            "ignore",
            message="Metric `SpearmanCorrcoef` will save all targets and predictions in the buffer. For large datasets, this may lead to large memory footprint.",
            category=UserWarning,
        )

        # compute spearman correlation between risk prediction
        corr = regression.SpearmanCorrCoef()(
            self.estimate.reshape(-1), other.estimate.reshape(-1)
        )

        # check for perfect positive monotonic relationship between two variables
        if 1 - torch.abs(corr) < 1e-15:
            return 1.0

        # compute t-stat
        t_stat = (self.cindex - other.cindex) / torch.sqrt(
            cindex1_se**2 + cindex2_se**2 - 2 * corr * cindex1_se * cindex2_se
        )

        # return p-value
        return torch.tensor(
            1 - stats.t.cdf(t_stat, df=N - 1),
            dtype=self.cindex.dtype,
            device=self.cindex.device,
        )  # student-t cdf not available on torch

    def _compare_bootstrap(self, other, n_bootstraps):
        """Bootstrap two-sample test to compare two concordance indices."""

        # c-index bootstraps given null hypothesis that cindex1
        # and cindex2 come from the same distribution
        cindex1_null = self._bootstrap_cindex(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )
        cindex2_null = self._bootstrap_cindex(
            metric="compare", other=other, n_bootstraps=n_bootstraps
        )

        # bootstrapped test statistics
        t_boot = cindex1_null - cindex2_null

        # observed test statistics
        t_obs = self.cindex - other.cindex

        # return p-value
        return torch.tensor(1) - (
            torch.tensor(1) + torch.sum(t_boot <= t_obs)
        ) / torch.tensor(n_bootstraps + 1)

    def _concordance_index_se(self):
        """Standard error of concordance index using Noether's method."""

        N = sum(self.weight)

        pc = (1 / (N * (N - 1))) * torch.sum(self.concordant)
        pd = (1 / (N * (N - 1))) * torch.sum(self.discordant)
        pcc = (1 / (N * (N - 1) * (N - 2))) * torch.sum(
            self.concordant * (self.concordant - 1)
        )
        pdd = (1 / (N * (N - 1) * (N - 2))) * torch.sum(
            self.discordant * (self.discordant - 1)
        )
        pcd = (1 / (N * (N - 1) * (N - 2))) * torch.sum(
            self.concordant * self.discordant
        )
        varp = (4 / (pc + pd) ** 4) * (
            pd**2 * pcc - 2 * pc * pd * pcd + pc**2 * pdd
        )

        return torch.sqrt(varp / N)

    def _bootstrap_cindex(
        self, metric: str, n_bootstraps: int, other=None
    ) -> torch.Tensor:
        """Compute bootstrap samples of the Concordance Index (C-index).

        Args:
            metric (str): Must be one of the following: "p_value", "confidence_interval", "compare".
                If "p_value", computes bootstrap samples of the concordance index given the sampling distribution
                under the null hypothesis (c-index = 0.5). If "confidence_interval", computes bootstrap
                samples of the c-index given the data distribution. If "compare", computes
                bootstrap samples of the c-index given the sampling distribution under the comparison test
                null hypothesis (c-index1 = cindex2).
            n_bootstraps (int): Number of bootstrap samples.
            other (optional, ConcordanceIndex):
                 Another instance of the ConcordanceIndex class representing cindex2.
                Only required for the ``metric`` is "compare".


        Returns:
            torch.Tensor: bootstrap samples of Concordance index.
        """

        # Initiate empty list to store concordance index
        cindexes = []

        # Get the bootstrap samples of concordance index
        for _ in range(n_bootstraps):
            if (
                metric == "p_value"
            ):  # bootstrap samples given null distribution (cindex = 0.5)
                estimate = copy.deepcopy(self.estimate)
                estimate = estimate[
                    torch.randperm(len(estimate)), :
                ]  # Shuffle estimate
                cindexes.append(
                    self(
                        estimate,
                        self.event,
                        self.time,
                        self.weight,
                        self.tmax,
                        instate=False,
                    )
                )  # Run Concordance index, without saving internal state
            elif (
                metric == "confidence_interval"
            ):  # bootstrap samples given data distribution
                index = torch.randint(
                    low=0,
                    high=len(self.event),
                    size=(
                        len(
                            self.event,
                        ),
                    ),
                )
                cindexes.append(
                    self(  # sample with replacement from sample
                        self.estimate[index, :][:, index],
                        self.event[index],
                        self.time[index],
                        self.weight,
                        self.tmax,
                        instate=False,
                    )
                )  # Run Concordance index, without saving internal state
            elif (
                metric == "compare"
            ):  # bootstrap samples given null distribution (cindex1 = cindex2)
                index = torch.randint(
                    low=0,
                    high=len(self.event) * 2,
                    size=(
                        len(
                            self.event,
                        ),
                    ),
                )
                estimate = torch.cat(
                    (
                        torch.cat((self.estimate, self.estimate), dim=1),
                        torch.cat((other.estimate, other.estimate), dim=1),
                    ),
                    dim=0,
                )  # create n_samples*2, n_samples*2 tensor
                cindexes.append(
                    self(  # sample with replacement from pooled sample
                        estimate[index, :][:, index],
                        torch.cat((self.event, other.event))[index],
                        torch.cat((self.time, other.time))[index],
                        torch.cat((self.weight, other.weight))[index],
                        self.tmax,
                        instate=False,
                    )
                )

        return torch.stack(cindexes)

    @staticmethod
    def _get_comparable_and_tied_time(
        event,
        time,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify comparable pairs and count tied time pairs.
        The function iterates through the sorted time points to identify comparable samples
        (those with the same time point) and count the number of tied time points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - comparable (torch.Tensor): Dictionary containing indices as keys
                and boolean masks as values, indicating which samples are comparable.
                - tied_time (torch.Tensor): Number of tied time points.

        Note:
        - 'comparable' dictionary maps indices to boolean masks, where True indicates
        that the corresponding sample is comparable to others at the same time point.
        - 'tied_time' counts the total number of tied time points.

        """
        # Number of samples
        n_samples = len(time)

        # Sort indices based on time
        order = torch.argsort(time)

        # Initialize dictionary to store comparable samples
        comparable = {}

        # Initialize count
        tied_time = 0

        # Initialize index for storing unique values
        i = 0

        # Iterate through the sorted time points
        while i < n_samples - 1:
            time_i = time[order[i]]

            # Find the range of samples with the same time point
            start = i + 1
            end = start
            while end < n_samples and time[order[end]] == time_i:
                end += 1

            # check for tied event times
            event_at_same_time = event[order[i:end]]
            censored_at_same_time = torch.logical_not(event_at_same_time)

            # Iterate through the sample with the same time point
            for j in range(i, end):
                # If event occurred at time
                if event[order[j]]:
                    # Create a boolean mask for comparability
                    mask = torch.zeros_like(time).bool()
                    mask[end:] = True

                    # Store the comparability mask for the event
                    mask[i:end] = censored_at_same_time
                    comparable[j] = mask
                    tied_time += censored_at_same_time.sum()
            i = end

        if len(comparable) == 0:
            raise ValueError("No comparable pairs, denominator is 0.")

        return comparable

    @staticmethod
    def _update_weight(
        time: torch.Tensor,
        weight: torch.Tensor,
        tmax: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Obtain subject-specific weight."""

        # If weight is not provided, use ones instead.
        weight_updated = torch.zeros_like(time)

        # If tmax is provided, truncate time after tmax (mask = False)
        masks = torch.ones_like(time).bool() if tmax is None else time < tmax
        weight_updated[masks] = (
            torch.tensor(1.0, dtype=weight_updated.dtype, device=time.device)
            if weight is None
            else weight[masks]
        )

        return weight_updated

    @staticmethod
    def _update_cindex_estimate(estimate: torch.Tensor) -> torch.Tensor:
        # Ensure estimate is (n_samples, n_samples) shape
        if estimate.ndim == 1:
            estimate = estimate.unsqueeze(1)

        if estimate.shape[1] == 1:
            estimate = estimate.expand((estimate.shape[0], estimate.shape[0]))

        return estimate


if __name__ == "__main__":
    import doctest

    # Run doctest
    results = doctest.testmod()
    if results.failed == 0:
        print("All tests passed.")
    else:
        print("Some doctests failed.")
        sys.exit(1)
