# A statistical introduction

<!-- ## Predictive accuracy evaluation metrics for survival model -->

* **Author**: Mélodie Monod

## Introduction

The evaluation metrics for assessing the predictive performance of a model depend greatly on the type of response variable. For instance, mean squared error (MSE) or mean absolute error (MAE) are commonly used for continuous response data. In contrast, for binary data, metrics such as true positive rate (TPR) and true negative rate (TNR) are utilized. This document presents evaluation metrics specifically designed for survival data, where the response variable is the time to an event.

## Evaluation metrics under binary response

To understand the evaluation metrics for time-to-event data, it is helpful to start with a review of the evaluation metrics used for binary outcomes, as the former are extensions of the latter.

Assume we have a binary response $Y_i \in \{0,1\}$, for any individual $i$. The model is a probabilistic classifier that outputs a score $\pi_i \in [0,1]$, which is an estimate of the probability $p(Y_i = 1)$.

The predicted response for individual $i$, denoted $\hat{Y}_i$, is obtained by comparing the score to a threshold $c$,

$$
\hat{Y}_i  = 
\begin{cases}
    1, \text{ if } \pi_i > c \\
    0, \text{ if } \pi_i \leq c. \\
\end{cases}
$$

Two key quantities are defined to evaluate the accuracy of the score: sensitivity and specificity, which are defined as

$$
\text{Sensitivity}(c) = p(\hat{Y}_i = 1 | Y_i = 1) = p(\pi_i > c | Y_i = 1),\\
\text{Specificity}(c) = p(\hat{Y}_i = 0 | Y_i = 0) = p(\pi_i \leq c | Y_i = 0).
$$

Note that the sensitivity is also referred to as the True Positive Rate (TPR), and the specificity is also known as the True Negative Rate (TNR), which equals 1 − False Positive Rate (FPR).

To visualize the predictive performance of the probabilistic classifier, the Receiver Operating Characteristic (ROC) curve plots the FPR on the x-axis and the TPR on the y-axis for all values of $c$. The Area Under the ROC Curve (AUC) summarizes the predictive performance across all values of $c$. It is given by

$$
\text{AUC} = \int_0^1 TPR(FPR(c)) dFPR(c).
$$

It can be shown that the AUC is equal to $p(\pi_i > \pi_j|Y_i = 1, Y_j = 0)$. This is the probability that, for a comparable pair, the individual without the event has a lower score than the individual with the event. This probability is also referred to as the C-index (denoted by C). In the binary context, the AUC is equal to the C-index.

<details>
<summary> Proof of the AUC's probabilistic interpretation </summary>
<br>

Let
$$
y(c) = \text{TPR}(c) = p(\pi_i > c | Y_i = 1),\\
x(c) = \text{FPR}(c) = p(\pi_i > c | Y_i = 0).
$$

Let us denote $f_1(c) = p(\pi_i = c | Y_i = 1)$ and $F_1(c) = p(\pi_i \leq c | Y_i = 1)$ and similarly $f_0(c) = p(\pi_i = c | Y_i = 0)$ and $F_0(c) = p(\pi_i \leq c | Y_i = 0)$. Notice that $y(c) = 1 - F_1(c)$ and $x(c) = 1 - F_0(c)$. 

The AUC is defined as
$$
\text{AUC} = \int_0^1 y(x(c)) dx(c).
$$

We use a change of variable $c = x(c)$. We have $\frac{dx}{dc}(c) = x'(c)$ and the limits become $x^{-1}(1) = - \infty$ and $x^{-1}(0) = \infty$. Therefore

$$
\begin{align*}
\text{AUC} &= \int_{\infty}^{-\infty} y(c) x'(c) dc \\
&= \int_{\infty}^{-\infty} (1- F_1(c)) (-f_0(c)) dc \\
&= \int_{-\infty}^\infty (1- F_1(c)) f_0(c) dc \\
&= \int_{-\infty}^\infty p(\pi_i > c | Y_i = 1) p(\pi_j = c | Y_j = 0) dc \\
&= p(\pi_i > \pi_j |  Y_i = 1, Y_j = 0)  \\
\end{align*}
$$
</details>

## Evaluation metrics under time-to-event response

Let's assume that individual $i$ has a time-to-event $X_i$ and a time-to-censoring $D_i$. We observe either the time to event or censoring, $T_i = \min(X_i, D_i)$, along with the event indicator $\delta_i = 1(X_i \leq D_i)$. A probabilistic survival model outputs a score $q_i(t) \in \mathbb{R}$, which estimates the event risk at time $t$  for individual $i$.


### Extension of sensitivity and specificity for time-to-event response

A time-to-event outcome can be viewed as a time-varying binary outcome using the counting process representation:

$$
N_i(t) = 1(X_i \leq t).
$$


The risk score chosen can either predicts prevalence or incidence (Heagerty and Zheng, 2005). For example, the cumulative hazard can be seen as a measure of prevalence because it measures the cumulative risk, whereas the instantaneous hazard can be seen as a measure of incidence as it measures the risk of an event in a very short, infinitesimally small time. Consequently, there are two different types of sensitivity depending on the chosen risk score.

#### Cumulative sensitivity

If the predicted classification of $i$ is based on a risk score measuring prevalence, it is defined as:

$$
\hat{N}_i(t) = 
\begin{cases}
    1, \text{ if } q_i(t) > c \\
    0, \text{ if } q_i(t) \leq c. \\
\end{cases}
$$

In this case, the sensitivity is referred to as cumulative sensitivity. It is the probability that a model correctly predicts that individual $i$ experiences an event before or at time $t$ and is defined as:

$$
\text{sensitivity}^{\mathbb{C}}(c,t) = p(\hat{N}_i(t) = 1 | {N}_i(t) =1 ) = p(q_i(t) > c| X_i \leq t )
$$

#### Incident sensitivity

If the prediction classification of $i$ is based on a risk score measuring incidence, it is defined as:

$$
d\hat{N}_i(t)  = 
\begin{cases}
    1, \text{ if } q_i(t) > c \\
    0, \text{ if } q_i(t) \leq c. \\
\end{cases}
$$

In this case, the sensitivity is referred to as incident sensitivity. It is the probability that a model correctly predicts that individual $i$ experiences an event at time $t$ and is defined as:

$$
\text{sensitivity}^{\mathbb{I}}(c,t) = p(d\hat{N}_i(t) = 1 | d{N}_i(t) =1 ) = p(q_i(t) > c| X_i = t )
$$

#### Dynamic specificity

Extending the concept from the binary response setting, specificity in the time-to-event response context is called dynamic specificity. It is the probability that a model correctly predicts that individual $i$ does not experience an event before or at time $t$. It is defined as:

$$
\text{specificity}^{\mathbb{D}}(c,t) = p(\hat{N}_i(t) = 0 | {N}_i(t) =0 ) = p(q_i(t) \leq c| X_i > t )
$$

Note that there is no distinction between prevalence and incidence in the context of specificity. This is because if we condition on $dN_i(t) = 0$, it implies either the individual had an event before $t$ and is thus excluded, or the individual has an event after $t$ which is equivalent to conditioning on $N_i(t) = 0$.

### Extension of the AUC for time-to-event response

#### Cumulative/dynamic AUC
If the risk score measures prevalence, a cumulative/dynamic ROC curve can be defined on the cumulative sensitivity against 1 - dynamic specificity. The area under the cumulative/dynamic ROC curve is called the cumulative/dynamic AUC (AUC C/D). It represents the probability that the model correctly identifies which of two comparable samples experiences an event before or at time $t$ based on their predicted risk scores. It is defined as:

$$
\begin{align*}
AUC^{\mathbb{C}/\mathbb{D}}(t) &= p(q_i(t)>q_j(t)|N_i(t) = 1, N_j(t) = 0) \\
&= p(q_i(t)>q_j(t)|X_i \leq t, X_j >t) 
\end{align*}
$$

The proof of the probabilitic interpretation of $AUC^{\mathbb{C}/\mathbb{D}}$ is similar to that in the binary response context.

#### Incident/dynamic AUC
If the risk score measures incidence, a incident/dynamic ROC curve can be defined on the incident sensitivity against 1 - dynamic specificity. The area under the incident/dynamic ROC curve is called the incident/dynamic AUC (AUC I/D). It represents the probability that the model correctly identifies which of two comparable samples experiences an event at time $t$ based on their predicted risk scores. It is defined as:

$$
\begin{align*}
\text{AUC}^{\mathbb{I}/\mathbb{D}}(t) &= p(q_i(t)>q_j(t)|dN_i(t) = 1, N_j(t) = 0) \\
& = p(q_i(t)>q_j(t)|X_i = t, X_j >t) 
\end{align*}
$$

The proof of the probabilitic interpretation of $AUC^{\mathbb{I}/\mathbb{D}}$ is similar to that in the binary response context.

### Extension of the C-index for time-to-event response

The C-index is the probability that a model correctly predict which of two comparable samples experiences an event first based on their predicted risk scores. 

$$
C = p(q_i(X_i) > q_j(X_i) | X_i < X_j)
$$

It can be shown that the C-index is related to the incidence/dynamic AUC, specifically $C = \int_t \text{AUC}^{\mathbb{I}/\mathbb{D}}(t) g(t) dt$ where $g(t) = 2f(t)S(t)$, $f(t)$ is the time-to-event densitity and $S(t)$ is the survival function.

<details>
<summary> Proof of the relation between the C-index and the AUC I/D </summary>
<br>


$$
\begin{align}
C &= p(q_i(X_i) > q_j(X_i) | X_i < X_j) \\
&= \frac{p(q_i(X_i) > q_j(X_i) , X_i < X_j)}{p(X_i < X_j)} \\
&= 2 p(q_i(X_i) > q_j(X_i) , X_i < X_j) \\
&= 2 \int_t p(q_i(X_i) > q_j(X_i) , X_i = t, X_j > t) dt \\
&= 2 \int_t p(q_i(X_i) > q_j(X_i) | X_i = t, X_j > t) p(X_i = t, X_j > t) dt \\
&= 2 \int_t \text{AUC}^{\mathbb{I}/\mathbb{D}}(t) p(X_i = t) p(X_j > t) dt \\
&= 2 \int_t \text{AUC}^{\mathbb{I}/\mathbb{D}}(t) f(t) S(t) dt \\
\end{align}
$$
where in the third line we used the fact that $p(X_i < X_j) = 1/2$ by independence, and in the sixth line, we again relied on the independence of the response.
</details>



## Discussion 

### How to estimate the AUC and C-index?
In other documentation, we presented the [estimators for the AUC]((../_autosummary/torchsurv.metrics.auc.html)) and the [estimators for the C-index](../_autosummary/torchsurv.metrics.cindex.html). In particular, these estimator needs to account for censored time-to-event data.

### What risk score should be used?
As explained previously, the risk score can either measure incidence or prevalence. An obvious candidate for measuring incidence is the instantaneous hazard, while the cumulative hazard is suitable for measuring prevalence. The choice of risk score depends on the question of interest and what the model should be good at predicting. For example, if the question of interest is how many deaths occur by time $t$, the risk score chosen should measure prevalence, and the evaluation metrics should be the AUC C/D.

### What evaluation metric should be used?

Choosing between AUC I/D and AUC C/D depends on the type of risk score, as previously explained. However, the question remains whether to use the AUC or the C-index. The AUCs are time-dependent and provide a measure of predictive performance at specific time points. In contrast, the concordance index (C-index) offers a global assessment of a fitted survival model over the entire observational period. It is recommended to use AUC instead of the concordance index for time-dependent predictions (e.g., 10-year mortality), as AUC is proper in this context, while the concordance index is not (Blanche et al., 2018).

Below, we summarize the pros and cons of the C-index, AUC I/D, and AUC C/D (Lambert and Chevret, 2016). 

|  | C-index | AUC cumulative/dynamic| AUC incident/dynamic|
|----------|----------|----------|----------|
| Pros | Global measure that does not depend on time | More clinically relevant, where one wishes to predict individuals who will have failed by time $t$. <br> Proper for time-dependent prediction. | Integral has probabilistic interpretation of the C-index. <br> Natural companion for hazard models. <br>Proper for time-dependent prediction. |
| Cons | Not proper for time-dependent prediction (e.g., 10-years mortality). | Integral does not have a nice probabilistic interpretation. <br> Requires a cumulative risk score (e.g., 1-survival function or cumulative hazard). | Require the exact event times to be recorded. <br> All subjects that failed before $t$ are ignored.|

### What are the limits of the AUC and the C-index?

There are three main limits of the AUC and C-index (Hartman et al., 2023):

* **Ties in predicted risk score (C-index)**:
A model that assigns the same risk score to individuals with extremely different survival experiences is not performing well, but the C-Index does not detect this inadequacy. 
On the other hand, a model that assigns the same risk score to individuals with very similar survival experiences may be appropriately capturing the similarities in underlying risk. 
While these two scenarios have very different interpretations, they are treated as the same in the C-Index calculation, since each of these pairs contributes a score of 0.5.

* **Time dependence (AUC and C-index)**:
Time-to-event outcome is dichotomized at each time point, which potentially generates many comparable pairs that are difficult to discriminate and are not clinically meaningful. 
For example, if a individual experiences the event of interest on day t of the study, and another individual, with similar underlying risk, experiences the event shortly afterwards on day t + 1, then these two individuals would be deemed comparable according to the time-dependent C-Index definition.

* **Discrimination (AUC and C-index)**:
The concordance index disregards the actual values of predicted risk scores – it is a ranking metric – and is unable to tell anything about calibration.


## The Brier-Score as an alternative to the AUC and C-index

### Brier-score and Integrated Brier Score

Suppose that the probabilistic survival model can output for any individual $i$ an estimate of the survival function $P(X_i > t)$, denoted by $\xi_i(t) \in [0,1]$. The Time-dependent Brier Score (BS) and integrated Brier Score (Graf et al., 1999) are defined as 

$$
\text{BS}(t) = \mathbb{E}\big[\big(1(X > t) - \xi(t)\big)^2\big],
$$

$$
\text{IBS} = \int_0^{t_{\text{max}}} \text{BS}(t) dW(t),
$$
where $W(t) = t / t_{\text{max}}$ and $t_{\text{max}}$ is the time-point until which we want to restrict the average.

### Discrimination (AUC, C-index) vs. calibration (Brier score) measure

The Brier score assesses both calibration and discrimination and serves as an alternative to the C-index and AUC. However, it is limited to models capable of estimating a survival function, thereby excluding models like the Cox proportional hazards model. Below, we outline the strengths and weaknesses of the C-index and AUC compared to the Brier score.

|  | C-index and AUC | Brier-Score |
|----------|----------|----------|
| Pros  | Extremely popular. <br> Simple to explain to clinical audience. <br> User-friendly scale (from 0.5 to 1).|Reflects both calibration and discrimination. <br>Can support the conclusions from a graphical calibration curve. |
| Cons   | Already discussed.     | Only applicable for models that can estimate a survival function.<br> Inadequate for very rare (or very frequent) events.<br>Less interpretable by clinicians.|

## Conclusion in practice

A meta-analysis by Zhou et al. (2022) recorded the proportion of evaluation metrics for survival models used from 2010 to 2021 in prominent journals such as "Annals of Statistics," "Biometrika," "Journal of the American Statistical Association," "Journal of the Royal Statistical Society, Series B," "Statistics in Medicine," "Artificial Intelligence in Medicine," and "Lifetime Data Analysis." This analysis indicates that the use of the C-index has been steadily increasing and has recently become a dominant predictive measure. In 2021, the C-index was used in more than 75% of the readings.
 
 
## Reference
 * Heagerty, P. J., & Zheng, Y. (2005). Survival Model Predictive Accuracy and ROC Curves. In Biometrics (Vol. 61, Issue 1, pp. 92–105). Oxford University Press (OUP). https://doi.org/10.1111/j.0006-341x.2005.030814.x
 * Lambert, J., & Chevret, S. (2016). Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves. In Statistical Methods in Medical Research (Vol. 25, Issue 5, pp. 2088–2102). SAGE Publications
 * Hartman, N., Kim, S., He, K., & Kalbfleisch, J. D. (2023). Pitfalls of the concordance index for survival outcomes. In Statistics in Medicine (Vol. 42, Issue 13, pp. 2179–2190). Wiley. https://doi.org/10.1002/sim.9717
 * Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999). Assessment and comparison of prognostic classification schemes for survival data. In Statistics in Medicine (Vol. 18, Issues 17–18, pp. 2529–2545). Wiley. https://doi.org/10.1002/(sici)1097-0258(19990915/30)18:17/18<2529::aid-sim274>3.0.co;2-5
* Hanpu Zhou; Hong Wang; Sizheng Wang; Yi Zou (2022). SurvMetrics: An R package for Predictive Evaluation Metrics in Survival Analysis. R Journal . Dec2022, Vol. 14 Issue 4, p1-12. 12p.
* Paul Blanche, Michael W Kattan, and Thomas A Gerds. The c-index is not proper for the evaluation of t-year predicted risks. Biostatistics, 20(2):347–357, February 2018.
