# Survival Models for Time-Varying Covariates

* **Author**: Mélodie Monod

In this document, we describe the mathematical formulation and assumptions behind the survival models' loss implemented in the `loss` module of `TorchSurv`, which can accommodate time-varying covariates.

Let individual $i$ have a time-to-event $X_i$ and a censoring time $D_i$. We observe  

$$
T_i = \min(X_i, D_i)
$$ 

along with the event indicator

$$
\delta_i = \mathbb{1}(X_i \le D_i),
$$ 

where $\delta_i = 1$ if the event is observed and $\delta_i = 0$ if the data is censored.  
Each individual also has covariates $\mathbf{x}_i(t)$ at multiple time points $t$, which may be high-dimensional.


### 1. Extended Cox Model 

The Extended Cox model loss is available through the function:

```python
from torchsurv.loss.cox import neg_partial_log_likelihood
```

The implementation only differs from the standard Cox model in that the `log_hz` argument must be two-dimensional, with rows corresponding to subjects and columns corresponding to the time points of each subject.


**Hazard function.**
The standard Cox proportional hazards model assumes a hazard function of the form:

$$
\lambda_i(t) = \lambda_0(t) \, \theta_i(t), \quad \text{with} \quad \theta_i(t) = \exp(\mathbf{x}_i(t)^\top \beta),
$$ 

where $\lambda_0(t)$ is the baseline hazard and $\theta_i(t)$ is the relative hazard at time $t$. 

**PyTorch NN output.**
In `TorchSurv`, the user may specify the relative hazard with a PyTorch neural network:

$$
    \theta_i(t) = f_\theta(\mathbf{x}_i(t)).
$$


**Loss function.** The loss function is defined as the negative partial log-likelihood:

$$
\text{npll} = - \sum_{i: \delta_i = 1} \left( \log \theta_i(T_i) - \log \sum_{j \in R(T_i)} \theta_j(T_i) \right),
$$ 
where $R(T_i)$ is the risk set at time $T_i$.

**Assumptions.**

- The function $f_\theta$ does not depend on time (i.e., recurrent neural networks or other time-dependent architectures cannot be used). In other words, time-varying covariates are allowed but not time-varying parameters.
- Proportional hazards: the relative hazard does not vary with time conditional on the covariates.  
- Independence between censoring and event times.  
- Correct specification of the covariate effect (linear in standard Cox, flexible in `TorchSurv` neural network).


### 2. Flexible Survival Model 

The Flexible Survival model loss is available through the function 
```python
from torchsurv.loss.survival import neg_log_likelihood
```

The argument `log_hz` argument must be two-dimensional, with rows corresponding to subjects and columns corresponding to the time points used in the trapezoidal approximation for the cumulative hazard.


**Hazard function.**
When no specific parametric form for the hazard function is assumed, `TorchSurv` allows the model to learn the hazard function $\lambda_i(t)$ directly from a neural network.  
In this framework, the network takes the covariates $\mathbf{x}_i$ (and optionally time $t$) as input and outputs an estimate of the instantaneous hazard rate $\lambda_i(t)$.

$$
\lambda_i(t) = f_{\lambda}(\mathbf{x}_i(t), t)
$$

**Loss function.** 
The loss function is defined as the negative log-likelihood given by:

$$
\text{nll} = - \sum_{i=1}^N \delta_i \log \lambda_i(T_i) + \sum_{i=1}^N \int_0^{T_i} \lambda_i(u) \, du,
$$

where the first term corresponds to the observed events and the second term integrates the predicted hazard over time to account for the survival component.

Since the integral $\int_0^{T_i} \lambda_i(u)\,du$ has no closed-form solution when $\lambda_i(t)$ is modeled by a neural network, it is numerically approximated in `TorchSurv` using the trapezoidal rule over a discretized time grid.  
This makes the flexible survival model the only loss function in `TorchSurv` that requires an approximation of the likelihood.

This model is particularly powerful when the true hazard does not follow a standard parametric form, such as Weibull or exponential. Indeed, if the neural network outputs hazards corresponding to these forms, this model will recover their respective log-likelihoods exactly.

**Assumptions.** 
- The hazard function $\lambda_i(t)$ is well-approximated by the neural network.  
- Censoring is independent of event times.  
- Observations are conditionally independent given the covariates.  
- Numerical integration (trapezoidal rule) is sufficiently accurate given the time discretization.


### FAQ

#### What are time-varying covariates?

Time-varying covariates are features or measurements of individuals that can change over time. For example, blood pressure, lab test results, or even sensor data may vary at different time points. Survival models can use this information to better estimate risk over time.

#### Can the Cox model handle time-varying covariates?

Yes, the extended Cox model can handle time-varying covariates. However, it assumes that the relative hazard function $\theta_i(t)$ does not vary due to time-dependent parameters. In practice, this means the function $f_\theta$ (or the regression coefficients $\beta$ in standard Cox) must remain fixed over time.


#### What type of neural networks (NNs) can be used with the Cox model for time-varying covariates?

In the extended Cox model implemented in `TorchSurv`, only neural networks that produce a fixed mapping from covariates to relative hazard can be used. This means that the network function $f_\theta$ must be independent of time; recurrent or other time-dependent architectures (e.g., RNNs, LSTMs, or temporal convolutional networks) violate this assumption. Essentially, each time-varying covariate vector $\mathbf{x}_i(t)$ is treated independently at its observed time points, and the same network parameters are applied across all times.


#### What is the difference between the Cox model and the flexible survival model?

The Cox model assumes proportional hazards and a neural network mapping covariates to relative hazard. The flexible survival model makes no assumptions about the hazard function form and allows a neural network to directly model the instantaneous hazard $\lambda_i(t)$ as a function of covariates and optionally time. This model requires numerical integration for the likelihood.


#### When should I use the flexible survival model instead of the Cox model?

The flexible survival model is recommended when the relationship between covariates and the hazard is complex or time-dependent, and you want to model it with a neural network. Unlike the Cox model, it does not assume proportional hazards and allows the hazard function to vary flexibly over time.

#### Can I use time-varying covariates with Weibull or Expoenntial model?

No. Parametric models like Weibull or Exponential assume a specific functional form for the hazard over time. Their hazard functions are already fully determined by the model parameters, so they cannot directly incorporate additional time-varying covariates.