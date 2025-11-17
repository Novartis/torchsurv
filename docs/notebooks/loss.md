# Survival Model Losses

* **Author**: Mélodie Monod

In this document, we describe the mathematical formulation and assumptions behind the survival models' loss implemented in the `loss` module of `TorchSurv`.

Let individual $i$ have a time-to-event $X_i$ and a censoring time $D_i$. We observe  

$$
T_i = \min(X_i, D_i)
$$ 

along with the event indicator

$$
\delta_i = \mathbb{1}(X_i \le D_i),
$$ 

where $\delta_i = 1$ if the event is observed and $\delta_i = 0$ if the data is censored.  
Each individual also has covariates $\mathbf{x}_i$, which may be high-dimensional (e.g., an image or a vector of clinical features).


### 1. Cox Model 

The Cox model loss is available through the function 
```python
from torchsurv.loss.cox import neg_partial_log_likelihood
```

**Hazard function.**
The standard Cox proportional hazards model assumes a hazard function of the form:

$$
\lambda_i(t) = \lambda_0(t) \, \theta_i, \quad \text{with} \quad \theta_i = \exp(\mathbf{x}_i^\top \beta),
$$ 

where $\lambda_0(t)$ is the baseline hazard and $\theta_i$ is the relative hazard. This is the model usually implemented in survival packages, like [`lifelines`'s CoxPHFitter](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) 

**PyTorch NN output.**
In `TorchSurv`, the user may specify the relative hazard with a PyTorch neural network:

$$
    \theta_i = f_\theta(\mathbf{x}_i).
$$


**Loss function.** The loss function is defined as the negative partial log-likelihood:

$$
\text{npll} = - \sum_{i: \delta_i = 1} \left( \log \theta_i - \log \sum_{j \in R(T_i)} \theta_j \right),
$$ 
where $R(T_i)$ is the risk set at time $T_i$.

**Assumptions.**

- Proportional hazards: the relative hazard $\theta_i$ does not vary with time.  
- Independence between censoring and event times.  
- Correct specification of the covariate effect (linear in standard Cox, flexible in `TorchSurv` neural network).



### 2. Weibull model

The Weibull model loss is available through the function 
```python
from torchsurv.loss.weibull import neg_log_likelihood
```

**Hazard function.**
The standard Weibull hazard is:

$$
h_i(t) = \frac{\rho_i}{\lambda_i} \left(\frac{t}{\lambda_i}\right)^{\rho_i - 1},
$$  

with shape $\rho$ and scale $\lambda$. In traditional implementations, like [`lifelines`'s WeibullAFTFitter](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html), a linear form on the covariates is assumed on the log shape and log scale.  


**PyTorch NN output.**
In `TorchSurv`, users can model the log shape and log scale using neural networks 

$$
\log \rho_i = f_\rho(\mathbf{x}_i), \quad \log \lambda_i = f_\lambda(\mathbf{x}_i).
$$



**Loss function.** The loss function is the negative log-likelihood:

$$
\text{nll} = - \sum_{i: \delta_i = 1} \log h_i(T_i) + \sum_{i=1}^N H_i(T_i),
$$  

where $H_i(T_i)$ is the cumulative hazard function that can be obtained in closed-form.

**Assumptions.** 

- Correct specification of the Weibull form (or ability of the network to capture deviations).  
- Independent censoring.  
- Covariates correctly affect the shape and scale parameters.

### 3. Exponential model

The exponential model is a special case of the Weibull with $\rho = 1$. The hazard is constant over time:

$$
h_i(t) = \frac{1}{\lambda_i}.
$$


The loss is defined similarly as the negative log-likelihood.


### 4. Flexible Survival model

The Flexible Survival model loss is available through the function 
```python
from torchsurv.loss.survival import neg_log_likelihood
```


**Hazard function.**
When no specific parametric form for the hazard function is assumed, `TorchSurv` allows the model to learn the hazard function $\lambda_i(t)$ directly from a neural network.  
In this framework, the network takes the covariates $\mathbf{x}_i$ (and optionally time $t$) as input and outputs an estimate of the instantaneous hazard rate $\lambda_i(t)$.

$$
\lambda_i(t) = f_{\lambda}(\mathbf{x}_i, t)
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


### FAQ: Choosing the Right Survival Model

#### When should I use the Cox model?

**Use the Cox model if:**
- You expect the hazard ratios between subjects to be constant over time.  
- You have right-censored data with independent censoring.  
- You prefer interpretability and simplicity.  
- You want to use either a linear predictor or a neural network to estimate relative risk.  

**Avoid if:** you suspect non-proportional hazards or time-varying effects.


#### When should I use the Weibull model?

**Use the Weibull model if:**
- You believe the hazard either increases or decreases monotonically with time.  
- You want to estimate both the shape ($\rho_i$) and scale ($\lambda_i$) parameters, potentially using neural networks.  
- You prefer a model that has a closed-form log-likelihood (no numerical approximation).

**Avoid if:** you suspect non-monotonic hazard shapes (e.g., U-shaped or multimodal risks).


#### When should I use the Exponential model?

**Use the Exponential model if:**
- You expect the event rate to be roughly constant through time.  
- You need a simple baseline model for benchmarking.  
- You value interpretability over flexibility.  

**Avoid if:** the hazard clearly changes with time or shows time-varying risk patterns.


#### When should I use the Flexible Survival model?

Use the **Flexible Survival model** when you do not want to impose any parametric assumption on the hazard function. Here, the hazard $\lambda_i(t)$ is learned directly from a neural network, which can capture complex, time-varying patterns.   Because this model has no closed form for the cumulative hazard, `TorchSurv` uses the trapezoidal rule to approximate the log-likelihood numerically.

**Use the Flexible Survival model if:**
- You suspect that the hazard changes in a complex, nonlinear way over time.  
- You have sufficient data and computational resources to fit a neural network-based hazard.  
- You accept a small numerical approximation error due to trapezoidal integration.  
- You need maximum flexibility and do not wish to assume a specific hazard form.  

**Avoid if:** interpretability or analytical tractability is a priority, or your dataset is small.



### Summary Table

| Model | Hazard Form | Time-varying Risk | Closed-form Likelihood | Best Used When... |
|--------|--------------|-------------------|------------------------|-------------------|
| **Cox** | $\lambda_i(t) = \lambda_0(t)\exp(f_{\theta}(\mathbf{x}_i))$ | ✗ (proportional only) | ✓  | You expect proportional hazards |
| **Weibull** | Parametric (monotonic) | ✗ | ✓ | You expect monotonic hazard shape |
| **Exponential** | Constant | ✗ | ✓ | You expect constant risk over time |
| **Flexible Survival** | Neural network output | ✓ | ✗ (numerical approximation) | You need full flexibility, no parametric form |
