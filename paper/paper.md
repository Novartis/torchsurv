---
title: 'TorchSurv: A Lightweight Package for Deep Survival Analysis'
tags:
  - Python package
  - Deep Survival Analysis
  - PyTorch
  - TorchSurv
authors:
  - name: Mélodie Monod
    orcid: 0000-0001-6448-2051
    affiliation: "1"
  - name: Peter Krusche
    affiliation: "1"
  - name: Qian Cao
    affiliation: "2"
  - name: Berkman Sahiner
    affiliation: "2"
  - name: Nicholas Petrick
    affiliation: "2"
  - name: David Ohlssen
    affiliation: "3"
  - name: Thibaud Coroller
    orcid: 0000-0001-7662-8724
    affiliation: "3"
  - name: Novartis Pharma AG, Switzerland
    index: 1
  - name: Center for Devices and Radiological Health, Food and Drug Administration, MD, USA
    index: 2
  - name: Novartis Pharmaceuticals Corporation, NJ, USA
    index: 3
date: 18 July 2024
bibliography: paper.bib
---

# Summary

`TorchSurv` (available on GitHub and PyPI) is a Python package that serves as a companion tool to perform deep survival modeling within the `PyTorch` environment [@paszke2019pytorch]. With its lightweight design, minimal input requirements, full `PyTorch` backend, and freedom from restrictive survival model parameterizations, `TorchSurv` facilitates efficient deep survival model implementation and is particularly beneficial for high-dimensional and complex input data scenarios.
`TorchSurv` has undergone rigorous testing on open source data and synthetically generated survival data that include edge cases. The package is comprehensively documented and contains illustrative examples. The latest documentation of `TorchSurv` can be found on [`TorchSurv`'s website](https://opensource.nibr.com/torchsurv/).

`TorchSurv` provides a user-friendly workflow for defining a survival model with parameters specified by a `PyTorch`-based (deep) neural network. At the core of `TorchSurv` lies its `PyTorch`-based calculation of log-likelihoods for prominent survival models, including the Cox proportional hazards model [@Cox1972] and the Weibull Accelerated Time Failure (AFT) model [@Carroll2003]. 
In survival analysis, each observation is associated with survival data denoted by $y$ (comprising the event indicator and the time-to-event or censoring) and covariates denoted by $x$. A survival model that is able to capture the complexity of the survival data $y$, is parametrized by parameters denoted by $\theta$. For instance, in the Cox proportional hazards model, the survival model parameters $\theta$ are the relative hazards. Within the `TorchSurv` framework, a `PyTorch`-based neural network is defined to act as a flexible function that takes the covariates $x$ as input and outputs the survival model parameters $\theta$. Estimation of the parameters $\theta$ is achieved via maximum likelihood estimation facilitated by backpropagation.
Additionally, `TorchSurv` offers evaluation metrics (the time-dependent Area Under the cure (AUC) under the Receiver operating characteristic curve (ROC), the Concordance index (C-index) and the Brier Score) to characterize the predictive performance of survival models.
Below is an overview of the workflow for model inference and evaluation with `TorchSurv`:

1. Initialize a `PyTorch`-based neural network that defines the function from the covariates to the survival model's parameters. In the context of the Cox relative hazards model for example, the parameters are the log relative hazards.
2. Initiate training: For each epoch on the training set,
    -  Draw survival data $y^{\text{train}}$ (i.e., event indicator and time-to-event or censoring) and covariates $x^{\text{train}}$ from the training set.
    - Obtain parameters $\theta^{\text{train}}$ based on drawn covariates $x^{\text{train}}$ using `PyTorch`-based neural network.
    - Calculate the loss given survival data $y^{\text{train}}$ and parameters $\theta^{\text{train}}$ using `TorchSurv`'s loss function. In the context of the Cox relative hazards model for example, the loss function is equal to the negative of the log partial likelihood.
    - Utilize backpropagation to update parameters $\theta^{\text{train}}$.
3. Obtain parameters $\theta^{\text{test}}$ based on covariates from the test set $x^{\text{test}}$ using the trained `PyTorch`-based neural network.
4. Evaluate the predictive performance of the model using `TorchSurv`'s evaluation metric functions (e.g., C-index) given parameters $\theta^{\text{test}}$ and survival data from the test set $y^{\text{test}}$.



# Statement of need

Survival analysis plays a crucial role in various domains, such as medicine and engineering. Deep learning offers promising avenues for building complex survival models. However, existing libraries often restrict users to predefined parameter forms and limit seamless integration with `PyTorch`. 

Table 1 compares the functionalities of `TorchSurv` with those of 
`auton-survival` [@nagpal2022auton], 
`pycox` [@Kvamme2019pycox], 
`torchlife` [@torchlifeAbeywardana],
`scikit-survival` [@polsterl2020scikit],
`lifelines` [@davidson2019lifelines], and 
`deepsurv` [@katzman2018deepsurv].
While several libraries offer survival modelling functionalities, as shown in Table 1, no existing library provides the flexibility to use a custom `PyTorch`-based neural network to define the survival model parameters $\theta$ given a set of covariates $x$. In existing libraries, users are limited to specific forms to define $\theta$ (e.g., linear function of covariates) and the log-likelihood functions available cannot be leveraged because they do not allow for seamless integration with `PyTorch`.
Specifically, the limitations on the log-likelihood functions include protected functions, specialized input requirements (format or class type), and reliance on external libraries like `NumPy` or `Pandas`. Dependence on external libraries hinders automatic gradient calculation within `PyTorch`. Additionally, the implementation of likelihood functions instead of log-likelihood functions, as done by some packages, introduces potential numerical instability.
With respect to the evaluation metrics, `scikit-survival` stands out as a comprehensive library. However, it lacks certain desirable features, including confidence intervals and comparison of the evaluation metric between two different models, and it is implemented with `NumPy`. 
Our package, `TorchSurv`, is specifically designed for use in Python, but we also provide a comparative analysis of its functionalities with popular `R` packages for survival analysis in Table 2. `R` packages do not make log-likelihood functions readily accessible and restrict users to specific forms to define  $\theta$. However, `R` has extensive libraries for evaluation metrics, such as the `RiskRegression` library [@riskRegressionpackage]. `TorchSurv` offers a comparable range of evaluation metrics, ensuring comprehensive model evaluation regardless of the chosen programming environment. 

The outputs of both the log-likelihood functions and the evaluation metrics functions have undergone thorough comparison with benchmarks generated with Python packages and R packages on open-source data and synthetic data. High agreement between the outputs is consistently observed, providing users with confidence in the accuracy and reliability of `TorchSurv`'s functionalities. The comparison is summarized in the [`TorchSurv`'s website](https://opensource.nibr.com/torchsurv/benchmarks.html).

# Functionality

## Loss functions

**Cox loss function.** The Cox loss function is defined as the negative of the Cox proportional hazards model's partial log-likelihood [@Cox1972]. The function requires the subject-specific log relative hazards and the survival data (i.e., event indicator and time-to-event or censoring). The log relative hazards should be obtained from a `PyTorch`-based model pre-specified by the user. In case of ties in the event times, the user can choose between the Breslow [@Breslow1975] and the Efron method [@Efron1977] to approximate the Cox partial log likelihood. We illustrate the use of the Cox loss function for a pseudo training loop in the code snippet below.

```python
from torchsurv.loss import cox
my_model = MyPyTorchModel() # PyTorch model for log hazards (1 output)
for data in dataloader:
    x, event, time = data
    log_hzs = my_model(x)  # torch.Size([64, 1]), if batch size is 64 
    loss = cox.neg_partial_log_likelihood(log_hzs, event, time)  
    loss.backward()  # native torch backend
```

**Weibull loss function.** The Weibull loss function is defined as the negative of the Weibull AFT's log-likelihood [@Carroll2003]. The function requires the subject-specific log parameters of the Weibull distribution (i.e., the log scale and the log shape) and the survival data. The log parameters of the Weibull distribution should be obtained from a `PyTorch`-based model pre-specified by the user. We illustrate the use of the Weibull loss function for a pseudo training loop in the code snippet below.

```python
from torchsurv.loss import weibull
my_model = MyPyTorchModel() # PyTorch model for log parameters (2 outputs)
for data in dataloader:
    x, event, time = data
    log_params = my_model(x)  # torch.Size([64, 2]), if batch size is 64 
    loss = weibull.neg_log_likelihood(log_params, event, time)  
    loss.backward()  # native torch backend
```

**Momentum**
 When training a model with a large file, the batch size is greatly limited by computational resources. This impacts the stability of model optimization, especially when rank-based loss is used. Inspired from MoCO \citep{he2020momentum}, we implemented a momentum loss that decouples batch size from survival loss, increasing the effective batch size and allowing robust train of a model, even when using a very limited batch size (e.g., $batch_{size} \leq 16$). We illustrate the use of momentum for a pseudo training loop in the code
snippet below.

```python
from torchsurv.loss import Momentum
my_model = MyPyTorchModel() # PyTorch model for log hazards (1 output)
my_loss = cox.neg_partial_log_likelihood  # any torchsurv loss
momentum = Momentum(backbone=my_model, loss=my_loss) 

for data in dataloader:
    x, event, time = data
    loss = model_momentum(x, event, time)  # torch.Size([64, 1])
    loss.backward()  # native torch backend

# Inference is computed with target network (k)
log_hzs = model_momentum.infer(x)  # torch.Size([64, 1])
```

## Evaluation Metrics Functions

The `TorchSurv` package offers a comprehensive set of metrics to evaluate the predictive performance of survival models, including the AUC, C-index, and Brier score. The inputs of the evaluation metrics functions are the individual risk score estimated on the test set and the survival data on the test set. The risk score measures the risk (or a proxy thereof) that a subject has an event.  We provide definitions for each metric and demonstrate their use through illustrative code snippets. 

**AUC.** The AUC measures the discriminatory capacity of a model at a given time $t$, i.e., the model’s ability to provide a reliable ranking of times-to-event based on estimated individual risk scores [@Heagerty2005;@Uno2007;@Blanche2013]. 

```python
from torchsurv.metrics import Auc
auc = Auc()
auc(log_hzs, event, time)  # AUC at each time 
auc(log_hzs, event, time, new_time=torch.tensor(10.))  # AUC at time 10
```

**C-index.** The C-index is a generalization of the AUC that represents the assessment of the discriminatory capacity of the model over time [@Harrell1996;@Uno_2011].

```python
from torchsurv.metrics import ConcordanceIndex
cindex = ConcordanceIndex()
cindex(log_hzs, event, time)  # c-index
```

**Brier Score.** The Brier score evaluates the accuracy of a model at a given time $t$. It represents the average squared distance between the observed survival status and the predicted survival probability [@Graf_1999]. The Brier score cannot be obtained for the Cox model because the survival function is not available, but it can be obtained for the Weibull model.

```python
from torchsurv.metrics import Brier
surv = survival_function(log_params, time) 
brier = Brier()
brier(surv, event, time)  # Brier score at each time
brier.integral() # integrated brier score
```


**Additional features.** In \texttt{TorchSurv}, the evaluation metrics can be obtained for time-dependent and time-independent risk scores (e.g., for proportional and non-proportional hazards). Additionally subjects can be optionally weighted (e.g., by the inverse probability of censoring weighting (IPCW)).
Lastly, functionalities including the confidence interval, one-sample hypothesis test to determine whether the metric is better than that of a random predictor, and two-sample hypothesis test to compare two evaluation metrics between  different models are implemented. For the hypothesis tests, the significance level, typically referred to as $\alpha$, can be modified as needed. The following code snippet exemplifies the aforementioned functionalities for the C-index.

```python
cindex.confidence_interval()  # CI, default alpha = .05
cindex.p_value(alternative='greater')  # pvalue, H0: c = 0.5, HA: c > 0.5
cindex.compare(cindex_other)  # pvalue, H0: c1 = c2, HA: c1 > c2
```

# Conflicts of interest

MM, PK, DO and TC are employees and stockholders of Novartis, a global pharmaceutical company.


# References
