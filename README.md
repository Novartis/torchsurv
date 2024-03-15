# Survival analysis made easy

`torchsurv` is a statistical package that serves as a companion tool for survival modeling on PyTorch. Its core functionalities include the computation of common survival models’ log-likelihood and predictive performance metrics. `torchsurv` requires minimal input specifications and does not impose parameter forms allowing to rapidly define, evaluate, and optimize deep survival models.

## TL;DR

Our idea is to **keep things simple**. You are free to use any model architecture you want! Our code has 100% PyTorch backend and behaves like any other functions (losses or metrics) you may be familiar with.

Our functions are designed to support you, not to make you jump through hoops. Here's a pseudo code illustrating how easy is it to use `torchsurv` to fit and evaluate a Cox proportional hazards model:

```python
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

# Pseudo training loop
for data in dataloader:
    x, event, time = data
    estimate = model(x)  # shape = torch.Size([64, 1]), if batch size is 64
    loss = cox.neg_partial_log_likelihood(estimate, event, time)  
    loss.backward()  # native torch backend

# You can check model performance using our evaluation metrics, e.g, the concordance index with
cindex = ConcordanceIndex()
cindex(estimate, event, time)  
cindex.p_value(method="noether", alternative="two_sided")  

# You can even compare the metrics between two models (e.g., vs. model B)
cindex.compare(cindexB)  
```

## Installation

First, install the package:

```bash
pip install torchsurv
```
or for local installation (from package root)

```bash
pip install -e . 
```

If you use Conda, you can install requirements into a conda environment
using the `environment.yml` file included in the `dev` subfolder of the source repository.

## Getting started

We recommend starting with the [introductory guide](notebooks/introduction), where you'll find an overview of the package's functionalities. 

## Usage

### Survival data

We simulate a random batch of 64 sujects. Each subject is associated with a binary event status (= ```True``` if event occured), a time-to-event or censoring and 16 covariates.

```python
>>> import torch
>>> _ = torch.manual_seed(52)
>>> n = 64
>>> x = torch.randn((n, 16))
>>> event = torch.randint(low=0, high=2, size=(n,)).bool()
>>> time = torch.randint(low=1, high=100, size=(n,)).float()
```

### Cox proportional hazards model 

The user is expected to have defined a model that outputs the estimated *log relative hazard* for each subject. For illustrative purposes, we define a simple linear model that generates a linear combination of the covariates. 

```python
>>> from torch import nn
>>> model_cox = nn.Sequential(nn.Linear(16, 1))
>>> log_hz = model_cox(x)
>>> print(log_hz.shape)
torch.Size([64, 1])
```

Given the estimated log relative hazard and the survival data, we calculate the current loss for the batch with:

```python
>>> from torchsurv.loss.cox import neg_partial_log_likelihood
>>> loss = neg_partial_log_likelihood(log_hz, event, time)
>>> print(loss)
tensor(4.1723, grad_fn=<MeanBackward0>)
```
We obtain the concordance index for this batch with:

```python
>>> from torchsurv.metrics.cindex import ConcordanceIndex
>>> with torch.no_grad():
>>>     log_hz = model_cox(x)
>>> cindex = ConcordanceIndex()
>>> print(cindex(log_hz, event, time))
tensor(0.4872)
```

We obtain the Area Under the Receiver Operating Characteristic Curve (AUC) at a new time $t = 50$ for this batch with:

```python
>>> from torchsurv.metrics.auc import Auc
>>> new_time = torch.tensor(50.)
>>> auc = Auc()
>>> print(auc(log_hz, event, time, new_time=50))
tensor([0.4737])
```

### Weibull accelerated failure time (AFT) model

The user is expected to have defined a model that outputs for each subject the estimated *log scale* and optionally the *log shape* of the Weibull distribution that the event density follows. In case the model has a single output, `torchsurv` assume that the shape is equal to 1, resulting in the event density to be an exponential distribution solely parametrized by the scale.

For illustrative purposes, we define a simple linear model that estimate two linear combinations of the covariates (log scale and log shape parameters). 
```python
>>> from torch import nn
>>> model = nn.Sequential(nn.Linear(16, 2))
>>> log_params = model(x)
>>> print(log_params.shape)
torch.Size([64, 2])
```

Given the estimated log scale and log shape and the survival data, we calculate the current loss for the batch with:

```python
>>> from torchsurv.loss.weibull import neg_log_likelihood
>>> loss = neg_log_likelihood(log_params, event, time)
>>> print(loss)
tensor(731.9636, grad_fn=<MeanBackward0>)
```

To evaluate the predictive performance of the model, we calculate subject-specific log hazard and survival function evaluated at all times with:

```python
>>> from torchsurv.loss.weibull import log_hazard
>>> from torchsurv.loss.weibull import survival_function
>>> with torch.no_grad():
>>>     log_params = model(x)
>>> log_hz = log_hazard(log_params, time)
>>> print(log_hz.shape)
torch.Size([64, 64])
>>> surv = survival_function(log_params, time)
>>> print(surv.shape)
torch.Size([64, 64])
```

We obtain the concordance index for this batch with:

```python
>>> from torchsurv.metrics.cindex import ConcordanceIndex
>>> cindex = ConcordanceIndex()
>>> print(cindex(log_hz, event, time))
tensor(0.4062)
```

We obtain the AUC at a new time $t =50$ for this batch with:

```python
>>> from torchsurv.metrics.auc import Auc
>>> new_time = torch.tensor(50.)
>>> log_hz_t = log_hazard(log_params, time=new_time)
>>> auc = Auc()
>>> print(auc(log_hz_t, event, time, new_time=new_time))
tensor([0.3509])
```

We obtain the integrated brier-score with:

```python
>>> from torchsurv.metrics.brier_score import BrierScore
>>> brier_score = BrierScore()
>>> bs = brier_score(surv, event, time)
>>> print(brier_score.integral())
tensor(0.4447)
```

## Contributing

We value contributions from the community to enhance and improve this project. If you'd like to contribute, please consider the following:

1. Create Issues: If you encounter bugs, have feature requests, or want to suggest improvements, please create an issue in the GitHub repository. Make sure to provide detailed information about the problem, including code for reproducibility, or enhancement you're proposing.

2. Fork and Pull Requests: If you're willing to address an existing issue or contribute a new feature, fork the repository, create a new branch, make your changes, and then submit a pull request. Please ensure your code follows our coding conventions and include tests for any new functionality.

By contributing to this project, you agree to license your contributions under the same license as this project.


## Contact

If you have any questions, suggestions, or feedback, feel free to reach out to [us](AUTHORS).

## Cite 

If you use this project in academic work or publications, we appreciate citing it using the following BibTeX entry:

```bibtex
@misc{projectname,
  title={Project Name},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your_username/your_repo}},
}
```

