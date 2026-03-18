# TorchSurv: Deep Survival Analysis in Pure PyTorch

*By Thibaud Coroller, Mélodie Monod, Peter Krusche, Qian Cao — Novartis & FDA*

---

Time-to-event prediction shows up everywhere: How long before a customer cancels their subscription? When will a turbine fail? How quickly will a borrower default on a loan? These are not classification problems. They are not regression problems. They are **survival analysis** problems — and most ML toolkits handle them poorly.

The core challenge is **censoring**. In real-world data, you rarely observe every event. A customer is still subscribed when you pull the data. A machine is still running. The loan is still open. Standard loss functions silently discard these observations or misrepresent them, introducing bias. Survival analysis handles this correctly by incorporating the information that an event *hasn't happened yet* — and that matters.

## Introducing TorchSurv

[TorchSurv](https://github.com/Novartis/torchsurv) is a lightweight Python library that brings survival analysis into the native PyTorch ecosystem. Rather than locking you into a fixed architecture, TorchSurv acts as a **companion to your own models**: you define the network, TorchSurv provides the loss functions and evaluation metrics.

Install it in one line:

```bash
pip install torchsurv
# or
conda install conda-forge::torchsurv
```

## Survival Analysis Is Not Just for Medicine

Survival analysis originated in clinical trials, but the mathematics applies to any domain where you are modeling *time until an event*:

| Domain | "Event" | "Time" |
|---|---|---|
| Telecom / SaaS | Customer churn | Subscription duration |
| HR / People Analytics | Employee attrition | Tenure |
| Predictive Maintenance | Equipment failure | Machine operating hours |
| Finance / Credit | Loan default | Time since origination |
| Meteorology | Extreme weather event | Days since last occurrence |

Each of these has censored observations — customers who haven't churned yet, machines that are still running — and TorchSurv handles them all with the same API.

## What TorchSurv Provides

**Loss functions** for the two most widely used survival models:

- **Cox Proportional Hazards**: your model outputs log relative hazards; TorchSurv computes the negative partial log-likelihood (with Breslow or Efron tie-handling).
- **Weibull AFT**: your model outputs log scale and log shape parameters; TorchSurv computes the negative log-likelihood.

**Evaluation metrics** with confidence intervals and hypothesis tests:

- **Concordance Index (C-index)**: does the model rank subjects correctly by risk?
- **Time-dependent AUC**: discrimination ability at a specific time horizon (e.g., 1-year failure rate)
- **Brier Score**: calibration of the predicted survival probabilities

All metrics support confidence intervals (bootstrap or analytical) and built-in statistical tests — including pairwise model comparison.

## 10 Lines to Train and Evaluate

```python
import torch
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

model = MyMLP(in_features=32, out_features=1)  # your architecture
optimizer = torch.optim.Adam(model.parameters())

for x, event, time in dataloader:
    estimate = model(x)                                          # log relative hazard
    loss = cox.neg_partial_log_likelihood(estimate, event, time) # handles censoring
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Evaluate
cindex = ConcordanceIndex()
cindex(estimate, event, time)
print(cindex.confidence_interval())          # analytical 95% CI
print(cindex.p_value(alternative="greater")) # test vs. random (0.5)
```

That's it. No special data formats. No wrapper classes. Your model returns a tensor; TorchSurv takes it from there.

## Comparing Two Models? One Method Call.

```python
from torchsurv.metrics.cindex import ConcordanceIndex

cindex_a = ConcordanceIndex()
cindex_b = ConcordanceIndex()

cindex_a(estimate_a, event, time)
cindex_b(estimate_b, event, time)

cindex_a.compare(cindex_b)  # returns p-value for H0: cindex_a == cindex_b
```

## Rigorously Benchmarked

TorchSurv outputs have been independently validated against established R packages (`survival`, `timeROC`, `riskRegression`) and Python packages (`lifelines`, `scikit-survival`, `pycox`) on both synthetic and real-world datasets. The full [benchmark report](https://opensource.nibr.com/torchsurv/benchmarks.html) is published in the documentation.

TorchSurv is also part of the **FDA's [Regulatory Science Tool Catalog](https://cdrh-rst.fda.gov/torchsurv-deep-learning-tools-survival-analysis)** (RST24AI17.01), developed in collaboration between Novartis and the Center for Devices and Radiological Health (CDRH).

## Why Not an Existing Library?

| Capability | TorchSurv | lifelines | pycox | scikit-survival |
|---|:---:|:---:|:---:|:---:|
| Custom PyTorch architecture | ✅ | ❌ | ⚠️ | ❌ |
| Pure PyTorch loss (autograd-compatible) | ✅ | ❌ | ⚠️ | ❌ |
| C-index with confidence interval | ✅ | ✅ | ✅ | ⚠️ |
| Time-dependent AUC | ✅ | ❌ | ⚠️ | ⚠️ |
| Brier Score | ✅ | ❌ | ✅ | ✅ |
| Model comparison (p-value) | ✅ | ❌ | ❌ | ❌ |

Existing libraries either lock you into a fixed model form or require non-PyTorch backends that break `autograd`. TorchSurv is designed from the ground up to be a drop-in addition to any PyTorch training loop.

## Get Started

- 📦 **Install**: `pip install torchsurv`
- 📖 **Docs**: [opensource.nibr.com/torchsurv](https://opensource.nibr.com/torchsurv/)
- 🗒️ **Intro notebook**: [Introduction to TorchSurv](https://opensource.nibr.com/torchsurv/notebooks/introduction.html)
- 💻 **GitHub**: [github.com/Novartis/torchsurv](https://github.com/Novartis/torchsurv)
- 📄 **Paper**: [JOSS 2024](https://joss.theoj.org/papers/02d7496da2b9cc34f9a6e04cabf2298d)

Whether you're predicting customer churn, machine failure, or financial default — if your problem involves *time until an event* and *incomplete observations*, TorchSurv is built for you.
