# LinkedIn Post — TorchSurv v0.2.0 + 100K Downloads

---

🔥 TorchSurv just crossed 100,000 PyPI downloads — and we're shipping v0.2.0.

TorchSurv is a lightweight, pure-PyTorch library for deep survival analysis — built by Novartis and the FDA, published in JOSS, and listed in the FDA's Regulatory Science Tool Catalog.

Here's what's new in v0.2.0:

📐 **Pydantic-powered validation** — We replaced our custom validation layer with pydantic ≥2.0 validators across all loss and metric modules. Inputs are now validated with structured, composable models instead of ad-hoc checks. Cleaner code, better error messages, fewer surprises.

⚡ **Vectorized internals** — Cox partial likelihood and cumulative hazard computations are now fully vectorized. Combined with hypothesis-based property tests and ruff + mypy strict mode, the codebase is more robust and easier to contribute to.

🧩 **PyTorch Ecosystem alignment** — v0.2.0 adds formal governance (GOVERNANCE.md, CODEOWNERS), validated compatibility with PyTorch 2.10–2.11, and CI powered by uv + astral-sh tooling. We're building to last.

TorchSurv gives you full freedom to define your own neural network architecture for survival modeling — no restrictive parameterizations, no framework lock-in. Just losses and metrics that work like any other PyTorch function.

If you work with time-to-event data, give it a look. ⭐ Star the repo: https://github.com/Novartis/torchsurv

#SurvivalAnalysis #PyTorch #DeepLearning #OpenSource #MachineLearning #FDA #Biostatistics #TorchSurv
