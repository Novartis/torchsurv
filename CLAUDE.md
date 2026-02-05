# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchSurv is a lightweight Python package for deep survival analysis using PyTorch. It provides loss functions and evaluation metrics for survival models without imposing specific parametric forms, enabling the use of custom PyTorch-based neural networks.

**Key Philosophy**: Keep things simple. Pure PyTorch implementations that behave like standard PyTorch functions.

**Recognition**: Part of FDA's Regulatory Science Tool Catalog (RST24AI17.01) - a collaborative project between Novartis and the FDA.

## Common Commands

### Environment Setup
```bash
# Create conda environment
conda create -y -n torchsurv python=3.10
conda env update -n torchsurv -f dev/environment.yml
conda activate torchsurv

# Install package in development mode
export PYTHONPATH=src
pip install -e .
```

### Testing
```bash
# Run all unit tests
export PYTHONPATH=src
python -m unittest discover -s tests -v

# Or use the provided script
./dev/run-unittests.sh

# Run with coverage (requires pytest)
pytest --cov=torchsurv --cov-report=html

# Run doctests
./dev/run-doctests.sh

# Run a single test file
PYTHONPATH=src python -m unittest tests.test_cox -v
```

### Code Quality
```bash
# Run all code quality checks (ruff, mypy, doctests)
./dev/codeqc.sh check

# Format code with ruff
ruff format .

# Run type checking
mypy src/torchsurv
```

### Documentation
```bash
# Build and serve documentation locally
./dev/build-docs.sh serve
# Opens at http://localhost:8000

# Build documentation only
./dev/build-docs.sh

# Build PDF (from docs directory)
cd docs && make latexpdf
```

### Git Hooks
```bash
# Install pre-commit hooks (one-time setup)
pre-commit install
```

## Architecture

### Module Structure

```
src/torchsurv/
├── loss/                    # Loss function implementations
│   ├── cox.py              # Cox proportional hazards (neg_partial_log_likelihood)
│   ├── weibull.py          # Weibull AFT (neg_log_likelihood_weibull)
│   ├── survival.py         # Custom survival models
│   └── momentum.py         # Momentum-based loss (nn.Module wrapper)
├── metrics/                 # Evaluation metrics (stateful classes)
│   ├── cindex.py           # ConcordanceIndex class
│   ├── auc.py              # Auc class
│   └── brier_score.py      # BrierScore class
├── stats/                   # Statistical utilities
│   ├── kaplan_meier.py     # KaplanMeierEstimator class
│   └── ipcw.py             # Inverse Probability of Censoring Weights
└── tools/                   # Helper utilities
    └── validate_data.py    # Data validation functions
```

### Design Patterns

**Loss Functions**: Pure functions (stateless) that accept tensors and return loss tensors with gradients. Handle edge cases like ties in event times (Cox supports standard/Efron/Breslow methods).

**Metrics**: Stateful classes that maintain computation results. Support confidence intervals, statistical tests (p-values), and pairwise model comparisons.

**Input Validation**: Centralized in `tools/validate_data.py`. Validates tensors, events (boolean), times, and model outputs with clear error messages.

**Time-Varying Covariates**: Loss functions support time-varying covariates through appropriate tensor shapes.

### Expected Model Outputs

- **Cox models**: Output log relative hazard, shape `[batch_size, 1]`
- **Weibull AFT models**: Output log scale and optionally log shape, shape `[batch_size, 2]` or `[batch_size, 1]` for exponential distribution

## Testing Strategy

All metrics and loss functions are benchmarked against external R and Python packages for correctness:

- Tests use pre-computed benchmark values stored in `tests/benchmark_data/*.json`
- Benchmarks generated from R packages (`survival`, `riskRegression`, `survAUC`, etc.) and Python packages (`lifelines`, `scikit-survival`)
- Test utilities are in `tests/utils.py`
- Coverage requirement: minimum 80% (enforced by pytest)

## Code Quality Standards

**Formatting**: Ruff with double quotes, 120-character line width, docstring formatting enabled.

**Linting**: Ruff strict rules (E, W, F, I, B, C4, UP). Tests excluded from some checks.

**Type Checking**: MyPy strict mode with `disallow_untyped_defs=true`. Full type annotations required throughout codebase. Scientific packages (torch, numpy, scipy) ignore missing imports.

**Documentation**: NumPy-style docstrings with doctest-compatible examples.

**Pre-commit Hooks**:
- Ruff formatting and linting
- Codespell for spell checking
- Standard checks (trailing whitespace, end-of-file, YAML/TOML validation, no debug statements, no private keys)

## Development Workflow

1. **Before making changes**: Read relevant files first, understand existing patterns
2. **Follow existing code style**: Use Ruff for formatting, MyPy for type checking
3. **Write tests**: Add tests with benchmark comparisons where appropriate
4. **Update documentation**: Include docstring examples that work with doctest
5. **Run quality checks**: Use `./dev/codeqc.sh check` before committing
6. **Pre-commit**: Hooks will auto-fix formatting issues

## Version and Release

- Version managed in `src/torchsurv/__init__.py` (currently v0.1.6)
- Version dynamically extracted by Hatchling build system
- Update `docs/CHANGELOG.md` for releases
- Distributed via PyPI and Conda-Forge

## Key Dependencies

**Runtime** (minimal by design):
- torch (PyTorch backend)
- scipy (statistical computations)
- numpy (numerical operations)
- torchmetrics (additional metrics)

**Development**:
- ruff (>=0.12.4) - linting and formatting
- mypy (>=1.7) - type checking
- pytest (>=7.0) - testing framework
- hypothesis (>=6.0) - property-based testing
- pre-commit (>=3.0) - git hooks

## Important Notes

- **Minimal dependencies**: Only 4 core dependencies to enable easy integration
- **Pure PyTorch**: Full PyTorch backend, behaves like native torch functions
- **Benchmark-driven**: All implementations validated against R/Python packages
- **Type safety**: Full type annotations with strict MyPy checking
- **Avoid over-engineering**: Keep implementations simple and focused on core functionality
