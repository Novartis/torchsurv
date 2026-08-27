Changelog
=========

Version 0.2.0
-------------

### Breaking Changes

* Dropped Python 3.8 support; now requires Python >=3.9 (#148)
* Bumped torch dependency to >=2.8 (#148)
* Replaced ``validate_data.py`` with pydantic-based validators (``pydantic>=2.0`` now required) (#148)

### Bug Fixes

* Fixed IPCW device mismatch issue (#147)
* Fixed the Brier score zero-standard-error warning, which tested the reduced vector rather than each element and so only fired when every standard error was zero (#167)

### Core Refactoring

* Refactored all loss modules (cox, weibull, momentum, survival) to use pydantic validators (#148)
* Refactored all metric modules (auc, brier_score, cindex) to use new validators (#148)
* Vectorized Cox partial likelihood and cumulative hazard computations (#148)
* Added ``__init__.py`` with ``__all__`` exports to all subpackages (#148)

### Development Tooling

* Migrated from conda to uv as primary development tool (#148)
* Added dependency-groups in ``pyproject.toml`` (dev, docs, publish) (#148)
* Added ruff + mypy strict mode to pre-commit and CI (#148)
* Modernized CI workflows to use uv + astral-sh/setup-uv@v5 (#148)
* Upgraded workflow tests with better Python/PyTorch version coverage (#158)

### Testing

* Added hypothesis-based property tests (#148)
* Added shared test fixtures via ``conftest.py`` (#148)
* Added pydantic validation tests (#148)

### Documentation

* Added GOVERNANCE.md, CODEOWNERS, and release info for PyTorch Ecosystem compliance (#156)
* Updated README with versioning, compatibility, and changelog links (#156)
* Fixed URL typos and broken links (#150)
* Added package overview docs and non-medical applications notebook (#148)

Version 0.1.6
-------------

* Fixed bugs (#118, #119, #137),
* Added Full Disclaimer and Link to RST page for TorchSurv (#125)
* Cox Proportional Hazards model:
  * Added Breslow’s estimator for the survival function (#127)
  * Added support for stratified models (#128)
* Added survival model parameterized by log-hazard, with loss computed using the trapezoidal rule (#138)
* Added support for time-dependent covariates (#138, #140)

Version 0.1.5
-------------

* Fixed bugs (#94), (#97)
* Improved codebase to match Torch.jit.script (#67)
* Added precommit (#105)
* Fixed logo issue (#95)

Version 0.1.4
-------------

* JOSS review edits (#45, #59, #66, #68, #77, #78)
* Torch.compile tests #81
* Improved documentation (#73, #76)
* Improved notebook #83

Version 0.1.3
-------------

* Tutorial dataset error on momentum.ipynb #50
* Fix issue #48 - log_hazard returns torch.Inf
* Fix warning with Spearman correlation #41
* Added in-depth statistical background to link AUC to C-index #39
* Created Conda Forge version #47
* Updated CICD builds #53

Version 0.1.2
-------------

* Updated package documentation with publication links & badges (#9, #14, #16, #19, #21, #22, #24)
* Fixed and documented package dependencies (#1)

Version 0.1.1
-------------

* Added `metrics` classes (AUC, Cindex, Brier score)
* Added `stats` class with Kaplan Meier
* Created Sphinx documentation
* Created `notebook` examples
* Added R benchmark comparison

Version 0.1.0
-------------

* Initial release of CoxPH and Weibull classes.
