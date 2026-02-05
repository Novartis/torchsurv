Changelog
=========

Version 0.2.0
-------------

### Breaking Changes
* **Minimum Python version increased to 3.9+**
* **Validation API replaced with Pydantic v2 models**
  * `validate_survival_data()` → `SurvivalData` model
  * `validate_model()` → `ModelParameters` model
  * Better structured error messages with field-level details
* **Added pydantic>=2.0 as dependency**

### Added
* Enum types for string parameters (TiesMethod, Reduction, ConfidenceMethod, Alternative)
* Structured validation with Pydantic v2 in `tools/validation.py`
* Type hints using Python 3.9+ syntax (dict[], list[], | union operator)
* New `torchsurv.types` module with enum definitions

### Changed
* Improved type hints throughout codebase
* Better error messages with Pydantic validation
* Ruff linting now includes PIE, SIM, and RUF rules

### Removed
* Python 3.8 support

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
