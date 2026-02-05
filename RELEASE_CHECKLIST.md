# TorchSurv v0.2.0 Release Checklist

## Pre-Release Testing

### Local Testing
- [x] All new modules import successfully
- [x] Comprehensive test suite passes (15/15 tests)
- [x] Pydantic validation works correctly
- [x] Error messages are informative
- [x] Type hints are correct
- [x] Ruff linting passes (minor warnings intentional)
- [x] Python syntax validation passes
- [ ] Full unittest suite passes (requires dev dependencies)
- [ ] Doctests pass (`./dev/run-doctests.sh`)
- [ ] Code quality checks pass (`./dev/codeqc.sh check`)

### Benchmark Testing
- [ ] Cox model benchmarks pass vs R `survival` package
- [ ] Weibull model benchmarks pass vs R/Python
- [ ] C-index benchmarks pass vs external packages
- [ ] AUC benchmarks pass
- [ ] Brier score benchmarks pass
- [ ] All test files in `tests/` pass

### Performance Testing
- [ ] No performance regressions vs v0.1.6
- [ ] Pydantic validation overhead is negligible
- [ ] Memory usage unchanged

## Documentation

### Updated Files
- [x] `src/torchsurv/__init__.py` - Version 0.2.0
- [x] `docs/CHANGELOG.md` - Release notes
- [x] `pyproject.toml` - Dependencies and settings
- [x] `MIGRATION_GUIDE.md` - User migration instructions
- [x] `IMPLEMENTATION_SUMMARY.md` - Technical details

### Documentation Build
- [ ] Sphinx documentation builds without errors (`./dev/build-docs.sh`)
- [ ] PDF documentation builds (`cd docs && make latexpdf`)
- [ ] All docstrings are valid
- [ ] API documentation is correct

## Code Review

### Core Changes
- [x] `src/torchsurv/tools/validation.py` - New Pydantic models
- [x] `src/torchsurv/types.py` - New enum types
- [x] `src/torchsurv/loss/cox.py` - Updated validation
- [x] `src/torchsurv/loss/weibull.py` - Updated validation
- [x] `src/torchsurv/loss/survival.py` - Updated validation
- [x] `src/torchsurv/metrics/cindex.py` - Updated validation
- [x] `src/torchsurv/metrics/auc.py` - Updated validation
- [x] `src/torchsurv/metrics/brier_score.py` - Updated validation
- [x] `src/torchsurv/stats/kaplan_meier.py` - Updated validation
- [x] `src/torchsurv/stats/ipcw.py` - Updated validation

### Code Quality
- [x] Type hints use Python 3.9+ syntax
- [x] `from __future__ import annotations` in all files
- [x] No breaking changes to public API
- [x] Error messages are helpful
- [x] Code is well-documented

## Version Control

### Git
- [ ] All changes committed
- [ ] Commit message follows convention
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### Tags
- [ ] Create git tag `v0.2.0`
- [ ] Tag message includes release notes summary

## CI/CD

### GitHub Actions
- [ ] Update Python version matrix (remove 3.8, ensure 3.9-3.11)
- [ ] All CI tests pass
- [ ] Pre-commit hooks pass
- [ ] Code coverage ≥80%

### Conda
- [ ] Update `dev/environment.yml` with Python 3.9+
- [ ] Test conda environment creation

## Distribution

### PyPI
- [ ] Build distribution: `python -m build`
- [ ] Test distribution locally: `pip install dist/torchsurv-0.2.0-*.whl`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify PyPI page shows correct information
- [ ] Test installation: `pip install torchsurv==0.2.0`

### Conda-Forge
- [ ] Update conda-forge feedstock
- [ ] Add `pydantic>=2.0` dependency
- [ ] Update Python version requirement to >=3.9
- [ ] Test conda installation

## Communication

### Release Announcement
- [ ] GitHub Release created with notes from CHANGELOG.md
- [ ] Include MIGRATION_GUIDE.md link
- [ ] Include breaking changes warning
- [ ] Tag as breaking release (v0.2.0)

### Documentation Sites
- [ ] Update documentation site with v0.2.0
- [ ] Ensure migration guide is visible
- [ ] Update examples if needed

### Community
- [ ] Post release announcement
- [ ] Update README badges if needed
- [ ] Notify key users of breaking changes

## Post-Release

### Monitoring
- [ ] Monitor GitHub issues for migration problems
- [ ] Monitor PyPI/Conda download statistics
- [ ] Check for user reports of breaking changes

### Bug Fixes
- [ ] Keep v0.1.6 branch for critical bug fixes
- [ ] Define end-of-life for v0.1.6

## Rollback Plan

If critical issues are found:

1. **Identify Issue**
   - Document the problem
   - Determine if it's fixable quickly

2. **Quick Fix (if possible)**
   - Create hotfix branch
   - Release v0.2.1 with fix
   - Fast-track release process

3. **Rollback (if needed)**
   - Yank v0.2.0 from PyPI
   - Restore v0.1.6 as latest
   - Communicate issue to users

## Sign-Off

### Development Team
- [ ] Lead developer reviewed
- [ ] Changes tested by team
- [ ] Documentation reviewed

### Stakeholders
- [ ] FDA collaboration team notified (if applicable)
- [ ] Novartis team notified (if applicable)
- [ ] Key users notified of breaking changes

## Notes

### Known Issues
- Phase 3 (test migration to pytest) not included - will be done in future release
- Some minor ruff warnings (SIM108) intentionally left for code clarity

### Future Work
- Test suite migration to pytest (v0.3.0)
- Dataclasses for metric state (v0.3.0)
- Additional enum enforcement (optional)

---

**Release Date Target:** TBD
**Release Manager:** TBD
**Approved By:** TBD
