import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "dev" / "classify_pr.py"
spec = importlib.util.spec_from_file_location("classify_pr", SCRIPT)
assert spec is not None and spec.loader is not None
classify_pr = importlib.util.module_from_spec(spec)
sys.modules["classify_pr"] = classify_pr
spec.loader.exec_module(classify_pr)


def make_diff(
    path: str, added: tuple[str, ...] = (), removed: tuple[str, ...] = (), old_path: str | None = None
) -> str:
    old = old_path or path
    body = "".join(f"-{line}\n" for line in removed) + "".join(f"+{line}\n" for line in added)
    return f"diff --git a/{old} b/{path}\n--- a/{old}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n{body}"


def tier(*diffs: str) -> tuple[str, list[str]]:
    return classify_pr.classify(classify_pr.parse_diff("".join(diffs)))


def test_docs_and_changelog_are_easy():
    result, _ = tier(make_diff("docs/index.rst", ("fixed typo",)), make_diff("docs/CHANGELOG.md", ("- fix (#1)",)))
    assert result == "easy"


def test_small_fix_outside_numerics_with_test_is_easy():
    result, _ = tier(
        make_diff("src/torchsurv/tools/validators.py", ("    x = 1",), ("    x = 0",)),
        make_diff("tests/test_validators.py", ("    assert x == 1",)),
    )
    assert result == "easy"


@pytest.mark.parametrize("pkg", ["loss/cox.py", "metrics/auc.py", "stats/ipcw.py"])
def test_numerics_need_human(pkg):
    result, reasons = tier(make_diff(f"src/torchsurv/{pkg}", ("    # comment",)))
    assert result == "needs-human"
    assert any("protected path" in r for r in reasons)


def test_all_change_needs_human():
    result, reasons = tier(make_diff("src/torchsurv/__init__.py", ('__all__ = ["Auc"]',)))
    assert result == "needs-human"
    assert any("public exports" in r for r in reasons)


def test_signature_change_needs_human():
    result, reasons = tier(
        make_diff("src/torchsurv/tools/validators.py", ("def f(a, b):",), ("def f(a):",)),
    )
    assert result == "needs-human"
    assert any("signature" in r for r in reasons)


def test_new_public_function_needs_human_but_private_helper_does_not():
    result, _ = tier(make_diff("src/torchsurv/tools/validators.py", ("def new_feature():",)))
    assert result == "needs-human"
    result, _ = tier(make_diff("src/torchsurv/tools/validators.py", ("def _helper():",)))
    assert result == "easy"


def test_deleted_test_needs_human():
    result, reasons = tier(make_diff("tests/test_cox.py", (), ("def test_ties():",)))
    assert result == "needs-human"
    assert any("removes a test" in r for r in reasons)


def test_skip_marker_needs_human():
    result, _ = tier(make_diff("tests/test_cox.py", ("@pytest.mark.skip",)))
    assert result == "needs-human"


def test_lowered_coverage_needs_human():
    result, _ = tier(make_diff("pyproject.toml", ('    "--cov-fail-under=70",',), ('    "--cov-fail-under=80",')))
    assert result == "needs-human"


def test_version_bump_needs_human():
    result, _ = tier(make_diff("pyproject.toml", ('version = "0.3.0"',), ('version = "0.2.0"',)))
    assert result == "needs-human"


def test_dependency_lock_bump_is_easy():
    result, _ = tier(make_diff("uv.lock", ('version = "1.2.4"',), ('version = "1.2.3"',)))
    assert result == "easy"


@pytest.mark.parametrize(
    "path", [".github/workflows/codeqc.yml", "dev/classify_pr.py", ".claude/settings.json", "CODEOWNERS"]
)
def test_gate_and_ci_config_need_human(path):
    result, _ = tier(make_diff(path, ("x",)))
    assert result == "needs-human"


def test_rename_out_of_protected_path_needs_human():
    result, _ = tier(make_diff("docs/cox.py", ("x",), old_path="src/torchsurv/loss/cox.py"))
    assert result == "needs-human"


def test_unknown_path_is_denied_by_default():
    result, reasons = tier(make_diff("setup_new_thing.sh", ("echo hi",)))
    assert result == "needs-human"
    assert any("allowlist" in r for r in reasons)


def test_oversize_diff_needs_human():
    result, reasons = tier(make_diff("docs/big.rst", tuple(f"line {i}" for i in range(200))))
    assert result == "needs-human"
    assert any("changed lines" in r for r in reasons)


def test_too_many_files_needs_human():
    result, reasons = tier(*(make_diff(f"docs/f{i}.rst", ("x",)) for i in range(6)))
    assert result == "needs-human"
    assert any("changed files" in r for r in reasons)


def test_empty_diff_needs_human():
    result, _ = tier("")
    assert result == "needs-human"
