#!/usr/bin/env python3
"""Classify a pull request as ``easy`` (auto-merge eligible) or ``needs-human``.

The tier is computed from the diff alone, never from anything Claude says about its own change.
Rules are documented in ``.github/claude-tiers.md``. Anything not explicitly allowed needs a human.

Usage: classify_pr.py --base <ref> --head <ref> [--github-output]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field

MAX_CHANGED_LINES = 150
MAX_CHANGED_FILES = 5

# Paths where any change needs a human.
HUMAN_PREFIXES = (
    "src/torchsurv/loss/",
    "src/torchsurv/metrics/",
    "src/torchsurv/stats/",
    ".github/",
    ".claude/",
    ".specify/",
    "dev/",
)
HUMAN_FILES = ("CODEOWNERS", "GOVERNANCE.md", ".pre-commit-config.yaml")

# Paths an easy PR may touch. Everything else needs a human.
EASY_PREFIXES = ("docs/", "src/torchsurv/", "tests/")
EASY_FILES = ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "pyproject.toml", "uv.lock")

FILE_HEADER = re.compile(r"^diff --git a/(?P<old>.+) b/(?P<new>.+)$")
REMOVED_TEST = re.compile(r"^-\s*(async\s+)?def test_")
ADDED_SKIP = re.compile(r"^\+.*(pytest\.mark\.(skip|xfail)|pytest\.(skip|xfail)\(|@unittest\.skip)")
COVERAGE_LINE = re.compile(r"^[-+].*cov-fail-under")
PYPROJECT_SENSITIVE = re.compile(r"^[-+]\s*(requires-python|version)\s*=")
REMOVED_API = re.compile(r"^-\s*(async\s+)?(def|class)\s+[A-Za-z]")
ADDED_PUBLIC_API = re.compile(r"^\+(async\s+)?(def|class)\s+[A-Za-z]")


@dataclass
class FileDiff:
    path: str
    old_path: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        return [f"+{line}" for line in self.added] + [f"-{line}" for line in self.removed]


def parse_diff(diff: str) -> list[FileDiff]:
    """Split a unified diff into per-file added and removed lines."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    for line in diff.splitlines():
        header = FILE_HEADER.match(line)
        if header:
            current = FileDiff(path=header["new"], old_path=header["old"])
            files.append(current)
        elif current is None or line.startswith(("+++", "---")):
            continue
        elif line.startswith("+"):
            current.added.append(line[1:])
        elif line.startswith("-"):
            current.removed.append(line[1:])
    return files


def _is_human_path(path: str) -> bool:
    return path.startswith(HUMAN_PREFIXES) or path in HUMAN_FILES


def _is_easy_path(path: str) -> bool:
    return path.startswith(EASY_PREFIXES) or path in EASY_FILES


def classify(files: list[FileDiff]) -> tuple[str, list[str]]:
    """Return ``("easy" | "needs-human", reasons)``."""
    reasons: list[str] = []

    if not files:
        return "needs-human", ["empty diff"]

    changed = sum(len(f.added) + len(f.removed) for f in files)
    if changed > MAX_CHANGED_LINES:
        reasons.append(f"{changed} changed lines (limit {MAX_CHANGED_LINES})")
    if len(files) > MAX_CHANGED_FILES:
        reasons.append(f"{len(files)} changed files (limit {MAX_CHANGED_FILES})")

    for f in files:
        paths = {f.path, f.old_path}
        for path in sorted(paths):
            if _is_human_path(path):
                reasons.append(f"{path}: protected path")
            elif not _is_easy_path(path):
                reasons.append(f"{path}: not on the easy allowlist")

        lines = f.lines
        is_test = f.path.startswith("tests/")
        is_src = f.path.startswith("src/torchsurv/")

        if is_test:
            if any(REMOVED_TEST.match(line) for line in lines):
                reasons.append(f"{f.path}: removes a test")
            if any(ADDED_SKIP.match(line) for line in lines):
                reasons.append(f"{f.path}: adds a skip or xfail")
        if any(COVERAGE_LINE.match(line) for line in lines):
            reasons.append(f"{f.path}: changes the coverage threshold")
        if f.path == "pyproject.toml" and any(PYPROJECT_SENSITIVE.match(line) for line in lines):
            reasons.append("pyproject.toml: changes version or requires-python")
        if is_src:
            if any("__all__" in line for line in lines) or f.path.endswith("__init__.py"):
                reasons.append(f"{f.path}: touches public exports")
            if any(REMOVED_API.match(line) for line in lines):
                reasons.append(f"{f.path}: removes or changes a def/class signature")
            if any(ADDED_PUBLIC_API.match(line) and not _is_private_def(line) for line in lines):
                reasons.append(f"{f.path}: adds a public def/class (new feature?)")

    unique = list(dict.fromkeys(reasons))
    return ("needs-human", unique) if unique else ("easy", ["all rules passed"])


def _is_private_def(line: str) -> bool:
    return re.match(r"^\+(async\s+)?(def|class)\s+_", line) is not None


def git_diff(base: str, head: str) -> str:
    return subprocess.run(
        ["git", "diff", "--no-color", "--no-ext-diff", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--github-output", action="store_true", help="also write tier/reason to $GITHUB_OUTPUT")
    args = parser.parse_args()

    tier, reasons = classify(parse_diff(git_diff(args.base, args.head)))
    print(json.dumps({"tier": tier, "reasons": reasons}, indent=2))

    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write(f"tier={tier}\n")
            out.write(f"reason={' '.join('; '.join(reasons).split())}\n")


if __name__ == "__main__":
    main()
