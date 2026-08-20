#!/usr/bin/env python3
"""Sync the PyTorch badge in README.md with the torch-compatibility CI matrix."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "torch-compatibility.yml"
README = REPO_ROOT / "README.md"

BADGE_PATTERN = re.compile(
    r"\[!\[PyTorch\]\(https://img\.shields\.io/badge/PyTorch-.*?-ee4c2c\?logo=pytorch&logoColor=white\)\]"
    r"\(.*?\)"
)
MATRIX_PATTERN = re.compile(r"torch-version:\s*\[([^\]]+)\]")


def extract_torch_versions() -> list[str]:
    """Parse major.minor versions from the CI matrix."""
    text = WORKFLOW.read_text()
    match = MATRIX_PATTERN.search(text)
    if not match:
        print(f"ERROR: Could not find torch-version matrix in {WORKFLOW}", file=sys.stderr)
        sys.exit(1)
    raw = match.group(1)
    versions = [v.strip().strip('"').strip("'") for v in raw.split(",")]
    # Strip to major.minor
    return [".".join(v.split(".")[:2]) for v in versions]


def build_badge(versions: list[str]) -> str:
    """Build a shields.io badge markdown string."""
    label = "%20|%20".join(versions)
    url = f"https://img.shields.io/badge/PyTorch-{label}-ee4c2c?logo=pytorch&logoColor=white"
    link = "https://github.com/Novartis/torchsurv/actions/workflows/torch-compatibility.yml"
    return f"[![PyTorch]({url})]({link})"


def main() -> int:
    versions = extract_torch_versions()
    new_badge = build_badge(versions)

    readme_text = README.read_text()
    if not BADGE_PATTERN.search(readme_text):
        print("ERROR: Could not find PyTorch badge in README.md", file=sys.stderr)
        return 1

    updated = BADGE_PATTERN.sub(re.escape(new_badge).replace("\\[", "[").replace("\\]", "]"), readme_text, count=1)

    # Simpler approach: just do a direct string replacement
    match = BADGE_PATTERN.search(readme_text)
    if match:
        old_badge = match.group(0)
        if old_badge == new_badge:
            print(f"PyTorch badge already up to date: {' | '.join(versions)}")
            return 0
        updated = readme_text.replace(old_badge, new_badge)
        README.write_text(updated)
        print(f"Updated PyTorch badge: {' | '.join(versions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
