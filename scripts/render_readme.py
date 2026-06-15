#!/usr/bin/env python
"""Render README.md from scripts/README.template using live metrics.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/render_readme.py

Exits 0 always; prints whether README.md changed or not.
Called by .githooks/pre-commit before each commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve repo root relative to this script so the hook works regardless of cwd.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from neuroslm.readme_metrics import build_metrics, render_template  # noqa: E402

_TEMPLATE_PATH = _REPO_ROOT / "scripts" / "README.template"
_README_PATH = _REPO_ROOT / "README.md"


def main() -> None:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    metrics = build_metrics(_REPO_ROOT)
    rendered = render_template(template, metrics)

    current = _README_PATH.read_text(encoding="utf-8") if _README_PATH.exists() else ""
    if rendered != current:
        _README_PATH.write_text(rendered, encoding="utf-8")
        print(f"[render_readme] README.md updated ({len(metrics)} metrics substituted)")
    else:
        print("[render_readme] README.md unchanged")


if __name__ == "__main__":
    main()
