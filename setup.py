"""Thin setup.py — modern config lives in pyproject.toml.

This file exists only for `pip install -e .` backward-compatibility with
older pip versions and for the editable-install workflow. All metadata
(name, version, deps, entry points like `brian = neuroslm.cli:main`)
is declared in pyproject.toml.

For a fresh install from scratch, use the wrapper scripts which create
a venv and pip-install everything:

    bash  scripts/install.sh        # Linux / macOS / git-bash on Windows
    pwsh  scripts/install.ps1       # PowerShell on Windows
"""
from setuptools import setup

setup()
