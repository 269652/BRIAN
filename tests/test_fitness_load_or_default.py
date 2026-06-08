# -*- coding: utf-8 -*-
"""TDD acceptance suite — `FitnessConfig.load_or_default` robustness +
the colab minimal-CPU fallback path.

Reproduces the Colab failure:

    FileNotFoundError: DNA file not found: dna/evol/arch.dna   ← expected on Colab
                                                                  (dna/ is .gitignored)
    During handling of the above exception, another exception occurred:
    FileNotFoundError: [Errno 2] No such file or directory: ''  ← the actual bug

`init_evolution()` correctly raised because the DNA wasn't checked in,
but the EXCEPT branch in `colab_train_minimal_cpu.main()` then called
``FitnessConfig.load_or_default("")`` — and `load_or_default` was
*supposed* to return the default when the path doesn't exist, but it
went straight into `cls.load("")` and blew up on `open("")`.

Two contracts under test:

  * `FitnessConfig.load_or_default(path)` must return the default for
    any *unreachable* path argument — including the empty string,
    `None`, and non-existent files.

  * The colab minimal-CPU script must survive a missing DNA file
    (the common case on a fresh Colab clone, since `dna/` is gitignored).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuroslm.fitness import FitnessConfig, FitnessObjective


# ──────────────────────────────────────────────────────────────────────
# load_or_default robustness — the root cause of the Colab crash
# ──────────────────────────────────────────────────────────────────────

class TestLoadOrDefaultRobustness:
    """`load_or_default` must NEVER call `cls.load()` with a path that
    cannot be opened.  Any unreachable path argument returns the default."""

    def test_empty_string_returns_default(self):
        """The exact Colab repro: `load_or_default("")` must return the
        default, not raise FileNotFoundError from inside `open("")`."""
        cfg = FitnessConfig.load_or_default("")
        assert isinstance(cfg, FitnessConfig)
        assert len(cfg.objectives) > 0, "default config must declare objectives"

    def test_none_returns_default(self):
        """`load_or_default(None)` is the most natural "no path" call —
        should also return the default rather than crash inside
        `Path(None)` or `open(None)`."""
        cfg = FitnessConfig.load_or_default(None)
        assert isinstance(cfg, FitnessConfig)
        assert len(cfg.objectives) > 0

    def test_no_arg_returns_default(self):
        """`load_or_default()` with no args should also work — the
        cleanest call shape for "I just want the default"."""
        cfg = FitnessConfig.load_or_default()
        assert isinstance(cfg, FitnessConfig)
        assert len(cfg.objectives) > 0

    def test_nonexistent_path_returns_default(self):
        """The documented happy-path: file simply doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            ghost = str(Path(tmp) / "definitely_not_here.json")
            cfg = FitnessConfig.load_or_default(ghost)
            assert isinstance(cfg, FitnessConfig)
            assert len(cfg.objectives) > 0

    def test_directory_path_returns_default(self):
        """If the path resolves to a *directory* (so `open(...)` would
        fail with IsADirectoryError), still fall back to default
        instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FitnessConfig.load_or_default(tmp)
            assert isinstance(cfg, FitnessConfig)
            assert len(cfg.objectives) > 0

    def test_existing_file_still_loads(self):
        """Regression guard: a valid path must still load correctly —
        we are only making the *fallback* more robust, not changing
        the happy path."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.json"
            payload = {
                "version": "1.0",
                "enabled": True,
                "objectives": [{
                    "name": "test_obj",
                    "metric": "loss",
                    "direction": "minimize",
                    "weight": 1.0,
                    "target": 0.0,
                    "metadata": {}
                }],
                "adaptation": None,
                "metadata": {}
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = FitnessConfig.load_or_default(str(path))
            assert len(cfg.objectives) == 1
            assert cfg.objectives[0].name == "test_obj"


# ──────────────────────────────────────────────────────────────────────
# colab_train_minimal_cpu fallback path — the actual user-visible bug
# ──────────────────────────────────────────────────────────────────────

class TestColabMinimalCPUFallback:
    """The colab minimal-CPU script must NOT crash when the DNA file
    is missing — the most common case on a fresh Colab `git clone`
    because `dna/` is gitignored.

    These tests don't run the full training (too slow for CPU CI); they
    only exercise the bootstrap path that historically crashed.
    """

    def test_main_survives_missing_dna(self, monkeypatch, tmp_path):
        """Run `main()` for zero steps from a cwd that lacks the DNA
        file.  Pre-fix behavior: FileNotFoundError on `open('')`.
        Post-fix: prints `[SKIP] evol.dna not found`, uses default
        fitness, runs to completion."""
        import colab_train_minimal_cpu as ccpu

        # Force tiny model + zero steps so the test stays fast.
        # `main` accepts the steps kwarg already.
        monkeypatch.chdir(tmp_path)   # cwd has no dna/ folder
        assert not (tmp_path / "dna" / "evol" / "arch.dna").exists()

        # `main(steps=0)` should perform setup but skip the training
        # loop body — it must return cleanly (or not raise).
        try:
            ccpu.main(steps=0, ood_every=1)
        except FileNotFoundError as e:
            pytest.fail(
                f"main() crashed on missing DNA fallback: {e!r} — "
                "the load_or_default('') anti-pattern is back"
            )

    def test_main_forwards_custom_dna_path(self, monkeypatch):
        """`main(dna_path=...)` must forward the supplied DNA path to
        init_evolution instead of always hard-coding dna/evol/arch.dna.
        """
        import colab_train_minimal_cpu as ccpu

        called = {}

        def _fake_init_evolution(path):
            called["path"] = path
            raise FileNotFoundError(path)

        monkeypatch.setattr("neuroslm.utils.init_evolution", _fake_init_evolution)

        # Keep execution fast: zero steps means setup-only.
        ccpu.main(steps=0, ood_every=1, dna_path="dna/custom/arch.dna")
        assert called["path"] == "dna/custom/arch.dna"


class TestTinyCliDnaRouting:
    """`brian train --preset=tiny` must actually honor --dna / --arch .dna.

    Regression target: command printed "Loading from DNA" but still ran
    the hardcoded evol.dna path in colab_train_minimal_cpu.
    """

    def _run_cmd_train(self, monkeypatch, args):
        from neuroslm import cli

        captured = {}

        class _Loader:
            def exec_module(self, module):
                def _main(**kwargs):
                    captured.update(kwargs)

                module.main = _main

        class _Spec:
            loader = _Loader()

        monkeypatch.setattr("importlib.util.spec_from_file_location",
                            lambda *a, **k: _Spec())
        monkeypatch.setattr("importlib.util.module_from_spec",
                            lambda spec: type("_M", (), {})())

        rc = cli.cmd_train(args)
        assert rc == 0
        return captured

    def test_tiny_cli_uses_arch_when_arch_is_dna(self, monkeypatch):
        import argparse

        args = argparse.Namespace(
            preset="tiny",
            arch="dna/evol/arch.dna",
            dna=None,
            steps=12,
            ood_every=34,
            batch=None,
            seq_len=None,
            d_sem=None,
        )
        captured = self._run_cmd_train(monkeypatch, args)
        assert captured["steps"] == 12
        assert captured["ood_every"] == 34
        assert captured["dna_path"] == "dna/evol/arch.dna"

    def test_tiny_cli_prefers_explicit_dna_over_arch(self, monkeypatch):
        import argparse

        args = argparse.Namespace(
            preset="tiny",
            arch="dna/wrong/arch.dna",
            dna="dna/right/arch.dna",
            steps=2,
            ood_every=3,
            batch=None,
            seq_len=None,
            d_sem=None,
        )
        captured = self._run_cmd_train(monkeypatch, args)
        assert captured["dna_path"] == "dna/right/arch.dna"
