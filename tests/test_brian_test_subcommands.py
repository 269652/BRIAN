"""Contracts for ``brian test {quick,fast,full}`` subcommands.

The unified test driver replaces ad-hoc ``pytest ...`` invocations
so every contributor (human and AI) hits the same exclusion list,
the same duration cache, and the same "recently-touched" heuristic.

Subcommand semantics pinned here:

  * ``brian test quick``  → run pytest on the 30 most-recently-modified
                            test FILES (by mtime). This is what you
                            want after editing a few test files — it
                            re-checks your active edits without
                            sitting through the full sweep.

  * ``brian test fast``   → run the 30 FASTEST individual tests (by
                            cached duration). Useful as a near-zero-
                            cost smoke check that exercises the bulk
                            of the import graph in <5 s. Falls back to
                            "quick" semantics when no duration cache
                            exists yet.

  * ``brian test full``   → the canonical full sweep. Excludes
                            historically-flaky / very-slow files
                            (``test_feature_flag_ablation``,
                            ``test_brian_compile``, ``tests/training/``)
                            and updates the duration cache so a future
                            ``brian test fast`` has data to chew on.

  * ``brian test [PATTERN]`` (legacy) — preserved for backward
                            compatibility, still runs ``tests/dsl/``
                            by default with the ``slow`` marker
                            gating.

  * ``brian test --help``  prints every subcommand at the top so the
                           CLI is self-documenting.

The duration cache lives at ``.neuro/test_durations.json`` and is
``{nodeid: seconds}``. ``brian test full`` rewrites it from the
``--durations=0`` pytest stream every time it succeeds. ``brian test
fast`` reads it, sorts ascending, takes the first 30, and runs them.

These tests deliberately do NOT invoke pytest as a subprocess (that
would be both slow and recursive). They stub ``_run`` and assert on
the argv that ``cmd_test_*`` would have passed to it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Public API surface
# ──────────────────────────────────────────────────────────────────────


class TestImports:
    def test_cmd_test_quick_is_importable(self):
        from neuroslm.cli import cmd_test_quick  # noqa: F401

    def test_cmd_test_fast_is_importable(self):
        from neuroslm.cli import cmd_test_fast  # noqa: F401

    def test_cmd_test_full_is_importable(self):
        from neuroslm.cli import cmd_test_full  # noqa: F401

    def test_duration_cache_path_constant_exists(self):
        from neuroslm.cli import TEST_DURATIONS_CACHE
        # Must live under .neuro/ so it doesn't pollute the repo root.
        s = str(TEST_DURATIONS_CACHE).replace("\\", "/")
        assert s.endswith(".neuro/test_durations.json"), (
            f"expected cache path to end with '.neuro/test_durations.json', "
            f"got {TEST_DURATIONS_CACHE!r}"
        )

    def test_subparser_registers_quick_fast_full(self):
        """``brian test`` must dispatch ``quick``/``fast``/``full`` to
        the dedicated command functions.

        The ``test`` subcommand uses manual dispatch (not argparse
        sub-subparsers) so it can still accept a legacy pytest
        path/pattern as the first positional. We verify dispatch
        end-to-end by parsing each magic-word invocation and checking
        the resolved Namespace's ``func`` is the right callable.
        """
        from neuroslm.cli import (
            _build_parser,
            cmd_test_quick,
            cmd_test_fast,
            cmd_test_full,
        )

        parser = _build_parser()

        # Capture which underlying command was invoked. The dispatcher
        # called by argparse is an inner function ``_dispatch_test``;
        # we stub _run so it doesn't actually try to run pytest, then
        # check the funcs were reached by patching each.
        for magic, expected in (
            ("quick", cmd_test_quick),
            ("fast", cmd_test_fast),
            ("full", cmd_test_full),
        ):
            called = {}

            def _stub(ns, m=magic):
                called["m"] = m
                return 0

            with patch(f"neuroslm.cli.{expected.__name__}", side_effect=_stub):
                ns = parser.parse_args(["test", magic])
                rc = ns.func(ns)
                assert rc == 0, (
                    f"`brian test {magic}` returned {rc!r}; expected 0 "
                    f"from the stubbed {expected.__name__}"
                )
                assert called.get("m") == magic, (
                    f"`brian test {magic}` did not route to "
                    f"{expected.__name__}"
                )

        # The --help text must still advertise all three magic words
        # so a developer typing `brian test -h` sees them.
        for action in parser._actions:
            if getattr(action, "dest", None) == "cmd":
                test_parser = action.choices.get("test")
                assert test_parser is not None
                desc = (test_parser.description or "") + \
                       " " + (test_parser.format_help() or "")
                for magic in ("quick", "fast", "full"):
                    assert magic in desc, (
                        f"`brian test --help` does not mention "
                        f"{magic!r}; description was {desc!r}"
                    )
                return
        pytest.fail("no `cmd` subparser found in CLI")


# ──────────────────────────────────────────────────────────────────────
# `brian test quick` — 30 most-recently-modified test files
# ──────────────────────────────────────────────────────────────────────


class TestQuick:
    def test_quick_picks_30_most_recent_test_files(self, tmp_path, monkeypatch):
        """When more than 30 ``test_*.py`` exist, quick mode picks
        the 30 with the newest mtime."""
        from neuroslm import cli as cli_mod

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create 50 files with monotonically increasing mtimes.
        # Files [20:50] will be the "30 newest" and must be the ones
        # passed to pytest.
        for i in range(50):
            f = tests_dir / f"test_synthetic_{i:02d}.py"
            f.write_text(
                "def test_noop():\n    assert True\n",
                encoding="utf-8",
            )
            mtime = time.time() + i  # later i → later mtime
            os.utime(f, (mtime, mtime))

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return 0

        monkeypatch.setattr(cli_mod, "_run", fake_run)

        rc = cli_mod.cmd_test_quick(argparse.Namespace(verbose=False))
        assert rc == 0
        cmd = captured["cmd"]
        # pytest invocation
        assert "pytest" in cmd
        # Extract the test-file arguments. They are everything that
        # ends with .py.
        passed_files = [c for c in cmd if c.endswith(".py")]
        assert len(passed_files) == 30, (
            f"expected exactly 30 test files, got {len(passed_files)}: "
            f"{passed_files}"
        )
        # All passed files must be from the latter half (i >= 20).
        for pf in passed_files:
            name = Path(pf).name
            n = int(name.replace("test_synthetic_", "").replace(".py", ""))
            assert n >= 20, (
                f"quick mode picked {name} (n={n}); should only pick "
                f"the 30 newest (n>=20)"
            )

    def test_quick_with_fewer_than_30_runs_them_all(self, tmp_path, monkeypatch):
        """When the tree has fewer than 30 test files, quick passes
        them all (no padding, no error)."""
        from neuroslm import cli as cli_mod

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        for i in range(5):
            (tests_dir / f"test_only_{i}.py").write_text(
                "def test_noop():\n    assert True\n",
                encoding="utf-8",
            )

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return 0

        monkeypatch.setattr(cli_mod, "_run", fake_run)

        rc = cli_mod.cmd_test_quick(argparse.Namespace(verbose=False))
        assert rc == 0
        passed = [c for c in captured["cmd"] if c.endswith(".py")]
        assert len(passed) == 5

    def test_quick_skips_non_test_files(self, tmp_path, monkeypatch):
        """Files not named ``test_*.py`` must be ignored even if their
        mtime is recent."""
        from neuroslm import cli as cli_mod

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        # 3 real test files, 3 distractors.
        (tests_dir / "test_a.py").write_text("def test_x(): pass\n")
        (tests_dir / "test_b.py").write_text("def test_x(): pass\n")
        (tests_dir / "test_c.py").write_text("def test_x(): pass\n")
        (tests_dir / "helper.py").write_text("# not a test\n")
        (tests_dir / "conftest.py").write_text("# fixture file\n")
        (tests_dir / "__init__.py").write_text("")

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run",
            lambda cmd, **kw: captured.update(cmd=cmd) or 0,
        )

        cli_mod.cmd_test_quick(argparse.Namespace(verbose=False))
        passed = [c for c in captured["cmd"] if c.endswith(".py")]
        names = {Path(p).name for p in passed}
        assert names == {"test_a.py", "test_b.py", "test_c.py"}, (
            f"quick mode picked up non-test files: {names}"
        )

    def test_quick_skips_canonical_exclusions(self, tmp_path, monkeypatch):
        """``quick`` mode must skip the same files / dirs that
        ``brian test full`` excludes — so it doesn't trip on
        pre-existing slow or broken files."""
        from neuroslm import cli as cli_mod

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        # Create one of the excluded files (broken on purpose) + one
        # under the excluded ``training/`` dir + one valid file.
        (tests_dir / "test_feature_flag_ablation.py").write_text(
            "= this is broken syntax on purpose ===\n"
        )
        (tests_dir / "test_brian_compile.py").write_text(
            "def test_real(): pass\n"
        )
        (tests_dir / "training").mkdir()
        (tests_dir / "training" / "test_under_training.py").write_text(
            "def test_real(): pass\n"
        )
        (tests_dir / "test_valid.py").write_text("def test_real(): pass\n")

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run",
            lambda cmd, **kw: captured.update(cmd=cmd) or 0,
        )

        cli_mod.cmd_test_quick(argparse.Namespace(verbose=False))
        passed = [c for c in captured["cmd"] if c.endswith(".py")]
        names = {Path(p).name for p in passed}
        assert "test_valid.py" in names, (
            f"quick mode must include the valid test file; got {names}"
        )
        for excluded in (
            "test_feature_flag_ablation.py",
            "test_brian_compile.py",
            "test_under_training.py",
        ):
            assert excluded not in names, (
                f"quick mode picked up {excluded!r} but that path is "
                f"on the canonical exclusion list; got {names}"
            )


# ──────────────────────────────────────────────────────────────────────
# `brian test fast` — 30 fastest cached node ids
# ──────────────────────────────────────────────────────────────────────


class TestFast:
    def test_fast_reads_cache_and_picks_30_fastest(self, tmp_path, monkeypatch):
        """When a duration cache exists, fast mode picks the 30
        nodeids with the smallest durations."""
        from neuroslm import cli as cli_mod

        cache = tmp_path / ".neuro" / "test_durations.json"
        cache.parent.mkdir(parents=True)
        # 100 fake nodeids with i seconds each → fastest 30 are i in [0..29]
        data = {f"tests/test_synthetic.py::test_{i:03d}": float(i)
                for i in range(100)}
        cache.write_text(json.dumps(data), encoding="utf-8")

        monkeypatch.setattr(cli_mod, "TEST_DURATIONS_CACHE", cache)
        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)

        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run",
            lambda cmd, **kw: captured.update(cmd=cmd) or 0,
        )

        rc = cli_mod.cmd_test_fast(argparse.Namespace(verbose=False))
        assert rc == 0
        # Extract nodeids (anything containing ::).
        nodes = [c for c in captured["cmd"] if "::" in c]
        assert len(nodes) == 30, f"expected 30 nodeids, got {len(nodes)}"
        # Every picked nodeid must come from the fastest 30 buckets
        # (i in [0..29]).
        for nid in nodes:
            i = int(nid.split("::test_")[-1])
            assert 0 <= i < 30, (
                f"fast mode picked slow test {nid} (i={i}); should "
                f"only pick i in [0..29]"
            )

    def test_fast_falls_back_to_quick_when_no_cache(self, tmp_path, monkeypatch):
        """When the cache is missing or empty, fast mode delegates
        to quick semantics so the developer still gets a useful run."""
        from neuroslm import cli as cli_mod

        cache = tmp_path / ".neuro" / "test_durations.json"
        # Don't create it.
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_alpha.py").write_text(
            "def test_x(): pass\n", encoding="utf-8"
        )

        monkeypatch.setattr(cli_mod, "TEST_DURATIONS_CACHE", cache)
        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)

        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run",
            lambda cmd, **kw: captured.update(cmd=cmd) or 0,
        )

        rc = cli_mod.cmd_test_fast(argparse.Namespace(verbose=False))
        assert rc == 0
        # In fallback mode, files (not nodeids) are passed.
        files = [c for c in captured["cmd"] if c.endswith(".py")]
        assert len(files) >= 1, "fallback should still run the test file"

    def test_fast_with_empty_cache_falls_back(self, tmp_path, monkeypatch):
        """An empty ``{}`` cache should be treated the same as missing
        (no nodeids to pick → fallback)."""
        from neuroslm import cli as cli_mod

        cache = tmp_path / ".neuro" / "test_durations.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}", encoding="utf-8")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_only.py").write_text(
            "def test_x(): pass\n", encoding="utf-8"
        )

        monkeypatch.setattr(cli_mod, "TEST_DURATIONS_CACHE", cache)
        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)

        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run",
            lambda cmd, **kw: captured.update(cmd=cmd) or 0,
        )

        rc = cli_mod.cmd_test_fast(argparse.Namespace(verbose=False))
        assert rc == 0
        files = [c for c in captured["cmd"] if c.endswith(".py")]
        assert len(files) >= 1


# ──────────────────────────────────────────────────────────────────────
# `brian test full` — canonical full sweep
# ──────────────────────────────────────────────────────────────────────


class TestFull:
    def test_full_runs_pytest_on_tests_dir(self, tmp_path, monkeypatch):
        """Full sweep must target the ``tests/`` directory."""
        from neuroslm import cli as cli_mod

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        cache = tmp_path / ".neuro" / "test_durations.json"
        monkeypatch.setattr(cli_mod, "TEST_DURATIONS_CACHE", cache)

        captured = {}
        # cmd_test_full uses _run_tee (single-pass tee+capture seam).
        monkeypatch.setattr(
            cli_mod, "_run_tee",
            lambda cmd: (captured.update(cmd=cmd) or (0, "")),
        )

        rc = cli_mod.cmd_test_full(argparse.Namespace(verbose=False))
        assert rc == 0
        cmd = captured["cmd"]
        assert "pytest" in cmd
        assert "tests/" in cmd or "tests" in cmd, (
            f"full sweep must target tests/, got cmd={cmd}"
        )

    def test_full_excludes_known_slow_files(self, tmp_path, monkeypatch):
        """The canonical exclusion list must be honoured so the sweep
        finishes in ~8 min and not 40+ min."""
        from neuroslm import cli as cli_mod

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            cli_mod, "TEST_DURATIONS_CACHE",
            tmp_path / ".neuro" / "test_durations.json",
        )

        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run_tee",
            lambda cmd: (captured.update(cmd=cmd) or (0, "")),
        )

        cli_mod.cmd_test_full(argparse.Namespace(verbose=False))
        joined = " ".join(captured["cmd"])
        # The historic exclusions discovered during the 2026-06-15
        # sweep. Anyone removing one of these must update this test
        # AND prove the removed file is fast enough to belong.
        for excl in (
            "tests/test_feature_flag_ablation.py",
            "tests/test_brian_compile.py",
            "tests/training",
        ):
            assert excl in joined, (
                f"full sweep must --ignore {excl} (excluded since "
                f"2026-06-15 sweep); got cmd={joined!r}"
            )

    def test_full_requests_durations_for_cache(self, tmp_path, monkeypatch):
        """Full sweep must pass ``--durations=0`` so we can persist
        every node's wall time into the cache for ``brian test fast``."""
        from neuroslm import cli as cli_mod

        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            cli_mod, "TEST_DURATIONS_CACHE",
            tmp_path / ".neuro" / "test_durations.json",
        )

        captured = {}
        monkeypatch.setattr(
            cli_mod, "_run_tee",
            lambda cmd: (captured.update(cmd=cmd) or (0, "")),
        )

        cli_mod.cmd_test_full(argparse.Namespace(verbose=False))
        joined = " ".join(captured["cmd"])
        assert "--durations=0" in joined, (
            "full sweep must pass --durations=0 to populate the cache"
        )

    def test_full_persists_durations_to_cache(self, tmp_path, monkeypatch):
        """When the pytest run emits a durations block, ``cmd_test_full``
        must parse it and write ``{nodeid: seconds}`` into the cache."""
        from neuroslm import cli as cli_mod

        cache = tmp_path / ".neuro" / "test_durations.json"
        monkeypatch.setattr(cli_mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(cli_mod, "TEST_DURATIONS_CACHE", cache)

        fake_output = (
            "============================== slowest durations ==============================\n"
            "1.23s call     tests/test_foo.py::test_a\n"
            "0.10s call     tests/test_bar.py::test_b\n"
            "================================ 2 passed in 1.45s ================================\n"
        )
        monkeypatch.setattr(
            cli_mod, "_run_tee",
            lambda cmd: (0, fake_output),
        )

        rc = cli_mod.cmd_test_full(argparse.Namespace(verbose=False))
        assert rc == 0
        assert cache.is_file(), (
            "cmd_test_full must write the duration cache when "
            "_run_tee returns a durations block"
        )
        data = json.loads(cache.read_text(encoding="utf-8"))
        assert data["tests/test_foo.py::test_a"] == pytest.approx(1.23)
        assert data["tests/test_bar.py::test_b"] == pytest.approx(0.10)


# ──────────────────────────────────────────────────────────────────────
# Duration parsing helper — the bit that turns pytest's textual
# `--durations=0` block into the JSON cache.
# ──────────────────────────────────────────────────────────────────────


class TestDurationParsing:
    def test_parse_pytest_durations_extracts_nodeid_and_seconds(self):
        from neuroslm.cli import _parse_pytest_durations

        # Realistic excerpt of pytest's ``--durations=0`` output.
        # Lines look like:  ``0.42s call  tests/foo.py::test_bar``
        # Other lines (setup/teardown rows for the same node) start
        # with the same node id; parser must take the MAX.
        output = """
============================== slowest durations ==============================
1.23s call     tests/test_alpha.py::test_one
0.05s setup    tests/test_alpha.py::test_one
0.02s teardown tests/test_alpha.py::test_one
0.10s call     tests/test_beta.py::test_two
0.30s call     tests/test_gamma.py::TestClass::test_three

(20 durations < 0.005s hidden.  Use -vv to show these durations.)
================================ 5 passed in 1.79s ================================
"""
        durations = _parse_pytest_durations(output)
        # Keys are the nodeids; values are the longest single-phase
        # duration in seconds.
        assert durations["tests/test_alpha.py::test_one"] == pytest.approx(1.23)
        assert durations["tests/test_beta.py::test_two"] == pytest.approx(0.10)
        assert durations["tests/test_gamma.py::TestClass::test_three"] == \
            pytest.approx(0.30)
        # Non-duration lines must be skipped.
        assert all("call" not in k and "setup" not in k for k in durations)

    def test_parse_pytest_durations_handles_empty(self):
        from neuroslm.cli import _parse_pytest_durations
        assert _parse_pytest_durations("") == {}
        assert _parse_pytest_durations("nothing here\nat all\n") == {}
