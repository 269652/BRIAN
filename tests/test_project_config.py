# -*- coding: utf-8 -*-
"""Tests for :mod:`neuroslm.project_config` — the workspace-level
``brian.toml`` loader.

The project config picks which architecture / DNA the
training-deploy-colab triplet (and the ``brian nfg`` convenience
command) treats as "current". It is intentionally tiny — six fields
total — and lives next to ``pyproject.toml`` so a one-line edit
re-targets every script in the repo.

Contract (frozen by these tests):

1. ``brian.toml`` is parsed into a :class:`ProjectConfig` with
   six attributes::

       arch          str   path to architecture folder (default
                           "architectures/current")
       dna           str   path to .dna file, or "" if DSL mode
                           (default "")
       nfg_output    str   default ".neuro/nfg.png"
       nfg_format    str   default "png"
       nfg_engine    str   default "dot"
       repo_root     Path  populated from search root, not file contents

2. ``load_project_config()`` walks up from the given start dir
   looking for a ``brian.toml``; if none found, returns a config
   with all default values and ``repo_root`` set to the original
   start dir.

3. ``ProjectConfig.is_dna_mode`` is True iff ``dna`` is set AND
   the file exists.  This is the single switch every training script
   reads.

4. ``ProjectConfig.resolve_arch_path()`` returns an absolute
   :class:`Path` to ``arch`` relative to ``repo_root``.

5. ``ProjectConfig.nfg_output_path()`` returns an absolute
   :class:`Path` to ``nfg_output`` relative to ``repo_root``, with
   parent directory created (the caller is expected to write to it).

6. ``ProjectConfig.training_target()`` returns the tuple
   ``("dna", dna_path)`` in DNA mode and ``("arch", arch_path)``
   otherwise. This is the canonical handoff into the training
   scripts (``vast_train_dsl_loop.sh`` etc.).

7. Env-var overrides:
       BRIAN_ARCH       overrides ``arch``
       BRIAN_DNA        overrides ``dna``
       BRIAN_NFG_OUTPUT overrides ``nfg_output``
   This keeps existing CI / vast.ai shell pipelines working.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. brian.toml parsing
# ---------------------------------------------------------------------------

class TestProjectConfigParsing:
    """``brian.toml`` deserialises into a ``ProjectConfig`` cleanly."""

    def test_module_importable(self):
        """The loader module is importable."""
        from neuroslm.project_config import ProjectConfig, load_project_config  # noqa: F401

    def test_defaults_when_file_missing(self, tmp_path):
        """A folder with no ``brian.toml`` returns sensible defaults."""
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.arch == "architectures/master"
        assert cfg.dna == ""
        assert cfg.nfg_output == ".neuro/nfg.png"
        assert cfg.nfg_format == "png"
        assert cfg.nfg_engine == "dot"
        assert cfg.repo_root == tmp_path.resolve()

    def test_explicit_arch_only(self, tmp_path):
        """A ``brian.toml`` with only an ``arch`` field overrides that one
        field and leaves the rest at defaults."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/my_custom"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.arch == "architectures/my_custom"
        assert cfg.dna == ""
        assert cfg.nfg_output == ".neuro/nfg.png"

    def test_full_config(self, tmp_path):
        """A complete ``brian.toml`` deserialises every field."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\n'
            'arch = "architectures/evol"\n'
            'dna  = "dna/evol/arch.dna"\n'
            '\n'
            '[nfg]\n'
            'output = ".neuro/nfg.svg"\n'
            'format = "svg"\n'
            'engine = "neato"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.arch == "architectures/evol"
        assert cfg.dna == "dna/evol/arch.dna"
        assert cfg.nfg_output == ".neuro/nfg.svg"
        assert cfg.nfg_format == "svg"
        assert cfg.nfg_engine == "neato"

    def test_deploy_scale_default_from_brian_toml(self, tmp_path):
        """``[deploy] scale`` deserialises into ``default_scale`` so the
        scale variant can be SET persistently (mirrors ``[deploy] machine``
        → ``default_machine``) instead of requiring ``--scale`` every run."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[deploy]\n'
            'machine = "A100"\n'
            'scale = "100m"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_scale == "100m"
        assert cfg.default_machine == "A100"

    def test_deploy_scale_empty_when_absent(self, tmp_path):
        """No ``[deploy] scale`` → ``default_scale`` is the empty string
        (connector / arch picks), never a crash."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/SmolLM"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_scale == ""

    def test_deploy_scale_env_override(self, tmp_path, monkeypatch):
        """``BRIAN_DEFAULT_SCALE`` overrides the brian.toml value."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[deploy]\nscale = "100m"\n', encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_DEFAULT_SCALE", "300m")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_scale == "300m"

    def test_walks_up_to_find_config(self, tmp_path):
        """``load_project_config`` walks parent dirs to find brian.toml."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/walked_up"\n',
            encoding="utf-8",
        )
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        cfg = load_project_config(start=deep)
        assert cfg.arch == "architectures/walked_up"
        assert cfg.repo_root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 2. is_dna_mode + path resolution
# ---------------------------------------------------------------------------

class TestModeAndPaths:
    def test_dna_mode_false_when_empty(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path)
        assert cfg.dna == ""
        assert cfg.is_dna_mode is False

    def test_dna_mode_false_when_file_missing(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            dna="dna/does_not_exist.dna")
        assert cfg.is_dna_mode is False

    def test_dna_mode_true_when_file_exists(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        dna_file = tmp_path / "dna" / "evol" / "arch.dna"
        dna_file.parent.mkdir(parents=True)
        dna_file.write_text("stub", encoding="utf-8")
        cfg = ProjectConfig(repo_root=tmp_path,
                            dna="dna/evol/arch.dna")
        assert cfg.is_dna_mode is True

    def test_resolve_arch_path_is_absolute(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            arch="architectures/master")
        resolved = cfg.resolve_arch_path()
        assert resolved.is_absolute()
        assert resolved == (tmp_path / "architectures" / "master").resolve()

    def test_resolve_dna_path_is_absolute(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            dna="dna/evol/arch.dna")
        resolved = cfg.resolve_dna_path()
        assert resolved.is_absolute()
        assert resolved == (tmp_path / "dna" / "evol" / "arch.dna").resolve()

    def test_resolve_dna_path_returns_none_when_empty(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path, dna="")
        assert cfg.resolve_dna_path() is None

    def test_nfg_output_path_creates_parent(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            nfg_output=".neuro/nfg.png")
        out = cfg.nfg_output_path()
        assert out == (tmp_path / ".neuro" / "nfg.png").resolve()
        assert out.parent.is_dir()  # parent created on access


# ---------------------------------------------------------------------------
# 3. training_target dispatch
# ---------------------------------------------------------------------------

class TestTrainingTarget:
    def test_dsl_mode_when_no_dna(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            arch="architectures/master")
        kind, path = cfg.training_target()
        assert kind == "arch"
        assert path == cfg.resolve_arch_path()

    def test_dna_mode_when_dna_set(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        dna_file = tmp_path / "dna" / "evol" / "arch.dna"
        dna_file.parent.mkdir(parents=True)
        dna_file.write_text("stub", encoding="utf-8")
        cfg = ProjectConfig(repo_root=tmp_path,
                            arch="architectures/master",
                            dna="dna/evol/arch.dna")
        kind, path = cfg.training_target()
        assert kind == "dna"
        assert path == cfg.resolve_dna_path()


# ---------------------------------------------------------------------------
# 4. Env-var overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    """Existing vast.ai shell pipelines pass ``ARCH=...`` and ``DNA=...``
    via the environment; the loader honours those for backwards compat
    (under a ``BRIAN_`` prefix so we don't collide with arbitrary
    user vars)."""

    def test_brian_arch_env_overrides_file(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/from_file"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_ARCH", "architectures/from_env")
        cfg = load_project_config(start=tmp_path)
        assert cfg.arch == "architectures/from_env"

    def test_brian_dna_env_overrides_file(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[current]\ndna = "dna/from_file.dna"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_DNA", "dna/from_env.dna")
        cfg = load_project_config(start=tmp_path)
        assert cfg.dna == "dna/from_env.dna"

    def test_brian_nfg_output_env_overrides_file(
            self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[nfg]\noutput = ".neuro/from_file.png"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_NFG_OUTPUT", ".neuro/from_env.svg")
        cfg = load_project_config(start=tmp_path)
        assert cfg.nfg_output == ".neuro/from_env.svg"
        # format auto-derived from the extension when not explicitly set
        # (sub-test for the nice-to-have helper)
        assert cfg.nfg_format == "svg"


# ---------------------------------------------------------------------------
# 5. Real repo smoke — the file at the workspace root parses cleanly
# ---------------------------------------------------------------------------

class TestRealRepoConfig:
    """The committed ``brian.toml`` at the workspace root must parse and
    must point ``arch`` at a folder that actually contains an
    ``arch.neuro``. Catches typos before they make it into a deploy."""

    def test_repo_brian_toml_parses(self):
        from neuroslm.project_config import load_project_config
        repo_root = Path(__file__).resolve().parent.parent
        if not (repo_root / "brian.toml").is_file():
            pytest.skip("brian.toml not yet present at repo root")
        cfg = load_project_config(start=repo_root)
        assert cfg.arch  # non-empty
        assert cfg.nfg_output  # non-empty

    def test_repo_brian_toml_arch_points_to_real_folder(self):
        from neuroslm.project_config import load_project_config
        repo_root = Path(__file__).resolve().parent.parent
        if not (repo_root / "brian.toml").is_file():
            pytest.skip("brian.toml not yet present at repo root")
        cfg = load_project_config(start=repo_root)
        arch_dir = cfg.resolve_arch_path()
        assert arch_dir.is_dir(), \
            f"brian.toml [current].arch = {cfg.arch!r} but " \
            f"{arch_dir} is not a directory"
        assert (arch_dir / "arch.neuro").is_file(), \
            f"{arch_dir}/arch.neuro is missing — the configured " \
            f"current arch is incomplete"


# ---------------------------------------------------------------------------
# 6. heat-overlay output path helper
# ---------------------------------------------------------------------------

class TestHeatOutputPath:
    """When ``brian compile nfg --current --heat`` is invoked, the
    output file must be derived from ``nfg_output`` by inserting
    ``.heat`` before the extension. The config object owns this
    derivation so the CLI and the parallel-session heat implementation
    don't drift apart on the file-name convention."""

    def test_heat_output_default(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            nfg_output=".neuro/nfg.png")
        out = cfg.nfg_output_path(heat=True)
        assert out == (tmp_path / ".neuro" / "nfg.heat.png").resolve()
        assert out.parent.is_dir()

    def test_heat_output_svg(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            nfg_output=".neuro/nfg.svg")
        out = cfg.nfg_output_path(heat=True)
        assert out == (tmp_path / ".neuro" / "nfg.heat.svg").resolve()

    def test_heat_output_handles_dotted_stem(self, tmp_path):
        """If the user already put dots in the stem (e.g.
        ``arch.v2.png``), the heat infix goes immediately before the
        final extension."""
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            nfg_output=".neuro/arch.v2.png")
        out = cfg.nfg_output_path(heat=True)
        assert out.name == "arch.v2.heat.png"

    def test_non_heat_path_unchanged(self, tmp_path):
        """``heat=False`` (the default) returns the plain configured
        output unchanged."""
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(repo_root=tmp_path,
                            nfg_output=".neuro/nfg.png")
        assert cfg.nfg_output_path() == cfg.nfg_output_path(heat=False)
        assert cfg.nfg_output_path().name == "nfg.png"


# ---------------------------------------------------------------------------
# 7. CLI integration — `brian compile nfg --current [--heat]`
# ---------------------------------------------------------------------------

class TestCompileNfgCurrentFlag:
    """``brian compile nfg --current`` reads the configured arch / DNA
    from ``brian.toml`` and writes to the configured ``nfg_output``.
    With ``--heat`` it writes to the ``.heat.<ext>`` sibling instead.

    These tests stub out the heavy hypergraph render — we only care
    that the CLI plumbs the resolved arch + the resolved output path
    through correctly. The actual render is covered by
    ``tests/test_nfg_graphviz.py``.
    """

    def test_current_flag_resolves_arch_from_brian_toml(
            self, tmp_path, monkeypatch):
        """With ``--current`` and no positional arg, the CLI reads
        ``[current].arch`` and renders that architecture."""
        from neuroslm import project_config
        from neuroslm import cli as _cli

        # Build a fake workspace with brian.toml + an arch folder.
        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/master"\n'
            '[nfg]\noutput = ".neuro/nfg.png"\n',
            encoding="utf-8",
        )
        arch_dir = tmp_path / "architectures" / "master"
        arch_dir.mkdir(parents=True)
        (arch_dir / "arch.neuro").write_text(
            'architecture master { d_sem: 4, dt: 0.01 }\n',
            encoding="utf-8",
        )

        # Force the loader to anchor at our tmp workspace.
        monkeypatch.setattr(
            project_config, "_default_start_dir",
            lambda: tmp_path, raising=False,
        )

        captured: dict = {}

        def _fake_render(ir, out_path, **kw):
            captured["out_path"] = str(out_path)
            captured["format"] = kw.get("format")
            return str(out_path)

        def _fake_lift(arch):
            captured["arch"] = str(arch)
            class _IR:
                nodes = []
                hyperedges = []
            return _IR()

        monkeypatch.setattr(
            "neuroslm.compiler.nfg_graphviz.render_hypergraph",
            _fake_render, raising=False,
        )
        monkeypatch.setattr(
            "neuroslm.compiler.hypergraph_ir.lift_arch_to_hypergraph",
            _fake_lift, raising=False,
        )

        import argparse
        args = argparse.Namespace(
            arch=None, current=True, heat=False,
            out=None, png=None,
            semantic=False, legacy=False,
            engine="dot", format="png",
        )
        rc = _cli.cmd_compile_nfg(args)
        assert rc == 0
        # The arch that got rendered is the configured one.
        assert "master" in captured["arch"]
        # The output landed at the configured nfg_output, NOT inside
        # the arch folder.
        assert captured["out_path"].endswith(
            str(Path(".neuro") / "nfg.png"))

    def test_current_flag_with_heat_writes_heat_sibling(
            self, tmp_path, monkeypatch):
        """``--current --heat`` writes to ``<nfg_output>`` with
        ``.heat`` inserted before the extension."""
        from neuroslm import project_config
        from neuroslm import cli as _cli

        (tmp_path / "brian.toml").write_text(
            '[current]\narch = "architectures/master"\n'
            '[nfg]\noutput = ".neuro/nfg.png"\n',
            encoding="utf-8",
        )
        arch_dir = tmp_path / "architectures" / "master"
        arch_dir.mkdir(parents=True)
        (arch_dir / "arch.neuro").write_text(
            'architecture master { d_sem: 4, dt: 0.01 }\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(
            project_config, "_default_start_dir",
            lambda: tmp_path, raising=False,
        )

        captured: dict = {}

        def _fake_render(ir, out_path, **kw):
            captured["out_path"] = str(out_path)
            return str(out_path)

        def _fake_lift(arch):
            class _IR:
                nodes = []
                hyperedges = []
            return _IR()

        monkeypatch.setattr(
            "neuroslm.compiler.nfg_graphviz.render_hypergraph",
            _fake_render, raising=False,
        )
        monkeypatch.setattr(
            "neuroslm.compiler.hypergraph_ir.lift_arch_to_hypergraph",
            _fake_lift, raising=False,
        )

        import argparse
        args = argparse.Namespace(
            arch=None, current=True, heat=True,
            out=None, png=None,
            semantic=False, legacy=False,
            engine="dot", format="png",
        )
        rc = _cli.cmd_compile_nfg(args)
        assert rc == 0
        # Output is the .heat. sibling, NOT plain nfg.png.
        assert captured["out_path"].endswith(
            str(Path(".neuro") / "nfg.heat.png")), \
            f"expected …/nfg.heat.png, got {captured['out_path']}"

    def test_current_and_positional_arch_are_mutually_exclusive(
            self, tmp_path):
        """Passing both ``--current`` and a positional arch is a
        user error — fail cleanly with a non-zero exit code."""
        from neuroslm import cli as _cli
        import argparse
        args = argparse.Namespace(
            arch="architectures/master",
            current=True, heat=False,
            out=None, png=None,
            semantic=False, legacy=False,
            engine="dot", format="png",
        )
        rc = _cli.cmd_compile_nfg(args)
        assert rc != 0

    def test_heat_without_current_is_allowed(
            self, tmp_path, monkeypatch):
        """``--heat`` can be combined with a positional arch (no
        ``--current``) — it just renders the standard NFG with the
        heat overlay, no filename change."""
        from neuroslm import cli as _cli

        arch_dir = tmp_path / "architectures" / "master"
        arch_dir.mkdir(parents=True)
        (arch_dir / "arch.neuro").write_text(
            'architecture master { d_sem: 4, dt: 0.01 }\n',
            encoding="utf-8",
        )

        captured: dict = {}

        def _fake_render(ir, out_path, **kw):
            captured["out_path"] = str(out_path)
            captured["heat"] = kw.get("heat")
            return str(out_path)

        def _fake_lift(arch):
            class _IR:
                nodes = []
                hyperedges = []
            return _IR()

        monkeypatch.setattr(
            "neuroslm.compiler.nfg_graphviz.render_hypergraph",
            _fake_render, raising=False,
        )
        monkeypatch.setattr(
            "neuroslm.compiler.hypergraph_ir.lift_arch_to_hypergraph",
            _fake_lift, raising=False,
        )

        import argparse
        heat_json = tmp_path / "heatmap.json"
        heat_json.write_text("{}", encoding="utf-8")
        args = argparse.Namespace(
            arch=str(arch_dir), current=False, heat=str(heat_json),
            out=None, png=None,
            semantic=False, legacy=False,
            engine="dot", format="png",
        )
        rc = _cli.cmd_compile_nfg(args)
        assert rc == 0
        # Heat payload propagated to the renderer.
        assert captured["heat"] == str(heat_json)
        # No ``.heat`` infix on the standalone path — only --current
        # triggers the configured-output rename.
        assert ".heat." not in captured["out_path"], \
            f"unexpected .heat infix on standalone path: {captured['out_path']}"
