# -*- coding: utf-8 -*-
"""TDD spec for the global-defaults + sticky-override system.

User requirement (literal):

    I can define it globally as fallback; the individual archs should
    still be able to overwrite it by appending a ``!`` to the respective
    config field of the arch.

Resolved precedence (high → low):

    1. CLI flag                          (`brian train --preset xxx`)
    2. Arch sticky                       (`preset!: "xxx"`)
    3. Global default                    (`brian.toml  [defaults] preset = "xxx"`)
    4. Arch default                      (`preset: "xxx"`)
    5. Built-in default                  (empty string / built-in fallback)

Three layers to lock down:

    A. **Parser** — `_split_top_level_kv_with_stickies()` recognises
       ``key!: value``, returns ``({"key": "value"}, {"key"})``.
       The legacy ``_split_top_level_kv`` shim must keep stripping the
       ``!`` so existing callers stay working.

    B. **ProjectConfig** — gains ``default_preset`` and
       ``default_hardware`` fields, read from a new ``[defaults]``
       section in ``brian.toml``; ``BRIAN_DEFAULT_PRESET`` and
       ``BRIAN_DEFAULT_HARDWARE`` env overrides.

    C. **apply_global_defaults()** — merges a ``ProjectConfig`` into a
       parsed ``TrainingConfig`` in place; returns the list of changes
       it made so the CLI can announce them.  Honours
       ``TrainingConfig._sticky_fields``.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# A. Parser — `!` sticky-override syntax
# ─────────────────────────────────────────────────────────────────────

class TestStickyParserSyntax:
    """``field!: value`` parses as bare ``field`` + a sticky flag."""

    def test_bang_strips_to_bare_key(self):
        from neuroslm.dsl.training_config import (
            _split_top_level_kv_with_stickies,
        )
        props, sticky = _split_top_level_kv_with_stickies(
            'preset!: "cheap_2k"'
        )
        assert "preset" in props
        assert props["preset"].strip('"') == "cheap_2k"
        assert "preset" in sticky

    def test_no_bang_means_not_sticky(self):
        from neuroslm.dsl.training_config import (
            _split_top_level_kv_with_stickies,
        )
        props, sticky = _split_top_level_kv_with_stickies(
            'preset: "cheap_2k"'
        )
        assert "preset" in props
        assert "preset" not in sticky

    def test_mixed_sticky_and_non_sticky_keys(self):
        from neuroslm.dsl.training_config import (
            _split_top_level_kv_with_stickies,
        )
        body = (
            'preset!: "cheap_2k"\n'
            'optimizer: "adamw"\n'
            'learning_rate!: 0.001\n'
        )
        props, sticky = _split_top_level_kv_with_stickies(body)
        assert sticky == {"preset", "learning_rate"}
        assert "optimizer" not in sticky
        # values still parsed correctly
        assert props["preset"].strip('"') == "cheap_2k"
        assert props["optimizer"].strip('"') == "adamw"
        assert props["learning_rate"] == "0.001"

    def test_legacy_split_strips_bang_and_returns_bare_key(self):
        """``_split_top_level_kv`` (legacy shim) must not leak ``!``
        into key names, otherwise every existing parser would silently
        ignore sticky-tagged keys."""
        from neuroslm.dsl.training_config import _split_top_level_kv
        props = _split_top_level_kv('preset!: "cheap_2k"')
        assert "preset" in props
        assert "preset!" not in props
        assert props["preset"].strip('"') == "cheap_2k"


class TestTrainingConfigTracksStickies:
    """Parsed configs carry which fields the arch insisted on with ``!``."""

    def test_default_config_has_empty_sticky_set(self):
        from neuroslm.dsl.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "_sticky_fields")
        assert cfg._sticky_fields == set()

    def test_no_bang_does_not_mark_sticky(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config('preset: "cheap_2k"')
        assert cfg.preset == "cheap_2k"
        assert "preset" not in cfg._sticky_fields

    def test_bang_marks_field_sticky(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config('preset!: "cheap_2k"')
        assert cfg.preset == "cheap_2k"
        assert "preset" in cfg._sticky_fields

    def test_bang_on_optimizer_marks_optimizer_sticky(self):
        from neuroslm.dsl.training_config import parse_training_config
        cfg = parse_training_config('optimizer!: "adamw"')
        assert cfg.optimizer == "adamw"
        assert "optimizer" in cfg._sticky_fields


# ─────────────────────────────────────────────────────────────────────
# B. ProjectConfig — new global defaults
# ─────────────────────────────────────────────────────────────────────

class TestProjectConfigDefaults:
    """``brian.toml`` ``[defaults]`` section feeds preset/hardware
    fallbacks. Existing ``[current]`` for arch/dna is untouched."""

    def test_defaults_section_parsed(self, tmp_path):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            "[defaults]\n"
            'preset = "cheap_2k"\n'
            'hardware = "RTX_3090"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_preset == "cheap_2k"
        assert cfg.default_hardware == "RTX_3090"

    def test_missing_defaults_section_returns_empty_strings(self, tmp_path):
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_preset == ""
        assert cfg.default_hardware == ""

    def test_defaults_dataclass_has_fields(self, tmp_path):
        from neuroslm.project_config import ProjectConfig
        cfg = ProjectConfig(
            repo_root=tmp_path,
            default_preset="cheap_2k",
            default_hardware="RTX_3090",
        )
        assert cfg.default_preset == "cheap_2k"
        assert cfg.default_hardware == "RTX_3090"

    def test_env_overrides_default_preset(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            '[defaults]\npreset = "from_file"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("BRIAN_DEFAULT_PRESET", "from_env")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_preset == "from_env"

    def test_env_overrides_default_hardware(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        monkeypatch.setenv("BRIAN_DEFAULT_HARDWARE", "A100_SXM4")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_hardware == "A100_SXM4"

    def test_arch_and_dna_still_under_current_section(self, tmp_path):
        """The new [defaults] section must coexist with the existing
        [current] section — neither should clobber the other."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            "[current]\n"
            'arch = "architectures/evol"\n'
            'dna  = "dna/evol/arch.dna"\n'
            "\n"
            "[defaults]\n"
            'preset   = "cheap_2k"\n'
            'hardware = "RTX_3090"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.arch == "architectures/evol"
        assert cfg.dna == "dna/evol/arch.dna"
        assert cfg.default_preset == "cheap_2k"
        assert cfg.default_hardware == "RTX_3090"


# ─────────────────────────────────────────────────────────────────────
# C. apply_global_defaults() — precedence resolution
# ─────────────────────────────────────────────────────────────────────

class TestApplyGlobalDefaults:
    """Merge a ``ProjectConfig`` into a parsed ``TrainingConfig`` in
    place; honour stickies."""

    def test_global_fills_when_arch_unset(self, tmp_path):
        from neuroslm.dsl.training_config import (
            TrainingConfig, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()                              # preset = ""
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        apply_global_defaults(arch, glb)
        assert arch.preset == "cheap_2k"

    def test_global_overrides_arch_default(self, tmp_path):
        """DEPRECATED semantics — kept as a regression guard for the
        new precedence (see test_hardware_aware_preset_selection.py).

        After the 2026-06-12 precedence flip, the arch's non-empty value
        ALWAYS wins over the global default, with or without ``!``.
        The global is now ONLY a fallback for empty arch fields.
        """
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')   # no `!`
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        apply_global_defaults(arch, glb)
        # Arch wins — the global cannot stomp it any more.
        assert arch.preset == "30m_p4"

    def test_sticky_arch_resists_global(self, tmp_path):
        """``preset!: "..."`` pins the arch — the global cannot stomp it."""
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset!: "30m_p4"')  # sticky
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        apply_global_defaults(arch, glb)
        assert arch.preset == "30m_p4"

    def test_empty_global_leaves_arch_alone(self, tmp_path):
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')
        glb = ProjectConfig(repo_root=tmp_path)            # all empty
        apply_global_defaults(arch, glb)
        assert arch.preset == "30m_p4"

    def test_returns_change_log(self, tmp_path):
        """``apply_global_defaults`` returns ``[(field, old, new), ...]``
        so the CLI can announce::

            preset: '' → cheap_2k (from brian.toml [defaults])

        Under the 2026-06-12 precedence (arch wins), a non-empty arch
        is silently left alone, so the change-log entry only fires when
        the global FILLS an empty arch field.
        """
        from neuroslm.dsl.training_config import (
            TrainingConfig, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()  # preset = "" (empty default)
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        changes = apply_global_defaults(arch, glb)
        assert ("preset", "", "cheap_2k") in changes

    def test_sticky_not_in_change_log(self, tmp_path):
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset!: "30m_p4"')
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        changes = apply_global_defaults(arch, glb)
        assert changes == []

    def test_returns_empty_list_when_no_changes(self, tmp_path):
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "cheap_2k"')
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        changes = apply_global_defaults(arch, glb)
        assert changes == []


# ─────────────────────────────────────────────────────────────────────
# D. CLI resolution helper — top-level precedence wiring
# ─────────────────────────────────────────────────────────────────────

class TestResolveEffectivePreset:
    """``_resolve_effective_preset(cli_preset, arch_cfg, project_cfg)``
    folds all four precedence layers into one decision and returns
    ``(effective_preset, change_log)``.

    Layer order (high → low):
        1. CLI flag
        2. Arch sticky
        3. Global default
        4. Arch default
    """

    def test_cli_wins_over_everything(self, tmp_path):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset!: "30m_p4"')   # sticky
        glb = ProjectConfig(repo_root=tmp_path, default_preset="cheap_2k")
        eff, changes = _resolve_effective_preset(
            cli_preset="t4_2k", arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "t4_2k"
        assert changes == []   # CLI is silent — no global merge happened

    def test_arch_sticky_beats_global(self, tmp_path):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset!: "30m_p4"')
        glb = ProjectConfig(repo_root=tmp_path, default_preset="cheap_2k")
        eff, changes = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "30m_p4"
        assert changes == []

    def test_global_beats_arch_default(self, tmp_path):
        """DEPRECATED — see ``test_hardware_aware_preset_selection.py``
        for the new precedence semantics.  Under 2026-06-12, the arch's
        non-empty value beats the global. This test now pins the new
        behaviour: ``preset: "30m_p4"`` (no ``!``) still wins.
        """
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')   # NOT sticky
        glb = ProjectConfig(repo_root=tmp_path, default_preset="cheap_2k")
        eff, changes = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "30m_p4"
        assert changes == []   # arch already had a value → no merge

    def test_arch_default_when_no_global(self, tmp_path):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')
        glb = ProjectConfig(repo_root=tmp_path)   # no global preset
        eff, changes = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "30m_p4"
        assert changes == []

    def test_returns_none_when_all_layers_empty(self, tmp_path, monkeypatch):
        """DEPRECATED — under the 2026-06-12 precedence the AUTO layer
        (hardware detection + VRAM fallback) means the resolver never
        actually returns ``None`` on a real machine. This test pins the
        new behaviour: with CUDA mocked off, AUTO returns ``"tiny"``.

        See ``tests/test_hardware_aware_preset_selection.py``
        ``TestResolveEffectivePresetNewPrecedence::test_auto_detection_picks_tiny_on_cpu``.
        """
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        from neuroslm import hardware
        monkeypatch.setattr(hardware, "_cuda_is_available", lambda: False)
        eff, _changes = _resolve_effective_preset(
            cli_preset=None,
            arch_cfg=TrainingConfig(),
            project_cfg=ProjectConfig(repo_root=tmp_path),
        )
        assert eff == "tiny"   # AUTO → CPU bucket
