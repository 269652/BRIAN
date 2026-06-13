# -*- coding: utf-8 -*-
"""TDD spec — hardware-aware preset selection with reversed precedence.

User requirement (literal, 2026-06-12):

    arch.neuro takes precedence over brian.toml and cli flags overwrite
    both arch.neuro and brian.toml

    [A100]  preset=large
    [T4]    preset=mid
    [CPU]   preset=tiny

    For legacy behaviour there should also be automatic hardware
    detection and choose preset so that it fits into VRAM.

Resolved precedence (high → low):

    1. CLI flag                          (`brian train --preset xxx`)
    2. Arch.neuro                        (`preset: "xxx"` — any non-empty)
    3. brian.toml [hardware.<DETECTED>]  (per-hardware override)
    4. brian.toml [defaults]             (workspace-wide fallback)
    5. Auto: detect_hardware()           (read GPU type from torch.cuda)
       → look up [hardware.<NAME>] preset
       → else pick_preset_for_vram(detected_gib)
    6. Built-in fallback                 ("rcc_bowtie_30m_p4")

NOTE: This **inverts** the precedence between arch and brian.toml
relative to the previous sticky-overrides spec.  The `!` parser support
stays (still useful as a forward-compat marker) but is no longer
consulted because arch ALWAYS wins now.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────
# A. Hardware detection — normalise torch device names to canonical IDs
# ─────────────────────────────────────────────────────────────────────

class TestNormaliseGpuName:
    """Map raw ``torch.cuda.get_device_name(0)`` output to a canonical
    hardware key that matches a ``brian.toml`` ``[hardware.<X>]`` block."""

    def test_a100_variants(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("NVIDIA A100-SXM4-40GB") == "A100"
        assert normalise_gpu_name("A100-PCIE-80GB") == "A100"
        assert normalise_gpu_name("NVIDIA A100 80GB PCIe") == "A100"

    def test_t4(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("Tesla T4") == "T4"
        assert normalise_gpu_name("NVIDIA T4") == "T4"

    def test_rtx_3090(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("NVIDIA GeForce RTX 3090") == "RTX_3090"

    def test_rtx_4090(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("NVIDIA GeForce RTX 4090") == "RTX_4090"

    def test_v100(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("Tesla V100-SXM2-32GB") == "V100"

    def test_h100(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("NVIDIA H100 80GB HBM3") == "H100"

    def test_unknown_returns_normalised_string(self):
        from neuroslm.hardware import normalise_gpu_name
        # Unknown GPUs are upper-cased, spaces→underscores
        assert normalise_gpu_name("Quadro RTX 6000") == "QUADRO_RTX_6000"

    def test_empty_string_returns_cpu(self):
        from neuroslm.hardware import normalise_gpu_name
        assert normalise_gpu_name("") == "CPU"


class TestDetectHardware:
    """``detect_hardware()`` returns the canonical name of the current
    GPU, or ``"CPU"`` when no CUDA device is available."""

    def test_returns_cpu_when_cuda_unavailable(self, monkeypatch):
        from neuroslm import hardware
        monkeypatch.setattr(
            hardware, "_cuda_is_available", lambda: False
        )
        assert hardware.detect_hardware() == "CPU"

    def test_returns_normalised_gpu_name(self, monkeypatch):
        from neuroslm import hardware
        monkeypatch.setattr(
            hardware, "_cuda_is_available", lambda: True
        )
        monkeypatch.setattr(
            hardware, "_cuda_device_name", lambda: "NVIDIA A100-SXM4-80GB"
        )
        assert hardware.detect_hardware() == "A100"


class TestDetectVram:
    """``detect_vram_gib()`` returns the active CUDA device's total
    memory in GiB; ``0.0`` on CPU."""

    def test_returns_zero_on_cpu(self, monkeypatch):
        from neuroslm import hardware
        monkeypatch.setattr(
            hardware, "_cuda_is_available", lambda: False
        )
        assert hardware.detect_vram_gib() == 0.0

    def test_returns_float_on_gpu(self, monkeypatch):
        from neuroslm import hardware
        monkeypatch.setattr(
            hardware, "_cuda_is_available", lambda: True
        )
        # 40 GiB in bytes
        monkeypatch.setattr(
            hardware, "_cuda_total_memory_bytes",
            lambda: 40 * 1024 ** 3,
        )
        assert abs(hardware.detect_vram_gib() - 40.0) < 0.01


class TestPickPresetForVram:
    """Heuristic that picks a preset based on available VRAM. Used as
    the last-ditch fallback when nothing else is configured."""

    def test_cpu_zero_vram_picks_tiny(self):
        from neuroslm.hardware import pick_preset_for_vram
        assert pick_preset_for_vram(0.0) == "tiny"

    def test_8gib_picks_t4_2k_or_similar_small(self):
        from neuroslm.hardware import pick_preset_for_vram
        out = pick_preset_for_vram(8.0)
        # Anything in the small bucket is fine — pin to t4_2k as the
        # documented "small CUDA" preset.
        assert out == "t4_2k"

    def test_24gib_picks_cheap_2k(self):
        """RTX 3090 / RTX 4090 fall in the 24 GiB bucket."""
        from neuroslm.hardware import pick_preset_for_vram
        assert pick_preset_for_vram(24.0) == "cheap_2k"

    def test_40gib_picks_large_class(self):
        """A100 40 GiB → large."""
        from neuroslm.hardware import pick_preset_for_vram
        out = pick_preset_for_vram(40.0)
        assert out in {"large", "30m_p4", "100m"}

    def test_80gib_picks_xl_or_larger(self):
        """A100 80 GiB / H100."""
        from neuroslm.hardware import pick_preset_for_vram
        out = pick_preset_for_vram(80.0)
        assert out in {"xl", "100m", "large"}

    def test_monotone_non_decreasing(self):
        """Bigger VRAM never picks a smaller preset (sanity guard)."""
        from neuroslm.hardware import pick_preset_for_vram, _PRESET_VRAM_TIERS
        # Make sure the tier table itself is sorted
        last_vram = -1.0
        for vram_floor, _preset in _PRESET_VRAM_TIERS:
            assert vram_floor >= last_vram, (
                f"_PRESET_VRAM_TIERS not sorted at {vram_floor}"
            )
            last_vram = vram_floor


# ─────────────────────────────────────────────────────────────────────
# B. ProjectConfig — new hardware/steps/hardware_presets fields
# ─────────────────────────────────────────────────────────────────────

class TestProjectConfigHardwarePresets:
    """``brian.toml`` ``[hardware.<NAME>]`` sections feed a
    ``ProjectConfig.hardware_presets`` mapping."""

    def test_hardware_presets_parsed(self, tmp_path):
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            "[hardware.A100]\n"
            'preset = "large"\n'
            "[hardware.T4]\n"
            'preset = "t4_2k"\n'
            "[hardware.CPU]\n"
            'preset = "tiny"\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.hardware_presets == {
            "A100": "large",
            "T4":   "t4_2k",
            "CPU":  "tiny",
        }

    def test_missing_section_returns_empty_dict(self, tmp_path):
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        assert cfg.hardware_presets == {}

    def test_default_hardware_field(self, tmp_path):
        """The workspace's ACTIVE hardware (drives both deploy and the
        per-hardware preset lookup) lives in ``[defaults] hardware``."""
        from neuroslm.project_config import load_project_config
        (tmp_path / "brian.toml").write_text(
            "[defaults]\n"
            'hardware = "A100"\n'
            'steps = 2000\n',
            encoding="utf-8",
        )
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_hardware == "A100"
        assert cfg.default_steps == 2000

    def test_default_steps_default_value(self, tmp_path):
        from neuroslm.project_config import load_project_config
        cfg = load_project_config(start=tmp_path)
        # 0 means "no opinion — caller picks"; non-positive sentinel.
        assert cfg.default_steps == 0

    def test_env_overrides_default_steps(self, tmp_path, monkeypatch):
        from neuroslm.project_config import load_project_config
        monkeypatch.setenv("BRIAN_DEFAULT_STEPS", "5000")
        cfg = load_project_config(start=tmp_path)
        assert cfg.default_steps == 5000


# ─────────────────────────────────────────────────────────────────────
# C. apply_global_defaults — NEW precedence (arch always wins)
# ─────────────────────────────────────────────────────────────────────

class TestApplyGlobalDefaultsArchWins:
    """REVERSED from the previous spec — arch's non-empty value always
    beats the global. Global is now ONLY a fallback for empty fields."""

    def test_arch_value_beats_global(self, tmp_path):
        """Even WITHOUT ``!``, an arch's ``preset: "30m_p4"`` resists
        the global ``cheap_2k``."""
        from neuroslm.dsl.training_config import (
            parse_training_config, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        apply_global_defaults(arch, glb)
        assert arch.preset == "30m_p4"   # arch wins

    def test_global_fills_when_arch_silent(self, tmp_path):
        """When the arch sets nothing, the global fills in."""
        from neuroslm.dsl.training_config import (
            TrainingConfig, apply_global_defaults,
        )
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()  # preset = "" (default)
        glb = ProjectConfig(
            repo_root=tmp_path, default_preset="cheap_2k"
        )
        apply_global_defaults(arch, glb)
        assert arch.preset == "cheap_2k"


# ─────────────────────────────────────────────────────────────────────
# D. _resolve_effective_preset — full precedence + hardware lookup
# ─────────────────────────────────────────────────────────────────────

class TestResolveEffectivePresetNewPrecedence:

    def test_cli_wins_over_everything(self, tmp_path):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')
        glb = ProjectConfig(
            repo_root=tmp_path,
            default_preset="cheap_2k",
            default_hardware="A100",
            hardware_presets={"A100": "large"},
        )
        eff, _ = _resolve_effective_preset(
            cli_preset="t4_2k", arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "t4_2k"

    def test_arch_beats_hardware_map(self, tmp_path):
        """``preset: "30m_p4"`` on the arch beats
        ``[hardware.A100] preset = "large"``."""
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config('preset: "30m_p4"')
        glb = ProjectConfig(
            repo_root=tmp_path,
            default_hardware="A100",
            hardware_presets={"A100": "large"},
        )
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "30m_p4"

    def test_hardware_map_beats_default_preset(self, tmp_path):
        """When the arch is silent, ``[hardware.A100] preset = "large"``
        wins over ``[defaults] preset = "cheap_2k"``."""
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()
        glb = ProjectConfig(
            repo_root=tmp_path,
            default_preset="cheap_2k",
            default_hardware="A100",
            hardware_presets={"A100": "large"},
        )
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "large"

    def test_default_preset_used_when_no_hardware_match(self, tmp_path):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()
        glb = ProjectConfig(
            repo_root=tmp_path,
            default_preset="cheap_2k",
            default_hardware="QUADRO_RTX_6000",   # no entry in map
            hardware_presets={"A100": "large"},
        )
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "cheap_2k"

    def test_auto_detection_used_when_nothing_configured(
            self, tmp_path, monkeypatch):
        """When neither CLI, arch, nor brian.toml set anything,
        auto-detect hardware + VRAM and pick a fitting preset."""
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        from neuroslm import hardware
        # Pretend we're running on a fresh 24 GiB RTX 3090
        monkeypatch.setattr(hardware, "_cuda_is_available", lambda: True)
        monkeypatch.setattr(
            hardware, "_cuda_device_name",
            lambda: "NVIDIA GeForce RTX 3090",
        )
        monkeypatch.setattr(
            hardware, "_cuda_total_memory_bytes",
            lambda: 24 * 1024 ** 3,
        )
        arch = TrainingConfig()
        glb = ProjectConfig(repo_root=tmp_path)   # all empty
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        # RTX 3090 has no [hardware.RTX_3090] entry → VRAM-based fallback
        # → 24 GiB tier → cheap_2k.
        assert eff == "cheap_2k"

    def test_auto_detection_picks_tiny_on_cpu(
            self, tmp_path, monkeypatch):
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        from neuroslm import hardware
        monkeypatch.setattr(hardware, "_cuda_is_available", lambda: False)
        arch = TrainingConfig()
        glb = ProjectConfig(repo_root=tmp_path)
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "tiny"

    def test_auto_detection_uses_hardware_map_when_present(
            self, tmp_path, monkeypatch):
        """When hardware auto-detected matches a [hardware.<NAME>] entry,
        use that entry (not the VRAM fallback)."""
        from neuroslm.cli import _resolve_effective_preset
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        from neuroslm import hardware
        monkeypatch.setattr(hardware, "_cuda_is_available", lambda: True)
        monkeypatch.setattr(
            hardware, "_cuda_device_name",
            lambda: "NVIDIA A100-SXM4-40GB",
        )
        monkeypatch.setattr(
            hardware, "_cuda_total_memory_bytes",
            lambda: 40 * 1024 ** 3,
        )
        arch = TrainingConfig()
        glb = ProjectConfig(
            repo_root=tmp_path,
            # No default_hardware set → trigger auto-detect
            hardware_presets={"A100": "large"},
        )
        eff, _ = _resolve_effective_preset(
            cli_preset=None, arch_cfg=arch, project_cfg=glb,
        )
        assert eff == "large"


# ─────────────────────────────────────────────────────────────────────
# E. _resolve_effective_steps — same precedence, simpler payload
# ─────────────────────────────────────────────────────────────────────

class TestResolveEffectiveSteps:
    """``--steps`` resolution mirrors preset precedence (CLI > arch >
    global). Used by ``cmd_train`` when ``args.steps`` is None."""

    def test_cli_wins(self, tmp_path):
        from neuroslm.cli import _resolve_effective_steps
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config("steps: 10000")
        glb = ProjectConfig(repo_root=tmp_path, default_steps=5000)
        assert _resolve_effective_steps(
            cli_steps=42, arch_cfg=arch, project_cfg=glb,
        ) == 42

    def test_arch_beats_global(self, tmp_path):
        from neuroslm.cli import _resolve_effective_steps
        from neuroslm.dsl.training_config import parse_training_config
        from neuroslm.project_config import ProjectConfig
        arch = parse_training_config("steps: 10000")
        glb = ProjectConfig(repo_root=tmp_path, default_steps=5000)
        assert _resolve_effective_steps(
            cli_steps=None, arch_cfg=arch, project_cfg=glb,
        ) == 10000

    def test_global_when_arch_silent(self, tmp_path):
        from neuroslm.cli import _resolve_effective_steps
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()  # steps default = 0
        glb = ProjectConfig(repo_root=tmp_path, default_steps=2000)
        assert _resolve_effective_steps(
            cli_steps=None, arch_cfg=arch, project_cfg=glb,
        ) == 2000

    def test_returns_none_when_nothing_set(self, tmp_path):
        from neuroslm.cli import _resolve_effective_steps
        from neuroslm.dsl.training_config import TrainingConfig
        from neuroslm.project_config import ProjectConfig
        arch = TrainingConfig()
        glb = ProjectConfig(repo_root=tmp_path)   # default_steps=0
        assert _resolve_effective_steps(
            cli_steps=None, arch_cfg=arch, project_cfg=glb,
        ) is None
