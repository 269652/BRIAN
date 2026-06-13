# -*- coding: utf-8 -*-
"""Workspace-level project configuration for BRIAN / NeuroSLM.

This module owns the single source of truth for **which architecture
or DNA every training/deploy/colab script targets**. It reads a tiny
``brian.toml`` at the repository root::

    # brian.toml
    [current]
    arch = "architectures/rcc_bowtie"   # current architecture folder
    dna  = ""                            # set to a .dna path for DNA-mode

    [nfg]
    output = ".neuro/nfg.png"            # where `brian compile nfg
                                          # --current` writes the diagram
    format = "png"                       # png | svg | pdf | dot
    engine = "dot"                       # dot | neato | sfdp | fdp | circo

Six fields, one decision per field. The training scripts
(``vast_train_dsl_loop.sh``, ``vast_train_dna_loop.sh``,
``_deploy_train.py``, ``colab_run.ipynb``) all dispatch through
:func:`ProjectConfig.training_target` so a one-line edit to
``brian.toml`` re-targets every script in the repo.

Env-var overrides (``BRIAN_ARCH``, ``BRIAN_DNA``, ``BRIAN_NFG_OUTPUT``)
keep existing vast.ai pipelines working — they take priority over the
file.

The contract is locked by ``tests/test_project_config.py``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# ─── tomllib (stdlib in 3.11+; tomli fallback for 3.10) ──────────────
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import]
else:  # pragma: no cover — exercised only on 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


# ─── defaults ────────────────────────────────────────────────────────

_DEFAULT_ARCH = "architectures/rcc_bowtie"
_DEFAULT_DNA = ""
_DEFAULT_NFG_OUTPUT = ".neuro/nfg.png"
_DEFAULT_NFG_FORMAT = "png"
_DEFAULT_NFG_ENGINE = "dot"
_DEFAULT_PRESET = ""
_DEFAULT_HARDWARE = ""
_DEFAULT_STEPS = 0  # 0 = "no opinion — caller picks"


# ─── data class ──────────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    """Parsed ``brian.toml``.

    All path-like attributes are stored verbatim (relative to
    ``repo_root``) so the on-disk file is human-readable. Use the
    ``resolve_*()`` helpers to get absolute :class:`Path` objects.
    """
    repo_root: Path
    arch: str = _DEFAULT_ARCH
    dna: str = _DEFAULT_DNA
    nfg_output: str = _DEFAULT_NFG_OUTPUT
    nfg_format: str = _DEFAULT_NFG_FORMAT
    nfg_engine: str = _DEFAULT_NFG_ENGINE
    # ── Global training defaults ──
    # Read from the ``[defaults]`` section of ``brian.toml``. Empty means
    # "no opinion — let the arch or CLI decide". Merged into a parsed
    # ``TrainingConfig`` by ``neuroslm.dsl.training_config.apply_global_defaults``
    # under the 2026-06-12 precedence (arch wins over global).
    # See ``docs/CLI.md`` § "Global defaults".
    default_preset: str = _DEFAULT_PRESET      # e.g. "cheap_2k", "t4_2k"
    default_hardware: str = _DEFAULT_HARDWARE  # e.g. "RTX_3090", "A100"
    default_steps: int = _DEFAULT_STEPS        # 0 = no opinion
    # ── Per-hardware preset map ──
    # ``[hardware.<NAME>] preset = "..."`` sections feed this dict.
    # Looked up by ``cli._resolve_effective_preset`` AFTER the arch's
    # own preset and BEFORE the workspace ``default_preset``.
    # Example::
    #     [hardware.A100] preset = "large"
    #     [hardware.T4]   preset = "t4_2k"
    #     [hardware.CPU]  preset = "tiny"
    hardware_presets: Dict[str, str] = field(default_factory=dict)

    # ── computed ──

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()

    @property
    def is_dna_mode(self) -> bool:
        """True iff ``dna`` points at a file that exists on disk.

        Used by every training script to dispatch
        DNA-loop vs DSL-loop without duplicating the check.
        """
        if not self.dna:
            return False
        return self.resolve_dna_path().is_file()  # type: ignore[union-attr]

    # ── path helpers ──

    def resolve_arch_path(self) -> Path:
        """Absolute path to the architecture folder."""
        p = Path(self.arch)
        if p.is_absolute():
            return p.resolve()
        return (self.repo_root / p).resolve()

    def resolve_dna_path(self) -> Optional[Path]:
        """Absolute path to the DNA file, or ``None`` if not set."""
        if not self.dna:
            return None
        p = Path(self.dna)
        if p.is_absolute():
            return p.resolve()
        return (self.repo_root / p).resolve()

    def nfg_output_path(self, heat: bool = False) -> Path:
        """Absolute path the ``brian compile nfg --current`` renderer
        writes to.

        ``heat=True`` inserts ``.heat`` before the final extension —
        e.g. ``.neuro/nfg.png`` → ``.neuro/nfg.heat.png``. The parallel
        heat-overlay implementation uses this so its output sits
        beside the plain NFG and the README can reference it.

        Side effect: ensures the parent directory exists (the caller
        is expected to write to the returned path).
        """
        p = Path(self.nfg_output)
        if not p.is_absolute():
            p = self.repo_root / p
        if heat:
            # Insert ``.heat`` immediately before the final extension.
            #   nfg.png       → nfg.heat.png
            #   arch.v2.png   → arch.v2.heat.png
            #   nfg           → nfg.heat  (extension-less is rare but
            #                    we handle it cleanly)
            if p.suffix:
                p = p.with_name(p.stem + ".heat" + p.suffix)
            else:
                p = p.with_name(p.name + ".heat")
        p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def training_target(self) -> Tuple[str, Path]:
        """Return ``("dna", path)`` in DNA-mode, else
        ``("arch", path)``. The single dispatch every training script
        uses.
        """
        if self.is_dna_mode:
            return ("dna", self.resolve_dna_path())  # type: ignore[return-value]
        return ("arch", self.resolve_arch_path())


# ─── loader ──────────────────────────────────────────────────────────

def _default_start_dir() -> Path:
    """The directory ``load_project_config`` walks up from when no
    ``start`` is provided. Pulled out as a module-level callable so
    tests can monkeypatch it.
    """
    return Path.cwd()


def _find_config_file(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a ``brian.toml``. Returns
    the path or ``None``."""
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        candidate = parent / "brian.toml"
        if candidate.is_file():
            return candidate
    return None


def _parse_toml(path: Path) -> dict:
    """Parse a TOML file; return an empty dict if tomllib is
    unavailable (3.10 without ``tomli`` installed)."""
    if tomllib is None:  # pragma: no cover
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _derive_nfg_format(output_path: str, explicit_format: str) -> str:
    """If the user changed ``nfg_output`` via env without also setting
    ``nfg_format``, infer the format from the file extension."""
    suffix = Path(output_path).suffix.lstrip(".").lower()
    if suffix in {"png", "svg", "pdf", "dot"}:
        return suffix
    return explicit_format


def load_project_config(
    start: Optional[Path] = None,
) -> ProjectConfig:
    """Load ``brian.toml`` from the workspace, applying env-var
    overrides.

    Parameters
    ----------
    start : Path, optional
        Directory to start the upward search from. Defaults to the
        current working directory.

    Returns
    -------
    ProjectConfig
        With ``repo_root`` set to either (a) the directory containing
        the discovered ``brian.toml``, or (b) the ``start`` directory
        if no config file was found.
    """
    if start is None:
        start = _default_start_dir()
    start = Path(start).resolve()

    config_path = _find_config_file(start)
    if config_path is not None:
        repo_root = config_path.parent
        data = _parse_toml(config_path)
    else:
        repo_root = start
        data = {}

    current = data.get("current", {}) if isinstance(data, dict) else {}
    nfg_section = data.get("nfg", {}) if isinstance(data, dict) else {}
    defaults_section = (
        data.get("defaults", {}) if isinstance(data, dict) else {}
    )
    # ``[hardware.A100] preset = "large"`` deserialises as
    # ``data["hardware"] == {"A100": {"preset": "large"}}``.
    hardware_section = (
        data.get("hardware", {}) if isinstance(data, dict) else {}
    )
    hardware_presets: Dict[str, str] = {}
    if isinstance(hardware_section, dict):
        for hw_name, hw_cfg in hardware_section.items():
            if isinstance(hw_cfg, dict) and "preset" in hw_cfg:
                hardware_presets[str(hw_name)] = str(hw_cfg["preset"])

    arch = str(current.get("arch", _DEFAULT_ARCH))
    dna = str(current.get("dna", _DEFAULT_DNA))
    nfg_output = str(nfg_section.get("output", _DEFAULT_NFG_OUTPUT))
    nfg_format = str(nfg_section.get("format", _DEFAULT_NFG_FORMAT))
    nfg_engine = str(nfg_section.get("engine", _DEFAULT_NFG_ENGINE))
    default_preset = str(defaults_section.get("preset", _DEFAULT_PRESET))
    default_hardware = str(
        defaults_section.get("hardware", _DEFAULT_HARDWARE)
    )
    default_steps = int(defaults_section.get("steps", _DEFAULT_STEPS))

    # ── env-var overrides (BRIAN_ prefix to avoid collisions) ──
    env_arch = os.environ.get("BRIAN_ARCH")
    env_dna = os.environ.get("BRIAN_DNA")
    env_nfg_out = os.environ.get("BRIAN_NFG_OUTPUT")
    env_nfg_format = os.environ.get("BRIAN_NFG_FORMAT")
    env_nfg_engine = os.environ.get("BRIAN_NFG_ENGINE")
    env_default_preset = os.environ.get("BRIAN_DEFAULT_PRESET")
    env_default_hardware = os.environ.get("BRIAN_DEFAULT_HARDWARE")
    env_default_steps = os.environ.get("BRIAN_DEFAULT_STEPS")

    if env_arch:
        arch = env_arch
    if env_dna is not None:
        # Allow BRIAN_DNA="" to *clear* a DNA setting from the file.
        dna = env_dna
    if env_nfg_out:
        nfg_output = env_nfg_out
        # Auto-derive format from the new extension unless the user
        # also set BRIAN_NFG_FORMAT explicitly.
        if not env_nfg_format:
            nfg_format = _derive_nfg_format(env_nfg_out, nfg_format)
    if env_nfg_format:
        nfg_format = env_nfg_format
    if env_nfg_engine:
        nfg_engine = env_nfg_engine
    if env_default_preset is not None:
        default_preset = env_default_preset
    if env_default_hardware is not None:
        default_hardware = env_default_hardware
    if env_default_steps:
        try:
            default_steps = int(env_default_steps)
        except ValueError:
            pass  # leave whatever the file said

    return ProjectConfig(
        repo_root=repo_root,
        arch=arch,
        dna=dna,
        nfg_output=nfg_output,
        nfg_format=nfg_format,
        nfg_engine=nfg_engine,
        default_preset=default_preset,
        default_hardware=default_hardware,
        default_steps=default_steps,
        hardware_presets=hardware_presets,
    )


__all__ = [
    "ProjectConfig",
    "load_project_config",
]
