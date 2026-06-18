# -*- coding: utf-8 -*-
"""Workspace-level project configuration for BRIAN / NeuroSLM.

This module owns the single source of truth for **which architecture
or DNA every training/deploy/colab script targets**. It reads a tiny
``brian.toml`` at the repository root::

    # brian.toml
    [current]
    arch = "architectures/current"      # current architecture folder
    dna  = ""                            # set to a .dna path for DNA-mode

    [nfg]
    output = ".neuro/nfg.png"            # where `brian compile nfg
                                          # --current` writes the diagram
    format = "png"                       # png | svg | pdf | dot
    engine = "dot"                       # dot | neato | sfdp | fdp | circo
    spring_gain = 0.9                    # K for spring engines
    panel_opacity = 1.0                  # 0.0–1.0 alpha for cluster panels

Six fields, one decision per field. The training scripts
(``vast_train_dsl_loop.sh``, ``vast_train_dna_loop.sh``,
``_deploy_train.py``, ``colab_run.ipynb``) all dispatch through
:func:`ProjectConfig.training_target` so a one-line edit to
``brian.toml`` re-targets every script in the repo.

Env-var overrides (``BRIAN_ARCH``, ``BRIAN_DNA``, ``BRIAN_NFG_OUTPUT``,
``BRIAN_NFG_SPRING_GAIN``, ``BRIAN_NFG_PANEL_OPACITY``) keep existing
vast.ai pipelines working — they take priority over the file.

The contract is locked by ``tests/test_project_config.py``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── tomllib (stdlib in 3.11+; tomli fallback for 3.10) ──────────────
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import]
else:  # pragma: no cover — exercised only on 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


# ─── defaults ────────────────────────────────────────────────────────

# Default arch is the live working-copy folder. Renamed on 2026-06-14
# from "architectures/rcc_bowtie": master/ holds the canonical
# hand-edited source-of-truth, current/ is the live arch the trainer
# consumes by default (and the one experiments may branch off).
_DEFAULT_ARCH = "architectures/master"
_DEFAULT_DNA = ""
_DEFAULT_NFG_OUTPUT = ".neuro/nfg.png"
_DEFAULT_NFG_FORMAT = "png"
# Plural list of formats to render in one ``brian compile nfg`` call.
# Defaults to PNG + SVG so users get both a raster image (chat embeds,
# inline preview) AND a vector image (zoom-in inspection without
# pixelation). Singular ``[nfg].format = "png"`` is still honoured
# as a 1-element list for backwards compat — see loader below.
_DEFAULT_NFG_FORMATS: List[str] = ["png", "svg"]
# PNG rasterization DPI. Graphviz default is 96 (looks pixelated when
# zooming on modern hi-DPI displays). 150 is ~1.56x sharper at ~2.4x
# file size — a reasonable middle for inline previews. SVG ignores it.
_DEFAULT_NFG_DPI = 150
_DEFAULT_NFG_ENGINE = "dot"
# Ideal edge length (K) for spring-based engines (fdp/sfdp/neato).
_DEFAULT_NFG_SPRING_GAIN = 0.9
# Cluster panel background opacity (0.0 = transparent, 1.0 = opaque).
_DEFAULT_NFG_PANEL_OPACITY = 1.0
# Show/hide cluster panel background rectangles entirely.
_DEFAULT_NFG_SHOW_PANELS = True
_DEFAULT_PRESET = ""
_DEFAULT_HARDWARE = ""
_DEFAULT_STEPS = 0  # 0 = "no opinion — caller picks"
_DEFAULT_BRANCH = ""  # "" = "no opinion — caller picks" (deploy will
                      # then fall through to _deploy_train.py's own default,
                      # which today is the git HEAD branch)
# ── checkpoint / log cadence ────────────────────────────────────────
# These three control how often the trainer logs, saves a local
# ``.pt``, and pushes it to Git LFS. Defaults restored after the
# H24 deploy lost a 3 k-step run when the box self-destroyed before
# the (then end-of-training-only) push. ``push_every`` was silently
# regressed to 0 during the DSL-trainer rewrite; the legacy
# ``train.py`` pushed after every save. We default to 500 so every
# new deploy is automatically protected.
#
# Source-of-truth precedence (highest → lowest):
#   1. CLI flag (``--push_every`` etc. on ``python -m neuroslm.train_dsl``)
#   2. Env var on the box (``LOG_EVERY``, ``SAVE_EVERY``, ``PUSH_EVERY``)
#   3. ``BRIAN_DEFAULT_*`` env override locally
#   4. ``[defaults]`` in ``brian.toml``
#   5. These constants
_DEFAULT_LOG_EVERY = 20
_DEFAULT_SAVE_EVERY = 500
_DEFAULT_PUSH_EVERY = 500

# ── checkpoint push backend (2026-06-15) ────────────────────────────
# Default flipped from Git-LFS to HuggingFace Hub after run 41063959
# hung at exactly step 500 — synchronous ``git push`` of a 569 MB LFS
# object inside the training loop raced the background log-pusher
# and never returned. HF Hub's single sync HTTPS PUT avoids the race
# entirely. The legacy LFS path is preserved (``push_backend = "lfs"``)
# for envs that can't reach HF Hub.
#
# Same precedence chain as the cadence fields above:
#   1. CLI flag (``--push_backend hf|lfs|none``)
#   2. Env on box (``CHECKPOINT_PUSH_BACKEND``)
#   3. ``BRIAN_DEFAULT_PUSH_BACKEND`` env locally
#   4. ``[defaults]`` in ``brian.toml``
#   5. This constant
_DEFAULT_PUSH_BACKEND = "hf"
_DEFAULT_HF_REPO_ID = "moritzroessler/BRIAN"

# ── push_optimizer (2026-06-15) ──────────────────────────────────────
# When False (default), :func:`push_checkpoint_to_hf` strips the
# ``optimizer`` key from the uploaded payload — saves ~2/3 of the
# file size (weights+m+v → weights only). The on-disk ``.pt`` always
# keeps the full state for same-box resume; the strip only affects
# what crosses the wire. Set True for the rare case where bandwidth
# is cheap and you want the HF copy to be a perfect resume target
# (no ~500-step LR-warmup-shape recovery curve).
_DEFAULT_PUSH_OPTIMIZER = False

# ── deploy platform (2026-06-18) ─────────────────────────────────────
# Which cloud provider to use for `brian deploy`. The connector for the
# chosen platform is resolved by `neuroslm.connectors.get_connector`.
# Override with BRIAN_DEFAULT_PLATFORM env or --platform CLI flag.
_DEFAULT_PLATFORM = "vast"

# ── deploy machine (2026-06-18) ──────────────────────────────────────
# Default GPU/machine tier when the connector supports it (currently
# Lightning AI). Substring match against the connector's enum, so values
# like "T4", "A10G", "A100", "L4" all work. Empty = let the connector
# pick its own default. Override with BRIAN_DEFAULT_MACHINE env or
# ``brian deploy --machine T4``.
_DEFAULT_MACHINE = ""


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
    # Plural list of formats to render. ``nfg_format`` (singular) is
    # kept as the FIRST element of ``nfg_formats`` for backwards compat
    # with code that reads only one format. The loader populates both.
    nfg_formats: List[str] = field(default_factory=lambda: list(_DEFAULT_NFG_FORMATS))
    # PNG rasterization DPI (graphviz ``-Gdpi=``). SVG output ignores it.
    nfg_dpi: int = _DEFAULT_NFG_DPI
    nfg_engine: str = _DEFAULT_NFG_ENGINE
    # Spring constant K for fdp/sfdp/neato engines.
    nfg_spring_gain: float = _DEFAULT_NFG_SPRING_GAIN
    # Cluster panel background alpha (0.0–1.0).
    nfg_panel_opacity: float = _DEFAULT_NFG_PANEL_OPACITY
    # Whether to render cluster panel background rectangles at all.
    nfg_show_panels: bool = _DEFAULT_NFG_SHOW_PANELS
    # ── Global training defaults ──
    # Read from the ``[defaults]`` section of ``brian.toml``. Empty means
    # "no opinion — let the arch or CLI decide". Merged into a parsed
    # ``TrainingConfig`` by ``neuroslm.dsl.training_config.apply_global_defaults``
    # under the 2026-06-12 precedence (arch wins over global).
    # See ``docs/CLI.md`` § "Global defaults".
    default_preset: str = _DEFAULT_PRESET      # e.g. "cheap_2k", "t4_2k"
    default_hardware: str = _DEFAULT_HARDWARE  # e.g. "RTX_3090", "A100"
    default_steps: int = _DEFAULT_STEPS        # 0 = no opinion
    default_branch: str = _DEFAULT_BRANCH      # "" = no opinion. Consumed
                                               # by ``cli.cmd_deploy`` as
                                               # the BRANCH env var passed
                                               # to ``_deploy_train.py``
                                               # when neither ``--branch``
                                               # nor ``$BRANCH`` is set.
    # ── Checkpoint / log cadence ──
    # Propagated via ``cli.cmd_deploy`` as ``LOG_EVERY`` /
    # ``SAVE_EVERY`` / ``PUSH_EVERY`` env vars all the way down to
    # ``vast_train_dsl_loop.sh`` which forwards them as
    # ``--log_every`` / ``--save_every`` / ``--push_every`` to
    # ``python -m neuroslm.train_dsl``. See module-level docstring
    # for precedence.
    default_log_every: int = _DEFAULT_LOG_EVERY
    default_save_every: int = _DEFAULT_SAVE_EVERY
    default_push_every: int = _DEFAULT_PUSH_EVERY
    # ── Checkpoint push backend (2026-06-15) ──
    # ``default_push_backend`` chooses between the HF Hub uploader
    # (default) and the legacy Git LFS uploader. ``default_hf_repo_id``
    # names the target HF repo for the HF backend; ignored when
    # ``default_push_backend == "lfs"``. Both propagate through
    # ``cli.cmd_deploy`` → ``_deploy_train.py`` → vast loop as the
    # ``CHECKPOINT_PUSH_BACKEND`` and ``HF_REPO_ID`` envs.
    default_push_backend: str = _DEFAULT_PUSH_BACKEND
    default_hf_repo_id: str = _DEFAULT_HF_REPO_ID
    # ``default_push_optimizer``: when False (the default),
    # ``push_checkpoint_to_hf`` strips the Adam state from the
    # uploaded ``.pt`` (saves ~2/3 of the file size for a 107 M
    # trunk). Local same-box resume is unaffected — the on-disk
    # ``.pt`` always has full optimiser state. See
    # ``neuroslm.checkpoint_push._maybe_strip_optimizer``.
    default_push_optimizer: bool = _DEFAULT_PUSH_OPTIMIZER
    # ── Deploy platform (2026-06-18) ──
    # Which cloud provider ``brian deploy`` uses. Resolved by
    # ``neuroslm.connectors.get_connector``. Overridden by the
    # ``--platform`` CLI flag and ``BRIAN_DEFAULT_PLATFORM`` env.
    default_platform: str = _DEFAULT_PLATFORM
    # ── Deploy machine (2026-06-18) ──
    # GPU/machine tier hint passed through to connectors that support
    # explicit machine selection (currently Lightning AI). Substring
    # match against the connector's enum, e.g. "T4", "A10G", "A100",
    # "L4". Empty string means "let the connector pick". Overridden by
    # the ``--machine`` CLI flag and ``BRIAN_DEFAULT_MACHINE`` env.
    default_machine: str = _DEFAULT_MACHINE

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

    def nfg_output_paths(self, heat: bool = False) -> List[Tuple[str, Path]]:
        """Plural version of :meth:`nfg_output_path` — returns one
        ``(format, path)`` tuple per entry in ``self.nfg_formats``.

        The path for each format is derived from ``self.nfg_output`` by
        swapping the suffix to match. Example with default config::

            self.nfg_output  = ".neuro/nfg.png"
            self.nfg_formats = ["png", "svg"]
            =>  [("png", "<repo>/.neuro/nfg.png"),
                 ("svg", "<repo>/.neuro/nfg.svg")]

        ``heat=True`` infixes ``.heat`` before each suffix, same rule
        as the singular helper. Parent dirs are created.

        Used by ``brian compile nfg --current`` to emit multiple formats
        in one pass.
        """
        out: List[Tuple[str, Path]] = []
        base = self.nfg_output_path(heat=heat)
        # The singular method already resolved + mkdir'd the parent. Swap
        # the suffix for each requested format. Preserve any infixed
        # ``.heat`` in the stem — Path.with_suffix() would treat ``.heat``
        # as the suffix and strip it (regression caught by
        # test_current_flag_with_heat_writes_heat_sibling), so we slice
        # the final extension off as a plain string and concatenate.
        base_str = str(base)
        last_dot = base_str.rfind(".")
        if last_dot >= 0 and "/" not in base_str[last_dot:] \
                and "\\" not in base_str[last_dot:]:
            # Strip ONLY the final ``.ext`` — leaves ``foo.heat`` and
            # ``foo.v2`` infixes intact.
            stem_str = base_str[:last_dot]
        else:
            stem_str = base_str
        for fmt in self.nfg_formats:
            ext = fmt if fmt.startswith(".") else f".{fmt}"
            out.append((fmt, Path(stem_str + ext)))
        return out

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
    deploy_section = (
        data.get("deploy", {}) if isinstance(data, dict) else {}
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
    # Plural ``formats`` wins over singular ``format``. Either is fine.
    # Normalises to a list of lowercase strings, deduped while preserving
    # order (so ``["png", "svg"]`` stays distinct from ``["svg", "png"]``).
    raw_formats = nfg_section.get("formats")
    if raw_formats is None:
        # Fall back to singular ``format`` (= 1-element list), then default.
        raw_formats = [nfg_section.get("format", _DEFAULT_NFG_FORMAT)]
    if isinstance(raw_formats, str):
        # Tolerate ``formats = "png"`` (string scalar instead of array).
        raw_formats = [raw_formats]
    seen: set = set()
    nfg_formats: List[str] = []
    for f in raw_formats:
        fl = str(f).strip().lower()
        if fl and fl not in seen:
            seen.add(fl)
            nfg_formats.append(fl)
    if not nfg_formats:
        nfg_formats = list(_DEFAULT_NFG_FORMATS)
    # ``nfg_format`` (singular) tracks the primary = first format.
    nfg_format = nfg_formats[0]
    nfg_dpi = int(nfg_section.get("dpi", _DEFAULT_NFG_DPI))
    nfg_engine = str(nfg_section.get("engine", _DEFAULT_NFG_ENGINE))
    nfg_spring_gain = float(nfg_section.get("spring_gain", _DEFAULT_NFG_SPRING_GAIN))
    nfg_panel_opacity = float(nfg_section.get("panel_opacity", _DEFAULT_NFG_PANEL_OPACITY))
    nfg_show_panels = bool(nfg_section.get("show_panels", _DEFAULT_NFG_SHOW_PANELS))
    default_preset = str(defaults_section.get("preset", _DEFAULT_PRESET))
    default_hardware = str(
        defaults_section.get("hardware", _DEFAULT_HARDWARE)
    )
    default_steps = int(defaults_section.get("steps", _DEFAULT_STEPS))
    default_branch = str(defaults_section.get("branch", _DEFAULT_BRANCH))
    default_log_every = int(
        defaults_section.get("log_every", _DEFAULT_LOG_EVERY)
    )
    default_save_every = int(
        defaults_section.get("save_every", _DEFAULT_SAVE_EVERY)
    )
    default_push_every = int(
        defaults_section.get("push_every", _DEFAULT_PUSH_EVERY)
    )
    default_push_backend = str(
        defaults_section.get("push_backend", _DEFAULT_PUSH_BACKEND)
    )
    default_hf_repo_id = str(
        defaults_section.get("hf_repo_id", _DEFAULT_HF_REPO_ID)
    )
    default_push_optimizer = bool(
        defaults_section.get("push_optimizer", _DEFAULT_PUSH_OPTIMIZER)
    )
    default_platform = str(
        deploy_section.get("platform", _DEFAULT_PLATFORM)
    )
    default_machine = str(
        deploy_section.get("machine", _DEFAULT_MACHINE)
    )

    # ── env-var overrides (BRIAN_ prefix to avoid collisions) ──
    env_arch = os.environ.get("BRIAN_ARCH")
    env_dna = os.environ.get("BRIAN_DNA")
    env_nfg_out = os.environ.get("BRIAN_NFG_OUTPUT")
    env_nfg_format = os.environ.get("BRIAN_NFG_FORMAT")
    # Comma-separated list, e.g. ``BRIAN_NFG_FORMATS=png,svg,pdf``.
    # Wins over the singular ``BRIAN_NFG_FORMAT`` when both are set.
    env_nfg_formats = os.environ.get("BRIAN_NFG_FORMATS")
    env_nfg_dpi = os.environ.get("BRIAN_NFG_DPI")
    env_nfg_engine = os.environ.get("BRIAN_NFG_ENGINE")
    env_nfg_spring_gain = os.environ.get("BRIAN_NFG_SPRING_GAIN")
    env_nfg_panel_opacity = os.environ.get("BRIAN_NFG_PANEL_OPACITY")
    env_nfg_show_panels = os.environ.get("BRIAN_NFG_SHOW_PANELS")
    env_default_preset = os.environ.get("BRIAN_DEFAULT_PRESET")
    env_default_hardware = os.environ.get("BRIAN_DEFAULT_HARDWARE")
    env_default_steps = os.environ.get("BRIAN_DEFAULT_STEPS")
    env_default_branch = os.environ.get("BRIAN_DEFAULT_BRANCH")
    env_default_log_every = os.environ.get("BRIAN_DEFAULT_LOG_EVERY")
    env_default_save_every = os.environ.get("BRIAN_DEFAULT_SAVE_EVERY")
    env_default_push_every = os.environ.get("BRIAN_DEFAULT_PUSH_EVERY")
    env_default_push_backend = os.environ.get(
        "BRIAN_DEFAULT_PUSH_BACKEND"
    )
    env_default_hf_repo_id = os.environ.get("BRIAN_DEFAULT_HF_REPO_ID")
    env_default_push_optimizer = os.environ.get(
        "BRIAN_DEFAULT_PUSH_OPTIMIZER"
    )
    env_default_platform = os.environ.get("BRIAN_DEFAULT_PLATFORM")
    env_default_machine = os.environ.get("BRIAN_DEFAULT_MACHINE")

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
    # ``BRIAN_NFG_FORMATS`` (plural, comma-sep) overrides the toml list
    # AND the singular env. Last write wins.
    if env_nfg_formats:
        seen_e: set = set()
        env_list: List[str] = []
        for f in env_nfg_formats.split(","):
            fl = f.strip().lower()
            if fl and fl not in seen_e:
                seen_e.add(fl)
                env_list.append(fl)
        if env_list:
            nfg_formats = env_list
            nfg_format = nfg_formats[0]
    elif env_nfg_format:
        # Singular env-format also re-syncs the plural list so callers
        # that loop over ``nfg_formats`` honour the env override.
        nfg_formats = [nfg_format]
    if env_nfg_dpi:
        try:
            nfg_dpi = int(env_nfg_dpi)
        except ValueError:
            pass  # leave whatever the file said
    if env_nfg_engine:
        nfg_engine = env_nfg_engine
    if env_nfg_spring_gain:
        try:
            nfg_spring_gain = float(env_nfg_spring_gain)
        except ValueError:
            pass
    if env_nfg_panel_opacity:
        try:
            nfg_panel_opacity = float(env_nfg_panel_opacity)
        except ValueError:
            pass
    if env_nfg_show_panels is not None and env_nfg_show_panels != "":
        nfg_show_panels = env_nfg_show_panels.strip().lower() not in ("0", "false", "no", "off")
    if env_default_preset is not None:
        default_preset = env_default_preset
    if env_default_hardware is not None:
        default_hardware = env_default_hardware
    if env_default_steps:
        try:
            default_steps = int(env_default_steps)
        except ValueError:
            pass  # leave whatever the file said
    if env_default_branch is not None:
        # Allow BRIAN_DEFAULT_BRANCH="" to clear a file setting.
        default_branch = env_default_branch
    if env_default_log_every:
        try:
            default_log_every = int(env_default_log_every)
        except ValueError:
            pass
    if env_default_save_every:
        try:
            default_save_every = int(env_default_save_every)
        except ValueError:
            pass
    if env_default_push_every is not None:
        # Allow BRIAN_DEFAULT_PUSH_EVERY="0" to explicitly *disable*
        # the push. Empty string keeps the file value.
        if env_default_push_every != "":
            try:
                default_push_every = int(env_default_push_every)
            except ValueError:
                pass
    if env_default_push_backend:
        # Empty string keeps the file value; any non-empty value
        # wins. Caller-side ``push_checkpoint`` dispatcher tolerates
        # unknown strings by warning + falling back to ``hf``.
        default_push_backend = env_default_push_backend
    if env_default_hf_repo_id:
        default_hf_repo_id = env_default_hf_repo_id
    if env_default_push_optimizer is not None and \
            env_default_push_optimizer != "":
        # Permissive parse: "true"/"1"/"yes"/"on" → True, anything
        # else (incl. "false"/"0"/"no"/"off") → False. Mirrors the
        # truthy semantics most CLI users expect from a bool env.
        default_push_optimizer = env_default_push_optimizer.strip().lower() \
            in ("1", "true", "yes", "on")
    if env_default_platform:
        default_platform = env_default_platform
    if env_default_machine is not None:
        # Allow empty string to clear a brian.toml setting.
        default_machine = env_default_machine

    return ProjectConfig(
        repo_root=repo_root,
        arch=arch,
        dna=dna,
        nfg_output=nfg_output,
        nfg_format=nfg_format,
        nfg_formats=nfg_formats,
        nfg_dpi=nfg_dpi,
        nfg_engine=nfg_engine,
        nfg_spring_gain=nfg_spring_gain,
        nfg_panel_opacity=nfg_panel_opacity,
        nfg_show_panels=nfg_show_panels,
        default_preset=default_preset,
        default_hardware=default_hardware,
        default_steps=default_steps,
        default_branch=default_branch,
        default_log_every=default_log_every,
        default_save_every=default_save_every,
        default_push_every=default_push_every,
        default_push_backend=default_push_backend,
        default_hf_repo_id=default_hf_repo_id,
        default_push_optimizer=default_push_optimizer,
        default_platform=default_platform,
        default_machine=default_machine,
        hardware_presets=hardware_presets,
    )


__all__ = [
    "ProjectConfig",
    "load_project_config",
]
