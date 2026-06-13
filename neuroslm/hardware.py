# -*- coding: utf-8 -*-
"""Hardware detection + VRAM-aware preset selection.

This module is the **only** place that knows about

  * how to read the current CUDA device name and total VRAM,
  * how to normalise vendor strings (``"NVIDIA A100-SXM4-40GB"``) into
    canonical keys (``"A100"``) that match a ``brian.toml``
    ``[hardware.<NAME>]`` section,
  * and which preset to pick when no human-configured override exists
    (``pick_preset_for_vram``).

All ``torch.cuda`` reads are funnelled through tiny ``_cuda_*`` helpers
so tests can monkey-patch them without needing a real GPU.

Used by :func:`neuroslm.cli._resolve_effective_preset` as the lowest-
priority fallback in the resolution chain::

    CLI > arch.neuro > [hardware.<DETECTED>] > [defaults] > AUTO

The locked spec lives in ``tests/test_hardware_aware_preset_selection.py``.
"""
from __future__ import annotations

import re
from typing import List, Tuple


# ─── torch.cuda shims (monkey-patched in tests) ─────────────────────

def _cuda_is_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device_name(device: int = 0) -> str:
    try:
        import torch
        return str(torch.cuda.get_device_name(device))
    except Exception:
        return ""


def _cuda_total_memory_bytes(device: int = 0) -> int:
    try:
        import torch
        return int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        return 0


# ─── Canonical-name normaliser ──────────────────────────────────────
#
# Pattern → canonical key. Order matters — the first regex that hits
# wins, so put MORE-specific patterns first (RTX_4090 before RTX_*).
# Patterns are case-insensitive and matched against the raw vendor
# string returned by ``torch.cuda.get_device_name``.
_GPU_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # NVIDIA datacenter
    (re.compile(r"\bH100\b", re.I), "H100"),
    (re.compile(r"\bA100\b", re.I), "A100"),
    (re.compile(r"\bA40\b", re.I), "A40"),
    (re.compile(r"\bA10\b", re.I), "A10"),
    (re.compile(r"\bA6000\b", re.I), "A6000"),
    (re.compile(r"\bA5000\b", re.I), "A5000"),
    (re.compile(r"\bA4000\b", re.I), "A4000"),
    (re.compile(r"\bV100\b", re.I), "V100"),
    (re.compile(r"\bL40\b", re.I), "L40"),
    (re.compile(r"\bL4\b",  re.I), "L4"),
    (re.compile(r"\bT4\b",  re.I), "T4"),
    # Consumer RTX 40xx / 30xx — most-specific first
    (re.compile(r"\bRTX[- _]?4090\b", re.I), "RTX_4090"),
    (re.compile(r"\bRTX[- _]?4080\b", re.I), "RTX_4080"),
    (re.compile(r"\bRTX[- _]?4070\b", re.I), "RTX_4070"),
    (re.compile(r"\bRTX[- _]?3090\b", re.I), "RTX_3090"),
    (re.compile(r"\bRTX[- _]?3080\b", re.I), "RTX_3080"),
    (re.compile(r"\bRTX[- _]?3070\b", re.I), "RTX_3070"),
]


def normalise_gpu_name(raw: str) -> str:
    """Map a raw vendor device-name string to a canonical hardware key.

    Returns ``"CPU"`` for the empty string. Unknown GPUs are
    upper-cased with spaces/dashes/dots collapsed to underscores so
    they can still be looked up via ``[hardware.<KEY>]`` if the user
    chooses to add a section for them.
    """
    s = (raw or "").strip()
    if not s:
        return "CPU"
    for pat, key in _GPU_PATTERNS:
        if pat.search(s):
            return key
    # Strip the leading "NVIDIA" prefix so unknown cards collapse to
    # something readable (avoids "NVIDIA_QUADRO_RTX_6000" noise).
    s = re.sub(r"^\s*NVIDIA\s+", "", s, flags=re.I)
    s = re.sub(r"[\s\-\.]+", "_", s).strip("_").upper()
    return s or "CPU"


# ─── Public detectors ───────────────────────────────────────────────

def detect_hardware() -> str:
    """Return the canonical key for the current accelerator, or
    ``"CPU"`` when no CUDA device is visible."""
    if not _cuda_is_available():
        return "CPU"
    return normalise_gpu_name(_cuda_device_name())


def detect_vram_gib() -> float:
    """Return the total VRAM of the active CUDA device in GiB, or
    ``0.0`` on CPU."""
    if not _cuda_is_available():
        return 0.0
    bytes_total = _cuda_total_memory_bytes()
    return bytes_total / (1024.0 ** 3)


# ─── VRAM → preset fallback ─────────────────────────────────────────
#
# Last-ditch table consulted when neither the CLI, nor the arch, nor
# brian.toml has a preference. Each tier says "I need AT LEAST this
# many GiB of VRAM"; the highest matching tier wins.
#
# Tier picks reflect the preset bundle this repo actually ships:
#   - "tiny"     — CPU / no CUDA
#   - "t4_2k"    — Tesla T4 (16 GiB, fp16, the documented small CUDA)
#   - "cheap_2k" — RTX 3090 / RTX 4090 / A5000 (24 GiB, bf16)
#   - "large"    — A100 40 GiB
#   - "xl"       — A100 80 GiB / H100
#
# Sorted ASCENDING by VRAM floor so the lookup loop can stop at the
# largest matching tier. The order is asserted by a regression test.
_PRESET_VRAM_TIERS: List[Tuple[float, str]] = [
    (0.0,   "tiny"),
    (8.0,   "t4_2k"),
    (20.0,  "cheap_2k"),
    (32.0,  "large"),
    (60.0,  "xl"),
]


def pick_preset_for_vram(gib: float) -> str:
    """Return the largest preset whose VRAM floor is ≤ ``gib``.

    ``gib == 0.0`` always returns the CPU bucket (``"tiny"``).
    Negative inputs are clamped to 0.
    """
    gib = max(0.0, float(gib))
    chosen = _PRESET_VRAM_TIERS[0][1]
    for vram_floor, preset in _PRESET_VRAM_TIERS:
        if gib >= vram_floor:
            chosen = preset
        else:
            break
    return chosen


__all__ = [
    "normalise_gpu_name",
    "detect_hardware",
    "detect_vram_gib",
    "pick_preset_for_vram",
]
