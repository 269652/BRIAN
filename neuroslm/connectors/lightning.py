# -*- coding: utf-8 -*-
"""Lightning AI training connector.

Launches a training run on Lightning AI Studios via the ``lightning-sdk``
Python package.  A Studio is created (or reused by name), started with the
requested GPU machine, the training command is submitted, and the Studio is
stopped after the run completes.

Prerequisites
─────────────
  pip install lightning-sdk        # Lightning AI SDK
  lightning login                  # authenticate once (stores token)

Machine selection (highest precedence wins)
───────────────────────────────────────────
  1. ``config.extra_env["LIGHTNING_MACHINE"]``  (CLI ``--machine``,
     brian.toml ``[deploy].machine``, or any caller-supplied override)
  2. ``config.scale``                            (CLI ``--scale``,
     re-purposed as a GPU hint when no ``--machine`` is given)
  3. ``Machine.T4``                              (sensible default)

Substring match — case-insensitive. Examples that resolve to T4::

    --machine T4
    --machine t4
    --scale t4_2k
    LIGHTNING_MACHINE=T4

Studio naming
─────────────
  brian-{config.label}   when label is set
  brian-train            fallback
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from neuroslm.connectors.base import BaseConnector, DeployConfig

if TYPE_CHECKING:
    try:  # ``lightning_sdk`` is the modern wheel name (no dot).
        from lightning_sdk import Machine  # type: ignore[import]
    except ImportError:  # Fall back to the legacy namespaced import.
        from lightning.sdk import Machine  # type: ignore[import]


def _import_lightning_sdk():
    """Import (Studio, Machine) from whichever Lightning SDK is installed.

    The wheel was renamed from ``lightning.sdk`` to ``lightning_sdk``
    (no dot) in mid-2024. We try the modern name first, then the legacy
    one, so the same connector code works against both releases.
    """
    try:
        from lightning_sdk import Studio, Machine  # type: ignore[import]
        return Studio, Machine, "lightning_sdk"
    except ImportError:
        pass
    try:
        from lightning.sdk import Studio, Machine  # type: ignore[import]
        return Studio, Machine, "lightning.sdk"
    except ImportError:
        return None, None, None


class LightningConnector(BaseConnector):
    """Launch training on Lightning AI Studios."""

    @classmethod
    def platform_name(cls) -> str:
        return "lightning"

    def launch(self, config: DeployConfig) -> int:
        Studio, Machine, sdk_name = _import_lightning_sdk()
        if Studio is None:
            print(
                "[lightning] lightning-sdk is not installed.\n"
                "  pip install lightning-sdk\n"
                "  lightning login\n"
                "  Docs: https://lightning.ai/docs/overview/studios",
                file=sys.stderr,
            )
            return 1

        studio_name = f"brian-{config.label}" if config.label else "brian-train"
        machine = self._resolve_machine(config, Machine)
        train_cmd = self._build_command(config)

        # Loud, pre-flight summary so any failure beyond this point is
        # post-launch (cloud-side), not configuration-side.
        print(f"[lightning] SDK         : {sdk_name}")
        print(f"[lightning] studio      : {studio_name}")
        print(f"[lightning] machine     : {machine}")
        print(f"[lightning] arch        : {config.arch or '(brian.toml default)'}")
        print(f"[lightning] steps       : {config.steps}")
        if config.resume_from:
            print(f"[lightning] resume_from : {config.resume_from}")
        print(f"[lightning] command     : {train_cmd}")

        studio = Studio(name=studio_name)
        studio.start(machine=machine)
        try:
            studio.run(train_cmd)
        finally:
            studio.stop()
        return 0

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _resolve_machine(config: DeployConfig, Machine) -> "Machine":
        """Map ``config`` → :class:`Machine` enum value.

        Precedence: ``extra_env["LIGHTNING_MACHINE"]`` > ``config.scale``
        > ``Machine.T4`` (default — matches ``[hardware.T4]`` in
        ``brian.toml``).
        """
        override = config.extra_env.get("LIGHTNING_MACHINE", "").lower()
        scale = (override or config.scale or "").lower()

        if "a100" in scale:
            return Machine.A100
        if "a10g" in scale:
            return Machine.A10G
        if "l4" in scale:
            return Machine.L4
        if "t4" in scale:
            return Machine.T4
        # Fall back to T4 — the default hardware tier in brian.toml
        # ``[hardware.T4]`` and the cheapest GPU on Lightning AI.
        return Machine.T4

    @staticmethod
    def _build_command(config: DeployConfig) -> str:
        arch = config.arch or "architectures/current"
        parts = [
            "python -m neuroslm.train_dsl",
            f"--arch {arch}",
            f"--steps {config.steps}",
        ]
        if config.log_every > 0:
            parts.append(f"--log_every {config.log_every}")
        if config.save_every > 0:
            parts.append(f"--save_every {config.save_every}")
        if config.push_every > 0:
            parts.append(f"--push_every {config.push_every}")
        if config.push_backend:
            parts.append(f"--push_backend {config.push_backend}")
        if config.resume_from:
            parts.append(f"--resume_from {config.resume_from}")
        if config.ood_every > 0:
            parts.append(f"--ood_every {config.ood_every}")
        return " ".join(parts)
