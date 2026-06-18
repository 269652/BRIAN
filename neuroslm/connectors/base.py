# -*- coding: utf-8 -*-
"""Platform-agnostic deploy configuration and connector base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeployConfig:
    """Everything a connector needs to launch one training run.

    ``arch`` is always a resolved local path by the time the connector
    sees it. If the original request was DNA-mode, ``cmd_deploy`` compiles
    the workspace and sets ``arch = str(workspace.arch_root)`` BEFORE
    calling ``connector.launch()``. ``source_dna`` carries the original
    DNA path for telemetry labels only.

    Cadence fields (``log_every``, ``save_every``, ``push_every``,
    ``ood_every``) default to 0, meaning "use the trainer's own default".
    Connectors skip zero-valued cadence fields to avoid overriding
    defaults that the on-box training loop already manages.
    """
    steps: int
    branch: Optional[str] = None
    arch: Optional[str] = None
    scale: Optional[str] = None
    label: Optional[str] = None
    resume_from: Optional[str] = None
    source_dna: Optional[str] = None
    ood_every: int = 0
    log_every: int = 0
    save_every: int = 0
    push_every: int = 0
    push_backend: str = "hf"
    hf_repo_id: str = "moritzroessler/BRIAN"
    push_optimizer: bool = False
    extra_env: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base for training platform connectors.

    A connector translates a :class:`DeployConfig` into a real launch
    on a specific cloud platform (vast.ai, lightning.ai, …). Each
    implementation is registered in :mod:`neuroslm.connectors` and
    selected by the ``[deploy] platform`` key in ``brian.toml`` or the
    ``--platform`` CLI flag.
    """

    @classmethod
    @abstractmethod
    def platform_name(cls) -> str:
        """Stable identifier used in ``brian.toml`` and ``--platform``."""

    @abstractmethod
    def launch(self, config: DeployConfig) -> int:
        """Launch a training run. Returns an exit code (0 = success)."""
