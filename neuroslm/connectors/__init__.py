# -*- coding: utf-8 -*-
"""Training platform connector registry.

Usage::

    from neuroslm.connectors import get_connector, DeployConfig
    config = DeployConfig(steps=10_000, branch="master")
    rc = get_connector("vast").launch(config)

Adding a new platform
─────────────────────
1. Create ``neuroslm/connectors/<name>.py`` with a class that inherits
   :class:`~neuroslm.connectors.base.BaseConnector` and implements
   ``platform_name()`` and ``launch()``.
2. Import it here and add it to ``_REGISTRY``.
3. Add the platform name to the ``choices`` list in
   ``neuroslm/cli.py``'s ``deploy`` subparser ``--platform`` argument.
"""
from __future__ import annotations

from neuroslm.connectors.base import (
    BaseConnector,
    DeployConfig,
    JobInfo,
    JobStatus,
    load_job,
    load_jobs,
    register_job,
    remove_job,
)
from neuroslm.connectors.lightning import LightningConnector
from neuroslm.connectors.vast import VastConnector

_REGISTRY: dict[str, type[BaseConnector]] = {
    VastConnector.platform_name(): VastConnector,
    LightningConnector.platform_name(): LightningConnector,
}

_PLATFORMS = sorted(_REGISTRY)


def get_connector(platform: str) -> BaseConnector:
    """Return an initialised connector for *platform*.

    Raises :exc:`ValueError` if the platform is not registered.
    """
    if platform not in _REGISTRY:
        raise ValueError(
            f"Unknown platform {platform!r}. "
            f"Available: {_PLATFORMS}"
        )
    return _REGISTRY[platform]()


def all_connectors() -> list[BaseConnector]:
    """Return one instantiated connector per registered platform.

    Used by ``brian ps`` to iterate every platform's ``list_jobs()``
    without hardcoding the platform list.
    """
    return [cls() for cls in _REGISTRY.values()]


__all__ = [
    "BaseConnector",
    "DeployConfig",
    "JobInfo",
    "JobStatus",
    "LightningConnector",
    "VastConnector",
    "all_connectors",
    "get_connector",
    "load_job",
    "load_jobs",
    "register_job",
    "remove_job",
]
