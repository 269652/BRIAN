# -*- coding: utf-8 -*-
"""Utilities for NeuroSLM.

Uses :pep:`562` lazy imports so importing just ``bootstrap_secrets``
(used by every connector) doesn't drag in the colab module — which
transitively imports ``torch`` and adds ~3-5 seconds to every CLI
startup. Lightweight helpers (``secrets``) load eagerly; heavy
helpers (``colab``) only when accessed.
"""
from neuroslm.utils.secrets import (
    bootstrap_secrets,
    detect_environment,
    get_secret,
    list_secret_providers,
    register_secret_provider,
    unregister_secret_provider,
)

__all__ = [
    # lazy (load on first access — see __getattr__ below)
    "init_evolution",
    "apply_patch_stack",
    "discover_patches",
    "get_last_checkpoint_step",
    "EvolutionaryTrainingContext",
    # eager (lightweight, no torch dependency)
    "get_secret",
    "bootstrap_secrets",
    "register_secret_provider",
    "unregister_secret_provider",
    "list_secret_providers",
    "detect_environment",
]


# ── PEP 562 lazy import for the colab helpers ────────────────────────
# These names trigger a full ``torch`` import the first time they're
# touched. Resolving them lazily keeps ``from neuroslm.utils.secrets
# import bootstrap_secrets`` (every connector's first call) snappy.
_LAZY_COLAB_NAMES = {
    "init_evolution",
    "apply_patch_stack",
    "discover_patches",
    "get_last_checkpoint_step",
    "EvolutionaryTrainingContext",
}


def __getattr__(name: str):
    """Lazy-load colab helpers; raises ``AttributeError`` otherwise."""
    if name in _LAZY_COLAB_NAMES:
        from neuroslm.utils import colab  # heavy: imports torch
        value = getattr(colab, name)
        # Cache on the module so subsequent accesses are O(1).
        globals()[name] = value
        return value
    raise AttributeError(
        f"module 'neuroslm.utils' has no attribute {name!r}"
    )
