# -*- coding: utf-8 -*-
"""Utilities for NeuroSLM."""
from neuroslm.utils.colab import init_evolution, apply_patch_stack, discover_patches, get_last_checkpoint_step, EvolutionaryTrainingContext
from neuroslm.utils.secrets import (
    get_secret,
    bootstrap_secrets,
    register_secret_provider,
    unregister_secret_provider,
    list_secret_providers,
    detect_environment,
)

__all__ = [
    "init_evolution",
    "apply_patch_stack",
    "discover_patches",
    "get_last_checkpoint_step",
    "EvolutionaryTrainingContext",
    # secrets helper (notebook-friendly cross-platform resolver)
    "get_secret",
    "bootstrap_secrets",
    "register_secret_provider",
    "unregister_secret_provider",
    "list_secret_providers",
    "detect_environment",
]
