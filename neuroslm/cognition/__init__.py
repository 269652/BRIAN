"""Cognition layer — the runtime half of the two-layer doctrine (§14).

The trunk learns language during pretraining; this package uses a
trained trunk to THINK: an always-on cognitive cycle with sensory
input, hippocampal episodic recall, NT-gated (basal-ganglia-style)
selection over candidate thoughts, and surprise-gated memory writes.
Hosted by ``neuroslm.chat_daemon.ChatDaemon`` (``brian chat --mind``).
"""
from neuroslm.cognition.runtime import (  # noqa: F401
    CognitiveRuntime,
    MindConfig,
    ThoughtScore,
    TickResult,
    selection_temperature,
)
