"""Intelligence-density mechanisms for NeuroSLM.

These modules turn parameters into reasoning depth, not just memorization
capacity. They are the primary lever for getting frontier-class behavior
out of a 100M-param substrate:

  * flow.py          — adaptive compute: per-token depth + ponder loop
  * mixture.py       — sparse MoE router (top-2, load-balanced)
  * memory_attention — retrieval-augmented cross-attention into memory
  * metrics.py       — quantifiable consciousness / intelligence metrics
  * reflection.py    — spontaneous self-reflection / identity formation
"""

from .flow import AdaptiveComputeBlock, PonderController
from .mixture import SparseMoE
from .memory_attention import MemoryCrossAttention
from .metrics import IntelligenceMetrics, IdentityDriftTracker
from .reflection import SpontaneousReflection

__all__ = [
    "AdaptiveComputeBlock", "PonderController",
    "SparseMoE",
    "MemoryCrossAttention",
    "IntelligenceMetrics", "IdentityDriftTracker",
    "SpontaneousReflection",
]
