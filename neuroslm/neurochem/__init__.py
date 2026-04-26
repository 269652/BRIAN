"""Neurochemistry subsystem.

Models neurotransmitters as named tensors with vesicle/release/reuptake
dynamics, receptors as gain-modulating projections on hidden states, and
named axonal pathways between brain-region modules.
"""
from .transmitters import TransmitterSystem, NT_NAMES
from .receptors import ReceptorBank
from .projections import Projection, ProjectionGraph
from .nuclei import VTA, NucleusAccumbens, LocusCoeruleus, RapheNuclei, BasalForebrain
from .homeostasis import Homeostasis

__all__ = [
    "TransmitterSystem", "NT_NAMES",
    "ReceptorBank",
    "Projection", "ProjectionGraph",
    "VTA", "NucleusAccumbens", "LocusCoeruleus", "RapheNuclei", "BasalForebrain",
    "Homeostasis",
]
