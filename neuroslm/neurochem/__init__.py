"""Neurochemistry subsystem.

Models neurotransmitters as named tensors with vesicle/release/reuptake
dynamics, receptors as gain-modulating projections on hidden states, and
named axonal pathways between brain-region modules.

Extended with:
  - Reuptake transporter dynamics (DAT, SERT, NET, EAAT, GAT)
  - Receptor desensitization / sensitization
  - Receptor-gated neural projections (CB1→DA disinhibition, D2 autoreceptors, etc.)
  - Full mesolimbic reward circuit (RPE, wanting/liking, opponent process)
  - LTP/LTD plasticity gating (NMDA, D1/D2, ACh, NE explore/exploit)
  - Substantia Nigra (SNc/SNr), PAG, hypothalamic CRH
"""
from .transmitters import TransmitterSystem, NT_NAMES
from .receptors import ReceptorBank
from .projections import Projection, ProjectionGraph
from .nuclei import (VTA, NucleusAccumbens, LocusCoeruleus, RapheNuclei,
                     BasalForebrain, SubstantiaNigra, PeriaqueductalGray,
                     HypothalamicCRH)
from .homeostasis import Homeostasis
from .reuptake import ReuptakeSystem
from .desensitization import ReceptorAdaptation
from .gated_projections import GatedProjectionGraph, GatedProjection, GATED_PROJECTIONS
from .mesolimbic_circuit import MesolimbicCircuit
from .plasticity import PlasticityGate

__all__ = [
    "TransmitterSystem", "NT_NAMES",
    "ReceptorBank",
    "Projection", "ProjectionGraph",
    "VTA", "NucleusAccumbens", "LocusCoeruleus", "RapheNuclei", "BasalForebrain",
    "SubstantiaNigra", "PeriaqueductalGray", "HypothalamicCRH",
    "Homeostasis",
    "ReuptakeSystem",
    "ReceptorAdaptation",
    "GatedProjectionGraph", "GatedProjection", "GATED_PROJECTIONS",
    "MesolimbicCircuit",
    "PlasticityGate",
]
