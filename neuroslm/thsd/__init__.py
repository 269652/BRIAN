# -*- coding: utf-8 -*-
"""THSD (Topological Hyper-Sheaf-Dynamics) Framework.

The mathematical foundation for architecture as a simplicial complex K
with cellular sheaves, cohomology guards, and integrated information
dynamics.

Public surface
--------------
* :class:`SimplexComplex`         — the underlying K
* :class:`CellularSheaf`          — stalks + Fisher metrics + restrictions
* :class:`CoboundaryOperator`     — δ⁰, δ¹ and H¹ contradiction tests
* :class:`PhiDynamicsComputer`    — IIT-style Φ via bipartition search
* :class:`SymbolicSimplex`        — discovery-operator 1/4 (Symbolic
                                    Expression Units; see
                                    ``docs/formal_framework.md`` §3).
"""

from neuroslm.thsd.engine import (
    CellularSheaf,
    CoboundaryOperator,
    PhiDynamicsComputer,
    SimplexComplex,
    SymbolicSimplex,
)

__all__ = [
    "CellularSheaf",
    "CoboundaryOperator",
    "PhiDynamicsComputer",
    "SimplexComplex",
    "SymbolicSimplex",
]
