# -*- coding: utf-8 -*-
"""NGL — the Neuro-Genetic Language and its algorithm-discovery harness.

See ``docs/dsl_subsystem_roadmap.md`` §NGL for the design. Public surface:

- ``language``  — register-machine core (``Program``, ``Instruction``, ``REGISTRY``)
- ``optimizer`` — SOTA optimizers expressed as NGL programs + a torch adapter
- ``evolve``    — genetic operators (mutate/crossover) + Pareto GA
- ``discovery`` — CPU discovery harness for optimizer / flow-modulation search
"""
from neuroslm.genetic.language import (  # noqa: F401
    Instruction,
    Memory,
    OpSpec,
    Program,
    REGISTRY,
    OP_FAMILIES,
)
