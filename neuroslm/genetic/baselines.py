# -*- coding: utf-8 -*-
"""Baseline algorithms with explicit tradeoffs — the search's starting points.

The trunk already trains with a known-good optimizer (Adam/AdamW). Discovery is
far more productive when it starts *from* that baseline and searches outward, than
from scratch. This registry holds the standard optimizers as NGL programs together
with their tradeoffs (per-step compute, optimizer-state memory, stability), so a
run can seed from the arch's current algorithm — or from several, to explore the
tradeoff frontier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from neuroslm.genetic.language import Program
from neuroslm.genetic.optimizer import (
    sgd_program, momentum_program, rmsprop_program, adam_program, lion_program,
)


@dataclass
class Baseline:
    name: str
    program: Program
    description: str
    cost: int          # instructions executed per param per step (compute)
    memory: int        # persistent optimizer-state buffers per param
    stability: str     # "low" | "medium" | "high"


def default_baselines() -> Dict[str, Baseline]:
    return {
        "sgd": Baseline(
            "sgd", sgd_program(), "plain gradient descent",
            cost=len(sgd_program().instructions), memory=0, stability="medium"),
        "momentum": Baseline(
            "momentum", momentum_program(), "SGD + heavy-ball momentum",
            cost=len(momentum_program().instructions), memory=1, stability="medium"),
        "rmsprop": Baseline(
            "rmsprop", rmsprop_program(), "per-coordinate rms normalization",
            cost=len(rmsprop_program().instructions), memory=1, stability="high"),
        "adam": Baseline(
            "adam", adam_program(), "bias-corrected first+second moments (trunk default)",
            cost=len(adam_program().instructions), memory=2, stability="high"),
        "lion": Baseline(
            "lion", lion_program(), "sign-momentum; cheap, memory-light",
            cost=len(lion_program().instructions), memory=1, stability="medium"),
    }


def seeds_for(names: List[str]) -> List[Program]:
    """Return the baseline programs for ``names`` (KeyError on an unknown name)."""
    reg = default_baselines()
    out = []
    for n in names:
        if n not in reg:
            raise KeyError(f"unknown baseline {n!r}; known: {sorted(reg)}")
        out.append(reg[n].program.copy())
    return out


def tradeoff_table() -> List[dict]:
    return [
        {"name": b.name, "cost": b.cost, "memory": b.memory,
         "stability": b.stability, "description": b.description}
        for b in default_baselines().values()
    ]
