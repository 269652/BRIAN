# -*- coding: utf-8 -*-
"""Evolve the KL-distillation λ-schedule as an NGL program (gap_nats → λ).

``BRIANHarness._distillation_lambda`` hardcodes a piecewise-linear ramp:
λ ramps from 0 at ``gap ≤ floor`` to ``lambda_max`` at ``gap ≥ ceiling``,
where ``gap = lm_loss_ema - cortex_loss_ema``. This module evolves that
mapping as an NGL program instead, using the same auto-evolve engine
``neuro_evolve.py`` uses for the trunk residual-stream modulation search
(``auto_evolve``/``Objective`` from ``evolve.py`` — no GA internals are
reimplemented here).

The fitness landscape is a proxy simulation of the actual physical
tension the ramp exists to resolve, not a synthetic language-modelling
task (there's no tensor-shaped analogue to reuse from ``neuro_evolve.py``
here — the schedule is a scalar function of a scalar gap):

  * gap > 0 (trunk behind the frozen cortex): distillation pulls the
    trunk's loss down toward the cortex — helpful, the whole point of
    the mechanism.
  * gap < 0 (trunk has already surpassed the cortex): the SAME pull now
    drags the trunk's loss back UP toward a now-worse teacher — harmful.
    A schedule that fails to collapse λ toward 0 once gap turns negative
    pays for it in the simulated final loss.

``distill_linear_program()`` is an EXACT NGL reconstruction of the
current piecewise-linear formula (via the identity
``relu(gap-floor) - relu(gap-ceiling) == clip-to-[0,ceiling-floor]``),
used both as the GA's protected elite seed and as the baseline to beat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch

from neuroslm.genetic.language import Instruction, Memory, Program
from neuroslm.genetic.evolve import Objective, auto_evolve
from neuroslm.genetic.neuro_evolve import NeuroanatomicPrior

_DIVERGED_LOSS = 1e6


def distill_linear_program(floor: float = 0.1, ceiling: float = 2.0,
                            lambda_max: float = 1.0) -> Program:
    """NGL reconstruction of BRIANHarness's piecewise-linear λ ramp.

    ``k = lambda_max / (ceiling - floor)``; then
    ``λ = k · (relu(gap - floor) - relu(gap - ceiling))`` is exactly
    0 below floor, the linear ramp between floor/ceiling, and
    ``lambda_max`` above ceiling — see the class docstring identity.
    """
    k = lambda_max / max(ceiling - floor, 1e-9)
    return Program(
        [
            Instruction("const", "t1", (), const=float(floor)),
            Instruction("const", "t2", (), const=float(ceiling)),
            Instruction("sub", "t3", ("t0", "t1")),
            Instruction("sub", "t4", ("t0", "t2")),
            Instruction("relu", "t5", ("t3",)),
            Instruction("relu", "t6", ("t4",)),
            Instruction("sub", "t7", ("t5", "t6")),
            Instruction("cscale", "t8", ("t7",), const=float(k)),
        ],
        n_scalar=4, n_tensor=10, out_reg="t8",
        meta={"name": "distill_linear", "floor": floor, "ceiling": ceiling,
              "lambda_max": lambda_max},
    )


def _make_lambda_fn(program: Program) -> Callable[[float], float]:
    """Turn an NGL program into a gap_nats -> λ callable.

    Mirrors ``neuro_evolve._make_modulator`` but for a scalar-in/scalar-out
    program instead of a residual-stream tensor gain.
    """

    def lambda_fn(gap_nats: float) -> float:
        mem = Memory(program.n_scalar, program.n_tensor)
        mem.write("t0", torch.tensor(float(gap_nats)))
        program.execute(mem)
        out = mem.read(program.out_reg)
        if not torch.is_tensor(out):
            return 0.0
        val = float(out.reshape(-1)[0]) if out.numel() > 0 else 0.0
        if not math.isfinite(val):
            return 0.0
        return val

    return lambda_fn


def simulate_distillation(program: Program, *, steps: int = 60,
                           lm_loss0: float = 3.0, cortex_loss: float = 1.2,
                           floor_loss: float = 0.8, intrinsic_rate: float = 0.06,
                           distill_rate: float = 0.35, lambda_cap: float = 3.0,
                           seed: int = 0) -> tuple:
    """Simulate a trunk learning under a candidate λ(gap) schedule.

    Each step: the trunk's own gradient descent pulls ``lm_loss`` toward
    ``floor_loss`` (its best achievable loss without any teacher help);
    the distillation term pulls ``lm_loss`` toward ``cortex_loss`` scaled
    by ``λ · gap`` — genuinely helpful when ``gap > 0``, genuinely harmful
    (pulls loss back UP) when ``gap < 0`` and λ hasn't collapsed. Returns
    ``(final_loss, trajectory, invalid)`` — ``invalid`` flags a λ that
    went non-finite or wildly outside ``[0, lambda_cap]`` before clamping.
    """
    lambda_fn = _make_lambda_fn(program)
    lm_loss = float(lm_loss0)
    rng = np.random.default_rng(seed)
    trajectory: List[float] = []
    invalid = False

    for _ in range(steps):
        gap = lm_loss - cortex_loss
        try:
            raw_lam = float(lambda_fn(gap))
        except Exception:
            raw_lam = 0.0
            invalid = True
        if not math.isfinite(raw_lam):
            raw_lam = 0.0
            invalid = True
        elif raw_lam < -1e-6 or raw_lam > lambda_cap * 4:
            invalid = True
        lam = max(0.0, min(lambda_cap, raw_lam))

        lm_loss -= intrinsic_rate * (lm_loss - floor_loss)
        lm_loss -= distill_rate * lam * gap * 0.1
        lm_loss += float(rng.normal(0.0, 0.01))
        lm_loss = max(lm_loss, 0.05)

        if not math.isfinite(lm_loss):
            invalid = True
            lm_loss = _DIVERGED_LOSS
            trajectory.append(lm_loss)
            break
        trajectory.append(lm_loss)

    final = trajectory[-1] if trajectory else float(lm_loss0)
    if not math.isfinite(final):
        final, invalid = _DIVERGED_LOSS, True
    return final, trajectory, invalid


def evaluate_schedule(program: Program, *, seed: int = 0, steps: int = 60,
                       prior: NeuroanatomicPrior | None = None) -> tuple:
    """Return ``(final_loss, plausibility)`` for a candidate λ-schedule."""
    prior = prior or NeuroanatomicPrior()
    final_loss, _, invalid = simulate_distillation(program, steps=steps, seed=seed)
    if invalid:
        final_loss = max(final_loss, _DIVERGED_LOSS)
    return final_loss, prior.score(program)


@dataclass
class DistillOutcome:
    best_program: Program
    best_final_loss: float
    best_plausibility: float
    baseline_final_loss: float
    history: List[float]
    front: List[Program]
    front_stats: List[dict]


def run_distill_evolution(*, seed: int = 0, pop_size: int = 16, generations: int = 8,
                          steps: int = 60, plausibility_weight: float = 0.3,
                          loss_weight: float = 1.0, progress=False) -> DistillOutcome:
    """Evolve a gap→λ schedule; Pareto over (−final_loss, +plausibility).

    ``distill_linear_program()`` (the current piecewise-linear formula) is
    seeded and protected by elitism, so the reported best is never worse
    than the mechanism already shipping in ``BRIANHarness``.
    """
    from neuroslm.genetic.discovery import _emit, _make_progress, _describe_champion
    rng = np.random.default_rng(seed)
    prior = NeuroanatomicPrior()
    baseline = distill_linear_program()
    baseline_loss, _ = evaluate_schedule(baseline, seed=seed, steps=steps, prior=prior)
    scale = max(baseline_loss, 1.0)
    if progress:
        _emit(progress, f"[distill] pop={pop_size} gens={generations} steps={steps} "
              f"| current piecewise-linear ramp final_loss={baseline_loss:.4f}")

    def evaluate(prog: Program) -> Objective:
        loss, plaus = evaluate_schedule(prog, seed=seed, steps=steps, prior=prior)
        return Objective((-loss_weight * loss / scale, plausibility_weight * plaus))

    on_gen = _make_progress(
        progress, "distill",
        fmt=lambda o: f"best_final_loss={-o.values[0] * scale / max(loss_weight, 1e-9):.4f} "
                      f"plaus={o.values[1] / max(plausibility_weight, 1e-9):.2f}",
        describe=_describe_champion,
    ) if progress else None

    result = auto_evolve(
        evaluate, rng,
        pop_size=pop_size, generations=generations,
        length=6, n_scalar=4, n_tensor=10,
        seeds=[baseline],
        weights=[1.0, 0.3],
        elite_frac=0.3,
        crossover_rate=0.5,
        on_generation=on_gen,
    )

    best_loss, best_plaus = evaluate_schedule(result.primary_program, seed=seed,
                                              steps=steps, prior=prior)
    front_stats = []
    for p in result.front:
        fl, pl = evaluate_schedule(p, seed=seed, steps=steps, prior=prior)
        front_stats.append({"final_loss": fl, "plausibility": pl,
                            "name": p.meta.get("name", "evolved")})
    return DistillOutcome(
        best_program=result.primary_program,
        best_final_loss=best_loss,
        best_plausibility=best_plaus,
        baseline_final_loss=baseline_loss,
        history=result.history,
        front=result.front,
        front_stats=front_stats,
    )


def install_distillation_schedule_from_store(harness, name: str, *, store_dir) -> dict:
    """Load a saved λ-schedule from ``modulations/<name>.neuro`` and wire it
    into a live ``BRIANHarness`` via ``install_distillation_schedule``.

    Simpler than the layer-modulation install path (``modulation_install.
    install_from_store``): a scalar gap->λ function has no "forward pass on
    a real batch" analogue to live-validate against, so this trusts the
    discovery run's own baseline-beats-current elitism guarantee (see
    ``run_distill_evolution``) instead of re-validating here. Raises
    ``KeyError`` (propagated from ``ModulationStore.get``) if ``name``
    isn't in the store.
    """
    from neuroslm.genetic.modulation_store import ModulationStore

    rec = ModulationStore(store_dir).get(name)
    fn = _make_lambda_fn(rec.program)
    harness.install_distillation_schedule(fn)
    return {"installed": name, "metrics": dict(rec.metrics)}
