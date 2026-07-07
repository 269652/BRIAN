# -*- coding: utf-8 -*-
"""CPU discovery harness — search NGL program space for ML algorithms.

Two search modes, both designed to run in seconds on CPU with tiny models:

1. ``run_optimizer_discovery`` — evolve update-rule programs, scoring each by
   actually training a tiny MLP on a synthetic task. Multi-objective Pareto over
   ``(-final_loss, -update_rule_cost)`` so the search trades convergence quality
   against per-step compute (efficiency / throughput).

2. ``run_flow_modulation_discovery`` — evolve programs that modulate the gradient
   flow of a tiny net; fitness adds an **effective-information / synergy** proxy
   from ``neuroslm.information`` so the search rewards modulation that raises
   integration, not just lowers loss.

This is the machinery the "outperform GPT-2 param-matched" goal needs. The
large-scale training that would cash the claim runs through ``brian deploy``;
here we prove the search finds strong, novel update rules on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from neuroslm.genetic.language import Memory, Program
from neuroslm.genetic.optimizer import (
    NGLOptimizer,
    SEED_OPTIMIZERS,
    sgd_program,
)
from neuroslm.genetic.evolve import Objective, auto_evolve, pareto_front
from neuroslm import information

_DIVERGED_LOSS = 1e4


# ---------------------------------------------------------------------------
# Tiny CPU benchmark model + task.
# ---------------------------------------------------------------------------
class _TinyMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x, return_hidden=False):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        if return_hidden:
            return y, h
        return y


def _build_regression(seed: int, d_in=8, d_hidden=16, n=128):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    teacher = _TinyMLP(d_in, d_hidden, 1)
    for p in teacher.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.7
    with torch.no_grad():
        y = teacher(X)
    torch.manual_seed(seed + 1)
    student = _TinyMLP(d_in, d_hidden, 1)
    return X, y, student


def _build_parity(seed: int, n_bits=6, d_hidden=24):
    from neuroslm.synthetic_tasks import parity_task

    X_np, y_np = parity_task(n_bits=n_bits, order=n_bits, n_samples=None)
    X = torch.tensor(X_np, dtype=torch.float32) * 2 - 1  # {0,1} -> {-1,1}
    y = torch.tensor(y_np, dtype=torch.long)
    torch.manual_seed(seed + 7)
    model = _TinyMLP(n_bits, d_hidden, 2)
    return X, y, model


@dataclass
class BenchmarkResult:
    final_loss: float
    curve: List[float]
    cost: int
    diverged: bool


def benchmark_optimizer(program: Program, *, steps: int = 40, seed: int = 0,
                        task: str = "regression") -> BenchmarkResult:
    """Train a tiny model for ``steps`` with the NGL update-rule and score it."""
    if task == "regression":
        X, y, model = _build_regression(seed)
        loss_fn = lambda pred: torch.mean((pred - y) ** 2)
    elif task == "parity":
        X, y, model = _build_parity(seed)
        loss_fn = lambda pred: nn.functional.cross_entropy(pred, y)
    else:
        raise ValueError(f"unknown task {task!r}")

    opt = NGLOptimizer(model.parameters(), program)
    curve: List[float] = []
    diverged = False
    for _ in range(steps):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred)
        lv = float(loss.detach())
        if not np.isfinite(lv) or lv > _DIVERGED_LOSS:
            diverged = True
            break
        loss.backward()
        opt.step()
        curve.append(lv)
    if not curve:
        final = _DIVERGED_LOSS
        diverged = True
    else:
        # final = post-training loss (recompute so it reflects the last update)
        with torch.no_grad():
            fl = float(loss_fn(model(X)).detach())
        final = fl if (np.isfinite(fl) and fl < _DIVERGED_LOSS) else _DIVERGED_LOSS
        if not np.isfinite(fl) or fl >= _DIVERGED_LOSS:
            diverged = True
        curve.append(final)
    return BenchmarkResult(final_loss=final, curve=curve,
                           cost=len(program.instructions), diverged=diverged)


# ---------------------------------------------------------------------------
# Optimizer discovery.
# ---------------------------------------------------------------------------
@dataclass
class DiscoveryOutcome:
    best_program: Program
    best_final_loss: float
    sgd_baseline_loss: float
    history: List[float]
    front: List[Program]
    best_ei: float = 0.0
    front_stats: List[dict] = field(default_factory=list)


def run_optimizer_discovery(
    *,
    seed: int = 0,
    pop_size: int = 24,
    generations: int = 10,
    steps: int = 30,
    task: str = "regression",
    include_sota_seeds: bool = True,
    cost_weight: float = 0.02,
) -> DiscoveryOutcome:
    """Evolve update-rule programs; return the best + the Pareto front.

    Objective (maximised): ``(-final_loss, -cost_weight*n_instructions)``.
    """
    rng = np.random.default_rng(seed)
    sgd_baseline = benchmark_optimizer(sgd_program(lr=0.02), steps=steps, seed=seed, task=task).final_loss

    def evaluate(prog: Program) -> Objective:
        res = benchmark_optimizer(prog, steps=steps, seed=seed, task=task)
        return Objective((-res.final_loss, -cost_weight * res.cost))

    seeds: List[Program] = []
    if include_sota_seeds:
        seeds = [fn() for fn in SEED_OPTIMIZERS.values()]
    else:
        seeds = [sgd_program(lr=0.02)]

    result = auto_evolve(
        evaluate,
        rng,
        pop_size=pop_size,
        generations=generations,
        length=6,
        n_scalar=8,
        n_tensor=12,
        seeds=seeds,
        weights=[1.0, 1.0],
        elite_frac=0.25,
        crossover_rate=0.5,
    )

    front_stats = []
    for p in result.front:
        r = benchmark_optimizer(p, steps=steps, seed=seed, task=task)
        front_stats.append({"final_loss": r.final_loss, "cost": r.cost,
                            "name": p.meta.get("name", "evolved")})
    return DiscoveryOutcome(
        best_program=result.best_program,
        best_final_loss=-result.best_objective.values[0],
        sgd_baseline_loss=sgd_baseline,
        history=result.history,
        front=result.front,
        front_stats=front_stats,
    )


# ---------------------------------------------------------------------------
# Flow-modulation discovery (effective-information objective).
# ---------------------------------------------------------------------------
def _ei_proxy(model: _TinyMLP, X: torch.Tensor, y: torch.Tensor) -> float:
    """Net synergy between two hidden groups and the target — an EI proxy.

    High net synergy = information that lives in the *joint* of hidden units and
    is irreducible to any single unit (the integration signature we want the
    modulation search to raise).
    """
    with torch.no_grad():
        _, h = model(X, return_hidden=True)
    hb = (h > 0).long().cpu().numpy()
    if hb.shape[1] < 2:
        return 0.0
    half = hb.shape[1] // 2
    x1 = hb[:, :half].sum(axis=1) % 2
    x2 = hb[:, half:].sum(axis=1) % 2
    yv = y.cpu().numpy()
    try:
        return float(information.net_synergy(x1, x2, yv))
    except Exception:
        return 0.0


def run_flow_modulation_discovery(
    *,
    seed: int = 0,
    pop_size: int = 16,
    generations: int = 6,
    steps: int = 25,
    ei_weight: float = 0.5,
) -> DiscoveryOutcome:
    """Evolve a gradient-flow modulation program wrapped around an SGD step.

    The evolved program transforms the raw gradient (``t0``) before the update;
    fitness combines task loss with an effective-information proxy, so the search
    is pushed toward modulation that both trains well *and* raises integration.
    """
    rng = np.random.default_rng(seed)
    X, y, _ = _build_parity(seed)

    def evaluate(prog: Program) -> Objective:
        torch.manual_seed(seed + 7)
        model = _TinyMLP(X.shape[1], 24, 2)
        # SGD step but with the gradient first passed through the NGL program
        params = list(model.parameters())
        mems = {id(p): Memory(prog.n_scalar, prog.n_tensor) for p in params}
        lr = 0.1
        diverged = False
        for _ in range(steps):
            model.zero_grad()
            loss = nn.functional.cross_entropy(model(X), y)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    mem = mems[id(p)]
                    mem.write("t0", p.grad.detach())
                    mem.write("t1", p.data)
                    prog.execute(mem)
                    mod = mem.read(prog.out_reg)
                    if mod.shape != p.grad.shape:
                        try:
                            mod = mod.reshape(p.grad.shape)
                        except RuntimeError:
                            mod = mod.mean().expand_as(p.grad)
                    p.data.add_(mod, alpha=-lr)
        with torch.no_grad():
            fl = float(nn.functional.cross_entropy(model(X), y))
        if diverged or not np.isfinite(fl) or fl > _DIVERGED_LOSS:
            return Objective((-_DIVERGED_LOSS, 0.0))
        ei = _ei_proxy(model, X, y)
        return Objective((-fl, ei_weight * ei))

    # seed with identity modulation (pass grad through) ≈ plain SGD
    from neuroslm.genetic.language import Instruction
    identity = Program([Instruction("cscale", "t5", ("t0",), const=1.0)],
                       n_scalar=8, n_tensor=12, out_reg="t5",
                       meta={"name": "identity_mod"})

    result = auto_evolve(
        evaluate,
        rng,
        pop_size=pop_size,
        generations=generations,
        length=5,
        n_scalar=8,
        n_tensor=12,
        seeds=[identity],
        weights=[1.0, 1.0],
        elite_frac=0.25,
    )
    best_obj = result.best_objective
    return DiscoveryOutcome(
        best_program=result.best_program,
        best_final_loss=-best_obj.values[0],
        sgd_baseline_loss=float("nan"),
        history=result.history,
        front=result.front,
        best_ei=best_obj.values[1] / max(ei_weight, 1e-9),
    )
