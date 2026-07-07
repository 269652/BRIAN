# -*- coding: utf-8 -*-
"""Exploration wired into training — search every N steps, keep-if-better, ledger.

Every ``explore_every`` steps the explorer runs a short NGL modulation search on
the *current* model, A/B-tests the winner (baseline = identity modulation), and
keeps it only if the metric improves — installing it into the running model
before training resumes. Every attempt is recorded to a persistent
``SearchLedger`` keyed by semantic signature, and prior-run **duds are skipped**
so a fresh run doesn't re-search the same space.

The explorer is model-agnostic: it takes a ``score_fn(program) -> metric`` (lower
is better). ``run_training_with_exploration`` wires it to a tiny CPU LM as the
runnable miniature; the same explorer attaches to the real trunk by supplying a
``score_fn`` that applies a modulation to the trunk's residual stream and returns
a validation metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from neuroslm.genetic.language import Program
from neuroslm.genetic.evolve import Objective, auto_evolve
from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.neuro_evolve import identity_modulation


@dataclass
class ExploreConfig:
    explore_every: int = 500
    pop_size: int = 16
    generations: int = 8
    length: int = 5
    n_scalar: int = 4
    n_tensor: int = 8
    tol: float = 1e-4          # metric must improve by more than this to "keep"
    dud_penalty: float = 1e9


@dataclass
class ExploreResult:
    step: int
    improved: bool
    baseline: float
    best_score: float
    best_program: Program
    n_evaluated: int
    n_skipped_duds: int


class TrainingExplorer:
    def __init__(self, ledger: SearchLedger, config: ExploreConfig = None,
                 run_id: str = "run"):
        self.ledger = ledger
        self.cfg = config or ExploreConfig()
        self.run_id = run_id

    def maybe_explore(self, step: int, score_fn: Callable[[Program], float]) -> Optional[ExploreResult]:
        if step > 0 and step % self.cfg.explore_every == 0:
            return self.explore(step, score_fn)
        return None

    def explore(self, step: int, score_fn: Callable[[Program], float]) -> ExploreResult:
        cfg = self.cfg
        rng = np.random.default_rng(hash((self.run_id, step)) % (2**32))
        baseline = float(score_fn(identity_modulation()))

        evaluated: dict = {}     # signature -> (program, score)
        skipped = [0]

        def evaluate(prog: Program) -> Objective:
            # skip patterns prior runs already found unhelpful (ledger not mutated
            # until after this explore, so this reflects PAST runs only)
            if self.ledger.is_dud(prog):
                skipped[0] += 1
                return Objective((-cfg.dud_penalty,))
            sig = self.ledger.signature(prog)
            if sig in evaluated:
                s = evaluated[sig][1]
            else:
                s = float(score_fn(prog))
                evaluated[sig] = (prog, s)
            return Objective((-s,))

        result = auto_evolve(
            evaluate, rng,
            pop_size=cfg.pop_size, generations=cfg.generations,
            length=cfg.length, n_scalar=cfg.n_scalar, n_tensor=cfg.n_tensor,
            seeds=[identity_modulation()],
            elite_frac=0.3, crossover_rate=0.5,
        )
        best_prog = result.best_program
        best_score = float(score_fn(best_prog))
        improved = best_score < baseline - cfg.tol

        # record every distinct pattern we searched (so future runs skip duds)…
        for sig, (prog, s) in evaluated.items():
            self.ledger.record(prog, outcome="searched", delta=s - baseline,
                               step=step, run_id=self.run_id)
        # …then the verdict on the winner (latest outcome wins on dedup)
        self.ledger.record(best_prog, outcome="kept" if improved else "rejected",
                           delta=best_score - baseline, metric_before=baseline,
                           metric_after=best_score, step=step, run_id=self.run_id)

        return ExploreResult(
            step=step, improved=improved, baseline=baseline, best_score=best_score,
            best_program=best_prog, n_evaluated=len(evaluated),
            n_skipped_duds=skipped[0],
        )


# ---------------------------------------------------------------------------
# Runnable miniature: a tiny-LM training loop with exploration wired in.
# ---------------------------------------------------------------------------
def run_training_with_exploration(*, total_steps: int = 2000, explore_every: int = 500,
                                  seed: int = 0, ledger: SearchLedger = None,
                                  pop_size: int = 12, generations: int = 4,
                                  inner_steps: int = 20) -> dict:
    """Train a tiny CPU LM; every ``explore_every`` steps search + keep-if-better.

    This is the miniature of "wire exploration into training": the residual-stream
    modulation is A/B-tested on the live model and installed only if it lowers
    validation perplexity. The same loop attaches to the real trunk on GPU.
    """
    import torch
    import torch.nn as nn
    from neuroslm.genetic.neuro_evolve import _TinyLM, _markov_corpus, _make_modulator

    ledger = ledger or SearchLedger(":memory:")
    train, val, vocab = _markov_corpus(seed)
    torch.manual_seed(seed + 3)
    model = _TinyLM(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    current_mod = identity_modulation()

    def val_ppl(prog: Program) -> float:
        modulate = _make_modulator(prog)
        with torch.no_grad():
            logits = model(val[:, :-1], modulate=modulate)
            vloss = loss_fn(logits.reshape(-1, vocab), val[:, 1:].reshape(-1))
        return float(np.exp(min(float(vloss), 20.0)))

    explorer = TrainingExplorer(
        ledger, ExploreConfig(explore_every=explore_every, pop_size=pop_size,
                              generations=generations), run_id=f"run-{seed}")

    explorations: List[dict] = []
    for step in range(1, total_steps + 1):
        modulate = _make_modulator(current_mod)
        opt.zero_grad()
        logits = model(train[:, :-1], modulate=modulate)
        loss = loss_fn(logits.reshape(-1, vocab), train[:, 1:].reshape(-1))
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        res = explorer.maybe_explore(step, val_ppl)
        if res is not None:
            if res.improved:
                current_mod = res.best_program   # install the winner
            explorations.append({
                "step": step, "improved": res.improved,
                "baseline_ppl": round(res.baseline, 4),
                "best_ppl": round(res.best_score, 4),
                "evaluated": res.n_evaluated, "skipped_duds": res.n_skipped_duds,
                "program": res.best_program.to_source() if res.improved else None,
            })

    return {
        "explorations": explorations,
        "final_val_ppl": val_ppl(current_mod),
        "installed_modulation": current_mod.to_source(),
        "ledger_stats": ledger.stats(),
    }
