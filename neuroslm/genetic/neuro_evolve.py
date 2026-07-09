# -*- coding: utf-8 -*-
"""Neuroanatomically-constrained auto-evolve of a trunk neuromodulation.

The SmolLM trunk's perplexity is too high; the classic lever is to reshape the
residual stream with a *neuromodulation* — a learned, biologically-motivated gain
control — rather than to keep scaling parameters. This module evolves that
modulation as an NGL program and injects it into a tiny CPU LM's residual stream,
scoring each candidate by validation perplexity **jointly with a neuroanatomic
realism prior** (Pareto), so the search improves language modelling without
drifting into biologically implausible mechanisms.

Honesty: training SmolLM to a competitive PPL needs a GPU (`brian deploy`). What
runs here is the *engine* — the same search, on a tiny model, proving it never
regresses below the unmodulated trunk and that the realism prior shapes
selection. The discovered gain law is what a deploy would wire into the real
trunk.

The realism prior is grounded in canonical neuromodulation principles:
- **Divisive normalization** (Carandini & Heeger 2012) — `div`/`rms`/`norm`.
- **Multiplicative gain modulation** (Salinas & Thier 2000) — `mul`.
- **Saturating / homeostatic dose-response** — bounded ops (`sigmoid`, `tanh`,
  `clip`, `softmax`); penalize unbounded runaway (`exp`, chained `outer`).
- **Metabolic economy** — shorter programs preferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from neuroslm.genetic.language import Instruction, Memory, Program, REGISTRY
from neuroslm.genetic.evolve import Objective, auto_evolve, pareto_front
from neuroslm.genetic.simplify import dead_code_eliminate

_DIVERGED_PPL = 1e6


# ---------------------------------------------------------------------------
# Neuroanatomic realism prior over an NGL modulation program.
# ---------------------------------------------------------------------------
_BOUNDED = {"sigmoid", "tanh", "clip", "softmax", "relu", "silu", "gelu"}
_DIVISIVE = {"div", "rms", "norm", "rmsnorm", "layernorm"}
_GAIN = {"mul", "cscale"}
_RUNAWAY = {"exp", "outer"}


class NeuroanatomicPrior:
    """Score an NGL modulation in [0,1] for biological plausibility."""

    def __init__(self, length_penalty: float = 0.03):
        self.length_penalty = length_penalty

    def score(self, program: Program) -> float:
        # Score the *effective* mechanism, not vestigial ops: dead code that never
        # reaches the output shouldn't earn (or lose) plausibility credit.
        program = dead_code_eliminate(program)
        ops = [ins.op for ins in program.instructions]
        if not ops:
            return 0.5  # empty = identity = cheap, neutral
        n = len(ops)
        bounded = sum(o in _BOUNDED for o in ops)
        divisive = sum(o in _DIVISIVE for o in ops)
        gain = sum(o in _GAIN for o in ops)
        runaway = sum(o in _RUNAWAY for o in ops)

        tonic = sum(o == "const" for o in ops)         # tonic (baseline) gain tone

        # canonical-motif credit (each capped so one op-type can't dominate)
        s = 0.30                                        # baseline plausibility
        s += 0.30 * min(1.0, bounded / max(1, n))      # homeostatic saturation
        s += 0.25 * min(1.0, divisive / max(1, n))     # divisive normalization
        s += 0.20 * min(1.0, gain / max(1, n))         # multiplicative gain
        s += 0.10 * min(1.0, tonic)                    # tonic gain tone (a flat
        #                                                gain level is biologically
        #                                                real — baseline NT tone)

        # penalties
        s -= 0.35 * min(1.0, runaway / max(1, n))      # implausible runaway
        s -= self.length_penalty * max(0, n - 4)       # metabolic economy
        return float(max(0.0, min(1.0, s)))


# ---------------------------------------------------------------------------
# Tiny CPU language model with a modulated residual stream.
# ---------------------------------------------------------------------------
class _TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 24, ctx: int = 12, n_heads: int = 2):
        super().__init__()
        self.d = d
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor, modulate=None) -> torch.Tensor:
        T = ids.shape[1]
        h = self.tok(ids) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h), attn_mask=mask,
                         need_weights=False)
        h = h + a
        if modulate is not None:
            h = modulate(h)                 # neuromodulation on the residual stream
        h = h + self.ff(self.n2(h))
        return self.head(self.nf(h))


def _markov_corpus(seed: int, vocab: int = 8, ctx: int = 12, n_seq: int = 96):
    """Sequences from a fixed order-1 Markov chain — learnable, bounded-entropy."""
    g = torch.Generator().manual_seed(seed)
    trans = torch.rand(vocab, vocab, generator=g)
    trans = trans / trans.sum(-1, keepdim=True)
    seqs = torch.zeros(n_seq, ctx + 1, dtype=torch.long)
    cur = torch.randint(0, vocab, (n_seq,), generator=g)
    seqs[:, 0] = cur
    for t in range(1, ctx + 1):
        probs = trans[cur]
        cur = torch.multinomial(probs, 1, generator=g).squeeze(1)
        seqs[:, t] = cur
    split = int(0.75 * n_seq)
    return seqs[:split], seqs[split:], vocab


def _make_modulator(program: Program):
    """Turn an NGL program into a residual-stream gain function h -> h * g(h)."""

    def modulate(h: torch.Tensor) -> torch.Tensor:
        # fresh memory per call: each forward is independent, so a stateful
        # program must not accumulate state across training steps (that leak was
        # a divergence source — state grew unbounded over a run).
        mem = Memory(program.n_scalar, program.n_tensor)
        mem.write("t0", h.detach())
        program.execute(mem)
        g = mem.read(program.out_reg)
        if not torch.is_tensor(g):
            return h
        if g.shape != h.shape:
            try:
                g = g.reshape(h.shape)
            except RuntimeError:
                g = g.mean()  # scalar broadcast gain
        g = torch.nan_to_num(g, nan=1.0, posinf=1.0, neginf=1.0).clamp(-8.0, 8.0)
        return h * g

    return modulate


def evaluate_modulation(program: Program, *, seed: int = 0, steps: int = 30,
                        prior: NeuroanatomicPrior | None = None,
                        device=None) -> Tuple[float, float]:
    """Train the tiny LM with this modulation; return (val_ppl, plausibility)."""
    prior = prior or NeuroanatomicPrior()
    dev = torch.device(device) if device is not None else torch.device("cpu")
    train, val, vocab = _markov_corpus(seed)
    train, val = train.to(dev), val.to(dev)
    torch.manual_seed(seed + 3)
    model = _TinyLM(vocab).to(dev)
    modulate = _make_modulator(program)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    diverged = False
    for _ in range(steps):
        opt.zero_grad()
        logits = model(train[:, :-1], modulate=modulate)
        loss = loss_fn(logits.reshape(-1, vocab), train[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    if diverged:
        return _DIVERGED_PPL, prior.score(program)
    with torch.no_grad():
        vlogits = model(val[:, :-1], modulate=modulate)
        vloss = loss_fn(vlogits.reshape(-1, vocab), val[:, 1:].reshape(-1))
    vloss = float(vloss)
    if not np.isfinite(vloss):
        return _DIVERGED_PPL, prior.score(program)
    ppl = float(np.exp(min(vloss, 20.0)))
    return ppl, prior.score(program)


def identity_modulation() -> Program:
    """Gain ≡ 1 — the unmodulated trunk baseline (and the GA's elite seed)."""
    return Program(
        [Instruction("const", "t5", (), const=1.0)],
        n_scalar=4, n_tensor=8, out_reg="t5",
        meta={"name": "identity"},
    )


# ---------------------------------------------------------------------------
# The evolution loop.
# ---------------------------------------------------------------------------
@dataclass
class TrunkOutcome:
    best_program: Program
    best_val_ppl: float
    best_plausibility: float
    baseline_val_ppl: float
    history: List[float]
    front: List[Program]
    front_stats: List[dict]


def run_trunk_evolution(*, seed: int = 0, pop_size: int = 16, generations: int = 8,
                        steps: int = 30, plausibility_weight: float = 1.0,
                        ppl_weight: float = 1.0, device: str = "cpu",
                        progress=False) -> TrunkOutcome:
    """Evolve a residual-stream neuromodulation; Pareto over (−PPL, +plausibility).

    The identity modulation is seeded and preserved by elitism, so the best
    candidate is never worse than the unmodulated trunk. The plausibility term
    keeps the search inside the neuroanatomically realistic region. ``device``
    scales the tiny-LM training onto a T4/cuda. ``progress`` streams a
    per-generation line.
    """
    from neuroslm.genetic.discovery import _resolve_device, _emit, _make_progress, _describe_champion
    rng = np.random.default_rng(seed)
    prior = NeuroanatomicPrior()
    dev = _resolve_device(device)
    baseline_ppl, _ = evaluate_modulation(identity_modulation(), seed=seed, steps=steps, device=dev)
    # normalise PPL into a comparable scale so the weighted scalar is balanced
    scale = max(baseline_ppl, 1.0)
    if progress:
        _emit(progress, f"[trunk] device={dev.type} pop={pop_size} gens={generations} "
              f"steps={steps} | unmodulated trunk ppl={baseline_ppl:.3f}")

    def evaluate(prog: Program) -> Objective:
        ppl, plaus = evaluate_modulation(prog, seed=seed, steps=steps, prior=prior, device=dev)
        return Objective((-ppl_weight * ppl / scale, plausibility_weight * plaus))

    on_gen = _make_progress(
        progress, "trunk",
        fmt=lambda o: f"best_ppl={-o.values[0] * scale / max(ppl_weight, 1e-9):.3f} "
                      f"plaus={o.values[1] / max(plausibility_weight, 1e-9):.2f}",
        describe=_describe_champion,
    ) if progress else None
    result = auto_evolve(
        evaluate, rng,
        pop_size=pop_size, generations=generations,
        length=5, n_scalar=4, n_tensor=8,
        seeds=[identity_modulation()],
        weights=[1.0, 0.3],   # PPL dominates; plausibility breaks ties / guards realism
        elite_frac=0.3,
        crossover_rate=0.5,
        on_generation=on_gen,
    )

    # recover raw ppl/plausibility for the reported best + front — report the
    # lowest-ppl champion (`primary_program`, monotonic by construction), not
    # the combined (ppl, plausibility) scalar's champion: a more "plausible"
    # program can legitimately outrank a lower-ppl one on that scalar, which
    # would otherwise make `best_val_ppl` look like it regressed.
    best_ppl, best_plaus = evaluate_modulation(result.primary_program, seed=seed, steps=steps, prior=prior, device=dev)
    front_stats = []
    for p in result.front:
        pp, pl = evaluate_modulation(p, seed=seed, steps=steps, prior=prior, device=dev)
        front_stats.append({"val_ppl": pp, "plausibility": pl,
                            "name": p.meta.get("name", "evolved")})
    return TrunkOutcome(
        best_program=result.primary_program,
        best_val_ppl=best_ppl,
        best_plausibility=best_plaus,
        baseline_val_ppl=baseline_ppl,
        history=result.history,
        front=result.front,
        front_stats=front_stats,
    )
