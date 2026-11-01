# -*- coding: utf-8 -*-
"""Controller φ — integrated information of the runtime's controller
(architecture.md §14.14).

**What is computed.** Integrated information φ as defined in IIT 2.0
(Balduzzi & Tononi 2008, PLoS Comput Biol 4(6):e1000091), *exactly*,
from an **interventional** state-by-node transition-probability matrix
over five binary macro-nodes the runtime genuinely has:

    SYNC   workspace phase coherence |mean e^{iθ}| ≥ 0.5
    DA     DA level above its baseline
    ACh    ACh level above its baseline
    GABA   GABA level above its baseline
    BORED  boredom ≥ 0.5

Each of the 2⁵ states is SET (phases locked or scrambled, NT logits
kicked ±1 around μ, boredom 0.75 / 0.1), one idle tick runs, and the
next state is read — averaged over ``samples`` seeded repeats. This is
a TPM by intervention, not by observation.

**What it is not.** Not IIT 3.0/4.0 Φ^max (which needs the full
cause-effect-structure machinery, e.g. PyPhi); not a property of the
LLM, whose billions of continuous units have no defined grain; and,
by IIT's own account, not a property of the hardware either. It is
the integration of the controller's causal graph at THIS grain, and
its ablation contrast is the evidential part.

φ(x₁) for current state x₁:
    whole   p(X₀ | x₁)       ∝ Π_i P(x₁ᵢ | X₀)        (uniform prior)
    part M  p(M₀ | m₁)       ∝ Σ_{rest₀} 2^{-|rest|} Π_{i∈M} P(x₁ᵢ | M₀, rest₀)
    EI(P)   = KL( p(X₀|x₁) ‖ Π_k p(M_k₀ | m_k₁) )     [bits]
    MIP     = argmin_P EI(P) / min_k |M_k|             (bipartitions)
    φ       = EI(MIP)
TPM rows are little-endian state indices (PyPhi convention).
"""
from __future__ import annotations

import itertools
import math
import random as _random
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONTROLLER_NODES: Tuple[str, ...] = ("SYNC", "DA", "ACh", "GABA", "BORED")
_NT_NODES = ("DA", "ACh", "GABA")
_KICK = 1.0
_BORED_ON, _BORED_OFF, _BORED_BAR = 0.75, 0.1, 0.5
_SYNC_BAR = 0.5


def _bits(index: int, n: int) -> Tuple[int, ...]:
    return tuple((index >> i) & 1 for i in range(n))


def _index(bits: Sequence[int]) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))


def _p_next(tpm: np.ndarray, prev: int, node: int, value: int) -> float:
    p_on = float(tpm[prev, node])
    return p_on if value else 1.0 - p_on


def _part_repertoire(tpm: np.ndarray, n: int, part: Sequence[int],
                     x1: Sequence[int]) -> Dict[Tuple[int, ...], float]:
    """p(M₀ | m₁) with the complement's past replaced by uniform noise."""
    rest = [i for i in range(n) if i not in part]
    rep: Dict[Tuple[int, ...], float] = {}
    for m0 in itertools.product((0, 1), repeat=len(part)):
        acc = 0.0
        for r0 in itertools.product((0, 1), repeat=len(rest)):
            prev = [0] * n
            for i, b in zip(part, m0):
                prev[i] = b
            for i, b in zip(rest, r0):
                prev[i] = b
            pi = _index(prev)
            like = 1.0
            for i in part:
                like *= _p_next(tpm, pi, i, x1[i])
            acc += like
        rep[m0] = acc / (2 ** len(rest))
    z = sum(rep.values())
    if z <= 0.0:
        return {k: 1.0 / len(rep) for k in rep}
    return {k: v / z for k, v in rep.items()}


def phi_iit2(tpm: np.ndarray, state: Sequence[int]
             ) -> Tuple[float, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """IIT 2.0 φ of ``state`` under ``tpm``; returns (φ bits, MIP)."""
    tpm = np.asarray(tpm, dtype=float)
    rows, n = tpm.shape if tpm.ndim == 2 else (0, 0)
    if n == 0 or rows != 2 ** n or len(state) != n:
        raise ValueError(f"tpm must be (2^n, n) for an n-node state; got "
                         f"{tpm.shape} and a {len(state)}-node state")
    x1 = tuple(int(b) for b in state)
    whole = np.array([np.prod([_p_next(tpm, s0, i, x1[i]) for i in range(n)])
                      for s0 in range(rows)])
    if whole.sum() <= 0.0:
        return 0.0, (tuple(range(n)), ())
    whole = whole / whole.sum()

    best = None
    for k in range(1, n // 2 + 1):
        for part in itertools.combinations(range(n), k):
            other = tuple(i for i in range(n) if i not in part)
            if k == n - k and part > other:
                continue
            ra = _part_repertoire(tpm, n, part, x1)
            rb = _part_repertoire(tpm, n, other, x1)
            ei = 0.0
            for s0 in range(rows):
                p = whole[s0]
                if p <= 0.0:
                    continue
                b = _bits(s0, n)
                q = ra[tuple(b[i] for i in part)] * rb[tuple(b[i] for i in other)]
                ei += p * math.log2(p / q) if q > 0 else float("inf")
            norm = ei / min(len(part), len(other))
            if best is None or norm < best[0]:
                best = (norm, ei, (tuple(part), other))
    return float(best[1]), best[2]


# ── interventions on a live CognitiveRuntime ─────────────────────────

def _coherence(theta: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


def apply_state(rt, state: Sequence[int]) -> None:
    """SET the controller into ``state`` (an intervention, do(·))."""
    if rt._osc is None:
        raise ValueError("controller φ needs the workspace oscillator on")
    sync, *nt_bits, bored = (int(b) for b in state)
    osc = rt._osc
    osc.theta = (np.zeros(osc.N) if sync
                 else osc._rng.uniform(0.0, 2 * math.pi, osc.N))
    for ch, on in zip(_NT_NODES, nt_bits):
        target = rt.nt._mu[ch] + (_KICK if on else -_KICK)
        rt.nt.perturb(ch, target - rt.nt._y[ch])
    rt._boredom = _BORED_ON if bored else _BORED_OFF


def read_state(rt) -> Tuple[int, ...]:
    levels = rt.nt.levels()
    base = rt.nt.baselines
    base = base() if callable(base) else base
    return (int(_coherence(rt._osc.theta) >= _SYNC_BAR),
            *(int(levels[ch] > base[ch]) for ch in _NT_NODES),
            int(rt._boredom >= _BORED_BAR))


def estimate_controller_tpm(seams, cfg, samples: int = 4, seed: int = 0,
                            seed_text: str = ("I watched the river flow "
                                              "past the old mill."),
                            progress=None) -> np.ndarray:
    """Interventional state-by-node TPM over CONTROLLER_NODES."""
    n = len(CONTROLLER_NODES)
    tpm = np.zeros((2 ** n, n))
    for idx in range(2 ** n):
        acc = np.zeros(n)
        for k in range(samples):
            s = seed * 100000 + idx * 97 + k
            if seams.reseed is not None:
                seams.reseed(s)
            rt = seams.runtime(cfg=cfg, rng=_random.Random(s))
            rt.memory.add(seed_text,
                          content_vec=list(seams.embed_fn(seed_text)),
                          context={"kind": "observed", "source": "phi"})
            apply_state(rt, _bits(idx, n))
            rt.tick()
            acc += np.array(read_state(rt))
        tpm[idx] = acc / samples
        if progress is not None and idx % 8 == 7:
            progress(f"    controller_phi: {idx + 1}/{2 ** n} states")
    return tpm


def probe_controller_phi(seams, cfg=None, samples: int = 4, seed: int = 0,
                         progress=None) -> dict:
    """φ of the controller, full vs the NT→oscillator feedback cut."""
    from neuroslm.cognition.runtime import MindConfig
    cfg = cfg or MindConfig()
    if not cfg.workspace_oscillator:
        cfg = replace(cfg, workspace_oscillator=True)
    n = len(CONTROLLER_NODES)
    conds = {}
    for name, ov in (("full", {}),
                     ("no_osc_feedback", {"osc_nt_feedback": False})):
        tpm = estimate_controller_tpm(seams, replace(cfg, **ov),
                                      samples=samples, seed=seed,
                                      progress=progress)
        per_state = []
        for idx in range(2 ** n):
            phi, mip = phi_iit2(tpm, _bits(idx, n))
            per_state.append({"state": list(_bits(idx, n)), "phi": phi,
                              "mip": [list(p) for p in mip]})
        phis = [p["phi"] for p in per_state]
        top = max(per_state, key=lambda p: p["phi"])
        conds[name] = {"phi_mean": float(np.mean(phis)),
                       "phi_max": float(max(phis)),
                       "argmax_state": top["state"], "argmax_mip": top["mip"],
                       "phi_per_state": per_state,
                       "tpm": tpm.round(4).tolist()}
        if progress is not None:
            progress(f"  controller_phi/{name}: φ_mean="
                     f"{conds[name]['phi_mean']:.3f} φ_max="
                     f"{conds[name]['phi_max']:.3f}")
    return {"indicator": "IIT/φ (IIT 2.0, controller macro-nodes)",
            "nodes": list(CONTROLLER_NODES), "samples": samples,
            "question": "How much integrated information does the "
                        "controller's interventional causal graph carry, "
                        "and does cutting the NT→workspace feedback "
                        "reduce it?",
            "conditions": conds,
            "contrasts": {"phi_mean_full_minus_no_osc_feedback":
                          conds["full"]["phi_mean"]
                          - conds["no_osc_feedback"]["phi_mean"]}}
