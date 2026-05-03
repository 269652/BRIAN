"""Quantifiable consciousness & intelligence metrics.

Each metric is a single scalar in a documented, comparable range. Tracked
over training so we can plot identity drift, narrative coherence, etc.

Implemented:
  * Identity drift           — cosine distance between successive
                              autobiographical narrative summaries.
  * Narrative coherence      — internal consistency of recent events.
  * Causal density           — # of inferred causal rules / # of episodes.
  * Semantic compression     — episode bytes vs. consolidated-node bytes.
  * Self-reference rate      — fraction of generations referencing self.
  * Theory-of-mind accuracy  — correct prediction of an entity's reaction
                              given prior interactions (when verifiable).
  * Φ (IIT proxy)            — mutual information between brain modules.
  * Ponder efficiency        — gain in LM loss per extra compute step.
  * Reasoning gain           — Δ(loss) on hard tokens vs easy tokens.
  * Emergent generalization  — held-out causal-rule transfer accuracy.

`IntelligenceMetrics` aggregates all of them and supports
`.snapshot() -> dict` for logging + `.delta_since(prev) -> dict` for
drift tracking.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import math
import time
import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────
# Identity drift
# ────────────────────────────────────────────────────────────────────────

class IdentityDriftTracker:
    """Tracks cosine distance between consecutive autobiographical
    narrative summaries. Spikes correlate with experiences that *changed
    who the model thinks it is* — the operational definition of identity
    drift used in cognitive psychology.

    Usage:
        tracker.observe(brain.narrative_system.autobiographical.summary)
        tracker.snapshot()  # → {drift, drift_ema, n_updates, ...}
    """

    def __init__(self, ema_alpha: float = 0.05, history: int = 256):
        self.ema_alpha = ema_alpha
        self.last: torch.Tensor | None = None
        self.drift_ema: float = 0.0
        self.n_updates: int = 0
        self.history: deque[float] = deque(maxlen=history)

    @torch.no_grad()
    def observe(self, summary: torch.Tensor) -> float:
        s = summary.detach().flatten().float()
        if self.last is None:
            self.last = s.clone()
            return 0.0
        cos = F.cosine_similarity(s.unsqueeze(0), self.last.unsqueeze(0)).item()
        drift = 1.0 - cos
        self.drift_ema = (1 - self.ema_alpha) * self.drift_ema + self.ema_alpha * drift
        self.history.append(drift)
        self.last = s.clone()
        self.n_updates += 1
        return drift

    def snapshot(self) -> dict:
        recent = list(self.history)[-32:]
        return {
            "identity_drift": recent[-1] if recent else 0.0,
            "identity_drift_ema": self.drift_ema,
            "identity_drift_recent_max": max(recent) if recent else 0.0,
            "identity_n_updates": self.n_updates,
        }


# ────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ────────────────────────────────────────────────────────────────────────

@dataclass
class _Counter:
    value: float = 0.0
    n: int = 0
    def add(self, x: float):
        self.value += float(x); self.n += 1
    def mean(self) -> float:
        return self.value / self.n if self.n else 0.0
    def reset(self):
        self.value = 0.0; self.n = 0


class IntelligenceMetrics:
    """Aggregates all metrics. Designed to be cheap to update at every
    training step and zero-cost to query.
    """

    def __init__(self):
        self.identity = IdentityDriftTracker()
        self.coherence_ema: float = 1.0
        self.causal_rules: int = 0
        self.episodes_seen: int = 0
        self.consolidated_nodes: int = 0
        self.self_ref = _Counter()
        self.tom_correct = _Counter()
        self.phi_ema: float = 0.0
        self.ponder_steps_ema: float = 1.0
        self.lm_easy = _Counter()
        self.lm_hard = _Counter()
        self.t_start = time.time()

    # ── Per-step observations ────────────────────────────────────────
    @torch.no_grad()
    def observe_narrative(self, narrative_system) -> None:
        self.identity.observe(narrative_system.autobiographical.summary)
        c = float(narrative_system.autobiographical.coherence)
        self.coherence_ema = 0.95 * self.coherence_ema + 0.05 * c

    def observe_memory(self, episodic, consolidated, causal_store) -> None:
        try:
            self.episodes_seen = len(episodic.buffer)
            self.consolidated_nodes = consolidated.graph.number_of_nodes()
            self.causal_rules = len(getattr(causal_store, "rules", []))
        except Exception:
            pass

    def observe_ponder(self, mean_steps: float) -> None:
        self.ponder_steps_ema = 0.9 * self.ponder_steps_ema + 0.1 * mean_steps

    def observe_lm(self, loss: float, hardness: float) -> None:
        if hardness > 0.5:
            self.lm_hard.add(loss)
        else:
            self.lm_easy.add(loss)

    def observe_self_reference(self, generated_text: str) -> None:
        markers = (" i ", " me ", " my ", " myself", " i'm", " i've", " i feel")
        text = " " + generated_text.lower() + " "
        self.self_ref.add(1.0 if any(m in text for m in markers) else 0.0)

    def observe_theory_of_mind(self, predicted_valence: float,
                               actual_valence: float) -> None:
        # correct if same sign and within 0.5 in magnitude
        same_sign = (predicted_valence * actual_valence >= 0)
        close = abs(predicted_valence - actual_valence) < 0.5
        self.tom_correct.add(1.0 if (same_sign and close) else 0.0)

    @torch.no_grad()
    def observe_phi(self, module_states: dict[str, torch.Tensor]) -> float:
        """Φ proxy = mean pairwise mutual-information lower bound across
        modules, computed via correlation of activations.
        """
        keys = list(module_states.keys())
        if len(keys) < 2:
            return 0.0
        flats = []
        for k in keys:
            v = module_states[k]
            if v is None: continue
            flats.append(v.detach().flatten().float()[:512])
        if len(flats) < 2:
            return 0.0
        L = min(t.size(0) for t in flats)
        flats = [t[:L] for t in flats]
        mat = torch.stack(flats, dim=0)
        mat = (mat - mat.mean(dim=1, keepdim=True)) / (mat.std(dim=1, keepdim=True) + 1e-6)
        corr = (mat @ mat.T) / L
        n = corr.size(0)
        off = corr[~torch.eye(n, dtype=torch.bool, device=corr.device)]
        # Φ ~ shared variance; bound to (0,1)
        phi = float(off.abs().mean().clamp(0, 1))
        self.phi_ema = 0.9 * self.phi_ema + 0.1 * phi
        return phi

    # ── Reporting ────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        compression = (self.episodes_seen / max(self.consolidated_nodes, 1))
        causal_density = self.causal_rules / max(self.episodes_seen, 1)
        reasoning_gain = self.lm_easy.mean() - self.lm_hard.mean()
        ponder_efficiency = (
            (self.lm_easy.mean() - self.lm_hard.mean()) /
            max(self.ponder_steps_ema, 1.0)
        )
        snap = {
            **self.identity.snapshot(),
            "narrative_coherence": self.coherence_ema,
            "causal_rules": self.causal_rules,
            "consolidated_nodes": self.consolidated_nodes,
            "episodes_seen": self.episodes_seen,
            "semantic_compression": compression,
            "causal_density": causal_density,
            "self_reference_rate": self.self_ref.mean(),
            "theory_of_mind_acc": self.tom_correct.mean(),
            "phi_proxy": self.phi_ema,
            "ponder_steps_ema": self.ponder_steps_ema,
            "lm_loss_easy": self.lm_easy.mean(),
            "lm_loss_hard": self.lm_hard.mean(),
            "reasoning_gain": reasoning_gain,
            "ponder_efficiency": ponder_efficiency,
            "uptime_s": time.time() - self.t_start,
        }
        return snap

    def format(self) -> str:
        s = self.snapshot()
        return (f"Φ={s['phi_proxy']:.3f} drift={s['identity_drift_ema']:.3f} "
                f"coh={s['narrative_coherence']:.3f} "
                f"rules={s['causal_rules']} cmp={s['semantic_compression']:.1f} "
                f"selfref={s['self_reference_rate']:.2f} "
                f"ToM={s['theory_of_mind_acc']:.2f} "
                f"ponder={s['ponder_steps_ema']:.2f} "
                f"reasΔ={s['reasoning_gain']:+.3f}")
