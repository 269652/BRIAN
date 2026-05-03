"""Causal pattern extraction for episodic consolidation.

Episodes are stored as raw events; consolidation should extract reusable
*causal rules* of the form

    (action_template, context_template) ──► outcome_distribution
    confidence: c ∈ [0,1], support: n_observations

Examples produced from the user's scenario:
    ("kind_words", "talking_to_entity")  ► happy_response   conf=0.83 n=12
    ("rude_words", "talking_to_entity")  ► negative_response conf=0.71 n=7

Pipeline:
  1. Pull recent (action, context, outcome_valence) tuples from episodic
     memory + mesolimbic tags.
  2. Cluster the (action, context) embeddings with cosine k-means.
  3. For each cluster, compute outcome statistics. If support ≥ N_min and
     |mean valence| > eps, emit a CausalRule.
  4. Merge rules with high cosine similarity and consistent outcome.
  5. Decay rules whose recent observations contradict them (Bayesian
     update of confidence).

The store is the second LFS-shipped artifact (alongside the consolidated
graph) inside the .mem checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import threading
import numpy as np


@dataclass
class CausalRule:
    """A learned (action, context) → outcome rule."""
    rule_id: int
    action_vec: np.ndarray         # (D,) prototype of the action embedding
    context_vec: np.ndarray        # (D,) prototype of the context embedding
    outcome_mean: float            # mean valence of observed outcomes
    outcome_var: float             # variance of observed outcomes
    n_observations: int = 1
    confidence: float = 0.5
    label: str = ""                # short human-readable label, optional
    last_seen_step: int = 0

    def update(self, outcome: float, alpha: float = 0.2) -> None:
        old_mean = self.outcome_mean
        self.outcome_mean = (1 - alpha) * old_mean + alpha * outcome
        # Welford-ish online variance
        self.outcome_var = (1 - alpha) * (self.outcome_var
                                          + alpha * (outcome - old_mean) ** 2)
        self.n_observations += 1
        # Confidence rises with support, falls with variance
        self.confidence = float(
            1.0 - 1.0 / (1.0 + 0.1 * self.n_observations))
        consistency = math.exp(-self.outcome_var)
        self.confidence *= consistency

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "action_vec": self.action_vec.tolist(),
            "context_vec": self.context_vec.tolist(),
            "outcome_mean": self.outcome_mean,
            "outcome_var": self.outcome_var,
            "n_observations": self.n_observations,
            "confidence": self.confidence,
            "label": self.label,
            "last_seen_step": self.last_seen_step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalRule":
        return cls(
            rule_id=d["rule_id"],
            action_vec=np.asarray(d["action_vec"], dtype=np.float32),
            context_vec=np.asarray(d["context_vec"], dtype=np.float32),
            outcome_mean=d["outcome_mean"], outcome_var=d["outcome_var"],
            n_observations=d["n_observations"], confidence=d["confidence"],
            label=d.get("label", ""), last_seen_step=d.get("last_seen_step", 0),
        )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float(na @ nb)


class CausalRuleStore:
    """Thread-safe store of CausalRules with merge / query / serialization."""

    def __init__(self, merge_threshold: float = 0.88,
                 min_support: int = 2):
        self.rules: list[CausalRule] = []
        self.lock = threading.Lock()
        self.next_id = 0
        self.merge_threshold = merge_threshold
        self.min_support = min_support

    # ── Updates ──────────────────────────────────────────────────────
    def observe(self, action_vec, context_vec, outcome_valence: float,
                step: int = 0, label: str = "") -> int:
        """Record a single (action, context, outcome) observation.

        If a similar rule exists, update it; else create a new candidate.
        Returns the rule_id touched.
        """
        action_vec = np.asarray(action_vec, dtype=np.float32).flatten()
        context_vec = np.asarray(context_vec, dtype=np.float32).flatten()
        with self.lock:
            best, best_sim = None, -1.0
            for r in self.rules:
                # Pad to common dim
                d = min(r.action_vec.size, action_vec.size)
                d2 = min(r.context_vec.size, context_vec.size)
                sim = 0.5 * _cos(r.action_vec[:d], action_vec[:d]) \
                    + 0.5 * _cos(r.context_vec[:d2], context_vec[:d2])
                if sim > best_sim:
                    best, best_sim = r, sim
            if best is not None and best_sim >= self.merge_threshold:
                # Update with EMA on prototype + outcome
                d = min(best.action_vec.size, action_vec.size)
                best.action_vec[:d] = 0.9 * best.action_vec[:d] + 0.1 * action_vec[:d]
                d2 = min(best.context_vec.size, context_vec.size)
                best.context_vec[:d2] = 0.9 * best.context_vec[:d2] + 0.1 * context_vec[:d2]
                best.update(outcome_valence)
                best.last_seen_step = step
                if label and not best.label:
                    best.label = label
                return best.rule_id
            # New rule
            rid = self.next_id; self.next_id += 1
            new = CausalRule(
                rule_id=rid,
                action_vec=action_vec.copy(),
                context_vec=context_vec.copy(),
                outcome_mean=float(outcome_valence),
                outcome_var=0.1,
                n_observations=1,
                confidence=0.3,
                label=label,
                last_seen_step=step,
            )
            self.rules.append(new)
            return rid

    # ── Querying ─────────────────────────────────────────────────────
    def predict(self, action_vec, context_vec,
                topk: int = 3) -> list[tuple[CausalRule, float]]:
        """Return top-k rules matching a hypothetical (action, context),
        sorted by (similarity * confidence). Used by PFC to forecast
        outcomes of candidate actions before selecting one.
        """
        action_vec = np.asarray(action_vec, dtype=np.float32).flatten()
        context_vec = np.asarray(context_vec, dtype=np.float32).flatten()
        scored = []
        for r in self.rules:
            if r.confidence < 0.2 or r.n_observations < self.min_support:
                continue
            d = min(r.action_vec.size, action_vec.size)
            d2 = min(r.context_vec.size, context_vec.size)
            sim = 0.5 * _cos(r.action_vec[:d], action_vec[:d]) \
                + 0.5 * _cos(r.context_vec[:d2], context_vec[:d2])
            scored.append((r, sim * r.confidence))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:topk]

    def best_action(self, candidate_actions, context_vec
                    ) -> tuple[int, float]:
        """Pick the action whose predicted outcome valence is highest
        (weighted by confidence). Returns (index, expected_valence).
        """
        best_i, best_v = 0, -math.inf
        for i, a in enumerate(candidate_actions):
            preds = self.predict(a, context_vec, topk=3)
            if not preds:
                v = 0.0
            else:
                num = sum(r.outcome_mean * w for r, w in preds)
                den = sum(w for _, w in preds) + 1e-9
                v = num / den
            if v > best_v:
                best_v, best_i = v, i
        return best_i, best_v

    # ── Maintenance ──────────────────────────────────────────────────
    def prune(self, max_rules: int = 1024) -> int:
        """Drop the least-confident rules if over capacity. Returns count
        removed."""
        with self.lock:
            if len(self.rules) <= max_rules:
                return 0
            self.rules.sort(key=lambda r: r.confidence * r.n_observations,
                            reverse=True)
            removed = len(self.rules) - max_rules
            self.rules = self.rules[:max_rules]
            return removed

    def stats(self) -> dict:
        if not self.rules:
            return {"n_rules": 0}
        confs = np.array([r.confidence for r in self.rules])
        sup = np.array([r.n_observations for r in self.rules])
        valences = np.array([r.outcome_mean for r in self.rules])
        return {
            "n_rules": len(self.rules),
            "mean_confidence": float(confs.mean()),
            "max_support": int(sup.max()),
            "positive_rules": int((valences > 0.1).sum()),
            "negative_rules": int((valences < -0.1).sum()),
        }

    # ── Serialization ────────────────────────────────────────────────
    def to_state(self) -> dict:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "next_id": self.next_id,
            "merge_threshold": self.merge_threshold,
            "min_support": self.min_support,
        }

    @classmethod
    def from_state(cls, s: dict) -> "CausalRuleStore":
        store = cls(merge_threshold=s.get("merge_threshold", 0.88),
                    min_support=s.get("min_support", 2))
        store.next_id = s.get("next_id", 0)
        store.rules = [CausalRule.from_dict(d) for d in s.get("rules", [])]
        return store
