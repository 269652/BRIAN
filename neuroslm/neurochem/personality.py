"""PersonalityVector and per-entity Bayesian trust → neurotransmitter coupling.

The PersonalityVector P is a low-dimensional, slowly-evolving state that:

  1. Distils long-term character traits (curiosity, agreeableness, vigilance,
     patience, hedonic tone) — these change on the scale of many thousand
     interactions, not per-step.
  2. Couples additively to the homeostatic baseline targets of selected
     neurotransmitters: a curious P → higher dopamine target, a patient
     P → higher serotonin target, a vigilant P → higher norepinephrine
     baseline.
  3. Carries a per-entity submodule: every tracked interactant has their
     own Bayesian Trust score, updated by interaction-outcome valence.
     The trust score biases NT release whenever the corresponding entity
     is active in the working set (ToM module signals current focus).

The vector is *distinct from episodic memory* — losing memory does not
reset personality, and personality changes only through repeated
high-confidence valence-tagged interactions, not from single events.

Math:

  Personality update (per consolidation step, NOT per forward):
    P ← (1 − η_P) · P + η_P · clamp(personality_drive, −1, 1)
  with η_P = 0.005 (≈ 200 consolidations to halve a trait).

  Bayesian trust per entity (Beta(α, β) on positive vs negative outcomes):
    α_e ← α_e + 0.5·(1 + v)    where v is the valence ∈ [−1, 1]
    β_e ← β_e + 0.5·(1 − v)
    trust(e) = α_e / (α_e + β_e)  ∈ [0, 1], confidence = (α_e + β_e)

  NT baseline bias from active entity:
    if entity e is in the working set with attention w_e:
      Δb_DA   = w_e · k_DA   · (trust(e) − 0.5)
      Δb_5HT  = w_e · k_5HT  · (trust(e) − 0.5)
      Δb_NE   = w_e · k_NE   · (0.5 − trust(e))   # untrusted → vigilant
    The Δb are added to the transmitters' learnable bias parameter.

This module is gated by the maturation phase: trust updates and
personality drift only happen *after* awakening (so random-init
interactions don't burn into character).
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn as nn


# Personality dimensions in fixed order (so checkpoints round-trip cleanly)
PERSONALITY_DIMS = (
    "curiosity",      # high → DA target raised, ACh raised
    "agreeableness",  # high → trust-bias amplified, 5HT raised
    "vigilance",      # high → NE baseline raised
    "patience",       # high → 5HT raised, GABA tone raised
    "hedonic_tone",   # high → DA + 5HT raised (general positive affect)
)
N_PERSONALITY = len(PERSONALITY_DIMS)
PERSONALITY_IDX = {n: i for i, n in enumerate(PERSONALITY_DIMS)}


# NT bias coefficients per personality dim — small numbers, layered on top
# of homeostasis's own bias clamp [-0.5, 0.5].
_NT_BIAS_FROM_PERSONALITY: dict[str, dict[str, float]] = {
    # personality_dim:  {NT_name: coefficient}
    "curiosity":     {"DA": +0.08, "ACh": +0.06, "NE": +0.02},
    "agreeableness": {"5HT": +0.06, "GABA": +0.04},
    "vigilance":     {"NE": +0.10, "ACh": +0.03, "GABA": -0.02},
    "patience":      {"5HT": +0.08, "GABA": +0.05, "DA": -0.02},
    "hedonic_tone":  {"DA": +0.07, "5HT": +0.05},
}


@dataclass
class EntityTrust:
    """Beta-Bernoulli posterior for one entity's trust."""
    entity_id: str
    alpha: float = 1.0      # positive-outcome pseudocount
    beta:  float = 1.0      # negative-outcome pseudocount
    last_interaction: float = field(default_factory=time.time)
    n_observations: int = 0

    def trust(self) -> float:
        """Posterior mean: α / (α + β)."""
        return float(self.alpha / (self.alpha + self.beta + 1e-9))

    def confidence(self) -> float:
        """Pseudocount mass — how strongly we believe the trust value."""
        return float(self.alpha + self.beta - 2.0)   # subtract the prior

    def update(self, valence: float) -> None:
        """valence ∈ [−1, 1]. Positive → α↑, negative → β↑."""
        v = max(-1.0, min(1.0, float(valence)))
        self.alpha += 0.5 * (1.0 + v)
        self.beta  += 0.5 * (1.0 - v)
        self.last_interaction = time.time()
        self.n_observations += 1


class PersonalityVector(nn.Module):
    """Holds the slowly-evolving Personality (P), the per-entity Trust map,
    and the NT-baseline biasing logic.

    Sits next to TransmitterSystem; the brain calls
    `personality.apply_bias(transmitters, active_entities)` once per
    consolidation step (NOT per forward pass).
    """

    # NT order must match transmitters.NT_NAMES
    _NT_NAMES = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")
    _NT_INDEX = {n: i for i, n in enumerate(_NT_NAMES)}

    def __init__(self,
                 eta_personality: float = 0.005,
                 entity_trust_prior: float = 1.0,
                 enable: bool = True):
        super().__init__()
        # Personality is a buffer (no gradients — updated by control loop)
        self.register_buffer(
            "P",
            torch.zeros(N_PERSONALITY, dtype=torch.float32))
        self.eta = eta_personality
        self.entity_trust_prior = entity_trust_prior
        self.enable = enable
        self.entities: Dict[str, EntityTrust] = {}
        # Awakening guard — set True by brain.py after maturation transition
        self._awakened = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def set_awakened(self, awakened: bool) -> None:
        self._awakened = bool(awakened)

    # ── trust API ────────────────────────────────────────────────────────────

    def observe_interaction(self, entity_id: str, valence: float) -> None:
        """Record an interaction outcome for an entity.

        Only updates state once the model has awakened — otherwise the
        valence is random-init noise and would burn into the trust prior.
        """
        if not self.enable or not self._awakened:
            return
        if not entity_id:
            return
        ent = self.entities.get(entity_id)
        if ent is None:
            ent = EntityTrust(entity_id=entity_id,
                              alpha=self.entity_trust_prior,
                              beta=self.entity_trust_prior)
            self.entities[entity_id] = ent
        ent.update(valence)

    def trust(self, entity_id: str) -> float:
        """Posterior trust mean ∈ [0, 1]; defaults to 0.5 for unknown
        entities (max-entropy prior)."""
        ent = self.entities.get(entity_id)
        return ent.trust() if ent is not None else 0.5

    def confidence(self, entity_id: str) -> float:
        ent = self.entities.get(entity_id)
        return ent.confidence() if ent is not None else 0.0

    # ── personality drift ────────────────────────────────────────────────────

    @torch.no_grad()
    def drift(self, drive: Dict[str, float]) -> None:
        """Drift P toward a sparse drive vector (one consolidation step).

        drive: e.g. {"curiosity": +0.3, "vigilance": -0.1}. Missing dims
        get drive=0 → toward zero (regression to neutral).
        """
        if not self.enable or not self._awakened:
            return
        d_vec = torch.zeros(N_PERSONALITY, dtype=self.P.dtype, device=self.P.device)
        for k, v in drive.items():
            i = PERSONALITY_IDX.get(k)
            if i is not None:
                d_vec[i] = max(-1.0, min(1.0, float(v)))
        self.P.mul_(1.0 - self.eta).add_(d_vec, alpha=self.eta)
        self.P.clamp_(-1.0, 1.0)

    # ── NT bias application ──────────────────────────────────────────────────

    @torch.no_grad()
    def apply_bias(self,
                   transmitters,            # neurochem.transmitters.TransmitterSystem
                   active_entities: Optional[Dict[str, float]] = None) -> None:
        """Apply personality-driven and entity-driven NT bias additively.

        active_entities: {entity_id: attention_weight_in_[0,1]} — usually
        the ToM module's current entity focus.

        Bias is applied to `transmitters.bias` (the learned per-NT bias
        already used by Homeostasis). We add a *small* personality-derived
        contribution and a per-entity trust contribution. The total bias
        remains clamped to [−0.5, 0.5] inside homeostasis.
        """
        if not self.enable or not self._awakened:
            return
        bias = transmitters.bias.data           # (N_NT,)
        device = bias.device
        delta = torch.zeros_like(bias)

        # Personality contribution
        P = self.P.to(device=device, dtype=bias.dtype)
        for dim_name, nt_coeffs in _NT_BIAS_FROM_PERSONALITY.items():
            di = PERSONALITY_IDX[dim_name]
            for nt_name, k in nt_coeffs.items():
                ni = self._NT_INDEX.get(nt_name)
                if ni is None:
                    continue
                delta[ni] += k * float(P[di].item())

        # Entity trust contribution
        if active_entities:
            for eid, w in active_entities.items():
                t = self.trust(eid) - 0.5          # ∈ [−0.5, +0.5]
                # Trusted entity → DA + 5HT bump
                # Untrusted entity → NE bump (vigilance)
                ww = max(0.0, min(1.0, float(w)))
                if "DA"  in self._NT_INDEX: delta[self._NT_INDEX["DA"]]  += 0.10 * ww * t
                if "5HT" in self._NT_INDEX: delta[self._NT_INDEX["5HT"]] += 0.06 * ww * t
                if "NE"  in self._NT_INDEX: delta[self._NT_INDEX["NE"]]  += 0.10 * ww * (-t)

        # Apply additively and clamp (Homeostasis also clamps, but be safe)
        bias.add_(delta).clamp_(-0.5, 0.5)

    # ── serialisation ────────────────────────────────────────────────────────

    def state_dict_summary(self) -> dict:
        return {
            "P": {n: float(self.P[i].item()) for i, n in enumerate(PERSONALITY_DIMS)},
            "n_entities": len(self.entities),
            "entities": {
                eid: {"trust": e.trust(), "confidence": e.confidence(),
                      "n_obs": e.n_observations}
                for eid, e in self.entities.items()
            },
            "awakened": self._awakened,
        }

    def save_state(self) -> dict:
        return {
            "P": self.P.detach().cpu().tolist(),
            "awakened": self._awakened,
            "entities": {
                eid: {"alpha": e.alpha, "beta": e.beta,
                      "last_interaction": e.last_interaction,
                      "n_observations": e.n_observations}
                for eid, e in self.entities.items()
            },
        }

    def load_state(self, state: dict) -> None:
        P = state.get("P")
        if P is not None:
            t = torch.tensor(P, dtype=self.P.dtype, device=self.P.device)
            if t.numel() == self.P.numel():
                self.P.copy_(t)
        self._awakened = bool(state.get("awakened", False))
        self.entities.clear()
        for eid, e in state.get("entities", {}).items():
            self.entities[eid] = EntityTrust(
                entity_id=eid,
                alpha=float(e["alpha"]),
                beta=float(e["beta"]),
                last_interaction=float(e.get("last_interaction", time.time())),
                n_observations=int(e.get("n_observations", 0)),
            )
