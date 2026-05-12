"""Qualia State Module — latent phenomenal-experience embedding.

Computes a *qualia embedding* that represents the subjective experiential
state of the agent at each cognitive tick (~10 Hz).  The qualia vector is
driven by:

  1. **Survival imperatives** — threat level, metabolic/homeostatic needs
  2. **Neurotransmitter levels** — DA (pleasure/motivation), NE (arousal/fear),
     5HT (calm/contentment), ACh (focus/curiosity), eCB (relaxation),
     GABA (inhibition), Glu (excitation)
  3. **Floating thought valence** — the semantic content of current thought
     (threatening → NE, calming → 5HT, rewarding → DA)
  4. **Self-model state** — agent's model of its own condition

The qualia embedding modulates the floating thought via a gated residual,
creating an oscillating thought-feeling loop analogous to human conscious
experience at ~10 Hz (alpha-band).

The module also produces **thought→NT feedback**: the content of thought
itself drives neurotransmitter release (e.g., ruminating on danger → NE,
imagining safety → 5HT), closing the loop between cognition and affect.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualiaState(nn.Module):
    """Latent qualia embedding — the agent's subjective experience state."""

    def __init__(self, d_sem: int, n_nt: int, d_qualia: int | None = None):
        super().__init__()
        self.d_sem = d_sem
        self.n_nt = n_nt
        self.d_qualia = d_qualia or d_sem

        # --- Inputs → qualia embedding ---
        # threat (1) + NT vector (n_nt) + thought valence (1) + self-state (d_sem)
        in_dim = 1 + n_nt + 1 + d_sem
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.d_qualia),
            nn.GELU(),
            nn.Linear(self.d_qualia, self.d_qualia),
            nn.Tanh(),  # bounded qualia space
        )

        # --- Qualia → thought modulation (gated residual) ---
        self.thought_gate = nn.Sequential(
            nn.Linear(self.d_qualia + d_sem, d_sem),
            nn.Sigmoid(),
        )
        self.thought_proj = nn.Linear(self.d_qualia, d_sem, bias=False)
        nn.init.zeros_(self.thought_proj.weight)  # start as identity on thought

        # --- Thought → NT feedback ---
        # Maps current thought embedding to NT release demands
        # This closes the cognition↔affect loop:
        #   threatening thought → NE release
        #   calming thought → 5HT release
        #   rewarding thought → DA release
        #   curious thought → ACh release
        self.thought_to_nt = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, n_nt),
            nn.Sigmoid(),  # NT demands in [0, 1]
        )

        # --- Valence detector ---
        # Classifies current thought as threatening/neutral/calming
        # Output: scalar in [-1, +1] (negative = threatening, positive = calming)
        self.valence_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 4),
            nn.GELU(),
            nn.Linear(d_sem // 4, 1),
            nn.Tanh(),
        )

        # Running EMA of qualia for smooth oscillation
        self.register_buffer("ema_qualia", torch.zeros(self.d_qualia))
        self.alpha = 0.3  # EMA smoothing (produces ~10 Hz oscillation feel)

        # ── Latent Qualia Manifold (Q) — homeostatic warp ────────────────
        # The Q-manifold biases the GWS broadcast mean by a direction in
        # d_sem space that scales with homeostatic deficit. When any
        # homeostat is critically low (< 0.2), the aversive direction
        # dominates: incoming sensory data is reinterpreted in the
        # "threatening / urgent" subspace. Healthy state produces only a
        # mild appetitive bias.
        #
        # Aversive direction is initialised at a much larger magnitude
        # than the appetitive direction so the warp behaviour is
        # asymmetric from step 0: starvation → strong warp; satiety →
        # gentle bias. Both directions are learnable.
        self.aversive_direction = nn.Parameter(
            torch.randn(d_sem) * 0.5)
        self.appetitive_direction = nn.Parameter(
            torch.randn(d_sem) * 0.05)
        # Per-homeostat (energy, hydration, integrity) gain into the warp.
        self.homeostatic_gain = nn.Parameter(
            torch.tensor([1.5, 1.0, 2.0], dtype=torch.float32))
        # Critical threshold below which the aversive component dominates.
        self.aversive_threshold = 0.20
        # Multiplier on the appetitive (surplus) contribution to keep it
        # subordinate to the aversive (deficit) contribution.
        self.appetitive_scale = 0.04
        # Last-computed warp magnitude (used by brain to gate κ_neg vesicle)
        self.register_buffer("last_aversive_pressure", torch.zeros(1))

    def forward(self, floating_thought: torch.Tensor,
                nt_vec: torch.Tensor,
                threat: torch.Tensor,
                z_self: torch.Tensor) -> dict:
        """
        Args:
            floating_thought: (B, d_sem) current thought embedding
            nt_vec: (B, n_nt) neurotransmitter levels
            threat: (B,) threat scalar from subconscious critic
            z_self: (B, d_sem) self-model state

        Returns dict with:
            qualia: (B, d_qualia) the qualia embedding
            modulated_thought: (B, d_sem) thought after qualia modulation
            thought_nt_demand: (B, n_nt) NT release demands from thought content
            thought_valence: (B,) how threatening/calming the current thought is
        """
        B = floating_thought.size(0)

        # 1) Compute thought valence
        thought_valence = self.valence_head(floating_thought).squeeze(-1)  # (B,)

        # 2) Assemble qualia input
        x = torch.cat([
            threat.unsqueeze(-1),           # (B, 1)
            nt_vec,                         # (B, n_nt)
            thought_valence.unsqueeze(-1),  # (B, 1)
            z_self,                         # (B, d_sem)
        ], dim=-1)

        # 3) Encode qualia
        qualia = self.encoder(x)  # (B, d_qualia)

        # 4) Smooth qualia with EMA for oscillatory dynamics
        with torch.no_grad():
            self.ema_qualia = (
                (1 - self.alpha) * self.ema_qualia
                + self.alpha * qualia.detach().mean(0)
            )

        # 5) Modulate floating thought with qualia (gated residual)
        gate_input = torch.cat([qualia, floating_thought], dim=-1)
        gate = self.thought_gate(gate_input)  # (B, d_sem)
        qualia_bias = self.thought_proj(qualia)  # (B, d_sem)
        modulated = floating_thought + gate * qualia_bias  # (B, d_sem)

        # 6) Thought → NT feedback
        thought_nt_demand = self.thought_to_nt(floating_thought)  # (B, n_nt)

        return {
            "qualia": qualia,
            "modulated_thought": modulated,
            "thought_nt_demand": thought_nt_demand,
            "thought_valence": thought_valence,
        }

    # ── Homeostatic warp of the GWS broadcast ────────────────────────────────

    def warp_broadcast(self,
                        broadcast: torch.Tensor,
                        homeostatic: torch.Tensor | list[float] | None = None,
                        ) -> torch.Tensor:
        """Apply a homeostatic-pressure warp to a GWS broadcast vector.

        broadcast:   (B, d_sem) — the mean GWS slot output for this pass
        homeostatic: (B, 3) or list/Tensor of [energy, hydration, integrity]
                     ∈ [0, 1]^3. If None, returns broadcast unchanged.

        When any homeostatic variable drops below `aversive_threshold`
        the aversive direction is added to the broadcast with magnitude
        proportional to the deficit. When all three are healthy, the
        appetitive direction nudges the broadcast slightly toward an
        exploratory bias. Net effect: low energy → all incoming sensory
        data is reinterpreted in the aversive subspace ("threatening" /
        "urgent"); high energy → the same data feels safe / interesting.

        The aversive pressure scalar is stored on `last_aversive_pressure`
        and read by the brain to gate κ_neg vesicle emission.
        """
        if homeostatic is None:
            return broadcast
        if not isinstance(homeostatic, torch.Tensor):
            homeostatic = torch.as_tensor(
                homeostatic, dtype=broadcast.dtype, device=broadcast.device)
        if homeostatic.dim() == 1:
            homeostatic = homeostatic.unsqueeze(0)
        # Deficit per channel (clamped to be non-negative)
        deficit = (self.aversive_threshold - homeostatic).clamp(min=0.0)
        # Weighted by per-homeostat gain
        gain = self.homeostatic_gain.to(
            device=homeostatic.device, dtype=homeostatic.dtype)
        weighted_deficit = (deficit * gain).sum(dim=-1, keepdim=True)   # (B, 1)
        # Surplus (when above threshold) contributes to appetitive bias
        surplus = (homeostatic - self.aversive_threshold).clamp(min=0.0)
        weighted_surplus = (surplus * gain).sum(dim=-1, keepdim=True)   # (B, 1)

        # Warp direction (aversive dominates over appetitive — see __init__)
        av = self.aversive_direction.to(dtype=broadcast.dtype).unsqueeze(0)
        ap = self.appetitive_direction.to(dtype=broadcast.dtype).unsqueeze(0)
        warped = (broadcast
                  + weighted_deficit * av
                  + self.appetitive_scale * weighted_surplus * ap)

        # Persist the aversive pressure scalar (mean over batch)
        with torch.no_grad():
            self.last_aversive_pressure.copy_(
                weighted_deficit.detach().mean().to(self.last_aversive_pressure.dtype))
        return warped

    def aversive_pressure(self) -> float:
        """Last computed aversive-pressure scalar — used by the brain to
        decide whether to emit a κ_neg vesicle this pass."""
        return float(self.last_aversive_pressure.item())
