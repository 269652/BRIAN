# -*- coding: utf-8 -*-
"""Active Inference / Free Energy Principle for NeuroSLM.

Canonical Friston (2017) hierarchical message passing:

  Superficial pyramidal cells (error units ε^l):
      ε^l = Π^l ⊙ (x^l − g^l(μ^{l+1}))
      — precision-weighted prediction errors propagated bottom-up

  Deep pyramidal cells (state units μ^l):
      μ^l = recognition(x^l, ε^l)
      — beliefs integrate sensory evidence + prediction errors

  Generative model (top-down predictions):
      g^l(μ^l) = pred of level l−1 given level l beliefs

  Free energy:
      F = Σ_l ||ε^l||^2  (accuracy) + complexity prior

Two-pass forward:
  Pass 1 (bottom-up):  compute initial μ^l with no top-down prior
  Pass 2 (top-down):   generate predictions downward, recompute ε^l

References:
  Friston et al. (2017) "Active Inference: A Process Theory" Neural Computation
  Bastos et al. (2012) "Canonical Microcircuits for Predictive Coding" Neuron
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class PredictiveLayer(nn.Module):
    """Canonical Friston predictive coding layer.

    Superficial pyramidal cells (error units):
        ε^l = Π^l ⊙ (x^l − prior_pred^l)
        Precision Π = exp(log_precision) is learned per feature.

    Deep pyramidal cells (state units):
        μ^l = recognition(x^l, ε^l)
        Integrates sensory evidence with precision-weighted prediction error.

    Generative model (top-down):
        pred^{l-1} = generative(μ^l)
        Zero-initialised so the network starts with uninformative priors.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Precision weights for superficial pyramidal error units
        self.log_precision = nn.Parameter(torch.zeros(d_in))

        # Deep pyramidal: recognition model μ^l = f(x, ε)
        self.recognition = nn.Sequential(
            nn.Linear(d_in * 2, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

        # Generative model: g^l(μ^l) → prediction of level l-1
        self.generative = nn.Linear(d_out, d_in, bias=False)
        nn.init.zeros_(self.generative.weight)  # uninformative prior at init

    def forward(self, x: torch.Tensor,
                prior_pred: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:          (B, d_in)  — sensory input or lower-layer belief
        prior_pred: (B, d_in)  — top-down generative prediction; None → zero prior

        Returns:
          posterior:  (B, d_out) — deep pyramidal state (belief μ^l)
          error:      (B, d_in)  — precision-weighted prediction error ε^l
          pred_down:  (B, d_in)  — generative prediction of level l−1
        """
        if prior_pred is None:
            prior_pred = torch.zeros_like(x)

        # Superficial pyramidal: ε^l = Π^l ⊙ (x − pred)
        precision = torch.exp(self.log_precision)   # (d_in,) — inverse variance
        error = precision * (x - prior_pred)

        # Deep pyramidal: μ^l integrates evidence + prediction error
        posterior = self.recognition(torch.cat([x, error], dim=-1))

        # Generative: what does this level predict about the level below?
        pred_down = self.generative(posterior)

        return posterior, error, pred_down


class HierarchicalPredictiveProcessor(nn.Module):
    """Canonical two-pass hierarchical predictive processor.

    Pass 1 — Bottom-up (recognition):
        For each layer l bottom to top: compute μ^l from x^l with no prior.
    Pass 2 — Top-down (generation):
        For each layer l top to bottom: generate pred^{l-1} = g^l(μ^l),
        recompute ε^l with the top-down prior, producing final error signals.

    Free energy = Σ_l ||ε^l||^2 + 0.05 × Σ_l ||μ^l||^2
    """

    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 3):
        super().__init__()
        dims = [d_in] + [d_hidden] * n_layers
        self.layers = nn.ModuleList([
            PredictiveLayer(dims[i], dims[i + 1])
            for i in range(n_layers)
        ])
        self.output_dim = d_hidden

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, d_in)

        Returns:
          posterior:    (B, d_hidden) — top-layer belief state μ^{L}
          free_energy:  ()            — scalar F = accuracy + complexity
        """
        n = len(self.layers)

        # --- Pass 1: Bottom-up — no top-down prior ---
        posteriors: list[torch.Tensor] = []
        current = x
        for layer in self.layers:
            post, _, _ = layer(current, prior_pred=None)
            posteriors.append(post)
            current = post

        # --- Pass 2: Top-down — recompute errors with generative predictions ---
        errors: list[torch.Tensor] = []
        td_pred: Optional[torch.Tensor] = None   # top layer has no prior from above
        for i in range(n - 1, -1, -1):
            inp = posteriors[i - 1] if i > 0 else x
            _, err, pred_down = self.layers[i](inp, prior_pred=td_pred)
            errors.insert(0, err)
            td_pred = pred_down   # this layer's prediction becomes prior for level below

        # --- Free energy ---
        accuracy   = sum(err.pow(2).mean() for err in errors)
        complexity = sum(0.05 * post.pow(2).mean() for post in posteriors)
        free_energy = accuracy + complexity

        return posteriors[-1], free_energy


class PrecisionWeightedAttention(nn.Module):
    """Multi-head attention with learned per-head precision weighting.

    Standard attention: softmax(QK^T / √d) V
    Precision attention: softmax(QK^T * precision / √d) V

    Where precision is a learned scalar per head, representing the
    inverse variance of the predictions at that head's feature scale.

    High precision → sharp, confident attention (exploitation)
    Low precision  → diffuse, exploratory attention (exploration)

    The precision values are modulated by NT state at inference:
      ACh ↑ → precision ↑ (focused attention)
      NE ↑  → precision ↑ on threat-related features
      DA ↑  → modulates what's attended (reward-relevant)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head   = d_model // n_heads
        self.n_heads  = n_heads
        self.d_model  = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

        # Learned per-head precision (log-scale for positivity)
        self.log_precision = nn.Parameter(torch.zeros(n_heads))

        # NT modulation of precision (4 NTs: DA, NE, 5HT, ACh)
        self.nt_precision_mod = nn.Linear(4, n_heads)

    def forward(self, x: torch.Tensor,
                nt_levels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:         (B, T, d_model)
        nt_levels: (B, 4) or (4,) — [DA, NE, 5HT, ACh]
        mask:      (B, T, T)
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                   for t in qkv]

        # Base precision
        precision = torch.exp(self.log_precision)   # (n_heads,)

        # NT modulation
        if nt_levels is not None:
            nt = nt_levels.float()
            if nt.dim() == 1:
                nt = nt.unsqueeze(0).expand(B, -1)
            nt_mod = torch.sigmoid(self.nt_precision_mod(nt))   # (B, n_heads)
            precision = precision.unsqueeze(0) * (0.5 + nt_mod)  # (B, n_heads)
        else:
            precision = precision.unsqueeze(0).expand(B, -1)    # (B, n_heads)

        # Precision-scaled attention
        scale = (self.d_head ** -0.5) * precision   # (B, n_heads)
        attn  = torch.einsum("bhid,bhjd->bhij", q, k)   # (B, n_heads, T, T)
        attn  = attn * scale[:, :, None, None]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn  = F.softmax(attn, dim=-1)
        out   = torch.einsum("bhij,bhjd->bhid", attn, v)
        out   = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out(out)


class EpistemicValueEstimator(nn.Module):
    """Estimates epistemic value (information gain) of a potential action.

    Epistemic value = expected reduction in posterior uncertainty.
    High epistemic value → taking this action will teach us something.

    Implemented as: H[p(o)] - E_a[H[p(o|a)]]
    Approximated by: entropy of current beliefs - conditional entropy.

    Novel application: drives exploration in language generation by
    selecting tokens / actions that maximally reduce uncertainty about
    the world model's hidden state.
    """

    def __init__(self, d_sem: int, n_action_types: int = 14):
        super().__init__()
        self.info_gain = nn.Sequential(
            nn.Linear(d_sem + n_action_types, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, 1),
            nn.Softplus(),   # always positive
        )
        # Uncertainty estimate for current world state
        self.uncertainty = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, 1),
            nn.Softplus(),
        )

    def forward(self, world_state: torch.Tensor,
                action_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        world_state:  (B, d_sem)
        action_probs: (B, n_action_types)
        Returns:
          epistemic_value: (B,)  — information gain from this action
          uncertainty:     (B,)  — current world-state uncertainty
        """
        x = torch.cat([world_state, action_probs], dim=-1)
        ig = self.info_gain(x).squeeze(-1)
        unc = self.uncertainty(world_state).squeeze(-1)
        return ig, unc


class FreeEnergyProcessor(nn.Module):
    """Top-level Free Energy module combining all active inference components.

    Wraps:
      • HierarchicalPredictiveProcessor — layered prediction error propagation
      • EpistemicValueEstimator — information-gain-driven exploration
      • Free-energy auxiliary loss — improves sample efficiency

    How it integrates with brain.py:
      1. Receives GWS slots (semantic context) as input
      2. Computes hierarchical prediction errors
      3. Returns a refined representation (posterior beliefs)
      4. Returns free_energy loss for auxiliary training signal
      5. Returns epistemic_value for action selection biasing
      6. Returns uncertainty for novelty / NE modulation

    The free_energy loss is added to the total loss with weight w_fe (~0.1)
    This acts as an auxiliary predictive coding loss that regularises
    internal representations toward predictable structures.
    """

    def __init__(self, d_sem: int, n_layers: int = 3,
                 n_action_types: int = 14):
        super().__init__()
        self.predictive = HierarchicalPredictiveProcessor(
            d_in=d_sem, d_hidden=d_sem, n_layers=n_layers)
        self.epistemic = EpistemicValueEstimator(d_sem, n_action_types)
        # Pragmatic value: expected reward from beliefs
        self.pragmatic = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, 1),
        )

    def forward(self, gws_state: torch.Tensor,
                action_probs: Optional[torch.Tensor] = None,
                nt_levels: Optional[torch.Tensor] = None
               ) -> dict:
        """
        gws_state:    (B, d_sem)  — current global workspace state
        action_probs: (B, n_action_types) — current BG action probabilities
        nt_levels:    (B, 4)     — [DA, NE, 5HT, ACh]

        Returns dict with:
          posterior:        (B, d_sem) — refined belief state
          free_energy:      ()         — scalar auxiliary loss
          epistemic_value:  (B,)       — info gain from action
          pragmatic_value:  (B,)       — expected reward from action
          uncertainty:      (B,)       — world-state uncertainty
        """
        # Hierarchical predictive processing
        posterior, free_energy = self.predictive(gws_state)

        # Epistemic and pragmatic value
        if action_probs is None:
            B = gws_state.shape[0]
            action_probs = torch.ones(B, 14, device=gws_state.device) / 14

        epistemic_val, uncertainty = self.epistemic(posterior, action_probs)
        pragmatic_val = self.pragmatic(posterior).squeeze(-1)

        # NT-modulated balance between epistemic (explore) and pragmatic (exploit).
        # Neurobiological grounding:
        #   DA ↑  → pragmatic dominates (reward is available, exploit it)
        #   NE ↑  → epistemic dominates (novelty/uncertainty, explore)
        #   ACh ↑ → sharpen accuracy term of free energy (attend to details)
        fe_weight = torch.ones(1, device=gws_state.device)
        ep_weight = torch.ones(1, device=gws_state.device)   # epistemic scale
        pr_weight = torch.ones(1, device=gws_state.device)   # pragmatic scale

        if nt_levels is not None:
            nt = nt_levels.float()
            if nt.dim() == 2:
                nt = nt.mean(0)
            da  = nt[0].clamp(0, 1) if nt.size(0) > 0 else torch.tensor(0.5)
            ne  = nt[1].clamp(0, 1) if nt.size(0) > 1 else torch.tensor(0.5)
            ach = nt[3].clamp(0, 1) if nt.size(0) > 3 else torch.tensor(0.5)
            # ACh sharpens accuracy attention
            fe_weight = (0.5 + 0.5 * ach)
            # DA promotes pragmatic value; NE promotes epistemic value
            pr_weight = (0.5 + da)          # in [0.5, 1.5]
            ep_weight = (0.5 + ne)          # in [0.5, 1.5]

        # Combined value signal passed to BG: weighted sum of explore/exploit
        combined_value = ep_weight * epistemic_val + pr_weight * pragmatic_val

        return {
            "posterior":        posterior,
            "free_energy":      free_energy * fe_weight,
            "epistemic_value":  epistemic_val,
            "pragmatic_value":  pragmatic_val,
            "combined_value":   combined_value,   # → BG action selection
            "uncertainty":      uncertainty,
            "epistemic_weight": float(ep_weight.mean()),
            "pragmatic_weight": float(pr_weight.mean()),
        }
