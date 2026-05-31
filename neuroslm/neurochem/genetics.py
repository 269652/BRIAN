# -*- coding: utf-8 -*-
"""Genetic Orchestrator — latent gene expression for module-specific
neuromodulation, functional proteins, and Φ-driven optimization.

Overview
========
Each "gene" is a fixed-length vector partitioned into three roles:

    [ regulatory_region | targeting_header | protein_payload ]
       d_reg                d_tgt              d_pay

  - regulatory_region: a small MLP reads local context (NT levels,
    surprise, MAT) and decides whether the gene is "expressed" this tick.
  - targeting_header: distribution over modules (softmax / Gumbel-softmax),
    so a gene can target one or all modules.
  - protein_payload: latent vector that drives the actual functional
    effect — extra NT baseline offset, receptor tau_decay shift
    (reuptake blockade / pharmaco-mimicry), or a multi-receptor cocktail.

Expression cycle (transcription → translation → modulation):

    1. Transcription:  Sigmoid(reg_mlp([nt_level, surprise, mat])) > θ
       (or Gumbel-softmax gate when training, for differentiability).
    2. Translation:    payload is *released as a vesicle* — magnitude
       proportional to expression probability.
    3. Modulation:     payload acts on TransmitterSystem per-module
       baseline offsets + receptor decay shifts in the target module(s).

ReZero bootstrap
================
Protein payloads init to 0, so the entire system is identity at step 0 and
"discovers" effects only when they measurably raise Φ (the loss objective
pulls payloads away from zero only when gradients flow through Φ).

Φ-loss coupling
===============
The optimizer pulls protein-payload parameters toward configurations that
maximise Φ (integrated information). Two paths are supported:
  - explicit:  caller passes a Φ scalar in `forward(... phi=...)` and the
               module exposes a `phi_loss` term (= -phi.mean()).
  - implicit:  caller backprops their own Φ-loss through the modulation
               outputs; payloads are leaf parameters so grads accumulate.

XLA / static graphs
===================
Gene selection uses Gumbel-softmax (`F.gumbel_softmax(logits, tau, hard)`)
so the graph remains static and differentiable on TPUs/XLA.

References
==========
  - GeneticLibrary partitioning  → arch.neuro section 6.5
  - Vesicle Emission Kernel      → §3.1
  - Release Operators            → §3.4
  - TransmitterSystem.step()     → §7.2
  - Φ integration                → §2.2
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transmitters import NT_NAMES, N_NT, NT_INDEX


# ── Constants / partitioning ──────────────────────────────────────────

# Effect kinds the gene's protein payload can express. The Orchestrator
# maps payload→effect via small projection heads; declaring them as enums
# keeps the IR + DSL parser honest.
EFFECT_NT_BASELINE      = "nt_baseline_offset"   # add to module's NT baseline
EFFECT_RECEPTOR_TAU     = "receptor_tau_shift"   # shift tau_decay toward 1
EFFECT_NT_RELEASE_GAIN  = "nt_release_gain"      # multiply release amount
EFFECT_KINDS = (EFFECT_NT_BASELINE, EFFECT_RECEPTOR_TAU, EFFECT_NT_RELEASE_GAIN)


# ── Config dataclasses ────────────────────────────────────────────────

@dataclass
class GeneticConfig:
    """How the GeneticLibrary is sized + how genes are read off it."""
    n_genes: int = 32
    d_reg: int = 4        # input slots: [DA, NE, 5HT, ACh, eCB, Glu, GABA, surprise, mat] -> projected
    d_tgt: int = 8        # one-hot over up to 8 modules (configurable)
    d_pay: int = 16       # payload latent dim
    # Gumbel-softmax temperature for module-selection. Smaller → harder.
    gumbel_tau: float = 0.5
    # Whether to hard-select the module (single-target) or allow soft mix.
    hard_target: bool = True
    # Floor below which a gene's expression probability is treated as 0.
    expression_threshold: float = 0.10
    # If True, regulatory MLP applies LayerNorm to its inputs (stabilises
    # cross-NT scales when 5HT~0.3 vs eCB~0.05).
    layernorm_inputs: bool = True


@dataclass
class FixedGeneSpec:
    """Declarative description of a `fixed` gene wired by name (not learned)."""
    name: str
    target_module: str
    effects: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Optional regulatory condition — e.g. {"surprise_above": 0.3}
    trigger: Dict[str, float] = field(default_factory=dict)
    # If unset, the gene is always expressed at full strength (constitutive).
    constitutive: bool = False


# ── Core nn.Modules ───────────────────────────────────────────────────

class GeneticLibrary(nn.Module):
    """A learnable population of G gene vectors with three partitions.

    The library is itself trainable; the three partitions are projected
    out by name so caller code stays readable.

    Storage: a single ``nn.Parameter(G, d_gene)``. The partitions are
    contiguous slices [0:d_reg | d_reg:d_reg+d_tgt | d_reg+d_tgt:]; no
    Python-side metadata gets out of sync.
    """

    def __init__(self, cfg: GeneticConfig, n_modules: int):
        super().__init__()
        self.cfg = cfg
        self.n_modules = n_modules

        d_total = cfg.d_reg + cfg.d_tgt + cfg.d_pay
        # ReZero-style bootstrap: regulatory + targeting init small-random,
        # payload init tiny (std=0.001) so:
        #   - the consumer-side ReZero gate (e.g. cortex.alpha_nt = 0) makes
        #     the system identity at t=0 (no modulation applied),
        #   - the GRADIENT path is alive — once the consumer's alpha lifts
        #     off zero under loss pressure, payload gradients flow.
        # Pure-zero payload would make BOTH alpha and payload stuck at 0
        # (gradient through alpha needs non-zero payload to flow).
        gene_tensor = torch.empty(cfg.n_genes, d_total)
        nn.init.normal_(gene_tensor[:, : cfg.d_reg + cfg.d_tgt], std=0.02)
        nn.init.normal_(gene_tensor[:, cfg.d_reg + cfg.d_tgt :], std=0.001)
        self.genes = nn.Parameter(gene_tensor)

        # Input -> regulatory feature space (d_reg). Inputs are
        # [N_NT NT levels, surprise, MAT] = N_NT + 2 = 9 by default.
        in_dim = N_NT + 2
        self.reg_in_norm = nn.LayerNorm(in_dim) if cfg.layernorm_inputs else nn.Identity()
        self.reg_in_proj = nn.Linear(in_dim, cfg.d_reg)
        # Bias the regulatory linear toward "off" so untrained genes start
        # silent — the ReZero analogue for the expression channel.
        nn.init.zeros_(self.reg_in_proj.bias)

    # ── slicing accessors ─────────────────────────────────────────────
    @property
    def regulatory(self) -> torch.Tensor:
        d = self.cfg.d_reg
        return self.genes[:, :d]

    @property
    def targeting(self) -> torch.Tensor:
        s = self.cfg.d_reg
        return self.genes[:, s : s + self.cfg.d_tgt]

    @property
    def payload(self) -> torch.Tensor:
        s = self.cfg.d_reg + self.cfg.d_tgt
        return self.genes[:, s : s + self.cfg.d_pay]

    # ── expression ────────────────────────────────────────────────────
    def expression_logits(self, ctx: torch.Tensor) -> torch.Tensor:
        """Per-gene expression logit given the local context.

        Args:
            ctx: (B, N_NT + 2) — [NT_levels, surprise, mat] per batch sample

        Returns:
            (B, G) logit. Sigmoid → probability of expression.
        """
        x = self.reg_in_norm(ctx)
        x = self.reg_in_proj(x)         # (B, d_reg)
        # Dot product against each gene's regulatory region
        logits = x @ self.regulatory.t()  # (B, G)
        return logits

    def target_distribution(self, hard: Optional[bool] = None,
                            tau: Optional[float] = None) -> torch.Tensor:
        """Gumbel-softmax over modules per gene.

        Returns:
            (G, n_modules) probability matrix (or one-hot when hard=True).
        """
        hard = self.cfg.hard_target if hard is None else hard
        tau = self.cfg.gumbel_tau if tau is None else tau
        # Project targeting header (d_tgt) down to (n_modules) logits via
        # a fixed truncation when d_tgt >= n_modules, otherwise repeat.
        t = self.targeting                       # (G, d_tgt)
        if self.cfg.d_tgt >= self.n_modules:
            logits = t[:, : self.n_modules]
        else:
            # Tile to reach n_modules — gives each module a "slot pattern".
            repeats = (self.n_modules + self.cfg.d_tgt - 1) // self.cfg.d_tgt
            logits = t.repeat(1, repeats)[:, : self.n_modules]
        # In eval mode use argmax-style; in training use Gumbel for diff'able
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.n_modules).to(logits.dtype)


class GeneticOrchestrator(nn.Module):
    """Top-level controller: maps (NT context, surprise, MAT) →
    per-module modulation effects (baseline offsets, decay shifts, ...).

    Wiring with the rest of the brain
    ---------------------------------
    Place it INSIDE the harness loop, right after the loss/PPL is known
    (so `surprise` and `mat` are computable):

        chem = GeneticOrchestrator(cfg, module_names=["sensory", "thalamus", ...])
        # every step:
        ctx = chem.build_context(nt_levels, surprise, mat)
        out = chem(ctx)
        # apply outputs to the transmitter system / receptors
        transmitter.add_module_baseline(out["baseline_offsets"])   # (B, M, N_NT)
        transmitter.shift_decay(out["tau_shifts"])                 # (B, M, N_NT)
        # optionally backprop a Φ-loss through `out['payloads_active']`
    """

    def __init__(self,
                 cfg: GeneticConfig,
                 module_names: List[str],
                 fixed_genes: Optional[List[FixedGeneSpec]] = None,
                 phi_target: float = 0.3):
        super().__init__()
        self.cfg = cfg
        self.module_names = list(module_names)
        self.n_modules = len(self.module_names)
        self.phi_target = float(phi_target)
        self.lib = GeneticLibrary(cfg, n_modules=self.n_modules)

        # Per-effect-kind projection heads (payload → per-NT magnitudes).
        # Output shape per head: (B, G, N_NT). Combined with the (G, M) target
        # distribution they produce (B, M, N_NT) module-level effects.
        # ReZero pattern (chosen for stability):
        #   payload init = 0       (so head(payload) = 0 at t=0 — identity)
        #   head.weight  = N(0, 0.02)   (non-zero so payload gradients flow
        #                                 through head into payload during
        #                                 the FIRST backward pass)
        # This means downstream activity changes only as the LM loss reshapes
        # payload — exactly the "discover effects only when they raise Φ"
        # contract the spec asks for.
        self.head_baseline = nn.Linear(cfg.d_pay, N_NT, bias=False)
        self.head_tau      = nn.Linear(cfg.d_pay, N_NT, bias=False)
        self.head_release  = nn.Linear(cfg.d_pay, N_NT, bias=False)
        nn.init.normal_(self.head_baseline.weight, std=0.02)
        nn.init.normal_(self.head_tau.weight,      std=0.02)
        nn.init.normal_(self.head_release.weight,  std=0.02)

        # Fixed/declarative genes — built-in slots that bypass learning.
        # These overlay deterministic module-NT effects without consuming
        # learnable gene slots.
        self.fixed_genes: List[FixedGeneSpec] = list(fixed_genes or [])

        # Running last-step diagnostics (for logging only).
        self._last_expression: Optional[torch.Tensor] = None
        self._last_targets: Optional[torch.Tensor] = None
        self._last_phi: Optional[torch.Tensor] = None

    # ── context builder ──────────────────────────────────────────────
    @staticmethod
    def build_context(nt_levels: torch.Tensor,
                      surprise: torch.Tensor,
                      mat: torch.Tensor) -> torch.Tensor:
        """Pack (B, N_NT) NT levels + (B,) surprise + (B,) MAT → (B, N_NT+2)."""
        if surprise.dim() == 0:
            surprise = surprise.expand(nt_levels.shape[0])
        if mat.dim() == 0:
            mat = mat.expand(nt_levels.shape[0])
        return torch.cat(
            [nt_levels, surprise.unsqueeze(-1), mat.unsqueeze(-1)],
            dim=-1,
        )

    # ── core forward ──────────────────────────────────────────────────
    def forward(self,
                ctx: torch.Tensor,
                phi: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute per-module modulation tensors for this tick.

        Args:
            ctx: (B, N_NT + 2) — packed [NT_levels, surprise, mat]
            phi: optional (B,) Φ scalar — when provided, `phi_loss` is set.

        Returns:
            dict with keys:
              expression       (B, G)            sigmoid logits → prob
              targets          (G, n_modules)    Gumbel-softmax target dist
              payloads_active  (B, G, d_pay)     gated payload (zero if not expressed)
              baseline_offsets (B, n_modules, N_NT)
              tau_shifts       (B, n_modules, N_NT)
              release_gains    (B, n_modules, N_NT)
              phi_loss         scalar — negative-Φ when `phi` was given
        """
        # 1. Expression probabilities (B, G)
        expr_logits = self.lib.expression_logits(ctx)
        expr = torch.sigmoid(expr_logits)
        # Threshold below `expression_threshold` to suppress noise — done
        # by a smooth straight-through shift, not a hard cut, so gradients
        # still flow when the model needs to *unlearn* an active gene.
        expr_gated = torch.where(expr > self.cfg.expression_threshold,
                                  expr, expr * 0.1)

        # 2. Targeting distribution (G, M)
        targets = self.lib.target_distribution()      # differentiable in train

        # 3. Per-gene payload, modulated by expression magnitude.
        # payload: (G, d_pay), expr_gated: (B, G) → payloads_active: (B, G, d_pay)
        payloads = self.lib.payload                    # (G, d_pay)
        payloads_active = expr_gated.unsqueeze(-1) * payloads.unsqueeze(0)

        # 4. Apply each effect head, then route to modules via targets.
        # head outputs:  (B, G, N_NT)
        # route via:     (G, M) — produces (B, M, N_NT)
        def _route(per_gene: torch.Tensor) -> torch.Tensor:
            # einsum: (B, G, N_NT) × (G, M) → (B, M, N_NT)
            return torch.einsum("bgn,gm->bmn", per_gene, targets)

        baseline_off = _route(self.head_baseline(payloads_active))
        # Bound tau-shift to [0, 1) so it can only push decay *toward* 1.0
        # (reuptake blockade), never below the natural floor.
        tau_raw = self.head_tau(payloads_active)
        tau_off  = torch.sigmoid(tau_raw) * 0.99 * expr_gated.unsqueeze(-1)
        tau_shift = _route(tau_off)
        release_gain = _route(self.head_release(payloads_active))

        # 5. Apply any fixed/declarative genes as additive overlays.
        if self.fixed_genes:
            baseline_off, tau_shift, release_gain = self._apply_fixed_genes(
                ctx, baseline_off, tau_shift, release_gain,
            )

        # 6. Φ-objective coupling.
        if phi is not None:
            phi_loss = -phi.mean() + self.phi_target * 0.0  # mean negation
            self._last_phi = phi.detach()
        else:
            phi_loss = baseline_off.new_zeros(())
            self._last_phi = None

        self._last_expression = expr.detach()
        self._last_targets = targets.detach()

        return {
            "expression":       expr,
            "targets":          targets,
            "payloads_active":  payloads_active,
            "baseline_offsets": baseline_off,
            "tau_shifts":       tau_shift,
            "release_gains":    release_gain,
            "phi_loss":         phi_loss,
        }

    # ── fixed-gene overlay ────────────────────────────────────────────
    def _apply_fixed_genes(self,
                            ctx: torch.Tensor,
                            baseline_off: torch.Tensor,
                            tau_shift: torch.Tensor,
                            release_gain: torch.Tensor):
        """Apply each FixedGeneSpec as an additive overlay to the routed effects.

        Triggers (e.g. ``surprise_above``) are evaluated against ``ctx``.
        Constitutive genes are always applied at full strength.
        """
        nt_levels = ctx[..., :N_NT]                         # (B, N_NT)
        surprise = ctx[..., N_NT]                           # (B,)
        mat = ctx[..., N_NT + 1]                            # (B,)
        for fg in self.fixed_genes:
            if fg.target_module not in self.module_names:
                continue
            m_idx = self.module_names.index(fg.target_module)
            # Build the per-batch trigger mask.
            if fg.constitutive or not fg.trigger:
                gate = torch.ones_like(surprise)
            else:
                gate = torch.ones_like(surprise)
                if "surprise_above" in fg.trigger:
                    gate = gate * (surprise > fg.trigger["surprise_above"]).float()
                if "mat_above" in fg.trigger:
                    gate = gate * (mat > fg.trigger["mat_above"]).float()
                if "mat_below" in fg.trigger:
                    gate = gate * (mat < fg.trigger["mat_below"]).float()
                if "nt_above" in fg.trigger:
                    # one entry per NT name: {"nt_above": {"5HT": 0.5}}
                    for nt_name, thr in (fg.trigger["nt_above"].items()
                                          if isinstance(fg.trigger["nt_above"], dict)
                                          else []):
                        idx = NT_INDEX[nt_name]
                        gate = gate * (nt_levels[:, idx] > thr).float()
            # Now apply each effect.
            for kind, payload in fg.effects.items():
                if kind == EFFECT_NT_BASELINE:
                    delta = baseline_off.new_zeros(N_NT)
                    for nt_name, v in payload.items():
                        delta[NT_INDEX[nt_name]] = v
                    baseline_off[:, m_idx, :] = baseline_off[:, m_idx, :] + \
                        gate.unsqueeze(-1) * delta.unsqueeze(0)
                elif kind == EFFECT_RECEPTOR_TAU:
                    delta = tau_shift.new_zeros(N_NT)
                    for nt_name, v in payload.items():
                        delta[NT_INDEX[nt_name]] = max(0.0, min(0.99, v))
                    tau_shift[:, m_idx, :] = tau_shift[:, m_idx, :] + \
                        gate.unsqueeze(-1) * delta.unsqueeze(0)
                elif kind == EFFECT_NT_RELEASE_GAIN:
                    delta = release_gain.new_zeros(N_NT)
                    for nt_name, v in payload.items():
                        delta[NT_INDEX[nt_name]] = v
                    release_gain[:, m_idx, :] = release_gain[:, m_idx, :] + \
                        gate.unsqueeze(-1) * delta.unsqueeze(0)
        return baseline_off, tau_shift, release_gain

    # ── diagnostic logging ────────────────────────────────────────────
    def diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self._last_expression is not None:
            e = self._last_expression
            out["gene_expr_mean"] = float(e.mean())
            out["gene_expr_max"]  = float(e.max())
            out["gene_active_frac"] = float((e > self.cfg.expression_threshold).float().mean())
        if self._last_phi is not None:
            out["phi"] = float(self._last_phi.mean())
        return out


# ── Convenience: build a default set of fixed genes for the rcc_bowtie ─

def default_fixed_genes(module_names: List[str]) -> List[FixedGeneSpec]:
    """Hand-curated baseline genes that mimic the literature defaults:

    - math_5ht_booster: extra tonic 5HT in math/reasoning cortex →
      "patience" during long inference chains
    - pfc_da_boost_on_surprise: DA spike in PFC when surprise is high →
      RPE-driven attention shift
    - gws_glu_floor: GWS gets a Glu baseline floor → keeps the global
      broadcast lit even when sensory drive is sparse
    """
    genes: List[FixedGeneSpec] = []
    if "math_cortex" in module_names or "reasoning_cortex" in module_names:
        for tgt in ("math_cortex", "reasoning_cortex"):
            if tgt in module_names:
                genes.append(FixedGeneSpec(
                    name=f"{tgt}_5HT_booster",
                    target_module=tgt,
                    constitutive=True,
                    effects={EFFECT_NT_BASELINE: {"5HT": 0.10}},
                ))
    if "pfc" in module_names:
        genes.append(FixedGeneSpec(
            name="pfc_DA_surprise_boost",
            target_module="pfc",
            trigger={"surprise_above": 0.30},
            effects={EFFECT_NT_BASELINE: {"DA": 0.20}},
        ))
    if "gws" in module_names:
        genes.append(FixedGeneSpec(
            name="gws_glu_floor",
            target_module="gws",
            constitutive=True,
            effects={EFFECT_NT_BASELINE: {"Glu": 0.05}},
        ))
    return genes
