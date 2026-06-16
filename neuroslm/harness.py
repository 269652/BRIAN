# -*- coding: utf-8 -*-
"""BRIANHarness — language-model training harness for DSL-compiled circuits.

The DSL gives you a bare `nn.Module` whose forward maps a per-population
input tensor to a dict of per-population outputs. To *train* this as a
language model you also need:

  * a vocabulary embedding (token IDs → d_sem vectors)
  * an LM head (d_sem → vocab logits)
  * sequence-aware forward (process a (batch, seq_len) of token IDs,
    return (batch, seq_len, vocab) logits)
  * a loss function (cross-entropy, optionally per-sample clipped)
  * optimizer + grad-accumulation + grad-clip plumbing
  * checkpoint save/load

`BRIANHarness` provides all of that, configured from a `TrainingConfig`
(read from arch.neuro's `training { ... }` block).

This is Phase A of the DSL→training port — the harness is feature-
complete for what's needed to train an SLM end-to-end with DSL-defined
architecture. Later phases (B–F per docs/dsl.md roadmap) port the
remaining Brain subsystems (vesicle pools, trophic system, sleep cycle,
maturity machinery, etc.) into both DSL constructs and harness hooks.
"""
from __future__ import annotations
import math
import os
from contextlib import nullcontext
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.training_config import TrainingConfig
from neuroslm.dsl.maturity import (
    MaturityTracker, AuxWeights, TotalLossConfig,
)
from neuroslm.dsl.bema_optimizer import BEMAController, BEMAConfig


# ── LR schedule (warmup + cosine) ──────────────────────────────────────
#
# Brain's training uses linear warmup over `warmup` steps to base_lr,
# then cosine decay to `min_lr_ratio * base_lr` over the remaining
# (total - warmup) steps. After `total`, clamps to min_lr (so crash-
# restarts past the budget don't oscillate).

def cosine_warmup_lr(step: int, base_lr: float, warmup: int, total: int,
                     min_lr_ratio: float = 0.1) -> float:
    """Standard warmup-then-cosine schedule used by Brain training."""
    if warmup > 0 and step < warmup:
        # Linear ramp from 0 → base_lr
        return base_lr * (step / warmup)
    # Cosine decay from base_lr → min_lr over [warmup, total]
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))  # clamp past total
    min_lr = base_lr * min_lr_ratio
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ── Stage 6 OOD push: μP scaling helpers ───────────────────────────────

def _mu_p_param_groups(model, base_lr: float, wd: float):
    """Build AdamW param groups with width-aware LR multipliers (μP).

    The rule (simplified Yang/Hu Tensor-Programs-IV):
      - "embedding-like" params (vocab × d_model, or 1-D γ/β): LR × 1
      - "hidden" params (d_model × d_model, etc.): LR × 1 / fan_in_ratio
        where fan_in_ratio = fan_in / base_width (typical base=256)
      - "output / lm_head" params: LR × 1 / d_model

    Effect: representation updates stay O(1) across widths so the same
    base_lr that worked at 51M continues to work at 1B+. At our current
    51M (d_model=384) with base_width=256 the multipliers are close to
    1, so this is a near no-op until we scale up.
    """
    base_width = 256
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Output head: scale by 1/d_model
        if "lm_head" in name:
            d_model = p.shape[-1] if p.dim() >= 2 else p.shape[0]
            mult = base_width / max(1, d_model)
        # 2-D hidden weights: scale by 1/fan_in_ratio
        elif p.dim() == 2:
            fan_in = p.shape[1]
            mult = base_width / max(1, fan_in)
        else:
            # 1-D (norms, biases) and embeddings: no scaling
            mult = 1.0
        groups.append({"params": [p], "lr": base_lr * mult,
                        "weight_decay": wd})
    return groups


def _llrd_param_groups(model, base_lr: float, wd: float, factor: float):
    """Build AdamW param groups with layer-wise LR decay (ULMFiT / DeBERTa).

    Per-block parameters get `lr_i = base_lr * factor^(depth - 1 - i)`
    so block 0 gets the smallest LR and the final block gets `base_lr`.
    Embedding / lm_head / norms get `base_lr` (no decay).

    Why: top layers learn quickly and tend to memorise; slowing them
    while keeping bottom layers fast preserves general representations
    and reduces train↔OOD generalisation gap by 20–30% in published
    runs (ULMFiT, DeBERTa, BERT). At factor=1.0 this is a no-op.

    Heuristic: a parameter is "in block i" if its name contains
    `blocks.{i}.` or `adapters.{i}.`. Everything else uses base_lr.
    """
    import re
    # Discover the number of blocks
    block_idx_re = re.compile(r"\.blocks\.(\d+)\.")
    max_idx = -1
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = block_idx_re.search(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    depth = max_idx + 1 if max_idx >= 0 else 1

    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = block_idx_re.search(name)
        if m:
            i = int(m.group(1))
            # Block 0 oldest → smallest LR; block depth-1 = base_lr
            mult = factor ** (depth - 1 - i)
        else:
            mult = 1.0
        groups.append({"params": [p], "lr": base_lr * mult, "weight_decay": wd})
    return groups


# ── H006: Capacity-Funneled Distillation (CFD) helpers ─────────────────
#
# These three free functions implement the three CFD stages described in
# docs/formal_framework.md §13 and the hypothesis card
# hypothesis/H006_capacity_funneled_distillation_implode.md. They are
# at module level (not on BRIANHarness) so they can be unit-tested
# independently of the harness state machine.
#
# Wiring into the training loop is conditional on
# MultiCortexConfig.cfd_enabled and lives inside
# BRIANHarness._cortex_fusion_aux_step below.


def cfd_topk_target(
    teacher_logits: torch.Tensor, K: int, T: float
) -> torch.Tensor:
    """Stage 1 — top-K rank-preserving sparsification.

    Replace the raw teacher softmax with a K-modes-plus-uniform-tail
    projection:

      * Keep the top-K teacher logits at their `softmax(teacher / T)`
        mass.
      * Spread the residual `1 - sum(top-K mass)` UNIFORMLY over the
        remaining V-K modes.

    This puts the imitation target inside the student's reachable
    softmax simplex whenever K ≤ student mode-resolution capacity, so
    the distillation KL has a reachable zero (not the asymptotic
    "always-positive" floor of naive Hinton).

    Args:
        teacher_logits: (..., V) raw teacher logits.
        K: number of top modes to keep (1 ≤ K ≤ V).
        T: temperature for the softmax. Same value as used by the
            downstream KL.

    Returns:
        target: (..., V) probability distribution. Sums to 1 along
        the last axis, exactly preserves the top-K masses of
        `softmax(teacher_logits / T)`, has uniform tail.
    """
    V = teacher_logits.size(-1)
    if K < 1 or K > V:
        raise ValueError(f"K must be in [1, {V}], got {K}")

    raw = F.softmax(teacher_logits / float(T), dim=-1)
    if K == V:
        # Tail is empty; nothing to redistribute.
        return raw

    topk_vals, topk_idx = teacher_logits.topk(K, dim=-1)
    # Mass on the top-K under the raw softmax (same indices).
    topk_mass = raw.gather(-1, topk_idx)
    # Residual mass to spread uniformly over the V-K tail.
    residual = (1.0 - topk_mass.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    tail_uniform = residual / float(V - K)

    # Build target: start with uniform tail value everywhere, then
    # overwrite the top-K positions with their raw mass.
    target = tail_uniform.expand_as(raw).clone()
    target.scatter_(-1, topk_idx, topk_mass)
    return target


def cfd_effective_temperature(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T_0: float,
    floor_multiplier: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """Stage 2 — entropy-matched temperature.

    Compute the per-batch effective temperature

        T_eff = T_0 · max(floor_multiplier, H(p_s) / H(p_t))

    where H is Shannon entropy of `softmax(. / T_0)` per row, averaged
    over the batch (under no_grad).

    When the student is at least as confident as the teacher
    (H(p_s) ≤ H(p_t)) the multiplier collapses to `floor_multiplier`
    (default 1.0, so T_eff = T_0). When the student is more uncertain,
    T_eff > T_0, softening the teacher's distribution to a level the
    student can plausibly match. This is the self-paced curriculum
    that reveals teacher detail as the student earns capacity.

    Args:
        student_logits: (..., V) student logits (any shape; entropy
            averaged over leading dims).
        teacher_logits: (..., V) teacher logits (same shape).
        T_0: base temperature.
        floor_multiplier: lower bound on the entropy ratio (default 1.0
            so T_eff ≥ T_0 always; set > 1.0 to enforce a minimum
            softening).
        eps: numerical guard against H(p_t) → 0.

    Returns:
        T_eff: scalar Python float (so it can be used as a multiplier
        in subsequent softmax calls without re-entering autograd).
    """
    with torch.no_grad():
        s_soft = F.softmax(student_logits / float(T_0), dim=-1)
        t_soft = F.softmax(teacher_logits / float(T_0), dim=-1)
        # Per-position entropy then mean over leading dims.
        log_s = F.log_softmax(student_logits / float(T_0), dim=-1)
        log_t = F.log_softmax(teacher_logits / float(T_0), dim=-1)
        H_s = -(s_soft * log_s).sum(dim=-1).mean().item()
        H_t = -(t_soft * log_t).sum(dim=-1).mean().item()
    ratio = H_s / max(H_t, eps)
    multiplier = max(float(floor_multiplier), ratio)
    return float(T_0) * multiplier


def cfd_grad_alignment_gate(
    distill_term: torch.Tensor,
    lm_logits: torch.Tensor,
    targets: torch.Tensor,
    lam_0: float,
) -> tuple[float, float]:
    """Stage 3 — gradient-alignment gate.

    Compute the cosine alignment between the distillation gradient and
    the LM-CE gradient measured at the pre-fusion logit tensor:

        g_align = cos(∇_{lm_logits} distill_term,
                      ∇_{lm_logits} CE(lm_logits, targets))

    Then the effective λ is

        λ_eff = λ_0 · (1 + g_align) / 2 ∈ [0, λ_0].

    By construction λ_eff = 0 when the teacher pulls AGAINST the LM
    objective (anti-aligned gradients) and λ_eff = λ_0 when fully
    aligned. This is the mechanical "no-harm floor" of the H006
    theorem (I).

    Args:
        distill_term: scalar tensor — the distillation KL term BEFORE
            multiplication by λ. Must have grad_fn pointing to
            `lm_logits` (or share a common subgraph with `lm_logits`).
        lm_logits: (B, T, V) — the PRE-FUSION student logits. Must be
            a non-leaf tensor (the grad-alignment is measured here).
        targets: (B, T) — token ids for the LM-CE gradient.
        lam_0: base λ value (the pre-Stage-3 scale).

    Returns:
        (lam_eff, g_align): lam_eff is the gated scalar to multiply
        the distill term by; g_align ∈ [-1, 1] is the cosine for
        telemetry.
    """
    V = lm_logits.size(-1)
    flat_t = targets.reshape(-1)
    lm_ce = F.cross_entropy(lm_logits.reshape(-1, V), flat_t)
    # retain_graph so the caller can still .backward() on the
    # downstream loss.
    g_l = torch.autograd.grad(
        lm_ce, lm_logits, retain_graph=True, create_graph=False
    )[0]
    g_d = torch.autograd.grad(
        distill_term, lm_logits, retain_graph=True, create_graph=False
    )[0]
    # Cosine over the full flattened logit-gradient field. Cheap (one
    # vector cosine) compared to the gradient computation itself.
    cos = F.cosine_similarity(
        g_l.reshape(1, -1), g_d.reshape(1, -1), dim=-1, eps=1e-8
    ).item()
    # Numerical guard — cosine_similarity can drift very slightly out
    # of [-1, 1] due to fp16/fp32 mixed precision.
    cos = max(-1.0, min(1.0, cos))
    lam_eff = float(lam_0) * 0.5 * (1.0 + cos)
    return lam_eff, cos


def cfd_topk_schedule(
    step: int, K_start: int, K_end: int, anneal_steps: int
) -> int:
    """Linear top-K anneal: K_start at step 0 → K_end at `anneal_steps`,
    then stays at K_end. Always returns an int ≥ 1.
    """
    if anneal_steps <= 0 or step >= anneal_steps:
        return int(K_end)
    if step <= 0:
        return int(K_start)
    progress = step / anneal_steps  # in (0, 1)
    K = K_start + (K_end - K_start) * progress
    return max(1, int(round(K)))


# ── CFDv2 — Generalisation-Funneled Distillation (GFD) helpers ─────────
#
# Two additional free functions extending CFD with the M2 (prior-residual
# sparsification) and M4 (pointwise-K via teacher PMI) mechanisms from
# docs/formal_framework.md §14. Plus a variable-K version of the Stage-1
# top-K projection that consumes the M4 K-per-position tensor.
#
# Design philosophy (see hypothesis/H006 v2 falsifier table):
#   * γ = 0 ⇒ M2 is a noop (full back-compat with CFDv1).
#   * Constant K ⇒ var-K top-K is bit-identical to scalar top-K.
# So enabling the v2 knobs at their defaults preserves H006's no-harm
# floor; the v2 mechanisms only *bias the funnel toward generalisation*,
# they cannot violate Stage-3's cosine-gate argument.


def cfd_prior_residual(
    teacher_logits: torch.Tensor,
    log_prior: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """M2 — prior-residual sparsification of the teacher distribution.

    Returns adjusted teacher logits such that the subsequent softmax
    yields the prior-residual distribution

        p_t^residual(v | c) ∝ p_t(v | c) / p_uni(v)^γ

    by linearity of log-softmax: subtract `γ · log p_uni(v)` from each
    teacher logit before the softmax normaliser.

    Args:
        teacher_logits: (..., V) raw teacher logits.
        log_prior: (V,) log-probabilities of the unigram prior over the
            shared vocabulary (must satisfy `logsumexp(log_prior) ≈ 0`,
            though we do not strictly enforce it — any constant shift
            cancels in the next softmax).
        gamma: residual strength in [0, 1]. 0 = identity (CFDv1 path),
            1 = full removal of unigram floor.

    Returns:
        adjusted: (..., V) logits with `−γ · log_prior` broadcast on
        the last axis. Pass into `cfd_topk_target` /
        `cfd_topk_target_var_k` (or any downstream Stage-1 op) exactly
        as if it were the raw teacher.
    """
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError(
            f"cfd_prior_residual: gamma must be in [0, 1], got {gamma}"
        )
    if log_prior.dim() != 1 or log_prior.size(0) != teacher_logits.size(-1):
        raise ValueError(
            f"cfd_prior_residual: log_prior shape "
            f"{tuple(log_prior.shape)} incompatible with teacher_logits "
            f"vocab dim {teacher_logits.size(-1)}"
        )
    if float(gamma) == 0.0:
        # Bit-identical noop branch — preserves back-compat numerically.
        return teacher_logits
    # log_prior broadcasts on the vocab axis. Cast to teacher dtype to
    # avoid spurious upcasts in mixed-precision training.
    return teacher_logits - float(gamma) * log_prior.to(
        teacher_logits.dtype
    ).to(teacher_logits.device)


def cfd_pointwise_k_from_pmi(
    teacher_logits: torch.Tensor,
    log_prior: torch.Tensor,
    K_min: int,
    K_max: int,
    scale: float = 2.0,
) -> torch.Tensor:
    """M4 — per-position top-K from the teacher's pointwise mutual
    information with the unigram prior.

    For each (batch, position) slot t, let v* = argmax_v p_t(v | c_t)
    be the teacher's top-1 token. Define

        PMI(t) := log p_t(v* | c_t) − log p_uni(v*)

    and the per-position K

        K(t) := clamp( K_max · exp( −max(PMI(t), 0) / scale ),
                       K_min, K_max )

    so K(t) ∈ [K_min, K_max], monotone-non-increasing in PMI(t):

      * high PMI (sharp contextual peak on a rare-prior token)
        ⇒ K close to K_min — concentrate the distillation signal,
      * low / zero PMI (uniform teacher or top-1 = a common token)
        ⇒ K close to K_max — broaden into a soft regulariser.

    Args:
        teacher_logits: (..., V).
        log_prior: (V,) unigram log-probabilities.
        K_min: lower bound on per-position K (must be ≥ 1).
        K_max: upper bound on per-position K (must be ≥ K_min, ≤ V).
        scale: PMI decay scale (nats). Default 2.0 gives a smooth
            interpolation across the typical PMI range of a language
            model (∼0–5 nats for high-entropy contexts).

    Returns:
        K_per_pos: integer tensor of shape `teacher_logits.shape[:-1]`,
        each entry in [K_min, K_max]. dtype = torch.long.
    """
    V = teacher_logits.size(-1)
    if K_min < 1:
        raise ValueError(f"cfd_pointwise_k_from_pmi: K_min ≥ 1, got {K_min}")
    if K_max < K_min:
        raise ValueError(
            f"cfd_pointwise_k_from_pmi: K_max ≥ K_min, got "
            f"K_min={K_min}, K_max={K_max}"
        )
    if K_max > V:
        raise ValueError(
            f"cfd_pointwise_k_from_pmi: K_max ≤ V, got "
            f"K_max={K_max}, V={V}"
        )
    if log_prior.dim() != 1 or log_prior.size(0) != V:
        raise ValueError(
            f"cfd_pointwise_k_from_pmi: log_prior shape "
            f"{tuple(log_prior.shape)} incompatible with V={V}"
        )

    with torch.no_grad():
        log_p_t = F.log_softmax(teacher_logits, dim=-1)
        # top-1 value & index per position
        top1_logp, top1_idx = log_p_t.max(dim=-1)
        # Gather prior log-prob at the top-1 token
        prior_logp = log_prior.to(teacher_logits.device).gather(
            0, top1_idx.reshape(-1)
        ).reshape(top1_idx.shape)
        pmi = (top1_logp - prior_logp).clamp_min(0.0)
        # K(t) = K_max · exp(−PMI / scale), then clamp to [K_min, K_max]
        K_real = float(K_max) * torch.exp(-pmi / float(scale))
        K_clamped = K_real.clamp(min=float(K_min), max=float(K_max))
        # Round to nearest int; dtype long so downstream gather/scatter
        # treats it as indices.
        return K_clamped.round().long()


def cfd_topk_target_var_k(
    teacher_logits: torch.Tensor,
    K_per_pos: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """Variable-K analogue of `cfd_topk_target` — each (batch,
    position) slot uses its own K from `K_per_pos`.

    For each slot t with K(t) modes:
      * keep the top-K(t) teacher softmax masses at their raw values,
      * spread the residual `1 − Σ top-K(t) mass` uniformly over the
        remaining V − K(t) tail.

    Implementation: build a full top-V-sorted softmax once, then for
    each slot mask the entries with rank ≥ K(t) and replace them with
    `(1 − retained_mass) / (V − K(t))`. Vectorised — O(B·T·V) time,
    no Python loop over positions.

    Args:
        teacher_logits: (..., V).
        K_per_pos: integer tensor with shape `teacher_logits.shape[:-1]`,
            each entry in [1, V]. Typically produced by
            `cfd_pointwise_k_from_pmi`.
        T: temperature.

    Returns:
        target: (..., V) probability distribution. Each per-position
        row sums to 1, has non-negative entries, exactly preserves the
        top-K(t) softmax masses, has uniform tail of size V − K(t).
    """
    V = teacher_logits.size(-1)
    if K_per_pos.shape != teacher_logits.shape[:-1]:
        raise ValueError(
            f"cfd_topk_target_var_k: K_per_pos shape "
            f"{tuple(K_per_pos.shape)} does not match leading dims of "
            f"teacher_logits {tuple(teacher_logits.shape[:-1])}"
        )
    if K_per_pos.min().item() < 1 or K_per_pos.max().item() > V:
        raise ValueError(
            f"cfd_topk_target_var_k: K_per_pos out of range, got "
            f"min={K_per_pos.min().item()}, max={K_per_pos.max().item()}, "
            f"V={V}"
        )

    raw = F.softmax(teacher_logits / float(T), dim=-1)
    # Sort descending → use rank position to compare against K(t).
    sorted_vals, sorted_idx = raw.sort(dim=-1, descending=True)
    # rank: (..., V) with values 0..V-1 along last axis
    rank = torch.arange(V, device=raw.device).expand_as(sorted_vals)
    # K_expand: (..., 1) → broadcasts on V axis
    K_expand = K_per_pos.unsqueeze(-1)
    keep_mask_sorted = rank < K_expand  # (..., V) bool
    # Retained per-position mass (sum of the kept ranks)
    retained = (sorted_vals * keep_mask_sorted).sum(dim=-1, keepdim=True)
    # Tail size per position = V − K(t)
    tail_size = (V - K_expand).clamp_min(1).to(raw.dtype)
    residual = (1.0 - retained).clamp_min(0.0)
    tail_uniform = residual / tail_size  # (..., 1) — broadcasts on V

    # Build target IN SORTED ORDER first, then un-sort with scatter.
    target_sorted = torch.where(
        keep_mask_sorted, sorted_vals, tail_uniform.expand_as(sorted_vals)
    )
    # Un-sort: place target_sorted[..., r] back at the original index
    # sorted_idx[..., r].
    target = torch.empty_like(raw)
    target.scatter_(-1, sorted_idx, target_sorted)
    return target


class BRIANHarness(nn.Module):
    """Wrap a DSL-compiled circuit for end-to-end LM training.

    Args:
        circuit: an `nn.Module` produced by `CodeGenerator.compile_to_module()`
                 — its forward must accept `(sensory_input, nt_levels=None)`
                 and return a dict of population outputs (must include a
                 sink population whose output gets routed to the LM head;
                 by default we pick `motor` if present, else the last
                 population's output).
        vocab_size: tokenizer vocab size
        d_sem: semantic dimension (must match circuit's d_sem)
        training_config: optional TrainingConfig; defaults to vanilla
        sink_population: name of the population whose output goes to the
                         LM head. Default 'motor' (rcc_bowtie convention).
    """

    def __init__(self,
                 circuit: nn.Module,
                 vocab_size: int,
                 d_sem: int,
                 training_config: Optional[TrainingConfig] = None,
                 sink_population: str = "motor"):
        super().__init__()
        self.circuit = circuit
        self.vocab_size = vocab_size
        self.d_sem = d_sem
        self.training_config = training_config or TrainingConfig()
        self.sink_population = sink_population

        # If a full ids→logits language model is supplied, the harness
        # delegates its forward to it (no separate embedding/head). This is
        # the N5 path: train the exact-match DSL transformer LM directly.
        self.language_model = None

        # Token embedding: (vocab_size, d_sem)
        self.embedding = nn.Embedding(vocab_size, d_sem)
        # LM head: (d_sem, vocab_size). Linear stores weight as (out=vocab, in=d_sem)
        self.lm_head = nn.Linear(d_sem, vocab_size, bias=False)

        # Optimizer is lazy-built on first train_step (so users can pass
        # parameters through to schedulers etc. before training).
        self._optimizer: Optional[torch.optim.Optimizer] = None
        # Grad accumulation counter
        self._accum_step = 0
        # Global step counter (drives the LR schedule)
        self._global_step = 0

        # LR schedule (set via set_schedule; None → constant LR)
        self._sched_warmup: Optional[int] = None
        self._sched_total: Optional[int] = None
        self._sched_min_ratio: float = 0.1

        # Mixed precision (set via enable_mixed_precision)
        self._amp_dtype: Optional[torch.dtype] = None
        self._grad_scaler: Optional[torch.cuda.amp.GradScaler] = None

        # Maturity + total-loss formula (matches Brain bit-for-bit).
        # `total_loss_config` carries the aux weight table; `maturity`
        # holds the rise-fast/fall-slow EMA fed by every train_step.
        self.total_loss_config: TotalLossConfig = TotalLossConfig()
        self.maturity: MaturityTracker = MaturityTracker()
        self._last_lm_loss_value: float = self.maturity.l_random
        # Stage 3+4 OOD-push controllers. _bema is lazily built in
        # _ensure_optimizer so it sees the constructed optimizer.
        self._bema: Optional[BEMAController] = None
        self._last_bema_info: Dict = {}
        self._nemori_skipped: int = 0
        # Genetics + runtime metric registry (see §6.5).
        # Lazily built in _ensure_genetics() so it sees the actual model.
        self._genetics_orch = None
        self._transmitter_sys = None
        self._genetics_module_names: list = []
        # Surprise EMA for the regulatory context fed into genes
        self._surprise_ema: float = 0.0
        # Cached last orchestrator modulation — reused on skipped steps
        # (genetics.update_every > 1) so the cortex's NT modulation
        # stays installed without rebuilding the graph every step.
        self._last_orch_modulation = None
        # Runtime metric registry — values queryable from anywhere in the
        # control flow node (other modules, schedulers, gene triggers).
        # Updated once per compute_loss step.
        self._metrics: Dict[str, float] = {}

        # Evolution heatmap hook — see neuroslm/evolution/harness_hook.py.
        # Attached lazily via :py:meth:`attach_heatmap_hook` so importing
        # the harness doesn't pull in the evolution subsystem until a run
        # actually wants incremental heat collection.
        self._heatmap_hook = None

        # ── Cortex fusion: distillation + NT-gated α state ──
        # Slot A (KL-distillation) and slot C (NT-mediated gating) are
        # both controlled by `training_config.multi_cortex` flags; the
        # state lives on the harness so it persists across train_step
        # calls without re-entering compute_loss with stale EMAs.
        #
        # Stash of pre-fusion logits — set inside forward() when the
        # fusion path is active, consumed in compute_loss() by the KL
        # term. None when fusion is off (back-compat).
        self._last_pre_fusion_lm_logits: Optional[torch.Tensor] = None
        self._last_pre_fusion_cortex_logits: Optional[torch.Tensor] = None
        # EMAs of cortex-only and trunk-only CE losses (both nats).
        # Used by:
        #   * `_distillation_lambda` for the λ_t ramp schedule
        #   * `_update_cortex_inhibition` for the NT-gating drive signal
        self._lm_loss_ema: float = 0.0
        self._cortex_loss_ema: float = 0.0
        # Inhibition level ∈ [0, 1]: 0 = cortex fully active (default at
        # init), 1 = cortex fully gated off. Driven up as trunk surpasses
        # cortex; saturates near 1 once the gap is large for long enough.
        self._cortex_inhibition_level: float = 0.0

        # ── PR2: OOD-intervention controller ──
        # Reads cfg.regularization (set by parse_training_config). When all
        # interventions are disabled the controller's aux loss is exactly
        # zero so this is a no-op for legacy archs. Vocab/d_model are
        # known here so the sub-modules (DAR discriminator, PCC projector,
        # CMD narrative head) can be sized correctly.
        self._build_reg_controller()
        # Optional caller-supplied function ids → (B,) domain labels.
        # When None, DAR self-disables with a one-time warning.
        self._domain_id_fn = None
        # Optional caller-supplied callable (harness → loss_tensor) that
        # runs an OOD anchor forward on a held-out prose batch. Required
        # by CDGA gradient surgery. When None, CDGA is dormant
        # (telemetry-only, no surgery applied).
        self._cdga_anchor_fn = None

        # ── VBB (Variational Bowtie Bottleneck) ──
        # Eagerly construct the sigma-head + log-beta scalar so they're
        # included by `self.parameters()` BEFORE _ensure_optimizer()
        # builds the optimizer on the first train_step. See
        # `_build_vbb_modules` for the math + rationale.
        self._build_vbb_modules()

        # ── Multi-Trunk-V2: specialist language cortex ensemble ──
        # Eagerly construct (same rationale as VBB) so its parameters
        # are visible to the optimizer.  Disabled-config ⇒ attribute set
        # to None, never built — preserves bit-for-bit reproducibility
        # of legacy single-cortex runs.  See architecture.md §5.7.
        self.multi_cortex = None
        self._build_multi_cortex()

        # ── Synthetic HPA axis (allostasis) ──
        # Slow homeostatic damper that distinguishes acute stress
        # (single bad batch — leave alone) from chronic stress
        # (positive-feedback runaway — damp NE / trophic growth / LR).
        # Disabled-config ⇒ attribute set to None, no metrics added,
        # bit-identical to legacy behaviour. See
        # neuroslm/neurochem/allostasis.py for the full math + cites.
        self.allostasis = None
        self._build_allostasis()

        # ── Geometric Information Funnel (GIF) ──
        # Three interlocking mechanisms that fix the train-PPL / OOD-PPL
        # gap divergence: (1) VBB α schedule, (2) OOD probe EMA,
        # (3) isotropy schedule. Disabled-config ⇒ all three are no-ops,
        # bit-identical to legacy behaviour. See lib/gif.neuro +
        # neuroslm/emergent/gif.py for math + rationale.
        self._gif = None
        self._build_gif()

    @classmethod
    def from_language_model(cls, language_model: nn.Module,
                            vocab_size: int, d_sem: int,
                            training_config: Optional[TrainingConfig] = None):
        """Build a harness that trains a full ids→logits LM directly.

        Used for the DSL transformer LM (build_language_model), which
        already contains its own embedding + blocks + lm_head. The harness
        contributes only the loss (with optional clipping), schedule, AMP,
        grad-accum, and checkpointing — no extra embedding/head.
        """
        h = cls.__new__(cls)
        nn.Module.__init__(h)
        h.circuit = nn.Identity()
        h.vocab_size = vocab_size
        h.d_sem = d_sem
        h.training_config = training_config or TrainingConfig()
        h.sink_population = ""
        h.language_model = language_model
        h.embedding = None
        h.lm_head = None
        h._optimizer = None
        h._accum_step = 0
        h._global_step = 0
        h._sched_warmup = None
        h._sched_total = None
        h._sched_min_ratio = 0.1
        h._amp_dtype = None
        h._grad_scaler = None
        h.total_loss_config = TotalLossConfig()
        h.maturity = MaturityTracker()
        h._last_lm_loss_value = h.maturity.l_random
        h._bema = None
        h._last_bema_info = {}
        h._nemori_skipped = 0
        h._genetics_orch = None
        h._transmitter_sys = None
        h._genetics_module_names = []
        h._surprise_ema = 0.0
        h._last_orch_modulation = None
        h._metrics = {}
        # Evolution heatmap hook — see neuroslm/evolution/harness_hook.py.
        # Mirror __init__ so the L6 NFG overlay path is safe to query
        # from this alternate constructor (DSL LM training).
        h._heatmap_hook = None
        # Cortex fusion state (slot A + slot C) — mirror __init__
        # so this alternate constructor doesn't crash the fusion path.
        h._last_pre_fusion_lm_logits = None
        h._last_pre_fusion_cortex_logits = None
        h._lm_loss_ema = 0.0
        h._cortex_loss_ema = 0.0
        h._cortex_inhibition_level = 0.0
        h._build_reg_controller()
        h._domain_id_fn = None
        h._cdga_anchor_fn = None
        # VBB modules (see __init__ — same eager-construction discipline).
        h._build_vbb_modules()
        # Multi-Trunk-V2 ensemble (see __init__).
        h.multi_cortex = None
        h._build_multi_cortex()
        # Synthetic HPA axis (allostasis) — see __init__.
        h.allostasis = None
        h._build_allostasis()
        # GIF controller — see __init__.
        h._gif = None
        h._build_gif()
        return h

    # ── Distributed training wrapping (DDP / FSDP) ────────────────────
    def enable_distributed(self, strategy: str = "ddp",
                            device_id: Optional[int] = None) -> None:
        """Wrap the language model in DDP or FSDP for multi-GPU training.

        Reads the DDP-related env vars set by torchrun (`RANK`,
        `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`). For
        single-process runs the strategy="single" path is a no-op.

        Call AFTER moving the harness to the correct CUDA device but
        BEFORE the first `train_step`.
        """
        if strategy == "single" or strategy == "":
            return
        if self.language_model is None:
            return
        import torch.distributed as dist
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.language_model = self.language_model.to(f"cuda:{local_rank}")
        if strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.language_model = DDP(
                self.language_model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,   # genetics + cortex hooks
            )
        elif strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.language_model = FSDP(self.language_model)
        else:
            raise ValueError(f"unknown dist_strategy {strategy!r}")

    # ── PR2: regularization controller (5 OOD interventions) ─────────

    def _build_reg_controller(self) -> None:
        """Build the RegularizationController from cfg.regularization.

        Safe to call multiple times — only the first call constructs.
        When `cfg.regularization` is None or all-disabled, the controller
        still exists but its `collect_aux` returns zero, so wiring into
        `compute_loss` is unconditional.
        """
        from neuroslm.regularizers import RegularizationController
        from neuroslm.dsl.regularization import RegularizationConfig
        cfg = getattr(self.training_config, "regularization", None)
        if cfg is None:
            cfg = RegularizationConfig()
            self.training_config.regularization = cfg
        # Pick a default chat_ratio from training config if available
        initial_chat_ratio = float(
            getattr(self.training_config, "chat_ratio", 0.6))
        self.reg_controller = RegularizationController(
            cfg, d_model=self.d_sem, vocab_size=self.vocab_size,
            initial_chat_ratio=initial_chat_ratio,
        )

    # ── Multi-Trunk-V2: specialist language cortex ensemble ─────────
    def _build_multi_cortex(self) -> None:
        """Build a `MultiCortexEnsemble` from `cfg.multi_cortex`.

        Behaviour:
          * `cfg.multi_cortex` missing OR `.enabled = False`
              → `self.multi_cortex = None`.  Legacy single-cortex
                runs reproduce bit-for-bit.
          * `.enabled = True, weights = "stub"`
              → builds a `StubSubCortex` ensemble via
                `build_default_ensemble`.  No network, no HF download.
                Used by tests and offline CI.
          * `.enabled = True, weights = "gpt2"`
              → calls `build_gpt2_ensemble` which lazy-imports
                `transformers` and downloads GPT-2 family checkpoints
                (`gpt2`, `gpt2-medium`, `distilgpt2`) the first time.
                Subsequent runs reuse the HF cache.

        The ensemble's parameters are registered as a child module
        (`self.multi_cortex = ensemble`) so they appear in
        `self.parameters()` and the lazy optimizer picks them up on
        the first `train_step`.

        Safe to call multiple times — idempotent when ensemble already
        built. Re-running with a different config is NOT supported
        (would orphan the previous optimizer's param refs).
        """
        cfg = getattr(self.training_config, "multi_cortex", None)
        if cfg is None or not cfg.enabled:
            self.multi_cortex = None
            self.cortex_lm_head = None
            self.cortex_mix_logit = None
            self.cortex_pre_head_norm = None
            return
        if self.multi_cortex is not None:
            return  # already built — idempotent

        # ── New path: `experts: [...]` roster → LMExpertEnsemble ───
        # When the operator declares an explicit per-expert roster
        # (the preferred path going forward), every expert returns
        # logits directly in trunk-vocab space via its OWN pretrained
        # LM head — no random Xavier projection, no rogue-dim band-aid
        # LayerNorm, no tied head plagiarised from the trunk's
        # untrained embedding. Initial CE on natural English drops
        # from `~ln(V)` (uniform baseline, the catastrophic legacy
        # default) to `~3-5` nats from step 0. See
        # `tests/training/test_lm_expert_harness_integration.py` for
        # the smoking-gun contract and `neuroslm/experts.py` for the
        # full mechanism (VocabBridge + per-tokenizer alignment).
        if getattr(cfg, "experts", None) is not None:
            # Lazy import to keep legacy paths free of the experts
            # module load (which imports transformers eagerly under
            # the hood of LMExpert).
            from neuroslm.experts import build_lm_expert_ensemble

            self.multi_cortex = build_lm_expert_ensemble(
                experts=cfg.experts,
                trunk_tokenizer=getattr(cfg, "trunk_tokenizer", "gpt2"),
                vocab_size=self.vocab_size,
                router_d_model=int(getattr(cfg, "router_d_model", 256)),
                lexical_bias_weight=float(getattr(
                    cfg, "lexical_bias_weight", 2.0)),
                bema_tau=float(getattr(cfg, "bema_tau", 0.5)),
            )
            # The ensemble already produces (B, T, V_trunk) logits, so
            # the legacy projection chain (cortex_pre_head_norm + the
            # tied cortex_lm_head) is BY DESIGN absent on this path.
            # `cortex_lm_head=None` is the harness's signal in forward()
            # to skip the chain entirely (see the fusion branch).
            self.cortex_lm_head = None
            self.cortex_pre_head_norm = None
            fusion_mode = getattr(cfg, "fusion_mode", "logits_mixture")
            if fusion_mode == "off":
                self.cortex_mix_logit = None
                return
            # Still build the mixing scalar so the operator can ramp
            # the ensemble in/out of the trunk logits over training
            # (identical surface to the legacy fusion path).
            fusion_init = float(getattr(cfg, "fusion_init", 0.5))
            fusion_init = min(max(fusion_init, 1e-6), 1.0 - 1e-6)
            init_logit = math.log(fusion_init / (1.0 - fusion_init))
            self.cortex_mix_logit = nn.Parameter(
                torch.tensor([init_logit], dtype=torch.float32)
            )
            return

        # Lazy import so legacy runs that never enable multi_cortex
        # never pay the import cost (cortex.py loads torch.nn extras
        # + sets up the DomainLexicon BPE table).
        from neuroslm.cortex import (
            build_default_ensemble, build_gpt2_ensemble,
            DEFAULT_GPT2_VARIANTS,
        )

        # The ensemble's d_target must equal d_sem so its (B, T, d_target)
        # output is dimensionally compatible with the fusion head. The
        # config's `router_d_model` is reinterpreted here as the d_target;
        # if it doesn't match d_sem we override it and warn (so fusion
        # works without the operator hand-tuning two related numbers).
        d_target = self.d_sem
        if cfg.router_d_model != d_target:
            import warnings
            warnings.warn(
                f"multi_cortex.router_d_model={cfg.router_d_model} != "
                f"harness.d_sem={d_target}; overriding to d_sem so the "
                f"fusion path lines up dimensionally.",
                RuntimeWarning,
            )

        if cfg.weights == "stub":
            self.multi_cortex = build_default_ensemble(
                vocab=self.vocab_size,
                d_model=d_target,
                domains=tuple(cfg.domains),
                lexical_bias_weight=cfg.lexical_bias_weight,
                bema_tau=cfg.bema_tau,
            )
        elif cfg.weights == "gpt2":
            # The default variants are keyed by the 4 standard domain
            # names. If the user customised `domains`, fall back to
            # mapping every domain to "gpt2" rather than failing — a
            # warning makes the substitution visible.
            if set(cfg.domains) <= set(DEFAULT_GPT2_VARIANTS.keys()):
                variants = {d: DEFAULT_GPT2_VARIANTS[d] for d in cfg.domains}
            else:
                import warnings
                warnings.warn(
                    f"multi_cortex.domains={cfg.domains} not in "
                    f"DEFAULT_GPT2_VARIANTS={list(DEFAULT_GPT2_VARIANTS)}; "
                    "mapping all to 'gpt2'. Customise weights:variants in "
                    "code if a different mapping is needed.",
                    RuntimeWarning,
                )
                variants = {d: "gpt2" for d in cfg.domains}
            self.multi_cortex = build_gpt2_ensemble(
                d_target=d_target,
                variants=variants,
                freeze_weights=cfg.freeze_weights,
                lexical_bias_weight=cfg.lexical_bias_weight,
                bema_tau=cfg.bema_tau,
            )
        else:  # pragma: no cover — parser already rejects this
            raise ValueError(
                f"unknown multi_cortex.weights {cfg.weights!r} "
                "(parser should have rejected this)"
            )

        # ── Fusion head: makes the ensemble's output actually reach
        #    the LM head, so the pretrained cortex features (frozen or
        #    fine-tuned) contribute to the loss from step 0. ────────
        fusion_mode = getattr(cfg, "fusion_mode", "logits_mixture")
        if fusion_mode == "off":
            self.cortex_lm_head = None
            self.cortex_mix_logit = None
            self.cortex_pre_head_norm = None
            return

        # cortex_pre_head_norm: LayerNorm before the tied head.
        # ────────────────────────────────────────────────────────────
        # GPT-2's residual stream has a single rogue dimension with
        # standard deviation ~80× the median per-dim std (Timkey &
        # van Schijndel 2021, "All Bark and No Bite"). Even after
        # the model's own `ln_f`, `last_hidden_state.std()` is
        # dominated by that one direction. `cortex_proj` is a plain
        # `nn.Linear(768, d_sem)` that preserves the anisotropy, and
        # the tied head `cortex_h @ embed.T` then translates it into
        # logit-spikes of magnitude 5-10 on whichever vocab tokens
        # happen to align with the rogue direction in embed-space.
        # Softmax interprets that as "I'm 99% sure of token X" and
        # gets it wrong nearly every time, producing the observed
        # `loss ≈ 13.84 > ln(50257) = 10.82` catastrophic init.
        #
        # LayerNorm normalises per-token across the feature axis,
        # rescaling the rogue dimension to be commensurate with the
        # rest. With this in place, initial CE matches the LM-trunk
        # baseline (≈ ln(vocab_size)) and training begins from a
        # sensible loss instead of recovering from a 3-nat deficit.
        # Validated by `scripts/diagnose_catastrophic_loss.py`.
        self.cortex_pre_head_norm = nn.LayerNorm(self.d_sem)

        # cortex_lm_head: (vocab_size, d_sem)
        # The Linear stores its weight as (out=vocab, in=d_sem), the
        # same shape as the LM's input embedding — required for tied-
        # weights init below.
        self.cortex_lm_head = nn.Linear(self.d_sem, self.vocab_size,
                                        bias=False)

        # ── Tied-weights init ───────────────────────────────────────
        # Standard transformer trick: share storage with the LM's input
        # embedding so the cortex's initial logits are geometrically
        # aligned with the model's token space (instead of being random
        # Xavier noise that gives "confidently wrong" predictions and
        # pushes initial CE above log(vocab_size)).
        #
        # We try, in order:
        #   1. self.language_model.embed (DSLLanguageModel — from
        #      neuroslm/dsl/nn_lang.py — exposes .embed as
        #      nn.Parameter(vocab, d_model)).
        #   2. self.embedding.weight (legacy BRIANHarness path with the
        #      hand-built per-token embedding).
        # If neither is available, fall back to Xavier init — safer
        # than Linear's default kaiming_uniform_ for an LM head.
        tied = False
        if (getattr(self, "language_model", None) is not None
                and hasattr(self.language_model, "embed")
                and isinstance(self.language_model.embed, nn.Parameter)
                and tuple(self.language_model.embed.shape)
                    == (self.vocab_size, self.d_sem)):
            # Hard tie — share the same parameter. Gradients flowing
            # through cortex_lm_head also update the LM's input embed,
            # which is the standard GPT-style weight-tying setup.
            self.cortex_lm_head.weight = self.language_model.embed
            tied = True
        elif (getattr(self, "embedding", None) is not None
                and tuple(self.embedding.weight.shape)
                    == (self.vocab_size, self.d_sem)):
            self.cortex_lm_head.weight = self.embedding.weight
            tied = True

        if not tied:
            # Xavier-normal init — at least centred, with std controlled,
            # so initial logits aren't catastrophic.
            nn.init.xavier_normal_(self.cortex_lm_head.weight)

        # cortex_mix_logit: learnable scalar; sigmoid → α ∈ (0, 1).
        # Init at logit(fusion_init) so the initial α matches the
        # configured value. Default fusion_init=0.5 ⇒ logit=0.
        fusion_init = float(getattr(cfg, "fusion_init", 0.5))
        fusion_init = min(max(fusion_init, 1e-6), 1.0 - 1e-6)
        init_logit = math.log(fusion_init / (1.0 - fusion_init))
        self.cortex_mix_logit = nn.Parameter(
            torch.tensor([init_logit], dtype=torch.float32)
        )

    # ── Synthetic HPA axis (allostasis) ───────────────────────────────

    def _build_allostasis(self) -> None:
        """Construct the AllostaticController if enabled in the config.

        When ``cfg.allostasis.enabled = False`` (default), the attribute
        stays ``None`` and every later check (``allostasis_step``,
        ``_apply_lr_damping``, telemetry) short-circuits. So this is the
        only place "is allostasis on?" is decided.

        Safe to call multiple times — idempotent.
        """
        cfg = getattr(self.training_config, "allostasis", None)
        if cfg is None or not cfg.enabled:
            self.allostasis = None
            return
        if self.allostasis is not None:
            return  # already built — idempotent
        # Lazy import keeps legacy runs (allostasis disabled) free of
        # the neurochem.allostasis import cost.
        from neuroslm.neurochem.allostasis import AllostaticController
        self.allostasis = AllostaticController(cfg)

    def _build_gif(self) -> None:
        """Construct the GIFController if enabled in the config.

        When ``cfg.gif.enabled = False`` (default), ``self._gif`` stays
        ``None`` and every later check short-circuits — bit-identical to
        legacy behaviour.
        """
        gif_cfg = getattr(self.training_config, "gif", None)
        if gif_cfg is None:
            self._gif = None
            return
        enabled = gif_cfg.get("enabled", False) if isinstance(gif_cfg, dict) else False
        if not enabled:
            self._gif = None
            return
        from neuroslm.emergent.gif import GIFController
        self._gif = GIFController.from_config(self.training_config)

    def _read_ne_gaba_levels(self) -> tuple[float, float]:
        """Sample current per-batch-mean NE and GABA levels.

        Reads from ``self._transmitter_sys`` if it has been built (the
        genetics pathway constructs it lazily). Returns ``(0, 0)`` when
        the transmitter system isn't available — the controller treats
        zeros as ``below baseline`` so no spurious stress is registered.

        Wrapped in try/except because the lazy genetics path may set
        ``level`` to a non-standard shape during reset; we never want
        an allostasis telemetry read to crash a training step.
        """
        ts = getattr(self, "_transmitter_sys", None)
        if ts is None or getattr(ts, "level", None) is None:
            return 0.0, 0.0
        try:
            from neuroslm.neurochem.transmitters import NT_INDEX
            level = ts.level  # (B, N_NT)
            if level.numel() == 0:
                return 0.0, 0.0
            mean = level.detach().mean(dim=0)  # (N_NT,)
            return (float(mean[NT_INDEX["NE"]].item()),
                    float(mean[NT_INDEX["GABA"]].item()))
        except Exception:  # pragma: no cover - telemetry must never crash a step
            return 0.0, 0.0

    def _allostasis_step(self, loss: float, grad_norm: float) -> None:
        """Advance the allostatic controller by one step, then publish
        telemetry to ``self._metrics``.

        Called from ``train_step`` after the forward + backward and
        before ``optimizer.step()``. Order matters:
          1. Read current NE/GABA (set by the most recent genetics-orch step).
          2. Update load/cort using those + ``loss`` + ``grad_norm``.
          3. Apply the LR multiplier to the optimizer's param groups (in
             ``_apply_lr_damping``, called right before ``optimizer.step``).

        No-op when ``self.allostasis is None``.
        """
        if self.allostasis is None:
            return
        ne_level, gaba_level = self._read_ne_gaba_levels()
        self.allostasis.step(
            ne_level=ne_level, gaba_level=gaba_level,
            loss=float(loss), grad_norm=float(grad_norm),
        )
        # Publish telemetry to the runtime metric registry (surfaces on
        # the per-step train log line via train_dsl._format_metrics_line).
        self._metrics.update(self.allostasis.telemetry())

    def _apply_lr_damping(self, optimizer: torch.optim.Optimizer) -> None:
        """Multiply every param-group's LR by ``allostasis.lr_multiplier()``.

        Called from ``train_step`` immediately before ``optimizer.step()``.
        The scheduler in ``train_step`` re-bases each param-group's LR
        from the SCHEDULED value (``self._last_lr``) on every step, so
        this damping is per-step (never compounds). Applies a uniform
        scalar multiplier to all groups, preserving any per-group LLRD
        ratios.

        No-op when allostasis is disabled or its lr_multiplier is 1.0.
        """
        if self.allostasis is None:
            return
        mult = self.allostasis.lr_multiplier()
        if mult >= 1.0 - 1e-9:
            return  # no damping ⇒ skip the param-group walk
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * mult

    # ── Cortex fusion helpers (slot A + slot C) ──────────────────────

    def _distillation_lambda(self, gap_nats: float) -> float:
        """λ_t for the KL-distillation aux loss.

        Piecewise-linear ramp in `gap_nats = lm_loss_ema - cortex_loss_ema`:
            gap ≤ floor         → 0.0           (trunk has caught up)
            floor < gap < ceil  → linear interp
            gap ≥ ceiling       → lambda_max    (trunk much worse)

        Item 3 (NT-mod). When either ``distillation_5ht_gain`` or
        ``distillation_da_gain`` is non-zero, the gap-ramp value is
        multiplied by

            nt_mult = clamp(1 + k_5HT * z_5HT - k_DA * z_DA, 0, 2)

        with ``z_X = 2 * (level - 0.5) ∈ [-1, +1]``. NT levels are read
        from ``self._nt_levels_for_distill`` (the harness pushes the
        homeostat readout there each step). Missing channels default
        to ``0.5`` so the multiplier collapses to ``1`` (identity).

        Returns 0.0 unconditionally when `multi_cortex.distillation_enabled`
        is False, so the entire term collapses to a no-op for back-compat.
        """
        cfg = getattr(self.training_config, "multi_cortex", None)
        if cfg is None or not getattr(cfg, "distillation_enabled", False):
            return 0.0
        floor = float(cfg.distillation_gap_floor)
        ceiling = float(cfg.distillation_gap_ceiling)
        lam_max = float(cfg.distillation_lambda_max)
        if gap_nats <= floor:
            base = 0.0
        elif gap_nats >= ceiling:
            base = lam_max
        else:
            base = lam_max * (gap_nats - floor) / (ceiling - floor)

        # Item 3: NT modulation. Skip the work when both gains are 0
        # so back-compat callers see bit-identical behaviour.
        k_5ht = float(getattr(cfg, "distillation_5ht_gain", 0.0))
        k_da  = float(getattr(cfg, "distillation_da_gain", 0.0))
        if k_5ht == 0.0 and k_da == 0.0:
            return base

        nt_levels = getattr(self, "_nt_levels_for_distill", None) or {}
        ht5 = float(nt_levels.get("5HT", 0.5))
        da  = float(nt_levels.get("DA", 0.5))
        z_5ht = 2.0 * (ht5 - 0.5)
        z_da  = 2.0 * (da - 0.5)
        nt_mult = 1.0 + k_5ht * z_5ht - k_da * z_da
        nt_mult = max(0.0, min(2.0, nt_mult))
        return base * nt_mult

    def _update_cortex_inhibition(
        self, lm_loss: float, cortex_loss: float,
    ) -> None:
        """Update `_cortex_inhibition_level` toward a [0, 1] target.

        Drive signal: `gap = cortex_loss - lm_loss` (positive when the
        trunk has *outperformed* the cortex). The target is
            target = clip(gap / inhibition_temperature, 0, 1)
        — so a gap equal to `inhibition_temperature` (1 nat by default)
        means "fully gate the cortex off". A negative gap (trunk worse)
        gives target=0, which pulls inhibition back toward 0 if it had
        previously risen.

        Update rule: `level ← (1-α) · level + α · target`. The unit-
        interval clamp on the target plus convex-combination dynamics
        keep `_cortex_inhibition_level ∈ [0, 1]` for all inputs.

        No-op when `inhibition_enabled` is False — the level stays at
        whatever it was (default 0 ⇒ no effect on α_eff).
        """
        cfg = getattr(self.training_config, "multi_cortex", None)
        if cfg is None or not getattr(cfg, "inhibition_enabled", False):
            return
        temperature = float(cfg.inhibition_temperature)
        ema_alpha = float(cfg.inhibition_ema_alpha)
        gap = cortex_loss - lm_loss
        target = max(0.0, min(1.0, gap / max(temperature, 1e-6)))
        self._cortex_inhibition_level = (
            (1.0 - ema_alpha) * self._cortex_inhibition_level
            + ema_alpha * target
        )
        # Defensive clamp (floats can drift on long runs)
        self._cortex_inhibition_level = max(
            0.0, min(1.0, self._cortex_inhibition_level)
        )

    def _effective_alpha(self) -> float:
        """`α_eff = sigmoid(cortex_mix_logit) · (1 - cortex_inhibition)`.

        Returns just `α` when inhibition is disabled (back-compat).
        Returns 0.0 when the fusion head isn't built (no cortex to mix).
        """
        if getattr(self, "cortex_mix_logit", None) is None:
            return 0.0
        alpha = float(torch.sigmoid(self.cortex_mix_logit).detach().item())
        cfg = getattr(self.training_config, "multi_cortex", None)
        if cfg is None or not getattr(cfg, "inhibition_enabled", False):
            return alpha
        return alpha * (1.0 - self._cortex_inhibition_level)

    def _reset_stashes_for_test(self) -> None:
        """Test-only: clear the pre-fusion logit stashes between
        intentionally-replayed forward passes (used to isolate the KL
        term in the gradient-flow contract)."""
        self._last_pre_fusion_lm_logits = None
        self._last_pre_fusion_cortex_logits = None

    def _cortex_fusion_aux_step(
        self,
        total: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Add KL-distillation (slot A) and advance the inhibition EMA
        (slot C). Returns the new total loss.

        Behaviour matrix:

          fusion off   → unchanged total; metrics dict gets no fusion keys.
          fusion on,
          A & C off    → unchanged total; α_eff metric updated for log.
          A on         → total += λ_t · T² · KL(softmax(cortex.detach()/T)
                                              || softmax(lm/T));
                          KL is built from the pre-fusion logit stashes
                          set by forward(). `lm_loss_ema` and
                          `cortex_loss_ema` are advanced from per-token
                          CE of the respective pre-fusion logits.
          C on         → `cortex_inhibition_level` advances toward
                          clip((cortex - lm) / temperature, 0, 1).

        The EMAs (`_lm_loss_ema`, `_cortex_loss_ema`) and the inhibition
        state are advanced UNCONDITIONALLY when fusion is active —
        useful even with distillation off, because tests sometimes set
        the EMAs manually to drive λ_t without running an extra step.
        """
        cfg = getattr(self.training_config, "multi_cortex", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return total
        if (self._last_pre_fusion_lm_logits is None
                or self._last_pre_fusion_cortex_logits is None):
            # Fusion config says on, but the forward path didn't stash
            # (e.g. fusion_mode="off" or cortex_lm_head not built).
            # Still expose alpha_effective for the telemetry contract.
            self._metrics["alpha_effective"] = self._effective_alpha()
            self._metrics["cortex_inhibition"] = float(self._cortex_inhibition_level)
            return total

        lm_logits = self._last_pre_fusion_lm_logits
        cortex_logits = self._last_pre_fusion_cortex_logits

        # ── Per-pathway CE for EMA driving signal ──
        # Both are honest cross-entropies on the same targets, computed
        # in float32 for numerical stability.
        with torch.no_grad():
            B, T, V = lm_logits.shape
            flat_t = targets.reshape(-1)
            ce_lm = F.cross_entropy(
                lm_logits.detach().float().reshape(-1, V), flat_t
            ).item()
            ce_cx = F.cross_entropy(
                cortex_logits.detach().float().reshape(-1, V), flat_t
            ).item()
        # EMA bootstrap: first observation seeds the EMA so we don't
        # have a "warming up from 0" artefact that lets λ saturate
        # immediately on step 1.
        if self._lm_loss_ema == 0.0:
            self._lm_loss_ema = ce_lm
        if self._cortex_loss_ema == 0.0:
            self._cortex_loss_ema = ce_cx
        ema_a = 0.1  # responsive enough to track training progress
        self._lm_loss_ema = (1 - ema_a) * self._lm_loss_ema + ema_a * ce_lm
        self._cortex_loss_ema = (
            (1 - ema_a) * self._cortex_loss_ema + ema_a * ce_cx
        )

        # ── Slot A: KL distillation ──
        # Hinton 2015: scale by T² so the gradient magnitude is
        # comparable to the CE term across different temperatures.
        #
        # GIF-2: when OOD probe is ready, use probe EMA instead of
        # pre-fusion lm_ema for the gap computation. This gives the
        # distillation/inhibition gates a TRUE generalisation signal
        # rather than the systematically-inflated pre-fusion CE.
        lm_ema_for_gap = self._lm_loss_ema
        if (self._gif is not None and self._gif.enabled
                and self._gif.ood_probe.is_ready):
            lm_ema_for_gap = self._gif.ood_probe.ema
        gap_for_lambda = lm_ema_for_gap - self._cortex_loss_ema
        lam = self._distillation_lambda(gap_for_lambda)
        if lam > 0.0 and getattr(cfg, "distillation_enabled", False):
            T_dist = float(cfg.distillation_temperature)
            # H006: when `cfd_enabled = True`, use the three-stage
            # Capacity-Funneled Distillation path instead of legacy
            # Hinton KL. Otherwise fall through to the legacy code for
            # bit-identical reproduction of pre-H24 runs.
            if getattr(cfg, "cfd_enabled", False):
                # ── CFD: Stage 1 + Stage 2 + Stage 3 ──
                teacher = cortex_logits.detach()
                student = lm_logits
                # Stage 1 prep: top-K schedule
                K_t = cfd_topk_schedule(
                    step=self._global_step,
                    K_start=int(cfg.cfd_topk_start),
                    K_end=int(cfg.cfd_topk_end),
                    anneal_steps=int(cfg.cfd_topk_anneal_steps),
                )
                # Stage 2: entropy-matched effective temperature
                T_eff = cfd_effective_temperature(
                    student, teacher, T_dist,
                    floor_multiplier=float(cfg.cfd_temperature_floor),
                )

                # ── CFDv2 (GFD): build / update unigram prior ──
                # Maintain an EMA of the observed target distribution
                # over training tokens. This is the cheap, training-
                # corpus-derived estimate of p_uni(v). Used by both M2
                # (prior-residual subtraction in log-space) and M4
                # (PMI-based per-position K).
                #
                # Activated only when at least one v2 mechanism is on;
                # otherwise the v1 path stays bit-identical.
                gamma = float(getattr(cfg, "cfd_prior_gamma", 0.0))
                use_pointwise_k = bool(
                    getattr(cfg, "cfd_pointwise_k_enabled", False)
                )
                if gamma > 0.0 or use_pointwise_k:
                    V_full = student.size(-1)
                    # EMA update — counts smoothed with 1/V additive
                    # prior so log_prior is always well-defined.
                    with torch.no_grad():
                        counts = torch.bincount(
                            targets.reshape(-1).to(student.device),
                            minlength=V_full,
                        ).to(student.dtype)
                        batch_prior = (counts + 1.0) / (
                            counts.sum() + float(V_full)
                        )
                        if (
                            getattr(self, "_cfd_unigram_ema", None) is None
                            or self._cfd_unigram_ema.numel() != V_full
                        ):
                            # First step OR vocab changed → seed.
                            self._cfd_unigram_ema = batch_prior.clone()
                        else:
                            ema_p = 0.05  # slow — prior stabilises fast
                            self._cfd_unigram_ema = (
                                (1.0 - ema_p) * self._cfd_unigram_ema
                                + ema_p * batch_prior
                            )
                        log_prior = torch.log(
                            self._cfd_unigram_ema.clamp_min(1e-12)
                        )

                    # M2 — prior-residual sparsification (γ > 0 only;
                    # γ = 0 is a noop branch inside the helper).
                    teacher_v2 = cfd_prior_residual(
                        teacher, log_prior, gamma=gamma
                    )
                    # M4 — per-position K from teacher PMI.
                    if use_pointwise_k:
                        K_per_pos = cfd_pointwise_k_from_pmi(
                            teacher_v2,
                            log_prior,
                            K_min=int(cfg.cfd_pointwise_k_min),
                            K_max=int(cfg.cfd_pointwise_k_max),
                            scale=float(
                                getattr(cfg, "cfd_pmi_scale", 2.0)
                            ),
                        )
                        target_pdf = cfd_topk_target_var_k(
                            teacher_v2, K_per_pos, T=T_eff
                        )
                        K_telemetry = float(K_per_pos.float().mean().item())
                    else:
                        target_pdf = cfd_topk_target(
                            teacher_v2, K=K_t, T=T_eff
                        )
                        K_telemetry = float(K_t)
                else:
                    # v1 path: raw teacher, global K.
                    target_pdf = cfd_topk_target(
                        teacher, K=K_t, T=T_eff
                    )
                    K_telemetry = float(K_t)

                # Per-token KL (correct reduction — fixes Followup F1
                # by construction since we never go through batchmean
                # for the CFD path).
                log_student = F.log_softmax(student / T_eff, dim=-1)
                kl_per_tok = F.kl_div(
                    log_student, target_pdf, reduction="none"
                ).sum(-1).mean()
                kl_term = kl_per_tok * (T_eff ** 2)
                # Stage 3: gradient-alignment gate
                lam_eff, g_align = cfd_grad_alignment_gate(
                    kl_term, student, targets, lam_0=lam
                )
                total = total + lam_eff * kl_term
                self._metrics["distill_kl"] = float(kl_term.detach().item())
                self._metrics["distill_lambda"] = float(lam_eff)
                self._metrics["cfd_T_eff"] = float(T_eff)
                self._metrics["cfd_K"] = float(K_telemetry)
                self._metrics["cfd_g_align"] = float(g_align)
                # v2 telemetry — only meaningful when v2 path active
                self._metrics["cfd_prior_gamma"] = float(gamma)
                self._metrics["cfd_pointwise_k"] = (
                    1.0 if use_pointwise_k else 0.0
                )
            else:
                # Legacy Hinton KL (the H21–H23 path; reproduces
                # bit-for-bit).
                teacher = cortex_logits.detach()
                student = lm_logits
                log_student = F.log_softmax(student / T_dist, dim=-1)
                soft_teacher = F.softmax(teacher / T_dist, dim=-1)
                kl = F.kl_div(
                    log_student, soft_teacher, reduction="batchmean"
                ) * (T_dist ** 2)
                total = total + lam * kl
                self._metrics["distill_kl"] = float(kl.detach().item())
                self._metrics["distill_lambda"] = float(lam)
        else:
            self._metrics["distill_kl"] = 0.0
            self._metrics["distill_lambda"] = float(lam)
            # Clear CFD telemetry when distillation is off so log
            # consumers don't see stale values from prior steps.
            if getattr(cfg, "cfd_enabled", False):
                self._metrics["cfd_T_eff"] = 0.0
                self._metrics["cfd_K"] = 0
                self._metrics["cfd_g_align"] = 0.0
                self._metrics["cfd_prior_gamma"] = 0.0
                self._metrics["cfd_pointwise_k"] = 0.0

        # ── Slot C: NT-mediated inhibition update ──
        # Drive signal: gap = cortex - lm (positive ⇒ trunk overtook
        # cortex ⇒ inhibition should rise). Update is a no-op when
        # inhibition_enabled=False (back-compat).
        #
        # GIF-2: when OOD probe is ready, use probe EMA as the trunk's
        # "true" loss so the inhibition gate tracks real generalisation.
        lm_for_inh = ce_lm
        if (self._gif is not None and self._gif.enabled
                and self._gif.ood_probe.is_ready):
            lm_for_inh = self._gif.ood_probe.ema
        self._update_cortex_inhibition(
            lm_loss=lm_for_inh, cortex_loss=ce_cx,
        )

        # ── Telemetry ──
        self._metrics["lm_loss_ema"] = float(self._lm_loss_ema)
        self._metrics["cortex_loss_ema"] = float(self._cortex_loss_ema)
        if self._gif is not None and self._gif.enabled and self._gif.ood_probe.is_ready:
            self._metrics["gif_ood_probe_ema"] = float(self._gif.ood_probe.ema)
        self._metrics["cortex_inhibition"] = float(self._cortex_inhibition_level)
        self._metrics["alpha_effective"] = self._effective_alpha()

        return total

    def set_domain_id_fn(self, fn) -> None:
        """Register an `ids → (B,) long tensor of domain labels` callable.

        DAR requires per-sample domain labels (0=text, 1=chat). The data
        loader is the natural owner of that information; this hook lets
        the training script install a labeling function without modifying
        the harness's forward signature. When None (default), DAR
        self-disables with a one-time warning.
        """
        self._domain_id_fn = fn

    def set_cdga_anchor_fn(self, fn) -> None:
        """Register a CDGA anchor function `(harness) -> loss_tensor`.

        The function is called inside `train_step` after the standard
        backward. It MUST:

          1. Sample / cache a held-out OOD prose batch (separate from
             both the training mixture AND the OOD evaluation slice).
          2. Run a forward through `harness.language_model` on that batch.
          3. Return the scalar LM cross-entropy loss tensor (autograd-
             traced — the controller will call `.backward()` on it after
             zeroing the parameter grads).

        When None (default), CDGA is dormant. Its config + telemetry
        still exist but no surgery is applied. See docs/CDGA.md for
        a worked example anchor function against WikiText-103.
        """
        self._cdga_anchor_fn = fn

    def current_chat_ratio(self) -> float:
        """Public read of the adaptive mixture controller's current ratio.

        The dataloader should call this on every batch when
        `cfg.regularization.adaptive_mixture.enabled` is true.
        """
        return float(self.reg_controller.adaptive_mixture.ratio())

    # ── Runtime metric registry — usable anywhere in the control flow ──
    def metric(self, name: str, default: float = 0.0) -> float:
        """Read a runtime metric (phi, mat, surprise, lm_loss, gene_expr_mean...).

        Updated once per `compute_loss` step. Returns `default` if not yet
        populated. Available metric names:
            mat         — Maturity Index (0..1)
            lm_loss     — language-modelling loss
            surprise    — |loss - ema_loss| / max(ema_loss, eps)
            phi         — runtime Phi proxy (logit-entropy-based)
            phi_loss    — -phi (negated for backward)
            gene_*      — diagnostics from the GeneticOrchestrator
        """
        return float(self._metrics.get(name, default))

    def metrics_snapshot(self) -> Dict[str, float]:
        """All currently published metrics."""
        return dict(self._metrics)

    # ── Schedule + mixed-precision configuration ─────────────────────

    def set_schedule(self, warmup: int, total: int,
                     min_lr_ratio: float = 0.1) -> None:
        """Enable warmup+cosine LR scheduling over `total` steps."""
        self._sched_warmup = warmup
        self._sched_total = total
        self._sched_min_ratio = min_lr_ratio

    def enable_mixed_precision(self, dtype: str = "bf16") -> None:
        """Enable autocast. dtype ∈ {bf16, fp16}. bf16 needs no GradScaler;
        fp16 gets one. CPU-only runs silently keep fp32."""
        if dtype == "bf16":
            self._amp_dtype = torch.bfloat16
            self._grad_scaler = None
        elif dtype == "fp16":
            self._amp_dtype = torch.float16
            # GradScaler only meaningful on CUDA
            if torch.cuda.is_available():
                self._grad_scaler = torch.cuda.amp.GradScaler()
        else:
            raise ValueError(f"unsupported amp dtype {dtype!r}; use bf16 or fp16")

    def _autocast_ctx(self):
        """Return an autocast context manager (or nullcontext if AMP off)."""
        if self._amp_dtype is None:
            return nullcontext()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.autocast(device_type=device_type, dtype=self._amp_dtype)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, ids: torch.Tensor,
                nt_levels: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """`ids` is `(batch, seq_len)`; returns logits `(batch, seq_len, vocab)`.

        The DSL circuit operates on `(batch * seq_len, d_sem)` — we
        flatten the time dimension into the batch dim, run the circuit
        per-token, then reshape logits back. This treats each token as an
        independent forward pass through the brain — a deliberately simple
        starting model. Later phases add proper sequence dynamics (state
        propagation across tokens via the back-edge buffers, attention
        over time, etc.).
        """
        with self._autocast_ctx():
            if self.language_model is not None:
                # N5 path: full DSL transformer LM owns embedding→blocks→head.
                logits = self.language_model(ids)
            else:
                # Legacy per-token circuit path (cognitive overlay).
                batch, seq_len = ids.shape
                x = self.embedding(ids)
                x_flat = x.reshape(batch * seq_len, self.d_sem)
                outputs = self.circuit(x_flat, nt_levels=nt_levels)
                sink = self._pick_sink_output(outputs)
                sink = sink.reshape(batch, seq_len, self.d_sem)
                logits = self.lm_head(sink)

            # ── Multi-Trunk-V2 logits-mixture fusion ───────────────
            # When the cortex ensemble + fusion head are both built,
            # late-fuse the ensemble's predictions into the LM logits.
            # This is what makes the pretrained cortex weights actually
            # contribute to the loss — without it, the ensemble is
            # constructed but never invoked, and ~700M GPT-2 parameters
            # sit dormant while a Xavier-init LM head guesses badly.
            #
            # final_logits = (1 - α_eff) · lm_logits + α_eff · cortex_logits
            # where:
            #   α     = sigmoid(self.cortex_mix_logit) ∈ (0, 1) — the
            #           learnable mixing scalar the optimizer trains.
            #   α_eff = α · (1 - inhibition_level) — NT-gated effective
            #           weight, where `inhibition_level ∈ [0, 1]` rises
            #           as the trunk surpasses the cortex (see
            #           `_update_cortex_inhibition`). When inhibition is
            #           disabled (default), α_eff = α (back-compat).
            #
            # NOTE: cortex_h is passed through cortex_pre_head_norm
            # FIRST. This LayerNorm kills GPT-2's rogue-dimension
            # anisotropy (Timkey & van Schijndel 2021) before the
            # tied head amplifies it into logit-spikes. Without this
            # normalisation, initial CE blows up to ~13.8 nats
            # (vs ln(50257) = 10.82 baseline) — see
            # scripts/diagnose_catastrophic_loss.py for the in-vitro
            # repro.
            #
            # The pre-fusion lm_logits and cortex_logits are stashed on
            # the harness so compute_loss can build the KL-distillation
            # aux loss without redoing the forward pass.
            if (self.multi_cortex is not None
                    and getattr(self, "cortex_mix_logit", None) is not None):
                # Two ensemble dialects share this fusion branch:
                #   * Legacy `MultiCortexEnsemble`: returns hidden states
                #     `(B, T, d_sem)`. We then run them through the
                #     anisotropy-suppression LayerNorm (`cortex_pre_head_norm`)
                #     and the tied `cortex_lm_head` to reach trunk-vocab.
                #   * New `LMExpertEnsemble` (set via cfg.experts): the
                #     pretrained heads are already on the inside, so the
                #     return value is already `(B, T, V_trunk)` and the
                #     entire random-projection chain is bypassed (both
                #     `cortex_lm_head` and `cortex_pre_head_norm` are
                #     `None` in this case — `_build_multi_cortex` skipped
                #     building them).
                cortex_out = self.multi_cortex(ids)
                if getattr(self, "cortex_lm_head", None) is not None:
                    # Legacy hidden-state path. The (B, T, d_sem) output
                    # needs a LayerNorm + tied-head trip to trunk vocab.
                    cortex_h = cortex_out                       # (B, T, d_sem)
                    if getattr(self, "cortex_pre_head_norm", None) is not None:
                        cortex_h = self.cortex_pre_head_norm(cortex_h)
                    cortex_logits = self.cortex_lm_head(cortex_h)
                else:
                    # New experts path: ensemble output IS the cortex
                    # logits, already in trunk-vocab space. Assert this
                    # at dev time — easier to catch a contract violation
                    # here than in the loss.
                    if __debug__:
                        # `cortex_out` should be `(B, T, V_trunk)`; check
                        # the last dim only (B/T can differ on padded batches).
                        assert cortex_out.dim() == 3 and \
                            cortex_out.shape[-1] == self.vocab_size, (
                                f"LMExpertEnsemble must return (B, T, "
                                f"{self.vocab_size}) logits; got "
                                f"{tuple(cortex_out.shape)}"
                            )
                    cortex_logits = cortex_out
                # Cast cortex_logits to match logits dtype/device (autocast
                # may place them in bf16/fp16, but tied embed is fp32).
                cortex_logits = cortex_logits.to(logits.dtype)

                # Stash pre-fusion logits for the KL-distillation aux
                # loss in compute_loss(). Both tensors are still in the
                # autograd graph — KL backward will follow them into
                # the trunk (student) and stop at .detach() on the
                # teacher (cortex). See _distillation_lambda for the
                # admission schedule.
                self._last_pre_fusion_lm_logits = logits
                self._last_pre_fusion_cortex_logits = cortex_logits

                # NT-gated effective α (slot C). When inhibition is
                # disabled the cfg path returns plain `α`, preserving
                # bit-for-bit equivalence with the pre-NT-gate behaviour.
                cfg_mc = getattr(self.training_config, "multi_cortex", None)
                inhibition_on = (
                    cfg_mc is not None
                    and getattr(cfg_mc, "inhibition_enabled", False)
                )
                alpha = torch.sigmoid(self.cortex_mix_logit)  # (1,)
                if inhibition_on:
                    # Inhibition is a non-trainable scalar (updated by
                    # the loss EMA loop). Multiply in autograd-graph so
                    # the optimizer still sees the gradient on α.
                    inhibition = torch.tensor(
                        self._cortex_inhibition_level,
                        dtype=alpha.dtype, device=alpha.device,
                    )
                    alpha_eff = alpha * (1.0 - inhibition)
                else:
                    alpha_eff = alpha
                logits = (1.0 - alpha_eff) * logits + alpha_eff * cortex_logits
            else:
                # Fusion path inactive — clear any stale stashes so a
                # later distillation lookup can't pick them up across a
                # config change.
                self._last_pre_fusion_lm_logits = None
                self._last_pre_fusion_cortex_logits = None
        # Return float32 logits regardless of autocast dtype, so loss math
        # downstream is numerically stable.
        return logits.float()

    def _pick_sink_output(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the tensor that goes to the LM head.

        Default: `outputs[sink_population]`. Fallback: the last
        population's output if the named sink isn't present (useful for
        ad-hoc circuits in tests).
        """
        if self.sink_population in outputs:
            return outputs[self.sink_population]
        # Fallback — last key inserted (Python 3.7+ dicts are ordered)
        return next(reversed(outputs.values())) if outputs else \
            torch.zeros(1, self.d_sem)

    # ── Loss ─────────────────────────────────────────────────────────

    def _build_vbb_modules(self) -> None:
        """Construct the Variational Bowtie Bottleneck (VBB) sub-modules.

        Adds two attributes:

        * ``self._vbb_sigma_head`` — an ``nn.Linear(d_sem, d_sem)`` that
          maps the motor mean ``μ = h_motor`` to a per-element
          log-variance ``log σ²``. Initialised to a constant
          ``log σ² ≈ -8`` (σ ≈ 0.018) so the very first reparam sample
          is near-deterministic and behaviour matches the legacy
          (pre-VBB) loss up to a tiny stochastic perturbation. The
          optimizer is free to grow σ as the KL term tightens; the
          MDRV-VBB Posterior Entropy Commitment (PEC) term makes
          σ = 0 a repulsive fixed point so the optimizer is *also*
          always nudged away from the degenerate channel.
        * ``self._vbb_log_beta`` — a single learnable scalar
          ``nn.Parameter`` whose softplus is the precision ``β`` of the
          predictive-coding likelihood. Initialised from
          ``training_config.vbb_beta_init`` (default 1.0) via the inverse
          softplus ``log(exp(β) − 1)``. The MDRV-VBB β-ceiling
          (``vbb_log_beta_max``) clamps this at runtime — the parameter
          is unclamped in storage; the clamp is applied differentiably
          in ``_compute_pc_reentry_loss``.

        Total parameter cost at ``d_sem = 512``: 512×512 + 512 + 1 =
        262 657 params ≈ 0.21 % of a 122 M model — negligible.

        When ``training_config.vbb_alpha <= 0`` we still construct both
        attributes as ``None`` so the codepath in
        ``_compute_pc_reentry_loss`` can detect the off-state cheaply.
        Eager construction is required: ``_ensure_optimizer`` runs
        before the first ``compute_loss``, so anything added lazily
        from inside the loss would be missed by AdamW's param list.
        """
        alpha = float(getattr(self.training_config, "vbb_alpha", 0.0))
        if alpha <= 0.0 or self.d_sem is None or int(self.d_sem) <= 0:
            self._vbb_sigma_head = None
            # Register as a plain attribute (not a Parameter) so it
            # doesn't pollute the optimizer when VBB is off.
            self._vbb_log_beta = None
            return
        d = int(self.d_sem)
        head = nn.Linear(d, d)
        # Zero-init the weight so log σ² = bias for the very first step
        # (independent of μ); bias = -8 ⇒ σ ≈ 0.018 ⇒ near-deterministic
        # encoder at init. The model picks up structure as the KL relaxes.
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, -8.0)
        self._vbb_sigma_head = head
        beta_init = float(getattr(self.training_config,
                                  "vbb_beta_init", 1.0))
        beta_init = max(beta_init, 1e-4)
        # Inverse softplus: log_beta such that softplus(log_beta) = β₀.
        # log(exp(x) − 1) is numerically fine for x ≥ ~1e-3; for very
        # small x we'd hit -inf, hence the floor above.
        raw = math.log(math.expm1(beta_init))
        self._vbb_log_beta = nn.Parameter(torch.tensor(raw,
                                                       dtype=torch.float32))

    def _compute_pc_reentry_loss(self, base_weight: float):
        r"""Return the NT-gated PC-reentry trunk loss term (scalar tensor)
        or ``None`` if not applicable this step.

        Two code paths:

        **Legacy path** (``vbb_alpha <= 0``) — bit-identical to the
        Jun 2026 v1 surgery: lazily build a :class:`PCReentryProbe`,
        run its internal SGD on detached activations for telemetry,
        then form ``r = ‖s − W·m‖²`` via ``residual_diff`` (probe-W is
        frozen, gradient flows into the trunk only), gate by
        ``max(0, 1 + 0.5·DA − 0.7·GABA)``, scale by ``pc_reentry_weight``.

        **VBB path** (``vbb_alpha > 0``) — the *Variational Bowtie
        Bottleneck*. Mathematically:

        .. math::

            q(h_m \mid x) = \\mathcal{N}(\\mu_\\theta(x),
                                          \\sigma_\\theta(x)^2 I), \\\\
            \\hat h_m = \\mu + \\sigma \\cdot \\epsilon,
            \\quad \\epsilon \\sim \\mathcal{N}(0, I), \\\\
            r = \\| h_s - W \\hat h_m \\|^2, \\\\
            \\beta = \\mathrm{softplus}(\\log\\beta), \\\\
            \\mathrm{KL} = \\tfrac{1}{2}
                \\sum_d (\\sigma_d^2 + \\mu_d^2 - 1 - \\log \\sigma_d^2), \\\\
            \\mathcal{L}_{\\mathrm{VBB}} = \\gamma_{\\mathrm{NT}}
                \\cdot \\bigl(\\beta r - \\log\\beta
                              + \\alpha\\,\\mathrm{KL}\\bigr).

        Why each term exists:

        * ``β·r``  is the precision-weighted prediction error
          (Friston). High β = confident predictor → loop demands a
          tight residual; low β = uncertain → relaxed.
        * ``−log β`` is the Gaussian normaliser; without it β would
          collapse to 0 and the residual term would vanish. Together
          they fix ``β`` at the closed-form equilibrium
          ``β* = 1 / (2·E[r])``  — i.e. ``β`` auto-tracks the inverse
          variance of the residual. This is what stops the C3 loop
          from being a self-distillation amplifier.
        * ``α·KL[q ‖ N(0,I)]`` is an Information Bottleneck (Tishby)
          placed at the bowtie waist — the narrowest cross-section
          of the network, so the IB has maximum leverage. It caps
          ``I(X; h_m)`` and forces the motor pole to keep only
          *generalisable* statistics, killing memorisation.

        The reparam noise injection at the C3 read-site additionally
        serves as stochastic gradient Langevin perturbation, biasing
        SGD toward flat minima (Keskar et al.).

        Composition with the NT gate is unchanged: the whole
        free-energy term is multiplied by ``γ_NT`` so DA/GABA tone the
        loop just as before.

        Returns
        -------
        torch.Tensor (scalar) or ``None``
            ``None`` when the motor / sensory stash is unavailable,
            shapes don't match, or the effective weight collapses to
            zero (e.g. high GABA quenches the gate). Else an
            autograd-tracked scalar to be added to the LM loss.
        """
        h_m = getattr(self.language_model, "_last_h_motor", None)
        h_s = getattr(self.language_model, "_last_h_sensory", None)
        if h_m is None or h_s is None:
            return None
        # Lazy-construct a probe sized to the population dim. The probe
        # is sized once and reused; if d_model changes mid-run (it
        # shouldn't) we drop and rebuild.
        dim = int(h_m.shape[-1])
        probe = getattr(self, "_pc_reentry_probe", None)
        if probe is None or getattr(probe, "dim", -1) != dim:
            try:
                from neuroslm.emergent.pc_reentry import PCReentryProbe
            except Exception:
                return None
            probe = PCReentryProbe(dim=dim, device=h_m.device)
            self._pc_reentry_probe = probe
        # Run probe's internal SGD on a detached copy (telemetry path).
        # We always pass the deterministic μ here so the probe learns
        # the noise-free mapping (cleaner W).
        try:
            probe.step(h_m.detach(), h_s.detach())
        except Exception:
            pass

        # ── NT gate (shared by both paths) ──────────────────────────
        gate_scalar = 1.0
        if getattr(self.training_config, "pc_reentry_nt_gate", False):
            obs = getattr(self, "_observer", None) \
                or getattr(self, "observer", None) \
                or getattr(self, "last_observer", None)
            nt = None
            if obs is not None and hasattr(obs, "nt"):
                try:
                    nt = obs.nt.levels()
                except Exception:
                    nt = None
            if nt is not None:
                da = float(nt.get("DA", 0.0))
                gaba = float(nt.get("GABA", 0.0))
                gate_scalar = max(0.0, 1.0 + 0.5 * da - 0.7 * gaba)

        # ── Branch on VBB ────────────────────────────────────────────
        # GIF-1: when GIF is enabled, use the scheduled α instead of
        # the static config value. The schedule ramps from α_start
        # (loose, 0.001) to α_end (tight, 0.05) over training.
        if self._gif is not None and self._gif.enabled:
            alpha = self._gif.vbb_alpha(self._global_step)
        else:
            alpha = float(getattr(self.training_config, "vbb_alpha", 0.0))
        use_vbb = (alpha > 0.0
                   and self._vbb_sigma_head is not None
                   and self._vbb_log_beta is not None)

        if not use_vbb:
            # Legacy path: plain ‖s − W·m‖² with frozen W.
            pc_diff = probe.residual_diff(h_m, h_s)
            if pc_diff is None:
                return None
            eff_w = base_weight * gate_scalar
            try:
                self._metrics["pc_reentry_loss"] = float(pc_diff.detach())
                self._metrics["pc_reentry_gate"] = float(gate_scalar)
                self._metrics["pc_reentry_eff_weight"] = float(eff_w)
            except Exception:
                pass
            if eff_w <= 0.0:
                return None
            return eff_w * pc_diff

        # VBB path. We migrate the head + scalar to the trunk device
        # lazily — the harness might be moved to CUDA after
        # construction. .to() is a no-op when already there.
        device = h_m.device
        sigma_head = self._vbb_sigma_head.to(device)
        log_beta_param = self._vbb_log_beta.to(device)

        mu = h_m.to(dtype=torch.float32)                  # (B, T, D)
        # Per-element log-variance from the head. We work in float32
        # to keep the exp() stable under bf16 autocast.
        log_var = sigma_head(mu)                          # (B, T, D)
        # Clamp the log-variance to a safe range so exp() can't blow
        # up under unlucky init / amp. log σ² ∈ [-12, 4] →
        # σ ∈ [~2.5e-3, ~7.4].
        log_var = log_var.clamp(-12.0, 4.0)
        sigma = (0.5 * log_var).exp()

        # Reparameterised sample (same shape as μ). The graph carries
        # gradient through both μ and σ.
        eps = torch.randn_like(mu)
        mu_sample = mu + sigma * eps

        # Residual using the frozen-W probe on the noised motor. The
        # probe handles dtype/device internally and returns a scalar
        # mean-squared residual. Gradient flows through `mu_sample`
        # (→ μ and σ) and through `h_s`.
        residual = probe.residual_diff(mu_sample, h_s)
        if residual is None:
            return None

        # ── β with ceiling — breaks β/σ co-collapse (MDRV-VBB) ───────
        # Without a ceiling the joint equilibrium (β→∞, σ→0) eventually
        # erases the bottleneck. Clamping log β at log_beta_max bounds
        # the precision reward, making the co-collapse non-optimal.
        log_beta_max = float(getattr(self.training_config,
                                     "vbb_log_beta_max", 0.0))
        if log_beta_max > 0.0:
            log_beta_eff = log_beta_param.clamp(max=log_beta_max)
        else:
            log_beta_eff = log_beta_param
        beta = F.softplus(log_beta_eff) + 1e-6

        # ── Free-bits KL floor (Kingma et al. 2016, §3.5) ────────────
        # Per-element KL: ½(σ² + μ² − 1 − log σ²), shape (B, T, D).
        kl_per_dim = 0.5 * (log_var.exp() + mu.pow(2) - 1.0 - log_var)
        free_bits = float(getattr(self.training_config, "vbb_free_bits", 0.0))
        if free_bits > 0.0:
            # Clamp each element to [δ, ∞) so the KL can never fully
            # vanish in any dimension.  The gradient of the clamped term
            # is zero *below* δ so the network cannot recoup loss by
            # shrinking σ further — making σ→0 provably non-optimal.
            kl_per_dim = kl_per_dim.clamp(min=free_bits)
        kl = kl_per_dim.mean()

        # ── HPB Phase 4 — Hyperbolic Bowtie Waist (HBW) ──────────────
        # When vbb_curvature > 0, switch the posterior from Euclidean
        # N(μ, σ²I) to a wrapped Gaussian on the Poincaré ball B^d_c.
        # The KL acquires an extra Jacobian-log-det term (Skopek et al.
        # 2020, eq. 12) which is ≥ 0 for any non-zero ‖μ‖ — strictly
        # upper-bounding the Euclidean KL and making σ-collapse harder
        # by the construction of the geometry alone. Free-bits is
        # applied to the per-dim Euclidean part inside the helper.
        vbb_curvature = float(getattr(self.training_config,
                                      "vbb_curvature", 0.0))
        if vbb_curvature > 0.0:
            try:
                from neuroslm.emergent.hyperbolic import wrapped_normal_kl
                kl = wrapped_normal_kl(mu, log_var, c=vbb_curvature,
                                       free_bits=free_bits)
            except Exception:
                # Fall back silently to the Euclidean KL on any
                # numerical hiccup (e.g. ‖μ‖ at the boundary). The
                # training loop must never crash on the math swap.
                pass

        # ── Posterior Entropy Commitment — Jeffreys σ-prior (PEC) ────
        # Adds −η · ½ · 𝔼[log σ²] = +η · H_Gauss[q] + const to loss.
        # As σ→0 the term diverges to +∞, making σ=0 a Lyapunov-
        # unstable fixed point: the collapse attractor is repulsive by
        # construction regardless of batch noise.  Mathematically this
        # is a Jeffreys prior p(σ_d) ∝ 1/σ_d on each scale parameter
        # (Jeffreys 1946; Berger & Bernardo 1992).
        entropy_eta = float(getattr(self.training_config,
                                    "vbb_entropy_eta", 0.0))
        pec_term = 0.0
        if entropy_eta > 0.0:
            pec_term = -entropy_eta * 0.5 * log_var.mean()

        # ── Free energy: β·r − log β + α·KL + PEC ────────────────────
        free_energy = beta * residual - torch.log(beta) + alpha * kl + pec_term

        eff_w = base_weight * gate_scalar
        try:
            self._metrics["pc_reentry_loss"] = float(residual.detach())
            self._metrics["pc_reentry_gate"] = float(gate_scalar)
            self._metrics["pc_reentry_eff_weight"] = float(eff_w)
            self._metrics["vbb_alpha"] = float(alpha)
            self._metrics["vbb_beta"] = float(beta.detach())
            self._metrics["vbb_kl"] = float(kl.detach())
            self._metrics["vbb_sigma_mean"] = float(sigma.detach().mean())
            self._metrics["vbb_free_energy"] = float(free_energy.detach())
            if entropy_eta > 0.0:
                pec_val = pec_term if isinstance(pec_term, float) \
                    else float(pec_term.detach())
                self._metrics["vbb_pec"] = pec_val
        except Exception:
            pass
        if eff_w <= 0.0:
            return None
        return eff_w * free_energy

    def _compute_mspcc_loss(self, base_weight: float):
        """HPB Phase 3 — Multi-Scale Predictive Coding Cascade loss.

        When ``training_config.mspcc`` is a dict with ``enabled = True``
        and the wrapped language model exposes ``_last_layer_outputs``
        (a list of per-block hidden states), run the per-layer VBB
        free-energy cascade (see :mod:`neuroslm.emergent.mspcc`) with
        the shared MDRV stabiliser hyperparameters.

        Returns ``None`` when MSPCC is disabled / the stash is missing /
        there are fewer than two layers.  Designed to be **additive**
        with the single-waist VBB at :meth:`_compute_pc_reentry_loss`.

        Parameters
        ----------
        base_weight
            Override for the MSPCC base λ_0. Falls back to
            ``mspcc["base_weight"]`` when 0.

        Returns
        -------
        torch.Tensor (scalar) or None
        """
        mspcc_cfg = getattr(self.training_config, "mspcc", None)
        if mspcc_cfg is None or not bool(mspcc_cfg.get("enabled", False)):
            return None
        lm = self.language_model
        if lm is None:
            return None
        layer_outs = getattr(lm, "_last_layer_outputs", None)
        if not layer_outs or len(layer_outs) < 2:
            return None
        try:
            from neuroslm.emergent.mspcc import compute_mspcc_loss
        except Exception:
            return None
        # Resolve hyperparameters.
        base_w = float(base_weight) if base_weight and float(base_weight) > 0 \
            else float(mspcc_cfg.get("base_weight", 0.05))
        decay = float(mspcc_cfg.get("layer_weight_decay", 0.5))
        # GIF-1: use scheduled α when GIF is enabled.
        if self._gif is not None and self._gif.enabled:
            alpha = self._gif.vbb_alpha(self._global_step)
        else:
            alpha = float(getattr(self.training_config, "vbb_alpha", 0.001))
        free_bits = float(getattr(self.training_config, "vbb_free_bits", 0.0))
        log_beta_max = float(getattr(self.training_config,
                                     "vbb_log_beta_max", 0.0))
        entropy_eta = float(getattr(self.training_config,
                                    "vbb_entropy_eta", 0.0))
        loss = compute_mspcc_loss(
            layer_outs,
            base_weight=base_w,
            layer_weight_decay=decay,
            alpha=alpha,
            free_bits=free_bits,
            log_beta_max=log_beta_max,
            entropy_eta=entropy_eta,
        )
        if loss is None:
            return None
        try:
            self._metrics["mspcc_loss"] = float(loss.detach())
            self._metrics["mspcc_base_weight"] = float(base_w)
            self._metrics["mspcc_n_pairs"] = int(len(layer_outs) - 1)
        except Exception:
            pass
        return loss

    # ── NT distribution seam ──────────────────────────────────────────

    def distribute_nt_levels(self, levels: Optional[Dict[str, float]]) -> None:
        """Fan out the live neuromodulator dict to every consumer.

        Single seam called by :meth:`compute_loss` whenever a fresh
        NT dict is available (the training loop sources it from the
        observer's :class:`~neuroslm.emergent.DrivenNTSystem`).

        Consumers (each one a no-op if the relevant gain is 0):

        * ``self._nt_levels_for_distill`` — read inside
          :meth:`_distillation_lambda` for the 5HT/DA-modulated
          distillation strength multiplier (Item 3).
        * ``self.multi_cortex.set_nt_levels(levels)`` if supported —
          the new :class:`LMExpertEnsemble` forwards to both:
            - its ``ThalamicRouter`` (NE → routing-softmax sharpness,
              Item 2),
            - its optional ``LateralInhibition`` (GABA → Mexican-hat
              κ, Item 4).
          The legacy ``MultiCortexEnsemble`` doesn't expose this
          method — we silently skip the push for back-compat.

        Passing ``None`` is a no-op so callers can be permissive.
        """
        if levels is None:
            return
        # Items 3 + (downstream uses): the gap-ramp ×NT-multiplier
        # distillation reads this dict by name in `_distillation_lambda`.
        self._nt_levels_for_distill = dict(levels)
        # Items 2 + 4: the new expert ensemble fans out internally to
        # router (NE) and lateral inhibition (GABA). Polymorphic guard
        # for the legacy MultiCortexEnsemble which has no such method.
        mc = getattr(self, "multi_cortex", None)
        if mc is not None and hasattr(mc, "set_nt_levels"):
            mc.set_nt_levels(levels)

    def compute_loss(self, ids: torch.Tensor, targets: torch.Tensor,
                     nt_levels: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Cross-entropy loss with optional per-sample clipping + aux losses.

        Per-sample clipping (p4 fix): compute per-sequence mean loss,
        clip each at `factor * batch_median`, then average. Suppresses
        outlier-sequence dominance of the gradient.

        Aux losses: every loss stashed via the `_last_*_loss` convention on
        the wrapped LM is aggregated using Brain's exact phase-gated weight
        formula —  `aux_w * phase_gate(mat, center, width) * w_aux * loss_aux`
        — so the trunk-affecting gradient matches Brain bit-for-bit. The
        PCH aux is the only one routed to the LM trunk via the language
        model's per-layer outputs; the rest (world/forward/motor/orchestrator)
        come from sidecar modules and are detached from the trunk in Brain,
        so they only affect the `loss` column, not the LM trajectory.

        Recognised stash keys (set by the DSL Brain aggregator or wrappers):
          * `_last_pred_coding_loss` → aux key 'pred_coding'  (trunk-affecting)
          * `_last_world_loss`       → 'world'
          * `_last_forward_loss`     → 'forward'
          * `_last_motor_loss`       → 'motor'
          * `_last_kl_world_loss`    → 'kl_world'
          * `_last_novel_aux_loss`   → 'novel'
          * `_last_cpc_loss`         → 'cpc'
          * `_last_phi_loss`         → 'phi'
        """
        # ── NT-distribution seam: push the live homeostat dict to every
        # consumer BEFORE the forward fires.  This is what activates
        # Items 2/3/4 — without it the router temp, lateral-inhibition
        # κ, and distillation λ all stay at their centre defaults
        # forever.  No-op when `nt_levels` is None (back-compat).
        if nt_levels is not None:
            self.distribute_nt_levels(nt_levels)

        # ── MAT-phase-gated mechanism multipliers ──
        # If the arch declared `mechanisms { dropout: { strength, phase_gate },
        # pct_trunk: { ... } }`, refresh the cortex's runtime multipliers
        # for this step. With phase_gate.center > 0 these start at ~0
        # (mechanism OFF, vanilla baseline training) and ramp to full
        # `strength` as maturity crosses `center`. Lets us reach the
        # ~70 train-PPL floor first, then engage OOD regularizers.
        if (self.language_model is not None
                and hasattr(self.language_model, "set_mat_multipliers")):
            mat_now = self.maturity.value()
            mech = self.training_config.mechanisms
            eff_drop = mech.effective_dropout(
                mat_now, fallback=self.training_config.dropout)
            eff_pct  = mech.effective_pct_trunk(
                mat_now, fallback=self.training_config.pct_trunk)
            self.language_model.set_mat_multipliers(eff_drop, eff_pct)

        # ── PRE-pass genetic step ──
        # Run the orchestrator BEFORE the forward so its (live, non-detached)
        # baseline_offsets enter the forward graph this step and gene
        # gradients flow back through the LM loss. Uses the previous step's
        # metrics (mat, surprise, phi proxy) as regulatory context — the
        # natural 1-step latency mirrors real transcription/translation.
        pre_phi_loss = None
        gcfg = self.training_config.genetics
        if gcfg.enabled:
            pre_phi_loss = self._step_genetics_pre(ids.shape[0], ids.device)

        logits = self(ids, nt_levels=nt_levels)

        # ── PR2-F: frequency-balanced CE replacement ──
        # If freq_balance is enabled AND statistics have been loaded
        # via reg_controller.freq_balance.set_frequencies(...), the
        # standard mean-CE is replaced with a per-token-weighted CE
        # that biases the gradient direction toward the OOD target.
        # This is the cheapest possible direction-aware OOD signal:
        # it acts on the LM gradient itself (not an aux loss), so it
        # composes additively with the other interventions in `total`.
        reg_cfg_early = getattr(self.training_config, "regularization", None)
        use_freq_balance = (
            reg_cfg_early is not None
            and reg_cfg_early.freq_balance.enabled
            and getattr(self.reg_controller.freq_balance, "_fitted", False)
        )
        if use_freq_balance:
            B, T, V = logits.shape
            flat_l = logits.reshape(-1, V)
            flat_t = targets.reshape(-1)
            per_tok = F.cross_entropy(flat_l, flat_t, reduction="none")
            loss_lm = self.reg_controller.freq_balance(per_tok, flat_t)
        else:
            loss_lm = self._compute_loss_from_logits(logits, targets)

        # ── Stage 8 OOD push: flooding loss ──
        # |loss - b| + b prevents the model from memorizing below floor b.
        flood = self.training_config.flooding_level
        if flood > 0:
            loss_lm = (loss_lm - flood).abs() + flood

        total = self.total_loss_config.w_lm * loss_lm
        mat = self.maturity.value()

        # ── PR2: OOD-intervention controller ──
        # Pulls the final non-detached hidden state out of the language
        # model (stashed during forward) and adds each enabled aux loss
        # (DAR / PCC / Isotropy / CMD) to `total`. AdaptiveMixture
        # observes the logits but contributes no loss term (the dataloader
        # reads its ratio via `harness.current_chat_ratio()`).
        #
        # GIF-3: when GIF is enabled, override the isotropy weight with
        # the scheduled value. This activates isotropy even when the
        # static config says `enabled: false` — GIF takes ownership.
        reg_cfg = getattr(self.training_config, "regularization", None)

        if self._gif is not None and self._gif.enabled and reg_cfg is not None:
            gif_iso_w = self._gif.isotropy_weight(self._global_step)
            if gif_iso_w > 0.0:
                iso_cfg = getattr(reg_cfg, "isotropy", None)
                if iso_cfg is not None:
                    iso_cfg.enabled = True
                    iso_cfg.weight = gif_iso_w
                    self._metrics["gif_isotropy_weight"] = gif_iso_w

        if (reg_cfg is not None and reg_cfg.any_enabled()
                and self.language_model is not None):
            h_last = getattr(self.language_model, "_last_hidden", None)
            if h_last is not None:
                # Per-sample CE for DAR's reweighting path. Computed
                # cheaply here from the already-materialised logits.
                with torch.no_grad():
                    B = logits.shape[0]
                    flat_l = logits.reshape(-1, logits.shape[-1])
                    flat_t = targets.reshape(-1)
                    per_tok = F.cross_entropy(flat_l, flat_t, reduction="none")
                    per_sample_ce = per_tok.reshape(B, -1).mean(dim=1)
                # Optional domain labels (DAR is no-op without them).
                domain_labels = None
                if self._domain_id_fn is not None:
                    try:
                        domain_labels = self._domain_id_fn(ids)
                    except Exception:
                        domain_labels = None
                reg_out = self.reg_controller.collect_aux(
                    h=h_last, lm_logits=logits,
                    per_sample_ce=per_sample_ce,
                    domain_labels=domain_labels,
                )
                total = total + reg_out["total"]
                # Publish per-intervention metrics for logging / brian ps.
                for key in ("dar", "pcc", "isotropy", "cmd"):
                    self._metrics[f"reg_{key}"] = float(
                        reg_out[key].detach().item())
                self._metrics["reg_total"] = float(
                    reg_out["total"].detach().item())
                self._metrics["reg_warmup"] = float(
                    reg_out["warmup_mult"].detach().item())
                self._metrics["chat_ratio"] = self.current_chat_ratio()
            elif not getattr(self, "_warned_missing_last_hidden", False):
                import sys
                print(
                    "[harness] warning: regularization enabled but "
                    "language_model._last_hidden is missing; PR2 aux losses "
                    "(DAR/PCC/Isotropy/CMD) are skipped this step.",
                    file=sys.stderr,
                )
                self._warned_missing_last_hidden = True

        if self.language_model is not None:
            stash_map = {
                "_last_pred_coding_loss": "pred_coding",
                "_last_world_loss":       "world",
                "_last_forward_loss":     "forward",
                "_last_motor_loss":       "motor",
                "_last_kl_world_loss":    "kl_world",
                "_last_novel_aux_loss":   "novel",
                "_last_cpc_loss":         "cpc",
                "_last_phi_loss":         "phi",
            }
            for stash_key, aux_key in stash_map.items():
                aux = getattr(self.language_model, stash_key, None)
                if aux is None or aux.numel() == 0:
                    continue
                w = self.total_loss_config.aux.scaled(aux_key, mat)
                total = total + w * aux

            # ── C3 reentry: NT-gated trunk loss (Jun 2026 surgery) ──
            # Predict the sensory hidden state from the motor hidden
            # state via the PC reentry probe's frozen-W projection;
            # add the residual to the LM loss so the trunk learns to
            # make the two populations mutually predictable. Gating:
            #   gate = max(0, 1 + 0.5·DA − 0.7·GABA)
            # so curiosity/reward (DA) strengthens the constraint and
            # cortical inhibition (GABA) relaxes it. Other NTs left
            # agnostic — the homeostat already touches them.
            pc_w = float(getattr(self.training_config,
                                 "pc_reentry_weight", 0.0))
            if pc_w > 0.0:
                pc_diff = self._compute_pc_reentry_loss(pc_w)
                if pc_diff is not None:
                    total = total + pc_diff

            # ── HPB Phase 3 — MSPCC trunk cascade (additive) ──
            # When training_config.mspcc is enabled, run a per-layer
            # MDRV-VBB free-energy on the stash of post-block outputs.
            # The deepest pair dominates the cascade by the geometric
            # weight schedule λ_ℓ = λ_0 · decay^((L-1)-ℓ); shallow
            # pairs contribute a smaller, decaying fraction. Composes
            # with the single-waist VBB above — both terms add to
            # `total`. NT-gating intentionally NOT applied here
            # because each layer pair lives at a different depth in
            # the cortical hierarchy and the NT gate is calibrated
            # for the bowtie waist specifically.
            mspcc_loss = self._compute_mspcc_loss(base_weight=0.0)
            if mspcc_loss is not None:
                total = total + mspcc_loss
        # Record the LM-portion so train_step can update MAT after the
        # backward pass without a second forward.
        self._last_lm_loss_value = float(loss_lm.detach().item())

        # ── Cortex fusion: distillation (slot A) + NT gating (slot C) ──
        # When the fusion path is active AND we stashed pre-fusion logits
        # during forward(), this is where:
        #   * the cortex teaches the trunk via KL distillation (A), and
        #   * the inhibition EMA is advanced toward the trunk-vs-cortex
        #     PPL gap (C — drives the *next* forward's α_eff).
        # All telemetry (kl, lambda, inhibition, alpha_eff) lands in
        # `_metrics` so the training-log line displays it.
        total = self._cortex_fusion_aux_step(total, targets)

        # ── Runtime metric registry update ──
        # Cheap runtime Phi proxy: per-token softmax entropy normalised
        # by ln(vocab). Computed on a SUBSAMPLE of tokens (64 per batch)
        # because materialising `softmax(logits) * log_softmax(logits)`
        # at (B, T, V) costs ~B·T·V·4 bytes — at B=8, T=2048, V=50257
        # that's 6.7 GiB per tensor (OOM on a 40 GiB A100 once the LM
        # activations are also resident).
        with torch.no_grad():
            B, T, V = logits.shape
            n_sample = min(64, B * T)
            idx = torch.randint(0, B * T, (n_sample,), device=logits.device)
            sampled = logits.detach().reshape(B * T, V).index_select(0, idx)
            logprobs = F.log_softmax(sampled, dim=-1)             # (n, V)
            ent = -(logprobs.exp() * logprobs).sum(dim=-1)        # (n,)
            phi_proxy = float(ent.mean()) / max(1e-6, math.log(self.vocab_size))
        # Surprise EMA (drives gene transcription triggers)
        if self._surprise_ema == 0.0:
            self._surprise_ema = self._last_lm_loss_value
        ema_alpha = 0.05
        surprise = abs(self._last_lm_loss_value - self._surprise_ema) / max(
            1e-6, self._surprise_ema)
        self._surprise_ema = (1 - ema_alpha) * self._surprise_ema + \
            ema_alpha * self._last_lm_loss_value

        self._metrics["mat"]      = float(mat)
        self._metrics["lm_loss"]  = float(self._last_lm_loss_value)
        self._metrics["surprise"] = float(surprise)
        self._metrics["phi"]      = phi_proxy

        # ── GeneticOrchestrator: add the pre-pass phi_loss (from the
        # _step_genetics_pre call earlier) to the total. The orchestrator
        # outputs are already in the forward graph this step so
        # `total.backward()` will accumulate gene-parameter gradients.
        if gcfg.enabled and pre_phi_loss is not None:
            total = total + gcfg.phi_weight * pre_phi_loss
            self._metrics["phi_loss"] = float(pre_phi_loss.detach())
        return total

    def attach_arch_genetics(self, arch_root) -> None:
        """Read `gene { ... }` / `protein { ... }` / `metric { ... }` blocks
        from arch.neuro and merge them into the runtime config.

        Called by train_dsl after the harness is built. Genes declared in
        DSL show up as `FixedGeneSpec`s in the orchestrator; declared
        proteins override `genetics.d_pay`; declared metrics populate
        `training_config.metric_exposures`.
        """
        try:
            from neuroslm.dsl.multifile import compile_folder
            ir = compile_folder(arch_root)
        except Exception:
            return
        # Override genetics config from the DSL
        if ir.genes or ir.proteins or ir.metrics:
            gcfg = self.training_config.genetics
            gcfg.enabled = True
            if ir.proteins:
                # First protein's payload_dim wins (one protein per orchestrator)
                gcfg.d_pay = ir.proteins[0].payload_dim
            self._dsl_genes = list(ir.genes)
        if ir.metrics:
            from neuroslm.dsl.training_config import MetricExpose
            self.training_config.metric_exposures = [
                MetricExpose(name=m.name, compute=m.compute,
                              expose_at=m.expose_at,
                              every_n_steps=m.every_n_steps)
                for m in ir.metrics
            ]

    def _ensure_genetics(self) -> None:
        """Lazily build the GeneticOrchestrator + TransmitterSystem on
        first use, sized from `training_config.genetics`.
        """
        if self._genetics_orch is not None:
            return
        from neuroslm.neurochem.genetics import (
            GeneticConfig, GeneticOrchestrator, default_fixed_genes,
            FixedGeneSpec, EFFECT_NT_BASELINE, EFFECT_RECEPTOR_TAU,
            EFFECT_NT_RELEASE_GAIN)
        from neuroslm.neurochem.transmitters import TransmitterSystem
        gcfg = self.training_config.genetics
        modules = list(gcfg.target_modules) or [
            "sensory", "thalamus", "gws", "pfc", "bg", "motor",
            "math_cortex", "reasoning_cortex",
        ]
        self._genetics_module_names = modules
        cfg = GeneticConfig(
            n_genes=gcfg.n_genes,
            d_pay=gcfg.d_pay,
            d_reg=4, d_tgt=max(8, len(modules)),
        )
        # Build fixed gene list: prefer DSL-declared genes; fall back to preset
        dsl_genes = getattr(self, "_dsl_genes", None) or []
        if dsl_genes:
            kind_map = {
                "nt_baseline_offset": EFFECT_NT_BASELINE,
                "receptor_tau_shift": EFFECT_RECEPTOR_TAU,
                "nt_release_gain":    EFFECT_NT_RELEASE_GAIN,
            }
            fixed = []
            for g in dsl_genes:
                if g.target not in modules:
                    continue
                effects = {kind_map[k]: v for k, v in g.effects.items()
                           if k in kind_map}
                fixed.append(FixedGeneSpec(
                    name=g.name, target_module=g.target,
                    constitutive=g.constitutive, trigger=g.trigger,
                    effects=effects,
                ))
        elif gcfg.fixed_genes_preset == "default":
            fixed = default_fixed_genes(modules)
        elif gcfg.fixed_genes_preset == "minimal":
            fixed = [FixedGeneSpec(
                name="gws_glu_floor", target_module="gws",
                constitutive=True,
                effects={EFFECT_NT_BASELINE: {"Glu": 0.05}})]
        else:
            fixed = []
        self._genetics_orch = GeneticOrchestrator(
            cfg, module_names=modules, fixed_genes=fixed,
            phi_target=gcfg.phi_target)
        # Put the orchestrator on the same device as the LM (best effort).
        try:
            dev = next(self.parameters()).device
            self._genetics_orch.to(dev)
        except StopIteration:
            pass
        self._transmitter_sys = TransmitterSystem(n_modules=len(modules))
        # Reset to batch=1; reset() is called from train_step on first batch.

    def _step_genetics_pre(self, batch_size: int, device):
        """PRE-pass genetic expression: orchestrator → set NT modulation
        on the cortex BEFORE forward, so gene-parameter gradients flow.

        Throttled by `genetics.update_every`: orchestrator forward + the
        full autograd graph through gene parameters only runs every Nth
        step. On skipped steps the cortex sees the most recent live
        modulation (cached) — bit-identical to running the orchestrator
        with last step's context.

        Diagnostics are throttled by `genetics.diagnostics_every` —
        the `.item()` syncs (gene_expr_mean/max/active_frac) are the
        single biggest per-step cost so we skip them in between.

        Returns the auxiliary `phi_loss` tensor on update steps; a
        zero scalar on skipped steps (so the total loss path is
        unchanged but no extra grads are accumulated).
        """
        self._ensure_genetics()
        if (self._transmitter_sys.level.shape[0] != batch_size
                or self._transmitter_sys.level.device != device):
            self._transmitter_sys.to(device)
            self._transmitter_sys.reset(batch_size, device)

        gcfg = self.training_config.genetics
        step = self._global_step
        do_update = (step % max(1, gcfg.update_every) == 0)

        if not do_update and self._last_orch_modulation is not None:
            # Skipped step: install a DETACHED copy of the last live
            # modulation. The cached tensor's autograd graph was freed
            # by the previous backward(); referencing it again would
            # try to traverse a freed graph ("Trying to backward through
            # the graph a second time"). Detaching strips the graph so
            # the cortex sees the values only — no grad flows through
            # genes on this step, which is the contract for skipped
            # steps (genes train every update_every steps).
            cached = self._last_orch_modulation.detach()
            if (self.language_model is not None
                    and hasattr(self.language_model, "set_nt_modulation")):
                self.language_model.set_nt_modulation(cached)
            return torch.zeros((), device=device)

        nt_levels = self._transmitter_sys.level
        prev_mat = float(self._metrics.get("mat", 0.0))
        prev_surprise = float(self._metrics.get("surprise", 0.0))
        prev_phi = float(self._metrics.get("phi", 0.0))
        surprise_t = torch.full((batch_size,), prev_surprise, device=device)
        mat_t      = torch.full((batch_size,), prev_mat,      device=device)
        phi_t      = torch.full((batch_size,), prev_phi,      device=device)
        ctx = self._genetics_orch.build_context(nt_levels, surprise_t, mat_t)
        out = self._genetics_orch(ctx, phi=phi_t)
        self._transmitter_sys.set_module_offsets(
            baseline_off=out["baseline_offsets"].detach(),
            tau_shift=out["tau_shifts"].detach(),
        )
        if (self.language_model is not None
                and hasattr(self.language_model, "set_nt_modulation")):
            self.language_model.set_nt_modulation(out["baseline_offsets"])
        self._last_orch_modulation = out["baseline_offsets"]
        # Diagnostics syncing is the big per-step CPU↔GPU cost; throttle
        # to `diagnostics_every` steps so most steps stay sync-free.
        if step % max(1, gcfg.diagnostics_every) == 0:
            diag = self._genetics_orch.diagnostics()
            for k, v in diag.items():
                key = k if k.startswith("gene_") else f"gene_{k}"
                self._metrics[key] = v
        return out["phi_loss"]

    def _compute_loss_from_logits(self, logits: torch.Tensor,
                                  targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss with optional label-smoothing + per-sample clip.

        Memory: the naïve `F.cross_entropy(logits.reshape(-1, V), ...)` path
        materialises an (N, V) gradient tensor during backward — at N=8192
        V=50257 fp32 that's 1.6 GiB, and label_smoothing > 0 also builds an
        (N, V) soft-target tensor, so peak backward memory is 4 such tensors
        (logits + grad + softmax + smoothed target) ≈ 6.4 GiB and OOMs the
        A100 once the trunk's activation graph is also resident.

        Fix: chunk the CE over the (B*T) dimension. Each chunk's backward
        only allocates an (CHUNK, V) gradient tensor — at CHUNK=1024 that's
        200 MB peak instead of 1.6 GiB.
        """
        batch, seq_len, vocab = logits.shape
        ls = self.training_config.label_smoothing
        clip = self.training_config.loss_clipping
        z_w = self.training_config.z_loss
        flat_logits = logits.reshape(-1, vocab)
        flat_targets = targets.reshape(-1)
        N = flat_logits.shape[0]
        # Threshold below which to skip chunking (small batches don't OOM).
        CHUNK = 1024

        def _per_token_ce() -> torch.Tensor:
            if N <= CHUNK:
                return F.cross_entropy(flat_logits, flat_targets,
                                        reduction="none", label_smoothing=ls)
            parts = []
            for s in range(0, N, CHUNK):
                e = min(s + CHUNK, N)
                parts.append(F.cross_entropy(
                    flat_logits[s:e], flat_targets[s:e],
                    reduction="none", label_smoothing=ls))
            return torch.cat(parts, dim=0)

        # ── Stage 10 OOD push: z-loss (PaLM/Gemma) ──
        # Penalises logit magnitude via α * logsumexp(logits)^2.
        # Chunked to avoid materialising a full (N, V) intermediate.
        def _z_loss() -> torch.Tensor:
            if N <= CHUNK:
                lse = torch.logsumexp(flat_logits, dim=-1)
                return (lse * lse).mean()
            acc = flat_logits.new_zeros(())
            for s in range(0, N, CHUNK):
                e = min(s + CHUNK, N)
                lse = torch.logsumexp(flat_logits[s:e], dim=-1)
                acc = acc + (lse * lse).sum()
            return acc / N

        if not clip.enabled:
            ce = _per_token_ce().mean()
        else:
            # Per-sample clipping path.
            per_token = _per_token_ce().reshape(batch, seq_len)
            per_seq = per_token.mean(dim=1)
            threshold = (per_seq.detach().median() * clip.factor).clamp(min=1e-8)
            clipped = torch.minimum(per_seq, threshold)
            ce = clipped.mean()

        if z_w > 0:
            ce = ce + z_w * _z_loss()
        return ce

    # ── Train step ───────────────────────────────────────────────────

    def train_step(self, ids: torch.Tensor, targets: torch.Tensor,
                   nt_levels: Optional[Dict[str, float]] = None) -> float:
        """One training step. Handles grad accumulation + clipping + optimizer.

        Returns the (un-scaled) per-call loss as a Python float, so the
        training loop can log it.
        """
        optimizer = self._ensure_optimizer()
        self.train()

        # Apply the LR schedule for the current global step (if configured)
        self._global_step += 1
        if self._sched_total is not None:
            lr = cosine_warmup_lr(
                step=self._global_step,
                base_lr=self.training_config.learning_rate,
                warmup=self._sched_warmup or 0,
                total=self._sched_total,
                min_lr_ratio=self._sched_min_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

        # ── GIF-2: OOD probe evaluation ──
        # Run the held-out OOD CE probe every N steps. The resulting
        # EMA is used by _cortex_fusion_aux_step to drive the
        # distillation λ-ramp and inhibition gate with a TRUE
        # generalisation signal instead of the pre-fusion EMA.
        if (self._gif is not None and self._gif.enabled
                and self._gif.ood_probe.should_eval(self._global_step)):
            if self.language_model is not None:
                device = ids.device if ids is not None else torch.device("cpu")
                ce = self._gif.ood_probe.evaluate(
                    self.language_model, device)
                self._metrics["gif_ood_probe_ce"] = ce

        loss = self.compute_loss(ids, targets, nt_levels=nt_levels)
        accum = max(1, self.training_config.grad_accum)

        # ── Stage 4 OOD push: NEMORI predictive-forgetting gate ──
        # Now MAT-phase-gated when mechanisms.nemori is declared:
        #     effective_floor = declared_floor × phase_gate(mat)
        # Falls back to the flat `nemori_floor` flag for back-compat.
        # Plus a hard warmup safety: never active before step 200 OR
        # while loss is still near the random-init ln(V) (early-init
        # bug fix from run 38608948).
        loss_f = float(loss.detach().item())
        mat = self.maturity.value()
        eff_nemori_floor = self.training_config.mechanisms.effective_nemori(
            mat, fallback=self.training_config.nemori_floor)
        nemori_active = (
            eff_nemori_floor > 0
            and self._last_lm_loss_value > 0
            and self._last_lm_loss_value < 0.85 * self.maturity.l_random
            and self._global_step > 200
        )
        if nemori_active:
            base = max(self._last_lm_loss_value, 1e-6)
            surprise = abs(loss_f - base) / base
            if surprise < eff_nemori_floor:
                self._nemori_skipped += 1
                self._last_lr = float(optimizer.param_groups[0]["lr"])
                self.maturity.update(loss_f)
                self._last_lm_loss_value = loss_f
                return loss_f

        scaled = loss / accum
        # fp16 needs gradient scaling; bf16 and fp32 don't.
        if self._grad_scaler is not None:
            self._grad_scaler.scale(scaled).backward()
        else:
            scaled.backward()

        self._accum_step += 1
        if self._accum_step >= accum:
            # ── PR2-G: CDGA gradient surgery ──
            # Runs AFTER grad accumulation finishes (so .grad holds the
            # summed g_train) and BEFORE clip + optimizer.step(). If
            # CDGA is enabled and an anchor function is registered, the
            # controller may overwrite .grad with the conflict-projected
            # g_aligned. On non-refresh steps this is a no-op.
            reg_cfg_late = getattr(self.training_config, "regularization", None)
            cdga_enabled = (
                reg_cfg_late is not None
                and reg_cfg_late.cdga.enabled
                and self._cdga_anchor_fn is not None
            )
            if cdga_enabled:
                # Refuse to mix CDGA with fp16 grad-scaling (the anchor
                # backward would interact poorly with .unscale_); emit
                # a single warning and skip surgery in that case.
                if self._grad_scaler is not None:
                    if not getattr(self, "_warned_cdga_fp16", False):
                        import sys
                        print(
                            "[harness] CDGA disabled: fp16 grad-scaling is "
                            "active. Re-run with bf16 or fp32 to enable "
                            "gradient surgery.",
                            file=sys.stderr,
                        )
                        self._warned_cdga_fp16 = True
                else:
                    def _anchor_callable():
                        # Clear g_train, run anchor forward+backward.
                        optimizer.zero_grad(set_to_none=True)
                        anchor_loss = self._cdga_anchor_fn(self)
                        anchor_loss.backward()
                        return anchor_loss

                    cdga_out = self.reg_controller.cdga.apply_surgery(
                        self, anchor_loss_fn=_anchor_callable)
                    if cdga_out["applied"]:
                        self._metrics["cdga_alpha"] = float(cdga_out["alpha"])
                        self._metrics["cdga_cosine"] = float(cdga_out["cosine"])
                        self._metrics["cdga_conflict"] = float(
                            cdga_out["conflict_coef"])
                    else:
                        self._metrics["cdga_alpha"] = float(cdga_out["alpha"])

            clip = self.training_config.grad_clip
            # clip_grad_norm_ returns the total norm *before* clipping —
            # capture it for native-format logging (gnorm).
            if self._grad_scaler is not None:
                self._grad_scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), clip if (clip and clip > 0) else 1e9)
                # ── Allostasis: advance HPA controller + damp LR ──
                # Read this step's loss + grad-norm into the load/cort
                # EMAs; multiply optimizer LR by lr_multiplier() to brake
                # under chronic stress. No-op when allostasis disabled.
                self._allostasis_step(loss=loss_f, grad_norm=float(gnorm))
                self._apply_lr_damping(optimizer)
                # BEMA wraps optimizer.step() with rollback detection.
                if self._bema is not None and self._bema.cfg.enabled:
                    self._grad_scaler.unscale_(optimizer)  # already done
                    info = self._bema.maybe_step(loss_f)
                    self._last_bema_info = info
                    self._grad_scaler.update()
                else:
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), clip if (clip and clip > 0) else 1e9)
                # ── Allostasis: advance HPA controller + damp LR ──
                # Same hook as the fp16 branch above — see _allostasis_step
                # and _apply_lr_damping for the math.
                self._allostasis_step(loss=loss_f, grad_norm=float(gnorm))
                self._apply_lr_damping(optimizer)
                if self._bema is not None and self._bema.cfg.enabled:
                    info = self._bema.maybe_step(loss_f)
                    self._last_bema_info = info
                else:
                    optimizer.step()
            self._last_gnorm = float(gnorm)
            optimizer.zero_grad(set_to_none=True)
            self._accum_step = 0

        self._last_lr = float(optimizer.param_groups[0]["lr"])
        # Tick the maturity tracker — drives the next step's aux weights.
        # Matches Brain's pattern: update_maturity() is called by train.py
        # AFTER the backward, with the (un-scaled, batch-level) LM loss.
        self.maturity.update(self._last_lm_loss_value)

        # ── Evolution heatmap hook (L2-wire) ─────────────────────
        # Fire AFTER optimizer.step() so .grad still holds the
        # backward we just ran. The hook is a no-op when disabled,
        # off-cadence, or never attached; failures are swallowed.
        if self._heatmap_hook is not None:
            self._heatmap_hook.step(self._global_step)

        return float(loss.detach().item())

    def attach_heatmap_hook(self, hook) -> None:
        """Attach an :class:`HeatmapHook` to fire after every train_step.

        The hook is invoked with the current global step *after* the
        backward + optimizer step have completed. The hook itself owns
        its cadence, alias map, publisher, and ``enabled`` flag — the
        harness side is purely a forward call.

        Pass ``None`` to detach.
        """
        self._heatmap_hook = hook

    def eval_step(self, ids: torch.Tensor, targets: torch.Tensor,
                  nt_levels: Optional[Dict[str, float]] = None) -> float:
        """Forward + loss with no grad, for validation."""
        self.eval()
        with torch.no_grad():
            return float(self.compute_loss(ids, targets, nt_levels=nt_levels).item())

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is not None:
            return self._optimizer
        opt_name = self.training_config.optimizer
        lr = self.training_config.learning_rate
        wd = self.training_config.weight_decay
        if opt_name == "adamw":
            # Stage 6: μP-aware AdamW param groups when mu_p_scaling is on.
            # Each parameter gets an LR multiplier based on its width so
            # representation-level updates stay O(1) as d_model scales.
            # At our 51M scale this is a near no-op; pays off at 1B+.
            llrd_factor = self.training_config.llrd
            if self.training_config.mu_p_scaling:
                groups = _mu_p_param_groups(self, base_lr=lr, wd=wd)
                self._optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
            elif llrd_factor > 0 and llrd_factor < 1.0:
                # Stage 11: layer-wise LR decay
                groups = _llrd_param_groups(self, base_lr=lr, wd=wd,
                                             factor=llrd_factor)
                self._optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
                print(f"[harness] LLRD enabled: factor={llrd_factor:.2f}, "
                      f"{len(groups)} param groups")
            else:
                self._optimizer = torch.optim.AdamW(
                    self.parameters(), lr=lr, weight_decay=wd
                )
        elif opt_name == "adafactor":
            # Lazy import — Adafactor isn't always installed
            try:
                from transformers.optimization import Adafactor
                self._optimizer = Adafactor(
                    self.parameters(),
                    lr=lr, scale_parameter=False, relative_step=False,
                )
            except ImportError:
                # Fallback to AdamW with a warning printed once
                print("[harness] adafactor not available; falling back to AdamW")
                self._optimizer = torch.optim.AdamW(
                    self.parameters(), lr=lr, weight_decay=wd
                )
        else:
            raise ValueError(f"unsupported optimizer {opt_name!r}")

        # Item 6: tiny NT-coupling Parameters (DrivenNTSystem.W_param)
        # live OUTSIDE the harness module tree (the observer is a
        # sidecar, not a submodule), so `self.parameters()` doesn't
        # pick them up. Register them as a separate param group so the
        # optimiser actually updates them. Same LR/WD as the main
        # group — there are only 35 scalars, so the choice is moot.
        extra = list(getattr(self, "_extra_trainable_params", []) or [])
        if extra:
            already = set()
            for g in self._optimizer.param_groups:
                for p in g["params"]:
                    already.add(id(p))
            fresh = [p for p in extra if id(p) not in already]
            if fresh:
                self._optimizer.add_param_group({
                    "params": fresh, "lr": lr, "weight_decay": wd,
                })
                print(
                    f"[harness] Item 6 / extras: added {len(fresh)} "
                    f"extra trainable Parameter(s) to optimiser as a "
                    f"new param group"
                )

        # Stage 3 OOD push: wrap the optimizer with BEMA when enabled.
        rw = self.training_config.bema_rollback_window
        if rw > 0:
            bema_cfg = BEMAConfig(
                enabled=True,
                rollback_window=rw,
                snapshot_every=self.training_config.bema_snapshot_every,
                cooldown=self.training_config.bema_cooldown,
                max_snapshots=4,
            )
            self._bema = BEMAController(self, self._optimizer, bema_cfg)
        return self._optimizer

    # ── Parameter scopes (p3 gradient isolation) ────────────────────

    def apply_param_scopes(self, scopes) -> None:
        """Apply declarative gradient isolation from `param_scope` blocks.

        For every scope with `gradient == "detached_from_main_loss"`, the
        parameters of its populations are frozen (`requires_grad=False`),
        so the main LM loss can't update them — exactly the p3 fix. A
        population in a `normal` scope keeps its params trainable.

        `scopes` is a list of `ParamScope` (from param_scopes.parse_*).
        Populations not named in any scope are left untouched.
        """
        for scope in scopes:
            freeze = (scope.gradient == "detached_from_main_loss")
            for pop_name in scope.populations:
                submodule = getattr(self.circuit, pop_name, None)
                if submodule is None:
                    # Population named in a scope but not in this circuit —
                    # skip silently (architectures can declare scopes that
                    # reference optional populations).
                    continue
                for p in submodule.parameters():
                    p.requires_grad = not freeze

    # ── Checkpoint ───────────────────────────────────────────────────

    # Subtrees whose source-of-truth lives outside the checkpoint and
    # therefore must NOT be serialised. Adding 1+ GB of frozen HF
    # weights on every save blew past GitHub LFS's 2 GiB per-file
    # limit on instance 41049651 (2026-06-15). The experts are
    # re-attached by ``_build_multi_cortex`` on resume — from the
    # local HF cache, which is byte-for-byte identical.
    # Regression-pinned by
    # ``tests/test_checkpoint_excludes_frozen_experts.py``.
    _CKPT_EXTERNAL_PREFIXES: tuple = (
        "multi_cortex.experts.",
    )

    @classmethod
    def _is_external_key(cls, key: str) -> bool:
        """True iff ``key`` points into a subtree whose weights are
        sourced externally (HuggingFace cache, etc.) and so must be
        excluded from save and tolerated as missing on load."""
        return any(key.startswith(p) for p in cls._CKPT_EXTERNAL_PREFIXES)

    def _persistable_state_dict(self) -> Dict[str, Any]:
        """``state_dict()`` with externally-sourced subtrees filtered
        out. Used by ``save_checkpoint`` to keep the on-disk payload
        below GitHub LFS's 2 GiB single-file ceiling."""
        return {
            k: v for k, v in self.state_dict().items()
            if not self._is_external_key(k)
        }

    def save_checkpoint(self, path: str, step: int = 0,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist model + optimizer state + step. Mirrors Brain's format
        loosely enough that a future merger can interoperate.

        Creates the parent directory if it does not already exist —
        the H24+ per-run subdir layout
        (``lfs_checkpoints/<RUN_ID>_<GIT>_<ARCH>/step<N>.pt``) is only
        materialised on the first save, and ``torch.save`` blows up
        otherwise. Regression-pinned by
        ``tests/test_checkpoint_path_layout.py``
        ::``TestCheckpointDirLayout::test_save_checkpoint_writes_into_run_subdir``.

        Externally-sourced subtrees (frozen HF experts, see
        ``_CKPT_EXTERNAL_PREFIXES``) are filtered out of the saved
        state-dict; they're re-attached on resume by
        ``_build_multi_cortex`` from the local HuggingFace cache. This
        keeps the on-disk file under GitHub LFS's 2 GiB per-file hard
        limit and saves ~1 GB per checkpoint in the production
        ``rcc_bowtie_30m_p4`` config. Regression-pinned by
        ``tests/test_checkpoint_excludes_frozen_experts.py``.
        """
        import os as _os
        parent = _os.path.dirname(_os.fspath(path))
        if parent:
            _os.makedirs(parent, exist_ok=True)
        payload = {
            "step": step,
            "model": self._persistable_state_dict(),
            "vocab_size": self.vocab_size,
            "d_sem": self.d_sem,
            "sink_population": self.sink_population,
        }
        if self._optimizer is not None:
            payload["optimizer"] = self._optimizer.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> int:
        """Load model + optimizer state. Returns the saved step.

        Rebuilds lazy submodules (``_genetics_orch``,
        ``_transmitter_sys``) *before* ``load_state_dict`` when the
        checkpoint contains their keys. Without this, resume from any
        real production run blows up with ``Unexpected key(s)``
        because those modules are normally built on the first
        ``_step_genetics_pre`` call and a freshly-instantiated
        harness has them as ``None``. Regression-pinned by
        ``tests/test_checkpoint_path_layout.py``
        ::``TestStateDictRoundTripAfterTraining``.

        Loads with ``strict=False`` but enforces a narrow tolerance:
        missing keys are only allowed when they belong to an
        externally-sourced subtree (``_CKPT_EXTERNAL_PREFIXES`` — i.e.
        frozen HF experts that the live harness already owns from
        ``_build_multi_cortex``). Any other missing key raises so
        accidental cross-architecture loads stay loud. Unexpected
        keys (the back-compat case where an OLD checkpoint still
        carries expert weights) are silently dropped — the live
        experts are the source of truth on resume.
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        model_keys = payload["model"].keys()
        # Lazy submodule rebuild: a fresh harness has these as None,
        # but production checkpoints carry their state. Build them
        # *exactly* the same way ``_step_genetics_pre`` does so the
        # state_dict shapes match.
        if (self._genetics_orch is None
                and any(k.startswith("_genetics_orch.") for k in model_keys)):
            self._ensure_genetics()
        # ``_ensure_genetics`` builds BOTH orchestrator + transmitter,
        # so the second condition is normally redundant. Keep it as a
        # belt-and-braces guard for future divergence.
        if (self._transmitter_sys is None
                and any(k.startswith("_transmitter_sys.") for k in model_keys)):
            self._ensure_genetics()
        # Transmitter buffers (``level``, ``vesicles``,
        # ``module_baseline_off``, ``module_tau_shift``) carry a
        # leading ``batch`` dim that is set by ``reset(batch_size,
        # device)`` on first training step. A fresh
        # ``_ensure_genetics()`` call constructs them at batch=1, but
        # production checkpoints were saved with batch=2 (or higher)
        # → state_dict shape mismatch on load. Infer the batch dim
        # from the saved tensor and reset BEFORE ``load_state_dict``.
        if (self._transmitter_sys is not None
                and "_transmitter_sys.level" in payload["model"]):
            saved_bs = int(payload["model"]["_transmitter_sys.level"].shape[0])
            cur_bs = int(self._transmitter_sys.level.shape[0])
            if saved_bs != cur_bs:
                self._transmitter_sys.reset(saved_bs, device)
        # strict=False so the new (filtered) format loads cleanly even
        # though the live harness has the frozen-expert submodules
        # attached. Then narrow the tolerance: any MISSING key that is
        # NOT under an external prefix is still a hard error — that's
        # the cross-architecture / wrong-checkpoint guard.
        result = self.load_state_dict(payload["model"], strict=False)
        # ``IncompatibleKeys`` is a named tuple
        # ``(missing_keys, unexpected_keys)``; pre-PyTorch-1.5 returns
        # ``None``. Guard for both.
        if result is not None:
            real_missing = [
                k for k in getattr(result, "missing_keys", [])
                if not self._is_external_key(k)
            ]
            if real_missing:
                raise RuntimeError(
                    "load_checkpoint: missing keys outside the allow-listed "
                    f"external subtree (first 5): {real_missing[:5]}. "
                    "This usually means the checkpoint was produced by a "
                    "different architecture than the live harness."
                )
            # Unexpected keys (e.g. old-format payload carrying full
            # expert weights) are intentionally ignored: the live
            # experts attached at init are the source of truth.
        if "optimizer" in payload and self._optimizer is not None:
            self._optimizer.load_state_dict(payload["optimizer"])
        elif self._optimizer is not None:
            # Checkpoint carries no optimizer state — almost certainly
            # an HF-Hub resume (we strip the Adam moments from cloud
            # uploads by default, see ``checkpoint_push.py``). The
            # optimiser will reinit from zero, which means the first
            # ~500 steps will look like an LR-warmup blip while Adam's
            # 2nd-moment EMA rebuilds. The weights themselves are
            # exactly the trained ones, so eval metrics stay valid —
            # only the gradient-step shape suffers transiently.
            print(
                "[harness] load_checkpoint: no optimizer state in payload "
                "— Adam moments will reinit from zero (expect ~500-step "
                "warmup-shape loss blip while the 2nd-moment EMA rebuilds; "
                "weights themselves are untouched).",
                flush=True,
            )
        return int(payload.get("step", 0))

    # ── Introspection (for train.py compatibility) ──────────────────

    def topology_summary(self) -> str:
        """Human-readable topology string. Mirrors Brain.topology_summary().

        Breaks the parameter count down three ways so the LFS
        checkpoint-size question never has to be asked again:

          * ``total``     — every tensor in the forward graph
                            (trunk + frozen HF experts).
          * ``trainable`` — parameters that actually receive gradients
                            (the trunk). This is the number that matters
                            for "model capacity that's being learned".
          * ``saved``     — parameters that ``save_checkpoint`` actually
                            writes to disk (= trainable + any frozen-but-
                            persisted heads). For the multi-cortex MoE
                            setup this should equal ``trainable`` because
                            the frozen HF experts are excluded via
                            :data:`_CKPT_EXTERNAL_PREFIXES`.

        The estimated checkpoint size assumes float32 (4 bytes/param)
        which is the BRIANHarness default; bf16 mixed-precision training
        still saves master weights in fp32 so the estimate stays honest.
        """
        all_params = list(self.parameters())
        n_total     = sum(p.numel() for p in all_params)
        n_trainable = sum(p.numel() for p in all_params if p.requires_grad)
        # Computing the persistable count walks the full state_dict
        # once; cheap on a 30 M-trunk and a one-off operation.
        n_saved     = sum(t.numel() for t in self._persistable_state_dict().values())
        ckpt_mb     = n_saved * 4 / (1024 * 1024)  # fp32 → MB
        n_pops      = sum(1 for _ in self.circuit.children())
        return (
            f"BRIANHarness:\n"
            f"  vocab_size = {self.vocab_size}\n"
            f"  d_sem      = {self.d_sem}\n"
            f"  parameters = {n_total:,}  "
            f"(trainable {n_trainable:,} · frozen {n_total - n_trainable:,})\n"
            f"  checkpoint = {n_saved:,} params ≈ {ckpt_mb:.1f} MB (fp32) "
            f"— frozen HF experts excluded\n"
            f"  circuit populations = {n_pops}\n"
            f"  sink population = {self.sink_population}\n"
            f"  loss clipping = {self.training_config.loss_clipping.enabled} "
            f"(factor={self.training_config.loss_clipping.factor})\n"
            f"  optimizer = {self.training_config.optimizer} "
            f"(lr={self.training_config.learning_rate})"
        )
