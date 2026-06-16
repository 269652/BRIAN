# -*- coding: utf-8 -*-
"""`training { ... }` block parser — pipeline config from arch.neuro.

The training block lets an architecture declare *how it expects to be
trained* alongside *what it is*. The BRIAN harness reads the result and
applies the requested transformations on top of the bare DSL circuit.

Supported fields:
    loss_clipping  — per-sample Huber-style robust loss aggregation
    quantization   — int8/int4 post-training or QAT (PyTorch dynamic)
    grad_accum     — gradient accumulation steps
    optimizer      — adamw | adafactor (matches existing presets)
    learning_rate  — base LR
    weight_decay   — AdamW weight decay
    grad_clip      — global gradient norm clip
    label_smoothing — cross-entropy label smoothing
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Sub-config dataclasses ─────────────────────────────────────────────

@dataclass
class LossClippingConfig:
    """Per-sample loss clipping (p4 fix, declarative version).

    method = "per_sample":   each sequence's per-token loss is clipped at
                              `factor * batch_median` before averaging,
                              so one outlier sequence can't dominate the
                              gradient.
    factor = 3.0:             3× median is the GPT-3 / Cerebras default.
    """
    enabled: bool = False
    method: str = "per_sample"
    factor: float = 3.0


@dataclass
class QuantizationConfig:
    """Optional post-training or training-time quantization."""
    enabled: bool = False
    bits: int = 8


# ── Phase-gated mechanism configs (MAT-coupled ramp-in) ─────────────────

@dataclass
class PhaseGate:
    """MAT-window phase gate: gate(mat) = 0.5*(1+tanh((mat-center)/width)).

    Lifted from Brain's `_phase_gate` so each mechanism activates only
    once the model has matured past `center`. Lets us ship mechanisms
    that hurt early training (dropout, Tonnetz, NEMORI) without
    capping the train PPL floor — they only kick in after the model
    has reached the "fitting" phase.
    """
    center: float = 0.0   # MAT level at which gate crosses 0.5 (off below)
    width: float = 0.10   # transition sharpness

    def value(self, mat: float) -> float:
        """Return gate value at the given MAT. 0 = off, 1 = full strength."""
        import math
        x = (float(mat) - float(self.center)) / max(1e-6, float(self.width))
        return 0.5 * (1.0 + math.tanh(x))


@dataclass
class MechanismsConfig:
    """All MAT-phase-gated mechanism knobs. Each is None when not declared.

    The harness reads each mechanism's `strength × gate.value(mat)` at
    every step. A mechanism with no entry behaves as the legacy
    constant-strength flag (dropout, pct_trunk, etc.) which still
    works for back-compat.
    """
    # Mechanism: (strength multiplier, phase gate). All optional —
    # when None, the legacy flat-strength flag controls the mechanism.
    dropout:    Optional[Tuple[float, PhaseGate]] = None
    pct_trunk:  Optional[Tuple[float, PhaseGate]] = None
    tonnetz:    Optional[Tuple[int, int, PhaseGate]] = None   # (period, bandwidth, gate)
    nemori:     Optional[Tuple[float, PhaseGate]] = None
    bema:       Optional[Tuple[int, PhaseGate]] = None

    def effective_dropout(self, mat: float, fallback: float) -> float:
        if self.dropout is None:
            return fallback
        strength, gate = self.dropout
        return strength * gate.value(mat)

    def effective_pct_trunk(self, mat: float, fallback: float) -> float:
        if self.pct_trunk is None:
            return fallback
        strength, gate = self.pct_trunk
        return strength * gate.value(mat)

    def effective_nemori(self, mat: float, fallback: float) -> float:
        if self.nemori is None:
            return fallback
        floor, gate = self.nemori
        return floor * gate.value(mat)

    def effective_tonnetz_period(self, mat: float, fallback: int) -> int:
        if self.tonnetz is None:
            return fallback
        period, _bw, gate = self.tonnetz
        # Tonnetz can't be "partial" — toggle on once gate > 0.5
        return period if gate.value(mat) > 0.5 else 0

    def effective_bema(self, mat: float, fallback: int) -> int:
        if self.bema is None:
            return fallback
        rw, gate = self.bema
        return rw if gate.value(mat) > 0.5 else 0


# ── Pass marks (declarative early-exit conditions) ─────────────────────

@dataclass
class PassMark:
    """One early-exit condition.

    `name` identifies the rule in logs. The other fields encode a
    metric-based check; the harness evaluates them at each step and
    triggers early exit when any condition fires.

    Three flavors:
      1. Threshold-at-step:  `metric` ∈ {train_ppl, ood_ppl, train_loss}
                              at `at_step` must be ≤ `max` (or ≥ `min`).
      2. Stability:          `metric` must be stable (relative range
                              < `tol`) over the last `window` steps.
      3. Falling-trend:      `metric` must be falling (newest <
                              window_min × (1 + `tol`)) over the last
                              `window` steps.
    """
    name: str = ""
    metric: str = "train_ppl"
    at_step: int = 0           # 0 = no step constraint (any time)
    max: Optional[float] = None
    min: Optional[float] = None
    window: int = 0            # >0 enables stability / trend check
    tol: float = 0.02          # relative tolerance for stability
    trend: str = ""            # "" | "stable" | "falling"
    action: str = "exit"       # only "exit" supported now
    # Minimum number of data points (samples within the window) required
    # before the trend check is allowed to fire. Defaults to 4 so noisy
    # OOD eval blips (50-window WikiText is ±3% noisy) can't trigger an
    # early-exit on a single bump. Set higher (e.g. 6) for stricter
    # stability requirements.
    min_evals: int = 4


@dataclass
class PassMarksConfig:
    """All declared early-exit rules. Empty list = no early exit."""
    rules: List[PassMark] = field(default_factory=list)


@dataclass
class HardwareConfig:
    """`hardware { ... }` block — declares the target hardware envelope
    for a training run. The deploy script reads these to filter vast.ai
    offers; the harness reads them to pick DDP/FSDP wrapping.

    Fields:
        gpu_name:        e.g. "A100_SXM4" | "H100_SXM" | "RTX_4090"
        num_gpus:        per instance (1 = single-GPU)
        min_gpu_mem_gib: per-GPU minimum memory
        min_reliability: vast.ai reliability filter (0..1)
        min_inet_mbps:   minimum inet_down for clone speed
        dist_strategy:   "single" | "ddp" | "fsdp"
                          DDP works up to ~1B params; FSDP for >1B.
        precision:       "fp32" | "bf16" | "fp16"
    """
    gpu_name: str = "A100_SXM4"
    num_gpus: int = 1
    min_gpu_mem_gib: int = 0
    min_reliability: float = 0.995
    min_inet_mbps: int = 200
    dist_strategy: str = "single"
    precision: str = "bf16"


@dataclass
class ScaleVariant:
    """One entry in a `scales { ... }` block — a parameter scale of the
    same architecture. Mirrors the BrainConfig dim knobs.

    Fields:
        name:        identifier (e.g. "30m_p4", "1b", "7b")
        d_model:     trunk width
        depth:       number of transformer blocks
        n_heads:     attention heads
        n_kv_heads:  KV heads for GQA (defaults to n_heads)
        max_ctx:     context length
        batch_size:  per-GPU batch
        seq_len:     sequence length per sample
        grad_accum:  gradient accumulation steps
        learning_rate: μP-friendly LR; defaults inherited from training{}
        approx_params: rough param count, informational
        hardware:    optional per-scale hardware override (else inherits)
    """
    name: str = ""
    d_model: int = 256
    depth: int = 6
    n_heads: int = 4
    n_kv_heads: int = 0
    max_ctx: int = 2048
    batch_size: int = 16
    seq_len: int = 2048
    grad_accum: int = 1
    learning_rate: float = 0.0
    approx_params: str = ""
    hardware: Optional["HardwareConfig"] = None


@dataclass
class ScalesConfig:
    """All declared scales + the active selection."""
    variants: Dict[str, ScaleVariant] = field(default_factory=dict)
    # Default scale when the env-var SCALE isn't set
    default: str = ""


@dataclass
class MetricExpose:
    """`metric <name> { expose_at: [...] }` — where a computed metric is
    materialised into the control-flow node.

    Used by the codegen + harness to keep the per-step metric-publish
    overhead bounded: only compute + publish each metric at the points
    that actually consume it.

    `compute`:
        "lm_logits"   — entropy of the per-token softmax (default Phi proxy)
        "iit_proxy"   — full IIT-4 graph proxy (heavy; once per N steps)
        "external"    — supplied by the caller; harness does nothing
    `expose_at`:
        list of node tags where the metric is published. Recognised tags:
        "lm_head", "gws", "pfc", "trunk", "gene_trigger", "all".
        Empty list = expose ONLY at the loss site (cheapest).
    """
    name: str = "phi"
    compute: str = "lm_logits"
    expose_at: List[str] = field(default_factory=list)
    every_n_steps: int = 1


@dataclass
class GeneticsConfig:
    """`genetics { ... }` block — enables the GeneticOrchestrator.

    When `enabled`, the harness builds a `GeneticOrchestrator` at
    construction time, runs it inside `compute_loss` each step, and adds
    its `phi_loss` (= -Phi) as an auxiliary loss with weight `phi_weight`.

    `fixed_genes_preset`:
      "default" — math/reasoning 5HT booster, PFC DA on surprise, GWS Glu floor
      "none"    — only learnable genes; no hand-curated overlays
      "minimal" — only the GWS glu floor (safest, single gene)
    """
    enabled: bool = False
    n_genes: int = 32
    d_pay: int = 16
    phi_weight: float = 0.01           # multiplier on -Phi added to total loss
    phi_target: float = 0.30           # target Phi (informational only)
    fixed_genes_preset: str = "default"
    target_modules: List[str] = field(default_factory=list)
    # Gene-selection objective. "phi" adds -Phi*phi_weight as aux loss
    # (genes train toward higher integrated information). "lm_loss"
    # forces phi_weight=0 so genes train ONLY through the cortex →
    # LM-loss gradient path — they're selected for whatever reduces
    # next-token prediction error. "ppl" is an alias for "lm_loss".
    optimize_for: str = "phi"
    # ── Performance knobs ──
    # update_every:   orchestrator runs every N training steps; cached
    #                 output is reused on skipped steps. ~25% load at
    #                 N=4 recovers most of the throughput hit while
    #                 still giving genes a smoothed training signal.
    # diagnostics_every: gene-expr .item() syncs are GPU→CPU sync
    #                 points; publishing every step throttles A100s.
    #                 Default 50 ⇒ refresh diagnostics 2% of steps.
    update_every: int = 4
    diagnostics_every: int = 50


@dataclass
class AllostasisConfig:
    """`allostasis { ... }` block — synthetic HPA axis (slow homeostat).

    Adds two slow state variables to the neuromod stack:

      * ``load(t)`` — fast EMA (τ ≈ ``1/load_ema_alpha``) of multi-modal
        stress signals (NE/GABA above tonic baseline, loss volatility,
        grad-norm spikes), saturated in ``[0, 1]``.

      * ``cort(t)`` — slow EMA (τ ≈ ``1/cort_ema_alpha``) of ``load``.
        This is the "cortisol" integrator that distinguishes **acute**
        stress (a single bad batch — load spikes briefly, cort barely
        moves) from **chronic** stress (sustained — cort climbs and
        dampers engage).

    and three multiplicative effectors, each ``mult = 1 − γ_* · cort``
    clamped to ``[0, 1]``:

      * ``ne_multiplier()``      — clips NE released by the genetic
        orchestrator (closes the LC positive-feedback loop).
      * ``trophic_multiplier()`` — multiplies BDNF growth signal
        (suppresses sprouting during sustained stress).
      * ``lr_multiplier()``      — multiplies optimizer LR right
        before ``optimizer.step()`` (consolidate, don't update).

    The 10× ratio between ``load_ema_alpha`` and ``cort_ema_alpha`` is a
    textbook stress-physiology default: LC noradrenergic responses are
    ~10× faster than HPA cortisol responses.

    Disabled by default — ``enabled=False`` ⇒ no controller built, no
    multipliers applied, no telemetry published. Back-compat with every
    pre-allostasis arch.neuro is bit-identical.
    """
    enabled: bool = False

    # ── EMA time constants ──
    # load_ema_alpha = 0.10 ⇒ ~10-step memory.
    # cort_ema_alpha = 0.02 ⇒ ~34-step half-life (the ~5× HPA rule).
    # Physiological LC-NE vs HPA-cortisol time-scale ratio is ~5-15×;
    # 0.02 sits at the responsive end of that range and matches the
    # integration window we need to damp the kind of runaway observed
    # in the rcc_bowtie_30m_p4 run: NE saturated over ~40 steps after
    # ~400 steps of mild stress building up. Cort integrates enough
    # of that history (by ~step 460) to bite when it matters.
    load_ema_alpha: float = 0.10
    cort_ema_alpha: float = 0.02

    # ── Stress-source weights ──
    # Mixed additively into a saturating sum ∈ [0, 1]. Weights need
    # not sum to 1 (the sum is clipped). Tune to expose the channels
    # most informative for your run.
    w_ne:   float = 0.30
    w_gaba: float = 0.20
    w_loss: float = 0.30
    w_grad: float = 0.20

    # ── Stress thresholds ──
    # Baselines above which a transmitter level counts as "stressed."
    # ne_baseline=0.25 sits just above NE_DEFAULTS baseline (0.15) so
    # tonic firing does NOT register as stress; gaba_baseline=0.20
    # similarly clears the GABA tonic baseline (0.10).
    ne_baseline:       float = 0.25
    gaba_baseline:     float = 0.20
    # Grad-norm above this is a "spike". Matches the threshold the
    # existing `Homeostasis` uses to boost GABA bias (5.0).
    grad_norm_ceiling: float = 5.0

    # ── Effector kill switches ──
    # Lets operators engage individual dampers without touching the
    # others (e.g. enable LR damping but leave NE alone for ablation).
    suppress_ne:      bool = True
    suppress_trophic: bool = True
    suppress_lr:      bool = True

    # ── Effector damping strengths ──
    # mult = 1 - γ · cort. At cort=1: ne ⇒ 30%, trophic ⇒ 0%, lr ⇒ 50%.
    # gamma_trophic=1.0 is intentional: during sustained crisis the
    # network should STOP growing entirely (BDNF is downregulated by
    # chronic cortisol in real brains — Smith & Vale 2006).
    gamma_ne:      float = 0.7
    gamma_trophic: float = 1.0
    gamma_lr:      float = 0.5


@dataclass
class ExpertSpec:
    """One row of the ``multi_cortex.experts: [...]`` roster.

    Each spec describes a HuggingFace causal-LM expert that contributes
    logits to the trunk's mixture. The expert's **native pretrained
    head** is what produces logits — this is the architectural difference
    from the legacy ``GPT2SubCortex`` path, which threw the head away
    and replaced it with a random Xavier projection (causing initial CE
    to sit at the ``ln(vocab_size)`` ceiling; see
    ``scripts/diagnose_cortex_init.py``).

    Fields:
      id      — HuggingFace model id (``gpt2-medium``,
                ``Qwen/Qwen2.5-0.5B``, ``microsoft/CodeGPT-small-py``, …)
      domain  — routing key (must be unique within a roster)
      freeze  — when ``True`` (default), every expert parameter has
                ``requires_grad=False``. Distillation gradients still
                flow into the trunk; the expert itself is the teacher.
      weight  — multiplicative prior on the router output for this
                expert. ``1.0`` (default) keeps the router untouched;
                values > 1 over-weight a domain (e.g. while a particular
                expert is being introduced to the ensemble).
    """
    id: str
    domain: str
    freeze: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(
                f"ExpertSpec.id must be a non-empty string, got {self.id!r}"
            )
        if not self.domain or not isinstance(self.domain, str):
            raise ValueError(
                f"ExpertSpec.domain must be a non-empty string, "
                f"got {self.domain!r}"
            )
        if self.weight < 0:
            raise ValueError(
                f"ExpertSpec.weight must be >= 0, got {self.weight}"
            )


@dataclass
class MultiCortexConfig:
    """`multi_cortex { ... }` block — wires the Multi-Trunk-V2 ensemble.

    When `enabled`, the harness builds a `MultiCortexEnsemble`
    (`neuroslm.cortex`) at construction time and routes the trunk's
    semantic state through it. With `weights="gpt2"` each sub-cortex
    loads a HuggingFace GPT-2 family checkpoint (general:gpt2,
    math:gpt2-medium, code:distilgpt2, chat:gpt2 by default). With
    `weights="stub"` random `StubSubCortex`s are built — used for unit
    tests and offline CI runs that must not hit the HF Hub.

    Fields:
      enabled              — master switch.  False ⇒ ensemble never built.
      n_cortices           — must equal `len(domains)` (validated).
      domains              — ordered list of cortex names (the keys the
                              router gates over).
      weights              — "stub" or "gpt2".  "gpt2" triggers HF
                              download via `build_gpt2_ensemble`.
      freeze_weights       — when True, GPT-2 backbone params have
                              requires_grad=False; only the projection
                              adapters + router train.
      lexical_bias_weight  — multiplier on the DomainLexicon prior added
                              to router logits before softmax.
      bema_tau             — Bregman-EMA smoothing constant on routing
                              weights (0.0 = no smoothing, 1.0 = frozen).
      router_d_model       — hidden width of the ThalamicRouter MLP.
      fusion_mode          — how the ensemble feeds the LM head:
                              * "logits_mixture" (default when enabled):
                                  final_logits = (1-α) · lm_logits + α · (cortex_h @ embed.T)
                                where α = sigmoid(learnable mix scalar).
                                This makes the pretrained cortex features
                                actually drive the loss from step 0.
                              * "off": build the ensemble (for routing
                                telemetry / aux objectives) but do NOT
                                touch the LM head. Legacy behaviour.
      fusion_init          — initial value of α (the mixing weight).
                              Defaults to 0.5. Set to 0.0 to ramp the
                              cortex in over training (start unchanged
                              from the LM-only baseline). Set higher to
                              trust the pretrained features more.
    """
    enabled: bool = False
    n_cortices: int = 4
    domains: List[str] = field(
        default_factory=lambda: ["math", "code", "chat", "general"]
    )
    # "stub" is the safe default — no network access, no HF download.
    # arch.neuro must explicitly opt-in to "gpt2" to pull weights.
    weights: str = "stub"
    # New (preferred) per-expert roster. When set (non-None), it
    # supersedes the ``weights`` shorthand and ``domains`` / ``n_cortices``
    # are auto-derived from the roster. Each expert's pretrained LM head
    # is used directly — no random projection chain.
    experts: Optional[List[ExpertSpec]] = None
    # When ``experts`` is set, this is the BPE tokenizer id used to
    # decode trunk ids into text so cross-tokenizer experts can
    # re-encode and bridge their vocab back to trunk-vocab space. Must
    # match the tokenizer the dataset was tokenised with.
    trunk_tokenizer: str = "gpt2"
    freeze_weights: bool = True
    lexical_bias_weight: float = 2.0
    bema_tau: float = 0.5
    router_d_model: int = 256
    # Logits-fusion mode + initial mixing weight. See class docstring.
    fusion_mode: str = "logits_mixture"
    fusion_init: float = 0.5

    # ── A: KL-distillation aux loss (trunk learns FROM cortex) ──
    # Hinton 2015 style: L_total += λ_t · T² · KL(softmax(cortex.detach()/T)
    # || softmax(lm/T)). The trunk distills the cortex's full output
    # distribution, not just argmax targets, so the ~500M of pretrained
    # GPT-2 knowledge actually transfers into the trunk's parameters.
    # `.detach()` on the teacher means the KL term contributes NO gradient
    # to the cortex — it's a one-way teacher. λ_t is a piecewise-linear
    # ramp in the EMA `gap = lm_loss_ema - cortex_loss_ema`:
    #   gap ≤ floor                → λ = 0           (trunk has caught up)
    #   floor < gap < ceiling      → linear interp
    #   gap ≥ ceiling              → λ = lambda_max  (trunk much worse)
    distillation_enabled: bool = False
    distillation_lambda_max: float = 1.0
    distillation_temperature: float = 4.0
    distillation_gap_floor: float = 0.1     # nats: below = no distill
    distillation_gap_ceiling: float = 2.0   # nats: above = full distill

    # ── C: NT-mediated α-gating (cortex retires as trunk improves) ──
    # The harness tracks `cortex_inhibition_level ∈ [0, 1]` as a slow
    # EMA driven by `cortex_loss_ema - lm_loss_ema` (positive when trunk
    # has overtaken cortex). The forward path uses
    #   α_eff = sigmoid(cortex_mix_logit) · (1 - cortex_inhibition_level)
    # so the cortex's contribution shrinks smoothly as the trunk outgrows
    # it. Once inhibition saturates near 1, the cortex contributes ~0 and
    # inference can skip the cortex forward entirely (~20× FLOP savings
    # on the rcc_bowtie_30m_p4 configuration).
    inhibition_enabled: bool = False
    inhibition_ema_alpha: float = 0.05
    inhibition_temperature: float = 1.0     # nats of gap → full inhibition

    # ── Item 2: NT → router temperature (NE-driven sharpness) ──
    # The locus coeruleus (NE) is the brain's gain channel. With
    # `router_temp_nt_gain > 0`, the harness pushes the homeostat's
    # NE level into `ThalamicRouter.set_nt_levels()` before each
    # forward and the router logits are multiplied by
    #   mult = clamp(1 + k_NE * 2 * (NE - 0.5), 0.1, 10.0)
    # before softmax. mult > 1 ⇒ sharper routing (winner-take-most),
    # mult < 1 ⇒ softer (mixture mode). Default 0.0 ⇒ identity.
    router_temp_nt_gain: float = 0.0

    # ── Item 3: NT → distillation λ (DA/5HT-driven trust schedule) ──
    # The harness already ramps λ piecewise-linearly in the EMA gap
    # `lm_loss_ema - cortex_loss_ema`. With these gains > 0, the
    # gap-ramp value is then multiplied by
    #   nt_mult = clamp(1 + k_5HT*z_5HT - k_DA*z_DA, 0, 2)
    # so the trunk leans HARDER on the cortex teacher when 5HT is
    # high (stress / conservatism) and LESS when DA is high (reward /
    # explore). Both default to 0.0 ⇒ identity (back-compat).
    distillation_5ht_gain: float = 0.0
    distillation_da_gain: float = 0.0

    # ── Item 4: Lateral expert inhibition (Mexican-hat / WTA via GABA) ──
    # When `lateral_inhibition_kappa > 0`, a `LateralInhibition` module
    # is inserted between the router and the per-expert mixing. It
    # implements divisive normalisation (Carandini & Heeger 2012-style):
    #   suppressed_i = w_i / (1 + κ_eff · Σ_{j≠i} w_j)
    #   w'_i         = suppressed_i / Σ_j suppressed_j
    # with κ_eff = κ_base · GABA_level (linear, [0, κ_base]).
    # The harness pushes the live GABA level via
    # `ensemble.set_nt_levels(...)` each step. Default 0.0 ⇒ identity.
    lateral_inhibition_kappa: float = 0.0

    # ── H006: Capacity-Funneled Distillation (CFD) ──────────────────
    # When `cfd_enabled = True`, the legacy Hinton KL path in
    # `_cortex_fusion_aux_step` is replaced with the three-stage CFD
    # pipeline (see docs/formal_framework.md §13 and
    # hypothesis/H006_capacity_funneled_distillation_implode.md):
    #
    #   Stage 1  top-K rank-preserving sparsification of the teacher
    #            distribution (K modes kept at softmax mass, V-K tail
    #            spread uniformly). Makes the imitation target lie
    #            inside the student's reachable softmax simplex.
    #   Stage 2  entropy-matched temperature
    #              T_eff = T_0 · max(1, H(student) / H(teacher))
    #            so the teacher's sharpness never exceeds what the
    #            student can plausibly match.
    #   Stage 3  gradient-alignment gate
    #              λ_eff = λ_0 · (1 + cos(∇distill, ∇LM)) / 2
    #            so the teacher can never harm the LM objective (mechanical
    #            no-harm floor).
    #
    # Also implicitly fixes Followup F1 (the F.kl_div reduction='batchmean'
    # vs LM 'mean' mismatch) because the CFD path uses per-token KL
    # reduction throughout.
    #
    # Default `False`: every existing arch.neuro reproduces bit-for-bit.
    # Only opt-in archs (e.g. H24+ with `cfd_enabled: true`) get the
    # new path.
    cfd_enabled: bool = False
    # Top-K schedule: K anneals linearly from `cfd_topk_start` to
    # `cfd_topk_end` over the first `cfd_topk_anneal_steps` training
    # steps, then stays at `cfd_topk_end` for the remainder.
    # Default schedule: 4 → 32 over the first 10000 steps.
    cfd_topk_start: int = 4
    cfd_topk_end: int = 32
    cfd_topk_anneal_steps: int = 10000
    # Temperature floor. The Stage-2 entropy-match output is clamped
    # to be ≥ this value (in particular the `max(1, ...)` clamp uses
    # this; set > 1.0 to forbid the raw temperature from ever going
    # below `cfd_temperature_floor · distillation_temperature`).
    cfd_temperature_floor: float = 1.0

    # ── CFDv2: Generalisation-Funneled Distillation (GFD) extensions ──
    # See docs/formal_framework.md §14 and the v2 falsifier rows in
    # hypothesis/H006_capacity_funneled_distillation_implode.md.
    #
    # Motivation: CFDv1's three stages bound the *training* loss but
    # cannot reduce the train↔OOD gap because all (context, target)
    # pairs are weighted equally. A stronger teacher fires sharper on
    # corpus-specific patterns ("the", "and" follow-ons) → student
    # learns the marginal faster → train PPL implodes but OOD diverges
    # (the H22/B6 SmolLM2 swap result: gap_ratio 2.87 → 6.55).
    #
    # v2 adds two cheap mechanisms that bias the funnel toward
    # *generalisation* signal without weakening the no-harm floor:
    #
    #   M2 — prior-residual sparsification
    #         Subtract γ · log p_uni(v) from the teacher logits before
    #         Stage 1. γ ∈ [0, 1]: 0 = noop (full back-compat), 1 =
    #         full removal of the unigram marginal so the distillation
    #         channel only carries CONTEXTUAL PMI signal.
    #
    #   M4 — pointwise K (PMI-driven, per-position)
    #         Replace the global K schedule with a per-position K(t)
    #         decreasing in the teacher's top-1 PMI:
    #            K(t) = clamp(K_max · exp(-PMI(t) / scale),
    #                         K_min, K_max)
    #         High-PMI positions (sharp contextual peaks on rare-prior
    #         tokens) get small K → concentrate the signal.
    #         Low-PMI positions (uniform teacher or peak-on-common)
    #         get large K → soft regulariser, do not over-commit.
    #
    # Defaults: γ=0 and pointwise_k_enabled=False ⇒ v2 is a no-op
    # (legacy global K-anneal schedule + raw teacher logits).
    cfd_prior_gamma: float = 0.0
    cfd_pointwise_k_enabled: bool = False
    cfd_pointwise_k_min: int = 2
    cfd_pointwise_k_max: int = 32
    # PMI decay scale in nats — controls how aggressively K drops with
    # increasing teacher PMI. Default 2.0 ≈ smooth interpolation across
    # typical LM context entropies. Smaller value ⇒ sharper drop.
    cfd_pmi_scale: float = 2.0


# ──────────────────────────────────────────────────────────────────────
# Multi-Objective Fitness — Phase A/F1 (central selection-pressure switch)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FitnessObjective:
    """A single objective in the `fitness.objectives { ... }` table.

    Each row resolves at training time to a scalar loss contribution:

        contribution = weight * schedule(maturity) * loss_value

    `schedule` selects the time-modulation curve:
      "constant" — weight stays fixed across training (default).
      "gated"    — weight is multiplied by the maturity phase gate
                    (matches the existing AuxWeights mechanism so the
                    legacy phi/world/forward losses can be migrated
                    into this table without behavioural drift).
      "linear"   — weight ramps linearly from 0 at step 0 to the
                    full value at the configured `warmup` step.
    """
    enabled: bool = False
    weight: float = 0.0
    schedule: str = "constant"


@dataclass
class FitnessConfig:
    """`fitness { ... }` block — central Multi-Objective-Fitness switch.

    Wires the existing aux-loss machinery + new objectives (symbolic,
    metabolic, piso, nis_plus) into a single declarative table so a
    single arch.neuro line per objective controls activation, weight
    and schedule.

    Master switch `enabled` keeps legacy archs bit-for-bit identical:
    when False, the harness ignores this entire block and uses the
    pre-existing `total_loss_config`.

    Recognised objective names (`objectives` dict keys):
      lm          — standard cross-entropy (matches `total_loss_config.w_lm`)
      phi         — IIT integrated information (existing GeneticsConfig.phi_weight)
      nis_plus    — NIS+ effective-information (NEW; wires existing NISPlus module)
      symbolic    — SymbolicHyperNeuron extractable-formula loss (NEW)
      piso        — Topological isomorphy loss (NEW; Procrustes-distance to Tonnetz)
      metabolic   — NRCSTK metabolic-budget overflow loss (NEW)
    """
    # Master switch. False ⇒ legacy single-objective training.
    enabled: bool = False

    # Per-objective specifications. Empty default = no new objectives;
    # the harness falls through to the pre-existing total_loss_config.
    objectives: Dict[str, FitnessObjective] = field(default_factory=dict)

    # ── Per-objective extras ──
    # Symbolic (when objectives["symbolic"].enabled):
    symbolic_n_units: int = 8
    symbolic_n_features: int = 16
    symbolic_tau_init: float = 1.0
    symbolic_tau_final: float = 0.05
    symbolic_sparsity_weight: float = 0.01

    # Metabolic (when objectives["metabolic"].enabled):
    metabolic_budget: float = 0.7       # fraction of mean activity allowed
    metabolic_prune_threshold: float = 0.05   # EMA below this ⇒ pruned

    # PIso (when objectives["piso"].enabled):
    piso_topology: str = "tonnetz"
    piso_target_dim: int = 256


@dataclass
class TrainingConfig:
    """Pipeline-level config the BRIAN harness consumes.

    Defaults match Brain's `rcc_bowtie_30m_p4` so an arch.neuro with no
    `training {}` block trains under the same conditions as the reference
    Brain run (matching loss/ppl/lm trajectory bit-for-bit when the trunk
    is the bit-identical port). The five new fields (batch_size, seq_len,
    steps, warmup_steps, min_lr_ratio) move runtime hyperparameters into
    the architecture spec instead of leaving them in shell-script defaults.
    """
    loss_clipping: LossClippingConfig = field(default_factory=LossClippingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    grad_accum: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    # Runtime hyperparameters — declarative form of the deploy script flags.
    # batch_size / seq_len affect tokens-per-step (Brain p4 uses 32 / 1024 →
    # 32k tokens/step; DSL must match for trajectory parity).
    batch_size: int = 4
    seq_len: int = 256
    # ``steps == 0`` means "no opinion — let the CLI / brian.toml decide".
    # Non-zero arch values win over ``brian.toml [defaults] steps`` per
    # the 2026-06-12 precedence (arch > global > CLI fallback).
    steps: int = 0
    warmup_steps: int = 300
    min_lr_ratio: float = 0.1
    # OOD-generalization knobs (added 2026-05-30 after first 10k run hit
    # gap_ratio 7.04). All default to 0 = off so prior runs reproduce.
    dropout: float = 0.0
    pct_strength: float = 0.0   # multiplies the PCH-aux weight in the trunk path
    # Forward-path PCT (Stage 1 of the OOD architecture push).
    # > 0 inserts a top-down prediction loop at every block boundary so
    # each layer is pulled toward what the next layer predicts. The
    # cited literature claims >=2x OOD-gap reduction vs the aux-only
    # PCH variant (`pct_strength`). Recommended start: 0.5.
    pct_trunk: float = 0.0
    # Stage 2 OOD push: Tonnetz toroidal attention mask. >0 enables the
    # mask with the given period (12 = musical octave is the standard
    # choice). The mask suppresses attention to positions outside a
    # circular-distance bandwidth + local window. 0 = standard causal
    # attention (no toroidal constraint).
    tonnetz_period: int = 0
    # Stage 3 OOD push: BEMA branching optimizer wrapper. When > 0,
    # rolls back the last `bema_rollback_window` optimizer updates if
    # the loss-EMA rises for that many consecutive steps. Addresses
    # the recurring "spike at step ~2400" pattern we keep hitting.
    # 0 = off (no wrapping). 50 is a sensible default.
    bema_rollback_window: int = 0
    bema_snapshot_every: int = 50
    bema_cooldown: int = 100
    # Stage 4 OOD push: NEMORI predictive-forgetting surprise gate.
    # > 0 enables a per-batch surprise check; batches with surprise
    # BELOW nemori_floor are SKIPPED (no gradient update). Reduces
    # I(X;Z) by refusing to learn from "expected" episodes. Surprise
    # = |loss - ema_loss| / max(ema_loss, eps). Typical floor 0.05-0.20.
    nemori_floor: float = 0.0
    # Stage 8 OOD push: flooding loss (Ishida et al. 2020). Loss is
    # transformed as |loss - b| + b where b = flooding_level. Prevents
    # the model from over-fitting by refusing to push train loss below b.
    # 0 = off (standard loss). Recommended: 3.5–4.5 for LM at 30M scale.
    flooding_level: float = 0.0
    # Stage 9 OOD push: stochastic depth (Huang et al. 2016). Each block
    # has a linearly increasing probability of being skipped (identity).
    # 0 = off. 0.1 means the deepest block is dropped 10% of the time.
    stochastic_depth: float = 0.0
    # Stage 10 OOD push: z-loss (PaLM/Gemma). Penalises logit magnitude
    # via `α * logsumexp(logits)^2`. Caps logit growth → numerical
    # stability + implicit regularization. PaLM showed 3-5% PPL drop AND
    # 10-15% OOD gap drop simultaneously. 1e-4 is the PaLM default.
    z_loss: float = 0.0
    # Stage 11 OOD push: layer-wise LR decay (ULMFiT / DeBERTa). Top
    # transformer blocks get a smaller LR than bottom ones:
    #     lr_i = base_lr * llrd_factor^(depth - 1 - i)
    # Lower layers learn general features fast; top layers learn slow
    # which prevents memorisation-style overfitting. 0 or 1.0 = off.
    # Typical values: 0.75–0.95. 0.85 is the ULMFiT default.
    llrd: float = 1.0
    # ── C3 reentry as NT-gated trunk loss (Jun 2026) ─────────────────
    # The PC reentry probe (predicting sensory hidden state from motor
    # hidden state via a learned projection W) is no longer telemetry
    # only — its residual `||s − W·m_prev||²` is added to the LM loss
    # with gradient flowing through BOTH populations, so the trunk
    # learns to make motor and sensory hidden states mutually
    # predictive (soft cycle-consistency / internal world-model
    # constraint). 0 = off (telemetry only). 0.1 is the from-scratch
    # default — empirically small enough to not destabilise early LM
    # learning, large enough to noticeably shape the trunk by ~5k steps.
    pc_reentry_weight: float = 0.0
    # When True, scale `pc_reentry_weight` by a neuromodulator gate:
    #   gate = max(0, 1 + 0.5·DA − 0.7·GABA)
    # so the auxiliary loss strengthens under curiosity/reward (DA up)
    # and weakens under cortical inhibition (GABA up). Other NTs are
    # left agnostic to avoid double-counting with the homeostat.
    pc_reentry_nt_gate: bool = False
    # ── Item 6: trainable NT coupling matrix W ────────────────────────
    # When True, the DrivenNTSystem exposes its 7×5 driver→channel
    # coupling matrix as an ``nn.Parameter`` of shape (7, 5). The
    # *float* OU dynamics in ``step_full`` are unchanged (still use
    # the float ``self._W`` dict), so this knob never alters the
    # behaviour of ``levels()`` on its own. Gradient flows back to
    # ``W_param`` only through the differentiable readout
    # ``predict_nt_tensor(drivers)`` — the harness can use that
    # tensor wherever NT levels modulate trainable parameters
    # (router temperature, distillation λ, lateral inhibition κ) to
    # let the optimiser refine the coupling end-to-end.
    nt_w_trainable: bool = False
    # ── Variational Bowtie Bottleneck (VBB) — Jun 2026 ────────────────
    # Upgrades the C3 PC-reentry from a plain squared residual into a
    # *variational free-energy* term at the bowtie waist. When
    # ``vbb_alpha > 0`` the harness:
    #   1. Treats the motor pole ``h_m`` as the mean ``μ`` of a Gaussian
    #      ``q(h_m|x) = N(μ, σ²I)`` whose log-variance is produced by a
    #      tiny learned head ``log σ = sigma_head(μ)``  (one Linear,
    #      d_sem→d_sem, ~262k params at d=512 — 0.2% overhead on 122M).
    #   2. Reparameterises ``h_m_sample = μ + σ·ε`` with ``ε ~ N(0,I)``.
    #   3. Computes the residual ``r = ‖s − W·h_m_sample‖²`` via the
    #      frozen-W probe (same path as the legacy loss, just on the
    #      noised motor).
    #   4. Adds a single learnable confidence scalar ``β`` (via softplus
    #      of ``log_beta``) and forms the free-energy term
    #         ``L = γ_NT · ( β·r − log β + α·KL[q ‖ N(0,I)] )``
    #      where the Gaussian-Gaussian KL has the closed form
    #         ``KL = ½ · Σ (σ² + μ² − 1 − log σ²)``  (per-element mean).
    # Why this matters
    # ----------------
    # • The KL is an *Information Bottleneck* (Tishby) placed exactly
    #   at the bowtie waist — provably the right spot for it.
    # • The ``β·r − log β`` pair is the precision-weighted prediction
    #   error of the Free Energy Principle (Friston). ``β`` self-
    #   throttles at the equilibrium ``β* = 1/(2·E[r])`` — the loop
    #   stops being a self-distillation amplifier and starts being a
    #   genuine generative model with calibrated confidence.
    # • The reparam noise at the waist is structurally placed exactly
    #   where the C3 loop reads, so the loop cannot memorise the
    #   trunk's fingerprint — biases SGD toward flat minima.
    # • Composes with ``pc_reentry_nt_gate``: the NT gate still
    #   multiplies the whole free-energy term, so DA/GABA tone the
    #   precision modulation just as before.
    # ``vbb_alpha = 0`` (default) → exact legacy ``residual_diff`` path,
    # bit-identical to pre-VBB behaviour. Recommended on-value: 1e-3.
    vbb_alpha: float = 0.0
    # Initial value of the learnable confidence scalar β (parameterised
    # internally as ``β = softplus(log_beta)``). 1.0 is unit-Gaussian
    # likelihood; the optimizer will move it toward the data's
    # equilibrium within ~1k optimizer steps.
    vbb_beta_init: float = 1.0
    # ── MDRV-VBB anti-collapse stabilisers (VBB-v2, Jun 2026) ──────
    # Three interlocking mechanisms that prevent posterior collapse
    # (σ→0, β→∞) while preserving the IB compression benefit.
    #
    # Free-bits KL floor (Kingma et al. 2016, §3.5):
    #   KL_eff = max(KL_per_dim, δ).mean()
    # Guarantees minimum H[q] per dimension; makes σ→0 non-optimal
    # by construction (gradient of clamped term is zero below δ, so
    # the network cannot recoup loss by shrinking σ further).
    # 0.0 = off (legacy behaviour). Recommended: 0.1 nats/dim.
    vbb_free_bits: float = 0.0
    # Hard ceiling on log β (prevents the β↑/σ↓ co-collapse):
    #   log_β_eff = clamp(log_β, max=vbb_log_beta_max)
    # Without a ceiling the joint equilibrium (raise β / collapse σ)
    # eventually dominates. 0.0 = uncapped. Recommended: 4.0
    # (β ≤ softplus(4) ≈ 4.0, residual cannot vanish by β inflation).
    vbb_log_beta_max: float = 0.0
    # Posterior Entropy Commitment weight η (Jeffreys σ-prior):
    #   PEC = −η · ½ · 𝔼[log σ²]
    # Adds η times the Gaussian differential entropy of q to the
    # objective (we minimise, so −η·H[q] → maximise entropy).
    # Equivalent to placing a Jeffreys prior p(σ) ∝ 1/σ on each
    # scale parameter: as σ→0, log σ²→−∞ and PEC→+∞, making σ=0
    # a repulsive (Lyapunov-unstable) fixed point. This provides the
    # mathematical guarantee that the prior summary cannot escape to
    # the degenerate channel regardless of batch noise.
    # 0.0 = off. Recommended: 0.001 (weak prior, leaves α·KL dominant).
    vbb_entropy_eta: float = 0.0
    # ── HPB Phase 4 — Hyperbolic Bowtie Waist (HBW) ──────────────────
    # Curvature parameter c > 0 of the Poincaré ball B^d_c on which the
    # VBB posterior is wrapped. The KL closed-form picks up a Jacobian
    # correction (d-1)·log(sinh(√c·‖μ‖)/(√c·‖μ‖)) ≥ 0 that strictly
    # upper-bounds the Euclidean KL for any non-zero μ — making
    # σ-collapse harder by the construction of the geometry itself.
    # Composes with all MDRV stabilisers (free-bits, β-ceiling, PEC).
    # 0.0 = off (Euclidean VBB, legacy path). Recommended: 1.0.
    vbb_curvature: float = 0.0
    # ── HPB Phase 3 — Multi-Scale Predictive Coding Cascade (MSPCC) ──
    # When set to a dict with `enabled: true`, applies the VBB free-
    # energy term to EVERY adjacent layer pair (ℓ, ℓ+1) of the trunk
    # with geometric decay: λ_ℓ = base_weight · decay^((L-1)-ℓ).
    # Shares vbb_alpha / vbb_free_bits / vbb_log_beta_max / vbb_entropy_eta
    # with the single-waist VBB. Composes additively with it.
    # `None` (default) ⇒ off; legacy single-waist VBB path is preserved.
    mspcc: Optional[Dict[str, Any]] = None
    # ── Novel-topology mechanisms (H15 / H16 / H19) ──────────────────
    # Each accepts a dict or `None` (= off). When all are None the
    # cortex is bit-identical to the legacy baseline (zero-init
    # discipline enforced inside neuroslm/dsl/novel_topology.py).
    # H16 — multi-scale grid-cell positional bias. Provable length-OOD.
    grid_positions: Optional[Dict[str, Any]] = None
    # H15 — episodic kNN memory. Read-blend via ReZero gate (alpha=0).
    episodic_memory: Optional[Dict[str, Any]] = None
    # H19 — local-context surprise head. Composes with H15 via
    # `episodic_memory.write_gate = "surprise"`.
    surprise_head: Optional[Dict[str, Any]] = None
    # Stage 7 OOD push: curriculum + trunk isolation. Curriculum string
    # selects a data ordering strategy ("easy_to_hard", "random",
    # "uniform"). Trunk isolation is enforced by an existing param_scope
    # mechanism (Brain side); audited in tests.
    curriculum: str = "random"
    crystallization_step: int = 0    # 0 = no curriculum boundary
    # Stage 6 OOD push: μP scaling. When True, applies width-aware
    # init + per-param-group LR multipliers so representation updates
    # stay O(1) as d_model scales. Only meaningful at 200M+ params.
    mu_p_scaling: bool = False
    # Preset for trunk-size selection. When set, arch.neuro overrides
    # the env-var PRESET. Lets a single arch.neuro lock in BOTH the
    # architecture AND the model scale.
    preset: str = ""
    # MAT-phase-gated mechanism activation. Each declared mechanism's
    # effective strength is `declared × gate.value(maturity)`. See
    # MechanismsConfig + PhaseGate above. Empty = use the flat-strength
    # legacy fields (dropout, pct_trunk, ...) instead.
    mechanisms: MechanismsConfig = field(default_factory=MechanismsConfig)
    # Declarative early-exit rules. Empty list = train to STEPS.
    pass_marks: PassMarksConfig = field(default_factory=PassMarksConfig)
    # GeneticOrchestrator — latent gene expression with Phi-loss coupling.
    # See architecture.md §6.5. Disabled by default.
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)
    # Multi-Trunk-V2 ensemble — 4 specialist language cortices routed by
    # a thalamic mixture-of-experts gate. See architecture.md §5.7.
    # Disabled by default → legacy single-cortex behaviour unchanged.
    multi_cortex: MultiCortexConfig = field(default_factory=MultiCortexConfig)
    # Synthetic HPA axis — slow homeostatic stress damping (allostasis).
    # Tracks `load` (fast EMA of multi-modal stress) and `cort` (slow
    # EMA of load). When cort climbs, it multiplicatively damps NE
    # release, trophic growth, and learning rate — the negative-feedback
    # loop that's missing in the existing neuromod stack.
    # Disabled by default → legacy behaviour preserved bit-for-bit.
    allostasis: AllostasisConfig = field(default_factory=AllostasisConfig)
    # Multi-Objective-Fitness central switchboard — wires existing aux
    # losses + new objectives (symbolic, metabolic, piso, nis_plus) into
    # a single declarative table. Disabled by default → legacy single-
    # objective training reproduces bit-for-bit.
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    # Hardware envelope + multi-scale variants. The deploy script reads
    # `hardware` to filter vast.ai offers; the harness reads `scales` +
    # the SCALE env var to pick which variant to instantiate.
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    scales: ScalesConfig = field(default_factory=ScalesConfig)
    # Declarative metric-exposure points. Each entry says "compute this
    # metric and publish it at these node tags." Reduces data overhead by
    # not exposing every metric everywhere.
    metric_exposures: List[MetricExpose] = field(default_factory=list)
    # OOD interventions (`regularization { ... }` block). Five composable
    # losses declared math-first in lib/regularizers.neuro. All default
    # disabled → zero behavioral change vs legacy arch.neuro. Parsed via
    # neuroslm.dsl.regularization.parse_regularization_block; harness
    # consumption lands in PR2.
    regularization: Any = None  # filled with RegularizationConfig in parse_training_config
    # Geometric Information Funnel (GIF) — 3-mechanism OOD-PPL fix.
    # Parsed as a raw dict; consumed by neuroslm.emergent.gif.GIFController.
    gif: Optional[dict] = None
    # Field names the arch declared with a trailing ``!`` (e.g.
    # ``preset!: "cheap_2k"``). These resist global-default merging in
    # :func:`apply_global_defaults`. Never set by users directly —
    # populated by :func:`parse_training_config` from the DSL source.
    # ``repr=False`` keeps debug output uncluttered.
    _sticky_fields: Set[str] = field(default_factory=set, repr=False)


# ── Constants for validation ───────────────────────────────────────────

_VALID_LOSS_METHODS = {"per_sample"}
_VALID_QUANT_BITS = {4, 8, 16}
_VALID_OPTIMIZERS = {"adamw", "adafactor"}
# Allowed values for `multi_cortex.weights`. "stub" = unit-test/CI safe
# (no network); "gpt2" = HF GPT-2 family ensemble (build_gpt2_ensemble).
# Add new providers here as they're implemented in neuroslm/cortex.py.
_VALID_MULTI_CORTEX_WEIGHTS = {"stub", "gpt2"}

# Allowed values for `multi_cortex.fusion_mode`. Controls how the
# ensemble feeds the LM head — see MultiCortexConfig docstring.
#   "logits_mixture": late fusion via tied cortex_lm_head + sigmoid mix.
#   "off":            build the ensemble but do not touch LM logits
#                     (legacy "telemetry-only" semantics).
_VALID_MULTI_CORTEX_FUSION_MODES = {"logits_mixture", "off"}

# Allowed `fitness.objectives` keys.  Adding a new objective here is the
# *only* place the parser needs to learn about it — the FitnessComposer
# discovers them by enumeration at runtime.  Each name maps to a concrete
# loss source documented in `neuroslm/fitness.py`.
_VALID_FITNESS_OBJECTIVES = {
    "lm", "phi", "nis_plus", "symbolic", "piso", "metabolic",
}
# Allowed `FitnessObjective.schedule` values — see the dataclass docstring.
_VALID_FITNESS_SCHEDULES = {"constant", "gated", "linear"}


# ── Parser ─────────────────────────────────────────────────────────────

def parse_training_config(body: str) -> TrainingConfig:
    """Parse the body of a `training { ... }` block (without the braces).

    Empty body → all defaults. Unknown top-level keys are silently ignored
    (forward-compat with future versions of this schema). Invalid values
    for known keys raise ValueError.
    """
    cfg = TrainingConfig()

    props, sticky = _split_top_level_kv_with_stickies(body)
    # Track every field the arch insisted on with `!`. `apply_global_defaults`
    # consults this set so the global `[defaults]` block can't stomp pins.
    cfg._sticky_fields = set(sticky)

    # Sub-blocks
    if "loss_clipping" in props:
        cfg.loss_clipping = _parse_loss_clipping(props["loss_clipping"])
    if "quantization" in props:
        cfg.quantization = _parse_quantization(props["quantization"])

    # Scalars
    if "grad_accum" in props:
        cfg.grad_accum = int(props["grad_accum"])
    if "optimizer" in props:
        opt = _strip_quotes(props["optimizer"])
        if opt not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {opt!r}; expected one of {sorted(_VALID_OPTIMIZERS)}"
            )
        cfg.optimizer = opt
    if "learning_rate" in props:
        cfg.learning_rate = float(props["learning_rate"])
    if "weight_decay" in props:
        cfg.weight_decay = float(props["weight_decay"])
    if "grad_clip" in props:
        cfg.grad_clip = float(props["grad_clip"])
    if "label_smoothing" in props:
        cfg.label_smoothing = float(props["label_smoothing"])
    # Runtime hyperparameters (deploy-script flags moved into the arch spec)
    if "batch_size" in props:
        cfg.batch_size = int(props["batch_size"])
    if "seq_len" in props:
        cfg.seq_len = int(props["seq_len"])
    if "steps" in props:
        cfg.steps = int(props["steps"])
    if "warmup_steps" in props:
        cfg.warmup_steps = int(props["warmup_steps"])
    if "min_lr_ratio" in props:
        cfg.min_lr_ratio = float(props["min_lr_ratio"])
    if "dropout" in props:
        cfg.dropout = float(props["dropout"])
    if "pct_strength" in props:
        cfg.pct_strength = float(props["pct_strength"])
    if "pct_trunk" in props:
        cfg.pct_trunk = float(props["pct_trunk"])
    if "tonnetz_period" in props:
        cfg.tonnetz_period = int(props["tonnetz_period"])
    if "bema_rollback_window" in props:
        cfg.bema_rollback_window = int(props["bema_rollback_window"])
    if "bema_snapshot_every" in props:
        cfg.bema_snapshot_every = int(props["bema_snapshot_every"])
    if "bema_cooldown" in props:
        cfg.bema_cooldown = int(props["bema_cooldown"])
    if "nemori_floor" in props:
        cfg.nemori_floor = float(props["nemori_floor"])
    if "flooding_level" in props:
        cfg.flooding_level = float(props["flooding_level"])
    if "stochastic_depth" in props:
        cfg.stochastic_depth = float(props["stochastic_depth"])
    if "z_loss" in props:
        cfg.z_loss = float(props["z_loss"])
    if "llrd" in props:
        cfg.llrd = float(props["llrd"])
    if "pc_reentry_weight" in props:
        cfg.pc_reentry_weight = float(props["pc_reentry_weight"])
    if "pc_reentry_nt_gate" in props:
        cfg.pc_reentry_nt_gate = _parse_bool(props["pc_reentry_nt_gate"])
    # ── Item 6: trainable NT coupling matrix W ──
    if "nt_w_trainable" in props:
        cfg.nt_w_trainable = _parse_bool(props["nt_w_trainable"])
    if "vbb_alpha" in props:
        cfg.vbb_alpha = float(props["vbb_alpha"])
    if "vbb_beta_init" in props:
        cfg.vbb_beta_init = float(props["vbb_beta_init"])
    if "vbb_free_bits" in props:
        cfg.vbb_free_bits = float(props["vbb_free_bits"])
    if "vbb_log_beta_max" in props:
        cfg.vbb_log_beta_max = float(props["vbb_log_beta_max"])
    if "vbb_entropy_eta" in props:
        cfg.vbb_entropy_eta = float(props["vbb_entropy_eta"])
    # HPB Phase 4 — Hyperbolic Bowtie Waist curvature.
    if "vbb_curvature" in props:
        cfg.vbb_curvature = float(props["vbb_curvature"])
    # HPB Phase 3 — Multi-Scale Predictive Coding Cascade.
    if "mspcc" in props:
        cfg.mspcc = _parse_novel_topology_dict(props["mspcc"])
    # Novel-topology mechanisms (H15/H16/H19) — parse as generic dicts.
    if "grid_positions" in props:
        cfg.grid_positions = _parse_novel_topology_dict(props["grid_positions"])
    if "episodic_memory" in props:
        cfg.episodic_memory = _parse_novel_topology_dict(props["episodic_memory"])
    if "surprise_head" in props:
        cfg.surprise_head = _parse_novel_topology_dict(props["surprise_head"])
    if "curriculum" in props:
        cfg.curriculum = _strip_quotes(props["curriculum"])
    if "crystallization_step" in props:
        cfg.crystallization_step = int(props["crystallization_step"])
    if "mu_p_scaling" in props:
        cfg.mu_p_scaling = _parse_bool(props["mu_p_scaling"])
    if "preset" in props:
        cfg.preset = _strip_quotes(props["preset"])
    if "mechanisms" in props:
        cfg.mechanisms = _parse_mechanisms(props["mechanisms"])
    if "pass_marks" in props:
        cfg.pass_marks = _parse_pass_marks(props["pass_marks"])
    if "genetics" in props:
        cfg.genetics = _parse_genetics(props["genetics"])
    if "multi_cortex" in props:
        cfg.multi_cortex = _parse_multi_cortex(props["multi_cortex"])
    if "allostasis" in props:
        cfg.allostasis = _parse_allostasis(props["allostasis"])
    if "fitness" in props:
        cfg.fitness = _parse_fitness(props["fitness"])
    if "gif" in props:
        cfg.gif = _parse_novel_topology_dict(props["gif"])
    if "hardware" in props:
        cfg.hardware = _parse_hardware(props["hardware"])
    if "scales" in props:
        cfg.scales = _parse_scales(props["scales"], cfg.hardware)

    # `metric <name> { ... }` blocks live at the same level as `mechanisms`
    # / `pass_marks`. Collect all of them.
    cfg.metric_exposures = _parse_metric_exposures(props)

    # ── Five OOD interventions (regularization { ... }) ──
    # Imported lazily to avoid a circular import at module load time.
    from .regularization import (
        RegularizationConfig, parse_regularization_block
    )
    if "regularization" in props:
        cfg.regularization = parse_regularization_block(
            _strip_braces(props["regularization"])
        )
    else:
        cfg.regularization = RegularizationConfig()
    return cfg


def _parse_hardware(body: str) -> HardwareConfig:
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    h = HardwareConfig()
    if "gpu_name" in props:        h.gpu_name = _strip_quotes(props["gpu_name"])
    if "num_gpus" in props:        h.num_gpus = int(props["num_gpus"])
    if "min_gpu_mem_gib" in props: h.min_gpu_mem_gib = int(props["min_gpu_mem_gib"])
    if "min_reliability" in props: h.min_reliability = float(props["min_reliability"])
    if "min_inet_mbps" in props:   h.min_inet_mbps = int(props["min_inet_mbps"])
    if "dist_strategy" in props:   h.dist_strategy = _strip_quotes(props["dist_strategy"])
    if "precision" in props:       h.precision = _strip_quotes(props["precision"])
    return h


def _parse_scales(body: str, fallback_hw: HardwareConfig) -> ScalesConfig:
    """Parse a `scales { name: { ... }, ... }` block."""
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    cfg = ScalesConfig()
    if "default" in props:
        cfg.default = _strip_quotes(props.pop("default"))
    for name, raw in props.items():
        rp = _split_top_level_kv(_strip_braces(raw))
        v = ScaleVariant(name=name)
        if "d_model" in rp:       v.d_model = int(rp["d_model"])
        if "depth" in rp:         v.depth = int(rp["depth"])
        if "n_heads" in rp:       v.n_heads = int(rp["n_heads"])
        if "n_kv_heads" in rp:    v.n_kv_heads = int(rp["n_kv_heads"])
        if "max_ctx" in rp:       v.max_ctx = int(rp["max_ctx"])
        if "batch_size" in rp:    v.batch_size = int(rp["batch_size"])
        if "seq_len" in rp:       v.seq_len = int(rp["seq_len"])
        if "grad_accum" in rp:    v.grad_accum = int(rp["grad_accum"])
        if "learning_rate" in rp: v.learning_rate = float(rp["learning_rate"])
        if "approx_params" in rp: v.approx_params = _strip_quotes(rp["approx_params"])
        if "hardware" in rp:
            v.hardware = _parse_hardware(rp["hardware"])
        else:
            v.hardware = fallback_hw
        cfg.variants[name] = v
    if not cfg.default and cfg.variants:
        cfg.default = next(iter(cfg.variants))
    return cfg


def _parse_metric_exposures(props: Dict[str, str]) -> List[MetricExpose]:
    """Collect every `metric_<name>: { ... }` or `metric: { name, ... }`
    entry. The parser supports both forms — flat per-metric keys
    (`metric_phi: { ... }`, `metric_mat: { ... }`) and a single grouped
    entry that won't trigger on the legacy `metric` -> string fields.
    """
    out: List[MetricExpose] = []
    for key, raw in props.items():
        if not key.startswith("metric_"):
            continue
        body = _strip_braces(raw)
        rp = _split_top_level_kv(body)
        m = MetricExpose(name=key[len("metric_"):])
        if "compute" in rp:
            m.compute = _strip_quotes(rp["compute"])
        if "every_n_steps" in rp:
            m.every_n_steps = int(rp["every_n_steps"])
        if "expose_at" in rp:
            v = rp["expose_at"].strip()
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            m.expose_at = [_strip_quotes(x.strip()) for x in v.split(",") if x.strip()]
        out.append(m)
    return out


def _parse_novel_topology_dict(body: str) -> Dict[str, Any]:
    """Parse a `<name> { key: value, ... }` block into a plain dict.

    Used for H15 / H16 / H19 novel-topology specs. Values are typed
    heuristically: bool → int → float → quoted string → bare string.
    The downstream factory in neuroslm.dsl.novel_topology coerces
    them to the right type with sensible defaults, so we don't try
    to be clever here.
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    out: Dict[str, Any] = {}
    for k, raw in props.items():
        s = raw.strip()
        if s.lower() in ("true", "false"):
            out[k] = (s.lower() == "true")
            continue
        try:
            out[k] = int(s)
            continue
        except (ValueError, TypeError):
            pass
        try:
            out[k] = float(s)
            continue
        except (ValueError, TypeError):
            pass
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            out[k] = s[1:-1]
            continue
        out[k] = s
    out.setdefault("enabled", True)
    return out


def _parse_genetics(body: str) -> GeneticsConfig:
    """Parse a `genetics { ... }` block.

    Keys: enabled, n_genes, d_pay, phi_weight, phi_target,
          fixed_genes_preset, target_modules: [a, b, c]
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    g = GeneticsConfig()
    if "enabled" in props:             g.enabled = _parse_bool(props["enabled"])
    if "n_genes" in props:             g.n_genes = int(props["n_genes"])
    if "d_pay" in props:               g.d_pay = int(props["d_pay"])
    if "phi_weight" in props:          g.phi_weight = float(props["phi_weight"])
    if "phi_target" in props:          g.phi_target = float(props["phi_target"])
    if "fixed_genes_preset" in props:  g.fixed_genes_preset = _strip_quotes(props["fixed_genes_preset"])
    if "target_modules" in props:
        raw = props["target_modules"].strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        g.target_modules = [_strip_quotes(x.strip()) for x in raw.split(",") if x.strip()]
    if "update_every" in props:       g.update_every = int(props["update_every"])
    if "diagnostics_every" in props:  g.diagnostics_every = int(props["diagnostics_every"])
    if "optimize_for" in props:
        g.optimize_for = _strip_quotes(props["optimize_for"])
        # "lm_loss" / "ppl" → genes only train via cortex grad path
        if g.optimize_for in ("lm_loss", "ppl"):
            g.phi_weight = 0.0
    return g


def _parse_experts_list(raw: str) -> List[ExpertSpec]:
    """Parse ``[ { id: "x", domain: "y", freeze: true, weight: 1.0 }, ... ]``
    into a list of :class:`ExpertSpec`.

    Whitespace + newline tolerant. Unknown per-expert fields are
    silently ignored (forward-compat for future flags like ``bridge_mode``).
    """
    s = raw.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(
            f"multi_cortex.experts must be a [...] list, got: {raw[:60]}"
        )
    s = s[1:-1].strip()
    if not s:
        # Empty list — handled by cross-field validation downstream
        return []

    # Split top-level `{...}` dict literals at commas not inside braces.
    rows: List[str] = []
    depth, in_str, start = 0, None, 0
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append(s[start:i + 1])
        i += 1

    if not rows:
        raise ValueError(
            f"multi_cortex.experts: could not find any "
            f"{{ ... }} entries in: {raw[:80]}"
        )

    seen_domains: set = set()
    out: List[ExpertSpec] = []
    for row in rows:
        body = _strip_braces(row)
        kv = _split_top_level_kv(body)
        if "id" not in kv:
            raise ValueError(
                f"multi_cortex.experts: missing required field `id` "
                f"in entry: {row[:80]}"
            )
        if "domain" not in kv:
            raise ValueError(
                f"multi_cortex.experts: missing required field `domain` "
                f"in entry: {row[:80]}"
            )
        eid = _strip_quotes(kv["id"])
        domain = _strip_quotes(kv["domain"])
        freeze = (
            _parse_bool(kv["freeze"]) if "freeze" in kv else True
        )
        weight = float(kv["weight"]) if "weight" in kv else 1.0
        if domain in seen_domains:
            raise ValueError(
                f"multi_cortex.experts: duplicate domain {domain!r} — "
                f"each expert must have a unique routing key"
            )
        seen_domains.add(domain)
        out.append(ExpertSpec(
            id=eid, domain=domain, freeze=freeze, weight=weight,
        ))
    return out


def _parse_multi_cortex(body: str) -> MultiCortexConfig:
    """Parse a `multi_cortex { ... }` block.

    Keys: enabled, n_cortices, domains: [a, b, ...], weights,
          freeze_weights, lexical_bias_weight, bema_tau, router_d_model

    Validates:
      * weights ∈ {"stub", "gpt2"}     — fail loudly on typos like "gtp2"
                                          so we don't trigger an HF
                                          download with a bad checkpoint
                                          name.
      * n_cortices == len(domains)     — mismatch is a guaranteed crash
                                          inside ThalamicRouter; catching
                                          it here surfaces a clear error
                                          at parse time.
      * 0.0 ≤ bema_tau < 1.0           — Bregman-EMA stability bound.
      * lexical_bias_weight ≥ 0        — softmax temperature must not
                                          flip sign.
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    m = MultiCortexConfig()

    if "enabled" in props:
        m.enabled = _parse_bool(props["enabled"])
    if "n_cortices" in props:
        m.n_cortices = int(props["n_cortices"])
    if "domains" in props:
        raw = props["domains"].strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        m.domains = [_strip_quotes(x.strip()) for x in raw.split(",") if x.strip()]
    if "weights" in props:
        w = _strip_quotes(props["weights"])
        if w not in _VALID_MULTI_CORTEX_WEIGHTS:
            raise ValueError(
                f"unknown multi_cortex weights {w!r}; "
                f"expected one of {sorted(_VALID_MULTI_CORTEX_WEIGHTS)}"
            )
        m.weights = w
    if "freeze_weights" in props:
        m.freeze_weights = _parse_bool(props["freeze_weights"])
    if "lexical_bias_weight" in props:
        m.lexical_bias_weight = float(props["lexical_bias_weight"])
    if "bema_tau" in props:
        m.bema_tau = float(props["bema_tau"])
    if "router_d_model" in props:
        m.router_d_model = int(props["router_d_model"])
    if "fusion_mode" in props:
        fm = _strip_quotes(props["fusion_mode"])
        if fm not in _VALID_MULTI_CORTEX_FUSION_MODES:
            raise ValueError(
                f"unknown multi_cortex fusion_mode {fm!r}; "
                f"expected one of {sorted(_VALID_MULTI_CORTEX_FUSION_MODES)}"
            )
        m.fusion_mode = fm
    if "fusion_init" in props:
        m.fusion_init = float(props["fusion_init"])

    # ── A: distillation parameters ──
    if "distillation_enabled" in props:
        m.distillation_enabled = _parse_bool(props["distillation_enabled"])
    if "distillation_lambda_max" in props:
        m.distillation_lambda_max = float(props["distillation_lambda_max"])
    if "distillation_temperature" in props:
        m.distillation_temperature = float(props["distillation_temperature"])
    if "distillation_gap_floor" in props:
        m.distillation_gap_floor = float(props["distillation_gap_floor"])
    if "distillation_gap_ceiling" in props:
        m.distillation_gap_ceiling = float(props["distillation_gap_ceiling"])

    # ── C: inhibition / NT-gated α ──
    if "inhibition_enabled" in props:
        m.inhibition_enabled = _parse_bool(props["inhibition_enabled"])
    if "inhibition_ema_alpha" in props:
        m.inhibition_ema_alpha = float(props["inhibition_ema_alpha"])
    if "inhibition_temperature" in props:
        m.inhibition_temperature = float(props["inhibition_temperature"])

    # ── Item 2: NT → router temperature ──
    if "router_temp_nt_gain" in props:
        m.router_temp_nt_gain = float(props["router_temp_nt_gain"])

    # ── Item 3: NT → distillation λ ──
    if "distillation_5ht_gain" in props:
        m.distillation_5ht_gain = float(props["distillation_5ht_gain"])
    if "distillation_da_gain" in props:
        m.distillation_da_gain = float(props["distillation_da_gain"])

    # ── Item 4: Lateral expert inhibition (κ_base; GABA pushed by harness) ──
    if "lateral_inhibition_kappa" in props:
        m.lateral_inhibition_kappa = float(props["lateral_inhibition_kappa"])

    # ── H006 CFD ──
    if "cfd_enabled" in props:
        m.cfd_enabled = _parse_bool(props["cfd_enabled"])
    if "cfd_topk_start" in props:
        m.cfd_topk_start = int(props["cfd_topk_start"])
    if "cfd_topk_end" in props:
        m.cfd_topk_end = int(props["cfd_topk_end"])
    if "cfd_topk_anneal_steps" in props:
        m.cfd_topk_anneal_steps = int(props["cfd_topk_anneal_steps"])
    if "cfd_temperature_floor" in props:
        m.cfd_temperature_floor = float(props["cfd_temperature_floor"])

    # ── CFDv2 (GFD) ──
    if "cfd_prior_gamma" in props:
        m.cfd_prior_gamma = float(props["cfd_prior_gamma"])
    if "cfd_pointwise_k_enabled" in props:
        m.cfd_pointwise_k_enabled = _parse_bool(
            props["cfd_pointwise_k_enabled"]
        )
    if "cfd_pointwise_k_min" in props:
        m.cfd_pointwise_k_min = int(props["cfd_pointwise_k_min"])
    if "cfd_pointwise_k_max" in props:
        m.cfd_pointwise_k_max = int(props["cfd_pointwise_k_max"])
    if "cfd_pmi_scale" in props:
        m.cfd_pmi_scale = float(props["cfd_pmi_scale"])

    # ── Per-expert roster (new path, replaces ``weights`` shorthand) ──
    # `experts: [ { id: "gpt2", domain: "general", freeze: true }, ... ]`
    # When present, supersedes ``weights`` and auto-derives `domains`
    # and `n_cortices` from the roster so they can't drift.
    if "experts" in props:
        m.experts = _parse_experts_list(props["experts"])
        # Auto-derive — operator can't get them out of sync
        m.domains = [e.domain for e in m.experts]
        m.n_cortices = len(m.experts)
        # Deprecation warning if legacy `weights` was also supplied
        if "weights" in props:
            import warnings
            warnings.warn(
                "multi_cortex: both `weights` and `experts` were supplied; "
                "`experts` takes precedence. Remove `weights` to silence "
                "this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
    if "trunk_tokenizer" in props:
        m.trunk_tokenizer = _strip_quotes(props["trunk_tokenizer"])

    # ── Cross-field validation ──
    if m.lexical_bias_weight < 0:
        raise ValueError(
            f"multi_cortex.lexical_bias_weight must be >= 0, "
            f"got {m.lexical_bias_weight}"
        )
    if not (0.0 <= m.bema_tau < 1.0):
        raise ValueError(
            f"multi_cortex.bema_tau must be in [0.0, 1.0), got {m.bema_tau}"
        )
    if m.n_cortices != len(m.domains):
        raise ValueError(
            f"multi_cortex.n_cortices ({m.n_cortices}) must equal "
            f"len(domains) ({len(m.domains)}); domains={m.domains}"
        )
    if not (0.0 <= m.fusion_init <= 1.0):
        raise ValueError(
            f"multi_cortex.fusion_init must be in [0.0, 1.0], "
            f"got {m.fusion_init}"
        )
    if m.distillation_lambda_max < 0:
        raise ValueError(
            f"multi_cortex.distillation_lambda_max must be >= 0, "
            f"got {m.distillation_lambda_max}"
        )
    if m.distillation_temperature <= 0:
        raise ValueError(
            f"multi_cortex.distillation_temperature must be > 0, "
            f"got {m.distillation_temperature}"
        )
    if not (m.distillation_gap_floor < m.distillation_gap_ceiling):
        raise ValueError(
            f"multi_cortex.distillation_gap_floor "
            f"({m.distillation_gap_floor}) must be < "
            f"distillation_gap_ceiling ({m.distillation_gap_ceiling})"
        )
    if not (0.0 < m.inhibition_ema_alpha <= 1.0):
        raise ValueError(
            f"multi_cortex.inhibition_ema_alpha must be in (0.0, 1.0], "
            f"got {m.inhibition_ema_alpha}"
        )
    if m.inhibition_temperature <= 0:
        raise ValueError(
            f"multi_cortex.inhibition_temperature must be > 0, "
            f"got {m.inhibition_temperature}"
        )
    # Reject empty `experts: []` roster — the whole point of the
    # block is to declare at least one expert.
    if m.experts is not None and len(m.experts) == 0:
        raise ValueError(
            "multi_cortex.experts: must contain at least one expert "
            "(or omit the field entirely to use the legacy `weights` path)"
        )
    # ── H006 CFD validation ──
    if m.cfd_enabled:
        if m.cfd_topk_start < 1:
            raise ValueError(
                f"multi_cortex.cfd_topk_start must be >= 1, "
                f"got {m.cfd_topk_start}"
            )
        if m.cfd_topk_end < m.cfd_topk_start:
            raise ValueError(
                f"multi_cortex.cfd_topk_end ({m.cfd_topk_end}) must be "
                f">= cfd_topk_start ({m.cfd_topk_start})"
            )
        if m.cfd_topk_anneal_steps < 0:
            raise ValueError(
                f"multi_cortex.cfd_topk_anneal_steps must be >= 0, "
                f"got {m.cfd_topk_anneal_steps}"
            )
        if m.cfd_temperature_floor < 1.0:
            raise ValueError(
                f"multi_cortex.cfd_temperature_floor must be >= 1.0 "
                f"(otherwise Stage 2 could SHARPEN the teacher), "
                f"got {m.cfd_temperature_floor}"
            )
        # ── CFDv2 (GFD) validation ──
        if not (0.0 <= m.cfd_prior_gamma <= 1.0):
            raise ValueError(
                f"multi_cortex.cfd_prior_gamma must be in [0.0, 1.0], "
                f"got {m.cfd_prior_gamma}"
            )
        if m.cfd_pointwise_k_enabled:
            if m.cfd_pointwise_k_min < 1:
                raise ValueError(
                    f"multi_cortex.cfd_pointwise_k_min must be >= 1, "
                    f"got {m.cfd_pointwise_k_min}"
                )
            if m.cfd_pointwise_k_max < m.cfd_pointwise_k_min:
                raise ValueError(
                    f"multi_cortex.cfd_pointwise_k_max "
                    f"({m.cfd_pointwise_k_max}) must be >= "
                    f"cfd_pointwise_k_min ({m.cfd_pointwise_k_min})"
                )
            if m.cfd_pmi_scale <= 0.0:
                raise ValueError(
                    f"multi_cortex.cfd_pmi_scale must be > 0.0, "
                    f"got {m.cfd_pmi_scale}"
                )
    return m


def _parse_allostasis(body: str) -> AllostasisConfig:
    """Parse an `allostasis { ... }` block into an `AllostasisConfig`.

    All keys are optional — anything omitted keeps the dataclass default
    (back-compat). Unknown keys are silently ignored (forward-compat,
    same contract as every other DSL sub-block parser).

    Validates:
      * 0 < load_ema_alpha ≤ 1, 0 < cort_ema_alpha ≤ 1 — bounded EMAs.
      * load_ema_alpha ≥ cort_ema_alpha — the time-scale-separation
        invariant. Equal alphas would erase the acute-vs-chronic
        distinction the controller is built on.
      * w_* ≥ 0 — negative weights would invert the stress signal.
      * gamma_* ≥ 0 — negative gammas would AMPLIFY rather than damp.
      * grad_norm_ceiling > 0.
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    a = AllostasisConfig()

    if "enabled" in props:
        a.enabled = _parse_bool(props["enabled"])

    # EMA constants
    if "load_ema_alpha" in props:
        a.load_ema_alpha = float(props["load_ema_alpha"])
    if "cort_ema_alpha" in props:
        a.cort_ema_alpha = float(props["cort_ema_alpha"])

    # Stress weights
    if "w_ne" in props:
        a.w_ne = float(props["w_ne"])
    if "w_gaba" in props:
        a.w_gaba = float(props["w_gaba"])
    if "w_loss" in props:
        a.w_loss = float(props["w_loss"])
    if "w_grad" in props:
        a.w_grad = float(props["w_grad"])

    # Stress baselines
    if "ne_baseline" in props:
        a.ne_baseline = float(props["ne_baseline"])
    if "gaba_baseline" in props:
        a.gaba_baseline = float(props["gaba_baseline"])
    if "grad_norm_ceiling" in props:
        a.grad_norm_ceiling = float(props["grad_norm_ceiling"])

    # Effector kill switches
    if "suppress_ne" in props:
        a.suppress_ne = _parse_bool(props["suppress_ne"])
    if "suppress_trophic" in props:
        a.suppress_trophic = _parse_bool(props["suppress_trophic"])
    if "suppress_lr" in props:
        a.suppress_lr = _parse_bool(props["suppress_lr"])

    # Effector damping strengths
    if "gamma_ne" in props:
        a.gamma_ne = float(props["gamma_ne"])
    if "gamma_trophic" in props:
        a.gamma_trophic = float(props["gamma_trophic"])
    if "gamma_lr" in props:
        a.gamma_lr = float(props["gamma_lr"])

    # Cross-field validation
    if not (0.0 < a.load_ema_alpha <= 1.0):
        raise ValueError(
            f"allostasis.load_ema_alpha must be in (0.0, 1.0], "
            f"got {a.load_ema_alpha}"
        )
    if not (0.0 < a.cort_ema_alpha <= 1.0):
        raise ValueError(
            f"allostasis.cort_ema_alpha must be in (0.0, 1.0], "
            f"got {a.cort_ema_alpha}"
        )
    if a.load_ema_alpha < a.cort_ema_alpha:
        raise ValueError(
            f"allostasis.load_ema_alpha ({a.load_ema_alpha}) must be >= "
            f"cort_ema_alpha ({a.cort_ema_alpha}) — the slow `cort` "
            "integrator must lag the fast `load` signal (10× ratio is "
            "physiological default; equal would erase the controller's "
            "acute-vs-chronic distinction)."
        )
    for nm, v in (("w_ne", a.w_ne), ("w_gaba", a.w_gaba),
                   ("w_loss", a.w_loss), ("w_grad", a.w_grad)):
        if v < 0:
            raise ValueError(
                f"allostasis.{nm} must be >= 0, got {v}"
            )
    for nm, v in (("gamma_ne", a.gamma_ne),
                   ("gamma_trophic", a.gamma_trophic),
                   ("gamma_lr", a.gamma_lr)):
        if v < 0:
            raise ValueError(
                f"allostasis.{nm} must be >= 0, got {v}"
            )
    if a.grad_norm_ceiling <= 0:
        raise ValueError(
            f"allostasis.grad_norm_ceiling must be > 0, "
            f"got {a.grad_norm_ceiling}"
        )
    return a


def _parse_fitness_objective(name: str, body: str) -> FitnessObjective:
    """Parse a single entry of the `objectives { ... }` table.

    Body shape: ``{ weight: 0.05, enabled: true, schedule: "gated" }``

    Validates:
      * weight >= 0       — negative weights flip the objective sign
                              and would push the loss toward +∞.
      * schedule in {"constant", "gated", "linear"} — typos like
        "exponentail" must fail at parse time, not silently produce a
        zero contribution.
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    o = FitnessObjective()
    if "enabled" in props:
        o.enabled = _parse_bool(props["enabled"])
    if "weight" in props:
        w = float(props["weight"])
        if w < 0:
            raise ValueError(
                f"fitness.objectives.{name}.weight must be >= 0, got {w}"
            )
        o.weight = w
    if "schedule" in props:
        sched = _strip_quotes(props["schedule"])
        if sched not in _VALID_FITNESS_SCHEDULES:
            raise ValueError(
                f"fitness.objectives.{name}.schedule={sched!r} not in "
                f"{sorted(_VALID_FITNESS_SCHEDULES)}"
            )
        o.schedule = sched
    return o


def _parse_fitness(body: str) -> FitnessConfig:
    """Parse a `fitness { ... }` block.

    Top-level keys:
      enabled                 — master switch (bool)
      objectives              — nested table of {name: {weight, enabled, schedule}}
      symbolic                — nested table of symbolic-specific knobs
      metabolic               — nested table of metabolic-specific knobs

    Validation:
      * objective names must be in _VALID_FITNESS_OBJECTIVES (catches typos)
      * symbolic.n_units > 0
      * 0 <= metabolic.budget <= 1
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    f = FitnessConfig()

    if "enabled" in props:
        f.enabled = _parse_bool(props["enabled"])

    # ── objectives sub-block ──
    if "objectives" in props:
        obj_body = _strip_braces(props["objectives"])
        obj_props = _split_top_level_kv(obj_body)
        for name, raw in obj_props.items():
            if name not in _VALID_FITNESS_OBJECTIVES:
                raise ValueError(
                    f"unknown fitness objective {name!r}; "
                    f"expected one of {sorted(_VALID_FITNESS_OBJECTIVES)}"
                )
            f.objectives[name] = _parse_fitness_objective(name, raw)

    # ── symbolic sub-block (per-objective extras) ──
    if "symbolic" in props:
        sym_body = _strip_braces(props["symbolic"])
        sp = _split_top_level_kv(sym_body)
        if "n_units" in sp:
            n = int(sp["n_units"])
            if n <= 0:
                raise ValueError(
                    f"fitness.symbolic.n_units must be > 0, got {n}"
                )
            f.symbolic_n_units = n
        if "n_features" in sp:
            nf = int(sp["n_features"])
            if nf <= 0:
                raise ValueError(
                    f"fitness.symbolic.n_features must be > 0, got {nf}"
                )
            f.symbolic_n_features = nf
        if "tau_init" in sp:
            f.symbolic_tau_init = float(sp["tau_init"])
        if "tau_final" in sp:
            f.symbolic_tau_final = float(sp["tau_final"])
        if "sparsity_weight" in sp:
            sw = float(sp["sparsity_weight"])
            if sw < 0:
                raise ValueError(
                    f"fitness.symbolic.sparsity_weight must be >= 0, got {sw}"
                )
            f.symbolic_sparsity_weight = sw

    # ── metabolic sub-block ──
    if "metabolic" in props:
        met_body = _strip_braces(props["metabolic"])
        mp = _split_top_level_kv(met_body)
        if "budget" in mp:
            b = float(mp["budget"])
            if not (0.0 <= b <= 1.0):
                raise ValueError(
                    f"fitness.metabolic.budget must be in [0.0, 1.0], got {b}"
                )
            f.metabolic_budget = b
        if "prune_threshold" in mp:
            pt = float(mp["prune_threshold"])
            if not (0.0 <= pt < 1.0):
                raise ValueError(
                    f"fitness.metabolic.prune_threshold must be in [0.0, 1.0), "
                    f"got {pt}"
                )
            f.metabolic_prune_threshold = pt

    # ── piso sub-block ──
    if "piso" in props:
        piso_body = _strip_braces(props["piso"])
        pp = _split_top_level_kv(piso_body)
        if "topology" in pp:
            f.piso_topology = _strip_quotes(pp["topology"])
        if "target_dim" in pp:
            f.piso_target_dim = int(pp["target_dim"])

    return f


def _parse_phase_gate(body: str) -> PhaseGate:
    """`{ center: 0.5, width: 0.10 }` -> PhaseGate."""
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    g = PhaseGate()
    if "center" in props: g.center = float(props["center"])
    if "width" in props:  g.width  = float(props["width"])
    return g


def _parse_mechanisms(body: str) -> MechanismsConfig:
    """Parse a `mechanisms { ... }` block.

    Each entry: `name: { strength|floor|period: ..., phase_gate: { ... } }`
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    m = MechanismsConfig()
    for name, mech_body in props.items():
        mech_props = _split_top_level_kv(_strip_braces(mech_body))
        gate = PhaseGate()
        if "phase_gate" in mech_props:
            gate = _parse_phase_gate(mech_props["phase_gate"])
        if name == "dropout":
            s = float(mech_props.get("strength", "0.1"))
            m.dropout = (s, gate)
        elif name == "pct_trunk":
            s = float(mech_props.get("strength", "1.0"))
            m.pct_trunk = (s, gate)
        elif name == "tonnetz":
            period = int(mech_props.get("period", "12"))
            bw     = int(mech_props.get("bandwidth", "3"))
            m.tonnetz = (period, bw, gate)
        elif name == "nemori":
            floor = float(mech_props.get("floor", "0.1"))
            m.nemori = (floor, gate)
        elif name == "bema":
            rw = int(mech_props.get("rollback_window", "50"))
            m.bema = (rw, gate)
        # Silently ignore unknown mechanism names (forward-compat).
    return m


def _parse_pass_marks(body: str) -> PassMarksConfig:
    """Parse a `pass_marks { ... }` block.

    Entry shape: `name: { metric: "train_ppl", at_step: 10000, max: 80 }`
                 `name: { metric: "ood_ppl", window: 2000, trend: "stable", tol: 0.02 }`
    """
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    rules = []
    for name, rule_body in props.items():
        rp = _split_top_level_kv(_strip_braces(rule_body))
        r = PassMark(name=name)
        if "metric" in rp:   r.metric  = _strip_quotes(rp["metric"])
        if "at_step" in rp:  r.at_step = int(rp["at_step"])
        if "max" in rp:      r.max     = float(rp["max"])
        if "min" in rp:      r.min     = float(rp["min"])
        if "window" in rp:   r.window  = int(rp["window"])
        if "tol" in rp:      r.tol     = float(rp["tol"])
        if "trend" in rp:    r.trend   = _strip_quotes(rp["trend"])
        if "action" in rp:   r.action  = _strip_quotes(rp["action"])
        if "min_evals" in rp: r.min_evals = int(rp["min_evals"])
        rules.append(r)
    return PassMarksConfig(rules=rules)


def _parse_loss_clipping(body: str) -> LossClippingConfig:
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    cfg = LossClippingConfig()
    if "enabled" in props:
        cfg.enabled = _parse_bool(props["enabled"])
    if "method" in props:
        method = _strip_quotes(props["method"])
        if method not in _VALID_LOSS_METHODS:
            raise ValueError(
                f"loss_clipping method {method!r}: expected one of {sorted(_VALID_LOSS_METHODS)}"
            )
        cfg.method = method
    if "factor" in props:
        cfg.factor = float(props["factor"])
    return cfg


def _parse_quantization(body: str) -> QuantizationConfig:
    body = _strip_braces(body)
    props = _split_top_level_kv(body)
    cfg = QuantizationConfig()
    if "enabled" in props:
        cfg.enabled = _parse_bool(props["enabled"])
    if "bits" in props:
        bits = int(props["bits"])
        if bits not in _VALID_QUANT_BITS:
            raise ValueError(
                f"quantization bits={bits}: expected one of {sorted(_VALID_QUANT_BITS)}"
            )
        cfg.bits = bits
    return cfg


# ── Loader: pull training config out of an architecture folder ─────────

def load_training_config_from_arch(arch_root) -> TrainingConfig:
    """Read arch.neuro and parse the `training { ... }` block if present.

    Returns defaults when:
      * no `training` block exists (most common)
      * arch.neuro doesn't exist (returns defaults so callers can degrade
        gracefully rather than crashing on a missing folder)
    """
    arch_path = Path(arch_root) / "arch.neuro"
    if not arch_path.is_file():
        return TrainingConfig()

    source = arch_path.read_text(encoding="utf-8")
    # Find `training { ... }` block, brace-aware
    body = _extract_block(source, "training")
    if body is None:
        return TrainingConfig()
    return parse_training_config(body)


# ── Global-defaults merge: workspace-level brian.toml → TrainingConfig ─

# Map ``ProjectConfig`` global-default attribute → ``TrainingConfig``
# field it fills when the arch is silent. Add a new pair here to teach
# the merge a new overridable knob (e.g. `("default_optimizer", "optimizer")`).
# ``hardware`` is intentionally absent until the harness consumes a
# string ``hardware`` field on ``TrainingConfig`` — for now the global
# ``default_hardware`` flows through the deploy script env var and the
# CLI's per-hardware preset map.
_GLOBAL_DEFAULTS_FIELD_MAP: List[Tuple[str, str]] = [
    ("default_preset", "preset"),
]


def _field_is_empty(value: Any) -> bool:
    """Treat empty strings, 0, None, and empty containers as "not set"."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (int, float)):
        # 0 is the "no opinion" sentinel for steps. Bools are also
        # ints in Python — falsy bools count as empty too.
        return value == 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def apply_global_defaults(
    arch_cfg: "TrainingConfig",
    project_cfg: Any,  # neuroslm.project_config.ProjectConfig
) -> List[Tuple[str, str, str]]:
    """Merge workspace-level defaults from ``brian.toml`` into ``arch_cfg``.

    **2026-06-12 precedence (REVERSED from sticky-overrides spec):**

      1. CLI flag                    (handled in the CLI, not here)
      2. Arch.neuro                  → any non-empty value WINS
      3. Global ``[defaults]``       → fills only EMPTY arch fields
      4. CLI / built-in fallback     (handled in the CLI)

    Operates in place on ``arch_cfg``. Returns ``[(field, old, new), ...]``
    listing every field this call changed, so the CLI can report::

        [global] preset: '' -> 'cheap_2k'

    The ``!`` parser support stays in :func:`parse_training_config` as
    a forward-compat marker, but is no longer consulted here because
    arch ALWAYS wins now.
    """
    changes: List[Tuple[str, str, str]] = []
    for proj_attr, arch_attr in _GLOBAL_DEFAULTS_FIELD_MAP:
        global_value = getattr(project_cfg, proj_attr, "") or ""
        if not global_value:
            continue                            # no opinion from the global
        old = getattr(arch_cfg, arch_attr, "")
        if not _field_is_empty(old):
            continue                            # arch already has a value — wins
        if old == global_value:
            continue                            # already matches, no change
        setattr(arch_cfg, arch_attr, global_value)
        changes.append((arch_attr, str(old), str(global_value)))
    return changes


# ── Helpers (mirror multifile.py's small parsers) ─────────────────────

def _strip_comments(source: str) -> str:
    """Replace `# ...` to end-of-line with spaces.

    Preserves line numbers and positions so other parsers see the same
    offsets, but removes comment characters that would otherwise confuse
    the brace/string walker (e.g. apostrophes in `Brain's reference run`
    were entering string mode and swallowing `{` / `}` counts).
    """
    out = []
    i, n = 0, len(source)
    in_str = None
    while i < n:
        ch = source[i]
        if in_str:
            out.append(ch)
            if ch == in_str and source[i - 1] != '\\':
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == '#':
            # Skip to end of line, replacing with spaces (preserve newline)
            while i < n and source[i] != '\n':
                out.append(' ')
                i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _extract_block(source: str, keyword: str) -> Optional[str]:
    """Find `<keyword> { ... }` at top level; return brace body or None."""
    # Strip comments first so apostrophes/braces inside `# ...` text can't
    # confuse the brace/string walker below.
    source = _strip_comments(source)
    pattern = re.compile(rf'\b{re.escape(keyword)}\s*\{{', re.MULTILINE)
    m = pattern.search(source)
    if not m:
        return None
    start = m.end() - 1  # position of `{`
    depth = 1
    i = start + 1
    in_str = None
    while i < len(source) and depth > 0:
        ch = source[i]
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError(f"unbalanced braces in `{keyword}` block")
    return source[start + 1 : i - 1]


def _split_top_level_kv_with_stickies(
    body: str,
) -> Tuple[Dict[str, str], Set[str]]:
    """Quote/paren/brace-aware key:value splitter with sticky-key support.

    A key suffixed with ``!`` (``preset!: "cheap_2k"``) is recorded in
    the returned ``sticky`` set with its bare name (``"preset"``).
    Sticky keys mean "this arch insists on the value — the global
    fallback in ``brian.toml`` ``[defaults]`` must not stomp it".

    Returns
    -------
    (props, sticky) : (Dict[str, str], Set[str])
        ``props`` is the normal key→value mapping with bare keys.
        ``sticky`` is the subset of keys that carried a ``!``.
    """
    out: Dict[str, str] = {}
    sticky: Set[str] = set()
    buf, depth, in_str = [], 0, None

    def flush() -> None:
        piece = "".join(buf).strip()
        if not piece or ":" not in piece:
            return
        k, v = piece.split(":", 1)
        key = k.strip()
        if key.endswith("!"):
            key = key[:-1].rstrip()
            sticky.add(key)
        out[key] = v.strip()

    for ch in body:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif (ch == "," or ch == "\n") and depth == 0:
            flush()
            buf = []
        else:
            buf.append(ch)
    flush()
    return out, sticky


def _split_top_level_kv(body: str) -> Dict[str, str]:
    """Legacy shim. Strips the ``!`` from sticky keys so existing
    callers that don't care about stickiness still see bare keys."""
    props, _sticky = _split_top_level_kv_with_stickies(body)
    return props


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1]
    return s


def _parse_bool(s: str) -> bool:
    return _strip_quotes(s).lower() in ("true", "yes", "1")
