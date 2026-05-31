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
from typing import Any, Dict, List, Optional, Tuple


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
    steps: int = 10000
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
    # Hardware envelope + multi-scale variants. The deploy script reads
    # `hardware` to filter vast.ai offers; the harness reads `scales` +
    # the SCALE env var to pick which variant to instantiate.
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    scales: ScalesConfig = field(default_factory=ScalesConfig)
    # Declarative metric-exposure points. Each entry says "compute this
    # metric and publish it at these node tags." Reduces data overhead by
    # not exposing every metric everywhere.
    metric_exposures: List[MetricExpose] = field(default_factory=list)


# ── Constants for validation ───────────────────────────────────────────

_VALID_LOSS_METHODS = {"per_sample"}
_VALID_QUANT_BITS = {4, 8, 16}
_VALID_OPTIMIZERS = {"adamw", "adafactor"}


# ── Parser ─────────────────────────────────────────────────────────────

def parse_training_config(body: str) -> TrainingConfig:
    """Parse the body of a `training { ... }` block (without the braces).

    Empty body → all defaults. Unknown top-level keys are silently ignored
    (forward-compat with future versions of this schema). Invalid values
    for known keys raise ValueError.
    """
    cfg = TrainingConfig()

    props = _split_top_level_kv(body)

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
    if "hardware" in props:
        cfg.hardware = _parse_hardware(props["hardware"])
    if "scales" in props:
        cfg.scales = _parse_scales(props["scales"], cfg.hardware)

    # `metric <name> { ... }` blocks live at the same level as `mechanisms`
    # / `pass_marks`. Collect all of them.
    cfg.metric_exposures = _parse_metric_exposures(props)
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
    return g


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


def _split_top_level_kv(body: str) -> Dict[str, str]:
    """Quote/paren/brace-aware key:value splitter."""
    out: Dict[str, str] = {}
    buf, depth, in_str = [], 0, None

    def flush() -> None:
        piece = "".join(buf).strip()
        if not piece or ":" not in piece:
            return
        k, v = piece.split(":", 1)
        out[k.strip()] = v.strip()

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
    return out


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
