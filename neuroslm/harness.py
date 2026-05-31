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
        # Runtime metric registry — values queryable from anywhere in the
        # control flow node (other modules, schedulers, gene triggers).
        # Updated once per compute_loss step.
        self._metrics: Dict[str, float] = {}

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
        h._metrics = {}
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
        loss_lm = self._compute_loss_from_logits(logits, targets)
        total = self.total_loss_config.w_lm * loss_lm
        mat = self.maturity.value()
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
        # Record the LM-portion so train_step can update MAT after the
        # backward pass without a second forward.
        self._last_lm_loss_value = float(loss_lm.detach().item())

        # ── Runtime metric registry update ──
        # Compute a runtime Phi proxy from the per-token logit entropy:
        #   higher entropy across the vocab = more "differentiated"
        #   predictions = higher integrated information proxy.
        # Cheap to compute (single softmax on already-computed logits).
        with torch.no_grad():
            logprobs = F.log_softmax(logits.detach(), dim=-1)
            ent = -(logprobs.exp() * logprobs).sum(dim=-1)        # (B, T)
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

        Uses the PREVIOUS step's metrics (mat/surprise/phi) as the
        regulatory context — the natural 1-step latency mirrors real
        gene transcription. Returns the auxiliary `phi_loss` tensor
        (a live tensor in the autograd graph for THIS step).
        """
        self._ensure_genetics()
        if (self._transmitter_sys.level.shape[0] != batch_size
                or self._transmitter_sys.level.device != device):
            self._transmitter_sys.to(device)
            self._transmitter_sys.reset(batch_size, device)
        nt_levels = self._transmitter_sys.level   # (B, N_NT)
        # Previous-step metrics — zero on the very first call.
        prev_mat = float(self._metrics.get("mat", 0.0))
        prev_surprise = float(self._metrics.get("surprise", 0.0))
        prev_phi = float(self._metrics.get("phi", 0.0))
        surprise_t = torch.full((batch_size,), prev_surprise, device=device)
        mat_t      = torch.full((batch_size,), prev_mat,      device=device)
        phi_t      = torch.full((batch_size,), prev_phi,      device=device)
        ctx = self._genetics_orch.build_context(nt_levels, surprise_t, mat_t)
        out = self._genetics_orch(ctx, phi=phi_t)
        # Transmitter state overlay (detached — it's a per-step state buffer)
        self._transmitter_sys.set_module_offsets(
            baseline_off=out["baseline_offsets"].detach(),
            tau_shift=out["tau_shifts"].detach(),
        )
        # LIVE (non-detached) modulation into the cortex — this is the
        # path through which gene gradients flow back from the LM loss.
        if (self.language_model is not None
                and hasattr(self.language_model, "set_nt_modulation")):
            self.language_model.set_nt_modulation(out["baseline_offsets"])
        diag = self._genetics_orch.diagnostics()
        for k, v in diag.items():
            self._metrics["gene_" + k if not k.startswith("gene_") else k] = v
        return out["phi_loss"]

    def _compute_loss_from_logits(self, logits: torch.Tensor,
                                  targets: torch.Tensor) -> torch.Tensor:
        batch, seq_len, vocab = logits.shape
        ls = self.training_config.label_smoothing
        clip = self.training_config.loss_clipping

        if not clip.enabled:
            # Standard cross-entropy with optional label smoothing
            return F.cross_entropy(
                logits.reshape(-1, vocab),
                targets.reshape(-1),
                label_smoothing=ls,
            )

        # Per-sample clipping: compute per-token CE, average to per-sequence,
        # clip each sequence at `factor * batch_median`, then average.
        per_token = F.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
            label_smoothing=ls,
        )
        per_token = per_token.reshape(batch, seq_len)
        per_seq = per_token.mean(dim=1)             # (batch,)

        # Median-based clip threshold; detached so the threshold itself
        # isn't a gradient target (otherwise the clip dampens its own
        # learning signal).
        threshold = (per_seq.detach().median() * clip.factor).clamp(min=1e-8)
        clipped = torch.minimum(per_seq, threshold)
        return clipped.mean()

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
            clip = self.training_config.grad_clip
            # clip_grad_norm_ returns the total norm *before* clipping —
            # capture it for native-format logging (gnorm).
            if self._grad_scaler is not None:
                self._grad_scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), clip if (clip and clip > 0) else 1e9)
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
        return float(loss.detach().item())

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
            if self.training_config.mu_p_scaling:
                groups = _mu_p_param_groups(self, base_lr=lr, wd=wd)
                self._optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
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

    def save_checkpoint(self, path: str, step: int = 0,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist model + optimizer state + step. Mirrors Brain's format
        loosely enough that a future merger can interoperate."""
        payload = {
            "step": step,
            "model": self.state_dict(),
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
        """Load model + optimizer state. Returns the saved step."""
        payload = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(payload["model"])
        if "optimizer" in payload and self._optimizer is not None:
            self._optimizer.load_state_dict(payload["optimizer"])
        return int(payload.get("step", 0))

    # ── Introspection (for train.py compatibility) ──────────────────

    def topology_summary(self) -> str:
        """Human-readable topology string. Mirrors Brain.topology_summary().
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_pops = sum(1 for _ in self.circuit.children())
        return (
            f"BRIANHarness:\n"
            f"  vocab_size = {self.vocab_size}\n"
            f"  d_sem      = {self.d_sem}\n"
            f"  parameters = {n_params:,}\n"
            f"  circuit populations = {n_pops}\n"
            f"  sink population = {self.sink_population}\n"
            f"  loss clipping = {self.training_config.loss_clipping.enabled} "
            f"(factor={self.training_config.loss_clipping.factor})\n"
            f"  optimizer = {self.training_config.optimizer} "
            f"(lr={self.training_config.learning_rate})"
        )
