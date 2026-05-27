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
from contextlib import nullcontext
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.training_config import TrainingConfig


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
        return h

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
        """Cross-entropy loss with optional per-sample clipping.

        Per-sample clipping (p4 fix): compute per-sequence mean loss,
        clip each at `factor * batch_median`, then average. Suppresses
        outlier-sequence dominance of the gradient.
        """
        logits = self(ids, nt_levels=nt_levels)
        loss = self._compute_loss_from_logits(logits, targets)
        return loss

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
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), clip if (clip and clip > 0) else 1e9)
                optimizer.step()
            self._last_gnorm = float(gnorm)
            optimizer.zero_grad(set_to_none=True)
            self._accum_step = 0

        self._last_lr = float(optimizer.param_groups[0]["lr"])
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
