# -*- coding: utf-8 -*-
"""Geometric Information Funnel (GIF) — Python backing for lib/gif.neuro.

Three mechanisms that fix the train-PPL / OOD-PPL gap divergence
identified in the GPT-2 vs SmolLM2 10k forensic (2026-06-16):

1. ``VBBAlphaSchedule`` — piecewise-linear ramp of ``vbb_alpha`` from
   a loose 0.001 during infancy to a tight 0.05 at maturity.
2. ``OODProbe`` — cached held-out OOD sequences + periodic CE eval
   that replaces the pre-fusion EMA with a true generalisation signal.
3. Isotropy schedule — ramped via the same schedule (reuses the
   existing ``IsotropyLoss`` in ``neuroslm/regularizers.py``).

All three ramps now share a single *progress* variable (0→1) that can
be driven adaptively from the gap ratio instead of fixed step counts.
When ``adaptive: true``, progress advances faster when the gap ratio
exceeds ``target_gap_ratio`` and holds steady when it's below target.
The static step-based schedule is kept as a minimum floor.

Contracts: tests/dsl/test_gif_config.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F


# ── GIF-1: VBB α schedule ──────────────────────────────────────────

@dataclass
class VBBAlphaSchedule:
    """Piecewise-linear ramp for the VBB information bottleneck weight.

    Matches ``gif_vbb_alpha_schedule`` in lib/gif.neuro exactly.
    When used inside the adaptive GIFController, the ``step`` passed
    to ``__call__`` is ignored in favour of the shared ``progress``.
    """
    alpha_start: float = 0.001
    alpha_end: float = 0.05
    ramp_start: int = 500
    ramp_end: int = 3000

    def __call__(self, step: int, *, progress: Optional[float] = None) -> float:
        """Return α for the given step or explicit progress (0-1)."""
        if progress is not None:
            p = max(0.0, min(1.0, progress))
        else:
            # Static fallback
            if step < self.ramp_start:
                p = 0.0
            elif step >= self.ramp_end:
                p = 1.0
            else:
                p = (step - self.ramp_start) / max(1, self.ramp_end - self.ramp_start)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * p

    @classmethod
    def from_config(cls, cfg) -> "VBBAlphaSchedule":
        """Build from a training_config that has gif.* keys.

        Accepts both the Python-native names (vbb_alpha_start/end) and
        the DSL-friendly names (vbb_alpha_min/max) for resilience.
        """
        gif = getattr(cfg, "gif", None) or {}
        if isinstance(gif, dict):
            return cls(
                alpha_start=float(gif.get("vbb_alpha_start",
                                   gif.get("vbb_alpha_min", 0.001))),
                alpha_end=float(gif.get("vbb_alpha_end",
                                 gif.get("vbb_alpha_max", 0.05))),
                ramp_start=int(gif.get("vbb_alpha_ramp_start",
                                gif.get("vbb_ramp_start", 500))),
                ramp_end=int(gif.get("vbb_alpha_ramp_end",
                              gif.get("vbb_ramp_end", 3000))),
            )
        return cls()


# ── GIF-2: OOD probe ───────────────────────────────────────────────

class OODProbe:
    """Maintains a cached set of OOD sequences and evaluates trunk CE.

    Usage in the harness::

        probe = OODProbe.from_config(training_config)
        probe.maybe_load(tokenizer, device)  # loads OOD sequences once

        # Inside train_step, every probe_every steps:
        if probe.should_eval(step):
            ce = probe.evaluate(language_model, device)
            # ce is the mean per-token CE on the OOD probe set
    """

    def __init__(
        self,
        n_seqs: int = 50,
        max_len: int = 512,
        probe_every: int = 100,
        ema_alpha: float = 0.1,
        dataset_name: str = "Salesforce/wikitext",
        dataset_config: str = "wikitext-103-v1",
        split: str = "test",
    ):
        self.n_seqs = n_seqs
        self.max_len = max_len
        self.probe_every = probe_every
        self.ema_alpha = ema_alpha
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split

        # State
        self._input_ids: Optional[torch.Tensor] = None  # (N, L)
        self._loaded = False
        self._ema: float = 0.0
        self._n_evals: int = 0

    @classmethod
    def from_config(cls, cfg) -> "OODProbe":
        """Build from training_config.gif dict.

        Accepts both Python-native names (ood_probe_seqs/every/ema_alpha)
        and the DSL-friendly names (probe_n_seqs/probe_every/probe_ema_beta).
        """
        gif = getattr(cfg, "gif", None) or {}
        if isinstance(gif, dict):
            return cls(
                n_seqs=int(gif.get("ood_probe_seqs",
                            gif.get("probe_n_seqs", 50))),
                max_len=int(gif.get("ood_probe_max_len", 512)),
                probe_every=int(gif.get("ood_probe_every",
                                 gif.get("probe_every", 100))),
                ema_alpha=float(gif.get("ood_probe_ema_alpha",
                                 1.0 - float(gif.get("probe_ema_beta", 0.9)))),
            )
        return cls()

    def maybe_load(self, tokenizer, device: torch.device) -> bool:
        """Load and tokenise OOD sequences once. Returns True on success."""
        if self._loaded:
            return True
        try:
            from datasets import load_dataset
            ds = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                trust_remote_code=True,
            )
            # Filter to non-empty paragraphs
            texts = [r["text"] for r in ds if r["text"].strip()]
            # Deterministic sample from the start (reproducible)
            texts = texts[: self.n_seqs * 3]  # oversample for filtering

            ids_list: List[torch.Tensor] = []
            for t in texts:
                if len(ids_list) >= self.n_seqs:
                    break
                enc = tokenizer.encode(t)
                if hasattr(enc, "ids"):
                    toks = enc.ids
                elif isinstance(enc, list):
                    toks = enc
                else:
                    toks = list(enc)
                if len(toks) < 16:
                    continue
                toks = toks[: self.max_len]
                ids_list.append(torch.tensor(toks, dtype=torch.long))

            if not ids_list:
                return False

            # Pad to uniform length
            max_l = max(t.size(0) for t in ids_list)
            padded = torch.zeros(len(ids_list), max_l, dtype=torch.long)
            for i, t in enumerate(ids_list):
                padded[i, : t.size(0)] = t

            self._input_ids = padded.to(device)
            self._loaded = True
            return True
        except Exception as e:
            print(f"[gif] OOD probe load failed: {e}", flush=True)
            return False

    def should_eval(self, step: int) -> bool:
        """Whether we should run the probe at this step."""
        return self._loaded and step > 0 and step % self.probe_every == 0

    @torch.no_grad()
    def evaluate(self, language_model, device: torch.device) -> float:
        """Evaluate trunk-only CE on cached OOD sequences.

        Returns the mean per-token cross-entropy (in nats). Updates
        the internal EMA.
        """
        if self._input_ids is None:
            return self._ema

        was_training = language_model.training
        language_model.eval()
        try:
            ids = self._input_ids.to(device)
            total_ce = 0.0
            total_tokens = 0

            # Evaluate in small batches to avoid OOM
            batch_size = min(8, ids.size(0))
            for start in range(0, ids.size(0), batch_size):
                batch = ids[start : start + batch_size]
                inp = batch[:, :-1]
                tgt = batch[:, 1:]

                # Forward through the trunk only (no cortex fusion)
                out = language_model(inp)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out

                B, T, V = logits.shape
                ce = F.cross_entropy(
                    logits.float().reshape(-1, V),
                    tgt.reshape(-1),
                    reduction="sum",
                    ignore_index=0,
                )
                # Count non-padding tokens
                n_tok = (tgt != 0).sum().item()
                total_ce += ce.item()
                total_tokens += max(n_tok, 1)

            mean_ce = total_ce / max(total_tokens, 1)

            # Update EMA
            if self._n_evals == 0:
                self._ema = mean_ce  # seed
            else:
                self._ema = (
                    (1 - self.ema_alpha) * self._ema
                    + self.ema_alpha * mean_ce
                )
            self._n_evals += 1

            return mean_ce
        finally:
            if was_training:
                language_model.train()

    @property
    def ema(self) -> float:
        """Current EMA of the OOD probe CE."""
        return self._ema

    @property
    def is_ready(self) -> bool:
        """Whether at least one eval has been run."""
        return self._n_evals > 0


# ── GIF-3: Isotropy schedule ──────────────────────────────────────

@dataclass
class IsotropySchedule:
    """Ramps isotropy weight from 0 to weight_max in lockstep with VBB.

    Matches ``gif_isotropy_schedule`` in lib/gif.neuro.
    When used inside the adaptive GIFController, the ``step`` passed
    to ``__call__`` is ignored in favour of the shared ``progress``.
    """
    weight_max: float = 0.01
    ramp_start: int = 500
    ramp_end: int = 3000

    def __call__(self, step: int, *, progress: Optional[float] = None) -> float:
        if progress is not None:
            p = max(0.0, min(1.0, progress))
        else:
            if step < self.ramp_start:
                p = 0.0
            elif step >= self.ramp_end:
                p = 1.0
            else:
                p = (step - self.ramp_start) / max(1, self.ramp_end - self.ramp_start)
        return self.weight_max * p

    @classmethod
    def from_config(cls, cfg) -> "IsotropySchedule":
        gif = getattr(cfg, "gif", None) or {}
        if isinstance(gif, dict):
            return cls(
                weight_max=float(gif.get("isotropy_weight_max",
                                  gif.get("iso_weight_max", 0.01))),
                ramp_start=int(gif.get("isotropy_ramp_start",
                                gif.get("vbb_ramp_start", 500))),
                ramp_end=int(gif.get("isotropy_ramp_end",
                              gif.get("vbb_ramp_end", 3000))),
            )
        return cls()


# ── Composite ──────────────────────────────────────────────────────

@dataclass
class GIFController:
    """Top-level controller that owns all three GIF mechanisms.

    Constructed once at harness init; queried each step for:
    - ``vbb_alpha(step)`` → scheduled α for the VBB loss
    - ``isotropy_weight(step)`` → scheduled weight for the isotropy reg
    - ``ood_probe`` → the OODProbe instance for EMA-based cortex gating

    **Adaptive mode** (``adaptive=True``):

    A single ``progress`` variable (0→1) drives both the VBB α and
    isotropy weight.  Each step, ``update()`` is called with the
    current ``lm_loss_ema`` from the harness.  When the OOD probe has
    data, it computes the PPL-space gap ratio:

        gap_ratio = exp(ood_ce_ema - lm_loss_ema)

    A proportional controller advances progress:

        error    = max(0, gap_ratio - target_gap_ratio)
        Δp       = ramp_gain × error + min_ramp_speed
        progress = clamp(progress + Δp, 0, 1)

    The static step-based schedule is kept as a *floor*: progress
    never falls below what the old linear ramp would give.  This
    guarantees that even if the OOD probe hasn't loaded yet, the
    ramp still advances at its baseline pace.

    When ``adaptive=False``, behaviour is identical to the original
    fixed step-based schedule.
    """
    vbb_schedule: VBBAlphaSchedule = field(default_factory=VBBAlphaSchedule)
    isotropy_schedule: IsotropySchedule = field(default_factory=IsotropySchedule)
    ood_probe: OODProbe = field(default_factory=OODProbe)
    enabled: bool = False

    # ── Adaptive ramp state ──
    adaptive: bool = False
    target_gap_ratio: float = 1.5     # desired PPL gap ceiling
    ramp_gain: float = 0.0002         # progress per unit gap error per step
    min_ramp_speed: float = 0.00005   # minimum Δp per step (prevents stall)
    _progress: float = 0.0            # shared ramp progress 0→1
    _last_gap_ratio: float = 0.0      # for telemetry

    def vbb_alpha(self, step: int) -> float:
        if self.adaptive:
            return self.vbb_schedule(step, progress=self._progress)
        return self.vbb_schedule(step)

    def isotropy_weight(self, step: int) -> float:
        if self.adaptive:
            return self.isotropy_schedule(step, progress=self._progress)
        return self.isotropy_schedule(step)

    @property
    def progress(self) -> float:
        """Current ramp progress (0→1)."""
        return self._progress

    @property
    def last_gap_ratio(self) -> float:
        """Most recent gap ratio used by the adaptive controller."""
        return self._last_gap_ratio

    def update(self, step: int, lm_loss_ema: float) -> None:
        """Advance the adaptive ramp based on current gap ratio.

        Called once per step from the harness.  When ``adaptive=False``
        this is a no-op.

        Args:
            step:  current global step (for the static floor)
            lm_loss_ema:  harness train-loss EMA (CE, nats)
        """
        if not self.adaptive:
            return

        # Static floor: what the old linear ramp would give
        rs = self.vbb_schedule.ramp_start
        re = self.vbb_schedule.ramp_end
        if step < rs:
            static_p = 0.0
        elif step >= re:
            static_p = 1.0
        else:
            static_p = (step - rs) / max(1, re - rs)

        # Adaptive component: driven by gap ratio when OOD data ready
        if self.ood_probe.is_ready and lm_loss_ema > 0.5:
            ood_ce = self.ood_probe.ema
            # Gap ratio in PPL space: exp(ood_ce) / exp(lm_ce)
            gap_ratio = math.exp(
                max(-20.0, min(20.0, ood_ce - lm_loss_ema))
            )
            self._last_gap_ratio = gap_ratio

            error = max(0.0, gap_ratio - self.target_gap_ratio)
            delta = self.ramp_gain * error + self.min_ramp_speed
            self._progress = min(1.0, self._progress + delta)
        # else: No OOD data yet — do NOT advance progress.
        # The static floor (below) provides the minimum baseline.
        # Advancing min_ramp_speed blind inflates VBB loss when KL is
        # enormous (25k+ nats early), drowning LM signal.

        # Never fall below the static floor
        self._progress = max(self._progress, static_p)

    @classmethod
    def from_config(cls, cfg) -> "GIFController":
        """Build all three mechanisms from training_config.gif."""
        gif = getattr(cfg, "gif", None) or {}
        enabled = bool(gif.get("enabled", False)) if isinstance(gif, dict) else False
        if not enabled:
            return cls(enabled=False)

        adaptive = bool(gif.get("adaptive", False))
        target_gap = float(gif.get("target_gap_ratio", 1.5))
        gain = float(gif.get("ramp_gain", 0.0002))
        min_speed = float(gif.get("min_ramp_speed", 0.00005))

        return cls(
            vbb_schedule=VBBAlphaSchedule.from_config(cfg),
            isotropy_schedule=IsotropySchedule.from_config(cfg),
            ood_probe=OODProbe.from_config(cfg),
            enabled=True,
            adaptive=adaptive,
            target_gap_ratio=target_gap,
            ramp_gain=gain,
            min_ramp_speed=min_speed,
        )
