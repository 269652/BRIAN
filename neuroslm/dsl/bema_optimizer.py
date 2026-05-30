# -*- coding: utf-8 -*-
"""Branching Ensemble Moving Average (BEMA) optimizer wrapper.

Stage 3 of the OOD push. Addresses the loss-spike pattern we keep hitting
(e.g. step 2360 of every recent run) by *rolling back optimizer updates*
when PPL trends up for N consecutive steps.

How it works (per-step):
  1. After harness.compute_loss(), record the LM loss.
  2. EMA-track loss → ema_loss; compute trend = ema_loss[t] - ema_loss[t-1].
  3. When trend > 0 for `rollback_window` consecutive steps:
       a. Discard the last `rollback_window` optimizer updates by
          restoring a snapshot of the parameters from before the rise.
       b. Reset adaptive optimizer state (AdamW moment buffers) to
          before-rise values too, so the recovery isn't immediately
          re-injected by the same gradient direction.
       c. Skip the next `cooldown` updates (no rollback during cooldown).
  4. Periodically snapshot params + optimizer state so a future rollback
     can restore them.

This is the "Branching" in BEMA: each snapshot is a branch; if the
current branch climbs PPL, fall back to a known-good branch. The
"Ensemble Moving Average" is the EMA used to detect a rise.

Implementation is **wrapper-only** (doesn't replace the optimizer):
existing AdamW / Adafactor instances are wrapped via `wrap_with_bema()`,
and `harness.train_step()` calls `bema.maybe_step()` instead of
`optimizer.step()` when BEMA is enabled.
"""
from __future__ import annotations
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch


@dataclass
class BEMAConfig:
    enabled: bool = False
    snapshot_every: int = 50         # steps between snapshots
    rollback_window: int = 50        # PPL-rising streak that triggers rollback
    max_snapshots: int = 4           # keep most-recent N snapshots
    cooldown: int = 100              # steps to skip rollback detection after a rollback
    ema_alpha: float = 0.05          # EMA smoothing for loss tracking


@dataclass
class BEMAState:
    """Live state of the BEMA controller (not user-tunable)."""
    step: int = 0
    ema_loss: float = float("nan")
    prev_ema_loss: float = float("nan")
    rising_streak: int = 0
    cooldown_left: int = 0
    rollbacks_performed: int = 0
    # Each snapshot = (step, param_state_dict, optimizer_state_dict)
    snapshots: Deque = field(default_factory=lambda: deque(maxlen=4))


class BEMAController:
    """Wrap an optimizer with PPL-rise rollback.

    Usage:
        bema = BEMAController(model, optimizer, BEMAConfig(enabled=True))
        # In train_step, AFTER loss.backward() AND clip_grad_norm_:
        bema.maybe_step(loss_value=float(loss.item()))
    """

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 cfg: BEMAConfig):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.state = BEMAState(snapshots=deque(maxlen=cfg.max_snapshots))

    def _snapshot(self) -> None:
        """Save a deep copy of params + optimizer state."""
        params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            opt_state = copy.deepcopy(self.optimizer.state_dict())
        except Exception:
            opt_state = None
        self.state.snapshots.append((self.state.step, params, opt_state))

    def _rollback(self) -> Tuple[int, bool]:
        """Restore the OLDEST snapshot still in the deque (rollback the
        full rising window). Returns (steps_rolled_back, success)."""
        if not self.state.snapshots:
            return 0, False
        rb_step, params, opt_state = self.state.snapshots[0]
        try:
            self.model.load_state_dict(params, strict=False)
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
        except Exception:
            return 0, False
        rolled = self.state.step - rb_step
        # Drop ALL snapshots — we just reset to the oldest; everything
        # after it is now stale.
        self.state.snapshots.clear()
        self.state.rollbacks_performed += 1
        self.state.cooldown_left = self.cfg.cooldown
        self.state.rising_streak = 0
        self.state.ema_loss = float("nan")
        return rolled, True

    def maybe_step(self, loss_value: float) -> Dict:
        """Run optimizer.step() then track for rollback. Returns a dict
        of metrics the train loop can log."""
        self.optimizer.step()
        self.state.step += 1
        info = {"bema_rollback": False, "bema_rolled_steps": 0,
                 "bema_streak": 0, "bema_ema_loss": loss_value}

        if not self.cfg.enabled:
            return info

        # Loss EMA
        if self.state.ema_loss != self.state.ema_loss:   # NaN check
            self.state.ema_loss = loss_value
        else:
            self.state.prev_ema_loss = self.state.ema_loss
            self.state.ema_loss = (
                (1 - self.cfg.ema_alpha) * self.state.ema_loss
                + self.cfg.ema_alpha * loss_value)

        info["bema_ema_loss"] = self.state.ema_loss

        # Cooldown — don't check or rollback inside cooldown window
        if self.state.cooldown_left > 0:
            self.state.cooldown_left -= 1
        else:
            # Rising streak detection (vs previous EMA)
            if (self.state.prev_ema_loss == self.state.prev_ema_loss
                    and self.state.ema_loss > self.state.prev_ema_loss):
                self.state.rising_streak += 1
            else:
                self.state.rising_streak = 0
            info["bema_streak"] = self.state.rising_streak

            # Rollback trigger
            if self.state.rising_streak >= self.cfg.rollback_window:
                rolled, ok = self._rollback()
                if ok:
                    info["bema_rollback"] = True
                    info["bema_rolled_steps"] = rolled

        # Periodic snapshot
        if (self.state.step % self.cfg.snapshot_every == 0
                and self.state.cooldown_left == 0):
            self._snapshot()

        return info
