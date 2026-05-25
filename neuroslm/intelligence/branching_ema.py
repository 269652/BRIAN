"""Surprise-Gated Branching EMA — divergence-resilient parameter averaging.

Addresses the synth-v1 'awakening collapse' pathology observed in the
2026-05-25 10k training run:

    step  2600: ppl  85   ← getting low
    step  3800: ppl  59   ← BEST point ever reached
    step  4800: ppl 257   ← major spike
    step  5600: ppl 426   ← catastrophic spike
    step  5800-10000: stuck at ppl 200-270, NEVER recovers to step-3800

best.pt got saved at step 4000 with lm_ema 4.748 (ppl ~115), well above
the actual lowest moment (step 3800 ppl 59) — the model touched the
low-loss basin but couldn't stay there. Aux losses, awakening dynamics,
and gradient noise progressively drove the trunk away from its best
trajectory; lm_ema EMA-smoothed the divergence so 'best' tracker missed
the lowest moments and saved the wrong point.

Mechanism (the user's spec, condensed)
--------------------------------------
A *meta-optimizer* layered on top of the standard optimizer. Two
shadow parameter sets are maintained:

  * params_stable  — a slow EMA of recent stable weights. Reverted to
                     when training is going pathologically (rising PPL).
  * params_best    — snapshot at the historical lowest EMA-smoothed PPL.
                     The 'free energy minimum' — what we collapse to at
                     checkpoint / inference time.

The mixing rate alpha_eff between current model weights and params_stable
is gated by the PPL velocity d(ppl)/dt:

    alpha_eff(t) = (1 / avg_ppl(t)) * exp(-gamma * |d(ppl)/dt|)

  - PPL rising fast → alpha_eff ≈ 0 → params_stable freezes (we don't
    let the bad trajectory contaminate the EMA).
  - PPL flat or falling → alpha_eff > 0 → EMA absorbs the current
    weights as 'stable'.

The 1/avg_ppl factor down-weights early training (when PPL is huge but
that's just initial descent, not divergence). gamma is the surprise
sensitivity.

At checkpoint save time, OR if a 'collapse trigger' fires (e.g. PPL
spike > N×best for K steps), we copy params_best → model — this is the
'Free-Energy Collapse'.

How this differs from existing related work
--------------------------------------------
* nn.AveragedModel / SWA      — uniform averaging over all steps; we
                                exponentially weight by PPL and gate by
                                d(PPL)/dt.
* LookAhead (Zhang 2019)      — fast/slow weights with FIXED alpha. We
                                make alpha *PPL-velocity-dependent*.
* Polyak averaging            — same as SWA at fixed lr.
* SAM (Foret 2020)            — flat-minima search via gradient ascent
                                on the loss landscape. We're not finding
                                flat minima; we're protecting the trunk
                                from EXISTING good minima being
                                vandalized by later bad updates.
* Checkpoint averaging        — happens post-hoc on saved ckpts; we do
                                it ONLINE with adaptive damping.

The novel contribution is the **velocity-gated mixing** which gives
asymmetric protection: easy to be 'absorbed' into params_stable when
training is good, hard when training is going wrong.

References (related literature)
-------------------------------
Zhang et al. (2019, NeurIPS) — LookAhead Optimizer.
Izmailov et al. (2018, UAI)  — Stochastic Weight Averaging (SWA).
Foret et al. (2020, ICLR)    — Sharpness-Aware Minimization.
"""
from __future__ import annotations
from collections import deque
import math
from typing import Optional
import torch
import torch.nn as nn


class BranchingEMA:
    """Surprise-gated EMA over a model's parameters.

    Parameters
    ----------
    model           : the nn.Module whose parameters are tracked. Held by
                      reference; we never re-bind, just snapshot weights.
    history_len     : number of past PPL samples kept for velocity
                      estimation. 10 is a reasonable default — enough
                      to smooth noise but short enough to react.
    gamma           : surprise-sensitivity. Higher → more aggressive
                      freezing when PPL rises. Default 0.5 gives
                      alpha_eff ≈ 0.6×base when |dPPL/dt|=1, ≈0.14×base
                      when |dPPL/dt|=4. The dPPL/dt is computed on
                      PPL VALUES (e.g. 100 → 200 is delta=100), not
                      log-ppl, so gamma is in 1/ppl-units.
    base_alpha_cap  : ceiling on alpha_eff so a tiny avg_ppl can't make
                      the EMA collapse to the latest weights in one step.
                      Default 0.01 (1% absorption per call). At avg_ppl=
                      50 the 1/avg_ppl factor is 0.02, which the cap
                      clips to 0.01.
    update_every    : call branching_ema.maybe_update(model, ppl) once
                      per train step. Internally we only update the EMA
                      every `update_every` steps (default 1 = every
                      step). Set higher to reduce overhead.

    Usage
    -----
        bema = BranchingEMA(brain, gamma=0.5)
        for step in range(steps):
            ...
            out = brain.forward_lm(ids, targets=...)
            out['loss'].backward()
            optimizer.step()
            bema.maybe_update(brain, ppl=ppl_now, step=step)
            # at save time:
            bema.maybe_collapse_to_best(brain, ppl_now)
            # at periodic save:
            bema.save_state(path)
    """

    def __init__(self, model: nn.Module,
                 history_len: int = 10,
                 gamma: float = 5.0,
                 base_alpha_cap: float = 0.01,
                 update_every: int = 1,
                 best_ema_alpha: float = 0.1):
        # gamma=5.0 in log-PPL units gives:
        #   mild rise   (1.2× per step, v≈0.18): gate = exp(-0.9)  ≈ 0.40
        #   moderate    (1.5×,         v≈0.40): gate = exp(-2.0)  ≈ 0.13
        #   sharp rise  (2.0×,         v≈0.69): gate = exp(-3.5)  ≈ 0.03
        #   catastrophic(5.0×,         v≈1.61): gate = exp(-8.0)  ≈ 3e-4
        # i.e. alpha is meaningfully attenuated for any rise, and crushed
        # toward zero for the catastrophic spike pattern we saw at synth-v1
        # step 5600 (ppl 426 against an ~80 basin).
        self.history: deque = deque(maxlen=int(history_len))
        self.gamma = float(gamma)
        self.base_alpha_cap = float(base_alpha_cap)
        self.update_every = max(1, int(update_every))
        self.best_ema_alpha = float(best_ema_alpha)

        # Two shadow param sets, one tensor per trainable param. Stored
        # on the same device as the model param (so updates are cheap).
        # Not registered as nn.Parameter — these aren't trained, just
        # bookkeeping.
        self._stable: dict[str, torch.Tensor] = {}
        self._best: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._stable[name] = p.detach().clone()
            self._best[name]   = p.detach().clone()

        # Tracks the lowest EMA-smoothed PPL we've seen — `params_best`
        # is the snapshot at that point. EMA smoothing avoids snapshotting
        # at a single lucky-batch dip.
        self._best_ema_ppl: float = float('inf')
        self._ppl_ema: Optional[float] = None
        self._n_collapses: int = 0
        self._n_freezes: int = 0   # how often alpha_eff clipped to ~0

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------
    def _ppl_velocity(self) -> float:
        """Discrete derivative d(log_ppl)/dt over the last few samples.

        We use LOG-PPL not raw PPL: raw differences scale with the PPL
        magnitude (a step from 5000->4000 is 1000, from 100->50 is 50),
        so gamma would need per-regime tuning. Log-PPL puts the velocity
        in dimensionless natural units of 'how many log-points are we
        moving per step' — gamma=0.5 then means alpha_eff drops by factor
        exp(-0.5) ≈ 0.6 for every 1 nat/step of upward drift.

        Returns 0.0 if history is too short. Positive value = PPL rising.
        """
        if len(self.history) < 3:
            return 0.0
        log_hist = [math.log(max(1e-3, h)) for h in self.history]
        n = len(log_hist)
        mid = n // 2
        first_half = sum(log_hist[i] for i in range(mid)) / max(1, mid)
        second_half = sum(log_hist[i] for i in range(mid, n)) / max(1, n - mid)
        dt = max(1, n - mid)
        return (second_half - first_half) / dt

    def _alpha_eff(self, ppl: float) -> float:
        """Surprise-gated mixing rate.

        Only positive velocity (PPL rising) dampens the gate. Fast descent
        is desirable — we want the stable shadow to absorb good weights
        promptly when training is going well. The asymmetric gating is
        the spirit of the spec: dampen on spikes, absorb on improvement.
        """
        if not self.history:
            return 0.0
        avg_ppl = sum(self.history) / len(self.history)
        velocity = self._ppl_velocity()
        base = 1.0 / max(1.0, avg_ppl)
        # max(0, velocity) — descent (velocity < 0) does NOT freeze us
        gate = math.exp(-self.gamma * max(0.0, velocity))
        a = base * gate
        return min(self.base_alpha_cap, a)

    def maybe_update(self, model: nn.Module,
                     ppl: float,
                     step: int) -> dict:
        """Update internal history + maybe absorb current weights into
        the stable EMA. Returns a small dict of diagnostics for logging."""
        if not (math.isfinite(ppl) and ppl > 0):
            # Skip degenerate PPL (NaN / 0 / negative). No update.
            return {"alpha_eff": 0.0, "velocity": 0.0, "skipped": True}

        self.history.append(float(ppl))
        # Track ppl_ema for logging (fast EMA so it tracks recent regime)
        if self._ppl_ema is None:
            self._ppl_ema = float(ppl)
        else:
            a = self.best_ema_alpha
            self._ppl_ema = (1 - a) * self._ppl_ema + a * float(ppl)

        # Track best as the LOW-PASS-FILTERED min: average over the last
        # ~history_len/2 samples, take that as 'current ppl regime', and
        # snapshot when it's a new low. Pure-min on raw ppl was too noisy
        # (a single lucky batch could be 0.1× the regime); EMA was too
        # slow (took 20+ steps to track a descent from 5000 to 50). The
        # short-window mean is the right compromise — robust to single-
        # batch outliers, fast to track legitimate basins.
        if len(self.history) >= 3:
            tail_n = max(3, len(self.history) // 2)
            regime_ppl = sum(list(self.history)[-tail_n:]) / tail_n
            if regime_ppl < self._best_ema_ppl:
                self._best_ema_ppl = regime_ppl
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in self._best:
                            # Param may have grown (NeuralGeometryAdapter BDNF
                            # grow_kern_rank); just resnapshot with the new
                            # shape. Falls through to .copy_ for unchanged
                            # shapes (no-op overhead).
                            if self._best[name].shape != p.shape:
                                self._best[name] = p.detach().clone()
                            else:
                                self._best[name].copy_(p.detach())

        # EMA update of stable shadow — gated by surprise
        if step % self.update_every != 0:
            return {"alpha_eff": 0.0, "velocity": self._ppl_velocity(),
                    "skipped": False, "deferred": True}

        alpha = self._alpha_eff(ppl)
        if alpha < 1e-6:
            self._n_freezes += 1
            return {"alpha_eff": alpha, "velocity": self._ppl_velocity(),
                    "frozen": True}

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in self._stable:
                    # Newly-registered param mid-training (e.g. BDNF grew a
                    # new adapter rank). Snapshot it once and skip mixing
                    # this step — there's nothing meaningful to EMA against.
                    self._stable[name] = p.detach().clone()
                    continue
                # Shape may have changed via BDNF growth (kern_a/kern_b
                # grow along their rank dim). When that happens, reset
                # the stable shadow to the current param state — losing
                # the EMA history for this param is fine because the
                # grown dims are zero-init anyway (the model itself is
                # mostly unchanged on the OLD dims, but the new dims
                # would have undefined-shape mixing arithmetic).
                if self._stable[name].shape != p.shape:
                    self._stable[name] = p.detach().clone()
                    continue
                self._stable[name].mul_(1.0 - alpha).add_(p.detach(), alpha=alpha)

        return {
            "alpha_eff": alpha,
            "velocity": self._ppl_velocity(),
            "ppl_ema": self._ppl_ema,
            "best_ema_ppl": self._best_ema_ppl,
            "frozen": False,
        }

    # ------------------------------------------------------------------
    # Free-energy collapse
    # ------------------------------------------------------------------
    def maybe_collapse_to_best(self, model: nn.Module,
                               current_ppl: float,
                               trigger_ratio: float = 3.0,
                               require_history: int = 5) -> bool:
        """If current PPL is wildly above the best (catastrophic spike),
        collapse model weights to params_best ('Free-Energy Collapse').
        Returns True iff a collapse happened.

        trigger_ratio : collapse only when current_ppl > trigger_ratio
                        × self._best_ema_ppl. Default 3.0.
        require_history : need at least this many PPL samples before a
                        collapse is allowed (avoid collapsing in the
                        first few steps when best_ema_ppl is huge anyway).
        """
        if len(self.history) < require_history:
            return False
        if not math.isfinite(self._best_ema_ppl):
            return False
        if current_ppl < trigger_ratio * self._best_ema_ppl:
            return False

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._best and self._best[name].shape == p.shape:
                    p.data.copy_(self._best[name])
                # else: shape diverged (BDNF growth between snapshot and
                # now). Leave that param alone — partial collapse on the
                # majority of params still recovers the trunk's good
                # state; the grown adapter rank stays at its current value.
        self._n_collapses += 1
        return True

    def collapse_to_best(self, model: nn.Module) -> None:
        """Unconditional collapse to params_best (call at checkpoint /
        inference time). The 'free energy minimum' selector."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._best:
                    p.data.copy_(self._best[name])

    def collapse_to_stable(self, model: nn.Module) -> None:
        """Copy params_stable → model. Alternative to collapse_to_best
        when 'best' might be a single lucky dip rather than a stable
        trajectory."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._stable:
                    p.data.copy_(self._stable[name])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "best_ema_ppl": self._best_ema_ppl,
            "ppl_ema":      self._ppl_ema,
            "history_len":  len(self.history),
            "n_freezes":    self._n_freezes,
            "n_collapses":  self._n_collapses,
        }

    # ------------------------------------------------------------------
    # State persistence (so resume picks up the EMA state)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "stable": {k: v.cpu() for k, v in self._stable.items()},
            "best":   {k: v.cpu() for k, v in self._best.items()},
            "history": list(self.history),
            "best_ema_ppl": self._best_ema_ppl,
            "ppl_ema": self._ppl_ema,
            "n_freezes": self._n_freezes,
            "n_collapses": self._n_collapses,
        }

    def load_state_dict(self, sd: dict, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in sd.get("stable", {}):
                self._stable[name] = sd["stable"][name].to(p.device).clone()
            if name in sd.get("best", {}):
                self._best[name] = sd["best"][name].to(p.device).clone()
        self.history = deque(sd.get("history", []), maxlen=self.history.maxlen)
        self._best_ema_ppl = float(sd.get("best_ema_ppl", float('inf')))
        self._ppl_ema = sd.get("ppl_ema", None)
        self._n_freezes = int(sd.get("n_freezes", 0))
        self._n_collapses = int(sd.get("n_collapses", 0))
