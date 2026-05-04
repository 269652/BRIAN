"""Anterior Cingulate Cortex (ACC) for NeuroSLM.

The ACC sits at the interface of the limbic system and the prefrontal cortex.
It performs two distinct functions based on its sub-regions:

  Dorsal ACC (dACC) — Cognitive control:
    • Conflict monitoring: detects when multiple responses simultaneously compete
    • Error-related negativity: amplifies learning signal after errors
    • Expected Value of Control (EVC): weighs whether cognitive effort is worth it
    • Drives ACh release (via NBM) to sharpen attention when conflict is high

  Rostral/Ventral ACC (rACC/vACC) — Affective regulation:
    • Top-down emotional regulation: suppresses amygdala response
    • Pain modulation: gates pain vs. reward signals
    • Self-referential processing: "is this relevant to me?"

Why the ACC is critical for intelligence:
  Without conflict monitoring, the model always uses the same amount of
  cognitive effort regardless of task difficulty. With it:
  - Easy, familiar inputs: low effort (fast, efficient)
  - Novel/conflicted inputs: high effort (slow, careful)
  - Error detection: rapid adaptation of strategy after mistakes

This is the neural mechanism behind System 1 / System 2 (Kahneman):
  Low conflict → System 1 (automatic)
  High conflict detected by ACC → recruit System 2 (deliberate)

ML implementation:
  1. Conflict detector: measures discordance between parallel processing streams
     (multiple candidate actions with similar logit values = high conflict)
  2. Error detector: compares predicted vs actual outcomes (PE signal)
  3. EVC estimator: should we spend more computation? (gating function)
  4. ACh demand: conflict → ACh release → sharper attention everywhere
  5. Effort gate: modulates ponder depth (how many thinking steps to use)

Novel contribution: The ACC gives the model explicit *metacognitive effort
allocation* — it can literally decide to think harder based on how confused
it is. No existing SLM has this at the architectural level.

References:
  Botvinick et al. (2001): Conflict Monitoring and Cognitive Control
  Shackman et al. (2011): The integration of negative affect, pain and cognitive control
  Holroyd & Coles (2002): The neural basis of human error processing
  Shenhav et al. (2013): The expected value of control
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .brain_module import BrainModule


class ConflictMonitor(nn.Module):
    """Detects response conflict from multiple competing representations.

    Conflict = how much do the competing candidates disagree with each other?
    Measured as the average pairwise distance (or entropy of routing weights).

    High conflict: competing representations pull in different directions.
    Low conflict: clear winner, minimal competition.
    """

    def __init__(self, d_sem: int):
        super().__init__()
        # Project each candidate to a conflict-relevant representation
        self.proj = nn.Linear(d_sem, d_sem // 2)
        # Summarise conflict as a scalar
        self.scorer = nn.Sequential(
            nn.Linear(d_sem // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, candidates: torch.Tensor,
                routing_weights: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        candidates:      (B, n_candidates, d_sem)
        routing_weights: (B, n_candidates) — router probabilities (optional)

        Returns:
          conflict_scalar: (B,)          — [0, 1] how conflicted are the candidates
          conflict_rep:    (B, d_sem//2) — conflict representation for ACC
        """
        B, N, D = candidates.shape
        proj = self.proj(candidates)   # (B, N, D/2)

        # Pairwise cosine distances → mean = conflict measure
        norm = F.normalize(proj, dim=-1)   # (B, N, D/2)
        sim_mat = torch.bmm(norm, norm.transpose(1, 2))  # (B, N, N)
        # Mean off-diagonal similarity (high sim = low conflict)
        eye_mask = torch.eye(N, device=proj.device).unsqueeze(0)
        off_diag = (sim_mat * (1 - eye_mask)).sum(-1).sum(-1) / max(N * (N - 1), 1)
        conflict = 1.0 - (off_diag + 1.0) / 2.0   # map [-1,1] → [1,0], then invert

        # If routing weights provided, also use entropy as conflict measure
        if routing_weights is not None:
            w = routing_weights.clamp(1e-6)
            ent = -(w * w.log()).sum(-1)              # (B,)
            max_ent = torch.tensor(float(N)).log()
            normalised_ent = (ent / max_ent).clamp(0, 1)  # (B,)
            conflict = 0.5 * conflict + 0.5 * normalised_ent

        # Conflict representation: weighted mean of candidates, weighted by
        # how much each diverges from the mean
        mean = proj.mean(1, keepdim=True)            # (B, 1, D/2)
        deviation = (proj - mean).abs().mean(-1)     # (B, N)
        conflict_rep = (proj * deviation.unsqueeze(-1)).sum(1)  # (B, D/2)

        return conflict.clamp(0, 1), conflict_rep


class AnteriorCingulateCortex(BrainModule):
    """ACC: conflict monitoring, error detection, effort gating.

    Parameters
    ----------
    d_sem      : semantic dimension
    n_nt       : neurotransmitter count
    effort_steps: max additional ponder steps to request (0-based)
    """

    def __init__(self, d_sem: int, n_nt: int = 8,
                 effort_steps: int = 4):
        super().__init__()
        self.d_sem       = d_sem
        self.effort_steps = effort_steps

        self.conflict_monitor = ConflictMonitor(d_sem)

        # Error detector: prediction PE → error signal
        self.error_proj = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, 1),
            nn.Sigmoid(),
        )

        # EVC: is it worth recruiting more effort?
        # Input: conflict + error + task_utility → effort investment
        self.evc = nn.Sequential(
            nn.Linear(d_sem // 2 + 2, effort_steps + 1),
            nn.Softmax(dim=-1),   # distribution over effort levels 0..effort_steps
        )

        # ACC → ACh demand: high conflict → release more ACh
        self.ach_demand = nn.Sequential(
            nn.Linear(d_sem // 2 + 1, 1),
            nn.Sigmoid(),
        )

        # Affective regulation: top-down suppression signal
        # Applied to amygdala valence to regulate emotional reactivity
        self.affect_reg = nn.Sequential(
            nn.Linear(d_sem, 1),
            nn.Sigmoid(),
        )

        # Error-based learning rate modulation: scale gradients after errors
        self.error_lr_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Softplus(),
        )

        # Running EMA of conflict (for baseline calibration)
        self.register_buffer("baseline_conflict", torch.tensor(0.5))
        self.register_buffer("baseline_error",    torch.tensor(0.3))
        self.ema_alpha = 0.01

    # ------------------------------------------------------------------

    def forward(self, candidates: torch.Tensor,
                prediction_error: Optional[torch.Tensor] = None,
                routing_weights: Optional[torch.Tensor] = None,
                task_utility: Optional[torch.Tensor] = None,
                pfc_rep: Optional[torch.Tensor] = None,
               ) -> dict:
        """
        candidates:       (B, n_candidates, d_sem) — competing representations
        prediction_error: (B,) [0,1] — how wrong was the last prediction
        routing_weights:  (B, n_candidates) — router probability distribution
        task_utility:     (B,) [0,1] — estimated reward if task succeeds
        pfc_rep:          (B, d_sem) — PFC state for affect regulation

        Returns dict with:
          conflict:          (B,)           — conflict level [0, 1]
          effort_dist:       (B, steps+1)   — distribution over effort levels
          effort_steps:      (B,)           — expected additional steps to take
          ach_demand:        (B,)           — ACh release demand [0, 1]
          affect_reg:        (B,)           — top-down emotional suppression [0, 1]
          error_lr_scale:    (B,)           — learning rate multiplier after errors
        """
        B = candidates.shape[0]
        device = candidates.device

        conflict, conflict_rep = self.conflict_monitor(candidates, routing_weights)

        # Error signal (use prediction_error if provided, else use conflict proxy)
        if prediction_error is not None:
            error = prediction_error.clamp(0, 1)
        else:
            error = conflict * 0.3

        # Update baselines
        with torch.no_grad():
            self.baseline_conflict = (
                (1 - self.ema_alpha) * self.baseline_conflict
                + self.ema_alpha * conflict.mean()
            )
            self.baseline_error = (
                (1 - self.ema_alpha) * self.baseline_error
                + self.ema_alpha * error.mean()
            )

        # Relative conflict (above baseline → more effort warranted)
        rel_conflict = (conflict - self.baseline_conflict).clamp(0, 1)

        # EVC: effort investment decision
        util = (task_utility if task_utility is not None
                else torch.ones(B, device=device) * 0.5)
        evc_input = torch.cat([
            conflict_rep,
            rel_conflict.unsqueeze(-1),
            util.unsqueeze(-1)
        ], dim=-1)
        effort_dist = self.evc(evc_input)                       # (B, steps+1)
        effort_expected = (effort_dist * torch.arange(
            self.effort_steps + 1, device=device, dtype=torch.float)).sum(-1)

        # ACh demand: conflict drives attention sharpening
        ach_inp = torch.cat([conflict_rep, rel_conflict.unsqueeze(-1)], dim=-1)
        ach = self.ach_demand(ach_inp).squeeze(-1)              # (B,)

        # Affective regulation (vACC: suppress amygdala when PFC engages)
        affect_sup = (self.affect_reg(pfc_rep).squeeze(-1)
                      if pfc_rep is not None
                      else torch.ones(B, device=device) * 0.5)

        # Error-driven learning rate modulation
        # After errors: amplify learning (surprise-induced plasticity)
        lr_scale = self.error_lr_gate(error.unsqueeze(-1)).squeeze(-1)  # (B,)

        return {
            "conflict":       conflict,
            "effort_dist":    effort_dist,
            "effort_steps":   effort_expected,
            "ach_demand":     ach,
            "affect_reg":     affect_sup,
            "error_lr_scale": lr_scale,
            "conflict_rep":   conflict_rep,
        }
