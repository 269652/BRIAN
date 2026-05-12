"""SurvivalCausalHead — measures how much an action a_t contributes to
the next-step survival variable S_{t+1}.

Sister module to ``ActualCausationHead`` (which estimates module→module
actual causation). This one is the dedicated head for the embodied loop:

  α_surv(a_t → ΔS_{t+1}) = ‖f(a_t) − f(a_baseline)‖² · σ(attn(a_t, ΔS_{t+1}))

where ``ΔS`` is the change in (energy, hydration, integrity) and the
baseline action is the EMA of recent action embeddings.

Hooks into the brain so:
  • the NAcc value head is trained against the *actual* survival reward,
  • DA release is gated by the *predicted-vs-actual* RPE,
  • the trophic system's BDNF release is amplified on positive RPE (the
    "survival success" → "lock in this pathway" coupling).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurvivalCausalHead(nn.Module):
    """Action → survival contribution head.

    Args:
        d_action: dimension of the BG action vector.
        n_survival_vars: 3 (energy, hydration, integrity).
        ema_decay: decay for baseline-action EMA.
    """

    def __init__(self,
                 d_action: int,
                 n_survival_vars: int = 3,
                 d_hidden: int = 32,
                 ema_decay: float = 0.95,
                 gate_threshold: float = 0.2):
        super().__init__()
        self.d_action = d_action
        self.n_survival_vars = n_survival_vars
        self.ema_decay = ema_decay
        self.gate_threshold = gate_threshold

        # Predicts ΔS_{t+1} given action a_t
        self.predictor = nn.Sequential(
            nn.Linear(d_action, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_survival_vars),
        )
        # Baseline-action EMA
        self.register_buffer("baseline_action",
                              torch.zeros(d_action))
        self.register_buffer("baseline_init",
                              torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def _update_baseline(self, action: torch.Tensor) -> None:
        a = action.detach()
        if a.dim() == 2:
            a = a.mean(0)
        if not bool(self.baseline_init.item()):
            self.baseline_action.copy_(a)
            self.baseline_init.fill_(True)
            return
        d = self.ema_decay
        self.baseline_action.mul_(d).add_(a, alpha=1 - d)

    def forward(self,
                action: torch.Tensor,
                ) -> tuple[torch.Tensor, float]:
        """Compute predicted ΔS_{t+1} and its causal-strength scalar α_surv.

        action: (B, d_action) or (d_action,)
        Returns (delta_S_pred, alpha_surv).

        ``alpha_surv`` is a scalar ∈ [0, 1] estimating how strongly the
        action diverges from the baseline in survival-relevant directions.
        """
        a = action if action.dim() == 2 else action.unsqueeze(0)
        self._update_baseline(a)
        delta_pred = self.predictor(a)                          # (B, 3)
        base = self.predictor(self.baseline_action.to(a.dtype).unsqueeze(0))
        diff = (delta_pred - base).pow(2).mean()
        alpha = float(torch.sigmoid(diff).item())
        return delta_pred, alpha

    def loss(self,
             action: torch.Tensor,
             actual_delta_S: torch.Tensor,
             ) -> torch.Tensor:
        """MSE auxiliary loss training the predictor on observed ΔS.

        action:        (B, d_action)
        actual_delta_S: (B, 3) — the post-step minus pre-step survival
                        vector. Caller is responsible for computing this
                        from the grid-world frames.
        """
        a = action if action.dim() == 2 else action.unsqueeze(0)
        d = actual_delta_S if actual_delta_S.dim() == 2 else actual_delta_S.unsqueeze(0)
        pred = self.predictor(a)
        return F.mse_loss(pred, d.detach())
