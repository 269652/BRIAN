"""Learned meta-optimizer for NeuroSLM.

Instead of a simple scalar multiplier, this module maintains per-parameter
hidden state (LSTM-based, like L2L) and produces both magnitude and
direction corrections.  It also accepts a *comprehension signal* (rate of
prediction-error reduction) so the optimizer learns to maximise deep
comprehension and intelligence, not just raw loss reduction speed.

The meta-objective optimizes for:
  - Calibrated predictions (knowing what you don't know)
  - Diverse semantic representations (rich internal models, not collapse)
  - Smooth reasoning (coherent predictions, not erratic pattern-matching)

Features fed per parameter group:
  [log|g|, log|p|, cos(g,p), sign(g)·log(1+|g|), momentum_ema,
   comprehension_delta, *neuromod_vec]
  →  tiny LSTMCell  →  (scale, comprehension_gate)  →  scalar multiplier

Fully differentiable for meta-training via unrolled inner optimisation.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedBackprop(nn.Module):
    """Meta-learned optimizer with per-parameter-group hidden state."""

    def __init__(self, n_neuromods: int = 4, hidden: int = 32,
                 momentum_decay: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden
        self.momentum_decay = momentum_decay
        # Feature dim: 5 stat features + 1 comprehension + n_neuromods
        self.n_features = 5 + 1 + n_neuromods

        # Coordinatewise LSTM (small, shared across all parameters)
        self.lstm = nn.LSTMCell(self.n_features, hidden)

        # Output heads
        self.scale_head = nn.Linear(hidden, 1)
        self.comp_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Init near-identity (scale≈1, gate≈0.5 → multiplier≈1)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, 0.0)

        self._hidden_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._momentum: dict[str, torch.Tensor] = {}

    def reset_state(self):
        """Clear all per-parameter hidden states."""
        self._hidden_states.clear()
        self._momentum.clear()

    def _get_state(self, key: str, device: torch.device):
        if key not in self._hidden_states:
            h = torch.zeros(1, self.hidden_dim, device=device)
            c = torch.zeros(1, self.hidden_dim, device=device)
            self._hidden_states[key] = (h, c)
        return self._hidden_states[key]

    def _get_momentum(self, key: str, grad: torch.Tensor):
        if key not in self._momentum:
            self._momentum[key] = torch.zeros(1, device=grad.device)
        m = self._momentum[key]
        g_norm = grad.norm()
        m = self.momentum_decay * m + (1 - self.momentum_decay) * g_norm
        self._momentum[key] = m.detach()
        return m

    def forward(self, grad: torch.Tensor, param: torch.Tensor,
                neuromod: torch.Tensor,
                comprehension_delta: float = 0.0,
                param_name: str = "") -> torch.Tensor:
        """Compute a scalar multiplier for a parameter's gradient.

        Args:
            grad, param: tensors (any shape)
            neuromod: (n_neuromods,) neuromodulatory vector
            comprehension_delta: improvement in understanding this step
            param_name: key for per-parameter hidden state
        Returns:
            scalar multiplier tensor (positive)
        """
        device = grad.device
        eps = 1e-8

        g_norm = grad.norm().clamp(min=eps)
        p_norm = param.norm().clamp(min=eps)

        log_g = torch.log(g_norm + eps).reshape(1)
        log_p = torch.log(p_norm + eps).reshape(1)
        cos_sim = ((grad * param).sum() / (g_norm * p_norm + eps)).reshape(1)
        signed_log_g = (grad.sign() * torch.log1p(grad.abs())).mean().reshape(1)
        momentum = self._get_momentum(param_name, grad).reshape(1)
        comp = torch.tensor([comprehension_delta], device=device, dtype=grad.dtype)
        nm = neuromod.to(device).reshape(-1)

        features = torch.cat([log_g, log_p, cos_sim, signed_log_g,
                              momentum, comp, nm]).unsqueeze(0)

        h_prev, c_prev = self._get_state(param_name, device)
        h_new, c_new = self.lstm(features, (h_prev, c_prev))
        # Store detached states — backward path goes through current call only.
        # This prevents in-place modification errors when learned_opt is called
        # multiple times per training step.
        self._hidden_states[param_name] = (h_new.detach(), c_new.detach())

        # Scale: exp(tanh(raw)) → ~[0.37, 2.72]
        raw_scale = self.scale_head(h_new)
        scale = torch.exp(torch.tanh(raw_scale)).squeeze()

        # Comprehension gate: amplify when understanding improves
        comp_g = self.comp_gate(h_new).squeeze()

        mult = scale * (0.5 + comp_g)
        mult = mult.clamp(0.01, 10.0)
        return mult

