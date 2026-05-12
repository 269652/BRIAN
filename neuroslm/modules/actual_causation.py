"""ActualCausationHead — IIT 4.0 actual-causation strength for module transitions.

IIT 4.0 defines actual causation between a source state s_t and an effect
e_{t+1} via the integrated information φ_c of the intervention that takes
the source from its current state to a *counterfactual* state. The full
combinatorial formulation is intractable here, so we use the standard
tractable proxy:

  α(i → j; t) = D_KL[ p(e_j^{t+1} | do(s_i^t = active)) ‖ p(e_j^{t+1} | do(s_i^t = baseline)) ]
              ≈ ‖f_j(z_i^t) − f_j(z_i^baseline)‖² · attention(z_i^t, z_j^{t+1})

where:

  • z_i^t = module i's output at time t (the *putative cause*),
  • z_j^{t+1} = module j's output at time t+1 (the *putative effect*),
  • f_j = a small learnable readout that estimates how each module's
    own state would transition under intervention on z_i,
  • attention(·,·) = a soft mask gating only those (i, j) pairs where
    j's state at t+1 actually attended to i at time t.

The output is a per-edge causal-strength score α ∈ [0, 1], normalised
across edges per forward pass.

The head is *intervention-free in the forward graph*: at training time
we compute the proxy with detached state pairs (no extra optimizer hits).
The "counterfactual baseline" is the module's running EMA of its own
output, used as the do(s_i = baseline) reference.

Usage:

    self.actual_causation = ActualCausationHead(
        n_modules=8, d_sem=cfg.d_sem)

    # in forward_lm, AFTER all modules have produced their outputs:
    alpha = self.actual_causation(prev_outputs, cur_outputs)
    # alpha: (n_modules, n_modules)

Emits κ_cause vesicles when α exceeds threshold (see brain.py wiring).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActualCausationHead(nn.Module):
    """IIT 4.0 actual-causation strength estimator over n module outputs.

    Args:
        n_modules: number of bow-tie module slots tracked.
        d_sem:     semantic dimension of module outputs.
        d_hidden:  internal MLP width (small — this is a side computation).
        ema_decay: decay for the baseline-state EMA used as do(s_i = baseline).
        gate_threshold: minimum α to count as a "real" causal link (for
                        downstream vesicle emission).
    """

    def __init__(self, n_modules: int, d_sem: int,
                 d_hidden: int = 64,
                 ema_decay: float = 0.95,
                 gate_threshold: float = 0.3):
        super().__init__()
        self.n_modules = n_modules
        self.d_sem = d_sem
        self.ema_decay = ema_decay
        self.gate_threshold = gate_threshold

        # Per-module readout f_j: z_i → estimated effect on j
        # Implemented as a single shared MLP conditioned on (i, j) one-hots.
        self.readout = nn.Sequential(
            nn.Linear(d_sem + 2 * n_modules, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_sem),
        )

        # Baseline-state EMA for each module (do(s = baseline) reference)
        self.register_buffer("baseline_state",
                              torch.zeros(n_modules, d_sem))
        self.register_buffer("baseline_init",
                              torch.zeros(1, dtype=torch.bool))

        # Pairwise attention readout: scalar gate for each (i, j) pair
        self.attn_q = nn.Linear(d_sem, d_hidden, bias=False)
        self.attn_k = nn.Linear(d_sem, d_hidden, bias=False)

        # Module index one-hots, registered as buffer for efficient lookup
        self.register_buffer(
            "one_hot",
            torch.eye(n_modules, dtype=torch.float32))

        # Running EMA of the alpha matrix (used by sleep cycle for pruning)
        self.register_buffer(
            "alpha_ema", torch.zeros(n_modules, n_modules))
        self._ema_alpha = 0.05  # slow tracker

    # ------------------------------------------------------------------
    # Baseline update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_baseline(self, cur_outputs: torch.Tensor) -> None:
        """EMA on each module's mean output (the do(s_i = baseline) ref).
        cur_outputs: (n_modules, d_sem)."""
        if not bool(self.baseline_init.item()):
            self.baseline_state.copy_(cur_outputs.detach())
            self.baseline_init.fill_(True)
            return
        d = self.ema_decay
        self.baseline_state.mul_(d).add_(cur_outputs.detach(), alpha=1 - d)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                prev_outputs: torch.Tensor,
                cur_outputs: torch.Tensor,
                ) -> torch.Tensor:
        """Compute actual-causation matrix α : (n, n).

        prev_outputs, cur_outputs: (n_modules, d_sem). For batch, pass
        mean-over-batch tensors. α[i, j] is the estimated causal strength
        of module i at time t on module j at time t+1.

        All math here is **no-grad** by default at the call site —
        the head is trained passively via direct optimization of its
        readout against the next-step module observations (an auxiliary
        loss that train.py adds as part of `_aux_w_scale`).
        """
        n = self.n_modules
        prev = prev_outputs[:n] if prev_outputs.size(0) >= n else \
               F.pad(prev_outputs, (0, 0, 0, n - prev_outputs.size(0)))
        cur  = cur_outputs[:n]  if cur_outputs.size(0)  >= n else \
               F.pad(cur_outputs,  (0, 0, 0, n - cur_outputs.size(0)))

        self._update_baseline(cur)

        # Build the (n_pairs, d_sem + 2n) input for the readout
        I = self.one_hot.to(prev.dtype)
        # For every (i, j) pair: concat(z_i_prev, onehot_i, onehot_j)
        prev_expanded = prev.unsqueeze(1).expand(-1, n, -1)  # (n, n, d)
        base_expanded = self.baseline_state.to(prev.dtype).unsqueeze(1).expand(-1, n, -1)
        i_onehot = I.unsqueeze(1).expand(-1, n, -1)          # (n, n, n)
        j_onehot = I.unsqueeze(0).expand(n, -1, -1)          # (n, n, n)

        # do(s_i = active):    feed z_i^t
        # do(s_i = baseline):  feed baseline_state[i]
        x_act = torch.cat([prev_expanded, i_onehot, j_onehot], dim=-1)
        x_base = torch.cat([base_expanded, i_onehot, j_onehot], dim=-1)

        # Predicted effect of intervention on j
        eff_act  = self.readout(x_act)   # (n, n, d)
        eff_base = self.readout(x_base)  # (n, n, d)

        # KL proxy: ‖f(act) − f(base)‖² (per pair)
        diff = (eff_act - eff_base).pow(2).mean(dim=-1)  # (n, n)

        # Attention mask: was j actually responsive to i at the observed pair?
        # Soft cosine similarity between cur[j] and the predicted effect.
        # → high attention iff the predicted effect aligned with what j did.
        q = self.attn_q(cur).unsqueeze(0).expand(n, -1, -1)   # (n, n, h)
        k = self.attn_k(eff_act)                                # (n, n, h)
        attn = torch.sigmoid((q * k).sum(dim=-1)               # (n, n)
                              / (self.attn_q.out_features ** 0.5))

        # Raw causal score, then normalise per pair
        alpha_raw = diff * attn
        # Soft normalisation across destinations: α[i, ·] sums to ≤ 1
        alpha = alpha_raw / (alpha_raw.sum(dim=-1, keepdim=True) + 1e-6)

        # Self-loops contribute no actual causation (they are the same node)
        eye = torch.eye(n, device=alpha.device, dtype=alpha.dtype)
        alpha = alpha * (1.0 - eye)

        # Update EMA — used by sleep cycle to prune persistently-weak edges
        with torch.no_grad():
            self.alpha_ema.mul_(1 - self._ema_alpha).add_(
                alpha.detach(), alpha=self._ema_alpha)

        return alpha

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def strong_edges(self,
                     alpha: torch.Tensor | None = None
                     ) -> list[tuple[int, int, float]]:
        """List (i, j, α) for pairs above gate_threshold."""
        a = alpha if alpha is not None else self.alpha_ema
        mask = a >= self.gate_threshold
        out: list[tuple[int, int, float]] = []
        n = self.n_modules
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    out.append((i, j, float(a[i, j].item())))
        return out

    def aux_loss(self,
                 prev_outputs: torch.Tensor,
                 cur_outputs: torch.Tensor) -> torch.Tensor:
        """Auxiliary objective that trains the readout to be predictive.

        Forces f_j(z_i^t) → cur_outputs[j] in expectation. This is what
        gives the do-intervention proxy its meaning: f is approximating
        the conditional transition function.

        Call this in brain.forward_lm aux-loss block; weight via
        `_aux_w_scale * 0.05`. Cheap (n² MLP evals) and naturally
        suppressed during infancy.
        """
        n = self.n_modules
        prev = prev_outputs[:n] if prev_outputs.size(0) >= n else \
               F.pad(prev_outputs, (0, 0, 0, n - prev_outputs.size(0)))
        cur  = cur_outputs[:n]  if cur_outputs.size(0)  >= n else \
               F.pad(cur_outputs,  (0, 0, 0, n - cur_outputs.size(0)))

        I = self.one_hot.to(prev.dtype)
        # We train each f_{i→j} to predict cur[j] from prev[i]
        prev_expanded = prev.unsqueeze(1).expand(-1, n, -1)
        i_onehot = I.unsqueeze(1).expand(-1, n, -1)
        j_onehot = I.unsqueeze(0).expand(n, -1, -1)
        x = torch.cat([prev_expanded, i_onehot, j_onehot], dim=-1)
        pred = self.readout(x)                                   # (n, n, d)
        tgt = cur.unsqueeze(0).expand(n, -1, -1).detach()        # (n, n, d)
        return F.mse_loss(pred, tgt)
