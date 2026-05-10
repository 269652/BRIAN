"""Meta-Plasticity / Fast Weights for NeuroSLM.

Fast weights (Schmidhuber 1992, Ba et al. 2016) are a second tier of
weights that update at inference time within a single forward pass.
They act as a differentiable short-term memory, storing associations
from the current context without any gradient steps.

The mechanism:
  W_fast = sum_{t'<t} η * v_{t'} ⊗ k_{t'}   (outer-product accumulation)
  read:    y_t = W_fast · q_t                 (associative retrieval)

In modern terms (Schlag et al. 2021), this is equivalent to *linear
attention* with an explicit outer-product memory matrix.  The advantage
over softmax attention is O(d²) memory but O(T·d²) computation — much
cheaper than O(T²·d) full attention for long sequences.

This implementation adds three improvements:

  1. Gated write: a write gate g_t ∈ (0,1)^d controls how much of
     each key-value association is stored.  This prevents catastrophic
     overwrite of earlier associations (like an LSTM forget gate for
     fast weights).

  2. Decaying accumulation: W_fast multiplied by decay factor λ at each
     step, giving exponential weighting to recent associations.

  3. Context-dependent plasticity rate η_t = f(context):
     The Hebbian learning rate itself is modulated by the current
     neuromodulator state (e.g., ACh ↑ → η ↑ → more plastic). This
     is the *meta-plasticity* mechanism observed in hippocampus CA3.

  4. LayerNorm stabilisation: W_fast accumulates unbounded so we apply
     a normalisation to the retrieved vector before returning.

The fast weight matrix is *not* a parameter — it lives in the forward
pass and is reset between sequences (or carried across turns in
interactive mode).

References:
  Schmidhuber (1992): Learning to control fast-weight memories
  Ba et al. (2016): Using Fast Weights to Attend to the Recent Past
  Schlag et al. (2021): Linear Transformers Are Secretly Fast Weight Programmers
  Munkhdalai & Yu (2017): Meta Networks
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FastWeightLayer(nn.Module):
    """Context-dependent fast-weight associative memory.

    Parameters
    ----------
    d_model      : input and output dimension
    decay        : exponential decay factor λ for W_fast (0.9–0.99)
    base_eta     : base Hebbian learning rate (scaled by context)
    n_heads      : number of parallel fast-weight heads
    """

    def __init__(self, d_model: int, decay: float = 0.95,
                 base_eta: float = 0.1, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.decay    = decay
        self.base_eta = base_eta

        # Key, value, query, write-gate projections
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, d_model)   # write gate

        # Context-dependent plasticity rate modulation
        # (takes NT state or GWS slot; maps to per-head η multiplier)
        self.eta_mod = nn.Sequential(
            nn.Linear(d_model, n_heads),
            nn.Softplus(),   # always positive
        )

        # Output projection and norm
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln       = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                W_fast: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:       (B, T, d_model)
        context: (B, d_model) — context for plasticity modulation (NT, GWS…)
                 If None, uses mean of x
        W_fast:  (B, n_heads, d_head, d_head) — carry-over from prev turn
                 If None, zero-initialise

        Returns:
          out:     (B, T, d_model)
          W_fast:  (B, n_heads, d_head, d_head) — updated fast-weight matrix
        """
        B, T, D = x.shape
        Dh = self.d_head

        if W_fast is None:
            W_fast = torch.zeros(B, self.n_heads, Dh, Dh, device=x.device, dtype=x.dtype)

        if context is None:
            context = x.mean(1)   # (B, D)

        # Per-head plasticity rate
        eta = self.base_eta * (self.eta_mod(context) + 1e-6)  # (B, n_heads)

        # Projections: (B, T, D) → split heads → (B, n_heads, T, d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.n_heads, Dh).permute(0, 2, 1, 3)
        q = self.q_proj(x).view(B, T, self.n_heads, Dh).permute(0, 2, 1, 3)
        g = torch.sigmoid(self.g_proj(x).view(B, T, self.n_heads, Dh).permute(0, 2, 1, 3))

        out_heads = []
        for t in range(T):
            kt = k[:, :, t, :]   # (B, H, Dh)
            vt = v[:, :, t, :]
            qt = q[:, :, t, :]
            gt = g[:, :, t, :]

            # Read from fast weights: y_t = W_fast · q_t
            # W_fast: (B, H, Dh, Dh),  qt: (B, H, Dh, 1)
            read = torch.einsum("bhij,bhj->bhi", W_fast, qt)  # (B, H, Dh)
            read_norm = F.layer_norm(read, [Dh])

            out_heads.append(read_norm)

            # Write: W_fast = λ*W_fast + η * g * (v ⊗ k)
            # outer product: (B, H, Dh, 1) × (B, H, 1, Dh)
            outer = torch.einsum("bhi,bhj->bhij",
                                 gt * vt,   # gated value
                                 kt)        # key

            eta_t = eta[:, :, None, None]   # (B, H, 1, 1)
            W_fast = self.decay * W_fast + eta_t * outer

        out_seq = torch.stack(out_heads, dim=2)           # (B, H, T, Dh)
        out_seq = out_seq.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.ln(x + self.out_proj(out_seq))

        return out, W_fast.detach()
