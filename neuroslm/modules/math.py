"""MathCortex — high-precision symbolic reasoning with differential attention.

Biologically: the intraparietal sulcus (IPS) and dorsolateral PFC (dlPFC)
together implement numerical cognition. IPS handles magnitude and spatial
quantity; dlPFC handles rule-based manipulation and working memory for
mathematical relations.

Computationally: implements differential attention (DiffAttn, Ye 2024) that
doubles effective SNR by cancelling attention noise with a second head.
Combines with a learnable symbolic memory pool for mathematical facts.

Activated by κ_math vesicles (type MATH in VesiclePool). The vesicle_gate
parameter scales the enrichment — 0.0 = module passes through unchanged.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_module import BrainModule
from .fast_weight import FastWeightLayer


# Topic index for this expert (must match TopicClassifier.N_TOPICS ordering)
TOPIC_MATH = 1


class MathCortex(BrainModule):
    """Symbolic-numeric reasoning using differential attention + fact memory.

    Architecture:
      1. Dual-query differential attention over a learned fact memory
         (two parallel attention heads whose outputs are subtracted to
         cancel correlated noise — the DiffAttn SNR-doubling trick)
      2. Small working-memory stack (FIFO, capacity = memory_size)
         that accumulates recent mathematical contexts across forward calls
      3. LayerNorm + residual around the enrichment

    Args:
        d_sem:       Semantic embedding dimension
        n_heads:     Number of attention heads inside each dual path
        memory_size: Capacity of the symbolic fact memory pool
    """

    def __init__(self, d_sem: int, n_heads: int = 4,
                 memory_size: int = 128,
                 enable_hfw: bool = True):
        super().__init__()
        self.d_sem      = d_sem
        self.n_heads    = max(1, n_heads)
        self.head_dim   = d_sem // self.n_heads

        # Learnable symbolic fact memory (math knowledge base)
        # Zero-init values so the cortex starts as identity
        self.fact_keys = nn.Parameter(
            torch.randn(memory_size, d_sem) * (1.0 / math.sqrt(d_sem)))
        self.fact_vals = nn.Parameter(torch.zeros(memory_size, d_sem))

        # Differential attention: two query projections, one key/value
        # attn_out = softmax(Q1 K^T) V  -  λ · softmax(Q2 K^T) V
        # λ is learned; starts near 0 so cortex starts near identity
        self.q1_proj = nn.Linear(d_sem, d_sem, bias=False)
        self.q2_proj = nn.Linear(d_sem, d_sem, bias=False)
        self.k_proj  = nn.Linear(d_sem, d_sem, bias=False)
        self.v_proj  = nn.Linear(d_sem, d_sem, bias=False)
        self.log_lam = nn.Parameter(torch.full((1,), -3.0))  # λ = sigmoid(-3) ≈ 0.05
        self.out_proj = nn.Linear(d_sem, d_sem, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start as zero-output

        # QK-Norm (stabilises attention scores, esp. for TPU bfloat16)
        self.qk_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)

        self.norm = nn.LayerNorm(d_sem)

        # Working memory: FIFO ring buffer (B-agnostic, accumulates over steps)
        self.register_buffer("_wm_keys", torch.zeros(memory_size, d_sem))
        self.register_buffer("_wm_vals", torch.zeros(memory_size, d_sem))
        self.register_buffer("_wm_ptr",  torch.zeros(1, dtype=torch.long))
        self._wm_size = memory_size

        # Hebbian Fast-Weight binding (final stage, episodic associations)
        # Dual-timescale: slow weights = fact memory; fast weights = inference-time
        # bindings between current GWS broadcast and discovered math patterns.
        self.hfw = FastWeightLayer(d_sem, decay=0.95, base_eta=0.1, n_heads=self.n_heads) \
                   if enable_hfw else None
        self._hfw_state: torch.Tensor | None = None  # carry-over W_fast across calls

    # ------------------------------------------------------------------
    # Differential attention over memory
    # ------------------------------------------------------------------
    def _diff_attn(self, x: torch.Tensor,
                   keys: torch.Tensor,
                   vals: torch.Tensor) -> torch.Tensor:
        """Differential cross-attention: x × memory.

        x:    (B, d_sem) — query
        keys: (M, d_sem) — memory keys
        vals: (M, d_sem) — memory values
        Returns: (B, d_sem) enrichment
        """
        B, D = x.shape
        M    = keys.size(0)
        H    = self.n_heads
        hd   = self.head_dim

        # Split into heads
        q1 = self.q1_proj(x).view(B, H, hd)          # (B, H, hd)
        q2 = self.q2_proj(x).view(B, H, hd)
        k  = self.k_proj(keys).view(M, H, hd)         # (M, H, hd)
        v  = self.v_proj(vals).view(M, H, hd)

        # QK-norm for stability
        q1 = self.qk_norm(q1); q2 = self.qk_norm(q2)

        # Per-head attention (cross-attention: B queries × M keys)
        scale = 1.0 / math.sqrt(hd)
        # (B, H, M)
        a1 = torch.einsum("bhd,mhd->bhm", q1, k) * scale
        a2 = torch.einsum("bhd,mhd->bhm", q2, k) * scale
        a1 = F.softmax(a1.float(), dim=-1).to(x.dtype)
        a2 = F.softmax(a2.float(), dim=-1).to(x.dtype)

        lam = torch.sigmoid(self.log_lam)             # λ ∈ (0, 1)
        # Differential: subtract noise-correlated attention
        a_diff = a1 - lam * a2                        # (B, H, M)

        # Value retrieval
        out = torch.einsum("bhm,mhd->bhd", a_diff, v)  # (B, H, hd)
        out = out.contiguous().view(B, D)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Working memory update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_wm(self, x: torch.Tensor) -> None:
        """Write current batch mean into the working-memory ring buffer."""
        mean_x = x.detach().mean(0)  # (d_sem,)
        idx = int(self._wm_ptr.item()) % self._wm_size
        self._wm_keys[idx] = mean_x
        self._wm_vals[idx] = mean_x  # identity: key = value for WM
        self._wm_ptr += 1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                vesicle_gate: float = 0.0,
                gws_context: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, d_sem) or (B, S, d_sem).
        vesicle_gate ∈ [0, 1]: strength of math-cortex activation.
        gws_context: optional (B, d_sem) — Global Workspace broadcast used as
                     plasticity context for the Hebbian fast-weight layer.
                     Enables dual-timescale binding (slow facts × fast episodic).
        Returns same shape as x.
        """
        if vesicle_gate < 1e-3:
            return x

        squeeze = x.dim() == 3
        if squeeze:
            x_flat = x.mean(1)  # (B, d_sem)
        else:
            x_flat = x          # (B, d_sem)

        h = self.norm(x_flat.float()).to(dtype=x_flat.dtype)

        # 1) Attend over learned fact memory
        fact_enrichment = self._diff_attn(h, self.fact_keys, self.fact_vals)

        # 2) Attend over working memory (recent contexts)
        has_wm = int(self._wm_ptr.item()) > 0
        if has_wm:
            n_filled = min(int(self._wm_ptr.item()), self._wm_size)
            wm_k = self._wm_keys[:n_filled]
            wm_v = self._wm_vals[:n_filled]
            wm_enrichment = self._diff_attn(h, wm_k, wm_v)
        else:
            wm_enrichment = torch.zeros_like(x_flat)

        enrichment = fact_enrichment + 0.3 * wm_enrichment
        enriched   = x_flat + vesicle_gate * enrichment

        # 3) Hebbian fast-weight binding (final stage)
        # Bind current GWS state ↔ discovered math pattern for within-inference
        # episodic recall.  Slow weights (fact memory) remain unchanged.
        if self.hfw is not None and vesicle_gate > 0.1:
            ctx = gws_context if gws_context is not None else enriched
            seq = enriched.unsqueeze(1)
            seq, self._hfw_state = self.hfw(seq, context=ctx,
                                             W_fast=self._hfw_state)
            enriched = seq.squeeze(1)

        # Update working memory with this forward's output
        self._update_wm(enriched)

        if squeeze:
            # Broadcast back: add enrichment residual to all slot positions
            delta = (enriched - x_flat).unsqueeze(1)  # (B, 1, d_sem)
            return x + delta
        return enriched

    def _disabled_output(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
