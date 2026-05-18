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
from .common import TransformerBlock, RMSNorm
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
                 enable_hfw: bool = True,
                 d_hidden: int | None = None,
                 n_blocks: int = 0,
                 max_ctx: int = 2048):
        """Args added for SRC-TEH token-level routing:

        d_hidden: per-token expert width. When None (legacy/test path) only
            the d_sem fact-memory path is active. When set, the cortex
            additionally constructs `n_blocks` TransformerBlocks at d_hidden
            and a learnable fact-memory cross-attention layer — invoked via
            `forward_tokens(x_tokens)` from the expert-choice router.
        n_blocks: depth of the token-level expert (default 3 per SRC-TEH).
        max_ctx: max sequence length per token-level call (router caps it).
        """
        super().__init__()
        self.d_sem      = d_sem
        self.n_heads    = max(1, n_heads)
        self.head_dim   = d_sem // self.n_heads
        self.d_hidden   = d_hidden
        self.n_blocks   = int(n_blocks)

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

        # ── Token-level expert stack (SRC-TEH) ────────────────────────────
        # Constructed only when d_hidden + n_blocks > 0. Otherwise the
        # cortex is the lightweight (legacy) fact-memory enrichment.
        if self.d_hidden is not None and self.n_blocks > 0:
            self.expert_blocks = nn.ModuleList([
                TransformerBlock(self.d_hidden, n_heads=self.n_heads,
                                 max_ctx=max_ctx)
                for _ in range(self.n_blocks)
            ])
            # Fact-memory cross-attention at the expert width.  Learnable
            # key/value bank specialised for arithmetic / symbolic facts.
            self.fact_keys_h = nn.Parameter(
                torch.randn(memory_size, self.d_hidden) * (1.0 / math.sqrt(self.d_hidden)))
            self.fact_vals_h = nn.Parameter(torch.zeros(memory_size, self.d_hidden))
            self.fact_norm   = RMSNorm(self.d_hidden)
            self.fact_q      = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
            self.fact_out    = nn.Linear(self.d_hidden, self.d_hidden, bias=False)
            nn.init.zeros_(self.fact_out.weight)
            self.expert_norm = RMSNorm(self.d_hidden)
        else:
            self.expert_blocks = None

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
                gws_context: torch.Tensor | None = None,
                maturity: float | None = None) -> torch.Tensor:
        """x: (B, d_sem) or (B, S, d_sem).

        vesicle_gate ∈ [0, 1]: topic-routing strength (math-cortex activation).
        gws_context: optional (B, d_sem) — Global Workspace broadcast used as
                     plasticity context for the Hebbian fast-weight layer.
                     Enables dual-timescale binding (slow facts × fast episodic).
        maturity: optional MAT scalar in [0, 1] — convex fade-in weight applied
                  to the expert residual:
                      h_out = (1 - m_eff)·h_in + m_eff·Expert(h_in)
                  with m_eff = max(maturity, 0.05) so a 5% noise broadcast
                  flows through even at very low maturity. None preserves
                  legacy (full-weight) behaviour.

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
            # Clone the slice — _update_wm later writes to self._wm_keys/_wm_vals
            # in-place; without the clone the saved-for-backward tensor would
            # have its version bumped and autograd would raise the
            # "modified by an inplace operation" error.
            wm_k = self._wm_keys[:n_filled].clone()
            wm_v = self._wm_vals[:n_filled].clone()
            wm_enrichment = self._diff_attn(h, wm_k, wm_v)
        else:
            wm_enrichment = torch.zeros_like(x_flat)

        enrichment = fact_enrichment + 0.3 * wm_enrichment
        # Maturity fade-in: m_eff = max(M, 0.05) → 5% noise floor pre-awakening,
        # full weight once M ≥ ~1.0. None → legacy full-weight passthrough.
        m_eff = 1.0 if maturity is None else max(float(maturity), 0.05)
        enriched   = x_flat + m_eff * vesicle_gate * enrichment

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

    # ------------------------------------------------------------------
    # Token-level expert pass (SRC-TEH)
    # ------------------------------------------------------------------
    def forward_tokens(self, x: torch.Tensor,
                       maturity: float | None = None) -> torch.Tensor:
        """Process a routed batch of tokens through the 3-block expert.

        x: (B, C, d_hidden) — tokens pulled by the ExpertChoiceRouter for
            this expert (C = capacity per expert).
        maturity: optional MAT scalar; gates a 5%-noise-floor → full residual.

        Returns (B, C, d_hidden) with expert enrichment applied as a residual.
        Falls back to identity when the token-level stack was not constructed
        (legacy d_hidden=None mode).
        """
        if self.expert_blocks is None:
            return x

        B, C, D = x.shape
        h = x
        # Layer stack — standard self-attention + SwiGLU at d_hidden.
        for blk in self.expert_blocks:
            h = blk(h)

        # Fact-memory cross-attention: queries from current tokens, K/V from
        # the learnable symbolic fact bank (specialised for arithmetic).
        q  = self.fact_q(self.fact_norm(h.float()).to(h.dtype))           # (B, C, D)
        # Single-head cross-attn (memory_size is small; keep simple).
        scale = 1.0 / math.sqrt(D)
        # cast bank to running dtype to avoid bf16/fp32 whiplash
        fk = self.fact_keys_h.to(dtype=h.dtype, device=h.device)
        fv = self.fact_vals_h.to(dtype=h.dtype, device=h.device)
        attn = torch.einsum("bcd,md->bcm", q, fk) * scale
        attn = F.softmax(attn.float(), dim=-1).to(h.dtype)
        enrich = torch.einsum("bcm,md->bcd", attn, fv)
        h = h + self.fact_out(enrich)
        h = self.expert_norm(h.float()).to(h.dtype)

        # SRC-TEH path: caller (brain.forward_lm) passes an already
        # phase-gated maturity, so we honour it directly without the legacy
        # 5% floor — that floor was useful for the d_sem residual path but
        # pumps unconditional noise into the trunk when applied at d_hidden.
        # Below MAT-phase 1e-3 the residual is effectively zero (passthrough).
        m_eff = 1.0 if maturity is None else max(float(maturity), 0.0)
        if m_eff < 1e-3:
            return x
        return x + m_eff * (h - x)

    def _disabled_output(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
