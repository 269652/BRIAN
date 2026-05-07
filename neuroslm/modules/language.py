"""Language cortex: combined Wernicke (comprehension) + Broca (production).

Contains the token embeddings, transformer stack, and LM head.
The hidden state at the last position is exposed as the "comprehension embedding"
projected into d_sem space for downstream modules.

Includes a **NeuralGeometryAdapter** — a meta-trainable layer that dynamically
reshapes the hidden-state manifold between transformer blocks.  The adapter
projects activations into a higher-dimensional "hyperbolic-like" space where
neurons can form richer inter-connections, then projects back.  The up/down
projections and a learned *connectivity kernel* are meta-trained so the network
discovers neural topologies that pack more linguistic understanding into fewer
parameters than a vanilla transformer.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import TransformerBlock, RMSNorm
from .neuro_attention import PredictiveCodingHead
from .differential_attention import DiffTransformerBlock
from .mixture_of_depths import MoDBlock


# ---------------------------------------------------------------------------
# Neural Geometry Adapter — meta-trainable higher-dimensional wiring
# ---------------------------------------------------------------------------

class NeuralGeometryAdapter(nn.Module):
    """Learns to reshape the hidden-state geometry between transformer blocks.

    Core idea: project d_hidden → d_hyper (larger), apply a learned
    *connectivity kernel* (low-rank + sparse gating), then project back.
    The connectivity kernel acts as a dynamic adjacency matrix in the
    higher-dimensional space, enabling neurons to form connections that
    do not exist in the original d_hidden topology.

    The adapter is deliberately lightweight:
      up:     (d_hidden → d_hyper) via linear
      kernel: low-rank (d_hyper, rank) @ (rank, d_hyper) + sigmoid gate
      down:   (d_hyper → d_hidden) via linear

    A residual connection and layer-norm ensure stability.
    The adapter parameters are included in the meta-training parameter set
    so the geometry itself is meta-learned.
    """

    def __init__(self, d_hidden: int, expansion: float = 2.0,
                 rank: int = 0, max_rank: int = 0):
        super().__init__()
        self.d_hyper = int(d_hidden * expansion)
        if rank <= 0:
            rank = max(8, self.d_hyper // 8)
        self.rank = rank
        self.max_rank = max_rank if max_rank > 0 else self.d_hyper // 2

        self.norm = RMSNorm(d_hidden)
        self.up = nn.Linear(d_hidden, self.d_hyper, bias=False)
        # Low-rank connectivity kernel
        self.kern_a = nn.Parameter(torch.randn(self.d_hyper, rank) * 0.01)
        self.kern_b = nn.Parameter(torch.randn(rank, self.d_hyper) * 0.01)
        # Per-dimension gate (sigmoid) — controls which hyper-dimensions
        # are "active connections" for this input
        self.gate = nn.Linear(self.d_hyper, self.d_hyper, bias=True)
        self.down = nn.Linear(self.d_hyper, d_hidden, bias=False)

        # Init down projection to zero so the adapter starts as identity
        nn.init.zeros_(self.down.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # gates start mostly closed

        # BDNF accumulator and cooldown for structural growth
        self._bdnf_accum: float = 0.0
        self._bdnf_cooldown: int = 0

    @torch.no_grad()
    def bdnf_grow(self, bdnf: float, phi: float,
                  growth_threshold: float = 1.5,
                  delta_rank: int = 4,
                  cooldown_steps: int = 200) -> bool:
        """Accumulate BDNF×Φ signal; grow kernel rank when threshold is crossed.

        Called between training steps (outside autograd graph).
        Returns True if growth happened.

        bdnf: BDNF proxy signal in [0, 1] (from TrophicSystem)
        phi:  Integrated information proxy in [0, 1] (from ConsciousnessMetrics)
        """
        if self._bdnf_cooldown > 0:
            self._bdnf_cooldown -= 1
            return False
        if self.rank >= self.max_rank:
            return False

        self._bdnf_accum += float(bdnf) * float(phi)
        if self._bdnf_accum < growth_threshold:
            return False

        # Grow: append delta_rank zero-initialized columns to kern_a, rows to kern_b
        new_rank = min(self.rank + delta_rank, self.max_rank)
        dr = new_rank - self.rank

        # kern_a: (d_hyper, rank) → (d_hyper, new_rank)
        new_a = torch.zeros(self.d_hyper, dr, device=self.kern_a.device,
                            dtype=self.kern_a.dtype)
        self.kern_a = nn.Parameter(torch.cat([self.kern_a.data, new_a], dim=1))

        # kern_b: (rank, d_hyper) → (new_rank, d_hyper)
        new_b = torch.zeros(dr, self.d_hyper, device=self.kern_b.device,
                            dtype=self.kern_b.dtype)
        self.kern_b = nn.Parameter(torch.cat([self.kern_b.data, new_b], dim=0))

        self.rank = new_rank
        self._bdnf_accum = 0.0
        self._bdnf_cooldown = cooldown_steps
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_hidden) → (B, T, d_hidden) with geometry-adapted residual."""
        h = self.norm(x)
        z = self.up(h)                                    # (B, T, d_hyper)
        # Connectivity kernel: low-rank transform in hyper-space
        # This is the "virtual wiring" — neurons interact through a
        # learned adjacency that doesn't exist in the base transformer
        k = z @ self.kern_a @ self.kern_b                 # (B, T, d_hyper)
        # Gating: sigmoid gate decides which hyper-connections are active
        g = torch.sigmoid(self.gate(z))                   # (B, T, d_hyper)
        z_new = F.silu(k) * g                             # gated activation
        out = self.down(z_new)                             # (B, T, d_hidden)
        return x + out                                     # residual


class LanguageCortex(nn.Module):
    def __init__(self, vocab_size: int, d_hidden: int, d_sem: int,
                 n_layers: int, n_heads: int, max_ctx: int,
                 n_kv_heads: int | None = None,
                 n_nt: int = 0,
                 hebbian_rank: int = 0,
                 geometry_expansion: float = 2.0,
                 gradient_checkpointing: bool = False,
                 mod_capacity: float = 0.5,
                 baseline: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.n_nt = n_nt
        self.tok_emb = nn.Embedding(vocab_size, d_hidden)

        # Interleaved architecture (novel hybrid):
        #   Layer pattern: [Standard, DiffAttn, MoD+DiffAttn, Standard, DiffAttn, MoD+DiffAttn, ...]
        #   - Standard blocks: Hebbian traces + NT modulation (in-context learning)
        #   - DiffAttn blocks: noise cancellation (hallucination reduction)
        #   - MoD blocks: dynamic compute allocation (efficiency)
        #   + NeuralGeometryAdapter after every block (meta-learnable wiring)
        self.blocks = nn.ModuleList()
        self.adapters = nn.ModuleList()
        if baseline:
            # Only standard TransformerBlocks, no adapters
            for i in range(n_layers):
                self.blocks.append(TransformerBlock(
                    d_hidden, n_heads, max_ctx, n_kv_heads,
                    n_nt=n_nt, hebbian_rank=hebbian_rank))
        else:
            for i in range(n_layers):
                pattern = i % 3
                if pattern == 0:
                    # Standard attention + Hebbian traces
                    self.blocks.append(TransformerBlock(
                        d_hidden, n_heads, max_ctx, n_kv_heads,
                        n_nt=n_nt, hebbian_rank=hebbian_rank))
                elif pattern == 1:
                    # Differential attention (noise cancellation)
                    self.blocks.append(DiffTransformerBlock(
                        d_hidden, n_heads, max_ctx, n_kv_heads, n_nt=n_nt))
                else:
                    # Mixture-of-Depths with differential attention
                    self.blocks.append(MoDBlock(
                        d_hidden, n_heads, max_ctx, n_kv_heads,
                        n_nt=n_nt, capacity_ratio=mod_capacity,
                        use_diff_attn=True))
                self.adapters.append(
                    NeuralGeometryAdapter(d_hidden, expansion=geometry_expansion)
                )

        # Novel: Predictive Coding — each layer predicts next layer's output
        # Deep supervision gives each layer its own gradient signal
        self.pred_coding = nn.ModuleList([
            PredictiveCodingHead(d_hidden) for _ in range(n_layers - 1)
        ]) if n_layers > 1 else nn.ModuleList()

        self.norm_f = RMSNorm(d_hidden)
        # Tied output head
        self.lm_head = nn.Linear(d_hidden, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Project last-layer hidden state into shared semantic space
        self.to_sem = nn.Linear(d_hidden, d_sem, bias=False)
        # Inverse projection: take a thought (d_sem) and condition generation
        self.from_sem = nn.Linear(d_sem, d_hidden, bias=False)

        # Proper init: small embeddings + scaled output projections.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.to_sem.weight, mean=0.0, std=0.02)
        # Zero-init the conditioning projection so initial training is a clean LM.
        nn.init.zeros_(self.from_sem.weight)
        for blk in self.blocks:
            for p in blk.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, thought: torch.Tensor | None = None,
                motor_bias: torch.Tensor | None = None,
                nt: torch.Tensor | None = None):
        """ids: (B, T). thought: optional (B, d_sem) injected as a prefix bias.
        motor_bias: optional (B, d_hidden) added to the LAST position's hidden
        state before the LM head.
        nt: optional (B, N_NT) neurotransmitter vector for attention modulation.

        Returns: (logits, sem, h, pred_coding_loss)
          pred_coding_loss: scalar predictive coding loss (0.0 if no PC heads).
        """
        h = self.tok_emb(ids)
        if thought is not None:
            bias = self.from_sem(thought).unsqueeze(1)  # (B, 1, d_hidden)
            h = h + bias

        # Compute predictive coding loss incrementally to avoid storing
        # all intermediate layer activations which increases peak memory.
        prev_layer = None
        pc_counter = 0
        pred_coding_loss = torch.tensor(0.0, device=h.device)

        def _run_block(blk, x, nt_vec):
            """Run one block, with gradient checkpointing for all block types."""
            if self.gradient_checkpointing and self.training:
                if nt_vec is None:
                    return torch.utils.checkpoint.checkpoint(
                        blk, x, use_reentrant=False)
                else:
                    return torch.utils.checkpoint.checkpoint(
                        lambda _x, _nt: blk(_x, nt=_nt), x, nt_vec,
                        use_reentrant=False)
            return blk(x, nt=nt_vec)

        # CALM early-exit state (inference only: no overhead during training)
        B, T = h.shape[:2]
        # Gradient checkpointing is also activated on XLA (TPU rematerialisation).
        # The existing self.gradient_checkpointing flag covers both CUDA and XLA.
        use_calm = (not self.training
                    and hasattr(self, 'adapters')
                    and len(self.adapters) > 0)
        if use_calm:
            n_layers_total = len(self.blocks)
            calm_frozen = torch.zeros_like(h)   # frozen hidden states for exited tokens
            calm_mask   = torch.zeros(B, T, dtype=torch.bool, device=h.device)

        if hasattr(self, 'adapters') and len(self.adapters) > 0:
            for i, (blk, adapter) in enumerate(zip(self.blocks, self.adapters)):
                # Inject frozen states for CALM-exited tokens before the block
                if use_calm and calm_mask.any():
                    h = torch.where(calm_mask.unsqueeze(-1), calm_frozen, h)

                h = _run_block(blk, h, nt)
                h = adapter(h)

                # CALM: evaluate per-token confidence; freeze tokens that are "done"
                if use_calm and hasattr(blk, 'calm_head'):
                    thresh = blk.calm_head.threshold(i, n_layers_total)
                    with torch.no_grad():
                        conf = blk.calm_head(h.detach())   # (B, T)
                    new_exits = (conf > thresh) & ~calm_mask
                    if new_exits.any():
                        calm_frozen = torch.where(new_exits.unsqueeze(-1), h, calm_frozen)
                        calm_mask   = calm_mask | new_exits

                if len(self.pred_coding) > 0:
                    if prev_layer is not None and pc_counter < len(self.pred_coding):
                        pred_coding_loss = pred_coding_loss + self.pred_coding[pc_counter](prev_layer, h)
                        pc_counter += 1  # only advance when a head is consumed
                    prev_layer = h

            # Apply any remaining frozen states after the final layer
            if use_calm and calm_mask.any():
                h = torch.where(calm_mask.unsqueeze(-1), calm_frozen, h)
        else:
            for i, blk in enumerate(self.blocks):
                h = _run_block(blk, h, nt)
                if len(self.pred_coding) > 0:
                    if prev_layer is not None and pc_counter < len(self.pred_coding):
                        pred_coding_loss = pred_coding_loss + self.pred_coding[pc_counter](prev_layer, h)
                        pc_counter += 1
                    prev_layer = h

        # Normalize predictive coding loss by number of heads if present
        if len(self.pred_coding) > 0 and pc_counter > 0:
            pred_coding_loss = pred_coding_loss / len(self.pred_coding)

        h = self.norm_f(h)
        if motor_bias is not None:
            # Add bias only to the last position (the one that will be sampled).
            h_last = h[:, -1:, :] + motor_bias.unsqueeze(1)
            logits = torch.cat([self.lm_head(h[:, :-1, :]),
                                self.lm_head(h_last)], dim=1)
        else:
            logits = self.lm_head(h)
        sem = self.to_sem(h.mean(dim=1))
        return logits, sem, h, pred_coding_loss

    def bdnf_grow_all(self, bdnf: float, phi: float) -> int:
        """Trigger BDNF-driven structural growth on all NeuralGeometryAdapters.

        Should be called between training steps (not inside the forward pass).
        Returns the number of adapters that grew.
        """
        if not hasattr(self, 'adapters') or len(self.adapters) == 0:
            return 0
        grew = 0
        for adapter in self.adapters:
            if isinstance(adapter, NeuralGeometryAdapter):
                if adapter.bdnf_grow(bdnf, phi):
                    grew += 1
        return grew
