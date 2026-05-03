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
                 rank: int = 0):
        super().__init__()
        self.d_hyper = int(d_hidden * expansion)
        if rank <= 0:
            rank = max(8, self.d_hyper // 8)
        self.rank = rank

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
                 mod_capacity: float = 0.5):
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

        # Collect layer hidden states for predictive coding
        layer_states = []
        pred_coding_loss = torch.tensor(0.0, device=ids.device)

        for i, (blk, adapter) in enumerate(zip(self.blocks, self.adapters)):
            if self.gradient_checkpointing and self.training:
                # checkpoint doesn't support kwargs well, so wrap in lambda
                h = torch.utils.checkpoint.checkpoint(
                    lambda x, _nt=nt: blk(x, nt=_nt), h, use_reentrant=False)
                h = torch.utils.checkpoint.checkpoint(
                    adapter, h, use_reentrant=False)
            else:
                h = blk(h, nt=nt)
                h = adapter(h)     # geometry-adapted residual after each block

            # Store for predictive coding
            if len(self.pred_coding) > 0:
                layer_states.append(h)

        # Predictive coding loss: each layer predicts the next
        if len(self.pred_coding) > 0 and len(layer_states) > 1:
            for i, pc_head in enumerate(self.pred_coding):
                if i + 1 < len(layer_states):
                    pred_coding_loss = pred_coding_loss + pc_head(
                        layer_states[i], layer_states[i + 1])
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
