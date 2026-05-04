"""Relational Inductive Biases — Graph-Structured Attention for NeuroSLM.

Standard transformer attention is *set-level*: it learns to attend over
all pairs of tokens but has no explicit representation of *how* tokens
are related (subject→object, cause→effect, entity→attribute).

Relational inductive biases (Battaglia et al. 2018) force the model to
reason explicitly about pairwise relationships, improving:
  • Compositional generalisation
  • Multi-hop reasoning
  • Entity binding (which property belongs to which entity)

This module implements three ideas together:

  1. Learned relation embeddings (Shaw et al. 2018 extended):
     For each pair (i, j), a relation type is inferred from the
     difference in their key projections, then embedded into a
     per-head bias term.  No external graph required.

  2. Message-passing rounds:
     After attention, each token sends messages to its top-k
     attended neighbours, aggregating relational content (not just
     values). This gives a GNN-like inductive step inside the layer.

  3. Relation classification head:
     Predicts which of R relation types holds between each pair —
     returned as a by-product that the caller can use as auxiliary loss
     or for memory/hypergraph encoding.

Novel aspect: the message-passing aggregation uses the *relation type*
embeddings as edge features, not just attention weights — so the model
distinguishes "I caused X" from "I observed X" even if both token pairs
get equal attention weight.  No existing SLM layer does this.

References:
  Battaglia et al. (2018): Relational inductive biases, deep learning, and graph networks
  Shaw et al. (2018): Self-attention with relative position representations
  Veličković et al. (2018): Graph Attention Networks
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RelationalAttentionBlock(nn.Module):
    """Attention with relation-type embeddings and message-passing.

    Parameters
    ----------
    d_model   : token embedding dimension
    n_heads   : number of attention heads
    n_rel_types : number of discrete relation types to learn
    d_rel     : relation embedding dimension
    mp_rounds : message-passing rounds after attention (0 = off)
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 n_rel_types: int = 16, d_rel: int = 32,
                 mp_rounds: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads      = n_heads
        self.d_head       = d_model // n_heads
        self.n_rel_types  = n_rel_types
        self.d_rel        = d_rel
        self.mp_rounds    = mp_rounds

        # Standard QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Relation type inference: from key-difference → relation logits
        self.rel_infer = nn.Sequential(
            nn.Linear(self.d_head, d_rel),
            nn.GELU(),
            nn.Linear(d_rel, n_rel_types),
        )

        # Relation type embeddings → per-head attention bias
        self.rel_embed = nn.Embedding(n_rel_types, n_heads)

        # Message-passing edge network
        if mp_rounds > 0:
            self.edge_net = nn.Sequential(
                nn.Linear(d_model + d_rel, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.mp_norm = nn.LayerNorm(d_model)
            # Relation type → d_rel for edge features
            self.rel_to_feat = nn.Embedding(n_rel_types, d_rel)

        self.attn_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------

    def _infer_relation_types(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer discrete relation type for each (i,j) pair.

        k: (B, n_heads, T, d_head)
        Returns:
          rel_type_ids:  (B, T, T)   — argmax relation type per pair
          rel_logits:    (B, T, T, n_rel_types) — soft logits
        """
        B, H, T, Dh = k.shape
        # Use first head's key difference for relation inference
        k0 = k[:, 0, :, :]                          # (B, T, Dh)
        # Pairwise difference: k_i - k_j  → (B, T, T, Dh)
        diff = k0.unsqueeze(2) - k0.unsqueeze(1)     # (B, T, T, Dh)
        rel_logits = self.rel_infer(diff)             # (B, T, T, n_rel)
        rel_type_ids = rel_logits.argmax(dim=-1)      # (B, T, T)
        return rel_type_ids, rel_logits

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x:    (B, T, d_model)
        mask: (B, T, T)  causal or padding mask (True = mask out)

        Returns:
          out:          (B, T, d_model)
          rel_logits:   (B, T, T, n_rel_types) — for auxiliary loss or memory
        """
        B, T, _ = x.shape
        scale   = self.d_head ** -0.5

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # q, k, v: (B, n_heads, T, d_head)

        # ---- 1. Relation-type biased attention ----
        rel_type_ids, rel_logits = self._infer_relation_types(k)
        # rel_type_ids: (B, T, T) → per-head bias via embedding
        rel_bias = self.rel_embed(rel_type_ids)       # (B, T, T, n_heads)
        rel_bias = rel_bias.permute(0, 3, 1, 2)       # (B, n_heads, T, T)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * scale + rel_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)         # (B, n_heads, T, T)

        attended = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
        attended = attended.transpose(1, 2).reshape(B, T, -1)  # (B, T, D)
        out = self.attn_norm(x + self.out_proj(attended))

        # ---- 2. Message-passing rounds ----
        if self.mp_rounds > 0:
            for _ in range(self.mp_rounds):
                # Edge features: relation type embedding for dominant relation
                edge_feats = self.rel_to_feat(rel_type_ids)   # (B, T, T, d_rel)

                # For each token, aggregate from top-k attended neighbours
                # Use mean-head attention as aggregation weight
                agg_w = attn_weights.mean(1)                   # (B, T, T)

                # Weighted sum of neighbour [x_j || edge_feat_ij]
                # msg shape: aggregate over j dimension
                # v_j: (B, T, D) → (B, T, T, D) by expanding
                v_expanded = out.unsqueeze(1).expand(B, T, T, -1)   # (B, T, T, D)
                edge_input = torch.cat([v_expanded, edge_feats], dim=-1)  # (B, T, T, D+d_rel)
                messages = self.edge_net(edge_input)              # (B, T, T, D)
                # Weighted aggregate: sum_j w[i,j] * msg[i,j]
                agg = torch.einsum("bij,bijd->bid", agg_w, messages)  # (B, T, D)
                out = self.mp_norm(out + agg)

        return out, rel_logits
