"""Memory cross-attention.

Turns the consolidated memory graph + narrative streams into a queryable
key/value bank that any transformer layer can attend to. This is the
"memory becomes compute" trick: a 100M-param model with retrieval over a
growing memory bank operates at the effective capacity of a much larger
parametric model, *and* its memory persists across runs (transferable
.mem files in Git LFS).

Two sources of K/V are merged:
  * Consolidated graph nodes (semantic embeddings of generalized concepts)
  * Narrative-stream summaries (autobiographical, world, top entities)

The retrieval is differentiable through the projections, but the memory
contents themselves are written non-differentiably (write-time selection
is gated by the comprehension gate).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCrossAttention(nn.Module):
    """Cross-attention from token states (queries) to memory K/V.

    Args:
        d_model: token state dim.
        d_mem: memory embedding dim (typically d_sem).
        n_heads: number of cross-attn heads.
        max_retrieved: max number of memory items per forward (top-k by sim).
    """

    def __init__(self, d_model: int, d_mem: int,
                 n_heads: int = 4, max_retrieved: int = 16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_mem = d_mem
        self.h = n_heads
        self.dk = d_model // n_heads
        self.max_retrieved = max_retrieved

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_mem, d_model)
        self.v_proj = nn.Linear(d_mem, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mem_keys: torch.Tensor | None,
                mem_vals: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D)
        mem_keys: (M, d_mem) or None — None ⇒ no-op (returns x unchanged).
        mem_vals: (M, d_mem) or None — defaults to mem_keys.
        """
        if mem_keys is None or mem_keys.numel() == 0:
            return x
        if mem_vals is None:
            mem_vals = mem_keys

        B, T, D = x.shape
        h_in = self.norm(x)

        # Top-K memory selection per batch element (avg over time first)
        with torch.no_grad():
            q_summary = h_in.mean(dim=1)               # (B, D)
            # Use first-D projection for similarity (cheap)
            mk_proj = mem_keys[:, :min(D, mem_keys.size(-1))]
            qs_proj = q_summary[:, :mk_proj.size(-1)]
            sim = F.normalize(qs_proj, dim=-1) @ F.normalize(mk_proj, dim=-1).T
            k = min(self.max_retrieved, mem_keys.size(0))
            _, top_idx = sim.topk(k, dim=-1)           # (B, k)

        # Gather memory per batch
        # (B, k, d_mem)
        mk = mem_keys[top_idx]
        mv = mem_vals[top_idx]

        Q = self.q_proj(h_in).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k_proj(mk).view(B, k, self.h, self.dk).transpose(1, 2)
        V = self.v_proj(mv).view(B, k, self.h, self.dk).transpose(1, 2)

        att = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, V)                     # (B,h,T,dk)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.o_proj(out)

        gate = self.gate(h_in)                         # (B, T, 1)
        return x + gate * out


# ────────────────────────────────────────────────────────────────────────
# Helper: build a fused memory bank from the brain's memory subsystems
# ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_memory_bank(consolidated, narrative_system, device,
                      max_items: int = 256, d_sem: int | None = None
                      ) -> torch.Tensor | None:
    """Concatenates consolidated-graph node vectors + narrative summaries
    into a single (M, d_sem) tensor. Returns None if memory is empty.
    """
    items = []

    # Consolidated graph nodes
    try:
        nodes = list(consolidated.graph.nodes(data=True))
        for _, data in nodes[-max_items:]:
            v = data.get("content_vec")
            if v is None:
                continue
            t = torch.as_tensor(v, dtype=torch.float32, device=device).flatten()
            items.append(t)
    except Exception:
        pass

    # Narrative summaries (auto, world, top-K entities)
    try:
        items.append(narrative_system.autobiographical.summary.detach().to(device))
        items.append(narrative_system.world.summary.detach().to(device))
        for stream in list(narrative_system.entities.values())[:8]:
            items.append(stream.summary.detach().to(device))
    except Exception:
        pass

    if not items:
        return None

    # Pad/truncate to common dim
    target = d_sem or items[0].size(0)
    fixed = []
    for t in items:
        if t.size(0) < target:
            t = F.pad(t, (0, target - t.size(0)))
        elif t.size(0) > target:
            t = t[:target]
        fixed.append(t)
    return torch.stack(fixed, dim=0)
