"""Hippocampus: episodic memory with DG/CA3/CA1 analogs + novelty signal.

- DG (dentate gyrus): sparse pattern separator (top-k mask) for storage keys.
- CA3: associative recall via cosine attention over store.
- CA1: output projection back into d_sem.
- Novelty: 1 - max cosine similarity to stored keys.

Memory store is a circular buffer of (key, value) tensors held as plain tensors
(not nn.Parameter) so it persists across batches but is not updated by SGD.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .genome_configurable import GenomeConfigurable


class Hippocampus(nn.Module, GenomeConfigurable):
    def __init__(self, d_sem: int, capacity: int, topk: int, sparse_k: int):
        super().__init__()
        self._genome_env = {}
        self.d_sem = d_sem
        self.capacity = capacity
        self.topk = topk
        self.sparse_k = sparse_k
        # Genome-tunable gates (defaults = standard behavior)
        self._recall_gate = 1.0    # how strongly recall fires
        self._store_gate = 1.0     # how strongly store fires
        self._novelty_bias = 0.0   # shift novelty threshold
        self._modulation_gain = 1.0  # NT modulation strength
        # Genome-driven algorithmic choices (set by configure_from_genome)
        self._recall_method = 'cosine'       # 'cosine' | 'attention' | 'surprise_weighted'
        self._sparsity_ratio = 0.5           # fraction of DG activations kept
        self._recall_temperature = 1.0       # softmax temperature for recall attention

        # DG: sparse projection
        self.dg = nn.Linear(d_sem, d_sem, bias=False)
        # CA3: learned attention recall (used when recall_method='attention')
        self.ca3_query = nn.Linear(d_sem, d_sem, bias=False)
        self.ca3_key = nn.Linear(d_sem, d_sem, bias=False)
        # CA1: output projection
        self.ca1 = nn.Linear(d_sem, d_sem, bias=False)

        # Memory store (non-trainable buffers)
        self.register_buffer("keys",   torch.zeros(capacity, d_sem))
        self.register_buffer("values", torch.zeros(capacity, d_sem))
        self.register_buffer("filled", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))

    def _sparsify(self, x: torch.Tensor) -> torch.Tensor:
        # Genome-driven sparsity: keep fraction of activations (DG pattern separation)
        k = max(1, int(self._sparsity_ratio * x.size(-1)))
        k = min(k, x.size(-1))
        topv, topi = x.topk(k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topi, 1.0)
        return x * mask

    def configure_from_genome(self, env: dict, structural=None):
        """Apply genome-compiled parameters to hippocampal behavior.
        The genome encodes BOTH scalar gates AND algorithmic choices:
          - recall_method: 'cosine' | 'attention' | 'surprise_weighted'
          - sparsity_ratio: fraction of DG activations kept (pattern separation density)
          - recall_temperature: softmax temperature for attention recall
          - recall_gate, store_gate, novelty_bias, modulation_gain: scalar gates
        """
        super().configure_from_genome(env, structural=structural)
        self._recall_gate = max(0.01, self.genv_float('recall_gate', 1.0))
        self._store_gate = max(0.01, self.genv_float('store_gate', 1.0))
        self._novelty_bias = self.genv_float('novelty_bias', 0.0)
        self._modulation_gain = max(0.01, self.genv_float('modulation_gain', 1.0))
        # Algorithmic choices from genome opcode patterns
        self._recall_method = self.genv_str('recall_method', 'cosine')
        self._sparsity_ratio = max(0.05, min(0.95, self.genv_float('sparsity_ratio', 0.5)))
        self._recall_temperature = max(0.1, self.genv_float('recall_temperature', 1.0))

    @torch.no_grad()
    def store(self, query: torch.Tensor, value: torch.Tensor) -> None:
        """Store (key, value) pairs. Both: (B, d_sem)."""
        key = self._sparsify(self.dg(query))
        B = key.size(0)
        for b in range(B):
            idx = int(self.write_ptr.item()) % self.capacity
            self.keys[idx] = key[b].detach()
            self.values[idx] = value[b].detach()
            self.filled[idx] = True
            self.write_ptr += 1

    def recall(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """query: (B, d_sem) → (recalls (B, topk, d_sem), novelty (B,)).

        Recall method is genome-driven:
          - 'cosine': standard cosine top-k (default)
          - 'attention': learned CA3 query/key attention with softmax
          - 'surprise_weighted': cosine similarity weighted by recency/novelty
        """
        B = query.size(0)
        if not self.filled.any():
            zeros = torch.zeros(B, self.topk, self.d_sem, device=query.device, dtype=query.dtype)
            ones = torch.ones(B, device=query.device, dtype=query.dtype)
            return zeros, ones

        key = self._sparsify(self.dg(query))                # (B, d_sem)
        keys = self.keys[self.filled]                       # (N, d_sem)
        vals = self.values[self.filled]                     # (N, d_sem)

        if self._recall_method == 'attention':
            # Learned attention: CA3 query/key projections + softmax
            q = self.ca3_query(key)                          # (B, d_sem)
            k = self.ca3_key(keys)                           # (N, d_sem)
            sim = torch.matmul(q, k.T) / (self.d_sem ** 0.5)  # (B, N)
            sim = sim / max(0.1, self._recall_temperature)
            attn = F.softmax(sim, dim=-1)                    # (B, N)
            k_actual = min(self.topk, sim.size(1))
            topv, topi = attn.topk(k_actual, dim=-1)
        elif self._recall_method == 'surprise_weighted':
            # Cosine similarity weighted by position (newer = higher weight)
            sim = F.cosine_similarity(
                key.unsqueeze(1), keys.unsqueeze(0), dim=-1)  # (B, N)
            N = sim.size(1)
            recency = torch.linspace(0.5, 1.0, N, device=sim.device)
            sim = sim * recency.unsqueeze(0)
            k_actual = min(self.topk, sim.size(1))
            topv, topi = sim.topk(k_actual, dim=-1)
        else:
            # Default cosine top-k recall
            sim = F.cosine_similarity(
                key.unsqueeze(1), keys.unsqueeze(0), dim=-1)  # (B, N)
            k_actual = min(self.topk, sim.size(1))
            topv, topi = sim.topk(k_actual, dim=-1)

        recalled = vals[topi]                               # (B, k, d_sem)
        recalled = self.ca1(recalled)
        # Genome gate: scale recall strength
        recalled = recalled * self._recall_gate
        # Pad if N < topk
        if k_actual < self.topk:
            pad = torch.zeros(B, self.topk - k_actual, self.d_sem,
                              device=query.device, dtype=query.dtype)
            recalled = torch.cat([recalled, pad], dim=1)

        novelty = 1.0 - topv[:, 0].clamp(-1, 1)             # (B,)
        # Genome bias: shift novelty threshold
        novelty = (novelty + self._novelty_bias).clamp(0, 1)
        return recalled, novelty
