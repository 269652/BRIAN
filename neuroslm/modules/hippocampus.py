"""Hippocampus: SOTA multi-dimensional memory enrichment of the Global Workspace.

Implements Complementary Learning Systems (CLS) theory with:
  - DG (Dentate Gyrus): sparse pattern separation via k-winners + random projection
  - CA3: autoassociative pattern completion (learned query-key attention)
  - CA1: mismatch / novelty detection (expected vs. actual)
  - Multi-dimensional retrieval:
      Semantic  — cosine similarity of content embeddings
      Temporal  — recency-weighted (exponential decay over buffer position)
      Mood      — NT state cosine similarity + valence distance
      Associate — multi-hop attention chaining from best semantic hit
  - Theta-gamma coupling: ACh gates encoding mode (high ACh) vs retrieval mode

The enrich_gws() method is the primary interface used by brain.py — it receives
the current GWS slots and floating thought, retrieves relevant memories across all
dimensions, and returns enriched GWS slots + a novelty signal.

References:
  McClelland et al. (1995) Why there are complementary learning systems.
  Hasselmo (2006) Role of acetylcholine in learning and memory.
  Lisman et al. (2017) Viewpoints: How the hippocampus contributes to memory.
  Kumaran et al. (2016) What learning systems do intelligent agents need?
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain_module import BrainModule


class Hippocampus(BrainModule):
    def __init__(self, d_sem: int, capacity: int, topk: int, sparse_k: int):
        super().__init__()
        self.d_sem = d_sem
        self.capacity = capacity
        self.topk = topk
        self.sparse_k = sparse_k

        # --- DG: sparse pattern separation ---
        # Random projection to higher dimension then sparsify (pattern orthogonalisation)
        dg_dim = min(d_sem * 2, 1024)
        self.dg_expand  = nn.Linear(d_sem, dg_dim, bias=False)
        self.dg_project = nn.Linear(dg_dim, d_sem, bias=False)
        # Freeze random projection (DG sparse codes are fixed, not learned)
        nn.init.orthogonal_(self.dg_expand.weight)
        self.dg_expand.weight.requires_grad = False

        # --- CA3: autoassociative recall ---
        self.ca3_query = nn.Linear(d_sem, d_sem, bias=False)
        self.ca3_key   = nn.Linear(d_sem, d_sem, bias=False)
        self.ca3_value = nn.Linear(d_sem, d_sem, bias=False)

        # --- CA1: mismatch / novelty ---
        # CA3 path: what we expect from memory
        self.ca3_to_ca1   = nn.Linear(d_sem, d_sem, bias=False)
        # Entorhinal path: what we actually see
        self.entorh_proj  = nn.Linear(d_sem, d_sem, bias=False)
        # Mismatch head: scalar novelty from (expected, actual)
        self.mismatch_mlp = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem), nn.GELU(),
            nn.Linear(d_sem, 1),
        )

        # --- Mood/temporal scoring ---
        # Projects NT-state + valence context into a weighting bias
        n_nt = 7  # DA, NE, 5HT, ACh, eCB, Glu, GABA
        self.mood_proj = nn.Linear(n_nt + 1, d_sem, bias=False)

        # --- GWS integration: cross-attention of slots onto recalls ---
        self.slot_query = nn.Linear(d_sem, d_sem, bias=False)
        self.recall_key = nn.Linear(d_sem, d_sem, bias=False)
        self.recall_val = nn.Linear(d_sem, d_sem, bias=False)
        self.out_proj   = nn.Linear(d_sem, d_sem)

        # --- Memory buffers (non-trainable, updated at each store()) ---
        self.register_buffer("keys",        torch.zeros(capacity, d_sem))
        self.register_buffer("values",      torch.zeros(capacity, d_sem))
        self.register_buffer("nt_states",   torch.zeros(capacity, n_nt))
        self.register_buffer("valences",    torch.zeros(capacity))
        self.register_buffer("timestamps",  torch.zeros(capacity))  # write-order index
        self.register_buffer("saliences",   torch.zeros(capacity))
        self.register_buffer("filled",      torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_ptr",   torch.zeros(1, dtype=torch.long))
        self._global_tick: int = 0

        # --- DNC temporal link matrix (sparse linked-list, XLA-friendly) ---
        # Replace the O(N²) dense NxN matrix with a constant-memory linked-list:
        #   _dnc_prev[i] = slot written immediately before slot i (-1 = unknown)
        #   _dnc_next[i] = slot written immediately after  slot i (-1 = unknown)
        #   _dnc_last    = most recently written slot index
        # Write cost: O(1). Temporal traversal: O(K) hops — fully unrolled so
        # XLA can trace the graph statically.  65K capacity → 512 KB vs 16 GB.
        _NEG = torch.full((capacity,), -1, dtype=torch.long)
        self.register_buffer("_dnc_prev", _NEG.clone())
        self.register_buffer("_dnc_next", _NEG.clone())
        self.register_buffer("_dnc_last", torch.full((1,), -1, dtype=torch.long))

    # ------------------------------------------------------------------
    # DG: sparse pattern separation
    # ------------------------------------------------------------------
    def _dg_sparse(self, x: torch.Tensor, mode: str = "encode") -> torch.Tensor:
        """Expand → sparsify (top-k winners) → project back.
        In encoding mode: higher sparsity (more orthogonal patterns).
        In retrieval mode: lower sparsity (richer, overlapping codes for completion).
        """
        h = self.dg_expand(x)          # (B, dg_dim)
        h = F.gelu(h)
        k = self.sparse_k if mode == "encode" else self.sparse_k * 2
        k = min(k, h.size(-1))
        topv, topi = h.topk(k, dim=-1)
        mask = torch.zeros_like(h).scatter_(-1, topi, 1.0)
        h = h * mask
        return self.dg_project(h)      # (B, d_sem)

    # ------------------------------------------------------------------
    # CA3: multi-dimensional recall
    # ------------------------------------------------------------------
    def _active_memories(self):
        """Return active (key, value, nt, valence, timestamp) tensors."""
        mask = self.filled
        return (self.keys[mask], self.values[mask],
                self.nt_states[mask], self.valences[mask],
                self.timestamps[mask], self.saliences[mask])

    def _active_indices(self) -> torch.Tensor:
        """Return integer indices of filled slots (for DNC link traversal)."""
        return self.filled.nonzero(as_tuple=True)[0]

    # ------------------------------------------------------------------
    # kNN — single matmul, XLA-traceable (~90% recall, Memorizing Transformers)
    # ------------------------------------------------------------------
    def _approx_knn(self, query: torch.Tensor, keys: torch.Tensor,
                    topk: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-matmul maximum-inner-product search (XLA-native).

        The chunked Python loop of the CUDA version is replaced by one large
        batched matmul that XLA compiles to a systolic-array matmul on TPU
        MXUs — the TPU's highest-throughput operation.

        For 65K keys at d=512 with bfloat16 this is a (B, 65K, d/2) matmul,
        fitting comfortably in TPU HBM at ~130 MB per core.

        Returns: (indices (B, topk), scores (B, topk))
        """
        q = F.normalize(query, dim=-1)   # (B, d)
        k = F.normalize(keys,  dim=-1)   # (N, d)
        sim = q @ k.T                    # (B, N) — one systolic matmul
        scores, indices = sim.topk(topk, dim=-1, largest=True, sorted=True)
        return indices, scores

    def _semantic_recall(self, dg_codes: torch.Tensor, topk: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """CA3 cosine-similarity recall using chunked approx kNN.
        Returns (recalls (B, topk, d), sim_scores (B, topk)).
        """
        keys, vals, *_ = self._active_memories()
        N = keys.size(0)
        B = dg_codes.size(0)
        device = dg_codes.device
        if N == 0:
            return (torch.zeros(B, topk, self.d_sem, device=device),
                    torch.zeros(B, topk, device=device))

        q = F.normalize(self.ca3_query(dg_codes), dim=-1)   # (B, d)
        k = self.ca3_key(keys)                               # (N, d)

        topi, topv = self._approx_knn(q, k, min(topk, N))   # (B, k_act)
        k_act = topi.size(1)

        v = self.ca3_value(vals)                             # (N, d)
        recalled = v[topi]                                   # (B, k_act, d)

        if k_act < topk:
            pad_v = torch.zeros(B, topk - k_act, self.d_sem, device=device)
            pad_s = torch.zeros(B, topk - k_act, device=device)
            recalled = torch.cat([recalled, pad_v], dim=1)
            topv     = torch.cat([topv,     pad_s], dim=1)

        return recalled, topv

    def _temporal_recall(self, dg_codes: torch.Tensor, topk: int
                         ) -> torch.Tensor:
        """Recency-weighted recall. More recent memories get an additive boost."""
        keys, vals, _, _, timestamps, _ = self._active_memories()
        N = keys.size(0)
        if N == 0:
            B = dg_codes.size(0)
            return torch.zeros(B, topk, self.d_sem, device=dg_codes.device)

        q = F.normalize(self.ca3_query(dg_codes), dim=-1)  # (B, d)
        k = F.normalize(self.ca3_key(keys), dim=-1)         # (N, d)
        sim = q @ k.T                                        # (B, N)

        # Recency weight: sigmoid ramp over write order (newer = higher)
        max_t = float(timestamps.max().item()) + 1e-8
        recency = timestamps / max_t              # (N,) in [0,1]
        sim = sim + 0.4 * recency.unsqueeze(0)    # bias toward recent

        k_act = min(topk, N)
        _, topi = sim.topk(k_act, dim=-1)
        v = self.ca3_value(vals)
        recalled = v[topi]                        # (B, k_act, d)

        B = dg_codes.size(0)
        if k_act < topk:
            pad = torch.zeros(B, topk - k_act, self.d_sem, device=dg_codes.device)
            recalled = torch.cat([recalled, pad], dim=1)
        return recalled

    def _mood_recall(self, dg_codes: torch.Tensor,
                     nt_vec: torch.Tensor,   # (B, n_nt)
                     valence: torch.Tensor,  # (B,)
                     topk: int) -> torch.Tensor:
        """Emotion/mood-congruent recall. Retrieves memories with similar NT
        state and valence — the neural basis of mood-congruent memory bias."""
        keys, vals, nt_stored, val_stored, _, _ = self._active_memories()
        N = keys.size(0)
        B = dg_codes.size(0)
        if N == 0:
            return torch.zeros(B, topk, self.d_sem, device=dg_codes.device)

        # Semantic similarity baseline
        q = F.normalize(self.ca3_query(dg_codes), dim=-1)  # (B, d)
        k = F.normalize(self.ca3_key(keys), dim=-1)         # (N, d)
        sem_sim = q @ k.T                                    # (B, N)

        # NT-state cosine similarity
        nt_q = F.normalize(nt_vec, dim=-1)              # (B, n_nt)
        nt_k = F.normalize(nt_stored, dim=-1)           # (N, n_nt)
        nt_sim = nt_q @ nt_k.T                          # (B, N)

        # Valence similarity: 1 - |val_q - val_stored|/2
        val_sim = 1.0 - (valence.unsqueeze(1) - val_stored.unsqueeze(0)).abs() / 2.0  # (B,N)

        # Mood-weighted score: content + NT mood + valence
        mood_score = 0.4 * sem_sim + 0.4 * nt_sim + 0.2 * val_sim

        k_act = min(topk, N)
        _, topi = mood_score.topk(k_act, dim=-1)
        v = self.ca3_value(vals)
        recalled = v[topi]                   # (B, k_act, d)

        if k_act < topk:
            pad = torch.zeros(B, topk - k_act, self.d_sem, device=dg_codes.device)
            recalled = torch.cat([recalled, pad], dim=1)
        return recalled

    def _associative_chain(self, best_recalls: torch.Tensor,
                            n_hops: int = 2, topk: int = 2) -> torch.Tensor:
        """Multi-hop associative chaining: use best recall as new query."""
        keys, vals, *_ = self._active_memories()
        if keys.size(0) == 0:
            return best_recalls
        chain = best_recalls
        for _ in range(n_hops):
            q_hop = F.normalize(self.ca3_query(chain.mean(1)), dim=-1)  # (B,d)
            k = F.normalize(self.ca3_key(keys), dim=-1)
            sim = q_hop @ k.T
            k_act = min(topk, keys.size(0))
            _, topi = sim.topk(k_act, dim=-1)
            v = self.ca3_value(vals)
            hop = v[topi]   # (B, k_act, d)
            chain = torch.cat([chain, hop], dim=1)
        return chain[:, :self.topk + n_hops * topk]   # trim to sensible size

    # ------------------------------------------------------------------
    # DNC temporal recall — O(K) hop traversal via sparse linked-list
    # ------------------------------------------------------------------
    def _dnc_temporal_recall(self, query: torch.Tensor, topk: int,
                              direction: str = "forward",
                              n_hops: int = 8) -> torch.Tensor:
        """Traverse write-order chains in O(K) hops via prev/next pointers.

        Each hop follows one edge in the linked-list:
          forward  → _dnc_next[slot]: what was written AFTER this slot
          backward → _dnc_prev[slot]: what was written BEFORE this slot

        The traversal is fully unrolled (static n_hops) so XLA can trace
        it as a fixed-depth computation graph without Python control flow.

        Returns: (B, topk, d_sem)
        """
        active_idx = self._active_indices()
        N = active_idx.size(0)
        B = query.shape[0]
        device = query.device

        if N == 0:
            return torch.zeros(B, topk, self.d_sem, device=device)

        # Initial read weights: cosine sim → best matching active slot
        q  = F.normalize(self.ca3_query(query), dim=-1)         # (B, d)
        k  = F.normalize(self.ca3_key(self.keys[active_idx]), dim=-1)  # (N, d)
        sim = q @ k.T                                           # (B, N)
        best_local = sim.argmax(dim=-1)                         # (B,) — local idx
        best_global = active_idx[best_local]                    # (B,) — global slot

        # Collect K hops; accumulate visited global indices
        visited = [best_global]                                 # list of (B,) tensors
        ptr = best_global                                       # (B,) current slot
        link_buf = self._dnc_next if direction == "forward" else self._dnc_prev

        for _ in range(n_hops - 1):
            # Clamp invalid (-1) to 0 before indexing; mask out results later
            safe_ptr = ptr.clamp(min=0)
            next_ptr = link_buf[safe_ptr]                       # (B,) — may be -1
            # Where the chain has ended, stay at current slot (masked below)
            ptr = torch.where(next_ptr >= 0, next_ptr, ptr)
            visited.append(ptr)

        # Stack visited slots: (B, n_hops) → retrieve values
        hop_idx = torch.stack(visited, dim=1)                   # (B, n_hops)
        k_take  = min(topk, n_hops)
        hop_idx = hop_idx[:, :k_take]
        v       = self.ca3_value(self.values[hop_idx.clamp(min=0)])  # (B, k_take, d)

        if k_take < topk:
            pad = torch.zeros(B, topk - k_take, self.d_sem, device=device)
            v   = torch.cat([v, pad], dim=1)
        return v

    # ------------------------------------------------------------------
    # CA1: mismatch / novelty detection
    # ------------------------------------------------------------------
    def _ca1_novelty(self, query: torch.Tensor,
                     recalled: torch.Tensor) -> torch.Tensor:
        """Compares entorhinal input (what we see) with CA3 reconstruction
        (what we expected). Mismatch = novelty signal (B,)."""
        expected = self.ca3_to_ca1(recalled.mean(1))   # (B, d_sem)
        actual   = self.entorh_proj(query)              # (B, d_sem)
        combined = torch.cat([expected, actual], dim=-1)
        mismatch = self.mismatch_mlp(combined).squeeze(-1)  # (B,)
        return torch.sigmoid(mismatch)

    # ------------------------------------------------------------------
    # GWS integration: enrich slots with recalled memories
    # ------------------------------------------------------------------
    def _integrate_into_gws(self, gws_slots: torch.Tensor,
                             recalls: torch.Tensor,
                             ach_gate: float) -> torch.Tensor:
        """Cross-attention of GWS slots onto all retrieved memories.
        ACh-gated: high ACh → stronger memory influence (encoding-boosted retrieval).
        """
        # q = slots, k/v = recalls
        Q = self.slot_query(gws_slots)                # (B, S, d)
        K = self.recall_key(recalls)                  # (B, R, d)
        V = self.recall_val(recalls)                  # (B, R, d)

        scale = math.sqrt(self.d_sem)
        attn = (Q @ K.transpose(-2, -1)) / scale     # (B, S, R)
        attn = F.softmax(attn, dim=-1)
        enrichment = attn @ V                         # (B, S, d)
        enrichment = self.out_proj(enrichment)

        # ACh gates encoding strength: high ACh = more memory influence
        gate = 0.2 + 0.6 * ach_gate                  # [0.2, 0.8]
        return gws_slots + gate * enrichment

    # ------------------------------------------------------------------
    # Primary interface: enrich_gws()
    # ------------------------------------------------------------------
    def enrich_gws(self, gws_slots: torch.Tensor,
                   floating_thought: torch.Tensor,
                   nt_levels: dict,
                   valence: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enrich Global Workspace slots with multi-dimensional hippocampal recall.

        Args:
            gws_slots:       (B, S, d_sem) broadcast slots
            floating_thought:(B, d_sem)
            nt_levels:       dict of NT scalars {name: float}
            valence:         (B,) current emotional valence (optional)

        Returns:
            enriched_slots   (B, S, d_sem) — GWS injected with recalled memories
            novelty          (B,) — CA1 mismatch signal
            all_recalls      (B, R, d_sem) — all retrieved memories (for PFC)
        """
        B, S, D = gws_slots.shape
        device = gws_slots.device

        ach = nt_levels.get("ACh", 0.5)
        ht  = nt_levels.get("5HT", 0.5)
        ne  = nt_levels.get("NE", 0.5)

        # Build query from GWS summary + floating thought
        query = gws_slots.mean(1) * 0.5 + floating_thought * 0.5   # (B, d)

        # Theta-gamma coupling: ACh > 0.6 → encoding mode
        dg_mode = "encode" if ach > 0.6 else "retrieve"
        dg_codes = self._dg_sparse(query, mode=dg_mode)

        topk = self.topk

        if not self.filled.any():
            zeros = torch.zeros(B, topk, self.d_sem, device=device)
            ones  = torch.ones(B, device=device)
            return gws_slots, ones, zeros

        # Multi-dimensional recall
        sem_recalls, sim_scores = self._semantic_recall(dg_codes, topk)
        temp_recalls            = self._temporal_recall(dg_codes, topk)

        # Mood recall needs NT vector
        n_nt = self.nt_states.size(-1)
        nt_vec_list = [nt_levels.get(n, 0.5)
                       for n in ["DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA"]]
        nt_t = torch.tensor(nt_vec_list, device=device, dtype=gws_slots.dtype
                            ).unsqueeze(0).expand(B, -1)
        val_t = valence if valence is not None else torch.zeros(B, device=device)
        mood_recalls = self._mood_recall(dg_codes, nt_t, val_t, topk)

        # Associative chaining from best semantic hits
        chain_recalls = self._associative_chain(sem_recalls, n_hops=2, topk=2)

        # DNC temporal forward traversal (what came after current query in write order)
        dnc_recalls = self._dnc_temporal_recall(query, topk, direction="forward")

        # Merge all recall streams (semantic, temporal, mood, associative, DNC)
        all_recalls = torch.cat([sem_recalls, temp_recalls,
                                 mood_recalls, chain_recalls,
                                 dnc_recalls], dim=1)  # (B, R, d)

        # CA1: novelty from semantic recalls
        novelty = self._ca1_novelty(query, sem_recalls)

        # Integrate into GWS
        enriched_slots = self._integrate_into_gws(gws_slots, all_recalls, ach)

        return enriched_slots, novelty, all_recalls

    # ------------------------------------------------------------------
    # Backward-compat interface used in brain.py
    # ------------------------------------------------------------------
    def recall(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple recall interface (B, d) → (recalls (B, topk, d), novelty (B,))."""
        dg_codes = self._dg_sparse(query, mode="retrieve")
        recalls, sim_scores = self._semantic_recall(dg_codes, self.topk)
        novelty = 1.0 - sim_scores[:, 0].clamp(-1.0, 1.0) * 0.5 + 0.5
        return recalls, novelty.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------
    @torch.no_grad()
    def store(self, query: torch.Tensor, value: torch.Tensor,
              nt_state: torch.Tensor | None = None,
              valence: float = 0.0, salience: float = 0.0,
              narrative_meta: dict | None = None) -> None:
        """Store (key, value) pairs. query/value: (B, d_sem).
        Also updates DNC temporal link matrix and precedence vector.

        ``narrative_meta`` (optional, **post-awakening only**): a JSON-ish
        dict carrying the structured narrative episode (content, entity_id,
        subject, predicate, object, predicted_action, observed_response).
        When present, the episode is appended to ``self.narrative_log``
        (capped at 4096) so the consolidation/sleep cycle can replay it
        as a distilled chunk rather than only a raw embedding.
        """
        key = self._dg_sparse(query, mode="encode").detach()
        B = key.size(0)
        n_nt = self.nt_states.size(-1)
        for b in range(B):
            idx = int(self.write_ptr.item()) % self.capacity
            self.keys[idx]       = key[b]
            self.values[idx]     = value[b].detach()
            self.nt_states[idx]  = (nt_state[b].detach()[:n_nt]
                                    if nt_state is not None
                                    else torch.zeros(n_nt, device=key.device))
            self.valences[idx]   = valence
            self.saliences[idx]  = salience
            self.timestamps[idx] = self._global_tick
            self.filled[idx]     = True

            # Sparse DNC: O(1) linked-list pointer update
            # _dnc_last → prev of current slot; current slot → _dnc_last
            last = int(self._dnc_last.item())
            if last >= 0:
                self._dnc_prev[idx] = last
                self._dnc_next[last] = idx
            self._dnc_last[0] = idx

            self.write_ptr      += 1
            self._global_tick   += 1

        # ── Narrative-episode log (post-awakening) ────────────────────────
        if narrative_meta is not None:
            if not hasattr(self, "narrative_log"):
                self.narrative_log: list[dict] = []
            self.narrative_log.append({
                **narrative_meta,
                "tick":     int(self._global_tick.item()
                                if hasattr(self._global_tick, "item")
                                else self._global_tick),
                "valence":  float(valence),
                "salience": float(salience),
            })
            # Keep bounded (oldest dropped first)
            if len(self.narrative_log) > 4096:
                self.narrative_log = self.narrative_log[-4096:]

    def recent_narrative_episodes(self, n: int = 256) -> list[dict]:
        """Return the last `n` narrative-meta dicts (empty if none recorded)."""
        if not hasattr(self, "narrative_log"):
            return []
        return list(self.narrative_log[-n:])

    # ------------------------------------------------------------------
    # forward() and _disabled_output() for BrainModule protocol
    # ------------------------------------------------------------------
    def forward(self, gws_slots, floating_thought, nt_levels,
                valence=None):
        return self.enrich_gws(gws_slots, floating_thought, nt_levels, valence)

    def _disabled_output(self, gws_slots, floating_thought, *args, **kwargs):
        B = gws_slots.size(0)
        ones = torch.ones(B, device=gws_slots.device)
        zeros = torch.zeros(B, self.topk, self.d_sem, device=gws_slots.device)
        return gws_slots, ones, zeros

    def to_device(self, device):
        return self.to(device)
