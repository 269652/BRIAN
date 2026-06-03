# -*- coding: utf-8 -*-
"""C5 — Bowtie-lattice specialisation probe.

Architecture today has *one* GWS of dim 256 with WTA temperature 0.1.
The PR-recommended surgery is to split it into K parallel narrow
workspaces (d=64×4) with lateral GABA inhibition between them — a
mixture-of-bowties competitive lattice that is the discrete analog of
the lateral-inhibition stabilising mechanism in real skyrmion lattices.

Before doing that surgery we want to *measure* whether the existing
single workspace is latently specialising — i.e. whether contiguous
slices of the d=256 activation already partition by token class.

The probe slices the GWS activation into K contiguous chunks of size
d/K, applies the input class label (`dialogue` | `prose` | `code` |
`other`), and computes the empirical conditional-probability lift:

    S = (1/K) Σ_k max_c P(c | active_k) / P(c)        ∈ [1, C]

where:
    active_k(x) = 1 if k = argmax_k ‖x_slice_k‖        (winning slice)

S = 1.0 means slices are uninformative about class; S ≈ C means each
slice cleanly maps to one class.

Stateless across runs; runs over the last `history` steps. Telemetry:
`lattice_spec`, `lattice_active_k`, `lattice_entropy`.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, Optional

import torch


class BowtieLatticeProbe:
    """Lateral-specialisation index for a (B, T, D) GWS-style activation.

    Parameters
    ----------
    dim : int
        Total feature dimensionality of the workspace activation.
    K : int
        Number of synthetic parallel workspaces to probe (must divide
        `dim`). Default 4.
    n_classes : int
        Cardinality of the class label space. Default 4
        (dialogue / prose / code / other).
    history : int
        Number of recent (slice-winner, class) pairs to keep for the
        specialisation estimate. Default 1024.
    """

    def __init__(self,
                 dim: int,
                 K: int = 4,
                 n_classes: int = 4,
                 history: int = 1024):
        if dim <= 0 or K <= 0 or n_classes <= 0:
            raise ValueError("dim, K, n_classes must all be positive")
        if dim % K != 0:
            raise ValueError(f"K={K} must divide dim={dim}")
        self.dim = int(dim)
        self.K = int(K)
        self.n_classes = int(n_classes)
        self._slice_size = dim // K
        # (class, slice) running counts.
        self._joint = torch.zeros(n_classes, K, dtype=torch.float64)
        self._class_count = torch.zeros(n_classes, dtype=torch.float64)
        self._slice_count = torch.zeros(K, dtype=torch.float64)
        self._total = 0.0
        # Recent-window mirror so we can age out old counts.
        self._history: deque = deque(maxlen=int(history))

    # ── Slice-winner ────────────────────────────────────────────────

    def slice_winner(self, h: torch.Tensor) -> torch.Tensor:
        """Return per-(batch,time) index of the winning slice.

        h: (B, T, D)  →  (B, T) int
        """
        if h.shape[-1] != self.dim:
            raise ValueError(
                f"BowtieLatticeProbe(dim={self.dim}) got D={h.shape[-1]}"
            )
        x = h.detach().float()
        # Reshape to (..., K, slice_size) and take per-slice L2 norm.
        new_shape = x.shape[:-1] + (self.K, self._slice_size)
        chunks = x.reshape(new_shape)
        norms = chunks.pow(2).sum(dim=-1)           # (..., K)
        return norms.argmax(dim=-1)                 # (...) int

    # ── Step ────────────────────────────────────────────────────────

    def step(self,
             h: Optional[torch.Tensor],
             class_label: Optional[int]) -> Dict[str, float]:
        """Update counts with one batch and return current stats.

        `class_label` is a single integer in [0, n_classes) — the
        batch-level class indicator. (For mixed batches, call repeatedly
        per-example or pre-classify; we keep the API simple here.)
        """
        if h is None or class_label is None:
            return self.stats()
        c = int(class_label)
        if not (0 <= c < self.n_classes):
            return self.stats()

        winners = self.slice_winner(h).reshape(-1).tolist()
        for k in winners:
            # If we're about to push out an item (deque is full), decrement
            # the oldest's contribution first — O(1) per step regardless
            # of history size.
            if (self._history.maxlen is not None
                    and len(self._history) == self._history.maxlen):
                old_c, old_k = self._history[0]
                self._joint[old_c, old_k] -= 1.0
                self._class_count[old_c] -= 1.0
                self._slice_count[old_k] -= 1.0
                self._total -= 1.0
            self._joint[c, k] += 1.0
            self._class_count[c] += 1.0
            self._slice_count[k] += 1.0
            self._total += 1.0
            self._history.append((c, k))

        return self.stats()

    # ── Specialisation index ────────────────────────────────────────

    def stats(self) -> Dict[str, float]:
        if self._total <= 0.0:
            return {"lattice_spec": 1.0, "lattice_active_k": 0,
                    "lattice_entropy": 0.0}

        # P(c) and P(c | k) = joint[c,k] / slice_count[k]
        p_c = self._class_count / self._total                 # (C,)
        # Lift per slice: max_c P(c|k) / P(c)
        slice_count_safe = self._slice_count.clamp_min(1.0)
        cond = self._joint / slice_count_safe.unsqueeze(0)    # (C, K)
        p_c_safe = p_c.clamp_min(1.0 / max(self._total, 1.0))
        lift = cond / p_c_safe.unsqueeze(1)                   # (C, K)
        per_slice_lift = lift.max(dim=0).values               # (K,)
        # Mask out unused slices (lift undefined if slice never won).
        active = self._slice_count > 0
        if active.any():
            spec = float(per_slice_lift[active].mean().item())
        else:
            spec = 1.0
        # Entropy of slice usage (high = uniform, low = collapsed).
        p_k = self._slice_count / self._total
        p_k_nz = p_k[p_k > 0]
        ent = float(-(p_k_nz * p_k_nz.log()).sum().item()) if p_k_nz.numel() else 0.0
        # Normalise to [0, 1] using log K as max entropy.
        ent_norm = ent / math.log(max(2, self.K))
        return {
            "lattice_spec":    spec,
            "lattice_active_k": int(active.sum().item()),
            "lattice_entropy": float(max(0.0, min(1.0, ent_norm))),
        }
