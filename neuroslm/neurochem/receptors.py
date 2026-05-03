"""Receptors: protein-like key/lock matching via multidimensional latent shapes.

Each NT molecule is a learnable latent shape vector (the "key").
Each receptor is a learnable binding-pocket shape vector (the "lock").
Binding affinity = cosine_similarity(NT_shape, receptor_shape) — exactly like
real protein geometry, where 3D complementarity determines binding strength.

This replaces hardcoded string matching (``if nt == 'NE'``) with a fully
differentiable, evolvable shape-matching system:

  - NTs can evolve new shapes → novel binding profiles
  - Receptors can evolve to accept multiple NTs (partial agonists)
  - Cross-reactivity emerges naturally from shape similarity
  - Antagonists have high affinity but inhibitory sign
  - The whole system is end-to-end trainable via backprop

Receptor types modelled (initialised to match their canonical NT):
  D1, D2     — DA receptors (excitatory / inhibitory)
  alpha1/2   — NE receptors (arousal)
  5HT1A/2A   — serotonin (mood / cortical gain)
  M1, nACh   — cholinergic (signal-to-noise, plasticity)
  CB1        — endocannabinoid (retrograde suppression)
  NMDA, AMPA — glutamate (plasticity gate, fast excitation)
  GABAA      — inhibitory
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transmitters import NT_INDEX, NT_NAMES, N_NT

# ── Latent shape dimensionality ──────────────────────────────────────
SHAPE_DIM = 16    # dimensionality of NT/receptor latent "protein" shape


# ════════════════════════════════════════════════════════════════════════
# NT Shape Registry — shared latent shape vectors for all 7 NTs
# ════════════════════════════════════════════════════════════════════════

class NTShapeRegistry(nn.Module):
    """Learnable latent "protein shapes" for every neurotransmitter.

    Each NT is a SHAPE_DIM vector.  Initialised with orthogonal seeds so
    each NT starts maximally distinct (like different molecular structures).
    Training can nudge them to create cross-reactivity, partial agonists, etc.
    """

    def __init__(self, shape_dim: int = SHAPE_DIM):
        super().__init__()
        self.shape_dim = shape_dim
        # Initialise with orthogonal vectors → maximal initial discrimination
        init = torch.empty(N_NT, shape_dim)
        nn.init.orthogonal_(init)
        self.shapes = nn.Parameter(init)            # (N_NT, SHAPE_DIM)

    def get_shape(self, nt_name: str) -> torch.Tensor:
        """Return normalised shape for a named NT.  (SHAPE_DIM,)"""
        idx = NT_INDEX[nt_name]
        return F.normalize(self.shapes[idx], dim=-1)

    def all_shapes(self) -> torch.Tensor:
        """Return normalised shapes for all NTs.  (N_NT, SHAPE_DIM)"""
        return F.normalize(self.shapes, dim=-1)

    def affinity_matrix(self) -> torch.Tensor:
        """(N_NT, N_NT) cosine similarity between all NT pairs."""
        normed = self.all_shapes()
        return normed @ normed.T


# ════════════════════════════════════════════════════════════════════════
# Legacy Receptor (string-matched) — kept for backward compat
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Receptor:
    nt: str           # which NT binds (legacy: by name)
    sign: float       # +1 excitatory, -1 inhibitory
    weight: float = 0.5


# ════════════════════════════════════════════════════════════════════════
# Latent Receptor — protein-shaped binding pocket
# ════════════════════════════════════════════════════════════════════════

class LatentReceptor(nn.Module):
    """A single receptor with a learnable binding-pocket shape.

    Binding affinity with any NT =
        sigmoid(temperature * cos_sim(pocket_shape, nt_shape))

    Initialised so the pocket shape matches the canonical NT's shape,
    giving high initial affinity for the "correct" NT.  Training can drift
    the shape to create partial agonists or antagonists.
    """

    def __init__(self, canonical_nt: str, sign: float, weight: float,
                 receptor_type: str = "",
                 shape_dim: int = SHAPE_DIM):
        super().__init__()
        self.canonical_nt = canonical_nt
        self.receptor_type = receptor_type
        self.sign = sign
        self.shape_dim = shape_dim

        # Learnable binding pocket shape — init to match canonical NT
        # (will be overwritten by _init_from_registry once registry exists)
        self.pocket = nn.Parameter(torch.randn(shape_dim) * 0.1)
        self.sensitivity = nn.Parameter(torch.tensor(weight))
        # Temperature controls sharpness of binding selectivity
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def init_from_registry(self, registry: NTShapeRegistry):
        """Align pocket shape to canonical NT + small noise (so it starts
        with high affinity but can evolve away)."""
        with torch.no_grad():
            base = registry.get_shape(self.canonical_nt).detach()
            noise = torch.randn_like(base) * 0.05
            self.pocket.copy_(base + noise)

    def affinity(self, nt_shapes: torch.Tensor) -> torch.Tensor:
        """Compute binding affinity with all NTs.

        Args:
            nt_shapes: (N_NT, SHAPE_DIM) normalised NT shapes

        Returns:
            (N_NT,) affinity scores in [0, 1]
        """
        pocket_norm = F.normalize(self.pocket, dim=-1)       # (SHAPE_DIM,)
        cos_sim = (pocket_norm @ nt_shapes.T)                # (N_NT,)
        return torch.sigmoid(self.temperature * cos_sim)     # (N_NT,)

    def binding_response(self, nt_shapes: torch.Tensor,
                         nt_levels: torch.Tensor) -> torch.Tensor:
        """Compute this receptor's scalar output given NT state.

        Args:
            nt_shapes: (N_NT, SHAPE_DIM)
            nt_levels: (B, N_NT)

        Returns:
            (B,) signed weighted response
        """
        aff = self.affinity(nt_shapes)                     # (N_NT,)
        # Effective level = sum(affinity_i * level_i) — partial agonist binding
        effective = (nt_levels * aff.unsqueeze(0)).sum(-1)  # (B,)
        return self.sign * self.sensitivity * effective     # (B,)


# ════════════════════════════════════════════════════════════════════════
# ReceptorBank — the main interface (now uses latent shapes)
# ════════════════════════════════════════════════════════════════════════

class ReceptorBank(nn.Module):
    """Bank of latent-shape receptors that yields a scalar gain (B,) per call.

    Accepts EITHER legacy Receptor specs OR LatentReceptor instances.
    When given legacy specs, auto-creates LatentReceptors from them.

    The bank needs an NTShapeRegistry to function at full capacity.
    Call `bind_registry(registry)` after construction (Brain does this).
    Without a registry it falls back to index-based lookup (legacy mode).
    """

    def __init__(self, receptors: Sequence[Receptor | LatentReceptor]):
        super().__init__()
        self._registry: Optional[NTShapeRegistry] = None
        self._legacy_mode = False

        # Convert legacy Receptor dataclasses → LatentReceptor modules
        latent_list = []
        for r in receptors:
            if isinstance(r, LatentReceptor):
                latent_list.append(r)
            elif isinstance(r, Receptor):
                latent_list.append(LatentReceptor(
                    canonical_nt=r.nt, sign=r.sign, weight=r.weight,
                    receptor_type="",
                ))
            else:
                raise TypeError(f"Expected Receptor or LatentReceptor, got {type(r)}")

        self.receptors_list = nn.ModuleList(latent_list)

        # Legacy fallback tensors (used when no registry is bound)
        signs = [r.sign for r in receptors]
        weights = [getattr(r, 'weight', 0.5) if isinstance(r, Receptor)
                   else float(r.sensitivity.data) for r in receptors]
        self._legacy_w = nn.Parameter(
            torch.tensor([w * s for w, s in zip(weights, signs)]))
        nt_names = [r.nt if isinstance(r, Receptor) else r.canonical_nt
                    for r in receptors]
        self.register_buffer(
            '_legacy_idx',
            torch.tensor([NT_INDEX[n] for n in nt_names], dtype=torch.long))

    def bind_registry(self, registry: NTShapeRegistry):
        """Bind the shared NT shape registry and init all receptor pockets."""
        self._registry = registry
        self._legacy_mode = False
        for r in self.receptors_list:
            r.init_from_registry(registry)

    def gain(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """nt_levels: (B, N_NT) → gain (B,) in [0.1, 2.0].

        If registry is bound: uses latent protein shape matching.
        Otherwise: falls back to legacy index-based lookup.
        """
        if self._registry is not None:
            return self._gain_latent(nt_levels)
        return self._gain_legacy(nt_levels)

    def _gain_latent(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """Shape-matched gain: each receptor computes affinity-weighted response."""
        nt_shapes = self._registry.all_shapes()          # (N_NT, SHAPE_DIM)
        total = torch.zeros(nt_levels.size(0), device=nt_levels.device)
        for receptor in self.receptors_list:
            total = total + receptor.binding_response(nt_shapes, nt_levels)
        return 0.1 + 1.9 * torch.sigmoid(total)

    def _gain_legacy(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """Legacy: index-based gain (backward compat)."""
        idx = self._legacy_idx.to(nt_levels.device)
        levels = nt_levels.index_select(1, idx)
        contrib = (levels * self._legacy_w).sum(-1)
        return 0.1 + 1.9 * torch.sigmoid(contrib)

    def modulate(self, x: torch.Tensor, nt_levels: torch.Tensor) -> torch.Tensor:
        """Apply gain to last-dim of x."""
        g = self.gain(nt_levels)
        shape = [g.size(0)] + [1] * (x.dim() - 1)
        return x * g.view(shape)

    def affinity_report(self) -> dict[str, dict[str, float]]:
        """Return per-receptor binding affinity to every NT (for diagnostics)."""
        if self._registry is None:
            return {}
        nt_shapes = self._registry.all_shapes()
        report = {}
        for r in self.receptors_list:
            aff = r.affinity(nt_shapes).detach().cpu()
            report[r.receptor_type or r.canonical_nt] = {
                NT_NAMES[i]: float(aff[i]) for i in range(N_NT)
            }
        return report
