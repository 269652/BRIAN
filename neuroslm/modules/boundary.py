"""BoundaryDetector: Fiedler Vector spectral bisection for MIP screening.

Implements Cheeger/Fiedler spectral graph theory to identify the Minimum
Information Partition (MIP) across the module interaction graph.  The Fiedler
vector (second eigenvector of the normalized graph Laplacian) defines the
optimal bipartition that minimises the Cheeger cut — and therefore minimises
the information flow across the partition boundary.

Theory
------
  Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
  Fiedler value λ₁  = second-smallest eigenvalue of L  (algebraic connectivity)
  Fiedler vector f  = corresponding eigenvector        (bipartition indicator)

  Cheeger's inequality:  h(G)/2  ≤  λ₁  ≤  2·h(G)
  → Low λ₁ (< 0.3) signals near-disconnection — homeostatic BDNF boost needed
  → High λ₁ → well-integrated graph — normal trophic dynamics apply

Biological mapping
------------------
  Module interactions are estimated from pairwise cosine similarity of recent
  activation vectors.  An EMA of the similarity matrix converges over steps,
  giving a smooth estimate of the effective connectivity graph.

Usage
-----
  detector = BoundaryDetector(n_modules=24)
  fiedler_val, fiedler_vec = detector.observe({"pfc": pfc_out, "gws": gws_out, ...})
  part_a, part_b = detector.mip_bipartition(list(module_outputs.keys()))
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class BoundaryDetector(nn.Module):
    """Spectral MIP detector using the algebraic Fiedler vector.

    Maintains an exponential moving average of the inter-module cosine
    similarity matrix.  Every ``recompute_every`` observations the
    normalised Laplacian eigensystem is solved (``torch.linalg.eigh``
    on a small n×n matrix) to extract the Fiedler value and vector.

    Parameters
    ----------
    n_modules       : upper bound on the number of modules tracked
    ema_decay       : EMA smoothing coefficient (0 < α < 1)
    recompute_every : how many ``observe()`` calls between eigen-solves
    """

    def __init__(self, n_modules: int = 32,
                 ema_decay: float = 0.95,
                 recompute_every: int = 10):
        super().__init__()
        self.n_modules = n_modules
        self.ema_decay = ema_decay
        self.recompute_every = recompute_every

        # EMA of pairwise cosine similarity (symmetric, non-negative)
        self.register_buffer("corr_ema", torch.eye(n_modules) * 0.5)
        # Last computed Fiedler eigenvalue (algebraic connectivity λ₁)
        self.register_buffer("fiedler_val", torch.tensor(1.0))
        # Fiedler vector (bipartition indicator — sign determines side)
        self.register_buffer("fiedler_vec", torch.zeros(n_modules))
        # EMA-corrected module name list for the current solve
        self._last_names: List[str] = []
        self._step = 0

    @torch.no_grad()
    def observe(self, module_outputs: Dict[str, torch.Tensor]
                ) -> Tuple[float, torch.Tensor]:
        """Update the EMA similarity matrix from current module activations.

        Parameters
        ----------
        module_outputs : mapping from module name → activation tensor
                         (any shape; will be mean-pooled to a 1-D vector)

        Returns
        -------
        fiedler_val : float   — algebraic connectivity (∈ [0, 2])
        fiedler_vec : Tensor  — bipartition indicator (length = n observed)
        """
        names = sorted(module_outputs.keys())
        n = min(len(names), self.n_modules)
        if n < 2:
            return float(self.fiedler_val), self.fiedler_vec[:max(n, 1)]

        # Build normalised module mean vectors
        vecs: List[torch.Tensor] = []
        for name in names[:n]:
            v = module_outputs[name].detach().float()
            if v.dim() > 1:
                v = v.mean(0)
            v = F.normalize(v, dim=0)
            vecs.append(v)

        stacked = torch.stack(vecs, dim=0)        # (n, d)
        corr = (stacked @ stacked.T).abs()        # (n, n) absolute cosine sim

        # EMA update
        old = self.corr_ema[:n, :n]
        self.corr_ema[:n, :n] = self.ema_decay * old + (1.0 - self.ema_decay) * corr

        self._last_names = names[:n]
        self._step += 1
        if self._step % self.recompute_every == 0:
            self._compute_fiedler(n)

        return float(self.fiedler_val), self.fiedler_vec[:n]

    def _compute_fiedler(self, n: int) -> None:
        """Solve for Fiedler value/vector via torch.linalg.eigh (exact, O(n³))."""
        C = self.corr_ema[:n, :n].float().clamp(min=0.0)

        # Degree matrix and normalised Laplacian
        deg = C.sum(dim=-1).clamp(min=1e-6)       # (n,)
        d_inv_sqrt = deg.rsqrt()                   # D^{-1/2}
        L = torch.eye(n, device=C.device, dtype=torch.float32) - (
            d_inv_sqrt.unsqueeze(1) * C * d_inv_sqrt.unsqueeze(0))

        try:
            # eigh returns eigenvalues in ascending order
            eigvals, eigvecs = torch.linalg.eigh(L)
            # λ₀ ≈ 0 (trivial), λ₁ = Fiedler value
            lam1 = float(eigvals[1].clamp(0.0, 2.0).item())
            fvec = eigvecs[:, 1]                   # Fiedler vector
        except Exception:
            lam1 = 1.0
            fvec = torch.zeros(n, device=C.device)

        self.fiedler_val.fill_(lam1)
        self.fiedler_vec[:n] = fvec

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def spectral_gap(self) -> float:
        """Algebraic connectivity λ₁ ∈ [0, 2].  Low → nearly disconnected."""
        return float(self.fiedler_val.item())

    @property
    def is_nearly_disconnected(self) -> bool:
        """True when the Fiedler value is below the disconnection threshold."""
        return self.spectral_gap < 0.3

    def mip_bipartition(self, module_names: List[str]
                        ) -> Tuple[List[str], List[str]]:
        """Return the MIP bipartition implied by the Fiedler vector sign.

        Modules with positive Fiedler-vector entries form partition A;
        negative entries form partition B.  This is the spectral bisection
        that minimises the Cheeger cut (minimum-information partition).

        Returns
        -------
        (part_a, part_b) : two lists of module names
        """
        n = min(len(module_names), self.n_modules)
        v = self.fiedler_vec[:n].cpu()

        part_a = [module_names[i] for i in range(n) if float(v[i]) >= 0.0]
        part_b = [module_names[i] for i in range(n) if float(v[i]) <  0.0]

        # Ensure neither partition is empty (degenerate graph)
        if not part_a:
            part_a, part_b = [module_names[0]], list(module_names[1:n])
        if not part_b:
            part_b, part_a = [module_names[-1]], list(module_names[:n - 1])

        return part_a, part_b

    def stats(self) -> dict:
        return {
            "fiedler_val":          float(self.fiedler_val.item()),
            "is_nearly_disconnected": bool(self.is_nearly_disconnected),
            "n_tracked":            len(self._last_names),
        }
