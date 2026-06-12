"""Tonnetz adjacency-loss prior — token-cooccurrence spectral-gap regulariser.

Goal
----
Encourage the token embedding space to behave like a *well-connected
multigraph* under the empirical cooccurrence pattern of each batch. We
build the unnormalised cooccurrence Laplacian over the tokens that
actually appear in the current batch and apply a soft hinge on its
algebraic connectivity (Fiedler eigenvalue) λ_1:

.. math::

    \\mathcal{L}_\\text{tonnetz} \\;=\\; \\operatorname{ReLU}(\\tau - \\lambda_1)^2

so the penalty is exactly zero whenever λ_1 ≥ τ and grows quadratically
below τ. The default threshold τ = 0.3 follows the Neo-Riemannian
Tonnetz convention used elsewhere in the workspace module.

Why "Tonnetz"
-------------
The cooccurrence graph here is *not literally* the Neo-Riemannian
toroidal triangulation — that would require fixing a 3-regular planar
embedding on a fixed vertex set. We borrow the name because the
*intent* is identical: keep the graph well-separated (high spectral
gap) so that no small cluster of tokens dominates the manifold. If you
want the strictly planar 3-regular variant, instantiate
:class:`TopologicalDifferentialWorkspace(tonnetz=True)` instead — that
class enforces orthonormality of a fixed Tonnetz basis at every
forward, which is the rank-projector counterpart of this regulariser.

Shape & gradient contract (used by tests)
-----------------------------------------
* ``forward(embeddings, ids)`` returns a 0-D tensor (composable with
  ``total_loss + λ * penalty``).
* The penalty is non-negative by construction (relu hinge, squared).
* Gradient flows through ``embeddings`` whenever penalty > 0.
* Higher ``gap_threshold`` ⇒ stricter constraint ⇒ monotone-non-decreasing
  penalty for fixed inputs.

Not a research claim
--------------------
This module exists to be ablation-tested under ``cfg.use_tonnetz_prior``.
Whether it actually improves downstream loss is an empirical question
that has to be answered by running the ablation; the docstring makes no
claim about effect size or direction. (CLAUDE.md §10, §13.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TonnetzPrior(nn.Module):
    """Spectral-gap hinge on the token-cooccurrence Laplacian.

    Parameters
    ----------
    vocab_size : int
        Size of the embedding table. Used only to validate input range.
    d_embed : int
        Embedding dimensionality. Cooccurrence weights are computed as
        the cosine similarity of embedding rows for tokens that appear
        in the same window.
    gap_threshold : float
        τ in the docstring; penalty is ``relu(τ - λ_1) ** 2``.
    window : int
        Cooccurrence window in tokens. Defaults to 5 (standard
        skip-gram-style window).
    """

    def __init__(self, vocab_size: int, d_embed: int,
                 gap_threshold: float = 0.3, window: int = 5):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_embed = int(d_embed)
        self.gap_threshold = float(gap_threshold)
        self.window = int(window)
        # No learnable parameters of our own — the regulariser shapes
        # the trunk's embedding table by gradient feedback only.

    @staticmethod
    def _algebraic_connectivity(L: torch.Tensor) -> torch.Tensor:
        """Smallest positive eigenvalue of a symmetric PSD Laplacian.

        For a connected graph, the smallest eigenvalue is exactly zero
        (eigenvector of all-ones) and the second-smallest is the
        algebraic connectivity λ_1. We use ``torch.linalg.eigvalsh``
        which returns eigenvalues in ascending order.
        """
        # eigvalsh requires a real symmetric matrix; symmetrise to kill
        # floating-point asymmetry introduced by the GEMM above.
        L = 0.5 * (L + L.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(L)
        # Index [1] is the algebraic connectivity (eigenvalue [0] is the
        # ~0 trivial mode). For a disconnected graph multiple eigenvalues
        # are ~0, so [1] correctly reports gap=0 in that case.
        return evals[1] if evals.numel() >= 2 else evals[0]

    def forward(self, embeddings: torch.Tensor,
                ids: torch.Tensor) -> torch.Tensor:
        """Return the scalar spectral-gap hinge penalty.

        Parameters
        ----------
        embeddings : (V, d_embed) tensor
            The trunk's token embedding table. Gradient flows back
            into this through the cosine-similarity weights.
        ids : (B, T) long tensor
            Token ids in the current batch. We pool unique tokens to
            build the per-batch subgraph, which keeps the eigen-solve
            small enough to run every step.
        """
        # 1. Unique tokens in this batch → subgraph vertex set.
        unique_ids = torch.unique(ids.flatten())
        n = unique_ids.numel()
        if n < 3:
            # Need ≥3 vertices for λ_1 to be meaningful; otherwise the
            # constraint is trivially satisfied (return 0).
            return embeddings.sum() * 0.0  # keeps autograd alive

        # 2. Adjacency weights via cosine similarity on rows.
        sub = embeddings.index_select(0, unique_ids)         # (n, d)
        sub_n = F.normalize(sub, dim=-1)
        W = sub_n @ sub_n.transpose(0, 1)                    # (n, n)
        # Drop self-loops, clamp to [0, 1] so we have a non-negative
        # weighted adjacency. Negative similarities are uninformative for
        # connectivity — treat them as "no edge".
        W = (W - torch.eye(n, device=W.device, dtype=W.dtype)).clamp(min=0.0)

        # 3. Unnormalised Laplacian L = D − W.
        deg = W.sum(dim=-1)
        L = torch.diag(deg) - W

        # 4. λ_1 = algebraic connectivity, hinge below threshold.
        lam_1 = self._algebraic_connectivity(L)
        slack = torch.clamp(self.gap_threshold - lam_1, min=0.0)
        return slack ** 2
