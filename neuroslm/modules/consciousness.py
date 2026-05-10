"""Consciousness metrics: measurable indicators of integrated information,
neural oscillations, coherence, and phenomenal binding.

Inspired by:
  - Integrated Information Theory (IIT 3.0): Φ as minimum information partition
  - Global Workspace Theory: broadcast / ignition as consciousness correlate
  - Neural oscillations: gamma (binding), theta (memory), alpha (idling)
  - Coherence: phase synchronisation across modules
  - Metacognition: confidence calibration as self-awareness proxy

IIT Φ implementation (this file):
  Φ ≈ min over bipartitions (A, B) of MI(A ; B)
  where MI is estimated from the Gram matrix of module output vectors.
  High Φ means NO bipartition can disconnect the system without losing
  mutual information — the hallmark of integrated experience.
  For n_modules ≤ 8 we enumerate all 2^(n-1)−1 bipartitions.
  For larger n we use spectral bisection (Fiedler vector of graph Laplacian).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


@torch.no_grad()
def estimate_fiedler(
    module_outputs: dict,
    n_power_iter: int = 20,
    device: Optional[torch.device] = None,
) -> tuple[float, Optional[torch.Tensor]]:
    """Power-iteration estimate of the Fiedler value (spectral gap λ₁).

    Builds a pairwise cosine-similarity graph from module output vectors,
    constructs the normalised Laplacian, then uses two rounds of deflated
    power iteration to approximate the second-smallest eigenvalue (Fiedler
    value) and its eigenvector (Fiedler vector).

    Returns:
      (fiedler_value, fiedler_vector | None)
      fiedler_value ≈ λ₁ ∈ [0, 2]:
        - Near 0 → graph almost disconnected (weak integration, low Φ risk)
        - Large  → well-connected major complex (strong integration)
      fiedler_vector: (n,) — identifies the approximate minimum bisection cut.
    """
    keys = [k for k, v in module_outputs.items() if torch.is_tensor(v)]
    n = len(keys)
    if n < 3:
        return 0.0, None

    if device is None:
        first = next((v for v in module_outputs.values() if torch.is_tensor(v)), None)
        device = first.device if first is not None else torch.device('cpu')

    vecs = []
    for k in keys:
        v = module_outputs[k]
        v = (v.mean(1) if v.dim() > 1 else v).detach().float()
        v = v.mean(0).flatten()
        vecs.append(F.normalize(v, dim=0).to(device))

    d = min(v.size(0) for v in vecs)
    vecs = [v[:d] for v in vecs]

    try:
        M = torch.stack(vecs)               # (n, d) unit-normed rows
        W = (M @ M.T).clamp(0.0, 1.0)      # cosine similarity graph
        W.fill_diagonal_(0.0)

        deg = W.sum(-1).clamp(min=1e-8)
        D_inv_sqrt = deg.pow(-0.5)
        L = (torch.eye(n, device=device)
             - D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0))

        # Shifted matrix A = I - L puts the dominant eigenvector first.
        A = torch.eye(n, device=device) - L

        # Power iteration for v₀ (the all-ones eigenvector, λ=0 in L)
        v0 = torch.ones(n, device=device) / math.sqrt(n)
        for _ in range(n_power_iter):
            v0 = A @ v0
            nrm = v0.norm()
            if nrm < 1e-10:
                break
            v0 = v0 / nrm
        v0 = v0 / v0.norm().clamp(min=1e-10)

        # Deflated power iteration for v₁ (Fiedler vector)
        Av0 = A @ v0
        A_def = A - torch.outer(Av0, v0)   # removes top eigenpair

        v1 = torch.randn(n, device=device)
        v1 = v1 - (v1 @ v0) * v0
        v1 = v1 / v1.norm().clamp(min=1e-10)
        for _ in range(n_power_iter):
            v1 = A_def @ v1
            v1 = v1 - (v1 @ v0) * v0      # re-orthogonalise
            nrm = v1.norm()
            if nrm < 1e-10:
                break
            v1 = v1 / nrm

        # Rayleigh quotient on L gives λ₁
        lambda1 = float((v1 @ (L @ v1)).item())
        lambda1 = max(0.0, min(2.0, lambda1))
        return lambda1, v1

    except Exception:
        return 0.0, None


class ConsciousnessMetrics(nn.Module):
    """Tracks and computes consciousness-related metrics each tick."""

    def __init__(self, d_sem: int, n_modules: int = 8, history_len: int = 64):
        super().__init__()
        self.d_sem = d_sem
        self.n_modules = n_modules
        self.history_len = history_len

        # Oscillation tracking buffers (circular)
        self.register_buffer("_tick", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_gamma_history",     torch.zeros(history_len))
        self.register_buffer("_theta_history",     torch.zeros(history_len))
        self.register_buffer("_alpha_history",     torch.zeros(history_len))
        self.register_buffer("_phi_history",       torch.zeros(history_len))
        self.register_buffer("_coherence_history", torch.zeros(history_len))

        self.register_buffer("_module_activities", torch.zeros(n_modules, d_sem))
        self.ignition_threshold = 0.6

    @torch.no_grad()
    def update(self, module_outputs: dict[str, torch.Tensor],
               gws_slots: torch.Tensor,
               floating_thought: torch.Tensor,
               novelty: torch.Tensor,
               routing: torch.Tensor) -> dict:
        idx = int(self._tick.item()) % self.history_len
        self._tick += 1

        B = floating_thought.size(0)
        device = floating_thought.device

        # 1) Gamma: binding coherence across GWS slots
        if gws_slots.size(1) > 1:
            slot_norms = F.normalize(gws_slots, dim=-1)
            sim_matrix = torch.bmm(slot_norms, slot_norms.transpose(1, 2))
            N = sim_matrix.size(1)
            mask = ~torch.eye(N, device=device, dtype=torch.bool)
            gamma = sim_matrix[:, mask].view(B, -1).mean()
        else:
            gamma = torch.tensor(0.0, device=device)
        gamma = gamma.clamp(0.0, 1.0)
        self._gamma_history[idx] = float(gamma)

        # 2) Theta: memory retrieval activity
        theta = novelty.mean().clamp(0.0, 1.0)
        self._theta_history[idx] = float(theta)

        # 3) Alpha: idling / suppression (high entropy routing → high alpha)
        routing_entropy = -(routing * (routing + 1e-8).log()).sum(-1).mean()
        max_entropy = math.log(routing.size(-1))
        alpha = (routing_entropy / max_entropy).clamp(0.0, 1.0)
        self._alpha_history[idx] = float(alpha)

        # 4) Φ: integrated information (MIP lower bound)
        phi = self._compute_phi_mip(module_outputs, device)
        self._phi_history[idx] = phi

        # 5) Coherence
        coherence = self._compute_coherence(module_outputs, gws_slots)
        self._coherence_history[idx] = coherence

        # 6) Ignition
        active_modules = sum(
            1 for v in module_outputs.values()
            if v.norm(dim=-1).mean() > self.ignition_threshold
        )
        ignition = active_modules / max(len(module_outputs), 1)

        # 7) Metacognition
        thought_magnitude = floating_thought.norm(dim=-1).mean()
        metacognition = torch.sigmoid(thought_magnitude - 1.0)

        # 8) Phenomenal binding
        binding = float(gamma) * float(coherence)

        return {
            "gamma":         float(gamma),
            "theta":         float(theta),
            "alpha":         float(alpha),
            "phi":           phi,
            "coherence":     coherence,
            "ignition":      ignition,
            "metacognition": float(metacognition),
            "binding":       binding,
            "tick":          int(self._tick.item()),
        }

    # ------------------------------------------------------------------
    # IIT Φ: minimum information partition lower bound
    # ------------------------------------------------------------------
    def _compute_phi_mip(self, module_outputs: dict[str, torch.Tensor],
                         device) -> float:
        """Φ ≈ min over bipartitions (A,B) of MI(A ; B).

        MI estimated via Gaussian approximation:
            MI(A;B) = 0.5 × (log det Σ_A + log det Σ_B − log det Σ_AB)

        For n ≤ 8 modules: enumerate all 2^(n-1)−1 bipartitions.
        For n > 8: use spectral bisection (Fiedler vector).

        A high Φ means no partition can disconnect the modules cheaply —
        the system is irreducibly integrated (IIT postulate 5).
        """
        keys = list(module_outputs.keys())
        n = min(len(keys), 8)
        if n < 2:
            return 0.0

        # Collect mean vectors, trim to shared dimension ≤ 256
        vecs = []
        for k in keys[:n]:
            v = module_outputs[k]
            v = v.mean(0) if v.dim() > 1 else v
            vecs.append(v.detach().float().flatten())
        d = min(min(v.size(0) for v in vecs), 256)
        vecs = [v[:d] for v in vecs]

        try:
            M = torch.stack(vecs)          # (n, d)
            M = M - M.mean(0, keepdim=True)

            # Gram matrix = cross-covariance structure (n, n)
            # Cov[i,j] = <vec_i, vec_j> / (d-1)
            cov = (M @ M.T) / max(d - 1, 1)
            eps = 1e-6 * torch.eye(n, device=device)

            if n <= 8:
                phi = self._phi_enumerate(cov, n, eps, device)
            else:
                phi = self._phi_spectral(cov, n, eps, device)

        except Exception:
            phi = 0.0

        return float(max(0.0, min(phi, 10.0)))

    def _phi_enumerate(self, cov: torch.Tensor, n: int,
                       eps: torch.Tensor, device) -> float:
        """Enumerate all 2^(n-1)−1 bipartitions, return min MI(A;B)."""
        logdet_full = torch.linalg.slogdet(cov + eps)[1]
        min_phi = float('inf')

        for mask_int in range(1, 1 << (n - 1)):   # half by symmetry
            A = [i for i in range(n) if (mask_int >> i) & 1]
            B = [i for i in range(n) if not (mask_int >> i) & 1]
            if not A or not B:
                continue

            ai = torch.tensor(A, device=device)
            bi = torch.tensor(B, device=device)
            cov_A  = cov[ai][:, ai]
            cov_B  = cov[bi][:, bi]

            try:
                eps_A  = 1e-6 * torch.eye(len(A), device=device)
                eps_B  = 1e-6 * torch.eye(len(B), device=device)
                ld_A   = torch.linalg.slogdet(cov_A + eps_A)[1]
                ld_B   = torch.linalg.slogdet(cov_B + eps_B)[1]
                # MI(A;B) = 0.5 (log det Σ_A + log det Σ_B − log det Σ_AB)
                mi = 0.5 * float(ld_A + ld_B - logdet_full)
                mi = max(0.0, mi)
                if mi < min_phi:
                    min_phi = mi
            except Exception:
                continue

        return 0.0 if min_phi == float('inf') else min_phi

    def _phi_spectral(self, cov: torch.Tensor, n: int,
                      eps: torch.Tensor, device) -> float:
        """Spectral bisection via Fiedler vector of the normalised Laplacian.
        Used when n > 8 (too many bipartitions to enumerate).
        """
        try:
            # Build similarity graph W: W[i,j] = |cov[i,j]| / sqrt(cov[i,i] * cov[j,j])
            diag = cov.diagonal().clamp(min=1e-8).sqrt()
            W = (cov.abs() / (diag.unsqueeze(1) * diag.unsqueeze(0))).clamp(0, 1)
            W.fill_diagonal_(0)

            # Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}
            deg = W.sum(-1).clamp(min=1e-8)
            D_inv_sqrt = deg.pow(-0.5)
            L = torch.eye(n, device=device) - D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)

            # Fiedler vector = eigenvector for second-smallest eigenvalue
            eigvals, eigvecs = torch.linalg.eigh(L)  # sorted ascending
            fiedler = eigvecs[:, 1]                   # (n,)

            # Bipartition: A = positive Fiedler, B = negative
            A = (fiedler >= 0).nonzero(as_tuple=True)[0].tolist()
            B = (fiedler <  0).nonzero(as_tuple=True)[0].tolist()
            if not A or not B:
                return 0.0

            ai = torch.tensor(A, device=device)
            bi = torch.tensor(B, device=device)
            logdet_full = torch.linalg.slogdet(cov + eps)[1]
            eps_A = 1e-6 * torch.eye(len(A), device=device)
            eps_B = 1e-6 * torch.eye(len(B), device=device)
            ld_A  = torch.linalg.slogdet(cov[ai][:, ai] + eps_A)[1]
            ld_B  = torch.linalg.slogdet(cov[bi][:, bi] + eps_B)[1]
            mi = max(0.0, 0.5 * float(ld_A + ld_B - logdet_full))
            return mi
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Coherence helper
    # ------------------------------------------------------------------
    def _compute_coherence(self, module_outputs: dict[str, torch.Tensor],
                           gws_slots: torch.Tensor) -> float:
        """Cosine alignment of module outputs with GWS broadcast mean."""
        gws_mean = gws_slots.mean(dim=(0, 1))
        gws_norm = F.normalize(gws_mean.unsqueeze(0), dim=-1)
        sims = []
        for v in module_outputs.values():
            v_flat = v.mean(0).flatten()[:gws_mean.size(0)]
            if v_flat.size(0) < gws_mean.size(0):
                continue
            sim = F.cosine_similarity(
                F.normalize(v_flat.unsqueeze(0), dim=-1), gws_norm).item()
            sims.append(sim)
        return sum(sims) / max(len(sims), 1)

    def oscillation_spectrum(self) -> dict:
        return {
            "gamma_mean":     float(self._gamma_history.mean()),
            "gamma_std":      float(self._gamma_history.std()),
            "theta_mean":     float(self._theta_history.mean()),
            "alpha_mean":     float(self._alpha_history.mean()),
            "phi_mean":       float(self._phi_history.mean()),
            "coherence_mean": float(self._coherence_history.mean()),
        }
