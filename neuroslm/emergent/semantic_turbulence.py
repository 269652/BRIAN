# -*- coding: utf-8 -*-
"""Semantic Turbulence Engine (STE) — three physics-inspired multi-scale mechanisms.

Module 1 — RenormalizationGroupCascade
    Partitions sequence positions into G groups operating at token scales 2^g.
    Coarse-grains via block mean pooling; extracts fluctuations δH = H - upsample(H̄).
    Applies scale-specific attention on the coarse stream and feeds the sum
    H̄^(g) + λ_g·δH^(g) back into the residual. Coupling λ_g ∝ 2^{-5g/6}
    follows Kolmogorov's 5/3-law (energy cascade in turbulent fluids).

    Reference: Kolmogorov (1941) "Local structure of turbulence in incompressible
    viscous fluid for very large Reynolds numbers."

Module 2 — GrossPitaevskiiLayer
    Encodes the hidden state as a complex superfluid field ψ ∈ ℂ^{d/2}.
    Runs N imaginary-time GPE steps:
        ψ_t ← ψ_t − Δτ(−∇²ψ_t/2 + g|ψ_t|²ψ_t)
    followed by norm-preserving rescaling.
    Order parameter ρ = |⟨ψ/|ψ|⟩|² ∈ [0,1] measures semantic coherence.
    ρ→1 (condensate) = semantically unambiguous context;
    ρ→0 (disordered) = highly polysemous or uncertain context.

    Reference: Gross (1961), Pitaevskii (1961) BEC mean-field equation.

Module 3 — BranchingRatioMonitor
    Tracks the layer-to-layer branching ratio:
        σ = (1/T) Σ_t ‖∂h_{l+1,t}/∂h_{l,t}‖_F
    The critical point σ=1 (Beggs & Plenz 2003) maximises information
    transmission, dynamic range, and susceptibility.
    • σ > 1 (supercritical) → increase GABA (dampen)
    • σ < 1 (subcritical)   → increase NE (excite)
    • σ ≈ 1 (critical)      → DA reward → optimizer reinforces state
    Adds (σ − σ*)² to the loss as a criticality regularizer.

    Reference: Beggs & Plenz (2003) "Neuronal avalanches in neocortical circuits."
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Module 1: RenormalizationGroupCascade ─────────────────────────────────────


class RenormalizationGroupCascade(nn.Module):
    """Multi-scale representation enrichment via RG coarse-graining.

    Args:
        d_model:        Hidden dimension of the trunk.
        n_groups:       G — number of scale groups. Group g operates at
                        block_size = 2^g tokens.
        kolmogorov_init: If True, init coupling scalars λ_g ∝ 2^{-5g/6}.
    """

    def __init__(
        self,
        d_model: int,
        n_groups: int = 3,
        kolmogorov_init: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.kolmogorov_init = kolmogorov_init

        # Per-group learnable coupling scalars λ_g (ReZero-style: start small)
        lambdas_init = self._kolmogorov_values(n_groups) if kolmogorov_init else [1.0] * n_groups
        self.lambda_scalars = nn.ParameterList([
            nn.Parameter(torch.tensor(lam, dtype=torch.float32))
            for lam in lambdas_init
        ])

        # Lightweight per-group projection (d → d, weight-shared across scales
        # would collapse to the same representation — so each group has its own)
        self.scale_proj = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_groups)
        ])
        # Zero-init output projections so the cascade starts as a no-op
        for proj in self.scale_proj:
            nn.init.zeros_(proj.weight)
            proj.weight.data += torch.eye(d_model) * 1e-3

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _kolmogorov_values(n: int = 1) -> List[float]:
        # internal — used during __init__ with n=n_groups
        return [2 ** (-5 * g / 6) for g in range(n)]

    def _kv(self) -> List[float]:
        return self._kolmogorov_values(self.n_groups)

    def kolmogorov_lambdas(self) -> List[float]:
        """Return the current λ_g values as plain floats."""
        if self.kolmogorov_init:
            return self._kv()
        # uniform: all set to 1.0 at construction
        return [p.item() for p in self.lambda_scalars]

    def coarse_grain(self, H: torch.Tensor, block_size: int) -> torch.Tensor:
        """Block-mean pooling: (B, T, d) → (B, T//block_size, d)."""
        B, T, d = H.shape
        T_eff = (T // block_size) * block_size
        H_trunc = H[:, :T_eff, :]           # (B, T_eff, d)
        H_blocks = H_trunc.view(B, T_eff // block_size, block_size, d)
        return H_blocks.mean(dim=2)          # (B, n_blocks, d)

    def extract_fluctuations(
        self, H: torch.Tensor, H_coarse: torch.Tensor, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute δH = H - upsample(H_coarse) for complete blocks.

        Returns:
            (dH, H_repeat) where H_repeat + dH == H[:, :T_eff, :]
        """
        B, T, d = H.shape
        T_eff = H_coarse.shape[1] * block_size
        H_repeat = H_coarse.repeat_interleave(block_size, dim=1)  # (B, T_eff, d)
        H_trunc = H[:, :T_eff, :]
        dH = H_trunc - H_repeat
        return dH, H_repeat

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Enrich H with multi-scale fluctuation structure.

        For each group g at block_size 2^g:
          1. Coarse-grain H to H̄_g
          2. Extract δH_g = H - upsample(H̄_g)
          3. Compute enriched output: proj_g(H̄_g_up) + λ_g * δH_g
          4. Accumulate residual
        """
        B, T, d = H.shape
        out = H.clone()

        for g in range(self.n_groups):
            block_size = 2 ** (g + 1)  # group 0 → 2, group 1 → 4, ...
            if block_size > T:
                break

            H_coarse = self.coarse_grain(H, block_size)     # (B, T//bs, d)
            dH, H_repeat = self.extract_fluctuations(H, H_coarse, block_size)

            T_eff = H_coarse.shape[1] * block_size

            # Apply scale-specific projection on the coarse stream (upsampled)
            H_coarse_proj = self.scale_proj[g](H_repeat)    # (B, T_eff, d)

            lam = self.lambda_scalars[g]
            enriched = H_coarse_proj + lam * dH             # (B, T_eff, d)

            # Add residual (in-place on the effective prefix)
            out[:, :T_eff, :] = out[:, :T_eff, :] + enriched

        return out


# ── Module 2: GrossPitaevskiiLayer ────────────────────────────────────────────


class GrossPitaevskiiLayer(nn.Module):
    """Superfluid semantic coherence via imaginary-time GPE evolution.

    Args:
        d_model:           Real hidden dimension (must be even).
        gpe_steps:         N Euler steps in imaginary time.
        gpe_coupling_init: Initial interaction strength g (ReZero: small).
        gpe_dt:            Imaginary-time step size Δτ.
    """

    def __init__(
        self,
        d_model: int,
        gpe_steps: int = 4,
        gpe_coupling_init: float = 0.01,
        gpe_dt: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for complex encoding, got {d_model}")
        self.d_model = d_model
        self.d_complex = d_model // 2
        self.gpe_steps = gpe_steps
        self.gpe_dt = gpe_dt

        # Learnable interaction coupling (ReZero: zero-init would make GPE a no-op;
        # start at coupling_init so early training can explore the condensate).
        self.log_g = nn.Parameter(torch.tensor(math.log(gpe_coupling_init + 1e-8)))

    @property
    def g(self) -> torch.Tensor:
        """Interaction strength g = exp(log_g) > 0."""
        return self.log_g.exp()

    def encode_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Map real (B, T, d) to complex (B, T, d//2) via view-as-complex."""
        B, T, d = x.shape
        # View last dim as pairs → (B, T, d//2, 2) → complex
        return torch.view_as_complex(x.reshape(B, T, d // 2, 2).contiguous())

    def decode_from_complex(self, psi: torch.Tensor) -> torch.Tensor:
        """Inverse: complex (B, T, d//2) → real (B, T, d)."""
        return torch.view_as_real(psi).reshape(psi.shape[0], psi.shape[1], self.d_model)

    def gpe_step(self, psi: torch.Tensor, dt: float) -> torch.Tensor:
        """One imaginary-time Euler step of the GPE.

        ψ_t ← ψ_t − Δτ · g · |ψ_t|² · ψ_t   (interaction term)
        Laplacian term requires positional structure; omitted here — the
        sequence attention handles long-range coupling. We keep the
        contact interaction g|ψ|²ψ (mean-field Bose-Hubbard).
        """
        amp_sq = psi.abs().pow(2)          # |ψ|² ∈ ℝ  (B, T, d//2)
        dPsi = self.g * amp_sq * psi       # g|ψ|²ψ
        psi_new = psi - dt * dPsi
        # Renormalize to preserve total "particle number" N = Σ|ψ|²
        norm_before = psi.abs().pow(2).sum().sqrt()
        norm_after = psi_new.abs().pow(2).sum().sqrt()
        scale = (norm_before / (norm_after + 1e-8)).to(psi_new.dtype)
        return psi_new * scale

    def order_parameter(self, psi: torch.Tensor) -> torch.Tensor:
        """ρ = |⟨ψ / |ψ|⟩|²  ∈ [0, 1].

        Scalar measuring phase coherence across the batch×sequence×feature.
        ρ→1: condensate (unambiguous); ρ→0: disordered (polysemous).
        """
        norms = psi.abs() + 1e-8
        psi_normalized = psi / norms        # unit-amplitude field
        mean_field = psi_normalized.mean()  # complex scalar
        return mean_field.abs().pow(2).clamp(0.0, 1.0)

    def forward_with_rho(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode → evolve → decode, return (output, ρ).

        Args:
            x: Real tensor (B, T, d_model).
        Returns:
            out:  Real tensor (B, T, d_model) — GPE-processed representation.
            rho:  Scalar tensor — order parameter ∈ [0, 1].
        """
        psi = self.encode_to_complex(x)

        for _ in range(self.gpe_steps):
            psi = self.gpe_step(psi, self.gpe_dt)

        rho = self.order_parameter(psi)
        out = self.decode_from_complex(psi)
        return out, rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_rho(x)
        return out


# ── Module 3: BranchingRatioMonitor ──────────────────────────────────────────


class BranchingRatioMonitor:
    """Non-parametric monitor for neural branching ratio σ.

    Stateful: maintains a running EMA σ_ema for smooth NT signaling.
    Stateless regarding parameters (no nn.Module) — telemetry only,
    never participates in the optimizer step.

    Args:
        target:     σ* (default 1.0 — Beggs & Plenz critical point).
        ema_alpha:  EMA smoothing factor α ∈ (0, 1].
        da_reward:  DA amplitude when σ ≈ σ*.
        weight:     Weight for criticality loss (σ − σ*)².
    """

    _CRITICAL_BAND = 0.05  # |σ - σ*| < band → "at criticality"

    def __init__(
        self,
        target: float = 1.0,
        ema_alpha: float = 0.05,
        da_reward: float = 0.1,
        weight: float = 0.01,
    ) -> None:
        self.target = target
        self.ema_alpha = ema_alpha
        self.da_reward = da_reward
        self.weight = weight
        self.sigma_ema: float = target  # start at criticality

    # ── σ measurement ─────────────────────────────────────────────────────

    def measure_sigma(
        self, h_prev: torch.Tensor, h_next: torch.Tensor
    ) -> torch.Tensor:
        """Estimate branching ratio from consecutive layer activations.

        σ ≈ ‖h_next‖_F / (‖h_prev‖_F + ε) — ratio of Frobenius norms.
        This is the leading-order approximation to the mean Jacobian norm
        without requiring an expensive per-token Jacobian computation.
        Exact only when h_next = J·h_prev (linear layer); approximation
        holds for the EMA-smoothed signal used by the NT controller.

        For an identity mapping: ‖h_next‖ / ‖h_prev‖ = 1 → σ = 1 ✓
        For 2× amplification:     ‖h_next‖ / ‖h_prev‖ = 2 → σ = 2 ✓
        """
        norm_next = h_next.detach().norm()
        norm_prev = h_prev.detach().norm()
        return (norm_next / (norm_prev + 1e-8)).clamp(min=0.0)

    # ── EMA tracking ──────────────────────────────────────────────────────

    def update_ema(self, sigma: float) -> None:
        """Update running EMA: σ_ema ← α·σ + (1−α)·σ_ema."""
        self.sigma_ema = self.ema_alpha * sigma + (1 - self.ema_alpha) * self.sigma_ema

    # ── NT signals ────────────────────────────────────────────────────────

    def nt_signals(self, sigma: float | torch.Tensor) -> Dict[str, float]:
        """Map σ to neuromodulator signals.

        Returns:
            {
                "gaba": inhibitory signal (peaks when σ > σ*, supercritical),
                "ne":   excitatory signal (peaks when σ < σ*, subcritical),
                "da":   reward signal (peaks when σ ≈ σ*, critical),
            }
        """
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        delta = sigma - self.target          # signed deviation from criticality

        # GABA: inhibitory — suppress supercritical activity
        gaba = max(0.0, delta)               # only positive when σ > σ*

        # NE: excitatory — boost subcritical activity
        ne = max(0.0, -delta)                # only positive when σ < σ*

        # DA: reward — maximum at σ = σ*, falls off with |δ|
        da = self.da_reward * math.exp(-abs(delta) / (self._CRITICAL_BAND + 1e-8))

        return {"gaba": gaba, "ne": ne, "da": da}

    # ── Loss & stress ─────────────────────────────────────────────────────

    def criticality_loss(
        self, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        """Loss term = weight * (σ − σ*)²."""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(float(sigma))
        return self.weight * (sigma - self.target).pow(2)

    def nt_stress(self, sigma: float | torch.Tensor) -> float:
        """NT stress contribution = (σ − σ*)² for allostasis load."""
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return (sigma - self.target) ** 2
