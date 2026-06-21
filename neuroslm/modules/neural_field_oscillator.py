# -*- coding: utf-8 -*-
"""Neural Field Oscillator (NFO) — residual-stream block.

Canonical Python lowering of ``lib/blocks/neural_field_oscillator.neuro``.

Three composable mechanisms in one residual block:

1. **Kuramoto graph synchronisation** of an M-oscillator complex field
   ``z = A·e^{iφ}`` lifted from the residual stream.  Coupling matrix
   ``K`` is a single-head causal softmax-attention over the polar
   embedding, so message-passing respects token order.

2. **Swift–Hohenberg amplitude flow** with cubic damping toward a
   learnable set-point ``A*`` — a Lyapunov-stable nonlinear gain
   control (no exploding-feature pathology, no LayerNorm wash-out).

3. **Coherence-gated readout** ``h_out = h_in + α·Wo·(g ⊙ z′)`` with
   ``g = R / max_c R`` — tokens whose oscillators are *locally
   synchronous* with their mean field get a louder voice in the
   write-back.  ``Wo`` is zero-init, so the first forward is
   bit-identical to the baseline residual stream (ReZero discipline).

The block exposes ``last_state`` with the polar field, the local order
parameter ``R`` (per-token mean coherence), the mean phase ``ψ`` and
the bipartition-coherence functional ``Phi_kappa`` — consumed by
``neuroslm.emergent.nfo_coherence.NFOCoherenceProbe`` for telemetry
and (Phase-2) by an optional auxiliary loss.

Hypotheses formalised against this module:

* **H015** — bipartition coherence is a closed-form Φ lower bound
  (``hypothesis/proofs/H015_kuramoto_coherence_phi_lower_bound.lean``).
* **H016** — coherence gate is information-preserving
  (``hypothesis/proofs/H016_coherence_gate_information_preserving.lean``).
* **H017** — Swift–Hohenberg amplitude flow is contractive
  (``hypothesis/proofs/H017_swift_hohenberg_contractive.lean``).
* **H018** — zero-init readout ⇒ baseline-identity forward
  (``hypothesis/proofs/H018_nfo_readout_zero_init_identity.lean``).

References
----------
* Kuramoto, Y. (1984) *Chemical Oscillations, Waves, and Turbulence*.
* Cross, M. & Hohenberg, P. (1993) *Pattern formation outside of
  equilibrium*, Rev. Mod. Phys. 65, 851 — Swift–Hohenberg dynamics.
* Singer, W. (1999) Neuronal synchrony: a versatile code for the
  definition of relations.  Neuron 24, 49 — binding-by-synchrony.
* Muller, L. et al. (2018) Cortical travelling waves: mechanisms and
  computational principles.  Nat. Rev. Neurosci. 19, 255.
* Tononi, G. (2016) Consciousness as integrated information.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


# ──────────────────────────────────────────────────────────────────────
# Tunables — frozen at the dataclass so DSL → module is one obvious call
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NFOConfig:
    """Hyperparameters for one Neural Field Oscillator block.

    Defaults are chosen so the block is a strict no-op at init
    (``alpha_init = 0``) and uses < 1 % of the host transformer block's
    parameter count for typical SmolLM-sized models (M = 32, d = 512:
    ≈ 35k params vs. 4.7M for the host transformer block).
    """

    # Oscillator field width.  Independent of d_model so two NFO blocks
    # can share an M while the trunk widens.  Practical range: 16 … 64.
    n_osc: int = 32

    # Number of Euler steps to integrate per forward.  More steps =
    # better coupling but more compute (linear in S).  S = 1 is enough
    # at training time; S = 3 helps at inference for long contexts.
    n_steps: int = 1

    # Step size for the Euler integrator.  ``dt`` is learnable per block
    # but kept bounded by ``dt_max`` for stability (H017 contractivity).
    dt_init: float = 0.10
    dt_max: float = 0.45

    # Initial Kuramoto coupling strength κ.  Zero-init means the
    # oscillator field is purely free at step 0 — perfect-information
    # baseline.  Learnable per block.
    kappa_init: float = 0.0
    kappa_max: float = 1.5

    # Initial readout gain α.  Combined with zero-init ``W_o``, this
    # guarantees a bit-identical baseline at step 0 (H018).
    alpha_init: float = 0.0

    # Swift–Hohenberg drive μ and set-point A*.
    mu_init: float = 0.5
    a_star_init: float = 1.0

    # Causal message-passing kernel — single-head softmax-attention over
    # the polar embedding (φ, log A) of dimension 2·M.
    kernel_temperature: float = 1.0

    # Numerical fudge factor for sqrt() / division-by-norm guards.
    eps: float = 1e-6

    # Telemetry budget — how many of the per-token statistics to expose.
    # All free, but we cap the size of ``last_state`` to keep the
    # observer overhead bounded.
    expose_phi_lower_bound: bool = True
    expose_polar_field: bool = False


# ──────────────────────────────────────────────────────────────────────
# Helpers — math kept readable and matches lib/blocks/*.neuro line-by-line
# ──────────────────────────────────────────────────────────────────────

def _complex_polar(z_re: torch.Tensor, z_im: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Polar coordinates of ``z = z_re + i·z_im`` with NaN-safe sqrt.

    See ``nfo_polar`` in ``lib/blocks/neural_field_oscillator.neuro``.
    """
    A = torch.sqrt(z_re * z_re + z_im * z_im + eps * eps)
    phi = torch.atan2(z_im, z_re)
    return A, phi


def _bipartition_coherence(R: torch.Tensor) -> torch.Tensor:
    """Closed-form Φ lower bound (H015 statement).

    Given per-token order-parameter ``R ∈ [0, 1]^{B×T×M}``, return the
    mean-field incoherence functional

        Φ_κ(R) := mean over (token × oscillator) of ``1 − R``.

    H015 (formalised in
    ``hypothesis/proofs/H015_kuramoto_coherence_phi_lower_bound.lean``)
    establishes that the *complement* ``1 − Φ_κ`` lower-bounds the
    sheaf-Laplacian Φ proxy of H001 by an additive coupling argument
    on the cut edges of any bipartition of the token graph.

    Returned as a scalar (batch+token+osc mean) so it slots straight
    into the metric observer.
    """
    return (1.0 - R.clamp(0.0, 1.0)).mean()


# ──────────────────────────────────────────────────────────────────────
# The block
# ──────────────────────────────────────────────────────────────────────

class NeuralFieldOscillator(nn.Module):
    """Drop-in residual block adding oscillator-field dynamics to a trunk.

    Forward signature is compatible with the residual contract
    ``h_out = h_in + Δh``: the block does not consume any extra inputs
    (NT, memory-kv) so it can be interleaved between any pair of
    transformer blocks (or appended at the trunk tail) without changing
    the surrounding plumbing.

    Examples
    --------
    >>> import torch
    >>> from neuroslm.modules.neural_field_oscillator import (
    ...     NeuralFieldOscillator, NFOConfig)
    >>> blk = NeuralFieldOscillator(d_model=64, cfg=NFOConfig(n_osc=16))
    >>> h = torch.randn(2, 8, 64)
    >>> y = blk(h)
    >>> torch.allclose(y, h)             # zero-init guarantees identity
    True
    """

    def __init__(self, d_model: int, cfg: Optional[NFOConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = NFOConfig()
        self.cfg = cfg
        self.d_model = int(d_model)
        self.M = int(cfg.n_osc)
        if self.M < 2:
            raise ValueError("n_osc must be >= 2")
        if self.M > self.d_model:
            raise ValueError("n_osc must be <= d_model")
        # ── Lift: residual → complex field ─────────────────────────
        self.norm = RMSNorm(self.d_model)
        self.lift_re = nn.Linear(self.d_model, self.M, bias=False)
        self.lift_im = nn.Linear(self.d_model, self.M, bias=False)
        # ── Message-passing kernel (single-head causal softmax) ───
        # Q, K project the polar embedding (φ, log A) ∈ ℝ^{2M} → ℝ^{M}.
        self.q_proj = nn.Linear(2 * self.M, self.M, bias=False)
        self.k_proj = nn.Linear(2 * self.M, self.M, bias=False)
        # ── Dynamics parameters ────────────────────────────────────
        # ω: per-oscillator intrinsic frequency, small Gaussian init.
        self.omega = nn.Parameter(torch.randn(self.M) * 0.05)
        # κ: Kuramoto coupling — sigmoid-bounded learnable scalar.
        # Init κ_init via inverse-sigmoid so the *raw* parameter matches
        # the spec, and the sigmoid output is bounded in (0, κ_max).
        # We keep them simple scalars; per-oscillator κ_i is a 1-line
        # extension if needed.
        self._raw_kappa = nn.Parameter(
            torch.logit(torch.tensor(
                max(cfg.kappa_init / max(cfg.kappa_max, 1e-6), 1e-6))))
        # dt: per-block step size — softplus to keep positive.
        self._raw_dt = nn.Parameter(
            torch.logit(torch.tensor(max(cfg.dt_init / max(cfg.dt_max, 1e-6), 1e-6))))
        # Swift–Hohenberg μ and A* per oscillator.
        self.mu = nn.Parameter(torch.full((self.M,), float(cfg.mu_init)))
        self.a_star = nn.Parameter(torch.full((self.M,), float(cfg.a_star_init)))
        # ── Readout back into the residual ─────────────────────────
        self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
        self.read_out = nn.Linear(self.M, self.d_model, bias=False)
        # ── Init discipline: zero-init readout ⇒ baseline-identity ──
        nn.init.zeros_(self.read_out.weight)
        # Sensible inits for the lift heads.  Small Gaussian keeps the
        # oscillator field bounded around |A| ≈ A*_init at step 0.
        nn.init.normal_(self.lift_re.weight, std=0.02)
        nn.init.normal_(self.lift_im.weight, std=0.02)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        # Last computed state, exposed to the metric observer.  Always
        # detached — never participates in the LM gradient.
        self.last_state: Dict[str, torch.Tensor] = {}

    # ── Tiny accessors that make the bounded params readable ─────────

    @property
    def kappa(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_kappa) * self.cfg.kappa_max

    @property
    def dt(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_dt) * self.cfg.dt_max

    # ── Forward ──────────────────────────────────────────────────────

    def _message_field(
        self,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
        A: torch.Tensor,
        phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal mean-field (z̄, R, ψ) under a single softmax-attention
        kernel over the polar embedding.

        Shapes (with M = self.M):
            z_re, z_im, A, phi   :: (B, T, M)
            returns
                zbar_re, zbar_im :: (B, T, M)
                R, psi           :: (B, T, M)

        The kernel is shape-correct under causal masking: the (T, T)
        attention score is left as a single head so the message
        composition stays cheap (O(T²·M)).  Each oscillator m sees
        the same K_ij row, so the per-oscillator mean-field is

            zbar_im = Σ_j K_ij · z_jm.

        i.e. tokens are aggregated, oscillators are not mixed.
        """
        B, T, M = z_re.shape
        # Polar embedding (φ, log A) — bounded, scale-equivariant.
        polar = torch.cat([phi, torch.log(A + self.cfg.eps)], dim=-1)  # (B, T, 2M)
        q = self.q_proj(polar)  # (B, T, M)
        k = self.k_proj(polar)  # (B, T, M)
        # (B, T, T) attention scores.
        scale = max(self.cfg.kernel_temperature, 1e-3) * math.sqrt(M)
        scores = torch.einsum("bti,bsi->bts", q, k) / scale
        # Causal mask.
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=z_re.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal
        K = torch.softmax(scores, dim=-1)
        # Mean field per oscillator.
        zbar_re = torch.einsum("bts,bsm->btm", K, z_re)
        zbar_im = torch.einsum("bts,bsm->btm", K, z_im)
        R, psi = _complex_polar(zbar_re, zbar_im, self.cfg.eps)
        return zbar_re, zbar_im, R, psi

    def _step(
        self,
        A: torch.Tensor,
        phi: torch.Tensor,
        z_re: torch.Tensor,
        z_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Euler step of the coupled (φ, A) Kuramoto/SH system.

        Returns the next ``(A, φ, z_re, z_im, R, ψ)`` so the caller can
        iterate or pass the polar pieces to the readout.
        """
        zbar_re, zbar_im, R, psi = self._message_field(z_re, z_im, A, phi)
        kappa = self.kappa
        dt = self.dt
        # Phase update: φ̇ = ω + κ·R·sin(ψ − φ).
        phi_dot = self.omega + kappa * R * torch.sin(psi - phi)
        phi_next = phi + dt * phi_dot
        # Amplitude update: cubic Lyapunov damping toward A* + coupling.
        Abar = torch.sqrt(zbar_re * zbar_re + zbar_im * zbar_im + self.cfg.eps * self.cfg.eps)
        A_dot = (
            self.mu * A
            - 0.25 * (A * A - self.a_star * self.a_star) * A
            + kappa * R * (Abar - A)
        )
        # Clamp to non-negative: amplitude is a physical magnitude (||z||)
        # and cannot be negative.  Without the clamp, large A can drive
        # A_dot ≈ -0.25*A³ to overshoot zero in one Euler step, and the
        # subsequent step then explodes to NaN (Euler instability for
        # stiff cubic nonlinearities, threshold A ≈ 2/sqrt(dt) ≈ 3).
        A_next = (A + dt * A_dot).clamp(min=self.cfg.eps)
        # Re-cartesianise.
        z_re_next = A_next * torch.cos(phi_next)
        z_im_next = A_next * torch.sin(phi_next)
        return A_next, phi_next, z_re_next, z_im_next, R, psi

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply oscillator-field dynamics and write a coherence-gated
        readout back into the residual stream.

        Parameters
        ----------
        h : torch.Tensor
            ``(B, T, d_model)``.  Float32/16/bf16 all OK; the dynamics
            run in fp32 internally to keep sin/atan2 well-conditioned.

        Returns
        -------
        torch.Tensor
            Same shape as ``h``.  At init, equals ``h`` exactly (H018).
        """
        in_dtype = h.dtype
        hf = h.float()
        x = self.norm(hf)
        # Lift to complex oscillator field z = u + i·v ∈ ℂ^M.
        z_re = self.lift_re(x)
        z_im = self.lift_im(x)
        A, phi = _complex_polar(z_re, z_im, self.cfg.eps)
        # Iterate the coupled dynamics for n_steps.
        R: Optional[torch.Tensor] = None
        psi: Optional[torch.Tensor] = None
        for _ in range(max(1, int(self.cfg.n_steps))):
            A, phi, z_re, z_im, R, psi = self._step(A, phi, z_re, z_im)
        assert R is not None and psi is not None  # for type-checker
        # Coherence gate: g = R / (max_c R + eps), broadcast per token.
        R_max = R.max(dim=-1, keepdim=True).values
        g = R / (R_max + self.cfg.eps)
        # Binding-by-synchrony readout: in-phase component of each
        # oscillator with its local mean field.
        y = g * A * torch.cos(phi - psi)              # (B, T, M)
        delta = self.alpha * self.read_out(y)         # (B, T, d_model)
        # ── Telemetry: never grad-leaks into LM loss ───────────────
        with torch.no_grad():
            self.last_state = {
                "R_mean": R.mean().detach(),
                "R_max": R_max.mean().detach(),
                "A_mean": A.mean().detach(),
                "A_std": A.std().detach(),
                "phi_circular_var": (1.0 - (torch.cos(phi).mean() ** 2 + torch.sin(phi).mean() ** 2)).clamp_min(0.0).detach(),
                "kappa": self.kappa.detach(),
                "dt": self.dt.detach(),
                "alpha": self.alpha.detach(),
            }
            if self.cfg.expose_phi_lower_bound:
                self.last_state["phi_kappa"] = _bipartition_coherence(R).detach()
            if self.cfg.expose_polar_field:
                self.last_state["A"] = A.detach()
                self.last_state["phi"] = phi.detach()
                self.last_state["R"] = R.detach()
        return (h + delta).to(dtype=in_dtype)


# ──────────────────────────────────────────────────────────────────────
# Factory — DSL hook
# ──────────────────────────────────────────────────────────────────────

def make_nfo(spec, d_model: int) -> Optional[NeuralFieldOscillator]:
    """Build an :class:`NeuralFieldOscillator` from a DSL spec dict.

    Spec keys (all optional; defaults from :class:`NFOConfig`):

        enabled              bool
        n_osc                int     — oscillator count per token
        n_steps              int     — Euler integration substeps
        dt_init, dt_max      float   — step-size bounds
        kappa_init, kappa_max float  — Kuramoto coupling bounds
        alpha_init           float   — readout gain (keep 0 for baseline)
        mu_init              float   — Swift–Hohenberg drive
        a_star_init          float   — amplitude set-point
        kernel_temperature   float   — softmax temperature
        expose_phi_lower_bound bool  — emit Φ_κ telemetry
        expose_polar_field   bool    — emit full polar tensors

    Returns ``None`` if ``enabled = False`` so the host trunk can skip
    instantiating the module entirely.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, dict):
        if spec.get("enabled", True) is False:
            return None
        cfg = NFOConfig(**{k: v for k, v in spec.items() if k in NFOConfig.__dataclass_fields__})
    elif spec is True:
        cfg = NFOConfig()
    else:
        raise TypeError(f"NFO spec must be bool|dict|None, got {type(spec).__name__}")
    return NeuralFieldOscillator(d_model=d_model, cfg=cfg)


__all__ = ["NeuralFieldOscillator", "NFOConfig", "make_nfo"]
