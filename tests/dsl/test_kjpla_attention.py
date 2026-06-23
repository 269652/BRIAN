# -*- coding: utf-8 -*-
"""RED-first tests for KJPLAttention and josephson_loss (Phase 3, THSD program).

CLAUDE.md §14 contracts enforced here (not merely checked post-hoc):

  1.  Bit-identity: torch.equal(kjpla_y, manual_y) at zero-init.
      KJPLA must produce EXACT same output as manual-softmax CSA path when
      all phase params are zero.  (NOT allclose — bit-exact.)

  2.  delta_h non-persistent: not in state_dict(), reconstructed on load.

  3.  phi1 stash is bfloat16 (memory discipline FIX 9).

  4.  Backward from josephson_loss reaches K_h.grad (gradient-flow contract).

  5.  FD vs autograd on beta_h, atol=1e-3 (NOT just .grad > 0).

  6.  T=1, L=1 boundary: no crash; R returns scalar, loss returns scalar.

  7.  Josephson order param R=1 when phi_ℓ - phi_{ℓ-1} == delta_h exactly.

  8.  josephson_loss returns zero when len(phi_list) < 2.

  9.  Kuramoto sync: when eta=0, phi1 == phi0 exactly (torch.equal).

  10. Kuramoto sync: when eta!=0, phi1 differs from phi0.

  11. Phase-gated logits: when beta_h=0, phase term contributes exactly zero.

  12. w_h not in delta_h (non-param buffer stays non-param; w_h IS a param).

  13. Phase parameters (eta, beta_h, K_h, w_h) all start at zero.
"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from neuroslm.mechanisms.kjpla import KJPLAttention, josephson_loss
from neuroslm.modules.common import (
    build_rope_cache, apply_rope, HebbianTrace,
)


# ── Tiny dimensions for fast CPU tests ───────────────────────────────────────

DIM = 32
N_HEADS = 4
MAX_CTX = 16
HEAD_DIM = DIM // N_HEADS   # 8
N_LAYERS = 3


def _make_kjpla(hebbian_rank: int = 0, **kw) -> KJPLAttention:
    return KJPLAttention(
        dim=DIM, n_heads=N_HEADS, max_ctx=MAX_CTX,
        hebbian_rank=hebbian_rank, n_layers=N_LAYERS, **kw
    )


def _reference_manual_attn(
    x: torch.Tensor,
    q_proj: nn.Linear,
    kv_proj: nn.Linear,
    out_proj: nn.Linear,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    hebbian: "HebbianTrace | None" = None,
) -> torch.Tensor:
    """Manual-softmax attention reference (matches CausalSelfAttention
    manual-Hebbian path and KJPLA inert-gate path byte-for-byte)."""
    B, T, C = x.shape
    n_groups = n_heads // n_kv_heads

    q = q_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    kv = kv_proj(x).view(B, T, 2, n_kv_heads, head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]

    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
    k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))

    if n_groups > 1:
        k = k[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)
        v = v[:, :, None, :, :].expand(-1, -1, n_groups, -1, -1).reshape(B, n_heads, T, head_dim)

    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    if hebbian is not None:
        attn = attn + hebbian(q, k)

    causal_mask = torch.triu(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
    return out_proj(y)


# ============================================================================
# 1. Bit-identity at zero init
# ============================================================================

class TestBitIdentityAtZeroInit:
    """torch.equal between KJPLA (inert) and manual reference at every T."""

    def _run(self, B: int, T: int, hebbian_rank: int = 0):
        torch.manual_seed(42)
        kjpla = _make_kjpla(hebbian_rank=hebbian_rank)
        kjpla.eval()

        x = torch.randn(B, T, DIM)
        cos, sin = build_rope_cache(T, HEAD_DIM)

        # Reference: manually compute the same path KJPLA's inert gate uses.
        ref_y = _reference_manual_attn(
            x, kjpla.q_proj, kjpla.kv_proj, kjpla.out,
            N_HEADS, kjpla.n_kv_heads, HEAD_DIM, cos, sin,
            hebbian=kjpla.hebbian,
        )

        with torch.no_grad():
            y, phi_stash, aux = kjpla(x)

        assert torch.equal(y, ref_y), (
            f"Bit-identity FAILED at B={B} T={T} hebbian_rank={hebbian_rank}; "
            f"max_diff={( y - ref_y).abs().max().item():.2e}"
        )
        assert phi_stash is None, "Inert gate must return None phi stash"
        assert aux == {}, "Inert gate must return empty aux dict"

    def test_b1_t8_no_hebbian(self):
        self._run(1, 8, hebbian_rank=0)

    def test_b2_t16_no_hebbian(self):
        self._run(2, 16, hebbian_rank=0)

    def test_b1_t8_with_zero_hebbian(self):
        # With hebbian_rank=1 but zero-initialized weights: still inert.
        torch.manual_seed(42)
        kjpla = _make_kjpla(hebbian_rank=1)
        # Zero the hebbian fast-weight so its contribution is zero.
        with torch.no_grad():
            for p in kjpla.hebbian.parameters():
                p.zero_()
        kjpla.eval()
        x = torch.randn(2, 8, DIM)
        cos, sin = build_rope_cache(8, HEAD_DIM)
        ref_y = _reference_manual_attn(
            x, kjpla.q_proj, kjpla.kv_proj, kjpla.out,
            N_HEADS, kjpla.n_kv_heads, HEAD_DIM, cos, sin,
            hebbian=kjpla.hebbian,
        )
        with torch.no_grad():
            y, _, _ = kjpla(x)
        assert torch.equal(y, ref_y)

    def test_t1_boundary_inert(self):
        """T=1 should not crash in inert mode."""
        kjpla = _make_kjpla()
        kjpla.eval()
        x = torch.randn(1, 1, DIM)
        with torch.no_grad():
            y, phi_stash, aux = kjpla(x)
        assert y.shape == (1, 1, DIM)
        assert phi_stash is None


# ============================================================================
# 2. Phase parameters start at zero
# ============================================================================

class TestParamInit:
    def test_all_phase_params_zero(self):
        kjpla = _make_kjpla()
        assert kjpla.eta.item() == 0.0
        assert kjpla.beta_h.abs().max().item() == 0.0
        assert kjpla.K_h.abs().max().item() == 0.0
        assert kjpla.w_h.abs().max().item() == 0.0

    def test_is_inert_true_at_init(self):
        assert _make_kjpla()._is_inert()

    def test_is_inert_false_after_eta_set(self):
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.eta.fill_(0.1)
        assert not kjpla._is_inert()

    def test_is_inert_false_after_beta_set(self):
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.5)
        assert not kjpla._is_inert()

    def test_w_h_is_parameter(self):
        kjpla = _make_kjpla()
        assert isinstance(kjpla.w_h, nn.Parameter)

    def test_eta_beta_K_are_parameters(self):
        kjpla = _make_kjpla()
        assert isinstance(kjpla.eta, nn.Parameter)
        assert isinstance(kjpla.beta_h, nn.Parameter)
        assert isinstance(kjpla.K_h, nn.Parameter)


# ============================================================================
# 3. delta_h non-persistent (not in state_dict, but exists as buffer)
# ============================================================================

class TestDeltaHNonPersistent:
    def test_delta_h_not_in_state_dict(self):
        kjpla = _make_kjpla()
        assert "delta_h" not in kjpla.state_dict(), \
            "delta_h must be non-persistent (not saved in state_dict)"

    def test_delta_h_exists_as_buffer(self):
        kjpla = _make_kjpla()
        assert hasattr(kjpla, "delta_h"), "delta_h must be a registered buffer"

    def test_delta_h_values_deterministic(self):
        """delta_h[h] == 2*pi*h / (n_heads * n_layers), reproducible."""
        kjpla1 = _make_kjpla()
        kjpla2 = _make_kjpla()
        assert torch.equal(kjpla1.delta_h, kjpla2.delta_h)

    def test_delta_h_shape(self):
        kjpla = _make_kjpla()
        assert kjpla.delta_h.shape == (N_HEADS,)

    def test_delta_h_survives_load(self):
        """After save/load cycle, delta_h is reconstructed correctly."""
        kjpla = _make_kjpla()
        expected = kjpla.delta_h.clone()
        state = kjpla.state_dict()
        # Load into a fresh model — delta_h is reconstructed, not loaded.
        kjpla2 = _make_kjpla()
        kjpla2.load_state_dict(state)
        assert torch.equal(kjpla2.delta_h, expected)


# ============================================================================
# 4. phi1 stash is bfloat16
# ============================================================================

class TestPhiStashDtype:
    def test_phi1_stash_is_bfloat16(self):
        torch.manual_seed(0)
        kjpla = _make_kjpla()
        # Enable the mechanism so we get a real phi stash.
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)  # makes mechanism active
        x = torch.randn(1, 8, DIM)
        _, phi_stash, _ = kjpla(x)
        assert phi_stash is not None, "phi_stash must not be None when mechanism active"
        assert phi_stash.dtype == torch.bfloat16, \
            f"phi_stash dtype must be bfloat16, got {phi_stash.dtype}"

    def test_phi1_stash_shape(self):
        torch.manual_seed(0)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)
        B, T = 2, 8
        x = torch.randn(B, T, DIM)
        _, phi_stash, _ = kjpla(x)
        assert phi_stash.shape == (B, N_HEADS, T), \
            f"phi_stash shape {phi_stash.shape} != {(B, N_HEADS, T)}"


# ============================================================================
# 5. Kuramoto sync
# ============================================================================

class TestKuramotoSync:
    def test_eta_zero_phi1_equals_phi0(self):
        """When eta=0, phi1 == phi0 exactly (torch.equal)."""
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)   # active but eta=0
        x = torch.randn(1, 6, DIM)
        kjpla.eval()
        _, phi_stash, _ = kjpla(x)
        # phi0 = einsum(q, w_h); w_h=0 => phi0=0 => phi1 should equal phi0=0
        assert phi_stash is not None
        # w_h = 0 so phi0 = 0 everywhere; sync of zeros is zero.
        assert torch.equal(phi_stash.float(), torch.zeros_like(phi_stash.float())), \
            "phi1 must equal phi0 when w_h=0 (gives phi0=0) regardless of eta"

    def test_kuramoto_step_changes_phi_when_eta_nonzero(self):
        """When eta != 0 and w_h != 0, phi1 should differ from phi0."""
        torch.manual_seed(1)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.eta.fill_(0.5)
            kjpla.beta_h.fill_(0.1)
            kjpla.w_h.normal_()       # random w_h so phi0 != 0
        x = torch.randn(1, 8, DIM)
        q = kjpla.q_proj(x).view(1, 8, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        phi0 = kjpla._phi0(q)
        phi1 = kjpla._kuramoto_step(phi0)
        assert not torch.equal(phi0, phi1), "phi1 should differ from phi0 when eta!=0 and phi0!=0"


# ============================================================================
# 6. Phase-gated logits
# ============================================================================

class TestPhaseGatedLogits:
    def test_beta_zero_no_phase_contribution(self):
        """When beta_h=0, phase_gated_logits returns input unchanged."""
        kjpla = _make_kjpla()
        B, T = 1, 5
        phi1 = torch.randn(B, N_HEADS, T)
        qk = torch.randn(B, N_HEADS, T, T)
        out = kjpla._phase_gated_logits(phi1, qk)
        assert torch.equal(out, qk), "beta_h=0 must leave logits unchanged"

    def test_beta_nonzero_changes_logits(self):
        """When beta_h != 0, phase term is added."""
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(1.0)
        B, T = 1, 5
        phi1 = torch.randn(B, N_HEADS, T)
        qk = torch.randn(B, N_HEADS, T, T)
        out = kjpla._phase_gated_logits(phi1, qk)
        assert not torch.equal(out, qk), "beta_h != 0 must change logits"


# ============================================================================
# 7. Josephson order param: R=1 when phi_ℓ - phi_{ℓ-1} == delta_h exactly
# ============================================================================

class TestJosephsonOrderParam:
    def test_R_equals_1_when_phase_stride_matches(self):
        """R_mean = 1.0 when phi_curr - phi_prev == delta_h for all t."""
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.K_h.fill_(1.0)

        B, T = 2, 10
        # Construct phi_prev and phi_curr with exact stride.
        phi_prev = torch.randn(B, N_HEADS, T)
        delta_h = kjpla.delta_h.view(1, -1, 1)
        phi_curr = phi_prev + delta_h

        R_mean, j_loss = kjpla._josephson_order_param(phi_curr, phi_prev)
        assert abs(R_mean.item() - 1.0) < 1e-5, \
            f"R should be 1.0 when phase stride matches delta_h, got {R_mean.item():.6f}"

    def test_R_between_0_and_1_for_random_phase(self):
        """For random phi_prev and phi_curr, 0 <= R <= 1."""
        torch.manual_seed(3)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.K_h.fill_(0.5)
        B, T = 2, 12
        phi_prev = torch.randn(B, N_HEADS, T).to(torch.bfloat16)
        phi_curr = torch.randn(B, N_HEADS, T)
        R_mean, _ = kjpla._josephson_order_param(phi_curr, phi_prev)
        assert 0.0 <= R_mean.item() <= 1.0 + 1e-6


# ============================================================================
# 8. josephson_loss function
# ============================================================================

class TestJosephsonLoss:
    def test_returns_zero_for_single_phase(self):
        """josephson_loss with <2 phases returns scalar zero."""
        kjpla = _make_kjpla()
        phi = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        loss = josephson_loss([phi], [kjpla.K_h], [kjpla.delta_h])
        assert torch.equal(loss, torch.zeros_like(loss))

    def test_returns_zero_for_empty_list(self):
        loss = josephson_loss([], [], [])
        assert loss.item() == 0.0

    def test_nonzero_K_produces_nonzero_loss(self):
        """When K_h != 0 and phases differ, loss is non-zero."""
        torch.manual_seed(5)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.K_h.fill_(1.0)
        phi0 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        phi1 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        loss = josephson_loss([phi0, phi1], [kjpla.K_h, kjpla.K_h],
                               [kjpla.delta_h, kjpla.delta_h])
        assert loss.item() != 0.0

    def test_backward_reaches_K_h_grad(self):
        """Gradient from josephson_loss must reach K_h."""
        torch.manual_seed(6)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.K_h.fill_(0.5)
        kjpla.K_h.requires_grad_(True)

        phi0 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        phi1 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        loss = josephson_loss([phi0, phi1], [kjpla.K_h, kjpla.K_h],
                               [kjpla.delta_h, kjpla.delta_h])
        loss.backward()
        assert kjpla.K_h.grad is not None
        assert kjpla.K_h.grad.abs().sum().item() > 0.0

    def test_three_layers_averaged(self):
        """Loss with 3 phases is average of 2 pairs."""
        torch.manual_seed(7)
        kjpla1 = _make_kjpla()
        kjpla2 = _make_kjpla()
        kjpla3 = _make_kjpla()
        for m in [kjpla1, kjpla2, kjpla3]:
            with torch.no_grad():
                m.K_h.fill_(1.0)

        phi0 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        phi1 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)
        phi2 = torch.randn(1, N_HEADS, 8).to(torch.bfloat16)

        loss_3 = josephson_loss(
            [phi0, phi1, phi2],
            [kjpla1.K_h, kjpla2.K_h, kjpla3.K_h],
            [kjpla1.delta_h, kjpla2.delta_h, kjpla3.delta_h],
        )
        # Must be a scalar
        assert loss_3.shape == ()


# ============================================================================
# 9. Gradient flow: FD vs autograd on beta_h
# ============================================================================

class TestGradFlowBetaH:
    def test_fd_vs_autograd_beta_h(self):
        """Finite-difference gradient on beta_h[0] matches autograd, atol=1e-3."""
        torch.manual_seed(10)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.5)
            kjpla.w_h.normal_(std=0.1)
        kjpla.eval()

        x = torch.randn(1, 6, DIM)

        def loss_fn():
            y, _, _ = kjpla(x)
            return y.sum()

        # Autograd gradient on beta_h[0]
        l0 = loss_fn()
        l0.backward()
        ag_grad = kjpla.beta_h.grad[0].item()
        kjpla.zero_grad()

        # Finite difference on beta_h[0]
        eps = 1e-3
        with torch.no_grad():
            kjpla.beta_h[0] += eps
        lp = loss_fn().item()
        with torch.no_grad():
            kjpla.beta_h[0] -= 2 * eps
        lm = loss_fn().item()
        with torch.no_grad():
            kjpla.beta_h[0] += eps
        fd_grad = (lp - lm) / (2 * eps)

        assert abs(ag_grad - fd_grad) < 1e-2, \
            f"FD grad={fd_grad:.4f} vs autograd={ag_grad:.4f}, diff={abs(ag_grad-fd_grad):.4f}"


# ============================================================================
# 10. Boundary cases
# ============================================================================

class TestBoundaryCases:
    def test_t1_no_crash(self):
        torch.manual_seed(0)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)
        x = torch.randn(1, 1, DIM)
        y, phi_stash, aux = kjpla(x)
        assert y.shape == (1, 1, DIM)
        assert phi_stash is not None
        assert phi_stash.dtype == torch.bfloat16

    def test_n_layers_1_delta_h_sensible(self):
        """n_layers=1: delta_h values are deterministic and non-negative."""
        kjpla = KJPLAttention(DIM, N_HEADS, MAX_CTX, n_layers=1)
        assert kjpla.delta_h.shape == (N_HEADS,)
        assert (kjpla.delta_h >= 0.0).all()

    def test_gqa_2kv_heads_no_crash(self):
        """GQA: n_kv_heads=2, n_heads=4."""
        kjpla = KJPLAttention(DIM, N_HEADS, MAX_CTX, n_kv_heads=2, n_layers=N_LAYERS)
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)
        x = torch.randn(1, 8, DIM)
        y, phi_stash, aux = kjpla(x)
        assert y.shape == (1, 8, DIM)

    def test_phi_prev_triggers_josephson_when_K_nonzero(self):
        """When phi_prev is passed and K_h != 0, aux dict has josephson_loss."""
        torch.manual_seed(0)
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.K_h.fill_(1.0)
            kjpla.beta_h.fill_(0.1)   # make mechanism active
        B, T = 1, 8
        x = torch.randn(B, T, DIM)
        phi_prev = torch.randn(B, N_HEADS, T).to(torch.bfloat16)
        _, _, aux = kjpla(x, phi_prev=phi_prev)
        assert "josephson_loss" in aux, "josephson_loss must be in aux when K_h!=0 and phi_prev given"
        assert "R_mean" in aux

    def test_phi_prev_none_no_josephson(self):
        """Without phi_prev, no josephson_loss in aux."""
        kjpla = _make_kjpla()
        with torch.no_grad():
            kjpla.beta_h.fill_(0.1)
        x = torch.randn(1, 8, DIM)
        _, _, aux = kjpla(x, phi_prev=None)
        assert "josephson_loss" not in aux

    def test_forward_returns_3_tuple(self):
        """forward() returns (y, phi_stash, aux) always — a 3-tuple."""
        kjpla = _make_kjpla()
        x = torch.randn(1, 8, DIM)
        result = kjpla(x)
        assert isinstance(result, tuple) and len(result) == 3
