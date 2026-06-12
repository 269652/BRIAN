"""Math contracts for HRR-bound key/value memory.

Pins Plate's (1995) HRR algebra properties + the HRRMemory module's
edge-endpoint contract:

  * Circular convolution is commutative, has 1-pulse identity.
  * Unbind inverts bind on its own argument (perfect retrieval at N=1).
  * Multiple bindings degrade gracefully: noise stays bounded.
  * HRRMemory.forward preserves shape, accepts (B,T,D), produces a
    different output than the input (it actually reads memory).
  * All parameters receive gradients.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroslm.modules.hrr_memory import (
    HRRMemory,
    hrr_bind,
    hrr_inverse,
    hrr_inverse_exact,
    hrr_superpose,
    hrr_unbind,
    hrr_unbind_exact,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Primitive properties — bind / inverse / unbind
# ──────────────────────────────────────────────────────────────────────


class TestHRRPrimitives:
    def test_bind_is_commutative(self):
        torch.manual_seed(0)
        a = torch.randn(64)
        b = torch.randn(64)
        ab = hrr_bind(a, b)
        ba = hrr_bind(b, a)
        # Circular convolution is commutative; numerical error is
        # entirely FFT round-trip.
        assert torch.allclose(ab, ba, atol=1e-5)

    def test_bind_has_pulse_identity(self):
        # The discrete impulse e_0 = [1, 0, 0, ...] is the binding
        # identity: a ⊛ e_0 == a.
        a = torch.randn(32)
        e0 = torch.zeros(32)
        e0[0] = 1.0
        out = hrr_bind(a, e0)
        assert torch.allclose(out, a, atol=1e-5)

    def test_inverse_is_involution_in_spectrum(self):
        # inverse(inverse(k)) == k for real k (FFT conj is involutive).
        torch.manual_seed(1)
        k = torch.randn(64)
        k_double = hrr_inverse(hrr_inverse(k))
        assert torch.allclose(k_double, k, atol=1e-5)

    def test_unbind_perfectly_retrieves_single_binding(self):
        """N=1 retrieval with the EXACT spectral inverse is bit-equal."""
        torch.manual_seed(2)
        k = torch.randn(128)
        v = torch.randn(128)
        m = hrr_bind(k, v)
        v_hat = hrr_unbind_exact(m, k)
        # Exact inverse: bind(k, exact_inv(k)) == δ → retrieval == v
        # up to a tight FFT round-trip tolerance.
        assert torch.allclose(v_hat, v, atol=1e-4), (
            f"exact-inverse retrieval failed: "
            f"max diff = {(v_hat - v).abs().max().item():.2e}"
        )

    def test_approximate_unbind_has_high_similarity(self):
        """Involution-inverse retrieval is approximate; cos should
        still clear 0.5 for random Gaussian keys at d=256 — well above
        the noise floor (random cos ≈ 0).

        The exact value depends on the spectral flatness of k; tighter
        bounds require a "unit-spectrum" key normalisation which we
        don't apply by default. The exact-inverse contract above pins
        the algebra rigorously.
        """
        torch.manual_seed(3)
        d = 256
        k = torch.randn(d)
        v = torch.randn(d)
        m = hrr_bind(k, v)
        v_hat = hrr_unbind(m, k)
        cos = torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), v_hat.unsqueeze(0)
        ).item()
        assert cos > 0.5, (
            f"approximate retrieval cos={cos:.4f}; expected > 0.5 for "
            f"random Gaussian keys at d=256 (well above noise floor)."
        )

    def test_unbinding_with_wrong_key_returns_noise(self):
        """Querying with an unrelated key must NOT return the stored value."""
        torch.manual_seed(4)
        d = 256
        k1 = torch.randn(d) / math.sqrt(d)
        k2 = torch.randn(d) / math.sqrt(d)
        v = torch.randn(d)
        m = hrr_bind(k1, v)
        # Unbind with the wrong key — should be near-orthogonal to v.
        wrong = hrr_unbind(m, k2)
        cos = torch.nn.functional.cosine_similarity(
            v.unsqueeze(0), wrong.unsqueeze(0)
        ).item()
        assert abs(cos) < 0.2, f"wrong-key retrieval cos={cos:.4f}, expected ≈0"

    def test_superposition_retrieval_better_than_random(self):
        """Plate Theorem 3.1: under superposition the signal still
        dominates noise. With d=2048 and N=4 (light load), median
        retrieval cos must clear 0.3 — far above random (cos ≈ 0).

        For tighter retrieval the literature recommends "unit-spectrum"
        keys (random phases, unit magnitudes) which we don't enforce by
        default; the HRRMemory module's optional ``normalize_keys`` flag
        improves things further by spatial normalisation.
        """
        torch.manual_seed(5)
        d, N = 2048, 4
        keys = [torch.randn(d) for _ in range(N)]
        vals = [torch.randn(d) for _ in range(N)]
        m = hrr_superpose(*zip(keys, vals))
        cosines = []
        for k, v in zip(keys, vals):
            v_hat = hrr_unbind(m, k)
            cosines.append(
                torch.nn.functional.cosine_similarity(
                    v.unsqueeze(0), v_hat.unsqueeze(0)
                ).item()
            )
        median_cos = sorted(cosines)[N // 2]
        assert median_cos > 0.3, (
            f"median retrieval cos={median_cos:.3f}; expected > 0.3 "
            f"for d=2048, N=4 (light superposition load)."
        )

    def test_dim_mismatch_in_bind_raises(self):
        with pytest.raises(ValueError, match="last dims"):
            hrr_bind(torch.randn(8), torch.randn(16))

    def test_dim_mismatch_in_unbind_raises(self):
        with pytest.raises(ValueError, match="last dims"):
            hrr_unbind(torch.randn(8), torch.randn(16))

    def test_empty_superpose_raises(self):
        with pytest.raises(ValueError):
            hrr_superpose()


# ──────────────────────────────────────────────────────────────────────
# 2. HRRMemory module — shape / behaviour contracts
# ──────────────────────────────────────────────────────────────────────


class TestHRRMemoryModule:
    def test_forward_preserves_shape(self):
        mod = HRRMemory(d_model=32, d_memory=64)
        x = torch.randn(2, 7, 32)
        y = mod(x)
        assert y.shape == x.shape

    def test_forward_runs_in_batch(self):
        mod = HRRMemory(d_model=16, d_memory=32)
        x = torch.randn(4, 11, 16)
        y = mod(x)
        assert y.shape == (4, 11, 16)
        assert torch.isfinite(y).all()

    def test_output_actually_depends_on_input(self):
        """If we change one token, the output for OTHER tokens must
        change too — because they all query the same per-batch memory.
        """
        torch.manual_seed(6)
        mod = HRRMemory(d_model=16, d_memory=64)
        mod.eval()
        x1 = torch.randn(1, 5, 16)
        x2 = x1.clone()
        x2[0, 0] = torch.randn(16)  # perturb only token 0
        y1 = mod(x1)
        y2 = mod(x2)
        # Token 4's output must change because the shared memory shifted.
        diff = (y1[0, 4] - y2[0, 4]).abs().max().item()
        assert diff > 1e-5, (
            f"perturbing token 0 didn't propagate to token 4 (diff={diff}); "
            "HRR memory not actually being read."
        )

    def test_wrong_input_shape_raises(self):
        mod = HRRMemory(d_model=16, d_memory=32)
        with pytest.raises(ValueError, match=r"\(B, T, D\)"):
            mod(torch.randn(5, 16))  # missing T axis

    def test_parameters_receive_gradients(self):
        mod = HRRMemory(d_model=16, d_memory=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        loss = mod(x).pow(2).sum()
        loss.backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum() > 0, f"{name} grad is identically zero"

    def test_normalize_keys_off_still_works(self):
        mod = HRRMemory(d_model=16, d_memory=32, normalize_keys=False)
        x = torch.randn(1, 4, 16)
        y = mod(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
