# -*- coding: utf-8 -*-
"""Memory-safety contracts for the harness logit-space fusion + loss path.

Root-cause record
==================

A recurring class of CUDA OOM on the SmolLM MoE deploys traces to a single
defect: a full ``(B, T, V)`` logit tensor — 3.07 GiB at B=16 T=2048
V=50257 in bf16 — is silently **promoted to fp32** (6.14 GiB) at several
points in ``BRIANHarness.forward`` / ``compute_loss``:

  1. **Fusion** (``harness.py`` additive_correction / logits_mixture):
     ``alpha_eff`` is ``sigmoid(self.cortex_mix_logit)`` — a scalar derived
     from an fp32 ``nn.Parameter``. The expression ``alpha_eff * logits``
     type-promotes the bf16 ``(B,T,V)`` trunk logits to fp32, doubling the
     allocation. This is the exact site of the 2026-06-23 OOM
     ("Tried to allocate 6.14 GiB" at the fusion add).

  2. **Return** (``return logits.float()``): forces the full fused tensor
     to fp32 even under bf16 autocast, so a 6.14 GiB fp32 tensor lives
     through the entire loss computation.

  3. **KL per-pathway CE** (``_cortex_fusion_aux_step``): two
     ``F.cross_entropy(lm_logits.detach().float().reshape(-1, V), ...)``
     calls each materialise a fp32 ``(B*T, V)`` tensor = 6.14 GiB transient.

  4. **DAR per-sample CE** (``compute_loss``): a full
     ``F.cross_entropy(flat_l, flat_t, reduction="none")`` over the whole
     ``(B*T, V)`` tensor — another fp32 spike.

The architectural fix makes the overflow impossible by design rather than
patching each site:

  * ``BRIANHarness._fuse_logits`` casts ``alpha_eff`` to the logit dtype
    BEFORE the multiply, so the ``(B,T,V)`` tensor never promotes to fp32.
  * ``forward`` returns logits in their compute dtype (bf16 under autocast,
    fp32 in eval) — no blanket ``.float()`` on the full tensor.
  * ``BRIANHarness._chunked_flat_ce`` computes cross-entropy over the
    flattened token dim in ``chunk``-sized slices, so the internal fp32
    softmax/grad is bounded to ``(chunk, V)`` instead of ``(B*T, V)``.
    Used for the KL per-pathway CE and the DAR per-sample CE.

These contracts pin the dtype behaviour and the numerical equivalence of
the chunked path, independent of the GPU it runs on.
"""
from __future__ import annotations

from unittest import mock

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
import torch.nn.functional as F  # noqa: E402


VOCAB = 64
D_SEM = 32


# ──────────────────────────────────────────────────────────────────────
# Fixtures — reuse the stub-fusion harness construction (no HF loading)
# ──────────────────────────────────────────────────────────────────────


class _FakeDSLLM(nn.Module):
    """Minimal trunk LM with fp32 params — matches the pattern in
    ``test_expert_correction_fusion.py``. Under bf16 autocast its
    ``F.linear`` head produces bf16 logits (the realistic training dtype)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_SEM, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self._last_hidden = None
        self._last_h_motor = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]
        self._last_hidden = h
        self._last_h_motor = h
        return F.linear(h, self.lm_head)


def _build_fusion_harness(fusion_mode: str = "additive_correction",
                          fusion_init: float = 0.4, seed: int = 0):
    from neuroslm.harness import BRIANHarness
    from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig

    cfg = TrainingConfig()
    mc = MultiCortexConfig()
    mc.enabled = True
    mc.n_cortices = 2
    mc.domains = ["general", "code"]
    mc.weights = "stub"
    mc.freeze_weights = False
    mc.lexical_bias_weight = 0.0
    mc.bema_tau = 0.5
    mc.router_d_model = D_SEM
    mc.fusion_mode = fusion_mode
    mc.fusion_init = fusion_init
    cfg.multi_cortex = mc
    lm = _FakeDSLLM(vocab=VOCAB, d_model=D_SEM, seed=seed)
    torch.manual_seed(seed)
    return BRIANHarness.from_language_model(
        language_model=lm, vocab_size=VOCAB, d_sem=D_SEM, training_config=cfg,
    )


# ──────────────────────────────────────────────────────────────────────
# _fuse_logits — the dtype-promotion fix
# ──────────────────────────────────────────────────────────────────────


class TestFuseLogitsDtype:
    """``_fuse_logits`` must never promote the (B,T,V) tensor to fp32 just
    because ``alpha_eff`` is an fp32 scalar."""

    def test_helper_exists(self):
        from neuroslm.harness import BRIANHarness
        assert callable(getattr(BRIANHarness, "_fuse_logits", None)), (
            "BRIANHarness must expose a _fuse_logits static helper"
        )

    def test_additive_keeps_bf16_with_fp32_alpha(self):
        from neuroslm.harness import BRIANHarness
        lm = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        cx = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        a = torch.tensor(0.3, dtype=torch.float32)  # sigmoid(fp32 param)
        out = BRIANHarness._fuse_logits(lm, cx, a, "additive_correction")
        assert out.dtype == torch.bfloat16, (
            f"additive fusion must keep bf16 (fp32 alpha must not promote "
            f"the (B,T,V) logits); got {out.dtype}"
        )

    def test_mixture_keeps_bf16_with_fp32_alpha(self):
        from neuroslm.harness import BRIANHarness
        lm = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        cx = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        a = torch.tensor(0.3, dtype=torch.float32)
        out = BRIANHarness._fuse_logits(lm, cx, a, "logits_mixture")
        assert out.dtype == torch.bfloat16, (
            f"mixture fusion must keep bf16; got {out.dtype}"
        )

    def test_additive_value_correct(self):
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(0)
        lm = torch.randn(2, 4, 8)
        cx = torch.randn(2, 4, 8)
        a = torch.tensor(0.3)
        out = BRIANHarness._fuse_logits(lm, cx, a, "additive_correction")
        expected = cx + 0.3 * lm
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_mixture_value_correct(self):
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(0)
        lm = torch.randn(2, 4, 8)
        cx = torch.randn(2, 4, 8)
        a = torch.tensor(0.3)
        out = BRIANHarness._fuse_logits(lm, cx, a, "logits_mixture")
        expected = 0.7 * lm + 0.3 * cx
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_additive_detaches_cortex(self):
        """additive_correction must cut the autograd path through cortex."""
        from neuroslm.harness import BRIANHarness
        lm = torch.randn(2, 4, 8, requires_grad=True)
        cx = torch.randn(2, 4, 8, requires_grad=True)
        a = torch.tensor(0.3)
        out = BRIANHarness._fuse_logits(lm, cx, a, "additive_correction")
        g = torch.autograd.grad(out.sum(), cx, allow_unused=True)[0]
        assert g is None, "cortex must be detached in additive_correction mode"

    def test_alpha_broadcasts_per_sample(self):
        """alpha_eff of shape (B,1,1) must broadcast cleanly over (B,T,V)."""
        from neuroslm.harness import BRIANHarness
        lm = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        cx = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        a = torch.rand(3, 1, 1, dtype=torch.float32)
        out = BRIANHarness._fuse_logits(lm, cx, a, "logits_mixture")
        assert out.shape == (3, 4, 8)
        assert out.dtype == torch.bfloat16


# ──────────────────────────────────────────────────────────────────────
# _chunked_flat_ce — bounded cross-entropy
# ──────────────────────────────────────────────────────────────────────


class TestChunkedFlatCE:
    """``_chunked_flat_ce`` must equal a full ``F.cross_entropy`` while only
    ever materialising a ``(chunk, V)`` fp32 intermediate."""

    def test_helper_exists(self):
        from neuroslm.harness import BRIANHarness
        assert callable(getattr(BRIANHarness, "_chunked_flat_ce", None)), (
            "BRIANHarness must expose a _chunked_flat_ce static helper"
        )

    def test_mean_matches_full(self):
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(0)
        logits = torch.randn(4, 1000, 64)  # N=4000 > chunk
        targets = torch.randint(0, 64, (4, 1000))
        ref = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
        got = BRIANHarness._chunked_flat_ce(
            logits, targets, reduction="mean", chunk=1024)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)

    def test_none_matches_full(self):
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(1)
        logits = torch.randn(4, 1000, 64)
        targets = torch.randint(0, 64, (4, 1000))
        ref = F.cross_entropy(
            logits.reshape(-1, 64), targets.reshape(-1), reduction="none")
        got = BRIANHarness._chunked_flat_ce(
            logits, targets, reduction="none", chunk=1024)
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)

    def test_sum_matches_full(self):
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(2)
        logits = torch.randn(4, 1000, 64)
        targets = torch.randint(0, 64, (4, 1000))
        ref = F.cross_entropy(
            logits.reshape(-1, 64), targets.reshape(-1), reduction="sum")
        got = BRIANHarness._chunked_flat_ce(
            logits, targets, reduction="sum", chunk=1024)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-3)

    def test_small_input_single_chunk(self):
        """N <= chunk must take the direct path and still be correct."""
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(3)
        logits = torch.randn(2, 8, 64)
        targets = torch.randint(0, 64, (2, 8))
        ref = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
        got = BRIANHarness._chunked_flat_ce(
            logits, targets, reduction="mean", chunk=1024)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

    def test_chunking_actually_happens(self):
        """For N > chunk the helper must issue MULTIPLE F.cross_entropy
        calls — proof that no full (B*T, V) tensor is built at once."""
        from neuroslm.harness import BRIANHarness
        logits = torch.randn(4, 1000, 64)  # N=4000
        targets = torch.randint(0, 64, (4, 1000))
        real_ce = F.cross_entropy
        calls = {"n": 0}

        def _counting_ce(*args, **kwargs):
            calls["n"] += 1
            return real_ce(*args, **kwargs)

        with mock.patch.object(F, "cross_entropy", _counting_ce):
            BRIANHarness._chunked_flat_ce(
                logits, targets, reduction="mean", chunk=1024)
        # ceil(4000 / 1024) = 4 chunks
        assert calls["n"] >= 4, (
            f"expected >=4 chunked CE calls for N=4000 chunk=1024; "
            f"got {calls['n']} (chunking not happening → full fp32 spike)"
        )

    def test_bf16_input_returns_finite_fp32_scalar(self):
        """bf16 logits must produce a finite loss; F.cross_entropy upcasts
        per chunk so the result is fp32 and the fp32 spike is bounded."""
        from neuroslm.harness import BRIANHarness
        torch.manual_seed(4)
        logits = (torch.randn(4, 1000, 64)).to(torch.bfloat16)
        targets = torch.randint(0, 64, (4, 1000))
        got = BRIANHarness._chunked_flat_ce(
            logits, targets, reduction="mean", chunk=1024)
        assert torch.isfinite(got).all()
        assert got.ndim == 0


# ──────────────────────────────────────────────────────────────────────
# Integration — forward under bf16 autocast must not produce fp32 (B,T,V)
# ──────────────────────────────────────────────────────────────────────


class TestForwardDtypeUnderAutocast:
    """End-to-end: the fused logits returned by ``forward`` must stay in the
    autocast compute dtype (bf16), not get promoted/forced to fp32."""

    @pytest.mark.parametrize("mode", ["additive_correction", "logits_mixture"])
    def test_fused_logits_stay_bf16_under_autocast(self, mode):
        h = _build_fusion_harness(fusion_mode=mode, seed=5)
        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.amp.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
            out = h.forward(ids)
        assert out.dtype == torch.bfloat16, (
            f"under bf16 autocast the fused (B,T,V) logits must stay bf16 — a "
            f"fp32 alpha_eff scalar or a blanket .float() promotes them to "
            f"fp32, doubling 3GiB→6GiB (the OOM at the fusion add); "
            f"got {out.dtype} in mode={mode}"
        )

    def test_eval_without_autocast_returns_fp32(self):
        """Without autocast (eval / CPU inference) the fp32 trunk params
        produce fp32 logits — the return must NOT downcast them."""
        h = _build_fusion_harness(fusion_mode="additive_correction", seed=6)
        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            out = h.forward(ids)
        assert out.dtype == torch.float32, (
            f"in eval (no autocast) the logits must remain fp32; got {out.dtype}"
        )
