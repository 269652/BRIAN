# -*- coding: utf-8 -*-
"""TDD: STE harness wiring uses _last_hidden (post-norm), not _last_h_motor (pre-norm).

Contract
========
DSLLanguageModel applies a final rmsnorm to block_outs[-1] before projecting
through lm_head:

    block_outs[-1]   → rmsnorm(gamma_f) → h_for_head → lm_head → logits
    _last_h_motor    ← block_outs[-1]                   (pre-norm stash)
    _last_hidden     ← h_for_head                       (post-norm stash)

The STE re-projects through lm_head: F.linear(h_gpe, lm_head).
h_gpe is derived from the RG cascade applied to the input to lm_head.
Therefore the STE's input to the RG cascade must be _last_hidden (post-norm),
not _last_h_motor (pre-norm).

This test pins that contract by:
  1. Setting _last_h_motor to all-zeros (orthogonal to meaningful signal)
  2. Setting _last_hidden to a nonzero tensor (the "real" normed state)
  3. Verifying the STE-produced logits match those computed from _last_hidden,
     not from _last_h_motor.

Run: brian test tests/training/test_ste_harness_norm.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from neuroslm.dsl.training_config import (
    SemanticTurbulenceConfig,
    TrainingConfig,
)


VOCAB = 128
D = 32   # must be even for GPE complex encoding


class _FakeNormedLM(nn.Module):
    """Minimal LM stub with distinct pre-norm and post-norm stashes.

    _last_h_motor  → zeros  (pre-norm: should NOT drive STE projection)
    _last_hidden   → ones   (post-norm: SHOULD drive STE projection)
    lm_head        → identity-ish Parameter so F.linear is non-trivial
    """

    def __init__(self, vocab: int = VOCAB, d: int = D) -> None:
        super().__init__()
        self.lm_head = nn.Parameter(
            torch.randn(vocab, d) * 0.01
        )
        # Fixed stashes — set once, read by the harness STE path
        self._last_h_motor = None    # will be set in reset()
        self._last_h_sensory = None
        self._last_hidden = None

    def reset(self, B: int = 1, T: int = 8) -> torch.Tensor:
        """Populate stashes; return logits the LM 'produced'."""
        pre_norm = torch.zeros(B, T, D)           # _last_h_motor: all-zero
        post_norm = torch.ones(B, T, D) * 0.5     # _last_hidden: nonzero
        self._last_h_motor = pre_norm
        self._last_h_sensory = pre_norm            # sensory = first block
        self._last_hidden = post_norm
        return F.linear(post_norm, self.lm_head)  # (B, T, V)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        return self.reset(B, T)


def _ste_cfg(enabled: bool = True) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.semantic_turbulence = SemanticTurbulenceConfig(
        enabled=enabled,
        n_rg_groups=1,      # minimal cascade
        kolmogorov_init=True,
        gpe_steps=0,        # zero GPE steps → identity (isolates the norm issue)
        gpe_coupling_init=0.0,
        gpe_dt=0.01,
        criticality_target=1.0,
        criticality_weight=0.0,  # no crit loss during test
    )
    return cfg


@pytest.fixture(scope="module")
def fake_lm() -> _FakeNormedLM:
    return _FakeNormedLM()


@pytest.fixture(scope="module")
def harness_cls():
    from neuroslm.harness import BRIANHarness
    return BRIANHarness


class TestSTEUsesPostNormHidden:
    """STE projection must be driven by _last_hidden, not _last_h_motor."""

    def test_ste_enabled_uses_last_hidden_not_motor(self, fake_lm, harness_cls):
        """When _last_h_motor == 0 and _last_hidden == 0.5, STE logits
        must not be zero (i.e. they come from _last_hidden)."""
        cfg = _ste_cfg(enabled=True)
        h = harness_cls.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D,
            training_config=cfg,
        )

        ids = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            logits = h(ids)

        # If the STE used _last_h_motor (zeros), all token logits would be zero.
        # If it uses _last_hidden (0.5 * ones), logits are nonzero.
        assert logits.shape == (1, 8, VOCAB)
        # With gpe_steps=0 (identity GPE) and near-zero init on RG projections,
        # the STE output ≈ _last_hidden itself → F.linear(0.5, lm_head) ≠ 0
        # (lm_head is random init, so all-zero input → zero output is the tell).
        logits_from_zeros = F.linear(
            torch.zeros(1, 8, D), fake_lm.lm_head
        )
        # The STE logits should differ from the all-zeros result
        assert not torch.allclose(logits, logits_from_zeros, atol=1e-6), (
            "STE logits look like they came from _last_h_motor (zeros) — "
            "expected them to come from _last_hidden (0.5 * ones)."
        )

    def test_ste_disabled_not_affected(self, fake_lm, harness_cls):
        """With STE disabled, the LM's own logits pass through unmodified."""
        cfg = _ste_cfg(enabled=False)
        h = harness_cls.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D,
            training_config=cfg,
        )
        assert h._ste_rg is None

    def test_ste_metrics_populated_when_enabled(self, fake_lm, harness_cls):
        """After a forward pass with STE enabled, ste_rho and ste_sigma
        must be present in the harness metrics dict."""
        cfg = _ste_cfg(enabled=True)
        h = harness_cls.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D,
            training_config=cfg,
        )
        ids = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            h(ids)
        assert "ste_rho" in h._metrics, (
            f"ste_rho missing from metrics; got keys: {list(h._metrics)}"
        )
        assert "ste_sigma" in h._metrics, (
            f"ste_sigma missing from metrics; got keys: {list(h._metrics)}"
        )


class TestSTEGradientIsolation:
    """RG cascade must NOT amplify CE gradient back into the trunk's hidden state.

    The STE should add a *residual delta* to trunk logits (not replace them) and
    must DETACH h_hidden before passing it to the RG cascade.

    Without detach (old/broken): CE → STE_logits → h_gpe → h_rg → (scale_proj·I) → h_hidden
    With scale_proj = c·I the RG Jacobian is (1+c)·I, so grad(h_hidden) is (1+c)× too large.

    With detach (new/correct): CE → trunk_logits → h_hidden (direct lm_head^T path only,
    no RG amplification). STE parameters still get CE gradient via the delta path.
    """

    def test_rg_does_not_amplify_trunk_gradient(self):
        """With scale_proj = c·I (c=3.0), old replacement code amplifies grad 4×.
        New residual code: grad(h_hidden) ≈ lm_head^T @ ∂CE/∂logits (direct, 1×).
        """
        from neuroslm.harness import BRIANHarness

        T = 8

        class _GradTrackingLM(nn.Module):
            """Fake LM exposing _last_hidden as a leaf tensor for grad inspection."""
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Parameter(torch.randn(VOCAB, D) * 0.01)
                # Leaf tensor — gradient will accumulate here after backward.
                self._h_leaf = (torch.ones(1, T, D) * 0.5).requires_grad_(True)
                self._last_h_motor = None
                self._last_h_sensory = None
                self._last_hidden = None

            def forward(self, ids):
                # Expose leaf tensor as _last_hidden so the harness STE can read it.
                self._last_hidden = self._h_leaf
                self._last_h_motor = self._h_leaf.detach()
                self._last_h_sensory = self._h_leaf.detach()
                return F.linear(self._h_leaf, self.lm_head)

        lm = _GradTrackingLM()
        cfg = _ste_cfg(enabled=True)
        h = BRIANHarness.from_language_model(
            language_model=lm, vocab_size=VOCAB, d_sem=D, training_config=cfg
        )

        # Override scale_proj to c·I so the RG Jacobian = (1+c)·I for homogeneous input.
        # With c=3: old code amplifies gradient 4×; new code leaves it at 1×.
        c = 3.0
        with torch.no_grad():
            for proj in h._ste_rg.scale_proj:
                proj.weight.data = torch.eye(D) * c

        ids = torch.zeros(1, T, dtype=torch.long)
        targets = torch.zeros(1, T, dtype=torch.long)

        logits = h(ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        loss.backward()

        actual_grad_norm = lm._h_leaf.grad.norm().item()

        # Expected gradient: only the direct trunk path lm_head^T @ ∂CE/∂logits.
        # Computed from the ACTUAL final logits so G_final is identical.
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)                # (1, T, V)
            one_hot_t = F.one_hot(targets, VOCAB).float()             # (1, T, V)
            G = (probs - one_hot_t) / (1 * T)                        # CE grad per position
            expected_grad = (G @ lm.lm_head.data)                    # (1, T, D)
        expected_grad_norm = expected_grad.norm().item()

        ratio = actual_grad_norm / (expected_grad_norm + 1e-12)
        # New code: ratio ≈ 1.0 (trunk direct path only).
        # Old replacement code: ratio ≈ (1+c) = 4.0 (RG amplification).
        assert ratio < 1.5, (
            f"Trunk gradient amplified {ratio:.2f}× by RG cascade (c={c}). "
            "STE must detach h_hidden before RG and add a residual delta "
            f"(not replace trunk logits). Expected ratio < 1.5."
        )
