# -*- coding: utf-8 -*-
"""TDD: cortex_pre_head_norm kills the GPT-2 rogue-dimension anisotropy.

Background
----------
Operator's GPU run on rcc_bowtie_30m_p4 (June 8 2026, Colab T4) shows:

    step  20 | loss 13.84 | ppl 1024606 | ...
    step  40 | loss 13.57 | ppl  780229 | ...
    step 240 | loss 12.81 | ppl  364482 | ...

``ln(50257) ≈ 10.82``. The model is making *confidently wrong*
predictions: loss exceeds the uniform-random baseline by ~3 nats and
perplexity exceeds the vocab size by 20×.

Root cause (empirically validated by
``scripts/diagnose_catastrophic_loss.py``):

  1. GPT-2's residual stream has a well-known rogue dimension whose
     standard deviation is ~80× the median per-dim std (Timkey & van
     Schijndel 2021, "All Bark and No Bite"). Even after the model's
     own ``ln_f``, ``last_hidden_state.std()`` is dominated by that
     single direction.

  2. ``MultiCortexEnsemble.projections[name]`` is a plain
     ``nn.Linear(768, d_sem)`` with default ``kaiming_uniform_`` init
     and **no LayerNorm**. The projection preserves the anisotropy:
     a random linear map of an anisotropic vector is still anisotropic.

  3. ``BRIANHarness.cortex_lm_head`` is tied to ``language_model.embed``
     (std=0.02). The tied head maps the projected anisotropic features
     to logits via ``cortex_h @ embed.T``. Whichever vocab tokens
     happen to align with the rogue direction in embed-space get
     logit-spikes of magnitude 5-10. Softmax interprets that as
     "I'm 99% sure the next token is X" — and gets it wrong nearly
     every time.

The fix: insert a ``LayerNorm(d_sem)`` between the projection and the
tied head. LayerNorm normalises per-token across the feature dim, so
the rogue dimension's variance is rescaled to be commensurate with
the rest. Cortex logits then sit at sane magnitudes from step 0 and
the initial CE matches the LM-trunk baseline (≈ ln(vocab)).

Contracts pinned by this suite
------------------------------

  A. ``cortex_pre_head_norm`` attribute exists and is a ``nn.LayerNorm``
     when fusion is enabled.
  B. It is registered as a child module so its γ/β are in
     ``harness.parameters()`` and the optimizer trains them.
  C. The forward path applies it before ``cortex_lm_head``: the
     standard deviation of ``cortex_logits`` at init must be bounded
     even when the cortex hidden has rogue-dim anisotropy 50× the
     median per-dim std.
  D. Initial cross-entropy on random labels must be within +0.5 nats
     of ``ln(vocab)`` when the cortex hidden is anisotropic — i.e. the
     pre-head norm is sufficient to suppress the catastrophic-loss
     regime entirely.
  E. The norm is a no-op when ``fusion_mode='off'`` (no head to
     normalise into) — back-compat with the disabled-fusion path.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig


VOCAB = 256          # bigger than D_SEM so ln(VOCAB) > 0 by a healthy margin
D_SEM = 32


class _FakeDSLLM(nn.Module):
    """Mirrors the DSL LM surface used by ``BRIANHarness.from_language_model``.

    Exposes ``self.embed`` as ``nn.Parameter(vocab, d_model)`` so the
    harness can tie ``cortex_lm_head`` to it (Contract 6 of the
    sister suite ``test_multi_cortex_logits_wired``).
    """

    def __init__(self, vocab: int = VOCAB, d_model: int = D_SEM,
                 seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self._last_hidden = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]
        self._last_hidden = h
        return F.linear(h, self.lm_head)


@pytest.fixture
def fake_lm() -> _FakeDSLLM:
    return _FakeDSLLM(seed=0)


@pytest.fixture
def stub_fusion_cfg() -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True,
        n_cortices=4,
        domains=["math", "code", "chat", "general"],
        weights="stub",
        freeze_weights=False,
        lexical_bias_weight=2.0,
        bema_tau=0.5,
        router_d_model=D_SEM,
    )
    return cfg


@pytest.fixture
def disabled_cfg() -> TrainingConfig:
    return TrainingConfig()


def _make_anisotropic_hidden(
    B: int, T: int, D: int, *, rogue_dim: int = 7, rogue_scale: float = 50.0,
) -> torch.Tensor:
    """Construct a (B, T, D) tensor with one rogue dimension whose
    standard deviation is ``rogue_scale`` × the median per-dim std.

    Reproduces the GPT-2 hidden-state pathology in vitro (no transformers
    dependency required for the test to run).
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, D, generator=g)
    x[..., rogue_dim] *= rogue_scale
    return x


# ──────────────────────────────────────────────────────────────────────
# Contract A — cortex_pre_head_norm attribute exists
# ──────────────────────────────────────────────────────────────────────

class TestCortexPreHeadNormStructure:
    """The pre-head norm must exist as a real ``nn.LayerNorm`` whenever
    the fusion head exists. If it's missing, the catastrophic-loss bug
    has regressed."""

    def test_pre_head_norm_attribute_exists_when_fusion_enabled(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        assert hasattr(h, "cortex_pre_head_norm"), (
            "cortex_pre_head_norm attribute missing — the rogue-dimension "
            "fix has been removed or never applied. See "
            "scripts/diagnose_catastrophic_loss.py for the failure mode."
        )
        assert h.cortex_pre_head_norm is not None, (
            "cortex_pre_head_norm is None when multi_cortex fusion is on"
        )
        assert isinstance(h.cortex_pre_head_norm, nn.LayerNorm), (
            f"cortex_pre_head_norm must be nn.LayerNorm (the only invariant "
            f"that kills GPT-2's rogue dim deterministically), got "
            f"{type(h.cortex_pre_head_norm).__name__}"
        )

    def test_pre_head_norm_has_correct_normalised_shape(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        ln = h.cortex_pre_head_norm
        assert ln.normalized_shape == (D_SEM,), (
            f"LayerNorm must normalise over the feature axis (D_SEM={D_SEM}), "
            f"got normalized_shape={ln.normalized_shape}"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract B — pre_head_norm parameters are in the optimizer's view
# ──────────────────────────────────────────────────────────────────────

class TestPreHeadNormRegistered:
    """If the LayerNorm is built but not registered as a child module,
    its γ/β are not in ``harness.parameters()`` and the lazy optimizer
    silently drops them. Pin the registration with parameter identity."""

    def test_pre_head_norm_params_in_harness_parameters(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        ln_param_ids = {id(p) for p in h.cortex_pre_head_norm.parameters()}
        harness_param_ids = {id(p) for p in h.parameters()}
        missing = ln_param_ids - harness_param_ids
        assert not missing, (
            f"{len(missing)}/{len(ln_param_ids)} LayerNorm parameters are "
            "NOT in harness.parameters(). The norm was constructed but not "
            "registered as a child module — γ/β won't be trained."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract C — forward path applies the norm before the tied head
# ──────────────────────────────────────────────────────────────────────

class TestPreHeadNormSuppressesAnisotropy:
    """The whole point of this fix: feeding an anisotropic vector to
    the tied head produces sane-magnitude logits.

    Strategy: synthesise a cortex hidden with a rogue dimension 50×
    the median std (GPT-2 in vitro), then verify that the harness's
    ``cortex_pre_head_norm + cortex_lm_head`` composition produces
    bounded logits. Without the LayerNorm, the rogue dim would punch
    through to logit-spikes of magnitude 5-10."""

    def test_cortex_logits_bounded_on_anisotropic_input(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )
        B, T = 4, 16
        cortex_h_aniso = _make_anisotropic_hidden(B, T, D_SEM)
        assert cortex_h_aniso.std(dim=(0, 1)).max() > 10.0, (
            "test setup broken — synthesised hidden is not anisotropic"
        )

        with torch.no_grad():
            sem_normed = h.cortex_pre_head_norm(cortex_h_aniso)
            cortex_logits = h.cortex_lm_head(sem_normed)

        # LayerNorm normalises ACROSS the feature axis per-token: each
        # (B, T) slot is mapped to mean=0, std=1. Per-token L2 norm is
        # therefore exactly sqrt(D_SEM) regardless of input anisotropy.
        # This is the invariant that kills the rogue dimension at the
        # downstream head: the head sees a bounded-magnitude input
        # instead of a vector dominated by one direction.
        per_token_norm = sem_normed.flatten(0, 1).norm(dim=-1)
        expected = math.sqrt(D_SEM)
        max_dev = (per_token_norm - expected).abs().max().item()
        assert max_dev < 1e-3, (
            f"LayerNorm output per-token L2 norm = {per_token_norm.mean():.4f} "
            f"(want ≈ sqrt(D_SEM) = {expected:.4f}), max deviation = "
            f"{max_dev:.6f}. The norm is not being applied across the "
            "feature axis."
        )

        # Cortex logits should sit at small magnitudes. The tied embed
        # has std≈0.02; the LayerNorm-normalised cortex_h has per-element
        # std≈1; the head output `cortex_h @ embed.T` therefore has
        # std ≈ 0.02 × sqrt(D_SEM) ≈ 0.11 and abs-max well under 1.
        # Without LayerNorm, the rogue-dim spike would blow the abs-max
        # past 5-10 (operator's catastrophic-loss regime).
        max_abs = cortex_logits.abs().max().item()
        assert max_abs < 3.0, (
            f"cortex_logits max|·| = {max_abs:.2f} (want < 3.0). "
            "The rogue dimension is still leaking through the tied head — "
            "LayerNorm in the forward path is not being applied."
        )

    def test_forward_pass_logits_finite_on_anisotropic_input(
        self, fake_lm, stub_fusion_cfg,
    ):
        """End-to-end: a full forward pass through the harness must
        produce finite logits even when the cortex feeds anisotropic
        hidden states. ``stub`` cortex builds isotropic hidden, so this
        test injects the anisotropy by monkey-patching the ensemble's
        forward output post-hoc."""
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )

        original_forward = h.multi_cortex.forward

        def aniso_forward(ids):
            base = original_forward(ids)            # (B, T, d_sem)
            # Inject rogue dim 7 with std 50× the rest.
            out = base.clone()
            out[..., 7] = out[..., 7] * 50.0
            return out

        h.multi_cortex.forward = aniso_forward       # type: ignore[method-assign]

        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            logits = h(ids)

        assert torch.isfinite(logits).all(), (
            "logits contain NaN/Inf after anisotropic cortex injection — "
            "pre-head norm is not in the forward path"
        )
        # Logits should sit at a sane magnitude that softmax can handle.
        assert logits.abs().max() < 50.0, (
            f"logits max|·|={logits.abs().max().item():.2f} (want < 50) — "
            "rogue dim is leaking through into the fused output"
        )


# ──────────────────────────────────────────────────────────────────────
# Contract D — initial CE on anisotropic input stays near ln(vocab)
# ──────────────────────────────────────────────────────────────────────

class TestInitialCrossEntropyNotCatastrophic:
    """The end-user-visible contract: at step 0, with the model untrained
    and the cortex providing anisotropic features, the cross-entropy
    against random labels must stay near ``ln(vocab)``. If CE exceeds
    ``ln(vocab) + 0.5``, the catastrophic-loss bug has regressed and
    GPU runs will display loss > log(vocab) again."""

    def test_initial_ce_under_anisotropic_cortex_within_0p5_nats(
        self, fake_lm, stub_fusion_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_fusion_cfg,
        )

        original_forward = h.multi_cortex.forward

        def aniso_forward(ids):
            base = original_forward(ids)
            out = base.clone()
            out[..., 7] = out[..., 7] * 50.0
            return out

        h.multi_cortex.forward = aniso_forward       # type: ignore[method-assign]

        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (4, 32))
        targets = torch.randint(0, VOCAB, (4, 32))
        with torch.no_grad():
            logits = h(ids)
            ce = F.cross_entropy(
                logits.reshape(-1, VOCAB), targets.reshape(-1)
            ).item()

        baseline = math.log(VOCAB)
        excess = ce - baseline
        assert excess < 0.5, (
            f"CE under anisotropic cortex = {ce:.4f} nats, "
            f"baseline ln({VOCAB}) = {baseline:.4f}, "
            f"excess = {excess:+.4f} nats (want < +0.5). "
            "Catastrophic-loss bug has regressed: cortex is making "
            "confidently wrong predictions at init. Check that "
            "cortex_pre_head_norm is applied BEFORE cortex_lm_head in "
            "BRIANHarness.forward (logits-mixture fusion path)."
        )


# ──────────────────────────────────────────────────────────────────────
# Contract E — back-compat: fusion_mode='off' builds no norm
# ──────────────────────────────────────────────────────────────────────

class TestPreHeadNormBackCompat:
    """When the user disables fusion (either by enabled=False or
    fusion_mode='off'), no pre-head norm should be built — it has no
    head to normalise into and would otherwise hold dangling parameters
    that ``state_dict()`` would serialise but no training signal would
    update."""

    def test_no_pre_head_norm_when_multi_cortex_disabled(
        self, fake_lm, disabled_cfg,
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=disabled_cfg,
        )
        assert getattr(h, "cortex_pre_head_norm", None) is None, (
            "cortex_pre_head_norm must be None when multi_cortex.enabled=False"
        )

    def test_no_pre_head_norm_when_fusion_mode_off(self, fake_lm):
        from neuroslm.harness import BRIANHarness
        cfg = TrainingConfig()
        cfg.multi_cortex = MultiCortexConfig(
            enabled=True,
            n_cortices=4,
            domains=["math", "code", "chat", "general"],
            weights="stub",
            router_d_model=D_SEM,
            fusion_mode="off",
        )
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert getattr(h, "cortex_pre_head_norm", None) is None, (
            "cortex_pre_head_norm must be None when fusion_mode='off'"
        )
