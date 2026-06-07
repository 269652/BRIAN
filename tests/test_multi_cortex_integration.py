# -*- coding: utf-8 -*-
"""End-to-end Multi-Trunk-V2 integration test (Step 4 of the 4-step
HuggingFace wiring sequence).

Covers the full pipeline from DSL source text → parsed config →
constructed `BRIANHarness` → live `MultiCortexEnsemble` forward pass.
Steps 1-3 each test their layer in isolation; this file is the
acceptance test that proves all three layers are properly connected.

Two flavours:
  * Stub backend — exercised by default; fast (≈ 0.5 s), no network.
  * GPT-2 backend — gated behind `MULTI_CORTEX_LIVE_HF=1` because it
    downloads ≈ 1.6 GB on first run and requires `transformers`.
    Skipped in CI; runs locally when the env var is set.

Why both:
  * Stub proves the WIRING is correct.
  * Live proves the WIRING + the HuggingFace BACKEND together work.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.training_config import parse_training_config


# Synthetic but production-shaped DSL snippet. Uses every multi_cortex
# field documented in arch.neuro Section 5.7.
DSL_STUB_BODY = """
    batch_size: 4,
    seq_len: 16,
    multi_cortex: {
        enabled:             true,
        n_cortices:          4,
        domains:             ["math", "code", "chat", "general"],
        weights:             "stub",
        freeze_weights:      true,
        lexical_bias_weight: 2.0,
        bema_tau:            0.5,
        router_d_model:      32
    }
"""

DSL_GPT2_BODY = """
    multi_cortex: {
        enabled:             true,
        n_cortices:          4,
        domains:             ["math", "code", "chat", "general"],
        weights:             "gpt2",
        freeze_weights:      true,
        lexical_bias_weight: 2.0,
        bema_tau:            0.5,
        router_d_model:      768
    }
"""

VOCAB = 64
D_SEM = 32


def _trivial_circuit() -> nn.Module:
    class Trivial(nn.Module):
        def forward(self, sensory_input, nt_levels=None):
            return {"motor": torch.zeros(sensory_input.shape[0], D_SEM)}
    return Trivial()


# ──────────────────────────────────────────────────────────────────────
# Stub flavour — runs by default, no network
# ──────────────────────────────────────────────────────────────────────

class TestDslToHarnessEndToEnd:
    """The headline integration test: a DSL snippet matching the
    arch.neuro multi_cortex block must end up as a live, forward-able
    `MultiCortexEnsemble` on a constructed harness."""

    def test_full_pipeline_with_stub_backend(self):
        from neuroslm.cortex import MultiCortexEnsemble
        from neuroslm.harness import BRIANHarness

        # 1. DSL parse
        cfg = parse_training_config(DSL_STUB_BODY)
        assert cfg.multi_cortex.enabled is True
        assert cfg.multi_cortex.weights == "stub"

        # 2. Harness construction (the wiring step)
        h = BRIANHarness(
            _trivial_circuit(), vocab_size=VOCAB,
            d_sem=D_SEM, training_config=cfg,
        )
        assert isinstance(h.multi_cortex, MultiCortexEnsemble)

        # 3. Forward through the ensemble
        B, T = 2, 8
        ids = torch.randint(0, VOCAB, (B, T))
        out = h.multi_cortex(ids)
        assert out.shape == (B, T, cfg.multi_cortex.router_d_model)
        # Output must be finite — proves the routing softmax actually fired.
        assert torch.isfinite(out).all(), \
            "ensemble output contains NaN/Inf — routing math broke"

    def test_ensemble_params_are_in_harness_optimizer_visibility(self):
        """Step 1+2+3 stack: the parameters reachable from
        `harness.parameters()` must include the ensemble's — otherwise
        a real `train_step` would skip them entirely."""
        from neuroslm.harness import BRIANHarness
        cfg = parse_training_config(DSL_STUB_BODY)
        h = BRIANHarness(
            _trivial_circuit(), vocab_size=VOCAB,
            d_sem=D_SEM, training_config=cfg,
        )
        harness_param_ids = {id(p) for p in h.parameters()}
        ensemble_param_ids = {id(p) for p in h.multi_cortex.parameters()}
        # Strict subset relationship: every ensemble param is a harness param.
        assert ensemble_param_ids.issubset(harness_param_ids), \
            f"{len(ensemble_param_ids - harness_param_ids)} ensemble params " \
            "are invisible to harness.parameters()"

    def test_legacy_dsl_without_multi_cortex_block_still_works(self):
        """Backward-compat: an arch.neuro with NO multi_cortex block
        must still parse + construct a harness — just with
        `multi_cortex = None`."""
        from neuroslm.harness import BRIANHarness
        body = """
            batch_size: 4,
            seq_len: 16,
            optimizer: "adamw"
        """
        cfg = parse_training_config(body)
        assert cfg.multi_cortex.enabled is False
        h = BRIANHarness(
            _trivial_circuit(), vocab_size=VOCAB,
            d_sem=D_SEM, training_config=cfg,
        )
        assert h.multi_cortex is None


# ──────────────────────────────────────────────────────────────────────
# GPT-2 flavour — gated, hits HuggingFace Hub (≈ 1.6 GB download)
# ──────────────────────────────────────────────────────────────────────

LIVE_HF = os.environ.get("MULTI_CORTEX_LIVE_HF") == "1"


@pytest.mark.skipif(
    not LIVE_HF,
    reason="set MULTI_CORTEX_LIVE_HF=1 to run the live HF download test "
           "(downloads ≈ 1.6 GB on first invocation)"
)
class TestDslToHarnessWithGpt2Weights:
    """End-to-end test of the `weights: \"gpt2\"` path. Downloads from
    HuggingFace Hub on first run; subsequent runs use the cache. Lives
    behind an env-var so CI never triggers the download."""

    def test_gpt2_ensemble_builds_from_dsl(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed — `pip install transformers`")

        from neuroslm.cortex import GPT2SubCortex, MultiCortexEnsemble
        from neuroslm.harness import BRIANHarness

        cfg = parse_training_config(DSL_GPT2_BODY)
        h = BRIANHarness(
            _trivial_circuit(), vocab_size=50257,  # GPT-2 vocab
            d_sem=768, training_config=cfg,
        )
        assert isinstance(h.multi_cortex, MultiCortexEnsemble)
        # Every sub-cortex must be a real GPT-2 backbone
        for sc in h.multi_cortex.sub_cortices:
            assert isinstance(sc, GPT2SubCortex), \
                f"sub-cortex {sc.name} is {type(sc).__name__}, not GPT2SubCortex"

    def test_gpt2_ensemble_forward_produces_finite_logits(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        from neuroslm.harness import BRIANHarness

        cfg = parse_training_config(DSL_GPT2_BODY)
        h = BRIANHarness(
            _trivial_circuit(), vocab_size=50257,
            d_sem=768, training_config=cfg,
        )
        B, T = 1, 4
        ids = torch.randint(0, 50257, (B, T))
        with torch.no_grad():
            out = h.multi_cortex(ids)
        assert out.shape == (B, T, 768)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
