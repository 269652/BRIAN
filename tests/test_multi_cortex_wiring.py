# -*- coding: utf-8 -*-
"""TDD acceptance suite — `BRIANHarness` wires `MultiCortexEnsemble`.

Step 3 of the 4-step Multi-Trunk-V2 ⇒ HuggingFace wiring sequence.

This is the test that closes the audit gap: the arch.neuro block
parses correctly (Step 1) and `transformers` is installable (Step 2),
but **nothing actually constructs the ensemble at training time**
unless we wire the harness. This file proves the wiring works.

Tests use `weights="stub"` exclusively — `MultiCortexEnsemble` is built
from `StubSubCortex`s with no network access. Step 4 will add a
gated end-to-end test that exercises the HF download path.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

VOCAB = 64
D_SEM = 32


@pytest.fixture
def disabled_cfg() -> TrainingConfig:
    """Legacy config — multi_cortex disabled. Harness must NOT build it."""
    return TrainingConfig()  # defaults: multi_cortex.enabled = False


@pytest.fixture
def stub_cfg() -> TrainingConfig:
    """Production-shaped config but using stub weights (no HF download)."""
    cfg = TrainingConfig()
    cfg.multi_cortex = MultiCortexConfig(
        enabled=True,
        n_cortices=4,
        domains=["math", "code", "chat", "general"],
        weights="stub",
        freeze_weights=True,
        lexical_bias_weight=2.0,
        bema_tau=0.5,
        router_d_model=D_SEM,
    )
    return cfg


@pytest.fixture
def trivial_circuit() -> nn.Module:
    """A no-op circuit good enough to construct BRIANHarness."""
    class Trivial(nn.Module):
        def forward(self, sensory_input, nt_levels=None):
            B = sensory_input.shape[0]
            return {"motor": torch.zeros(B, D_SEM)}
    return Trivial()


# ──────────────────────────────────────────────────────────────────────
# Disabled path — backwards compatibility
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexDisabledByDefault:
    """When `cfg.multi_cortex.enabled = False`, the harness must NOT build
    an ensemble — preserving bit-for-bit reproducibility of legacy runs."""

    def test_harness_init_does_not_build_ensemble(
        self, trivial_circuit, disabled_cfg
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=disabled_cfg)
        # The attribute must exist for safe access elsewhere, but be None.
        assert hasattr(h, "multi_cortex")
        assert h.multi_cortex is None

    def test_from_language_model_does_not_build_ensemble(self, disabled_cfg):
        from neuroslm.harness import BRIANHarness
        lm = nn.Sequential(nn.Embedding(VOCAB, D_SEM),
                           nn.Linear(D_SEM, VOCAB))
        h = BRIANHarness.from_language_model(
            language_model=lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=disabled_cfg,
        )
        assert hasattr(h, "multi_cortex")
        assert h.multi_cortex is None


# ──────────────────────────────────────────────────────────────────────
# Enabled path — STUB weights (offline-safe)
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexStubWiring:
    """With `weights="stub"` the harness must build a real
    MultiCortexEnsemble — no network, but a fully connected routing
    graph so the rest of the pipeline can use `harness.multi_cortex`."""

    def test_harness_init_builds_ensemble_when_enabled(
        self, trivial_circuit, stub_cfg
    ):
        from neuroslm.cortex import MultiCortexEnsemble
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=stub_cfg)
        assert h.multi_cortex is not None
        assert isinstance(h.multi_cortex, MultiCortexEnsemble)

    def test_ensemble_has_correct_number_of_sub_cortices(
        self, trivial_circuit, stub_cfg
    ):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=stub_cfg)
        assert len(h.multi_cortex.sub_cortices) == 4

    def test_ensemble_domains_match_config(self, trivial_circuit, stub_cfg):
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=stub_cfg)
        names = [sc.domain for sc in h.multi_cortex.sub_cortices]
        assert names == ["math", "code", "chat", "general"]

    def test_ensemble_parameters_are_registered_for_optimizer(
        self, trivial_circuit, stub_cfg
    ):
        """If the ensemble's params aren't visible from `harness.parameters()`
        the optimizer will never train them. This was the silent-failure
        mode of the original PR — fail loudly here."""
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=stub_cfg)
        harness_params = {id(p) for p in h.parameters()}
        ensemble_params = list(h.multi_cortex.parameters())
        assert len(ensemble_params) > 0, "ensemble has no params"
        # Every ensemble param must be reachable from the harness.
        missing = [p for p in ensemble_params
                   if id(p) not in harness_params]
        assert not missing, (
            f"{len(missing)} ensemble params not in harness.parameters(); "
            "MultiCortexEnsemble was not registered as a child Module"
        )

    def test_ensemble_forward_smokes_through(
        self, trivial_circuit, stub_cfg
    ):
        """The wired ensemble must accept the harness's standard
        (input_ids, hidden_state) interface without crashing."""
        from neuroslm.harness import BRIANHarness
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=stub_cfg)
        B, T = 2, 8
        input_ids = torch.randint(0, VOCAB, (B, T))
        hidden = torch.zeros(B, T, D_SEM)
        out = h.multi_cortex(input_ids=input_ids, hidden=hidden)
        # MultiCortexEnsemble returns a tensor (B, T, d_target)
        assert out.shape == (B, T, D_SEM), \
            f"expected (B, T, d_target)=(2,8,{D_SEM}), got {tuple(out.shape)}"

    def test_from_language_model_also_wires_ensemble(self, stub_cfg):
        """The alternate construction path used by the DSL transformer LM
        must wire the ensemble identically — otherwise `brian train` and
        `brian eval` produce divergent graphs."""
        from neuroslm.cortex import MultiCortexEnsemble
        from neuroslm.harness import BRIANHarness
        lm = nn.Sequential(nn.Embedding(VOCAB, D_SEM),
                           nn.Linear(D_SEM, VOCAB))
        h = BRIANHarness.from_language_model(
            language_model=lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=stub_cfg,
        )
        assert isinstance(h.multi_cortex, MultiCortexEnsemble)
        assert len(h.multi_cortex.sub_cortices) == 4


# ──────────────────────────────────────────────────────────────────────
# Custom-domain path — proves the harness honours the config, not a
# hard-coded constant
# ──────────────────────────────────────────────────────────────────────

class TestMultiCortexHonoursConfig:
    """Mutating the config must change what the harness builds — this
    guards against a regression where the harness ignores the config
    and always builds the default 4-cortex GPT-2 ensemble."""

    def test_two_cortex_config_yields_two_cortices(self, trivial_circuit):
        from neuroslm.harness import BRIANHarness
        cfg = TrainingConfig()
        cfg.multi_cortex = MultiCortexConfig(
            enabled=True, n_cortices=2, weights="stub",
            domains=["alpha", "beta"], router_d_model=D_SEM,
        )
        h = BRIANHarness(trivial_circuit, vocab_size=VOCAB,
                         d_sem=D_SEM, training_config=cfg)
        assert len(h.multi_cortex.sub_cortices) == 2
        assert [sc.domain for sc in h.multi_cortex.sub_cortices] \
            == ["alpha", "beta"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
