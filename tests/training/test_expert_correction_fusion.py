# -*- coding: utf-8 -*-
"""TDD: P1 additive correction, P2 seq_len forwarding, P3 context gate.

P1 — Additive Correction Mode (fusion_mode="additive_correction"):
    Instead of: logits = (1-α)·trunk + α·cortex     (harmful competition)
    Use:        logits = cortex.detach() + α·trunk   (trunk corrects experts)
    The trunk no longer competes with frozen experts; it learns the DELTA.

P2 — Context Length Forwarding:
    DeployConfig carries seq_len / batch_size; LightningConnector forwards
    them as --seq_len / --batch to train_dsl.py so ctx=128 default is
    no longer hardcoded into every Lightning deploy.

P3 — Context-Dependent Fusion Gate:
    α_gate = sigmoid(cortex_mix_logit + W·h_trunk_last)
    Instead of a single global scalar α, the trunk produces a per-sample
    alpha conditioned on its own last hidden state. Expert trust varies
    by context type (prose/code/math).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 64
D_SEM = 32


# ─────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────

class _FakeDSLLM(nn.Module):
    """Minimal trunk LM — stores _last_h_motor for context gate."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_SEM, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self.lm_head = nn.Parameter(torch.randn(vocab, d_model, generator=g) * 0.02)
        self._last_hidden = None
        self._last_h_motor = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed[ids]          # (B, T, D)
        self._last_hidden = h
        self._last_h_motor = h       # P3: harness reads this
        return F.linear(h, self.lm_head)


@pytest.fixture
def fake_lm():
    return _FakeDSLLM(seed=0)


# ─────────────────────────────────────────────────────────────────────────
# P1 — Additive Correction Mode
# ─────────────────────────────────────────────────────────────────────────

class TestAdditiveCorrection:
    """Fusion mode 'additive_correction': logits = cortex.detach() + α·trunk."""

    # ── P1.1: DSL parser accepts the new mode ─────────────────────────

    def test_P1_parser_accepts_additive_correction(self):
        """_parse_multi_cortex must not raise for fusion_mode='additive_correction'."""
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex(
            "enabled: true, "
            "n_cortices: 2, "
            "domains: [\"general\", \"code\"], "
            "weights: \"stub\", "
            'fusion_mode: "additive_correction"'
        )
        assert mc.fusion_mode == "additive_correction"

    def test_P1_logits_mixture_still_valid(self):
        """Existing mode must still parse cleanly."""
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex(
            "enabled: true, "
            "n_cortices: 2, "
            "domains: [\"general\", \"code\"], "
            "weights: \"stub\""
        )
        assert mc.fusion_mode == "logits_mixture"

    def test_P1_unknown_fusion_mode_raises(self):
        """Bad fusion_mode must still raise ValueError."""
        from neuroslm.dsl.training_config import _parse_multi_cortex
        with pytest.raises(ValueError, match="unknown.*fusion_mode"):
            _parse_multi_cortex(
                "enabled: true, "
                "n_cortices: 1, "
                "domains: [\"general\"], "
                "weights: \"stub\", "
                'fusion_mode: "typo_mode"'
            )

    # ── P1.2: harness uses cortex-as-base, trunk-as-delta ─────────────

    def _build_harness(self, fusion_mode: str, fusion_init: float = 0.5,
                       fake_lm_seed: int = 0):
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
        lm = _FakeDSLLM(vocab=VOCAB, d_model=D_SEM, seed=fake_lm_seed)
        return BRIANHarness.from_language_model(
            language_model=lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )

    def test_P1_additive_output_equals_cortex_plus_alpha_times_trunk(self, fake_lm):
        """In additive_correction mode the formula is cortex.detach() + α·trunk."""
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
        mc.fusion_mode = "additive_correction"
        mc.fusion_init = 0.3
        cfg.multi_cortex = mc

        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )

        ids = torch.randint(0, VOCAB, (2, 8))
        with torch.no_grad():
            output = h.forward(ids)

        trunk_logits = h._last_pre_fusion_lm_logits.detach()
        cortex_logits = h._last_pre_fusion_cortex_logits.detach()
        alpha = torch.sigmoid(h.cortex_mix_logit).item()

        expected = (cortex_logits + alpha * trunk_logits).float()
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4,
            msg="additive_correction output must equal cortex + α·trunk")

    def test_P1_additive_differs_from_mixture(self, fake_lm):
        """Additive and mixture modes produce different outputs for the same inputs.

        We verify algebraically from the stashed pre-fusion tensors that the two
        formulas diverge, then confirm the harness outputs match expectations.
        Avoids relying on absolute tolerance on tiny logit magnitudes.
        """
        from neuroslm.harness import BRIANHarness
        from neuroslm.dsl.training_config import MultiCortexConfig, TrainingConfig

        ids = torch.randint(0, VOCAB, (2, 8))

        def _build_and_run(fusion_mode, seed):
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
            mc.fusion_init = 0.4
            cfg.multi_cortex = mc
            lm = _FakeDSLLM(vocab=VOCAB, d_model=D_SEM, seed=seed)
            torch.manual_seed(seed)
            h = BRIANHarness.from_language_model(
                language_model=lm, vocab_size=VOCAB, d_sem=D_SEM,
                training_config=cfg,
            )
            with torch.no_grad():
                out = h.forward(ids)
            return h, out

        h_add, out_add = _build_and_run("additive_correction", seed=7)

        # Recover the pre-fusion tensors from the additive run.
        C = h_add._last_pre_fusion_cortex_logits.detach()
        T = h_add._last_pre_fusion_lm_logits.detach()
        alpha = torch.sigmoid(h_add.cortex_mix_logit).item()

        # The two formulas are algebraically different when C ≠ T:
        #   additive:  C + α·T
        #   mixture:  (1-α)·T + α·C
        # Difference: C·(1-α) + T·(2α-1) — zero only when C=T (or α=0.5 exactly).
        expected_add = (C + alpha * T).float()
        expected_mix = ((1.0 - alpha) * T + alpha * C).float()

        formula_max_diff = (expected_add - expected_mix).abs().max().item()
        assert formula_max_diff > 1e-9, (
            f"Pre-fusion tensors C and T must not be identical; "
            f"max formula diff={formula_max_diff:.2e} — stub cortex may equal trunk"
        )

        # Harness output in additive mode must match the additive formula.
        torch.testing.assert_close(out_add, expected_add, rtol=1e-4, atol=1e-4,
            msg="additive output must equal expected_add = C + α·T")

        # A mixture-mode run should produce expected_mix, not expected_add.
        h_mix, out_mix = _build_and_run("logits_mixture", seed=7)
        C_m = h_mix._last_pre_fusion_cortex_logits.detach()
        T_m = h_mix._last_pre_fusion_lm_logits.detach()
        alpha_m = torch.sigmoid(h_mix.cortex_mix_logit).item()
        expected_mix2 = ((1.0 - alpha_m) * T_m + alpha_m * C_m).float()
        torch.testing.assert_close(out_mix, expected_mix2, rtol=1e-4, atol=1e-4,
            msg="mixture output must equal expected_mix = (1-α)·T + α·C")

    def test_P1_cortex_gradient_blocked_in_additive_mode(self, fake_lm):
        """In additive mode the fused output has no autograd path through cortex_logits.

        The invariant under test: output = cortex_logits.detach() + α·trunk_logits.
        Because of .detach(), torch.autograd.grad(output, cortex_logits) must be None.
        We test this directly using autograd.grad with allow_unused=True.
        """
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
        mc.fusion_mode = "additive_correction"
        mc.fusion_init = 0.3
        cfg.multi_cortex = mc

        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )

        ids = torch.randint(0, VOCAB, (2, 8))

        # Run forward WITHOUT no_grad so the graph is retained.
        output = h.forward(ids)
        cortex_pre = h._last_pre_fusion_cortex_logits

        if cortex_pre is not None and cortex_pre.requires_grad:
            # output = cortex_logits.detach() + α·trunk → no path through cortex_pre
            grad = torch.autograd.grad(
                output.sum(), cortex_pre, allow_unused=True, retain_graph=False
            )[0]
            assert grad is None, (
                "In additive_correction mode, output must NOT be connected to "
                "cortex_logits via autograd (cortex_logits.detach() cuts the path). "
                f"Got grad with max={grad.abs().max().item():.4e}"
            )

        # Positive check: trunk parameters DO receive gradient from the output.
        targets = torch.randint(0, VOCAB, (2, 8))
        output2 = h.forward(ids)
        loss = F.cross_entropy(output2.view(-1, VOCAB), targets.view(-1))
        loss.backward()
        assert fake_lm.lm_head.grad is not None and fake_lm.lm_head.grad.abs().max() > 0, (
            "Trunk lm_head must receive gradient in additive_correction mode"
        )


# ─────────────────────────────────────────────────────────────────────────
# P2 — seq_len and batch_size forwarding through DeployConfig
# ─────────────────────────────────────────────────────────────────────────

class TestSeqLenForwarding:
    """DeployConfig.seq_len / .batch_size must flow through to train command."""

    def test_P2_deploy_config_has_seq_len_field(self):
        """DeployConfig must carry seq_len defaulting to 0."""
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=100)
        assert hasattr(cfg, "seq_len"), "DeployConfig must have seq_len field"
        assert cfg.seq_len == 0, "seq_len default must be 0 (meaning 'use trainer default')"

    def test_P2_deploy_config_has_batch_size_field(self):
        """DeployConfig must carry batch_size defaulting to 0."""
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=100)
        assert hasattr(cfg, "batch_size"), "DeployConfig must have batch_size field"
        assert cfg.batch_size == 0, "batch_size default must be 0"

    def test_P2_seq_len_forwarded_in_train_command(self):
        """When seq_len > 0, _build_train_command must include --seq_len."""
        from neuroslm.connectors.lightning import LightningConnector
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=1000, seq_len=256)
        cmd = LightningConnector._build_train_command(cfg, "~/logs/test.log")
        assert "--seq_len 256" in cmd, (
            f"--seq_len 256 not found in train command:\n{cmd}"
        )

    def test_P2_batch_size_forwarded_in_train_command(self):
        """When batch_size > 0, _build_train_command must include --batch."""
        from neuroslm.connectors.lightning import LightningConnector
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=1000, batch_size=2)
        cmd = LightningConnector._build_train_command(cfg, "~/logs/test.log")
        assert "--batch 2" in cmd, (
            f"--batch 2 not found in train command:\n{cmd}"
        )

    def test_P2_zero_seq_len_not_forwarded(self):
        """seq_len=0 means 'use trainer default' — must NOT appear in command."""
        from neuroslm.connectors.lightning import LightningConnector
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=1000)  # seq_len defaults to 0
        cmd = LightningConnector._build_train_command(cfg, "~/logs/test.log")
        assert "--seq_len" not in cmd, (
            "--seq_len must be absent when config.seq_len == 0"
        )

    def test_P2_zero_batch_size_not_forwarded(self):
        """batch_size=0 means 'use trainer default' — must NOT appear in command."""
        from neuroslm.connectors.lightning import LightningConnector
        from neuroslm.connectors.base import DeployConfig
        cfg = DeployConfig(steps=1000)
        cmd = LightningConnector._build_train_command(cfg, "~/logs/test.log")
        assert "--batch" not in cmd, (
            "--batch must be absent when config.batch_size == 0"
        )


# ─────────────────────────────────────────────────────────────────────────
# P3 — Context-Dependent Fusion Gate
# ─────────────────────────────────────────────────────────────────────────

class TestContextGate:
    """context_gate_enabled: trunk's last hidden state modulates fusion alpha."""

    def test_P3_context_gate_disabled_by_default(self):
        """context_gate_enabled must default to False for backward compat."""
        from neuroslm.dsl.training_config import MultiCortexConfig
        mc = MultiCortexConfig()
        assert hasattr(mc, "context_gate_enabled"), (
            "MultiCortexConfig must have context_gate_enabled field"
        )
        assert mc.context_gate_enabled is False, (
            "context_gate_enabled must default to False"
        )

    def test_P3_context_gate_dim_field_exists(self):
        """MultiCortexConfig must have context_gate_dim defaulting to 64."""
        from neuroslm.dsl.training_config import MultiCortexConfig
        mc = MultiCortexConfig()
        assert hasattr(mc, "context_gate_dim"), (
            "MultiCortexConfig must have context_gate_dim field"
        )
        assert mc.context_gate_dim == 64, (
            "context_gate_dim must default to 64"
        )

    def test_P3_parser_accepts_context_gate_fields(self):
        """_parse_multi_cortex must accept context_gate_enabled and context_gate_dim."""
        from neuroslm.dsl.training_config import _parse_multi_cortex
        mc = _parse_multi_cortex(
            "enabled: true, "
            "n_cortices: 2, "
            "domains: [\"general\", \"code\"], "
            "weights: \"stub\", "
            "context_gate_enabled: true, "
            "context_gate_dim: 32"
        )
        assert mc.context_gate_enabled is True
        assert mc.context_gate_dim == 32

    def test_P3_harness_builds_context_gate_when_enabled(self, fake_lm):
        """When context_gate_enabled=True, harness must have self.context_gate."""
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
        mc.context_gate_enabled = True
        mc.context_gate_dim = D_SEM  # match d_sem so Linear is square
        cfg.multi_cortex = mc

        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        assert hasattr(h, "context_gate") and h.context_gate is not None, (
            "BRIANHarness must build self.context_gate when context_gate_enabled=True"
        )
        assert isinstance(h.context_gate, nn.Linear), (
            "context_gate must be an nn.Linear"
        )

    def test_P3_context_gate_absent_when_disabled(self, fake_lm):
        """When context_gate_enabled=False, context_gate must be None."""
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
        mc.context_gate_enabled = False
        cfg.multi_cortex = mc

        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        gate = getattr(h, "context_gate", None)
        assert gate is None, (
            "context_gate must be None when context_gate_enabled=False"
        )

    def test_P3_alpha_varies_per_sample_when_gate_enabled(self, fake_lm):
        """With context_gate enabled, two different inputs produce different α."""
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
        mc.context_gate_enabled = True
        mc.context_gate_dim = D_SEM
        cfg.multi_cortex = mc

        # Non-zero context gate weights so alpha actually varies
        torch.manual_seed(0)
        h = BRIANHarness.from_language_model(
            language_model=fake_lm, vocab_size=VOCAB, d_sem=D_SEM,
            training_config=cfg,
        )
        # Force non-zero gate weights so different inputs produce different alphas
        nn.init.normal_(h.context_gate.weight, std=1.0)
        nn.init.zeros_(h.context_gate.bias)

        ids_a = torch.zeros(1, 4, dtype=torch.long)    # all token 0
        ids_b = torch.full((1, 4), VOCAB - 1, dtype=torch.long)  # all last token

        with torch.no_grad():
            h.forward(ids_a)
            # After forward: stash the last h_motor hidden state
            h_a = h.language_model._last_h_motor[:, -1, :]  # (1, D)
            alpha_a = torch.sigmoid(h.cortex_mix_logit + h.context_gate(h_a)).item()

            h.forward(ids_b)
            h_b = h.language_model._last_h_motor[:, -1, :]  # (1, D)
            alpha_b = torch.sigmoid(h.cortex_mix_logit + h.context_gate(h_b)).item()

        assert abs(alpha_a - alpha_b) > 1e-4, (
            f"Context gate must produce different alpha per context; "
            f"got alpha_a={alpha_a:.6f} alpha_b={alpha_b:.6f} (diff too small)"
        )


# ─────────────────────────────────────────────────────────────────────────
# H28 — the SmolLM arch must use logits_mixture (standalone-trunk objective)
# ─────────────────────────────────────────────────────────────────────────

class TestSmolLMUsesLogitsMixture:
    """The live SmolLM arch must train the trunk toward a STANDALONE
    distribution, so it must use ``fusion_mode=logits_mixture``, not
    ``additive_correction``.

    additive_correction (``fused = cortex.detach() + α·trunk``) makes the
    trunk learn a residual ``(target − cortex)/α`` that cannot stand alone —
    run 43125941 (α=0.5, T=2) had the trunk-only OOD ppl RISE 24k→88k while
    distillation fought the residual gradient. logits_mixture
    (``fused = (1-α)·trunk + α·cortex``) makes the trunk own ``1-α`` of the
    output, so it learns the full prediction and distillation reinforces it.
    See findings.md H28.
    """

    _ARCH = Path(__file__).resolve().parents[2] / "architectures" / "SmolLM"

    def test_fusion_mode_is_logits_mixture(self):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        mc = load_training_config_from_arch(str(self._ARCH)).multi_cortex
        assert mc.fusion_mode == "logits_mixture", (
            f"SmolLM must use logits_mixture for a standalone trunk; got "
            f"{mc.fusion_mode!r}. additive_correction makes the trunk a "
            f"residual that can never be dropped from the fused output.")

    def test_cortex_weight_lets_trunk_own_majority(self):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        mc = load_training_config_from_arch(str(self._ARCH)).multi_cortex
        # In logits_mixture α is the CORTEX weight; the trunk owns (1-α).
        assert mc.fusion_init < 0.5, (
            f"fusion_init (cortex weight) must be < 0.5 so the trunk owns the "
            f"majority of the output and gets the dominant learning gradient; "
            f"got {mc.fusion_init}")
