# -*- coding: utf-8 -*-
"""Phase A — BRIANHarness: wraps a DSL circuit for language-model training.

The harness is the layer between a DSL-compiled architecture
(`compile_folder(...) → nn.Module`) and the training loop. It owns:

  * the token embedding (vocab × d_sem)
  * the LM head (d_sem × vocab)
  * the forward pass: ids → embed → DSL circuit → LM head → logits
  * the loss: cross-entropy with optional per-sample clipping
  * the optimizer step (AdamW), grad accumulation, grad clipping
  * checkpoint save/load (round-trips with no precision loss)

This test file pins down the public contract before the implementation.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.training_config import TrainingConfig, LossClippingConfig
from neuroslm.harness import BRIANHarness


ARCH_ROOT = Path(__file__).resolve().parent.parent / "architectures" / "master"


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def small_circuit():
    """A small DSL circuit compiled from the rcc_bowtie folder at low d_sem
    so tests run on CPU in reasonable time."""
    ir = compile_folder(ARCH_ROOT)
    Cls = CodeGenerator(ir, module_name="HarnessTestCircuit").compile_to_module()
    return Cls(d_sem=64)


@pytest.fixture
def harness(small_circuit):
    cfg = TrainingConfig()
    return BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64,
                        training_config=cfg)


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:
    def test_wraps_circuit(self, small_circuit):
        h = BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64)
        assert isinstance(h, nn.Module)
        assert h.vocab_size == 512
        assert h.d_sem == 64

    def test_has_embedding_and_lm_head(self, harness):
        # Embedding shape (vocab_size, d_sem)
        assert harness.embedding.weight.shape == (512, 64)
        # LM head shape (d_sem, vocab_size)  (Linear stores as (out, in) = (vocab, d_sem))
        assert harness.lm_head.weight.shape == (512, 64)

    def test_uses_default_training_config_when_none(self, small_circuit):
        h = BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64,
                        training_config=None)
        assert h.training_config.loss_clipping.enabled is False


# ── Forward pass ──────────────────────────────────────────────────────

class TestForward:
    def test_forward_shapes(self, harness):
        ids = torch.randint(0, 512, (2, 16))   # (batch, seq_len)
        logits = harness(ids)
        assert logits.shape == (2, 16, 512)    # (batch, seq, vocab)

    def test_forward_no_nans(self, harness):
        ids = torch.randint(0, 512, (2, 16))
        logits = harness(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_gradient_flows_through_embedding(self, harness):
        ids = torch.randint(0, 512, (2, 8))
        logits = harness(ids)
        logits.sum().backward()
        assert harness.embedding.weight.grad is not None
        assert harness.embedding.weight.grad.abs().sum() > 0


# ── Loss computation ──────────────────────────────────────────────────

class TestLoss:
    def test_loss_is_scalar(self, harness):
        ids = torch.randint(0, 512, (2, 16))
        targets = torch.randint(0, 512, (2, 16))
        loss = harness.compute_loss(ids, targets)
        assert loss.dim() == 0   # scalar
        assert loss.item() > 0   # CE on random > 0

    def test_loss_clipping_disabled_matches_vanilla_ce(self, small_circuit):
        # Note: the DSL circuit mutates internal state (back-edge buffers)
        # on every forward call. So both losses must be computed from the
        # *same* logits, captured by a single forward pass.
        cfg = TrainingConfig()
        h = BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64,
                        training_config=cfg)
        torch.manual_seed(0)
        ids = torch.randint(0, 512, (2, 16))
        targets = torch.randint(0, 512, (2, 16))

        logits = h(ids)
        harness_loss = h._compute_loss_from_logits(logits, targets)
        ref_loss = nn.functional.cross_entropy(
            logits.reshape(-1, 512), targets.reshape(-1)
        )
        assert torch.allclose(harness_loss, ref_loss, atol=1e-6)

    def test_loss_clipping_suppresses_outlier_sequence(self, small_circuit):
        # Compare clipped vs unclipped on the *same* logits — synthesise
        # the logits directly so we control which sequences look bad.
        h = BRIANHarness(
            circuit=small_circuit, vocab_size=512, d_sem=64,
            training_config=TrainingConfig(
                loss_clipping=LossClippingConfig(enabled=True, factor=2.0)
            ),
        )
        # Hand-crafted logits: 4 sequences of length 16, 3 confident on
        # targets, the 4th confident on the WRONG token.
        torch.manual_seed(0)
        batch, seq, vocab = 4, 16, 512
        targets = torch.randint(0, vocab, (batch, seq))

        # Build "easy" logits: very high score on the target token
        logits = torch.full((batch, seq, vocab), -10.0)
        logits.scatter_(2, targets.unsqueeze(-1), 10.0)

        # Sabotage sequence 3 — wrong target token gets the high score
        wrong = (targets[3] + 1) % vocab
        logits[3] = torch.full((seq, vocab), -10.0)
        logits[3].scatter_(1, wrong.unsqueeze(-1), 10.0)

        # Clipped loss
        clipped = h._compute_loss_from_logits(logits, targets)

        # Same logits, clipping off
        h_noclip = BRIANHarness(
            circuit=small_circuit, vocab_size=512, d_sem=64,
            training_config=TrainingConfig(
                loss_clipping=LossClippingConfig(enabled=False)
            ),
        )
        unclipped = h_noclip._compute_loss_from_logits(logits, targets)

        # Clipping suppresses the outlier sequence → strictly lower loss
        assert clipped.item() < unclipped.item()


# ── Training step ─────────────────────────────────────────────────────

class TestTrainStep:
    def test_optimizer_step_changes_params(self, harness):
        # Capture a parameter snapshot, do a step, see it move
        params_before = {n: p.clone() for n, p in harness.named_parameters()}
        ids = torch.randint(0, 512, (2, 16))
        targets = torch.randint(0, 512, (2, 16))
        loss = harness.train_step(ids, targets)
        params_after = {n: p for n, p in harness.named_parameters()}

        moved = sum(1 for n in params_before
                    if not torch.allclose(params_before[n], params_after[n]))
        assert moved > 0, "expected at least one parameter to update"
        assert loss > 0

    def test_grad_accumulation(self, small_circuit):
        cfg = TrainingConfig()
        cfg.grad_accum = 4
        h = BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64,
                        training_config=cfg)

        # First 3 train_step calls accumulate, the 4th applies and zeros.
        ids = torch.randint(0, 512, (2, 16))
        targets = torch.randint(0, 512, (2, 16))

        params_before = {n: p.clone() for n, p in h.named_parameters()}
        for _ in range(3):
            h.train_step(ids, targets)   # accumulate
        params_after_3 = {n: p.clone() for n, p in h.named_parameters()}

        # After 3 accumulation steps, no actual optimizer step happened yet
        for n in params_before:
            assert torch.allclose(params_before[n], params_after_3[n]), \
                f"{n} moved before grad-accum threshold"

        h.train_step(ids, targets)  # 4th call → optimizer fires
        params_after_4 = {n: p for n, p in h.named_parameters()}
        moved = sum(1 for n in params_before
                    if not torch.allclose(params_before[n], params_after_4[n]))
        assert moved > 0


# ── Checkpoint round-trip ─────────────────────────────────────────────

class TestCheckpoint:
    def test_save_load_round_trip(self, small_circuit, tmp_path):
        h1 = BRIANHarness(circuit=small_circuit, vocab_size=512, d_sem=64)
        ckpt_path = tmp_path / "ckpt.pt"
        h1.save_checkpoint(str(ckpt_path), step=42)
        assert ckpt_path.exists()

        # Fresh circuit + harness, then load
        ir = compile_folder(ARCH_ROOT)
        Cls = CodeGenerator(ir, module_name="ReloadCircuit").compile_to_module()
        h2 = BRIANHarness(circuit=Cls(d_sem=64), vocab_size=512, d_sem=64)
        step = h2.load_checkpoint(str(ckpt_path))
        assert step == 42

        # Every parameter must match exactly
        sd1 = h1.state_dict()
        sd2 = h2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for k in sd1:
            assert torch.allclose(sd1[k], sd2[k]), f"{k} did not round-trip"


# ── Topology summary (for compatibility with train.py) ───────────────

class TestIntrospection:
    def test_topology_summary_returns_string(self, harness):
        s = harness.topology_summary()
        assert isinstance(s, str)
        assert "vocab" in s.lower() or "embedding" in s.lower()
        assert "d_sem" in s.lower() or "64" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
