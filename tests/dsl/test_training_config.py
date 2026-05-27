# -*- coding: utf-8 -*-
"""Tests for the `training { ... }` block in arch.neuro.

The block declares pipeline-level behavior the BRIAN harness applies on
top of the compiled DSL circuit:

    training {
        loss_clipping: { enabled: true, method: "per_sample", factor: 3.0 }
        quantization:  { enabled: false, bits: 8 }
        grad_accum: 4
        optimizer: "adamw"
        learning_rate: 0.0003
        weight_decay: 0.01
        grad_clip: 1.0
        label_smoothing: 0.05
    }

This stage delivers a parser that turns the block into a structured
`TrainingConfig` dataclass with sane defaults for everything that's
omitted. The harness consumes it as a plain Python object.
"""
import pytest
from pathlib import Path

from neuroslm.dsl.training_config import (
    TrainingConfig,
    LossClippingConfig,
    QuantizationConfig,
    parse_training_config,
    load_training_config_from_arch,
)


# ── Defaults ──────────────────────────────────────────────────────────

class TestDefaults:
    def test_empty_block(self):
        cfg = parse_training_config("")
        # Loss clipping defaults to OFF (preserves vanilla behavior)
        assert cfg.loss_clipping.enabled is False
        assert cfg.loss_clipping.method == "per_sample"
        assert cfg.loss_clipping.factor == 3.0
        # Quantization defaults to OFF
        assert cfg.quantization.enabled is False
        # Standard training hyperparameters
        assert cfg.grad_accum == 1
        assert cfg.optimizer == "adamw"
        assert cfg.learning_rate == 3e-4
        assert cfg.grad_clip == 1.0


# ── Loss clipping subblock ────────────────────────────────────────────

class TestLossClipping:
    def test_enable_loss_clipping(self):
        body = '''
            loss_clipping: {
                enabled: true,
                method: "per_sample",
                factor: 3.0
            }
        '''
        cfg = parse_training_config(body)
        assert cfg.loss_clipping.enabled is True
        assert cfg.loss_clipping.factor == 3.0

    def test_custom_clip_factor(self):
        body = '''
            loss_clipping: { enabled: true, factor: 5.5 }
        '''
        cfg = parse_training_config(body)
        assert cfg.loss_clipping.factor == 5.5

    def test_unknown_method_rejected(self):
        body = '''
            loss_clipping: { enabled: true, method: "bogus" }
        '''
        with pytest.raises(ValueError, match="method"):
            parse_training_config(body)


# ── Quantization subblock ─────────────────────────────────────────────

class TestQuantization:
    def test_int8_quantization(self):
        body = '''
            quantization: { enabled: true, bits: 8 }
        '''
        cfg = parse_training_config(body)
        assert cfg.quantization.enabled is True
        assert cfg.quantization.bits == 8

    def test_invalid_bits_rejected(self):
        body = '''
            quantization: { enabled: true, bits: 17 }
        '''
        with pytest.raises(ValueError, match="bits"):
            parse_training_config(body)


# ── Top-level fields ──────────────────────────────────────────────────

class TestTopLevelFields:
    def test_optimizer_and_lr(self):
        body = '''
            optimizer: "adamw",
            learning_rate: 0.0005,
            weight_decay: 0.02
        '''
        cfg = parse_training_config(body)
        assert cfg.optimizer == "adamw"
        assert cfg.learning_rate == 0.0005
        assert cfg.weight_decay == 0.02

    def test_grad_accum_and_clip(self):
        body = '''
            grad_accum: 8,
            grad_clip: 0.5,
            label_smoothing: 0.1
        '''
        cfg = parse_training_config(body)
        assert cfg.grad_accum == 8
        assert cfg.grad_clip == 0.5
        assert cfg.label_smoothing == 0.1


# ── load_training_config_from_arch — integration with arch.neuro ──────

class TestLoadFromArchFolder:
    def test_loads_from_arch_neuro(self, tmp_path):
        (tmp_path / "arch.neuro").write_text('''
            architecture test_arch { d_sem: 256 }

            training {
                loss_clipping: { enabled: true, factor: 3.0 }
                grad_accum: 4
                optimizer: "adamw"
            }
        ''', encoding="utf-8")

        cfg = load_training_config_from_arch(tmp_path)
        assert cfg.loss_clipping.enabled is True
        assert cfg.grad_accum == 4
        assert cfg.optimizer == "adamw"

    def test_arch_neuro_without_training_block_returns_defaults(self, tmp_path):
        (tmp_path / "arch.neuro").write_text(
            'architecture test_arch { d_sem: 256 }', encoding="utf-8"
        )
        cfg = load_training_config_from_arch(tmp_path)
        # Defaults preserved when training block absent
        assert cfg.loss_clipping.enabled is False
        assert cfg.grad_accum == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
