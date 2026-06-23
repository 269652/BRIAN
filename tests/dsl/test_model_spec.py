# -*- coding: utf-8 -*-
"""RED-first tests for the DSL v2 ModelSpec parser.

The new `model { ... }` block unifies all standard LM architectures under the
THSD sheaf framework: kind=gpt2|llama|qwen|brian maps to specific sheaf configs.
Every test will RED until neuroslm/dsl/model_spec.py is implemented.
"""
from __future__ import annotations
import pytest
from neuroslm.dsl.model_spec import ModelSpec, SheafConfig, parse_model_block


class TestSheafConfigDefaults:
    def test_default_norm_is_rmsnorm(self):
        s = SheafConfig()
        assert s.norm in ("rmsnorm", "layernorm")

    def test_default_pos_is_rope(self):
        s = SheafConfig()
        assert s.pos in ("rope", "learned", "none")

    def test_required_fields_are_optional_at_construction(self):
        SheafConfig()  # should not raise


class TestModelSpecDefaults:
    def test_kind_is_required_or_has_default(self):
        spec = ModelSpec()
        assert spec.kind is not None

    def test_weights_is_none_by_default(self):
        spec = ModelSpec()
        assert spec.weights is None

    def test_sheaf_is_sheafconfig(self):
        spec = ModelSpec()
        assert isinstance(spec.sheaf, SheafConfig)


class TestParseGPT2Block:
    DSL = """
    model {
        kind: gpt2
        weights: "hf:openai-community/gpt2"
        sheaf {
            dim: 768
            depth: 12
            heads: 12
            kv_heads: 12
            context: 1024
            vocab: 50257
            pos: learned
            ff_mult: 4.0
            ff_act: gelu
            norm: layernorm
            tie_embed: true
            bias: true
        }
    }
    """

    def test_parse_kind(self):
        spec = parse_model_block(self.DSL)
        assert spec.kind == "gpt2"

    def test_parse_weights(self):
        spec = parse_model_block(self.DSL)
        assert spec.weights == "hf:openai-community/gpt2"

    def test_parse_dim(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.dim == 768

    def test_parse_depth(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.depth == 12

    def test_parse_heads(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.heads == 12

    def test_parse_context(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.context == 1024

    def test_parse_pos_learned(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.pos == "learned"

    def test_parse_ff_act_gelu(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.ff_act == "gelu"

    def test_parse_norm_layernorm(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.norm == "layernorm"

    def test_parse_bias_true(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.bias is True

    def test_parse_tie_embed_true(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.tie_embed is True


class TestParseSmolLM2Block:
    DSL = """
    model {
        kind: llama
        weights: "hf:HuggingFaceTB/SmolLM2-135M"
        sheaf {
            dim: 576
            depth: 30
            heads: 9
            kv_heads: 3
            context: 2048
            vocab: 49152
            pos: rope
            rope_base: 10000
            ff_mult: 3.5
            ff_act: swiglu
            norm: rmsnorm
            tie_embed: true
            bias: false
        }
    }
    """

    def test_parse_kind_llama(self):
        spec = parse_model_block(self.DSL)
        assert spec.kind == "llama"

    def test_parse_kv_heads(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.kv_heads == 3

    def test_parse_rope_base(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.rope_base == 10000

    def test_parse_ff_act_swiglu(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.ff_act == "swiglu"

    def test_parse_norm_rmsnorm(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.norm == "rmsnorm"

    def test_parse_bias_false(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.bias is False

    def test_parse_ff_mult(self):
        spec = parse_model_block(self.DSL)
        assert abs(spec.sheaf.ff_mult - 3.5) < 1e-9


class TestParseQwenBlock:
    DSL = """
    model {
        kind: qwen
        weights: "hf:Qwen/Qwen2.5-0.5B"
        sheaf {
            dim: 896
            depth: 24
            heads: 14
            kv_heads: 2
            context: 32768
            vocab: 151936
            pos: rope
            rope_base: 1000000
            ff_mult: 3.5
            ff_act: swiglu
            norm: rmsnorm
            tie_embed: true
            bias: false
            qkv_bias: true
        }
    }
    """

    def test_parse_kind_qwen(self):
        spec = parse_model_block(self.DSL)
        assert spec.kind == "qwen"

    def test_parse_rope_base_1m(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.rope_base == 1_000_000

    def test_parse_vocab_151936(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.vocab == 151936

    def test_parse_qkv_bias_true(self):
        spec = parse_model_block(self.DSL)
        assert spec.sheaf.qkv_bias is True


class TestParseValidation:
    def test_unknown_kind_raises(self):
        dsl = "model { kind: unknown_model }"
        with pytest.raises((ValueError, KeyError)):
            parse_model_block(dsl)

    def test_negative_dim_raises(self):
        dsl = "model { kind: gpt2, sheaf { dim: -1 } }"
        with pytest.raises((ValueError, AssertionError)):
            parse_model_block(dsl)

    def test_empty_block_gives_defaults(self):
        dsl = "model { kind: gpt2 }"
        spec = parse_model_block(dsl)
        assert spec.kind == "gpt2"
        assert spec.sheaf is not None
