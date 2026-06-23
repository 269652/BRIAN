# -*- coding: utf-8 -*-
"""Tests for DSL v3 ModelSpec parser — unified THSD grammar.

All SOTA model architectures are now fully expressed in nested sub-blocks:
  embed { }         — token + positional embedding
  coboundary { }    — C⁰→C¹ attention operator (mha/gqa/mla/kjpla)
  transition { }    — C¹→C⁰ FFN (mlp/swiglu/moe/liouville_symplectic)
  norm { }          — normalisation type + placement
  output { }        — lm_head weight tying
  diagnostic { }    — topological/diagnostic aux modules

`kind: gpt2|llama|qwen|deepseek` is sugar that pre-fills sub-block defaults.
"""
from __future__ import annotations
import pytest
from neuroslm.dsl.model_spec import (
    ModelSpec, SheafConfig,
    EmbedConfig, CoboundaryConfig, RopeConfig, MLAConfig,
    TransitionConfig, MoEConfig, NormConfig, OutputConfig, DiagnosticConfig,
    parse_model_block,
)


# ── Dataclass defaults ────────────────────────────────────────────────────────

class TestCoboundaryConfigDefaults:
    def test_default_type_is_valid(self):
        c = CoboundaryConfig()
        assert c.type in ("gqa", "mha", "mla", "kjpla")

    def test_default_qkv_bias_false(self):
        assert CoboundaryConfig().qkv_bias is False

    def test_default_rope_is_none(self):
        assert CoboundaryConfig().rope is None


class TestTransitionConfigDefaults:
    def test_default_type_is_valid(self):
        t = TransitionConfig()
        assert t.type in ("swiglu", "mlp", "geglu", "moe", "liouville_symplectic")

    def test_default_ff_mult_positive(self):
        assert TransitionConfig().ff_mult > 0

    def test_default_moe_none(self):
        assert TransitionConfig().moe is None


class TestSheafConfigSubBlocks:
    def test_has_embed(self):
        assert isinstance(SheafConfig().embed, EmbedConfig)

    def test_has_coboundary(self):
        assert isinstance(SheafConfig().coboundary, CoboundaryConfig)

    def test_has_transition(self):
        assert isinstance(SheafConfig().transition, TransitionConfig)

    def test_has_norm(self):
        assert isinstance(SheafConfig().norm, NormConfig)

    def test_has_output(self):
        assert isinstance(SheafConfig().output, OutputConfig)

    def test_diagnostic_none_by_default(self):
        assert SheafConfig().diagnostic is None


# ── GPT-2 DSL (mha + mlp + layernorm + learned pos) ─────────────────────────

class TestParseGPT2Block:
    DSL = """
    model {
        weights: "hf:openai-community/gpt2"
        sheaf {
            dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257

            embed    { tokens: learned, position: learned }

            coboundary {
                type: mha
                bias: true
            }

            transition {
                type: mlp
                ff_mult: 4.0
                activation: gelu
                bias: true
            }

            norm     { type: layernorm, placement: pre, eps: 1e-5 }
            output   { tie_embed: true }
        }
    }
    """

    def test_weights(self):
        assert parse_model_block(self.DSL).weights == "hf:openai-community/gpt2"

    def test_dim(self):
        assert parse_model_block(self.DSL).sheaf.dim == 768

    def test_depth(self):
        assert parse_model_block(self.DSL).sheaf.depth == 12

    def test_vocab(self):
        assert parse_model_block(self.DSL).sheaf.vocab == 50257

    def test_embed_position_learned(self):
        assert parse_model_block(self.DSL).sheaf.embed.position == "learned"

    def test_coboundary_type_mha(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.type == "mha"

    def test_coboundary_bias_true(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.bias is True

    def test_transition_type_mlp(self):
        assert parse_model_block(self.DSL).sheaf.transition.type == "mlp"

    def test_transition_ff_mult_4(self):
        assert abs(parse_model_block(self.DSL).sheaf.transition.ff_mult - 4.0) < 1e-9

    def test_transition_activation_gelu(self):
        assert parse_model_block(self.DSL).sheaf.transition.activation == "gelu"

    def test_transition_bias_true(self):
        assert parse_model_block(self.DSL).sheaf.transition.bias is True

    def test_norm_type_layernorm(self):
        assert parse_model_block(self.DSL).sheaf.norm.type == "layernorm"

    def test_norm_eps(self):
        assert abs(parse_model_block(self.DSL).sheaf.norm.eps - 1e-5) < 1e-12

    def test_output_tie_embed_true(self):
        assert parse_model_block(self.DSL).sheaf.output.tie_embed is True


# ── SmolLM2-135M DSL (gqa + rope + swiglu + rmsnorm) ────────────────────────

class TestParseSmolLM2Block:
    DSL = """
    model {
        weights: "hf:HuggingFaceTB/SmolLM2-135M"
        sheaf {
            dim: 576, depth: 30, heads: 9, kv_heads: 3, context: 2048, vocab: 49152

            embed    { tokens: learned, position: none }

            coboundary {
                type: gqa
                rope { base: 10000 }
            }

            transition {
                type: swiglu
                ff_mult: 3.5
            }

            norm     { type: rmsnorm, placement: pre, eps: 1e-5 }
            output   { tie_embed: true }
        }
    }
    """

    def test_kv_heads(self):
        assert parse_model_block(self.DSL).sheaf.kv_heads == 3

    def test_embed_position_none(self):
        assert parse_model_block(self.DSL).sheaf.embed.position == "none"

    def test_coboundary_type_gqa(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.type == "gqa"

    def test_coboundary_rope_base(self):
        cb = parse_model_block(self.DSL).sheaf.coboundary
        assert cb.rope is not None
        assert cb.rope.base == 10000

    def test_coboundary_no_qkv_bias(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.qkv_bias is False

    def test_transition_type_swiglu(self):
        assert parse_model_block(self.DSL).sheaf.transition.type == "swiglu"

    def test_transition_ff_mult(self):
        assert abs(parse_model_block(self.DSL).sheaf.transition.ff_mult - 3.5) < 1e-9

    def test_norm_type_rmsnorm(self):
        assert parse_model_block(self.DSL).sheaf.norm.type == "rmsnorm"


# ── Qwen2.5-0.5B DSL (gqa + qkv_bias + large rope_base) ────────────────────

class TestParseQwenBlock:
    DSL = """
    model {
        weights: "hf:Qwen/Qwen2.5-0.5B"
        sheaf {
            dim: 896, depth: 24, heads: 14, kv_heads: 2, context: 32768, vocab: 151936

            embed    { tokens: learned, position: none }

            coboundary {
                type: gqa
                qkv_bias: true
                rope { base: 1000000 }
            }

            transition {
                type: swiglu
                ff_mult: 3.5
            }

            norm     { type: rmsnorm, placement: pre, eps: 1e-6 }
            output   { tie_embed: true }
        }
    }
    """

    def test_vocab(self):
        assert parse_model_block(self.DSL).sheaf.vocab == 151936

    def test_coboundary_qkv_bias_true(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.qkv_bias is True

    def test_coboundary_rope_base_1m(self):
        cb = parse_model_block(self.DSL).sheaf.coboundary
        assert cb.rope is not None
        assert cb.rope.base == 1_000_000

    def test_norm_eps_1e6(self):
        assert abs(parse_model_block(self.DSL).sheaf.norm.eps - 1e-6) < 1e-15

    def test_coboundary_bias_default_false(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.bias is False


# ── BRIAN (kjpla + liouville_symplectic + topo_charge) ──────────────────────

class TestParseBRIANBlock:
    DSL = """
    model {
        sheaf {
            dim: 576, depth: 30, heads: 9, kv_heads: 3, context: 2048, vocab: 49152

            embed    { tokens: learned, position: none }

            coboundary {
                type: kjpla
                rope { base: 10000 }
                josephson_strength: 0.1
                entropy_eps: 0.01
            }

            transition {
                type: liouville_symplectic
                ff_mult: 3.5
                noether_strength: 0.01
            }

            diagnostic {
                type: topo_charge
                alpha: 0.01
                gamma: 0.005
            }

            norm     { type: rmsnorm, placement: pre, eps: 1e-5 }
            output   { tie_embed: true }
        }
    }
    """

    def test_coboundary_type_kjpla(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.type == "kjpla"

    def test_coboundary_josephson_strength(self):
        cb = parse_model_block(self.DSL).sheaf.coboundary
        assert abs(cb.josephson_strength - 0.1) < 1e-9

    def test_coboundary_entropy_eps(self):
        cb = parse_model_block(self.DSL).sheaf.coboundary
        assert abs(cb.entropy_eps - 0.01) < 1e-9

    def test_transition_type_liouville(self):
        assert parse_model_block(self.DSL).sheaf.transition.type == "liouville_symplectic"

    def test_transition_noether_strength(self):
        t = parse_model_block(self.DSL).sheaf.transition
        assert abs(t.noether_strength - 0.01) < 1e-9

    def test_diagnostic_not_none(self):
        assert parse_model_block(self.DSL).sheaf.diagnostic is not None

    def test_diagnostic_type_topo_charge(self):
        assert parse_model_block(self.DSL).sheaf.diagnostic.type == "topo_charge"

    def test_diagnostic_alpha(self):
        d = parse_model_block(self.DSL).sheaf.diagnostic
        assert abs(d.alpha - 0.01) < 1e-9

    def test_diagnostic_gamma(self):
        d = parse_model_block(self.DSL).sheaf.diagnostic
        assert abs(d.gamma - 0.005) < 1e-9

    def test_no_weights(self):
        assert parse_model_block(self.DSL).weights is None


# ── DeepSeek-style (MLA + MoE) ───────────────────────────────────────────────

class TestParseDeepSeekBlock:
    DSL = """
    model {
        sheaf {
            dim: 7168, depth: 61, heads: 128, kv_heads: 128, context: 163840, vocab: 129280

            embed    { tokens: learned, position: none }

            coboundary {
                type: mla
                kv_lora_rank: 512
                q_lora_rank: 1536
                qk_nope_dim: 128
                qk_rope_dim: 64
                v_dim: 128
                rope { base: 10000, scaling: yarn }
            }

            transition {
                type: moe
                n_experts: 256
                n_active: 8
                shared_experts: 1
                ff_mult: 1.0
                activation: swiglu
            }

            norm     { type: rmsnorm, placement: pre, eps: 1e-6 }
            output   { tie_embed: false }
        }
    }
    """

    def test_coboundary_type_mla(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.type == "mla"

    def test_coboundary_kv_lora_rank(self):
        cb = parse_model_block(self.DSL).sheaf.coboundary
        assert cb.mla is not None
        assert cb.mla.kv_lora_rank == 512

    def test_coboundary_q_lora_rank(self):
        assert parse_model_block(self.DSL).sheaf.coboundary.mla.q_lora_rank == 1536

    def test_coboundary_rope_scaling_yarn(self):
        rope = parse_model_block(self.DSL).sheaf.coboundary.rope
        assert rope is not None
        assert rope.scaling == "yarn"

    def test_transition_type_moe(self):
        assert parse_model_block(self.DSL).sheaf.transition.type == "moe"

    def test_transition_moe_n_experts(self):
        t = parse_model_block(self.DSL).sheaf.transition
        assert t.moe is not None
        assert t.moe.n_experts == 256

    def test_transition_moe_n_active(self):
        assert parse_model_block(self.DSL).sheaf.transition.moe.n_active == 8

    def test_transition_moe_shared_experts(self):
        assert parse_model_block(self.DSL).sheaf.transition.moe.shared_experts == 1

    def test_output_tie_embed_false(self):
        assert parse_model_block(self.DSL).sheaf.output.tie_embed is False


# ── kind: macro sugar ─────────────────────────────────────────────────────────

class TestKindMacroExpansion:
    def test_kind_gpt2_sets_coboundary_mha(self):
        dsl = "model { kind: gpt2, sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.coboundary.type == "mha"

    def test_kind_gpt2_sets_transition_mlp(self):
        dsl = "model { kind: gpt2, sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.transition.type == "mlp"

    def test_kind_gpt2_sets_norm_layernorm(self):
        dsl = "model { kind: gpt2, sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.norm.type == "layernorm"

    def test_kind_gpt2_sets_embed_position_learned(self):
        dsl = "model { kind: gpt2, sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.embed.position == "learned"

    def test_kind_llama_sets_coboundary_gqa(self):
        dsl = "model { kind: llama, sheaf { dim: 576, depth: 30, heads: 9, kv_heads: 3, context: 2048, vocab: 49152 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.coboundary.type == "gqa"

    def test_kind_llama_sets_transition_swiglu(self):
        dsl = "model { kind: llama, sheaf { dim: 576, depth: 30, heads: 9, kv_heads: 3, context: 2048, vocab: 49152 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.transition.type == "swiglu"

    def test_kind_llama_sets_norm_rmsnorm(self):
        dsl = "model { kind: llama, sheaf { dim: 576, depth: 30, heads: 9, kv_heads: 3, context: 2048, vocab: 49152 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.norm.type == "rmsnorm"

    def test_explicit_blocks_override_kind(self):
        dsl = """model {
            kind: gpt2
            sheaf {
                dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257
                coboundary { type: gqa }
            }
        }"""
        spec = parse_model_block(dsl)
        # Explicit coboundary block overrides kind macro
        assert spec.sheaf.coboundary.type == "gqa"


# ── Validation ────────────────────────────────────────────────────────────────

class TestParseValidation:
    def test_unknown_coboundary_type_raises(self):
        dsl = "model { sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257, coboundary { type: unknown_attn } } }"
        with pytest.raises(ValueError):
            parse_model_block(dsl)

    def test_unknown_transition_type_raises(self):
        dsl = "model { sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257, transition { type: unknown_mlp } } }"
        with pytest.raises(ValueError):
            parse_model_block(dsl)

    def test_negative_dim_raises(self):
        dsl = "model { sheaf { dim: -1, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        with pytest.raises((ValueError, AssertionError)):
            parse_model_block(dsl)

    def test_empty_block_gives_defaults(self):
        dsl = "model { sheaf { dim: 768, depth: 12, heads: 12, context: 1024, vocab: 50257 } }"
        spec = parse_model_block(dsl)
        assert spec.sheaf.dim == 768
        assert spec.sheaf.coboundary is not None
        assert spec.sheaf.transition is not None
