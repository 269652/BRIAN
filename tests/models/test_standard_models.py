# -*- coding: utf-8 -*-
"""Structural tests for GPT-2, SmolLM2, and Qwen2.5 model classes.

Tests use the DSL v3 sub-block API (coboundary/transition/norm/embed/output).
No HF weights required — only architecture structure and output shapes.
Logit-equivalence tests are in test_hf_equivalence.py (@pytest.mark.slow).
"""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn

from neuroslm.models.gpt2 import GPT2Model
from neuroslm.models.llama import LlamaModel
from neuroslm.dsl.model_spec import (
    ModelSpec, SheafConfig,
    EmbedConfig, CoboundaryConfig, RopeConfig,
    TransitionConfig, NormConfig, OutputConfig,
)


# ── Tiny spec builders (DSL v3 sub-block API) ─────────────────────────────────

def _gpt2_tiny() -> ModelSpec:
    spec = ModelSpec(kind="gpt2")
    s = spec.sheaf
    s.dim, s.depth, s.heads, s.kv_heads = 64, 2, 4, 4
    s.context, s.vocab = 32, 256
    s.embed      = EmbedConfig(tokens="learned", position="learned")
    s.coboundary = CoboundaryConfig(type="mha", qkv="fused", bias=True)
    s.transition = TransitionConfig(type="mlp", ff_mult=4.0,
                                    activation="gelu", bias=True)
    s.norm   = NormConfig(type="layernorm", eps=1e-5)
    s.output = OutputConfig(tie_embed=True)
    return spec


def _llama_tiny() -> ModelSpec:
    spec = ModelSpec(kind="llama")
    s = spec.sheaf
    s.dim, s.depth, s.heads, s.kv_heads = 64, 2, 4, 2
    s.context, s.vocab = 32, 256
    s.embed      = EmbedConfig(tokens="learned", position="none")
    s.coboundary = CoboundaryConfig(type="gqa", rope=RopeConfig(base=10000))
    s.transition = TransitionConfig(type="swiglu", ff_mult=3.5)
    s.norm   = NormConfig(type="rmsnorm", eps=1e-5)
    s.output = OutputConfig(tie_embed=True)
    return spec


def _qwen_tiny() -> ModelSpec:
    spec = ModelSpec(kind="qwen")
    s = spec.sheaf
    s.dim, s.depth, s.heads, s.kv_heads = 64, 2, 4, 2
    s.context, s.vocab = 32, 256
    s.embed      = EmbedConfig(tokens="learned", position="none")
    s.coboundary = CoboundaryConfig(type="gqa", qkv_bias=True,
                                    rope=RopeConfig(base=1_000_000))
    s.transition = TransitionConfig(type="swiglu", ff_mult=3.5)
    s.norm   = NormConfig(type="rmsnorm", eps=1e-6)
    s.output = OutputConfig(tie_embed=True)
    return spec


# ── GPT-2 ─────────────────────────────────────────────────────────────────────

class TestGPT2ModelStructure:
    def test_constructs_from_spec(self):
        assert GPT2Model(_gpt2_tiny()) is not None

    def test_output_shape(self):
        torch.manual_seed(0)
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        with torch.no_grad():
            logits = model(torch.randint(0, 256, (1, 8)))
        assert logits.shape == (1, 8, 256)

    def test_batch_shape(self):
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        with torch.no_grad():
            logits = model(torch.randint(0, 256, (3, 8)))
        assert logits.shape == (3, 8, 256)

    def test_tie_embed_shares_weight(self):
        model = GPT2Model(_gpt2_tiny())
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_no_tie_embed_independent_weight(self):
        spec = _gpt2_tiny()
        spec.sheaf.output.tie_embed = False
        model = GPT2Model(spec)
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_has_position_embedding(self):
        assert hasattr(GPT2Model(_gpt2_tiny()), "pos_embed")

    def test_uses_layernorm(self):
        assert any(isinstance(m, nn.LayerNorm)
                   for m in GPT2Model(_gpt2_tiny()).modules())

    def test_causal_mask_prevents_future_access(self):
        torch.manual_seed(42)
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        ids1 = torch.randint(0, 256, (1, 8))
        ids2 = ids1.clone()
        ids2[0, 5] = (ids2[0, 5] + 1) % 256
        with torch.no_grad():
            l1, l2 = model(ids1), model(ids2)
        assert torch.allclose(l1[0, :5], l2[0, :5]), \
            "Positions 0-4 must be unchanged when position 5 changes"

    def test_backward_does_not_crash(self):
        model = GPT2Model(_gpt2_tiny())
        model(torch.randint(0, 256, (1, 4))).sum().backward()


class TestGPT2WeightMapping:
    def test_state_dict_has_expected_keys(self):
        keys = set(GPT2Model(_gpt2_tiny()).state_dict().keys())
        for k in ("embed_tokens.weight", "pos_embed.weight",
                  "blocks.0.ln_1.weight", "blocks.0.attn.c_attn.weight",
                  "blocks.0.attn.c_proj.weight",
                  "blocks.0.mlp.c_fc.weight", "blocks.0.mlp.c_proj.weight"):
            assert k in keys, f"Missing key: {k}"

    def test_hf_mapping_exists(self):
        from neuroslm.models.gpt2 import hf_to_model_state_dict
        assert callable(hf_to_model_state_dict)


# ── LLaMA ─────────────────────────────────────────────────────────────────────

class TestLlamaModelStructure:
    def test_constructs_from_spec(self):
        assert LlamaModel(_llama_tiny()) is not None

    def test_output_shape(self):
        torch.manual_seed(0)
        model = LlamaModel(_llama_tiny())
        model.eval()
        with torch.no_grad():
            logits = model(torch.randint(0, 256, (1, 8)))
        assert logits.shape == (1, 8, 256)

    def test_no_position_embedding(self):
        assert not hasattr(LlamaModel(_llama_tiny()), "pos_embed")

    def test_uses_rmsnorm(self):
        from neuroslm.modules.common import RMSNorm
        assert any(isinstance(m, RMSNorm)
                   for m in LlamaModel(_llama_tiny()).modules())

    def test_tie_embed_shares_weight(self):
        model = LlamaModel(_llama_tiny())
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_gqa_kv_heads(self):
        model = LlamaModel(_llama_tiny())
        block = model.blocks[0]
        assert block.attn.k_proj.out_features < block.attn.q_proj.out_features, \
            "GQA: kv_heads=2 < heads=4 → smaller k/v projections"

    def test_causal_mask_prevents_future_access(self):
        torch.manual_seed(42)
        model = LlamaModel(_llama_tiny())
        model.eval()
        ids1 = torch.randint(0, 256, (1, 8))
        ids2 = ids1.clone()
        ids2[0, 5] = (ids2[0, 5] + 1) % 256
        with torch.no_grad():
            l1, l2 = model(ids1), model(ids2)
        assert torch.allclose(l1[0, :5], l2[0, :5])

    def test_hf_mapping_exists(self):
        from neuroslm.models.llama import hf_to_model_state_dict
        assert callable(hf_to_model_state_dict)


class TestLlamaWeightMapping:
    def test_state_dict_has_llama_keys(self):
        keys = set(LlamaModel(_llama_tiny()).state_dict().keys())
        for k in ("embed_tokens.weight",
                  "blocks.0.attn.q_proj.weight",
                  "blocks.0.attn.k_proj.weight",
                  "blocks.0.attn.v_proj.weight",
                  "blocks.0.attn.o_proj.weight",
                  "blocks.0.mlp.gate_proj.weight",
                  "blocks.0.mlp.up_proj.weight",
                  "blocks.0.mlp.down_proj.weight",
                  "norm.weight"):
            assert k in keys, f"Missing key: {k}"


# ── Qwen via LlamaModel ───────────────────────────────────────────────────────

class TestQwenViaLlama:
    def test_qwen_coboundary_builds_llamamodel(self):
        from neuroslm.models import build_model
        assert isinstance(build_model(_qwen_tiny()), LlamaModel)

    def test_output_shape(self):
        from neuroslm.models import build_model
        model = build_model(_qwen_tiny())
        model.eval()
        with torch.no_grad():
            logits = model(torch.randint(0, 256, (1, 6)))
        assert logits.shape == (1, 6, 256)

    def test_qkv_bias_present_when_set(self):
        model = LlamaModel(_qwen_tiny())
        # q_proj should have bias when qkv_bias=True
        assert model.blocks[0].attn.q_proj.bias is not None


# ── build_model factory ───────────────────────────────────────────────────────

class TestBuildModelFactory:
    def test_coboundary_mha_returns_gpt2model(self):
        from neuroslm.models import build_model
        assert isinstance(build_model(_gpt2_tiny()), GPT2Model)

    def test_coboundary_gqa_returns_llamamodel(self):
        from neuroslm.models import build_model
        assert isinstance(build_model(_llama_tiny()), LlamaModel)

    def test_unknown_coboundary_type_raises(self):
        from neuroslm.models import build_model
        spec = _llama_tiny()
        spec.sheaf.coboundary.type = "completely_unknown"
        with pytest.raises(ValueError):
            build_model(spec)
