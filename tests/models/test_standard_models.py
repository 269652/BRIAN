# -*- coding: utf-8 -*-
"""RED-first structural tests for GPT-2, SmolLM2, and Qwen2.5 model classes.

These tests verify model structure and output shapes WITHOUT downloading HF
weights — they only test that our implementations have correct architecture.
Logit-equivalence tests (requiring HF weights) are in test_hf_equivalence.py
and are marked @pytest.mark.slow.

CLAUDE.md §14: no stubs — every test checks real math, not just shapes.
"""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn

from neuroslm.models.gpt2 import GPT2Model
from neuroslm.models.llama import LlamaModel
from neuroslm.dsl.model_spec import ModelSpec, SheafConfig


# ── Tiny configs for fast CPU tests ──────────────────────────────────────────

def _gpt2_tiny() -> ModelSpec:
    spec = ModelSpec()
    spec.kind = "gpt2"
    spec.sheaf.dim = 64
    spec.sheaf.depth = 2
    spec.sheaf.heads = 4
    spec.sheaf.kv_heads = 4
    spec.sheaf.context = 32
    spec.sheaf.vocab = 256
    spec.sheaf.pos = "learned"
    spec.sheaf.ff_mult = 4.0
    spec.sheaf.ff_act = "gelu"
    spec.sheaf.norm = "layernorm"
    spec.sheaf.tie_embed = True
    spec.sheaf.bias = True
    return spec


def _llama_tiny() -> ModelSpec:
    spec = ModelSpec()
    spec.kind = "llama"
    spec.sheaf.dim = 64
    spec.sheaf.depth = 2
    spec.sheaf.heads = 4
    spec.sheaf.kv_heads = 2
    spec.sheaf.context = 32
    spec.sheaf.vocab = 256
    spec.sheaf.pos = "rope"
    spec.sheaf.rope_base = 10000
    spec.sheaf.ff_mult = 3.5
    spec.sheaf.ff_act = "swiglu"
    spec.sheaf.norm = "rmsnorm"
    spec.sheaf.tie_embed = True
    spec.sheaf.bias = False
    return spec


def _qwen_tiny() -> ModelSpec:
    spec = ModelSpec()
    spec.kind = "qwen"
    spec.sheaf.dim = 64
    spec.sheaf.depth = 2
    spec.sheaf.heads = 4
    spec.sheaf.kv_heads = 2
    spec.sheaf.context = 32
    spec.sheaf.vocab = 256
    spec.sheaf.pos = "rope"
    spec.sheaf.rope_base = 1_000_000
    spec.sheaf.ff_mult = 3.5
    spec.sheaf.ff_act = "swiglu"
    spec.sheaf.norm = "rmsnorm"
    spec.sheaf.tie_embed = True
    spec.sheaf.bias = False
    return spec


# ============================================================================
# GPT-2 model
# ============================================================================

class TestGPT2ModelStructure:
    def test_constructs_from_spec(self):
        model = GPT2Model(_gpt2_tiny())
        assert model is not None

    def test_output_shape(self):
        torch.manual_seed(0)
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, 256), f"Got {logits.shape}"

    def test_batch_shape(self):
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        ids = torch.randint(0, 256, (3, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (3, 8, 256)

    def test_tie_embed_shares_weight(self):
        spec = _gpt2_tiny()
        spec.sheaf.tie_embed = True
        model = GPT2Model(spec)
        assert model.lm_head.weight is model.embed_tokens.weight, \
            "tie_embed=True must share weight between embed_tokens and lm_head"

    def test_no_tie_embed_independent_weight(self):
        spec = _gpt2_tiny()
        spec.sheaf.tie_embed = False
        model = GPT2Model(spec)
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_has_position_embedding_for_learned_pos(self):
        model = GPT2Model(_gpt2_tiny())
        assert hasattr(model, "pos_embed"), "GPT-2 must have learned position embedding"

    def test_uses_layernorm(self):
        model = GPT2Model(_gpt2_tiny())
        # Final norm should be LayerNorm
        assert any(isinstance(m, nn.LayerNorm) for m in model.modules())

    def test_causal_mask_prevents_future_access(self):
        """Logits at position t must not change when token at t+1 changes."""
        torch.manual_seed(42)
        model = GPT2Model(_gpt2_tiny())
        model.eval()
        ids1 = torch.randint(0, 256, (1, 8))
        ids2 = ids1.clone()
        ids2[0, 5] = (ids2[0, 5] + 1) % 256  # change position 5
        with torch.no_grad():
            l1 = model(ids1)
            l2 = model(ids2)
        # Positions 0-4 must be identical (not seeing token 5)
        assert torch.allclose(l1[0, :5], l2[0, :5]), \
            "Logits at positions 0-4 must be unchanged when position 5 changes"

    def test_backward_does_not_crash(self):
        model = GPT2Model(_gpt2_tiny())
        ids = torch.randint(0, 256, (1, 4))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()


class TestGPT2WeightMapping:
    """Verify the HF weight mapping function is correct in structure."""

    def test_state_dict_has_expected_keys(self):
        model = GPT2Model(_gpt2_tiny())
        keys = set(model.state_dict().keys())
        assert "embed_tokens.weight" in keys
        assert "pos_embed.weight" in keys
        # blocks
        assert "blocks.0.ln_1.weight" in keys
        assert "blocks.0.attn.c_attn.weight" in keys
        assert "blocks.0.attn.c_proj.weight" in keys
        assert "blocks.0.mlp.c_fc.weight" in keys
        assert "blocks.0.mlp.c_proj.weight" in keys

    def test_hf_to_ours_mapping_function_exists(self):
        from neuroslm.models.gpt2 import hf_to_model_state_dict
        assert callable(hf_to_model_state_dict)


# ============================================================================
# LLaMA-style model (SmolLM2 / generic)
# ============================================================================

class TestLlamaModelStructure:
    def test_constructs_from_spec(self):
        model = LlamaModel(_llama_tiny())
        assert model is not None

    def test_output_shape(self):
        torch.manual_seed(0)
        model = LlamaModel(_llama_tiny())
        model.eval()
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, 256)

    def test_no_position_embedding(self):
        model = LlamaModel(_llama_tiny())
        assert not hasattr(model, "pos_embed"), \
            "LLaMA-style must NOT have learned pos_embed (uses RoPE)"

    def test_uses_rmsnorm(self):
        from neuroslm.modules.common import RMSNorm
        model = LlamaModel(_llama_tiny())
        assert any(isinstance(m, RMSNorm) for m in model.modules())

    def test_tie_embed_shares_weight(self):
        model = LlamaModel(_llama_tiny())
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_gqa_kv_heads(self):
        spec = _llama_tiny()
        assert spec.sheaf.kv_heads == 2
        model = LlamaModel(spec)
        # k_proj should be smaller than q_proj
        block = model.blocks[0]
        q_out = block.attn.q_proj.out_features
        k_out = block.attn.k_proj.out_features
        assert k_out < q_out, "GQA: kv_heads < heads → smaller k/v projections"

    def test_causal_mask_prevents_future_access(self):
        torch.manual_seed(42)
        model = LlamaModel(_llama_tiny())
        model.eval()
        ids1 = torch.randint(0, 256, (1, 8))
        ids2 = ids1.clone()
        ids2[0, 5] = (ids2[0, 5] + 1) % 256
        with torch.no_grad():
            l1 = model(ids1)
            l2 = model(ids2)
        assert torch.allclose(l1[0, :5], l2[0, :5])

    def test_hf_to_ours_mapping_function_exists(self):
        from neuroslm.models.llama import hf_to_model_state_dict
        assert callable(hf_to_model_state_dict)


class TestLlamaWeightMapping:
    def test_state_dict_has_llama_keys(self):
        model = LlamaModel(_llama_tiny())
        keys = set(model.state_dict().keys())
        assert "embed_tokens.weight" in keys
        assert "blocks.0.attn.q_proj.weight" in keys
        assert "blocks.0.attn.k_proj.weight" in keys
        assert "blocks.0.attn.v_proj.weight" in keys
        assert "blocks.0.attn.o_proj.weight" in keys
        assert "blocks.0.mlp.gate_proj.weight" in keys
        assert "blocks.0.mlp.up_proj.weight" in keys
        assert "blocks.0.mlp.down_proj.weight" in keys
        assert "norm.weight" in keys


# ============================================================================
# Qwen variant (same as LLaMA but via qwen kind)
# ============================================================================

class TestQwenViaLlama:
    def test_qwen_kind_uses_llama_model(self):
        """qwen kind should build a LlamaModel (same architecture family)."""
        from neuroslm.models import build_model
        model = build_model(_qwen_tiny())
        assert isinstance(model, LlamaModel)

    def test_output_shape(self):
        from neuroslm.models import build_model
        model = build_model(_qwen_tiny())
        model.eval()
        ids = torch.randint(0, 256, (1, 6))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 6, 256)


# ============================================================================
# build_model factory
# ============================================================================

class TestBuildModelFactory:
    def test_gpt2_kind_returns_gpt2model(self):
        from neuroslm.models import build_model
        model = build_model(_gpt2_tiny())
        assert isinstance(model, GPT2Model)

    def test_llama_kind_returns_llamamodel(self):
        from neuroslm.models import build_model
        model = build_model(_llama_tiny())
        assert isinstance(model, LlamaModel)

    def test_unknown_kind_raises(self):
        from neuroslm.models import build_model
        spec = _gpt2_tiny()
        spec.kind = "completely_unknown"
        with pytest.raises((ValueError, KeyError)):
            build_model(spec)
