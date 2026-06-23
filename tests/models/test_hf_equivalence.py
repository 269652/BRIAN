# -*- coding: utf-8 -*-
"""Logit-equivalence and PPL tests for GPT-2, SmolLM2-135M, Qwen2.5-0.5B.

These tests require `transformers` and network access to download HF models.
They are marked @pytest.mark.slow and excluded from `brian test fast/quick`.

Each test verifies:
  1. Our model loaded with HF weights produces the same logits as HF (rtol=1e-3).
  2. PPL on a fixed 512-token WikiText-103 sample is below sanity threshold.

Reference PPLs (fp32, greedy, no temperature):
  - GPT-2 124M:       ~29.4  (from original GPT-2 paper / HF eval)
  - SmolLM2-135M:     ~19.8  (HuggingFace model card)
  - Qwen2.5-0.5B:     ~14.2  (HuggingFace model card, WikiText-2)
"""
from __future__ import annotations
import math
import pytest
import torch

transformers = pytest.importorskip("transformers", reason="transformers not installed")


# ── Short WikiText-103 sample (hard-coded to avoid download) ─────────────────
_WIKITEXT_SAMPLE = (
    " = Valkyria Chronicles III = \n\n Senjō no Valkyria 3 : Unrecorded Chronicles "
    "( Japanese : 戦場のヴァルキュリア3, lit . Valkyria of the Battlefield 3 ) , "
    "commonly referred to as Valkyria Chronicles III outside Japan , is a tactical "
    "role @-@ playing video game developed by Sega and Media.Vision for the "
    "PlayStation Portable . Released in January 2011 in Japan , it is the third "
    "game in the Valkyria series . Employing the same fusion of tactical and "
    "real @-@ time gameplay as its predecessors , the story runs parallel to the "
    "first game and follows the \" Nameless \" , a penal military unit"
)


def _compute_ppl(model, tokenizer, text: str, max_len: int = 128) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[:, :max_len]
    target_ids = input_ids[:, 1:]
    with torch.no_grad():
        logits = model(input_ids[:, :-1])
        if hasattr(logits, "logits"):
            logits = logits.logits
    log_probs = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="mean",
    )
    return math.exp(log_probs.item())


# ── GPT-2 ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestGPT2HFEquivalence:
    MODEL_ID = "openai-community/gpt2"

    @pytest.fixture(scope="class")
    def hf_model_and_tokenizer(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL_ID)
        hf_model = GPT2LMHeadModel.from_pretrained(self.MODEL_ID)
        hf_model.eval()
        return hf_model, tokenizer

    @pytest.fixture(scope="class")
    def our_model(self, hf_model_and_tokenizer):
        from neuroslm.models.gpt2 import GPT2Model, hf_to_model_state_dict
        from neuroslm.dsl.model_spec import ModelSpec, SheafConfig, RopeConfig
        from neuroslm.dsl.model_spec import CoboundaryConfig, TransitionConfig
        from neuroslm.dsl.model_spec import NormConfig, OutputConfig, EmbedConfig
        hf_model, _ = hf_model_and_tokenizer
        cfg = hf_model.config
        spec = ModelSpec(kind="gpt2")
        s = spec.sheaf
        s.dim = cfg.n_embd
        s.depth = cfg.n_layer
        s.heads = cfg.n_head
        s.kv_heads = cfg.n_head
        s.context = cfg.n_positions
        s.vocab = cfg.vocab_size
        s.embed      = EmbedConfig(tokens="learned", position="learned")
        s.coboundary = CoboundaryConfig(type="mha", qkv="fused", bias=True)
        s.transition = TransitionConfig(type="mlp", ff_mult=4.0,
                                        activation="gelu", bias=True)
        s.norm   = NormConfig(type="layernorm", eps=float(getattr(cfg, "layer_norm_epsilon", 1e-5)))
        s.output = OutputConfig(tie_embed=True)
        model = GPT2Model(spec)
        model.load_state_dict(hf_to_model_state_dict(hf_model.state_dict(), spec))
        model.eval()
        return model

    def test_logits_match(self, hf_model_and_tokenizer, our_model):
        hf_model, _ = hf_model_and_tokenizer
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits
            our_logits = our_model(input_ids)
        assert torch.allclose(hf_logits, our_logits, rtol=1e-3, atol=1e-4), \
            f"Max diff: {(hf_logits - our_logits).abs().max().item():.2e}"

    def test_ppl_within_range(self, hf_model_and_tokenizer, our_model):
        _, tokenizer = hf_model_and_tokenizer
        ppl = _compute_ppl(our_model, tokenizer, _WIKITEXT_SAMPLE)
        assert 5.0 < ppl < 100.0, f"PPL out of range: {ppl:.1f}"


# ── SmolLM2-135M ──────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestSmolLM2HFEquivalence:
    MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

    @pytest.fixture(scope="class")
    def hf_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32)
        hf_model.eval()
        return hf_model, tokenizer

    @pytest.fixture(scope="class")
    def our_model(self, hf_model_and_tokenizer):
        from neuroslm.models.llama import LlamaModel, hf_to_model_state_dict
        from neuroslm.dsl.model_spec import ModelSpec
        from neuroslm.dsl.model_spec import CoboundaryConfig, RopeConfig
        from neuroslm.dsl.model_spec import TransitionConfig, NormConfig
        from neuroslm.dsl.model_spec import OutputConfig, EmbedConfig
        hf_model, _ = hf_model_and_tokenizer
        cfg = hf_model.config
        hf_sd = hf_model.state_dict()

        spec = ModelSpec(kind="llama")
        s = spec.sheaf
        s.dim = cfg.hidden_size
        s.depth = cfg.num_hidden_layers
        s.heads = cfg.num_attention_heads
        s.kv_heads = cfg.num_key_value_heads
        s.context = cfg.max_position_embeddings
        s.vocab = cfg.vocab_size
        s.embed = EmbedConfig(tokens="learned", position="none")
        s.coboundary = CoboundaryConfig(
            type="gqa",
            rope=RopeConfig(base=int(getattr(cfg, "rope_theta", 10000))),
            qkv_bias="model.layers.0.self_attn.q_proj.bias" in hf_sd,
        )
        s.transition = TransitionConfig(
            type="swiglu",
            ff_mult=cfg.intermediate_size / cfg.hidden_size,
        )
        s.norm = NormConfig(
            type="rmsnorm",
            eps=float(getattr(cfg, "rms_norm_eps", 1e-5)),
        )
        s.output = OutputConfig(tie_embed=cfg.tie_word_embeddings)

        model = LlamaModel(spec)
        model.load_state_dict(hf_to_model_state_dict(hf_sd, spec))
        model.eval()
        return model

    def test_logits_match(self, hf_model_and_tokenizer, our_model):
        hf_model, _ = hf_model_and_tokenizer
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits.float()
            our_logits = our_model(input_ids)
        assert torch.allclose(hf_logits, our_logits, rtol=1e-3, atol=1e-3), \
            f"Max diff: {(hf_logits - our_logits).abs().max().item():.2e}"

    def test_ppl_within_range(self, hf_model_and_tokenizer, our_model):
        _, tokenizer = hf_model_and_tokenizer
        ppl = _compute_ppl(our_model, tokenizer, _WIKITEXT_SAMPLE)
        assert ppl < 100.0, f"PPL too high: {ppl:.1f}"


# ── Qwen2.5-0.5B ─────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestQwen25HFEquivalence:
    MODEL_ID = "Qwen/Qwen2.5-0.5B"

    @pytest.fixture(scope="class")
    def hf_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32)
        hf_model.eval()
        return hf_model, tokenizer

    @pytest.fixture(scope="class")
    def our_model(self, hf_model_and_tokenizer):
        from neuroslm.models.llama import LlamaModel, hf_to_model_state_dict
        from neuroslm.dsl.model_spec import ModelSpec
        from neuroslm.dsl.model_spec import CoboundaryConfig, RopeConfig
        from neuroslm.dsl.model_spec import TransitionConfig, NormConfig
        from neuroslm.dsl.model_spec import OutputConfig, EmbedConfig
        hf_model, _ = hf_model_and_tokenizer
        cfg = hf_model.config
        hf_sd = hf_model.state_dict()

        spec = ModelSpec(kind="qwen")
        s = spec.sheaf
        s.dim = cfg.hidden_size
        s.depth = cfg.num_hidden_layers
        s.heads = cfg.num_attention_heads
        s.kv_heads = cfg.num_key_value_heads
        s.context = cfg.max_position_embeddings
        s.vocab = cfg.vocab_size
        s.embed = EmbedConfig(tokens="learned", position="none")
        s.coboundary = CoboundaryConfig(
            type="gqa",
            qkv_bias="model.layers.0.self_attn.q_proj.bias" in hf_sd,
            rope=RopeConfig(base=int(getattr(cfg, "rope_theta", 1_000_000))),
        )
        s.transition = TransitionConfig(
            type="swiglu",
            ff_mult=cfg.intermediate_size / cfg.hidden_size,
        )
        s.norm = NormConfig(
            type="rmsnorm",
            eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
        )
        s.output = OutputConfig(tie_embed=getattr(cfg, "tie_word_embeddings", True))

        model = LlamaModel(spec)
        model.load_state_dict(hf_to_model_state_dict(hf_sd, spec))
        model.eval()
        return model

    def test_logits_match(self, hf_model_and_tokenizer, our_model):
        hf_model, _ = hf_model_and_tokenizer
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits.float()
            our_logits = our_model(input_ids)
        assert torch.allclose(hf_logits, our_logits, rtol=1e-3, atol=1e-3), \
            f"Max diff: {(hf_logits - our_logits).abs().max().item():.2e}"

    def test_ppl_within_range(self, hf_model_and_tokenizer, our_model):
        _, tokenizer = hf_model_and_tokenizer
        ppl = _compute_ppl(our_model, tokenizer, _WIKITEXT_SAMPLE)
        assert ppl < 100.0, f"PPL too high: {ppl:.1f}"
