# -*- coding: utf-8 -*-
"""LLaMA-family compatibility shim — mechanics are expressed in arch.neuro DSL.

The DSL coboundary { type: gqa, rope { ... }, qkv_bias: ... }
+ transition { type: swiglu ... } drives the generic TransformerLM.
This module keeps legacy import paths working.
"""
from neuroslm.models.transformer_lm import TransformerLM, hf_to_model_state_dict

LlamaModel = TransformerLM

__all__ = ["LlamaModel", "hf_to_model_state_dict"]
