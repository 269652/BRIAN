# -*- coding: utf-8 -*-
"""GPT-2 compatibility shim — mechanics are expressed in arch.neuro DSL.

The DSL coboundary { type: mha, bias: true } + transition { type: mlp ... }
drives the generic TransformerLM. This module keeps legacy import paths working.
"""
from neuroslm.models.transformer_lm import TransformerLM, hf_to_model_state_dict

GPT2Model = TransformerLM

__all__ = ["GPT2Model", "hf_to_model_state_dict"]
