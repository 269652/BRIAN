# -*- coding: utf-8 -*-
"""Standard LM model factory.

`build_model(spec)` returns a TransformerLM configured entirely by the
ModelSpec DSL — no enum dispatch, no class lookup table.
The coboundary/transition/norm/embed blocks in arch.neuro ARE the mechanic.
"""
from __future__ import annotations

from neuroslm.dsl.model_spec import ModelSpec
from neuroslm.models.transformer_lm import TransformerLM


def build_model(spec: ModelSpec) -> TransformerLM:
    """Instantiate a TransformerLM from a ModelSpec DSL spec.

    The spec's sheaf sub-blocks determine every architectural choice:
      coboundary.type  mha/gqa/swa/kjpla → attention operator
      transition.type  mlp/swiglu/geglu   → FFN operator
      norm.type        layernorm/rmsnorm  → normalizer
      embed.position   learned/none       → positional encoding
    """
    return TransformerLM(spec)
