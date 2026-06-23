# -*- coding: utf-8 -*-
"""Standard LM model implementations (GPT-2, LLaMA/SmolLM2/Qwen families).

Use `build_model(spec: ModelSpec) -> nn.Module` to get a model from a
DSL v2 ModelSpec.  The returned model has parameter names compatible with
the HF weight-loading adapters in each module.
"""
from __future__ import annotations

from neuroslm.dsl.model_spec import ModelSpec

_KIND_TO_CLASS = None  # lazy import to avoid circular deps at package load


def build_model(spec: ModelSpec):
    """Factory: return an nn.Module for the given ModelSpec.

    kind=gpt2  → GPT2Model
    kind=llama → LlamaModel
    kind=qwen  → LlamaModel  (same architecture family)
    """
    global _KIND_TO_CLASS
    if _KIND_TO_CLASS is None:
        from neuroslm.models.gpt2 import GPT2Model
        from neuroslm.models.llama import LlamaModel
        _KIND_TO_CLASS = {
            "gpt2": GPT2Model,
            "llama": LlamaModel,
            "qwen": LlamaModel,   # Qwen2 = LLaMA architecture
            "mistral": LlamaModel,
        }

    cls = _KIND_TO_CLASS.get(spec.kind)
    if cls is None:
        raise ValueError(
            f"Unknown model kind {spec.kind!r}; "
            f"expected one of {sorted(_KIND_TO_CLASS)}"
        )
    return cls(spec)
