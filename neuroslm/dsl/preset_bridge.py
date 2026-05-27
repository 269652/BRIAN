# -*- coding: utf-8 -*-
"""Bridge a BrainConfig preset to the DSL transformer LM (Phase N6).

Training performance is dominated by the transformer trunk. To match a
preset like `rcc_bowtie_30m_p4`, the DSL LM is built at that preset's
exact trunk dimensions — same d_model (d_hidden), depth (lang_layers),
heads (lang_heads), context (lang_ctx), and vocab. This makes a DSL
training run architecturally the same language model as the preset's
trunk, so its loss curve is directly comparable.

The cognitive bowtie subsystems (Phases N7+) layer on top of this trunk;
they shape Φ/IIT/reasoning but not the raw LM loss, which the trunk
determines.
"""
from __future__ import annotations
from typing import Dict, Optional


def dsl_lm_config_from_preset(preset_name: str) -> Dict:
    """Read a BrainConfig preset and extract the DSL LM trunk dimensions.

    Returns a dict with: d_model, depth, n_heads, n_kv_heads, max_ctx, vocab.
    Raises KeyError for an unknown preset.
    """
    from neuroslm.config import PRESETS
    if preset_name not in PRESETS:
        raise KeyError(f"unknown preset {preset_name!r}; "
                       f"available: {sorted(PRESETS)}")
    c = PRESETS[preset_name]()

    n_heads = getattr(c, "lang_heads", None) or getattr(c, "n_heads", 8)
    kv = getattr(c, "lang_kv_heads", None)
    return {
        "d_model": c.d_hidden,
        "depth": c.lang_layers,
        "n_heads": n_heads,
        "n_kv_heads": kv if kv else n_heads,
        "max_ctx": c.lang_ctx,
        "vocab": c.vocab_size,
    }


def build_lm_from_preset(preset_name: str,
                         vocab_override: Optional[int] = None,
                         max_ctx_override: Optional[int] = None):
    """Build the DSL transformer LM at a preset's trunk dimensions.

    Overrides for vocab/max_ctx keep tests fast while preserving the
    architecture-defining dims (d_model, depth, heads).
    """
    from neuroslm.dsl.nn_lang import build_language_model
    cfg = dsl_lm_config_from_preset(preset_name)
    vocab = vocab_override or cfg["vocab"]
    max_ctx = max_ctx_override or cfg["max_ctx"]
    return build_language_model(
        vocab=vocab, d_model=cfg["d_model"], depth=cfg["depth"],
        n_heads=cfg["n_heads"], max_ctx=max_ctx, n_kv_heads=cfg["n_kv_heads"],
    )
