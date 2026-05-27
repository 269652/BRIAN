# -*- coding: utf-8 -*-
"""Phase N6 — bridge a BrainConfig preset to the DSL transformer LM.

Training performance (loss / perplexity) is determined by the transformer
trunk. To match `rcc_bowtie_30m_p4`, the DSL LM must be built at that
preset's exact trunk dimensions (d_hidden, lang_layers, lang_heads,
lang_ctx, vocab). This bridge reads the preset and sizes the DSL LM
accordingly, so a DSL run is architecturally the same language model as
P4's trunk.
"""
import pytest
import torch

from neuroslm.dsl.preset_bridge import dsl_lm_config_from_preset, build_lm_from_preset


class TestConfigBridge:
    def test_reads_p4_trunk_dims(self):
        cfg = dsl_lm_config_from_preset("rcc_bowtie_30m_p4")
        # P4 trunk: d_hidden=384, 4 layers, 6 heads, ctx=1024, vocab=50257
        assert cfg["d_model"] == 384
        assert cfg["depth"] == 4
        assert cfg["n_heads"] == 6
        assert cfg["vocab"] == 50257
        assert cfg["max_ctx"] == 1024

    def test_head_dim_divisible(self):
        cfg = dsl_lm_config_from_preset("rcc_bowtie_30m_p4")
        assert cfg["d_model"] % cfg["n_heads"] == 0

    def test_unknown_preset_errors(self):
        with pytest.raises(KeyError):
            dsl_lm_config_from_preset("does_not_exist_preset")


class TestBuildFromPreset:
    def test_builds_runnable_lm(self):
        # Build at the real P4 trunk config but tiny vocab/ctx for a fast
        # CPU forward sanity check (dims that matter for wiring are kept).
        lm = build_lm_from_preset("rcc_bowtie_30m_p4",
                                   vocab_override=256, max_ctx_override=64)
        ids = torch.randint(0, 256, (2, 16))
        logits = lm(ids)
        assert logits.shape == (2, 16, 256)
        assert not torch.isnan(logits).any()

    def test_param_count_reasonable_at_full_scale(self):
        # At full P4 trunk config the DSL LM should be tens of millions of
        # params (dominated by the 50257-row embedding + lm_head).
        lm = build_lm_from_preset("rcc_bowtie_30m_p4")
        n = sum(p.numel() for p in lm.parameters())
        # embedding + head alone: 2 * 50257 * 384 ≈ 38.6M
        assert n > 30e6, f"param count {n/1e6:.1f}M unexpectedly small"
        assert n < 120e6, f"param count {n/1e6:.1f}M unexpectedly large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
