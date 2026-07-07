# -*- coding: utf-8 -*-
"""Per-arch/preset run heatmap store.

Each training run records the latest heatmap for its (arch, preset) so you can
always see where the gradient/information heat concentrated on the most recent run
of each configuration — the map that shows *where* a wild gnorm lives.
"""
import json

from neuroslm.genetic.heatmap_store import (
    HeatmapStore,
    RunHeatmap,
    heatmap_from_grad_norms,
)


class TestHeatmapFromGradNorms:
    def test_ranks_hot_pathways(self):
        rh = heatmap_from_grad_norms(
            "architectures/SmolLM", "rcc_bowtie_30m_p4",
            {"blocks.0.attn.Wq": 80.0, "blocks.0.mlp.w1": 5.0, "embed": 0.5},
            step=160, git_commit="deadbeef", top_k=2)
        assert rh.arch == "SmolLM"          # basename-normalised
        assert rh.preset == "rcc_bowtie_30m_p4"
        assert rh.step == 160
        # hottest pathway first
        assert rh.summary["hot"][0][0] == "blocks.0.attn.Wq"
        assert rh.summary["max"] == 80.0
        assert len(rh.summary["hot"]) == 2


class TestStore:
    def test_path_is_arch_preset_scoped(self, tmp_path):
        store = HeatmapStore(tmp_path)
        p = store.path("SmolLM", "rcc_bowtie_30m_p4")
        assert p.name == "rcc_bowtie_30m_p4.json"
        assert p.parent.name == "SmolLM"

    def test_record_then_load_roundtrips(self, tmp_path):
        store = HeatmapStore(tmp_path)
        rh = heatmap_from_grad_norms("SmolLM", "p4", {"a": 3.0, "b": 1.0}, step=100)
        store.record(rh)
        loaded = store.load("SmolLM", "p4")
        assert loaded.step == 100
        assert loaded.entries == {"a": 3.0, "b": 1.0}
        # json on disk is valid + scoped
        json.loads(store.path("SmolLM", "p4").read_text())

    def test_latest_run_overwrites(self, tmp_path):
        store = HeatmapStore(tmp_path)
        store.record(heatmap_from_grad_norms("SmolLM", "p4", {"a": 1.0}, step=100))
        store.record(heatmap_from_grad_norms("SmolLM", "p4", {"a": 9.0}, step=500))
        loaded = store.load("SmolLM", "p4")
        assert loaded.step == 500          # latest wins
        assert loaded.entries == {"a": 9.0}

    def test_list_all_finds_configs(self, tmp_path):
        store = HeatmapStore(tmp_path)
        store.record(heatmap_from_grad_norms("SmolLM", "p4", {"a": 1.0}, step=1))
        store.record(heatmap_from_grad_norms("gpt2", "small", {"b": 1.0}, step=1))
        got = set(store.list_all())
        assert ("SmolLM", "p4") in got and ("gpt2", "small") in got


class TestCollectorFromModel:
    def test_records_from_named_parameters(self, tmp_path):
        import torch
        import torch.nn as nn
        from neuroslm.genetic.heatmap_store import record_training_run

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        x = torch.randn(3, 4)
        model(x).sum().backward()   # populate .grad

        store = HeatmapStore(tmp_path)
        rh = record_training_run(store, "SmolLM", "p4", model, step=42)
        assert rh.step == 42
        assert store.load("SmolLM", "p4").step == 42
        assert len(rh.entries) > 0   # captured per-parameter grad heat
