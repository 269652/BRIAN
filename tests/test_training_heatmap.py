# -*- coding: utf-8 -*-
"""TDD: TrainingHeatmap (L1) — incremental per-element heat over the IR.

Heat is an EMA of a per-element signal (gradient magnitude) accumulated
during training, keyed by HypergraphIR ids (nodes = modules, edges =
paths). The heatmap is the substrate for hot-path identification and
epigenetic mutation.
"""
import tempfile
from pathlib import Path

import pytest

from neuroslm.evolution.heatmap import TrainingHeatmap, HeatEntry


class TestIncrementalUpdate:
    def test_first_update_sets_heat_to_signal(self):
        hm = TrainingHeatmap(beta=0.2)
        hm.update({"population:gws": 0.5}, kinds={"population:gws": "node"})
        assert hm.heat("population:gws") == pytest.approx(0.5)
        assert hm.entries["population:gws"].kind == "node"

    def test_ema_moves_toward_new_signal(self):
        hm = TrainingHeatmap(beta=0.5)
        hm.update({"e": 1.0}, kinds={"e": "edge"})
        hm.update({"e": 0.0})
        # EMA with beta=0.5: 0.5*1.0 + 0.5*0.0 = 0.5
        assert hm.heat("e") == pytest.approx(0.5)

    def test_step_and_update_counts_advance(self):
        hm = TrainingHeatmap()
        hm.update({"a": 0.1}, kinds={"a": "node"})
        hm.update({"a": 0.2})
        hm.update({"a": 0.3})
        assert hm.step == 3
        assert hm.entries["a"].updates == 3

    def test_explicit_step_is_recorded(self):
        hm = TrainingHeatmap()
        hm.update({"a": 0.1}, kinds={"a": "node"}, step=500)
        assert hm.step == 500

    def test_new_elements_added_on_later_updates(self):
        hm = TrainingHeatmap()
        hm.update({"a": 0.1}, kinds={"a": "node"})
        hm.update({"b": 0.9}, kinds={"b": "edge"})
        assert set(hm.entries) == {"a", "b"}
        assert hm.entries["b"].kind == "edge"


class TestNormalizationAndRanking:
    def test_normalized_is_in_unit_interval_with_max_one(self):
        hm = TrainingHeatmap()
        hm.update({"a": 0.2, "b": 0.8, "c": 0.4},
                  kinds={"a": "node", "b": "node", "c": "edge"})
        norm = hm.normalized()
        assert max(norm.values()) == pytest.approx(1.0)
        assert all(0.0 <= v <= 1.0 for v in norm.values())
        assert norm["b"] == pytest.approx(1.0)

    def test_rank_is_descending_by_heat(self):
        hm = TrainingHeatmap()
        hm.update({"a": 0.2, "b": 0.8, "c": 0.4},
                  kinds={"a": "node", "b": "node", "c": "node"})
        ranked = hm.rank()
        assert [i for i, _ in ranked] == ["b", "c", "a"]


class TestHotColdIdentification:
    def test_hot_and_cold_paths_by_normalized_threshold(self):
        hm = TrainingHeatmap()
        hm.update(
            {"hot1": 1.0, "hot2": 0.9, "mid": 0.4, "cold": 0.02},
            kinds={"hot1": "edge", "hot2": "node", "mid": "edge", "cold": "edge"},
        )
        hot = set(hm.hot_paths(threshold=0.7))
        cold = set(hm.cold_paths(threshold=0.1))
        assert hot == {"hot1", "hot2"}
        assert "cold" in cold and "hot1" not in cold

    def test_kind_filter(self):
        hm = TrainingHeatmap()
        hm.update(
            {"n": 1.0, "e": 0.95},
            kinds={"n": "node", "e": "edge"},
        )
        assert hm.hot_paths(threshold=0.7, kind="edge") == ["e"]
        assert hm.hot_paths(threshold=0.7, kind="node") == ["n"]

    def test_empty_heatmap_has_no_hot_or_cold(self):
        hm = TrainingHeatmap()
        assert hm.normalized() == {}
        assert hm.hot_paths() == []
        assert hm.cold_paths() == []


class TestPersistence:
    def test_to_from_dict_roundtrip(self):
        hm = TrainingHeatmap(beta=0.3)
        hm.update({"a": 0.5, "b": 0.1}, kinds={"a": "node", "b": "edge"}, step=10)
        hm2 = TrainingHeatmap.from_dict(hm.to_dict())
        assert hm2.step == 10
        assert hm2.beta == pytest.approx(0.3)
        assert hm2.heat("a") == pytest.approx(hm.heat("a"))
        assert hm2.entries["b"].kind == "edge"

    def test_save_and_load_incrementally(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "heatmap.json"
            hm = TrainingHeatmap()
            hm.update({"a": 0.2}, kinds={"a": "node"})
            hm.save(str(path))
            # Update again and overwrite — load reflects the latest state.
            hm.update({"a": 0.6})
            hm.save(str(path))
            loaded = TrainingHeatmap.load(str(path))
            assert loaded.step == 2
            assert loaded.heat("a") == pytest.approx(hm.heat("a"))

    def test_saved_json_is_human_readable_dict(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "heatmap.json"
            hm = TrainingHeatmap()
            hm.update({"population:gws": 0.7}, kinds={"population:gws": "node"})
            hm.save(str(path))
            blob = json.loads(path.read_text())
            assert "entries" in blob and "step" in blob
            assert blob["entries"]["population:gws"]["kind"] == "node"
