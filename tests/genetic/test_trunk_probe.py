# -*- coding: utf-8 -*-
"""Read-only trunk probe — gather first discovery data on a real hidden state.

The probe searches a residual modulation of the trunk's final hidden state and
re-projects the LM head to measure whether it lowers next-token CE — without ever
touching the training forward or weights (so it can't perturb a live run).
"""
import torch
import torch.nn as nn

from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.modulation_store import ModulationStore
from neuroslm.genetic.training_explorer import probe_hidden_modulation, ExploreConfig


def _fixture(seed=0, B=2, T=6, D=16, V=32):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, T, D, generator=g)
    head = nn.Linear(D, V, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(V, D, generator=g) * 0.1)
    targets = torch.randint(0, V, (B, T), generator=g)
    return hidden, (lambda h: head(h)), targets


class TestProbe:
    def test_returns_a_summary_without_touching_inputs(self):
        hidden, head_fn, targets = _fixture()
        h0 = hidden.clone()
        led = SearchLedger(":memory:")
        out = probe_hidden_modulation(hidden, head_fn, targets, ledger=led,
                                      config=ExploreConfig(pop_size=8, generations=3),
                                      step=500, run_id="run-0")
        assert set(out) >= {"baseline_ce", "best_ce", "delta_ce", "improved", "saved"}
        assert out["baseline_ce"] > 0
        # the probe is read-only — it must not mutate the hidden tensor
        assert torch.allclose(hidden, h0)

    def test_baseline_matches_unmodulated_head(self):
        import torch.nn.functional as F
        hidden, head_fn, targets = _fixture(seed=1)
        led = SearchLedger(":memory:")
        out = probe_hidden_modulation(hidden, head_fn, targets, ledger=led,
                                      config=ExploreConfig(pop_size=6, generations=2),
                                      step=0)
        ref = float(F.cross_entropy(head_fn(hidden).reshape(-1, 32), targets.reshape(-1)))
        assert abs(out["baseline_ce"] - ref) < 1e-4

    def test_best_never_worse_than_baseline(self):
        # identity is always in the search seed, so best ≤ baseline
        hidden, head_fn, targets = _fixture(seed=2)
        led = SearchLedger(":memory:")
        out = probe_hidden_modulation(hidden, head_fn, targets, ledger=led,
                                      config=ExploreConfig(pop_size=8, generations=3),
                                      step=1)
        assert out["best_ce"] <= out["baseline_ce"] + 1e-4
        assert out["delta_ce"] >= -1e-4

    def test_improvement_is_persisted_with_delta(self, tmp_path):
        hidden, head_fn, targets = _fixture(seed=3)
        led = SearchLedger(":memory:")
        store = ModulationStore(tmp_path / "mods")
        out = probe_hidden_modulation(hidden, head_fn, targets, ledger=led,
                                      store=store,
                                      config=ExploreConfig(pop_size=10, generations=4),
                                      step=500, run_id="run-0")
        if out["improved"]:
            assert out["saved"] is not None
            recs = store.list_all()
            assert len(recs) == 1
            assert recs[0].metrics["delta_ce"] > 0
        else:
            assert store.list_all() == []
