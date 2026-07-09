# -*- coding: utf-8 -*-
"""Layer headroom scan + multi-site probe — search only *optimizable* regions.

H46 proved the terminal hidden state is a null site: it is the single most
end-to-end-optimized point in the network, so a fixed post-hoc modulation there
has nothing to exploit. The fix is not a deeper search at a dead site — it is
measuring, per layer, whether the TRUE loss still has slack under cheap
structured perturbations of that layer's output (re-run through the real tail),
then spending the GA budget only where slack exists.
"""
import math

import torch
import torch.nn.functional as F

from neuroslm.genetic.layer_probe import (
    SiteReport,
    headroom_scan,
    select_sites,
    probe_optimizable_regions,
)


# ── synthetic tails with KNOWN optimization structure ──────────────────────
def _make_synthetic(vocab=4, B=2, T=6, D=8):
    """Three sites with controlled loss geometry, sharing one baseline
    (as forward_from_layer exactness guarantees on a real sequential net):

    site 0 — tail ignores the hidden entirely       → insensitive (skip)
    site 1 — CE minimized at 0.8*h (not at h)       → damping improves (slack)
    site 2 — CE minimized exactly at the given h    → any perturbation hurts

    Each tail's correct-class logit is offset so that the UNPERTURBED hidden
    yields z=0 at every site — one consistent baseline CE across sites.
    """
    g = torch.Generator().manual_seed(0)
    layers = [torch.randn(B, T, D, generator=g) for _ in range(3)]
    targets = torch.zeros(B, T, dtype=torch.long)   # true class is always 0
    opt1 = layers[1] * 0.8                          # site 1's optimum ≠ its hidden
    off1 = ((layers[1] - opt1) ** 2).mean(dim=-1)   # unperturbed dist at site 1

    def tail_fn(k, h):
        logits = torch.zeros(h.shape[0], h.shape[1], vocab)
        if k == 0:
            return logits                            # constant → CE constant
        if k == 1:
            z = -((h - opt1) ** 2).mean(dim=-1) + off1   # z=0 unperturbed
        else:
            z = -((h - layers[2]) ** 2).mean(dim=-1)     # z=0 unperturbed (at opt)
        logits[:, :, 0] = z
        return logits

    return layers, tail_fn, targets


class TestHeadroomScan:
    def test_scan_measures_slack_where_it_exists(self):
        layers, tail_fn, targets = _make_synthetic()
        reports = headroom_scan(layers, tail_fn, targets, seed=0)
        assert len(reports) == 3
        assert all(isinstance(r, SiteReport) for r in reports)
        r0, r1, r2 = reports
        # site 0: the tail ignores it — no leverage at all
        assert r0.sensitivity < 1e-9
        # site 1: damping reaches the optimum — measurable positive slack
        assert r1.improvement > 1e-4
        # site 2: already optimal — every battery perturbation hurts
        assert r2.improvement < 0

    def test_select_sites_prefers_slack_and_skips_insensitive(self):
        layers, tail_fn, targets = _make_synthetic()
        reports = headroom_scan(layers, tail_fn, targets, seed=0)
        picked = select_sites(reports, top_k=2)
        assert picked[0] == 1              # the site with real slack leads
        assert 0 not in picked             # insensitive site is never searched

    def test_select_sites_falls_back_to_most_promising_when_no_slack(self):
        layers, tail_fn, targets = _make_synthetic()
        reports = headroom_scan(layers, tail_fn, targets, seed=0)
        # drop the slack site — only the insensitive + the converged remain
        rest = [r for r in reports if r.layer != 1]
        picked = select_sites(rest, top_k=1)
        assert picked == [2]               # speculative: least-negative, sensitive


class TestProbeOptimizableRegions:
    def _cortex(self, depth=3):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        torch.manual_seed(0)
        return build_dsl_language_cortex(vocab=61, d_model=24, depth=depth,
                                         n_heads=4, max_ctx=16, dropout=0.0)

    def test_end_to_end_on_real_trunk(self, tmp_path):
        from neuroslm.genetic.modulation_store import ModulationStore
        from neuroslm.genetic.training_explorer import ExploreConfig
        m = self._cortex()
        m.train()          # probe must handle (and restore) training mode
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, 61, (2, 12), generator=g)
        targets = torch.randint(0, 61, (2, 12), generator=g)
        state = {k: v.clone() for k, v in m.state_dict().items()}

        out = probe_optimizable_regions(
            m, ids, targets, store=ModulationStore(tmp_path / "mods"),
            config=ExploreConfig(pop_size=8, generations=2), step=500,
            run_id="test", top_k=1, seed=0)

        assert set(out) >= {"baseline_ce", "best_ce", "delta_ce", "improved",
                            "saved", "evaluated", "reports", "searched"}
        assert math.isfinite(out["baseline_ce"]) and out["baseline_ce"] > 0
        assert out["best_ce"] <= out["baseline_ce"] + 1e-4   # floored at identity
        assert len(out["reports"]) == 3                       # one per layer
        assert out["searched"] and all(0 <= k < 3 for k in out["searched"])
        # strictly read-only + mode restored
        for k, v in m.state_dict().items():
            assert torch.equal(v, state[k]), f"probe mutated {k}"
        assert m.training, "probe did not restore training mode"
        if out["improved"]:
            assert out["saved"]
            recs = ModulationStore(tmp_path / "mods").list_all()
            assert len(recs) >= 1

    def test_probe_baseline_matches_true_forward_ce(self):
        m = self._cortex()
        g = torch.Generator().manual_seed(2)
        ids = torch.randint(0, 61, (2, 10), generator=g)
        targets = torch.randint(0, 61, (2, 10), generator=g)
        from neuroslm.genetic.training_explorer import ExploreConfig
        out = probe_optimizable_regions(
            m, ids, targets, config=ExploreConfig(pop_size=6, generations=1),
            step=0, top_k=1, seed=0)
        m.eval()
        with torch.no_grad():
            ref = float(F.cross_entropy(m(ids).reshape(-1, 61),
                                        targets.reshape(-1)))
        assert abs(out["baseline_ce"] - ref) < 1e-4
