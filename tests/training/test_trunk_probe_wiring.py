# -*- coding: utf-8 -*-
"""`_run_trunk_probe` wiring — the read-only discovery probe on a (fake) trunk.

Exercises the train_dsl integration without a 1.1B harness: a minimal fake
harness that stashes ``_last_hidden`` and exposes ``lm_head``, run through
`_run_trunk_probe`, must produce a summary and persist any winner under the
given root (never the repo).
"""
import torch
import torch.nn as nn

from neuroslm.dsl import nn_ops
from neuroslm.train_dsl import _run_trunk_probe


class _FakeLM(nn.Module):
    def __init__(self, d=16, v=32):
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.lm_head = nn.Parameter(torch.randn(v, d) * 0.1)
        self._last_hidden = None
        self._cosine_head = False
        self.head_temperature = 1.0


class _FakeHarness(nn.Module):
    def __init__(self, d=16, v=32):
        super().__init__()
        self.language_model = _FakeLM(d, v)

    def forward(self, ids):
        h = self.language_model.embed(ids)              # (B, T, d)
        self.language_model._last_hidden = h
        return nn_ops.linear(h, self.language_model.lm_head)


def test_probe_runs_and_persists_under_given_root(tmp_path):
    torch.manual_seed(0)
    harn = _FakeHarness()
    ids = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    out = _run_trunk_probe(harn, ids, targets, step=500, arch_name="fake",
                           preset_name="p", root=tmp_path)
    assert out is not None
    assert out["baseline_ce"] > 0
    assert out["best_ce"] <= out["baseline_ce"] + 1e-4        # floored at identity
    # a persisted winner (if any) lives under the given root, never the repo
    mods = list((tmp_path / "modulations").glob("*.neuro")) if (tmp_path / "modulations").exists() else []
    if out["saved"]:
        assert len(mods) == 1


def test_probe_returns_none_without_language_model():
    class _Bare(nn.Module):
        language_model = None
    out = _run_trunk_probe(_Bare(), torch.zeros(1, 4, dtype=torch.long),
                           torch.zeros(1, 4, dtype=torch.long),
                           step=1, arch_name="x", preset_name="y", root="/tmp")
    assert out is None


def test_real_cortex_gets_the_multi_site_probe(tmp_path):
    # The class real training builds (DSLLanguageCortex) exposes
    # forward_from_layer → the probe must take the multi-site path:
    # per-layer headroom reports + searching only sites with slack.
    from neuroslm.dsl.nn_lang import build_dsl_language_cortex

    class _Harness(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.language_model = build_dsl_language_cortex(
                vocab=61, d_model=24, depth=3, n_heads=4, max_ctx=16,
                dropout=0.0)

        def forward(self, ids):
            return self.language_model(ids)

    harn = _Harness()
    harn.train()
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 61, (2, 12), generator=g)
    targets = torch.randint(0, 61, (2, 12), generator=g)
    state = {k: v.clone() for k, v in harn.state_dict().items()}

    out = _run_trunk_probe(harn, ids, targets, step=500, arch_name="cortex",
                           preset_name="p", root=tmp_path,
                           pop=8, gens=2, sites=1)
    assert out is not None
    assert "reports" in out and len(out["reports"]) == 3   # multi-site path
    assert out["searched"] and len(out["searched"]) == 1
    assert out["best_ce"] <= out["baseline_ce"] + 1e-4
    for k, v in harn.state_dict().items():
        assert torch.equal(v, state[k]), f"probe mutated {k}"
    assert harn.language_model.training, "probe did not restore training mode"


def test_probe_only_loop_runs_without_training(tmp_path):
    # `--explore_only`: discovery on the current model state, no optimizer, no
    # backward — weights must be untouched across every round, and each round
    # consumes a FRESH batch (recurrence across batches is the install filter).
    from neuroslm.dsl.nn_lang import build_dsl_language_cortex
    from neuroslm.train_dsl import run_probe_only

    class _Harness(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.language_model = build_dsl_language_cortex(
                vocab=61, d_model=24, depth=3, n_heads=4, max_ctx=16,
                dropout=0.0)

    class _Source:
        def __init__(self):
            self.calls = 0
            self._g = torch.Generator().manual_seed(9)

        def next(self):
            self.calls += 1
            return (torch.randint(0, 61, (2, 12), generator=self._g),
                    torch.randint(0, 61, (2, 12), generator=self._g))

    harn, src = _Harness(), _Source()
    state = {k: v.clone() for k, v in harn.state_dict().items()}
    results = run_probe_only(harn, src, rounds=3, arch_name="cortex",
                             preset_name="p", root=tmp_path,
                             pop=6, gens=2, length=5, sites=1, push=False)
    assert len(results) == 3
    assert src.calls == 3                      # fresh batch per round
    for k, v in harn.state_dict().items():
        assert torch.equal(v, state[k]), f"probe-only loop mutated {k}"
