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
    assert (tmp_path / ".neuro" / "search_ledger.json").exists()


def test_probe_returns_none_without_language_model():
    class _Bare(nn.Module):
        language_model = None
    out = _run_trunk_probe(_Bare(), torch.zeros(1, 4, dtype=torch.long),
                           torch.zeros(1, 4, dtype=torch.long),
                           step=1, arch_name="x", preset_name="y", root="/tmp")
    assert out is None
