# -*- coding: utf-8 -*-
"""Expert-cortex probe — discovery on the frozen pretrained LMs (H53 follow-up).

The frozen experts (SmolLM2/CodeGPT/Qwen, PPL≈50 territory) are the strongest
discovery target: they were optimized for THEIR pretraining distribution, not
BRIAN's data mixture, so domain-shift slack is well-defined — and durable,
because frozen weights never move under a banked winner. The probe searches an
NGL modulation of the expert's final hidden, scored by the expert's OWN
next-token CE in its OWN token space (no vocab bridge — that's a distillation
concern, not a discovery one).

All tests run offline against a fake expert (tiny backbone + head + byte-level
tokenizer) — no transformers download.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from neuroslm.genetic.expert_probe import (
    ProbedExpert,
    expert_batch,
    probe_expert,
    run_expert_discovery,
    texts_from_stream,
)


# ── offline fake expert ─────────────────────────────────────────────────────
class _FakeTokenizer:
    """Byte-level: every character is a token id (mod vocab)."""
    def __init__(self, vocab=97):
        self.vocab = vocab

    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) % self.vocab for c in text]}


class _FakeBackbone(nn.Module):
    def __init__(self, vocab=97, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)

    def forward(self, input_ids=None):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def _fake_expert(vocab=97, d=16, max_ctx=64):
    torch.manual_seed(0)
    backbone = _FakeBackbone(vocab, d)
    head = nn.Linear(d, vocab, bias=False)
    for p in list(backbone.parameters()) + list(head.parameters()):
        p.requires_grad = False
    return ProbedExpert(model_id="fake/expert-1", lm=None,
                        backbone=backbone, lm_head=head,
                        tokenizer=_FakeTokenizer(vocab), max_ctx=max_ctx)


_TEXTS = ["The quick brown fox jumps over the lazy dog. " * 8,
          "Pack my box with five dozen liquor jugs. " * 8]


class TestExpertBatch:
    def test_windows_have_requested_shape(self):
        ex = _fake_expert()
        ids, targets = expert_batch(ex, _TEXTS, batch=3, seq_len=16)
        assert ids.shape == (3, 16) and targets.shape == (3, 16)
        # next-token contract: targets are ids shifted by one
        flat = ids[0].tolist() + [targets[0, -1].item()]
        assert targets[0].tolist() == flat[1:]

    def test_short_corpus_tiles_instead_of_crashing(self):
        ex = _fake_expert()
        ids, targets = expert_batch(ex, ["tiny."], batch=2, seq_len=32)
        assert ids.shape == (2, 32)

    def test_seq_len_capped_at_expert_max_ctx(self):
        ex = _fake_expert(max_ctx=8)
        ids, _ = expert_batch(ex, _TEXTS, batch=2, seq_len=128)
        assert ids.shape[1] <= 8


class TestProbeExpert:
    def test_probe_returns_summary_and_persists(self, tmp_path):
        from neuroslm.genetic.modulation_store import ModulationStore
        from neuroslm.genetic.training_explorer import ExploreConfig
        ex = _fake_expert()
        ids, targets = expert_batch(ex, _TEXTS, batch=2, seq_len=16)
        out = probe_expert(ex, ids, targets,
                           store=ModulationStore(tmp_path / "mods"),
                           config=ExploreConfig(pop_size=8, generations=2),
                           round_idx=1)
        assert set(out) >= {"baseline_ce", "best_ce", "delta_ce", "improved",
                            "saved", "evaluated", "model_id", "headroom"}
        assert math.isfinite(out["baseline_ce"]) and out["baseline_ce"] > 0
        assert out["best_ce"] <= out["baseline_ce"] + 1e-4   # floored at identity
        if out["improved"]:
            assert out["saved"]
            recs = ModulationStore(tmp_path / "mods").list_all()
            assert recs and "expert" in recs[0].name

    def test_probe_never_touches_expert_weights(self):
        ex = _fake_expert()
        state = {k: v.clone() for k, v in ex.backbone.state_dict().items()}
        head_w = ex.lm_head.weight.clone()
        ids, targets = expert_batch(ex, _TEXTS, batch=2, seq_len=16)
        from neuroslm.genetic.training_explorer import ExploreConfig
        probe_expert(ex, ids, targets,
                     config=ExploreConfig(pop_size=6, generations=2),
                     round_idx=0)
        for k, v in ex.backbone.state_dict().items():
            assert torch.equal(v, state[k])
        assert torch.equal(ex.lm_head.weight, head_w)

    def test_baseline_matches_direct_ce(self):
        import torch.nn.functional as F
        ex = _fake_expert()
        ids, targets = expert_batch(ex, _TEXTS, batch=2, seq_len=16)
        from neuroslm.genetic.training_explorer import ExploreConfig
        out = probe_expert(ex, ids, targets,
                           config=ExploreConfig(pop_size=6, generations=1),
                           round_idx=0)
        with torch.no_grad():
            h = ex.backbone(input_ids=ids).last_hidden_state
            ref = float(F.cross_entropy(
                ex.lm_head(h).reshape(-1, 97), targets.reshape(-1)))
        assert abs(out["baseline_ce"] - ref) < 1e-4


class TestFp32Measurement:
    def test_bf16_expert_is_scored_in_fp32(self):
        # Experts load in bf16 (ensemble memory). A CE computed in bf16 is
        # quantized to ~1/32-nat steps — coarser than the Δs the probe hunts
        # (seen live: baseline_ce=4.1875, improve=+0.03125, all n/32). The
        # probe must run the head + CE in fp32 on the fp32-cast hidden.
        import torch.nn.functional as F
        from neuroslm.genetic.training_explorer import ExploreConfig
        ex = _fake_expert()
        ex.backbone.to(torch.bfloat16)
        ex.lm_head.to(torch.bfloat16)
        ids, targets = expert_batch(ex, _TEXTS, batch=2, seq_len=16)
        out = probe_expert(ex, ids, targets,
                           config=ExploreConfig(pop_size=6, generations=1),
                           round_idx=0)
        with torch.no_grad():
            h32 = ex.backbone(input_ids=ids).last_hidden_state.float()
            ref = float(F.cross_entropy(
                F.linear(h32, ex.lm_head.weight.float()).reshape(-1, 97),
                targets.reshape(-1)))
        assert abs(out["baseline_ce"] - ref) < 1e-5


class TestRunExpertDiscovery:
    def test_multi_round_over_fake_experts(self, tmp_path):
        results = run_expert_discovery(
            experts=[_fake_expert()], rounds=2, batch=2, seq_len=16,
            pop=6, gens=2, length=5,
            texts_fn=lambda n: (_TEXTS * n)[:n],
            store_root=tmp_path, push=False)
        assert len(results) == 2
        assert all(r["model_id"] == "fake/expert-1" for r in results)


class TestTextsFromStream:
    def test_offline_fallback_uses_bundled_corpus(self, monkeypatch):
        import neuroslm.genetic.expert_probe as ep

        def _boom(mode="text"):
            raise RuntimeError("offline")
        monkeypatch.setattr("neuroslm.data.open_stream", _boom)
        texts = texts_from_stream(4)
        assert len(texts) == 4
        assert all(isinstance(t, str) and t for t in texts)

    def test_provider_advances_across_rounds(self, monkeypatch):
        # Recurrence evidence requires FRESH text each round — a provider that
        # re-opens the stream and returns the same first-N texts every round
        # makes every "independent" probe measure the identical batch (seen
        # live: two rounds with bit-identical baseline_ce). The stateful
        # provider must keep drawing forward.
        from neuroslm.genetic.expert_probe import make_texts_provider

        def _boom(mode="text"):
            raise RuntimeError("offline")
        monkeypatch.setattr("neuroslm.data.open_stream", _boom)
        provider = make_texts_provider(min_chars=10)
        round1 = provider(2)
        round2 = provider(2)
        assert len(round1) == 2 and len(round2) == 2
        assert round1 != round2      # the stream advanced
