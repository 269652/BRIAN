# -*- coding: utf-8 -*-
"""`forward_from_layer` — re-run the trunk tail from an intermediate hidden.

This is the enabler for discovery on *optimizable regions*: a modulation applied
at block k's output must be scored by the TRUE next-token loss, i.e. re-projected
through the real remaining blocks + PCT + final norm + LM head — not through a
proxy. Contract: with the unmodified stashed hidden it reproduces the original
logits exactly (eval mode), and it never mutates weights or stashes.
"""
import torch

from neuroslm.dsl.nn_lang import build_dsl_language_cortex


def _model(depth=4, pct_trunk=0.0, cosine_head=False):
    torch.manual_seed(0)
    m = build_dsl_language_cortex(
        vocab=97, d_model=32, depth=depth, n_heads=4, max_ctx=32,
        dropout=0.0, pct_trunk=pct_trunk, cosine_head=cosine_head,
    )
    m.eval()
    return m


def _ids(B=2, T=16, vocab=97, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


class TestExactReproduction:
    def test_every_layer_reproduces_forward_logits(self):
        m = _model(depth=4)
        ids = _ids()
        with torch.no_grad():
            ref = m(ids)
            outs = m._last_layer_outputs
            assert outs is not None and len(outs) == 4
            for k in range(4):
                got = m.forward_from_layer(k, outs[k])
                assert torch.allclose(got, ref, atol=1e-5), f"layer {k} tail diverged"

    def test_reproduces_with_pct_trunk_active(self):
        # PCT couples the tail to ALL block outputs; the tail must consume the
        # stashed lower-layer outputs so the correction matches the forward.
        m = _model(depth=4, pct_trunk=0.5)
        # give topdown weights real (non-zero-init) values so PCT actually acts
        with torch.no_grad():
            for w in m.topdown_w:
                w.add_(torch.randn_like(w) * 0.05)
        ids = _ids(seed=2)
        with torch.no_grad():
            ref = m(ids)
            outs = m._last_layer_outputs
            for k in range(4):
                got = m.forward_from_layer(k, outs[k])
                assert torch.allclose(got, ref, atol=1e-5), f"layer {k} PCT tail diverged"

    def test_reproduces_with_cosine_head(self):
        m = _model(depth=3, cosine_head=True)
        ids = _ids(seed=3)
        with torch.no_grad():
            ref = m(ids)
            outs = m._last_layer_outputs
            got = m.forward_from_layer(1, outs[1])
            assert torch.allclose(got, ref, atol=1e-5)


class TestModulationActuallyFlows:
    def test_modulated_hidden_changes_logits(self):
        m = _model(depth=4)
        ids = _ids(seed=4)
        with torch.no_grad():
            ref = m(ids)
            outs = m._last_layer_outputs
            got = m.forward_from_layer(1, outs[1] * 1.5)
            assert not torch.allclose(got, ref, atol=1e-3)


class TestReadOnly:
    def test_never_mutates_weights_or_stashes(self):
        m = _model(depth=3)
        ids = _ids(seed=5)
        with torch.no_grad():
            ref = m(ids)
            outs_before = [o.clone() for o in m._last_layer_outputs]
            hidden_before = m._last_hidden.clone()
            state_before = {k: v.clone() for k, v in m.state_dict().items()}
            m.forward_from_layer(1, outs_before[1] * 2.0)
        for k, v in m.state_dict().items():
            assert torch.equal(v, state_before[k]), f"weight {k} mutated"
        assert torch.equal(m._last_hidden, hidden_before), "_last_hidden overwritten"
        for a, b in zip(m._last_layer_outputs, outs_before):
            assert torch.equal(a, b), "_last_layer_outputs overwritten"
