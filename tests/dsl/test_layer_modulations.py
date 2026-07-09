# -*- coding: utf-8 -*-
"""Installed layer modulations — banked discovery winners take effect in training.

`DSLLanguageCortex._layer_modulations` maps block index → callable(h)->h applied
at exactly the probe site (block output, post adapter/gain — what
`_last_layer_outputs[k]` stashes). Contract: empty mapping is bit-identical to
baseline; an installed modulation changes the forward at its site;
`forward_from_layer` stays bit-exact (tail re-applies modulations for deeper
sites only); a modulation that throws is bypassed + auto-uninstalled rather
than crashing a training run.
"""
import torch

from neuroslm.dsl.nn_lang import build_dsl_language_cortex


def _model(depth=4):
    torch.manual_seed(0)
    m = build_dsl_language_cortex(vocab=61, d_model=24, depth=depth,
                                  n_heads=4, max_ctx=16, dropout=0.0)
    m.eval()
    return m


def _ids(seed=1, B=2, T=12, vocab=61):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g)


class TestNoOpByDefault:
    def test_empty_mapping_is_bit_identical(self):
        m = _model()
        ids = _ids()
        with torch.no_grad():
            ref = m(ids)
        assert m._layer_modulations == {}
        with torch.no_grad():
            again = m(ids)
        assert torch.equal(ref, again)


class TestInstalledModulationTakesEffect:
    def test_applied_at_the_site(self):
        m = _model()
        ids = _ids(seed=2)
        with torch.no_grad():
            ref = m(ids)
        m._layer_modulations[1] = lambda h: h * 1.5
        with torch.no_grad():
            got = m(ids)
        assert not torch.allclose(got, ref, atol=1e-3)
        # the stashed layer output reflects the modulation (probe-site parity)
        with torch.no_grad():
            plain_ref = m._last_layer_outputs[1]
        del m._layer_modulations[1]
        with torch.no_grad():
            m(ids)
        assert not torch.allclose(m._last_layer_outputs[1], plain_ref, atol=1e-4)

    def test_forward_from_layer_stays_exact_with_installed_modulations(self):
        m = _model()
        m._layer_modulations[2] = lambda h: h * torch.sigmoid(h)
        ids = _ids(seed=3)
        with torch.no_grad():
            ref = m(ids)
            outs = m._last_layer_outputs
            for k in range(4):
                got = m.forward_from_layer(k, outs[k])
                assert torch.allclose(got, ref, atol=1e-5), f"layer {k} diverged"

    def test_gradient_still_flows_through_installed_modulation(self):
        m = _model(depth=2)
        m.train()
        m._layer_modulations[0] = lambda h: h * 1.1
        ids = _ids(seed=4)
        logits = m(ids)
        loss = logits.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


class TestFailSafe:
    def test_throwing_modulation_is_bypassed_and_uninstalled(self):
        m = _model()
        ids = _ids(seed=5)
        with torch.no_grad():
            ref = m(ids)

        def _boom(h):
            raise RuntimeError("bad candidate")

        m._layer_modulations[1] = _boom
        with torch.no_grad():
            got = m(ids)                      # must not raise
        assert torch.allclose(got, ref, atol=1e-6)
        assert 1 not in m._layer_modulations  # self-healed
