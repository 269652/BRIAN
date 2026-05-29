# -*- coding: utf-8 -*-
"""Stateful DSL blocks — the language extension for cognitive subsystems.

`state name: (shape) init=spec` registers a persistent buffer (not a
parameter). The forward body reads state vars as locals at the prelude
and writes them back to the buffer at the postlude — exactly the
state-machine semantics a NeurotransmitterSystem, vesicle pool, or
trophic system needs.

This is what makes the bowtie subsystems first-class DSL constructs.
"""
import pytest
import torch

from neuroslm.dsl.nn_lang import compile_layer


class TestStateDecls:
    def test_state_creates_buffer_with_init(self):
        src = '''
        layer Box() {
            state x: (1,) init=constant(0.5)
            forward(_) {
                return x
            }
        }
        '''
        m = compile_layer(src)()
        # state is a buffer, not a parameter
        assert "x" in dict(m.named_buffers())
        assert "x" not in dict(m.named_parameters())
        assert torch.allclose(m.x, torch.tensor([0.5]))

    def test_state_updates_persist_across_calls(self):
        src = '''
        layer Counter() {
            state c: (1,) init=constant(0.0)
            forward(step) {
                c = c + step
                return c
            }
        }
        '''
        m = compile_layer(src)()
        step = torch.tensor([1.0])
        m(step); m(step); m(step)
        assert torch.allclose(m.c, torch.tensor([3.0]))

    def test_state_only_no_params_no_grad(self):
        """State updates should not require/build a grad graph by default."""
        src = '''
        layer Decay() {
            state v: (1,) init=constant(1.0)
            forward(_) {
                v = v * 0.5
                return v
            }
        }
        '''
        m = compile_layer(src)()
        for _ in range(4):
            m(torch.tensor([0.0]))
        # 1.0 * 0.5^4 = 0.0625
        assert torch.allclose(m.v, torch.tensor([0.0625]), atol=1e-7)


# ── NT system as a DSL stateful block ──────────────────────────────────

NT_SYSTEM_DSL = '''
layer NTSystem() {
    state DA:    (1,) init=constant(0.10)
    state NE:    (1,) init=constant(0.15)
    state fiveHT:(1,) init=constant(0.30)
    state ACh:   (1,) init=constant(0.20)
    state eCB:   (1,) init=constant(0.05)
    state Glu:   (1,) init=constant(0.40)
    state GABA:  (1,) init=constant(0.10)

    forward(activity) {
        rel    = 0.02 * tanh(activity)
        DA     = DA     + rel - 0.4 * (DA     - 0.10)
        NE     = NE     + rel - 0.4 * (NE     - 0.15)
        fiveHT = fiveHT + rel - 0.4 * (fiveHT - 0.30)
        ACh    = ACh    + rel - 0.4 * (ACh    - 0.20)
        eCB    = eCB    + rel - 0.4 * (eCB    - 0.05)
        Glu    = Glu    + rel - 0.4 * (Glu    - 0.40)
        GABA   = GABA   + rel - 0.4 * (GABA   - 0.10)
        return DA
    }
}
'''


class TestNTSystemAsDSL:
    def test_matches_metrics_ntsystem_reference(self):
        from neuroslm.dsl.metrics import NTSystem as RefNT
        ref = RefNT()
        dsl = compile_layer(NT_SYSTEM_DSL)()

        torch.manual_seed(0)
        for _ in range(10):
            act = torch.rand(1).item()
            ref.step(activity=act)
            dsl(torch.tensor([act]))

        ref_levels = ref.levels()
        dsl_levels = {
            "DA": float(dsl.DA), "NE": float(dsl.NE),
            "5HT": float(dsl.fiveHT), "ACh": float(dsl.ACh),
            "eCB": float(dsl.eCB), "Glu": float(dsl.Glu),
            "GABA": float(dsl.GABA),
        }
        for k in ref_levels:
            diff = abs(ref_levels[k] - dsl_levels[k])
            assert diff < 1e-6, f"NT[{k}] diverged: ref={ref_levels[k]} dsl={dsl_levels[k]}"

    def test_seven_state_buffers_no_parameters(self):
        dsl = compile_layer(NT_SYSTEM_DSL)()
        buf_names = {n for n, _ in dsl.named_buffers()}
        assert buf_names == {"DA", "NE", "fiveHT", "ACh", "eCB", "Glu", "GABA"}
        assert list(dsl.parameters()) == []   # state, not params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
