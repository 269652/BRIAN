# -*- coding: utf-8 -*-
"""Phase B — `param_scope` blocks: declarative gradient isolation (p3 fix).

The p3 architectural fix isolated bio-side parameters so they couldn't be
mutated by the main LM loss (they're trained by their own auxiliary
objectives). In the DSL this becomes a declaration in arch.neuro:

    param_scope trunk {
        populations: [sensory, thalamus, gws, pfc, motor]
    }
    param_scope bio {
        populations: [amygdala, hippo, vta],
        gradient: "detached_from_main_loss"
    }

The harness reads these and sets requires_grad=False on the parameters of
populations in a `detached_from_main_loss` scope — exactly what the p3
fix did via Brain.partition_trunk_bio_params().
"""
import pytest
import torch
from pathlib import Path

from neuroslm.dsl.param_scopes import (
    ParamScope,
    parse_param_scopes,
    load_param_scopes_from_arch,
)
from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.harness import BRIANHarness


ARCH_ROOT = Path(__file__).resolve().parent.parent.parent / "architectures" / "master"


# ── Parser ─────────────────────────────────────────────────────────────

class TestParseParamScopes:
    def test_single_scope(self):
        src = '''
            param_scope trunk {
                populations: [sensory, thalamus, gws]
            }
        '''
        scopes = parse_param_scopes(src)
        assert len(scopes) == 1
        assert scopes[0].name == "trunk"
        assert scopes[0].populations == ["sensory", "thalamus", "gws"]
        assert scopes[0].gradient == "normal"  # default

    def test_detached_scope(self):
        src = '''
            param_scope bio {
                populations: [amygdala, hippo],
                gradient: "detached_from_main_loss"
            }
        '''
        scopes = parse_param_scopes(src)
        assert scopes[0].gradient == "detached_from_main_loss"
        assert scopes[0].populations == ["amygdala", "hippo"]

    def test_multiple_scopes(self):
        src = '''
            param_scope trunk { populations: [sensory, motor] }
            param_scope bio {
                populations: [amygdala],
                gradient: "detached_from_main_loss"
            }
        '''
        scopes = parse_param_scopes(src)
        assert len(scopes) == 2
        names = {s.name for s in scopes}
        assert names == {"trunk", "bio"}

    def test_invalid_gradient_policy_rejected(self):
        src = '''
            param_scope x { populations: [a], gradient: "bogus_policy" }
        '''
        with pytest.raises(ValueError, match="gradient"):
            parse_param_scopes(src)

    def test_no_scopes_returns_empty(self):
        assert parse_param_scopes("architecture x { d_sem: 256 }") == []


# ── Load from arch.neuro ──────────────────────────────────────────────

class TestLoadFromArch:
    def test_loads_from_folder(self, tmp_path):
        (tmp_path / "arch.neuro").write_text('''
            architecture x { d_sem: 256 }
            param_scope trunk { populations: [a, b] }
            param_scope bio { populations: [c], gradient: "detached_from_main_loss" }
        ''', encoding="utf-8")
        scopes = load_param_scopes_from_arch(tmp_path)
        assert len(scopes) == 2

    def test_missing_arch_returns_empty(self, tmp_path):
        assert load_param_scopes_from_arch(tmp_path) == []


# ── Harness applies the isolation ─────────────────────────────────────

class TestHarnessAppliesScopes:
    def _harness(self):
        ir = compile_folder(ARCH_ROOT)
        Cls = CodeGenerator(ir, module_name="ScopeTestCircuit").compile_to_module()
        return BRIANHarness(circuit=Cls(d_sem=64), vocab_size=256, d_sem=64)

    def test_detached_scope_freezes_params(self):
        h = self._harness()
        # thalamus and claustrum are `gated` → they have a `gate` Parameter.
        # Put thalamus in a detached scope; its gate must end up frozen.
        scopes = [
            ParamScope(name="bio", populations=["thalamus"],
                       gradient="detached_from_main_loss"),
        ]
        h.apply_param_scopes(scopes)

        thal_params = list(h.circuit.thalamus.parameters())
        assert len(thal_params) > 0   # gated population has a gate param
        assert all(not p.requires_grad for p in thal_params), \
            "thalamus params should be frozen (detached scope)"

    def test_normal_scope_leaves_params_trainable(self):
        h = self._harness()
        scopes = [
            ParamScope(name="trunk", populations=["thalamus"],
                       gradient="normal"),
        ]
        h.apply_param_scopes(scopes)
        thal_params = list(h.circuit.thalamus.parameters())
        assert all(p.requires_grad for p in thal_params)

    def test_detached_params_get_no_gradient(self):
        h = self._harness()
        scopes = [
            ParamScope(name="bio", populations=["claustrum"],
                       gradient="detached_from_main_loss"),
        ]
        h.apply_param_scopes(scopes)

        ids = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        loss = h.compute_loss(ids, targets)
        loss.backward()

        # Claustrum's gate param must have no gradient (frozen)
        for p in h.circuit.claustrum.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
