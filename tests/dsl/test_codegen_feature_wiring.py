# -*- coding: utf-8 -*-
"""Codegen contracts for the feature-wiring pipeline (CLAUDE.md §14).

When a ``feature`` block is ``active: true`` AND a ``synapse`` references
it via ``feature: "<feature>.<endpoint>"``, the generated ``nn.Module``
MUST:

  1. Import the feature's ``impl:`` class.
  2. Instantiate it in ``__init__`` with the feature's ``params``.
  3. Call it on the synapse's pre-population activation in ``forward``
     (replacing the canonical ``weight * (x_pre @ W)`` term).
  4. Multiply the result by the synapse ``weight`` so existing
     edge-weighting semantics are preserved.

When the feature is ``active: false``, the synapse falls back to the
canonical edge — the feature is documented and binding is preserved
but bypassed at runtime. This is the §14-compliant ablation switch:
flip a single flag, get back the baseline circuit verbatim.

Unknown / mismatched references must error at compile time, NOT at
runtime — the whole point of the DSL is to catch wiring mistakes before
weights are spun up.

Anchors: §14 (no stubs / no scaffolds), §10 (mechanism studies).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.compiler import NeuroMLCompiler


# ──────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────


# Minimal arch that wires hyperbolic attention via a feature reference.
# d_sem == feature.params.d_model so the attention impl's shape contract
# (B, T, d_model) is satisfied without resize gymnastics.
_ARCH_TEMPLATE = """
architecture toy {{ d_sem: 32, dt: 0.01 }}

population a {{ count: 8, dynamics: "rate_code" }}
population b {{ count: 8, dynamics: "static" }}

equation hyp_attn_eq {{
    params: [d_model, n_heads, c],
    formula: "Attn(Q,K,V) = softmax(-d_hyp(expmap0(Q), expmap0(K)) / sqrt(d_head)) @ V"
}}

feature hyperbolic_attention {{
    equation: hyp_attn_eq,
    active:   {active},
    impl:     "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
    params: {{
        d_model: 32,
        n_heads: 4,
        c:       1.0
    }},
    endpoints: {{
        edge: {{
            kind:   "edge",
            inputs: [x_pre],
            output: y
        }}
    }}
}}

synapse a -> b {{
    feature: "hyperbolic_attention.edge",
    weight:  0.5
}}
"""


def _compile_source(src: str, module_name: str = "FeatTestCircuit"):
    """Helper — compile DSL source straight to an executable nn.Module class."""
    ir = NeuroMLCompiler.compile(src)
    return CodeGenerator(ir, module_name=module_name).compile_to_module()


def _generate_source(src: str, module_name: str = "FeatTestCircuit") -> str:
    """Helper — return the generated Python source as a string (for greps)."""
    ir = NeuroMLCompiler.compile(src)
    return CodeGenerator(ir, module_name=module_name).generate()


# ──────────────────────────────────────────────────────────────────────
# 1. Active feature → import + instantiation + call
# ──────────────────────────────────────────────────────────────────────


class TestActiveFeatureWiring:
    """When ``active: true`` the impl class must appear in the generated
    code at all three required sites (import, init, forward)."""

    def test_emits_import_for_active_feature_impl(self):
        src = _generate_source(_ARCH_TEMPLATE.format(active="true"))
        assert (
            "from neuroslm.modules.hyperbolic_attention "
            "import HyperbolicMultiHeadAttention"
        ) in src, (
            "Active feature with `impl:` must produce a real Python "
            "import — anything else is a §14 stub."
        )

    def test_instantiates_impl_in_init_with_feature_params(self):
        src = _generate_source(_ARCH_TEMPLATE.format(active="true"))
        # Constructor must be called with the feature's params dict.
        # We don't pin keyword order, just that the right kwargs land.
        assert "self.feature_hyperbolic_attention = HyperbolicMultiHeadAttention(" in src
        assert "d_model=32" in src
        assert "n_heads=4" in src
        # Curvature came in as float 1.0 → must remain a float literal.
        assert "c=1.0" in src

    def test_forward_calls_impl_for_wired_synapse(self):
        src = _generate_source(_ARCH_TEMPLATE.format(active="true"))
        # The synapse a->b must route through self.feature_<name>,
        # multiplied by its weight (0.5) — NOT through self.syn_0_w.
        assert "self.feature_hyperbolic_attention(" in src, (
            "Active wired synapse must actually call the feature impl."
        )
        # The synapse weight must still gate the contribution.
        assert "0.5" in src

    def test_active_wired_synapse_does_not_use_canonical_buffer(self):
        """When the feature is active, the canonical ``syn_i_w`` matmul
        is skipped — its random buffer is wasted otherwise and
        existence in forward would silently mix the two edge functions.
        """
        src = _generate_source(_ARCH_TEMPLATE.format(active="true"))
        # The buffer is still allocated (cheap, harmless, keeps state
        # dict stable across active flips) but the forward must not
        # multiply against self.syn_0_w when the feature takes over.
        forward_section = src.split("def forward(self, sensory_input")[1]
        assert "self.syn_0_w" not in forward_section, (
            "Active feature wiring must not also apply the canonical "
            "linear edge — that would double-count the transmission."
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Inactive feature → canonical fallback
# ──────────────────────────────────────────────────────────────────────


class TestInactiveFeatureFallback:
    """``active: false`` MUST disable the wiring at runtime — even when
    a synapse points at the feature. The synapse falls back to the
    canonical ``weight * (x_pre @ W)``.
    """

    def test_inactive_feature_emits_no_import(self):
        src = _generate_source(_ARCH_TEMPLATE.format(active="false"))
        assert "HyperbolicMultiHeadAttention" not in src, (
            "An inactive feature must not pollute the generated module "
            "with its impl class — that would defeat the ablation switch."
        )

    def test_inactive_feature_falls_back_to_canonical_edge(self):
        src = _generate_source(_ARCH_TEMPLATE.format(active="false"))
        # The legacy linear edge must be present:
        forward_section = src.split("def forward(self, sensory_input")[1]
        assert "self.syn_0_w" in forward_section, (
            "Inactive feature must yield the canonical edge function "
            "with the random-init weight buffer."
        )

    def test_inactive_feature_module_runs_like_baseline(self):
        """Concrete contract: the runtime output equals the baseline
        (no-feature) output for the same input, up to weight-init noise.
        We compare structural shapes + no-crash; numerical parity is
        covered by the canonical-edge tests in test_codegen_syn_mod.py.
        """
        Cls = _compile_source(_ARCH_TEMPLATE.format(active="false"),
                              module_name="InactiveCircuit")
        circuit = Cls(d_sem=32)
        x = torch.randn(2, 32)
        out = circuit(x)
        assert "b" in out
        assert out["b"].shape == (2, 32)


# ──────────────────────────────────────────────────────────────────────
# 3. End-to-end: compile, instantiate, run
# ──────────────────────────────────────────────────────────────────────


class TestEndToEndActiveFeatureRuns:
    """Compile the active-feature arch into a real nn.Module, push a
    tensor through it, and confirm the impl was actually invoked."""

    def test_compiled_module_forward_runs_without_error(self):
        Cls = _compile_source(_ARCH_TEMPLATE.format(active="true"),
                              module_name="ActiveCircuit")
        circuit = Cls(d_sem=32)
        x = torch.randn(2, 32)
        out = circuit(x)
        assert "b" in out
        assert out["b"].shape == (2, 32)

    def test_compiled_module_holds_real_impl_attribute(self):
        from neuroslm.modules.hyperbolic_attention import (
            HyperbolicMultiHeadAttention,
        )
        Cls = _compile_source(_ARCH_TEMPLATE.format(active="true"),
                              module_name="ActiveCircuit2")
        circuit = Cls(d_sem=32)
        attr = getattr(circuit, "feature_hyperbolic_attention", None)
        assert attr is not None, "impl not registered as module attribute"
        assert isinstance(attr, HyperbolicMultiHeadAttention), (
            f"expected HyperbolicMultiHeadAttention, got {type(attr).__name__}"
        )

    def test_active_and_inactive_produce_different_outputs(self):
        """The strongest §14 contract: flipping ``active`` must
        observably change the forward pass. Otherwise the wiring is a
        no-op scaffold, which is forbidden.
        """
        torch.manual_seed(0)
        Cls_on = _compile_source(_ARCH_TEMPLATE.format(active="true"),
                                 module_name="OnCircuit")
        Cls_off = _compile_source(_ARCH_TEMPLATE.format(active="false"),
                                  module_name="OffCircuit")
        torch.manual_seed(42)
        circ_on = Cls_on(d_sem=32)
        torch.manual_seed(42)
        circ_off = Cls_off(d_sem=32)

        x = torch.randn(4, 32)
        out_on = circ_on(x)["b"]
        out_off = circ_off(x)["b"]

        assert out_on.shape == out_off.shape
        # The two paths run completely different math (Möbius attention
        # vs random linear projection); their outputs must diverge by
        # an obvious margin, not float noise.
        diff = (out_on - out_off).abs().max().item()
        assert diff > 1e-3, (
            f"active vs inactive outputs differ by only {diff:.2e}; "
            "feature wiring is not actually being invoked at runtime."
        )

    def test_compiled_module_impl_parameters_are_trainable(self):
        """The impl's nn.Parameters must show up in
        ``circuit.parameters()``, otherwise the optimiser can't reach
        them — a silent §14 violation."""
        Cls = _compile_source(_ARCH_TEMPLATE.format(active="true"),
                              module_name="ParamCircuit")
        circuit = Cls(d_sem=32)
        impl = circuit.feature_hyperbolic_attention
        impl_param_ids = {id(p) for p in impl.parameters()}
        circuit_param_ids = {id(p) for p in circuit.parameters()}
        missing = impl_param_ids - circuit_param_ids
        assert not missing, (
            f"{len(missing)} impl parameters are not reachable from "
            "circuit.parameters() — they will not be trained."
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Reference resolution — short form, unknown refs, kind mismatch
# ──────────────────────────────────────────────────────────────────────


class TestFeatureReferenceResolution:
    def test_short_form_feature_ref_resolves_to_sole_endpoint(self):
        """``feature: "hyperbolic_attention"`` (no `.endpoint`) is
        valid iff the feature has exactly one endpoint."""
        src_short = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "static" }
        equation eq { params: [d_model, n_heads, c],
                      formula: "Attn(Q,K,V) = softmax(-d_hyp(expmap0(Q), expmap0(K)) / sqrt(d_head)) @ V" }
        feature hyperbolic_attention {
            equation: eq,
            active:   true,
            impl:     "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
            params: { d_model: 32, n_heads: 4, c: 1.0 },
            endpoints: {
                edge: { kind: "edge", inputs: [x_pre], output: y }
            }
        }
        synapse a -> b { feature: "hyperbolic_attention", weight: 1.0 }
        """
        # No exception → short-form resolved correctly.
        src = _generate_source(src_short, module_name="ShortFormCircuit")
        assert "self.feature_hyperbolic_attention(" in src

    def test_unknown_feature_ref_raises_at_codegen(self):
        """A synapse pointing at a nonexistent feature must fail loudly
        at codegen, before any tensor is allocated."""
        src_bad = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "static" }
        synapse a -> b { feature: "nonexistent.edge", weight: 1.0 }
        """
        ir = NeuroMLCompiler.compile(src_bad)
        with pytest.raises(Exception, match=r"(?i)unknown feature|no feature"):
            CodeGenerator(ir, module_name="BadCircuit").generate()

    def test_unknown_endpoint_raises_at_codegen(self):
        """Feature exists but the endpoint name is wrong → loud error."""
        src_bad = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "static" }
        equation eq { params: [d_model, n_heads, c],
                      formula: "Attn(Q,K,V) = softmax(-d_hyp(expmap0(Q), expmap0(K)) / sqrt(d_head)) @ V" }
        feature hyperbolic_attention {
            equation: eq, active: true,
            impl: "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
            params: { d_model: 32, n_heads: 4, c: 1.0 },
            endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
        }
        synapse a -> b { feature: "hyperbolic_attention.nope", weight: 1.0 }
        """
        ir = NeuroMLCompiler.compile(src_bad)
        with pytest.raises(Exception, match=r"(?i)endpoint"):
            CodeGenerator(ir, module_name="BadEpCircuit").generate()

    def test_active_feature_missing_impl_raises_when_wired(self):
        """Documentation-only feature (no ``impl:``) referenced by a
        synapse → §14 violation, must error."""
        src_bad = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "static" }
        equation eq { params: [d_model, n_heads, c],
                      formula: "Attn(Q,K,V) = softmax(-d_hyp(expmap0(Q), expmap0(K)) / sqrt(d_head)) @ V" }
        feature docs_only {
            equation: eq, active: true,
            endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
        }
        synapse a -> b { feature: "docs_only.edge", weight: 1.0 }
        """
        ir = NeuroMLCompiler.compile(src_bad)
        with pytest.raises(Exception, match=r"(?i)impl"):
            CodeGenerator(ir, module_name="NoImplCircuit").generate()
