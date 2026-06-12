# -*- coding: utf-8 -*-
"""Contracts for the wirable-feature DSL extensions.

Pins the surface that lets a `feature` block declare:
  * `impl: "<dotted.python.path>"` — the class implementing the equation,
  * `endpoints: { <name>: { kind, inputs, output, params } }` — the
    wiring surfaces the feature exposes,
and lets a `synapse` declare `feature: <feature_name>.<endpoint>` to
route its edge function through the feature implementation.

Per CLAUDE.md §14 every test pins behaviour, not shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroslm.dsl.compiler import NeuroMLCompiler, FeatureIR


# ──────────────────────────────────────────────────────────────────────
# 1. Feature `impl` field — dotted Python path to implementation class
# ──────────────────────────────────────────────────────────────────────


class TestFeatureImplField:
    def test_impl_field_is_parsed(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation foo_eq {
            params: [x],
            formula: "y = x"
        }
        feature foo {
            equation: foo_eq,
            active: false,
            impl: "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention"
        }
        """
        ir = NeuroMLCompiler().compile(source)
        assert len(ir.features) == 1
        feat = ir.features[0]
        assert (
            feat.impl
            == "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention"
        )

    def test_impl_omitted_defaults_to_empty_string(self):
        """A feature without an `impl` is a pure-math declaration —
        still valid (used for documentation-only features), but the
        codegen will refuse to wire it into a synapse."""
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation foo_eq { params: [x], formula: "y = x" }
        feature foo { equation: foo_eq, active: false }
        """
        ir = NeuroMLCompiler().compile(source)
        assert ir.features[0].impl == ""


# ──────────────────────────────────────────────────────────────────────
# 2. Feature `endpoints` block — wiring surfaces
# ──────────────────────────────────────────────────────────────────────


class TestFeatureEndpoints:
    def test_single_endpoint_parses(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation att_eq { params: [Q, K, V], formula: "softmax(QK)V" }
        feature attn {
            equation: att_eq,
            active: false,
            impl: "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
            endpoints: {
                edge: {
                    kind: "edge",
                    inputs: [x_pre],
                    output: y
                }
            }
        }
        """
        ir = NeuroMLCompiler().compile(source)
        feat = ir.features[0]
        assert len(feat.endpoints) == 1
        ep = feat.endpoints[0]
        assert ep.name == "edge"
        assert ep.kind == "edge"
        assert ep.inputs == ["x_pre"]
        assert ep.output == "y"

    def test_multiple_endpoints_parse(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation eq { params: [x], formula: "y = x" }
        feature multi {
            equation: eq,
            active: false,
            impl: "x.y.Z",
            endpoints: {
                forward: { kind: "edge", inputs: [x_pre], output: y },
                modulator: { kind: "modulator", inputs: [c, gain], output: y }
            }
        }
        """
        ir = NeuroMLCompiler().compile(source)
        eps = ir.features[0].endpoints
        assert {e.name for e in eps} == {"forward", "modulator"}
        kinds = {e.name: e.kind for e in eps}
        assert kinds["forward"] == "edge"
        assert kinds["modulator"] == "modulator"

    def test_endpoints_omitted_yields_empty_list(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation eq { params: [x], formula: "y = x" }
        feature no_ep { equation: eq, active: false }
        """
        ir = NeuroMLCompiler().compile(source)
        assert ir.features[0].endpoints == []

    def test_endpoint_with_params_carries_them(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population p { count: 8, dynamics: "rate_code" }
        equation eq { params: [x], formula: "y = x" }
        feature f {
            equation: eq,
            active: false,
            impl: "a.b.C",
            endpoints: {
                e1: {
                    kind: "edge",
                    inputs: [x_pre],
                    output: y,
                    params: { scale: 1.5, mode: "soft" }
                }
            }
        }
        """
        ir = NeuroMLCompiler().compile(source)
        ep = ir.features[0].endpoints[0]
        assert float(ep.params["scale"]) == pytest.approx(1.5)
        assert ep.params["mode"] == "soft"


# ──────────────────────────────────────────────────────────────────────
# 3. Synapse `feature:` field — wiring a feature endpoint into an edge
# ──────────────────────────────────────────────────────────────────────


class TestSynapseFeatureReference:
    def test_synapse_with_feature_ref_parses(self):
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "rate_code" }
        equation eq { params: [Q, K, V], formula: "softmax(QK)V" }
        feature attn {
            equation: eq,
            active: false,
            impl: "neuroslm.modules.hyperbolic_attention.HyperbolicMultiHeadAttention",
            endpoints: { edge: { kind: "edge", inputs: [x_pre], output: y } }
        }
        synapse a -> b {
            feature: "attn.edge",
            weight: 1.0
        }
        """
        ir = NeuroMLCompiler().compile(source)
        assert len(ir.synapses) == 1
        syn = ir.synapses[0]
        assert syn.feature_ref == "attn.edge"

    def test_synapse_without_feature_ref_has_none(self):
        """Backwards compatibility: existing synapses without
        `feature:` must continue to work unchanged."""
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "rate_code" }
        synapse a -> b { weight: 0.5 }
        """
        ir = NeuroMLCompiler().compile(source)
        syn = ir.synapses[0]
        assert syn.feature_ref is None

    def test_synapse_short_form_feature_ref(self):
        """If the feature has exactly one endpoint, the synapse may
        omit the endpoint name: ``feature: "attn"`` resolves to that
        endpoint at validation time."""
        source = """
        architecture toy { d_sem: 32, dt: 0.01 }
        population a { count: 8, dynamics: "rate_code" }
        population b { count: 8, dynamics: "rate_code" }
        equation eq { params: [x], formula: "y = x" }
        feature solo {
            equation: eq,
            active: false,
            impl: "a.b.C",
            endpoints: { only: { kind: "edge", inputs: [x_pre], output: y } }
        }
        synapse a -> b {
            feature: "solo",
            weight: 1.0
        }
        """
        ir = NeuroMLCompiler().compile(source)
        # Short form is preserved as-is in the IR; resolution to
        # "solo.only" happens at validation/codegen time.
        assert ir.synapses[0].feature_ref == "solo"
