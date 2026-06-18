# -*- coding: utf-8 -*-
"""TDD: op-driven HypergraphExecutor — IR attrs control operations.

The hypergraph IS the program; PyTorch IS the interpreter. Every edge and
node carries attrs that select which mathematical operation runs. Nothing
is hardcoded in the executor; all dispatch goes through op_registry.

Contracts pinned here:
  OD-1   standard_synapse: weight=0 disconnects edge (output = no-edge baseline).
  OD-2   standard_synapse: weight attr scales the edge contribution.
  OD-3   multiplicative_modulation: gain × NT level scales node output.
  OD-4   multiplicative_modulation: NT absent → no modulation applied.
  OD-5   additive_modulation: gain × NT level adds to node output.
  OD-6   additive_modulation: NT absent → output unchanged.
  OD-7   feature attr takes priority over equation attr for edge dispatch.
  OD-8   hyperbolic_attention feature: correct shape, grad flows.
  OD-9   rope_torus feature: correct shape, grad flows.
  OD-10  hrr_memory feature: correct shape, grad flows.
  OD-11  predictive_coding_residual feature: correct shape, grad flows.
  OD-12  surprise_gated_moe feature: correct shape, grad flows.
  OD-13  node act:silu attr switches activation.
  OD-14  node act:tanh attr switches activation.
  OD-15  unknown equation falls back to standard_synapse (no crash).
  OD-16  no-attrs IR (old-style) is backward-compat (existing tests hold).
  OD-17  modulation edges do NOT add to edge_projections.
  OD-18  grad flows through standard_synapse (weight is differentiable scale).
  OD-19  grad flows through multiplicative_modulation (non-parametric op).
  OD-20  resolve_edge_op returns correct op for each equation name.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from neuroslm.compiler.hypergraph_ir import (
    HypergraphIR, HyperNode, HyperEdge, SourceMap,
)
from neuroslm.compiler.hypergraph_executor import HypergraphExecutor
from neuroslm.compiler.op_registry import resolve_edge_op, resolve_node_op


# ── helpers ──────────────────────────────────────────────────────────────────

D = 32   # small d_model that divides cleanly by n_heads=4
B = 3    # batch size


def _pop(name: str, **attrs) -> HyperNode:
    return HyperNode(id=f"population:{name}", kind="population", name=name,
                     attrs=dict(attrs))


def _syn(src: str, dst: str, **attrs) -> HyperEdge:
    return HyperEdge(id=f"synapse:{src}->{dst}", kind="synapse",
                     members=[src, dst], attrs=dict(attrs))


def _mod(nt: str, dst: str, **attrs) -> HyperEdge:
    return HyperEdge(id=f"modulation:{nt}->{dst}", kind="modulation",
                     members=[nt, dst], attrs=dict(attrs))


def _ir(*nodes, edges=()) -> HypergraphIR:
    return HypergraphIR(nodes=list(nodes), hyperedges=list(edges),
                        source_map=SourceMap(""))


def _ex(*nodes, edges=(), d=D) -> HypergraphExecutor:
    return HypergraphExecutor(_ir(*nodes, edges=edges), d_model=d)


def _ex_seeded(*nodes, edges=(), d=D, seed=0) -> HypergraphExecutor:
    torch.manual_seed(seed)
    return HypergraphExecutor(_ir(*nodes, edges=edges), d_model=d)


# ── OD-1: weight=0 disconnects edge ─────────────────────────────────────────

class TestWeightZeroDisconnects:
    def test_weight_zero_equals_no_edge(self):
        nodes = [_pop("src"), _pop("dst")]
        edge_w0  = _syn("src", "dst", weight="0.0", equation="@standard_synapse")
        ex_with  = _ex_seeded(*nodes, edges=[edge_w0], seed=7)
        ex_none  = _ex_seeded(*nodes, seed=7)   # no edge at all

        x = torch.randn(B, D)
        # weight=0 nullifies edge contribution → same as if edge absent
        assert torch.allclose(ex_with(x)["dst"], ex_none(x)["dst"], atol=1e-6)


# ── OD-2: weight scales the edge contribution ────────────────────────────────

class TestWeightScaling:
    def test_different_weights_produce_different_dst_outputs(self):
        nodes = [_pop("src"), _pop("dst")]
        ex1 = _ex_seeded(*nodes,
                          edges=[_syn("src", "dst", weight="1.0",
                                      equation="@standard_synapse")], seed=0)
        ex2 = _ex_seeded(*nodes,
                          edges=[_syn("src", "dst", weight="0.5",
                                      equation="@standard_synapse")], seed=0)
        x = torch.randn(B, D)
        assert not torch.allclose(ex1(x)["dst"], ex2(x)["dst"])

    def test_weight_linearly_scales_edge_delta(self):
        # With src→dst weight w, the edge adds w*proj(src_out) to dst's incoming.
        # dst_w2 - dst_no_edge = 2*(dst_w1 - dst_no_edge) when all else equal.
        nodes = [_pop("src"), _pop("dst")]
        ex_w1  = _ex_seeded(*nodes, edges=[_syn("src", "dst", weight="1.0",
                                                equation="@standard_synapse")], seed=1)
        ex_w2  = _ex_seeded(*nodes, edges=[_syn("src", "dst", weight="2.0",
                                                equation="@standard_synapse")], seed=1)
        ex_none = _ex_seeded(*nodes, seed=1)

        x = torch.randn(B, D)
        out_w1   = ex_w1(x)["dst"]
        out_w2   = ex_w2(x)["dst"]
        out_none = ex_none(x)["dst"]

        delta1 = out_w1 - out_none
        delta2 = out_w2 - out_none
        # delta2 ≈ 2 * delta1  (linear in weight before ReLU)
        # This only holds where neither result is clipped by ReLU on the same side.
        # We check the norm ratio as a proxy (not pixel-exact due to relu saturation).
        ratio = delta2.norm() / (delta1.norm() + 1e-8)
        assert 1.5 < ratio < 2.5, f"expected ratio≈2, got {ratio:.3f}"


# ── OD-3: multiplicative modulation scales node output ───────────────────────

class TestMultiplicativeModulation:
    def _ir_with_mod(self, gain="0.5") -> HypergraphIR:
        return _ir(
            _pop("pfc"),
            edges=[_mod("dopamine", "pfc", gain=gain, effect="multiplicative",
                        equation="@multiplicative_modulation")],
        )

    def test_doubling_nt_doubles_scale(self):
        ex = HypergraphExecutor(self._ir_with_mod(gain="1.0"), d_model=D)
        x = torch.randn(B, D)
        # nt=1.0 → scale factor = 1.0*1.0 = 1.0   (node_out * 1.0)
        # nt=2.0 → scale factor = 1.0*2.0 = 2.0   (node_out * 2.0)
        out1 = ex(x, nt_levels={"dopamine": 1.0})["pfc"]
        out2 = ex(x, nt_levels={"dopamine": 2.0})["pfc"]
        assert torch.allclose(out2, 2.0 * out1, atol=1e-5)

    def test_gain_scales_effect(self):
        ex_g1 = HypergraphExecutor(self._ir_with_mod(gain="1.0"), d_model=D)
        ex_g2 = HypergraphExecutor(self._ir_with_mod(gain="2.0"), d_model=D)
        x = torch.randn(B, D)
        nt = {"dopamine": 1.0}
        out1 = ex_g1(x, nt_levels=nt)["pfc"]
        out2 = ex_g2(x, nt_levels=nt)["pfc"]
        # gain=2 → double the scale → out2 = 2 * out1
        # But ex_g1 and ex_g2 have different random init for node_layers,
        # so we can only compare with same seed.
        torch.manual_seed(5)
        ex_a = HypergraphExecutor(self._ir_with_mod(gain="1.0"), d_model=D)
        torch.manual_seed(5)
        ex_b = HypergraphExecutor(self._ir_with_mod(gain="2.0"), d_model=D)
        oa = ex_a(x, nt_levels=nt)["pfc"]
        ob = ex_b(x, nt_levels=nt)["pfc"]
        assert torch.allclose(ob, 2.0 * oa, atol=1e-5)


# ── OD-4: multiplicative modulation absent when NT not in levels ─────────────

class TestMultiplicativeModulationAbsent:
    def test_no_nt_means_no_modulation(self):
        ir = _ir(
            _pop("pfc"),
            edges=[_mod("dopamine", "pfc", gain="0.5", effect="multiplicative",
                        equation="@multiplicative_modulation")],
        )
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        # without dopamine in nt_levels, modulation is skipped → identity on node_out
        out_no_nt = ex(x, nt_levels={})["pfc"]
        out_no_nt2 = ex(x)["pfc"]               # nt_levels=None also skips
        # with dopamine present, output differs (scaled by 0.5*1.0 = 0.5)
        out_with_nt = ex(x, nt_levels={"dopamine": 1.0})["pfc"]
        assert torch.allclose(out_no_nt, out_no_nt2, atol=1e-6)
        assert not torch.allclose(out_no_nt, out_with_nt)


# ── OD-5: additive modulation adds gain×NT to node output ────────────────────

class TestAdditiveModulation:
    def _ir_add(self, gain="1.0") -> HypergraphIR:
        return _ir(
            _pop("hippo"),
            edges=[_mod("acetylcholine", "hippo", gain=gain, effect="additive",
                        equation="@additive_modulation")],
        )

    def test_additive_modulation_offsets_output(self):
        torch.manual_seed(9)
        ex = HypergraphExecutor(self._ir_add(gain="1.0"), d_model=D)
        x = torch.randn(B, D)
        out_no_nt  = ex(x, nt_levels={})["hippo"]
        out_nt_1   = ex(x, nt_levels={"acetylcholine": 1.0})["hippo"]
        out_nt_2   = ex(x, nt_levels={"acetylcholine": 2.0})["hippo"]
        # gain=1, nt=1 → add 1.0;  nt=2 → add 2.0
        # difference between nt=2 and nt=1 == 1.0 (scalar broadcast to all dims)
        diff_2_1 = out_nt_2 - out_nt_1
        diff_1_0 = out_nt_1 - out_no_nt
        assert torch.allclose(diff_2_1, diff_1_0, atol=1e-5)

    def test_additive_absent_when_nt_missing(self):
        torch.manual_seed(9)
        ex = HypergraphExecutor(self._ir_add(), d_model=D)
        x = torch.randn(B, D)
        out_no  = ex(x, nt_levels={})["hippo"]
        out_yes = ex(x, nt_levels={"acetylcholine": 1.0})["hippo"]
        assert not torch.allclose(out_no, out_yes)


# ── OD-7: feature attr takes priority over equation attr ─────────────────────

class TestFeaturePriority:
    def test_feature_overrides_equation_in_op_lookup(self):
        # edge with both feature: and equation: — feature must win
        attrs_feature_wins = {
            "weight": "1.0",
            "equation": "@standard_synapse",
            "feature": "hyperbolic_attention.edge",
        }
        attrs_eq_only = {
            "weight": "1.0",
            "equation": "@standard_synapse",
        }
        op_feature = resolve_edge_op(attrs_feature_wins)
        op_eq      = resolve_edge_op(attrs_eq_only)
        # They must be different objects (different dispatch)
        assert type(op_feature) is not type(op_eq)


# ── OD-8 through OD-12: feature ops shape + grad contracts ───────────────────

def _feature_ir(feature_name: str) -> HypergraphIR:
    return _ir(
        _pop("src"),
        _pop("dst"),
        edges=[_syn("src", "dst", weight="1.0",
                    feature=f"{feature_name}.edge",
                    equation="@standard_synapse")],
    )


def _check_feature_shape_and_grad(feature_name: str):
    ir = _feature_ir(feature_name)
    ex = HypergraphExecutor(ir, d_model=D)
    x = torch.randn(B, D)
    out = ex(x)
    assert out["dst"].shape == (B, D), \
        f"{feature_name}: expected ({B},{D}), got {out['dst'].shape}"
    loss = out["dst"].sum()
    loss.backward()
    # At least one parameter in the executor must have received a gradient
    has_grad = any(p.grad is not None for p in ex.parameters())
    assert has_grad, f"{feature_name}: no parameter received a gradient"


class TestFeatureOps:
    def test_hyperbolic_attention_shape_and_grad(self):
        _check_feature_shape_and_grad("hyperbolic_attention")

    def test_rope_torus_shape_and_grad(self):
        _check_feature_shape_and_grad("rope_torus")

    def test_hrr_memory_shape_and_grad(self):
        _check_feature_shape_and_grad("hrr_memory")

    def test_predictive_coding_residual_shape_and_grad(self):
        _check_feature_shape_and_grad("predictive_coding_residual")

    def test_surprise_gated_moe_shape_and_grad(self):
        _check_feature_shape_and_grad("surprise_gated_moe")

    def test_feature_module_is_not_plain_linear(self):
        # The feature op should instantiate a domain-specific module,
        # not a plain nn.Linear (which is what standard_synapse uses).
        ir = _feature_ir("hyperbolic_attention")
        ex = HypergraphExecutor(ir, d_model=D)
        syn_key = ex._safe_key("synapse:src->dst")
        module = ex.edge_projections[syn_key]
        assert not type(module) is nn.Linear, \
            "hyperbolic_attention should use HyperbolicMultiHeadAttention, not nn.Linear"


# ── OD-13/OD-14: node activation dispatch ────────────────────────────────────

class TestNodeActivation:
    def test_silu_activation_differs_from_relu(self):
        torch.manual_seed(3)
        ex_relu = HypergraphExecutor(
            _ir(_pop("a", act="relu")), d_model=D)
        torch.manual_seed(3)
        ex_silu = HypergraphExecutor(
            _ir(_pop("a", act="silu")), d_model=D)
        x = torch.randn(B, D) - 0.5  # include negatives so relu/silu differ
        out_relu = ex_relu(x)["a"]
        out_silu = ex_silu(x)["a"]
        assert not torch.allclose(out_relu, out_silu), \
            "act:silu and act:relu produced identical output — dispatch broken"

    def test_tanh_activation_bounded(self):
        ex = HypergraphExecutor(_ir(_pop("a", act="tanh")), d_model=D)
        x = torch.randn(B, D) * 10  # large inputs to stress the bound
        out = ex(x)["a"]
        assert out.abs().max() <= 1.0 + 1e-5, \
            "tanh output should be in [-1, 1]"

    def test_default_activation_is_relu_nonnegative(self):
        ex = HypergraphExecutor(_ir(_pop("a")), d_model=D)
        x = torch.randn(B, D)
        out = ex(x)["a"]
        assert (out >= 0).all(), "default activation (relu) must be non-negative"


# ── OD-15: unknown equation falls back gracefully ────────────────────────────

class TestUnknownEquationFallback:
    def test_unknown_equation_does_not_crash(self):
        ir = _ir(
            _pop("src"), _pop("dst"),
            edges=[_syn("src", "dst", weight="1.0",
                        equation="@some_unknown_equation_xyz")],
        )
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        out = ex(x)  # must not raise
        assert out["dst"].shape == (B, D)

    def test_unknown_equation_fallback_still_differentiable(self):
        ir = _ir(
            _pop("src"), _pop("dst"),
            edges=[_syn("src", "dst", equation="@nonexistent")],
        )
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D)
        ex(x)["dst"].sum().backward()
        assert any(p.grad is not None for p in ex.parameters())


# ── OD-16: backward compat — no-attrs IR uses standard_synapse ───────────────

class TestBackwardCompat:
    def test_no_attrs_ir_produces_correct_shape(self):
        ir = _ir(_pop("a"), _pop("b"),
                 edges=[HyperEdge(id="synapse:a->b", kind="synapse",
                                  members=["a", "b"])])
        ex = HypergraphExecutor(ir, d_model=D)
        out = ex(torch.randn(B, D))
        assert out["b"].shape == (B, D)

    def test_no_attrs_edge_creates_linear_projection(self):
        ir = _ir(_pop("a"), _pop("b"),
                 edges=[HyperEdge(id="synapse:a->b", kind="synapse",
                                  members=["a", "b"])])
        ex = HypergraphExecutor(ir, d_model=D)
        key = ex._safe_key("synapse:a->b")
        assert isinstance(ex.edge_projections[key], nn.Linear)


# ── OD-17: modulation edges don't add to edge_projections ────────────────────

class TestModulationNoProjection:
    def test_modulation_edge_not_in_edge_projections(self):
        ir = _ir(
            _pop("a"),
            edges=[_mod("dopamine", "a", gain="0.5", effect="multiplicative",
                        equation="@multiplicative_modulation")],
        )
        ex = HypergraphExecutor(ir, d_model=D)
        assert len(ex.edge_projections) == 0


# ── OD-18/OD-19: gradient flow through dispatch paths ────────────────────────

class TestGradientFlowDispatch:
    def test_grad_through_standard_synapse_weight(self):
        # weight is a Python float applied as a scalar multiply — not a Parameter.
        # The gradient must still flow through the nn.Linear projection itself.
        ir = _ir(_pop("src"), _pop("dst"),
                 edges=[_syn("src", "dst", weight="0.7",
                             equation="@standard_synapse")])
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        ex(x)["dst"].sum().backward()
        assert x.grad is not None
        proj = ex.edge_projections[ex._safe_key("synapse:src->dst")]
        assert proj.weight.grad is not None

    def test_grad_through_multiplicative_modulation(self):
        # Modulation is non-parametric, but the node output before modulation
        # must still carry gradients (they flow through the node_layer).
        ir = _ir(
            _pop("pfc"),
            edges=[_mod("dopamine", "pfc", gain="0.5", effect="multiplicative",
                        equation="@multiplicative_modulation")],
        )
        ex = HypergraphExecutor(ir, d_model=D)
        x = torch.randn(B, D, requires_grad=True)
        ex(x, nt_levels={"dopamine": 1.0})["pfc"].sum().backward()
        assert x.grad is not None
        pfc_layer = ex.node_layers[ex._safe_key("pfc")]
        assert pfc_layer.weight.grad is not None


# ── OD-20: resolve_edge_op returns correct op for each equation ───────────────

class TestResolveEdgeOp:
    def test_standard_synapse_resolves(self):
        from neuroslm.compiler.op_registry import _EDGE_REGISTRY
        op = resolve_edge_op({"equation": "@standard_synapse"})
        assert type(op).__name__ == "StandardSynapseOp"

    def test_multiplicative_modulation_resolves(self):
        op = resolve_edge_op({"equation": "@multiplicative_modulation"},
                             kind="modulation")
        assert type(op).__name__ == "MultiplicativeModulationOp"

    def test_additive_modulation_resolves(self):
        op = resolve_edge_op({"equation": "@additive_modulation"},
                             kind="modulation")
        assert type(op).__name__ == "AdditiveModulationOp"

    def test_hyperbolic_attention_resolves_via_feature(self):
        op = resolve_edge_op({"feature": "hyperbolic_attention.edge"})
        assert type(op).__name__ == "HyperbolicAttentionOp"

    def test_rope_torus_resolves_via_feature(self):
        op = resolve_edge_op({"feature": "rope_torus.edge"})
        assert type(op).__name__ == "RopeTorusOp"

    def test_hrr_memory_resolves_via_feature(self):
        op = resolve_edge_op({"feature": "hrr_memory.edge"})
        assert type(op).__name__ == "HRRMemoryOp"

    def test_predictive_coding_residual_resolves_via_feature(self):
        op = resolve_edge_op({"feature": "predictive_coding_residual.edge"})
        assert type(op).__name__ == "PredictiveCodingResidualOp"

    def test_surprise_gated_moe_resolves_via_feature(self):
        op = resolve_edge_op({"feature": "surprise_gated_moe.edge"})
        assert type(op).__name__ == "SurpriseGatedMoEOp"

    def test_empty_attrs_falls_back_to_standard_synapse(self):
        op = resolve_edge_op({})
        assert type(op).__name__ == "StandardSynapseOp"
