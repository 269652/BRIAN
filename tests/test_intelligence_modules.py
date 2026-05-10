"""Tests for the `intelligence/` package: adaptive compute, MoE, CPC,
memory attention, oscillations, reflection, metrics, active inference.
"""
from __future__ import annotations
import math
import torch
import pytest


# ── Flow / PonderController ─────────────────────────────────────────────
def test_ponder_controller_shape():
    from neuroslm.intelligence.flow import PonderController
    p = PonderController(d_model=32, max_steps=4)
    x = torch.randn(2, 8, 32)
    out = p(x)
    assert out.shape == (2, 8)
    assert (out >= 0).all() and (out <= 1).all()


def test_ponder_regularization_loss():
    from neuroslm.intelligence.flow import PonderController
    # Build a fake list of halt prob tensors
    probs = [torch.full((2, 4), 0.5) for _ in range(3)]
    loss = PonderController.regularization_loss(probs, target_mean_steps=2.0)
    assert torch.is_tensor(loss)
    assert math.isfinite(float(loss.item()))


def test_adaptive_compute_block_runs():
    from neuroslm.intelligence.flow import AdaptiveComputeBlock
    blk = AdaptiveComputeBlock(d_model=32, n_heads=4, max_steps=3)
    x = torch.randn(2, 8, 32)
    out = blk(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == x.shape


# ── Sparse MoE ──────────────────────────────────────────────────────────
def test_moe_forward_shape_and_aux():
    from neuroslm.intelligence.mixture import SparseMoE
    m = SparseMoE(d_model=32, n_experts=4, d_ff=64, top_k=2)
    x = torch.randn(2, 8, 32)
    y, aux = m(x)
    assert y.shape == x.shape
    assert torch.is_tensor(aux)
    assert math.isfinite(float(aux.item()))
    util = m.expert_utilization()
    assert len(util) == 4
    assert all(0.0 <= v <= 1.0 for v in util.values())


# ── Memory cross attention ──────────────────────────────────────────────
def test_memory_cross_attention_shape():
    from neuroslm.intelligence.memory_attention import MemoryCrossAttention
    m = MemoryCrossAttention(d_model=32, d_mem=32, n_heads=4, max_retrieved=4)
    q = torch.randn(2, 6, 32)
    mem = torch.randn(8, 32)
    out = m(q, mem)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == q.shape


def test_memory_cross_attention_no_mem_passthrough():
    from neuroslm.intelligence.memory_attention import MemoryCrossAttention
    m = MemoryCrossAttention(d_model=32, d_mem=32, n_heads=4)
    q = torch.randn(2, 6, 32)
    out = m(q, None)
    assert torch.equal(out, q)


# ── CPC ─────────────────────────────────────────────────────────────────
def test_cpc_forward_loss_positive():
    from neuroslm.intelligence.contrastive_predictive_coding import ContrastivePredictiveCoding
    cpc = ContrastivePredictiveCoding(d_model=32, max_steps=3, n_negatives=8)
    x = torch.randn(2, 10, 32)
    loss, info = cpc(x)
    assert torch.is_tensor(loss)
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) >= 0.0


# ── Oscillations ────────────────────────────────────────────────────────
def test_oscillations_record_and_spectrum():
    from neuroslm.intelligence.oscillations import NeuralOscillationTracker
    o = NeuralOscillationTracker(d_model=16, n_regions=4, window_size=16)
    for _ in range(8):
        for mod_idx in range(4):
            o.record(mod_idx, torch.randn(2, 16))
        o.tick()
    spec = o.compute_spectrum()
    d = spec.as_dict()
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        assert band in d
        assert math.isfinite(d[band])
    assert d["dominant_band"] in ("delta", "theta", "alpha", "beta", "gamma")


# ── Reflection ──────────────────────────────────────────────────────────
def test_spontaneous_reflection_identity_score():
    from neuroslm.intelligence.reflection import SpontaneousReflection
    r = SpontaneousReflection(d_sem=16)

    class _Stream:
        def __init__(self):
            self.summary = torch.randn(16)

    class _Sys:
        def __init__(self):
            self.autobiographical = _Stream()
            self.world = _Stream()
            self.entities: dict = {}

    cand = torch.randn(16)
    sys_ = _Sys()
    score = r.identity_score(cand, sys_)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0   # identity_head uses sigmoid


def test_spontaneous_reflection_reflect():
    from neuroslm.intelligence.reflection import SpontaneousReflection
    r = SpontaneousReflection(d_sem=16)

    class _Stream:
        def __init__(self):
            self.summary = torch.randn(16)

    class _Sys:
        def __init__(self):
            self.autobiographical = _Stream()
            self.world = _Stream()
            self.entities: dict = {}

    seed = r.reflect(_Sys())
    assert seed.shape == (16,)
    assert math.isfinite(float(seed.sum().item()))


# ── Metrics ─────────────────────────────────────────────────────────────
def test_identity_drift_tracker_observes():
    from neuroslm.intelligence.metrics import IdentityDriftTracker
    t = IdentityDriftTracker()
    t.observe(torch.randn(16))
    drift = t.observe(torch.randn(16))
    snap = t.snapshot()
    assert "identity_drift" in snap
    assert "identity_drift_ema" in snap
    assert math.isfinite(snap["identity_drift_ema"])


def test_intelligence_metrics_snapshot_has_all_keys():
    from neuroslm.intelligence.metrics import IntelligenceMetrics
    m = IntelligenceMetrics()
    m.observe_lm(2.0, hardness=0.1)
    m.observe_lm(3.0, hardness=0.9)
    m.observe_ponder(2.5)
    m.observe_self_reference("i am thinking about it")
    m.observe_theory_of_mind(0.5, 0.4)
    snap = m.snapshot()
    for k in ("phi_proxy", "narrative_coherence", "causal_density",
              "semantic_compression", "self_reference_rate",
              "theory_of_mind_acc", "ponder_steps_ema",
              "reasoning_gain", "ponder_efficiency"):
        assert k in snap, f"missing {k}"


def test_intelligence_metrics_observe_phi():
    from neuroslm.intelligence.metrics import IntelligenceMetrics
    m = IntelligenceMetrics()
    mods = {f"m{i}": torch.randn(2, 4, 32) for i in range(4)}
    phi = m.observe_phi(mods)
    assert 0.0 <= phi <= 1.0
    assert 0.0 <= m.phi_ema <= 1.0


# ── Active inference ────────────────────────────────────────────────────
def test_free_energy_processor():
    from neuroslm.intelligence.active_inference import FreeEnergyProcessor
    fe = FreeEnergyProcessor(d_sem=32, n_layers=2)
    out = fe(torch.randn(2, 32))
    assert isinstance(out, dict)
    for k in ("posterior", "free_energy", "epistemic_value",
              "pragmatic_value"):
        assert k in out, f"missing {k}"
    assert out["posterior"].shape == (2, 32)
    assert math.isfinite(float(out["free_energy"].mean().item()))


def test_free_energy_with_nt_modulation():
    from neuroslm.intelligence.active_inference import FreeEnergyProcessor
    fe = FreeEnergyProcessor(d_sem=16, n_layers=2)
    out = fe(torch.randn(2, 16),
             action_probs=torch.softmax(torch.randn(2, 14), dim=-1),
             nt_levels=torch.tensor([[0.8, 0.2, 0.5, 0.6]] * 2))
    assert "epistemic_value" in out
    assert "pragmatic_value" in out


def test_hierarchical_predictive_processor():
    from neuroslm.intelligence.active_inference import HierarchicalPredictiveProcessor
    p = HierarchicalPredictiveProcessor(d_in=32, d_hidden=32, n_layers=3)
    out = p(torch.randn(2, 32))
    if isinstance(out, tuple):
        post, fe_loss = out
        assert post.shape == (2, 32)
        assert math.isfinite(float(fe_loss.mean().item()))
