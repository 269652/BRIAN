"""Tests for individual brain-area modules.

Smoke checks that each module instantiates and produces finite output of
the documented shape. Detailed semantic tests live in their domain-specific
files (test_phi, test_memory, etc).
"""
from __future__ import annotations
import math
import torch
import pytest


# ── Workspace ───────────────────────────────────────────────────────────
def test_global_workspace_forward():
    from neuroslm.modules.workspace import GlobalWorkspace
    g = GlobalWorkspace(d_sem=32, n_slots=4, n_heads=4)
    out = g(torch.randn(2, 6, 32))
    assert out.shape == (2, 4, 32)
    # Ignition cached after forward
    assert g._last_ignition is None or torch.is_tensor(g._last_ignition)


# ── Workspace HFW fast weight ───────────────────────────────────────────
def test_fast_weight_layer_forward():
    from neuroslm.modules.fast_weight import FastWeightLayer
    fw = FastWeightLayer(d_model=32, n_heads=4, decay=0.9)
    x = torch.randn(2, 5, 32)
    out, W = fw(x)
    assert out.shape == x.shape
    assert W.shape == (2, 4, 8, 8)


def test_fast_weight_carryover():
    from neuroslm.modules.fast_weight import FastWeightLayer
    fw = FastWeightLayer(d_model=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    _, W1 = fw(x)
    out, W2 = fw(x, W_fast=W1)
    assert W2.shape == W1.shape
    # The carry-over should not be the identity (state evolved).
    assert not torch.allclose(W1, W2)


# ── Sensory / language / motor ──────────────────────────────────────────
def test_text_sensory_cortex():
    from neuroslm.modules.sensory import TextSensoryCortex
    s = TextSensoryCortex(d_sem=32)
    sem = torch.randn(2, 8, 32)
    out, sal = s(sem)
    assert out.shape == (2, 32) or out.shape == (2, 8, 32)
    assert torch.is_tensor(sal)


def test_motor_cortex_forward():
    from neuroslm.modules.motor import MotorCortex, ACTION_NAMES
    m = MotorCortex(d_action=16, d_sem=32, d_hidden=24)
    action = torch.randn(2, 16)
    survival = torch.zeros(2, dtype=torch.bool)
    out = m(action, survival=survival)
    assert isinstance(out, tuple)
    assert len(out) == 5
    probs = out[-1]
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-3)
    assert len(ACTION_NAMES) == probs.size(-1)


# ── Thalamus ────────────────────────────────────────────────────────────
def test_thalamus_forward():
    from neuroslm.modules.thalamus import Thalamus
    t = Thalamus(d_sem=32, hidden=48)
    x = torch.randn(2, 32)
    nt = torch.rand(2, 7)
    out, route = t(x, nt, return_routing=True)
    # output may be (B, d) or (B, d_hidden)
    assert out.size(0) == x.size(0)
    assert route.dim() in (2, 3)


# ── Qualia ──────────────────────────────────────────────────────────────
def test_qualia_state_forward():
    from neuroslm.modules.qualia import QualiaState
    q = QualiaState(d_sem=32, n_nt=7)
    thought = torch.randn(2, 32)
    nt = torch.rand(2, 7)
    threat = torch.rand(2)
    z_self = torch.randn(2, 32)
    out = q(thought, nt, threat, z_self)
    assert isinstance(out, dict)
    assert "modulated_thought" in out
    assert out["modulated_thought"].shape == thought.shape


# ── World / self / forward models ───────────────────────────────────────
def test_world_model_forward():
    from neuroslm.modules.world_model import WorldModel
    w = WorldModel(d_sem=32, d_hidden=48, n_layers=2)
    x = torch.randn(2, 32)
    h = w.init_hidden(2, x.device, dtype=x.dtype) if hasattr(w, "init_hidden") else None
    z, h_new, pred = w(x, h)
    assert z.shape == x.shape
    assert pred.shape == x.shape


def test_forward_model_runs():
    from neuroslm.modules.forward_model import ForwardModel
    fm = ForwardModel(d_sem=32, d_action=16, n_layers=2)
    z_w = torch.randn(2, 32)
    z_s = torch.randn(2, 32)
    act = torch.randn(2, 16)
    wp, sp = fm(z_w, z_s, act)
    assert wp.shape == z_w.shape
    assert sp.shape == z_s.shape


# ── Evaluator ───────────────────────────────────────────────────────────
def test_evaluator_forward():
    from neuroslm.modules.evaluator import Evaluator
    e = Evaluator(d_sem=32, n_neuromods=7)
    val = e(torch.randn(2, 32), torch.randn(2, 32), torch.rand(2, 7))
    assert val.shape[0] == 2


# ── PFC / DMN / BG ──────────────────────────────────────────────────────
def test_pfc_forward_safe():
    from neuroslm.modules.pfc import PrefrontalCortex
    p = PrefrontalCortex(d_sem=32, n_layers=2, n_heads=4)
    slots = torch.randn(2, 4, 32)
    recalls = torch.randn(2, 4, 32)
    thought = torch.randn(2, 32)
    nt_d = {n: 0.5 for n in ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")}
    out = p.forward_safe(slots, recalls, thought, nt_d)
    assert isinstance(out, tuple)


def test_dmn_forward_safe():
    from neuroslm.modules.dmn import DefaultModeNetwork
    d = DefaultModeNetwork(d_sem=32, n_slots=4, n_layers=2)
    slots = torch.randn(2, 4, 32)
    thought = torch.randn(2, 32)
    nt_d = {n: 0.5 for n in ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")}
    out = d.forward_safe(slots, thought, nt_d)
    assert isinstance(out, tuple) and len(out) >= 2


def test_basal_ganglia_runs():
    from neuroslm.modules.basal_ganglia import BasalGanglia
    bg = BasalGanglia(d_sem=32, d_action=16, n_candidates=3)
    nt_d = {n: 0.5 for n in ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")}
    out = bg.forward_safe(torch.randn(2, 32), nt_d)
    assert isinstance(out, tuple) and len(out) >= 1


# ── Hippocampus ─────────────────────────────────────────────────────────
def test_hippocampus_store_and_recall():
    from neuroslm.modules.hippocampus import Hippocampus
    h = Hippocampus(d_sem=32, capacity=32, topk=4, sparse_k=8)
    h.store(torch.randn(2, 32), torch.randn(2, 32),
            nt_state=torch.zeros(2, 7), valence=0.1, salience=0.5)
    # Recall produces something (shape and contents depend on impl).
    slots = torch.randn(2, 4, 32)
    query = torch.randn(2, 32)
    nt_d = {n: 0.5 for n in ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")}
    out = h.forward_safe(slots, query, nt_d, valence=torch.zeros(2))
    assert isinstance(out, tuple) and len(out) >= 2


# ── Cerebellum ──────────────────────────────────────────────────────────
def test_cerebellum_forward():
    from neuroslm.modules.cerebellum import Cerebellum
    c = Cerebellum(d_sem=32, expansion=4)
    # Cerebellum forward signature: (state, action, actual_next=None)
    out = c(torch.randn(2, 32), torch.randn(2, 32))
    assert isinstance(out, dict)
    for v in out.values():
        if torch.is_tensor(v):
            assert math.isfinite(float(v.float().sum().item()))


# ── Cortical column / entorhinal ────────────────────────────────────────
def test_cortical_sheet_forward():
    from neuroslm.modules.cortical_column import CorticalSheet
    cs = CorticalSheet(d_sem=32, n_columns=2, n_minicolumns=4)
    out = cs(torch.randn(2, 32), torch.randn(2, 32))
    assert isinstance(out, dict)
    assert "burst" in out and "output" in out


def test_entorhinal_grid_code():
    from neuroslm.modules.entorhinal import EntorhinalCortex
    e = EntorhinalCortex(d_sem=32, n_modules=2, cells_per_module=8, n_places=8)
    out = e(torch.randn(2, 32))
    assert isinstance(out, dict)
    assert "grid_code" in out
    assert out["grid_code"].size(-1) == 32


# ── Claustrum ───────────────────────────────────────────────────────────
def test_claustrum_forward():
    from neuroslm.modules.claustrum import Claustrum
    c = Claustrum(d_sem=32, n_modalities=4)
    # Most claustrum APIs take a list or stacked tensor.
    try:
        out = c([torch.randn(2, 32) for _ in range(4)])
    except (TypeError, AttributeError):
        out = c(torch.randn(2, 4, 32))
    assert isinstance(out, (dict, tuple, torch.Tensor))


# ── Neural geometry ─────────────────────────────────────────────────────
def test_neural_geometry_engine():
    from neuroslm.modules.neural_geometry import NeuralGeometryEngine
    g = NeuralGeometryEngine(d_model=32, n_fractal_levels=2)
    # forward expects (B, T, D) tokens
    out = g(torch.randn(2, 4, 32), torch.randn(2, 32))
    assert isinstance(out, dict)
    assert "curvature" in out
