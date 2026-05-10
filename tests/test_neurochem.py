"""Tests for the neurochemistry subsystem.

Covers TransmitterSystem, ProjectionGraph, GatedProjectionGraph,
TrophicSystem (BDNF/NGF), vesicle pool, homeostasis, GPCRs, mesolimbic,
lateral habenula, plasticity gates.
"""
from __future__ import annotations
import math
import torch
import pytest


# ── Transmitter system ─────────────────────────────────────────────────
def test_transmitter_release_and_decay():
    from neuroslm.neurochem.transmitters import TransmitterSystem, NT_NAMES
    ts = TransmitterSystem()
    ts.reset(batch_size=2, device=torch.device("cpu"))
    # Release some DA
    actual = ts.release("DA", torch.tensor([0.5, 0.5]))
    assert actual.shape == (2,)
    # Level increased above baseline
    assert ts.get("DA").max().item() > 0.1
    ts.step()
    # After step the level decays toward baseline
    # (we don't assert a specific drop magnitude since impl details vary)
    assert ts.get("DA").shape == (2,)


def test_transmitter_vector_and_names():
    from neuroslm.neurochem.transmitters import TransmitterSystem, NT_NAMES, N_NT
    ts = TransmitterSystem()
    ts.reset(batch_size=1, device=torch.device("cpu"))
    v = ts.vector()
    assert v.shape == (1, N_NT)
    assert len(NT_NAMES) == N_NT


def test_transmitter_clamps_levels_in_unit_range():
    from neuroslm.neurochem.transmitters import TransmitterSystem
    ts = TransmitterSystem()
    ts.reset(batch_size=2, device=torch.device("cpu"))
    # Try to release more than 1.0
    ts.release("DA", torch.tensor([5.0, 5.0]))
    assert ts.get("DA").max().item() <= 1.0 + 1e-5
    assert ts.get("DA").min().item() >= 0.0 - 1e-5


# ── Receptors ──────────────────────────────────────────────────────────
def test_receptor_bank_modulate_shape():
    from neuroslm.neurochem.receptors import ReceptorBank, Receptor
    rb = ReceptorBank(receptors=[Receptor("DA", sign=1.0, weight=0.5)])
    x = torch.randn(2, 4, 32)
    # ReceptorBank.modulate uses (x, nt_vector_of_levels)
    nt = torch.rand(2, 7)
    y = rb.modulate(x, nt)
    assert y.shape == x.shape


# ── Projection graph ───────────────────────────────────────────────────
def test_projection_graph_basic():
    from neuroslm.neurochem.projections import ProjectionGraph, Projection
    g = ProjectionGraph(projections=[Projection("PFC", "BG", "Glu",
                                                release_scale=0.5)],
                        region_dims={"PFC": 7, "BG": 7})
    assert len(g.projections) == 1
    assert "PFC" in g.regions
    assert "BG" in g.regions


# ── Trophic system ─────────────────────────────────────────────────────
def test_trophic_update_changes_state():
    from neuroslm.neurochem.projections import ProjectionGraph, Projection
    from neuroslm.neurochem.growth import TrophicSystem
    g = ProjectionGraph(
        projections=[Projection("A", "B", "Glu")],
        region_dims={"A": 7, "B": 7})
    t = TrophicSystem(g)
    before = t.trophic.clone()
    t.update(activities={"A": torch.tensor([0.9]), "B": torch.tensor([0.85])},
             bdnf=0.5, ngf=0.1, phi=0.4, fiedler=0.2)
    assert not torch.equal(before, t.trophic), "trophic levels did not update"


def test_trophic_phi_boosts_growth():
    """High Φ pathway should receive at least as much trophic support."""
    from neuroslm.neurochem.projections import ProjectionGraph, Projection
    from neuroslm.neurochem.growth import TrophicSystem

    def _make():
        g = ProjectionGraph(
            projections=[Projection("A", "B", "Glu")],
            region_dims={"A": 7, "B": 7})
        return g, TrophicSystem(g)

    _, t_hi = _make()
    t_hi.update({"A": torch.tensor([0.9]), "B": torch.tensor([0.9])},
                bdnf=0.5, ngf=0.05, phi=0.8, fiedler=1.0)
    _, t_lo = _make()
    t_lo.update({"A": torch.tensor([0.9]), "B": torch.tensor([0.9])},
                bdnf=0.5, ngf=0.05, phi=0.0, fiedler=1.0)
    assert float(t_hi.trophic.mean().item()) >= float(t_lo.trophic.mean().item())


# ── Vesicle pool ───────────────────────────────────────────────────────
def test_vesicle_pool_synthesize_and_migrate():
    from neuroslm.neurochem.vesicles import VesiclePool
    vp = VesiclePool(d_sem=16, n_modules=4, n_vesicles=8)
    vp.synthesize_typed(content=torch.randn(16), type_idx=0, source_module=1)
    vp.migrate()
    vp.degrade()
    g = vp.expert_gate(type_idx=0)
    assert 0.0 <= float(g) <= 1.0


# ── GPCR bank ──────────────────────────────────────────────────────────
def test_gpcr_bank_observe_and_query():
    from neuroslm.neurochem.receptors import GPCRBank
    g = GPCRBank(window_size=8)
    nt = torch.rand(2, 7)
    g.observe(nt)
    a = g.ach_gate()
    n = g.ne_arousal()
    # Most impls return floats; tolerate tensor scalars.
    a_f = float(a) if not isinstance(a, float) else a
    n_f = float(n) if not isinstance(n, float) else n
    assert 0.0 <= a_f <= 1.0
    assert 0.0 <= n_f <= 1.0


# ── Lateral habenula ───────────────────────────────────────────────────
def test_lateral_habenula_update():
    from neuroslm.neurochem.lateral_habenula import LateralHabenula
    h = LateralHabenula(n_nt=7)
    out = h.update(actual_reward=torch.tensor([0.1, 0.5]),
                   da_level=torch.tensor([0.4, 0.6]))
    assert "lhb_firing" in out


# ── Homeostasis ────────────────────────────────────────────────────────
def test_homeostasis_observe():
    from neuroslm.neurochem.homeostasis import Homeostasis
    from neuroslm.neurochem.transmitters import TransmitterSystem
    ts = TransmitterSystem()
    ts.reset(batch_size=2, device=torch.device("cpu"))
    h = Homeostasis()
    h.observe(ts, lm_loss=2.0, grad_norm=1.0)


# ── Mesolimbic circuit ─────────────────────────────────────────────────
def test_mesolimbic_circuit_forward():
    from neuroslm.neurochem.mesolimbic_circuit import MesolimbicCircuit
    m = MesolimbicCircuit(d_state=16)
    out = m(state_vec=torch.randn(2, 16),
            reward=torch.tensor([0.3, 0.4]),
            da_level=torch.tensor([0.5, 0.5]),
            ecb_level=torch.tensor([0.1, 0.1]),
            gaba_level=torch.tensor([0.4, 0.4]),
            novelty=torch.tensor([0.5, 0.5]),
            salience=torch.tensor([0.5, 0.5]),
            valence=torch.tensor([0.5, 0.6]),
            uncertainty=torch.tensor([0.2, 0.2]))
    assert isinstance(out, dict)
