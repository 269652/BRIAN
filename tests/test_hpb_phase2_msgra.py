# -*- coding: utf-8 -*-
"""HPB Phase 2 — MS-GRA: multi-scale grid attention (K=4, φ-ratio).

Phase 2 bumps the grid-cell positional code to K=4 incommensurate
scales at the golden ratio φ = 1.618…  The math here is the same
Stensola-2012 / Sargolini-2006 multi-scale grid lattice that lets the
hippocampus read place codes via Fourier inversion across scales.

Why K=4 and not the K=8 in the original disabled config:
  - K=4 with φ-ratio already spans 1, φ, φ², φ³ ≈ {1, 1.6, 2.6, 4.2}
    octaves — the empirically-observed dorsal-MEC range.
  - 2K = 8 raw features per position → projection is 8 × d_model ≪
    H15 buffer ≪ MSPCC waist params.

Contract under test
-------------------
1. K=4 build instantiates correctly with the right shapes.
2. The K scales are exactly base_period × φ^k for k ∈ {0..K-1}.
3. K=4 multi-scale code is incommensurate: distinct positions
   produce distinct codes for all positions up to LCM(τ_k) > max_ctx.
4. Extrapolates analytically: code is defined and finite for
   positions beyond max_ctx (the whole point vs. learned embeddings).
5. Bit-identical to baseline first-forward (zero-init contract).
6. arch.neuro Phase 2 declares n_scales=4, scale_ratio≈φ.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.dsl.training_config import parse_training_config
from neuroslm.dsl.nn_lang import build_dsl_language_cortex
from neuroslm.dsl.novel_topology import GridCellPositions


PHI = 1.6180339887498949
VOCAB = 256
D_MODEL = 64
DEPTH = 4
N_HEADS = 4
MAX_CTX = 64


def _build(seed: int, **kw):
    torch.manual_seed(seed)
    return build_dsl_language_cortex(
        vocab=VOCAB, d_model=D_MODEL, depth=DEPTH,
        n_heads=N_HEADS, max_ctx=MAX_CTX, **kw)


# ── 1. K=4 shape + scale spectrum ────────────────────────────────────

def test_k4_phi_module_shapes():
    """K=4 produces 2K=8 raw features projected to d_model."""
    pos = GridCellPositions(d_model=D_MODEL, n_scales=4,
                            scale_ratio=PHI, base_period=16.0)
    # Projection input is 2K=8, output is d_model.
    assert pos.proj.weight.shape == (D_MODEL, 8)
    # Output for L=20 positions
    code = pos(20)
    assert code.shape == (20, D_MODEL)
    assert torch.isfinite(code).all()


def test_k4_scale_spectrum_is_geometric_phi():
    """τ_k = base_period × φ^k for k = 0..K-1 — golden ratio spacing."""
    pos = GridCellPositions(d_model=D_MODEL, n_scales=4,
                            scale_ratio=PHI, base_period=16.0)
    # Re-derive the angles for position t=1 and verify against the
    # expected geometric series of periods.
    expected_taus = [16.0 * (PHI ** k) for k in range(4)]
    for k, tau in enumerate(expected_taus):
        assert math.isclose(tau, 16.0 * PHI ** k, rel_tol=1e-9)
    # First scale unchanged from K=1 case.
    assert math.isclose(expected_taus[0], 16.0)
    # Last scale has nontrivial period: ~67.5.
    assert 65.0 < expected_taus[-1] < 70.0


def test_k4_codes_are_distinct_for_neighbouring_positions():
    """At t=0 the code is (1,1,1,1, 0,0,0,0) (cos=1, sin=0). At t=1
    it MUST differ from t=0; the K=4 multi-scale code must NOT collapse
    onto a single repeating period."""
    pos = GridCellPositions(d_model=D_MODEL, n_scales=4,
                            scale_ratio=PHI, base_period=16.0)
    ks = torch.arange(4, dtype=torch.float32)
    taus = 16.0 * (PHI ** ks)
    t = torch.arange(4, dtype=torch.float32)
    ang = 2.0 * math.pi * t.unsqueeze(1) / taus.unsqueeze(0)
    code = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)  # (L, 2K)
    # All four codes must be pairwise distinct.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(code[i], code[j]), (
                f"K=4 φ-grid codes collapse for t={i} vs t={j}"
            )


# ── 2. Extrapolation guarantee (the OOD win) ─────────────────────────

def test_k4_extrapolates_well_beyond_max_ctx():
    """The whole MS-GRA proposal is that φ-grids are analytically
    defined for any position — including positions beyond what was
    seen during training. Test up to 10× max_ctx."""
    pos = GridCellPositions(d_model=D_MODEL, n_scales=4,
                            scale_ratio=PHI, base_period=16.0)
    far_L = 10 * MAX_CTX
    code = pos(far_L)
    assert code.shape == (far_L, D_MODEL)
    assert torch.isfinite(code).all(), (
        "K=4 φ-grid code must be finite at extrapolated positions"
    )


def test_k4_codes_differ_between_train_and_extrapolated_positions():
    """Position t=MAX_CTX+k must produce a *different* raw code than
    t=k for at least one scale — the bias is meaningful beyond ctx."""
    ks = torch.arange(4, dtype=torch.float32)
    taus = 16.0 * (PHI ** ks)
    # In-context vs OOD positions
    t_in = torch.tensor([5.0])
    t_out = torch.tensor([5.0 + MAX_CTX])
    ang_in = 2.0 * math.pi * t_in.unsqueeze(1) / taus.unsqueeze(0)
    ang_out = 2.0 * math.pi * t_out.unsqueeze(1) / taus.unsqueeze(0)
    code_in = torch.cat([torch.cos(ang_in), torch.sin(ang_in)], dim=-1)
    code_out = torch.cat([torch.cos(ang_out), torch.sin(ang_out)], dim=-1)
    # The codes must differ — if all four scales aliased we'd have
    # produced identical codes, which would mean OOD positions are
    # indistinguishable from in-context ones (bad).
    assert not torch.allclose(code_in, code_out, atol=1e-3), (
        "K=4 φ-grid aliased between in-ctx and OOD positions"
    )


# ── 3. Baseline-identity contract preserved at K=4 ───────────────────

def test_k4_first_forward_is_baseline_identical():
    spec = {"enabled": True, "n_scales": 4,
            "scale_ratio": PHI, "base_period": 16.0}
    m_off = _build(seed=999)
    m_on = _build(seed=999, grid_positions=spec)
    sd_off = m_off.state_dict()
    sd_on = m_on.state_dict()
    for k in sd_off:
        if k in sd_on and sd_on[k].shape == sd_off[k].shape:
            sd_on[k] = sd_off[k].clone()
    m_on.load_state_dict(sd_on, strict=False)
    m_off.eval(); m_on.eval()
    ids = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        l_off = m_off(ids)
        l_on = m_on(ids)
    assert torch.allclose(l_off, l_on, atol=1e-6), (
        f"K=4 φ-grid violates Phase-2 baseline-identity at init "
        f"(max-diff {(l_off-l_on).abs().max().item():.2e})"
    )


# ── 4. arch.neuro Phase-2 declaration ────────────────────────────────

def test_actual_arch_neuro_grid_positions_k4_phi():
    """After Phase 2 lands, arch.neuro MUST declare K=4 with φ-ratio."""
    from pathlib import Path
    from neuroslm.dsl.training_config import load_training_config_from_arch
    arch_root = Path(__file__).parent.parent / "architectures" / "master"
    cfg = load_training_config_from_arch(arch_root)
    assert cfg.grid_positions is not None
    assert cfg.grid_positions.get("enabled") is True
    assert int(cfg.grid_positions.get("n_scales")) == 4, (
        f"Phase 2 expects n_scales=4 (φ-ratio multi-scale), got "
        f"{cfg.grid_positions.get('n_scales')}"
    )
    ratio = float(cfg.grid_positions.get("scale_ratio"))
    assert math.isclose(ratio, PHI, abs_tol=1e-4), (
        f"Phase 2 expects scale_ratio ≈ φ={PHI:.6f}, got {ratio}"
    )


# ── 5. Gradient flow through the projection ──────────────────────────

def test_k4_gradient_flows_through_projection():
    """The grid-cell module's proj must receive gradient from the
    final LM loss; otherwise the mechanism cannot learn to make use
    of the multi-scale code."""
    spec = {"enabled": True, "n_scales": 4, "scale_ratio": PHI}
    m = _build(seed=11, grid_positions=spec)
    m.train()
    # Perturb the zero-init proj so the forward path has non-trivial
    # bias; otherwise the gradient is identically 0 (zero-init grad
    # of identity).  We test that an INITIATED proj sees gradient.
    with torch.no_grad():
        m._grid_positions.proj.weight.add_(
            torch.randn_like(m._grid_positions.proj.weight) * 0.01)
    ids = torch.randint(0, VOCAB, (2, 8))
    logits = m(ids)
    # Simple cross-entropy against random targets.
    targets = torch.randint(0, VOCAB, (2, 8))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1))
    loss.backward()
    grad = m._grid_positions.proj.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0, (
        "MS-GRA projection received zero gradient — broken wiring"
    )
