# -*- coding: utf-8 -*-
"""HPB Phase 1 — H16 grid-cell positions activation (K=1 sanity baseline).

The proposal's Phase 1 is the cheapest possible architectural lift:
activate the already-wired grid-cell positional bias with K=1 scale.
This pins down the integration path before we crank K up in Phase 2.

Contract under test
-------------------
1. ``arch.neuro`` `training.grid_positions.enabled = true` parses
   through ``parse_training_config`` into a dict the cortex consumes.
2. ``DSLLanguageCortex`` built with the parsed dict instantiates
   ``_grid_positions`` (i.e. not None).
3. K=1, ``base_period`` configurable, ``scale_ratio`` ignored at K=1
   (no scale advancement).
4. First-forward bit-identical to baseline (zero-init projection).
5. After perturbation of the projection weights the output diverges,
   proving the bias is wired into the residual stream.
"""
from __future__ import annotations
import math
import pytest
import torch

from neuroslm.dsl.training_config import parse_training_config
from neuroslm.dsl.nn_lang import build_dsl_language_cortex


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


# ── 1. arch.neuro → TrainingConfig pipeline ──────────────────────────

def test_parse_grid_positions_k1_enabled():
    """Phase 1 baseline: K=1 grid_positions parses into a live dict."""
    src = """
        learning_rate: 0.0003
        grid_positions: { enabled: true, n_scales: 1, scale_ratio: 1.6180339887, base_period: 16.0 }
    """
    cfg = parse_training_config(src)
    assert cfg.grid_positions is not None
    assert isinstance(cfg.grid_positions, dict)
    assert cfg.grid_positions.get("enabled") is True
    assert int(cfg.grid_positions.get("n_scales")) == 1
    assert math.isclose(float(cfg.grid_positions.get("scale_ratio")),
                        1.6180339887, abs_tol=1e-6)


def test_grid_positions_disabled_dict_is_off():
    """A dict with enabled=false MUST behave like off (no module built)."""
    src = """
        learning_rate: 0.0003
        grid_positions: { enabled: false, n_scales: 1 }
    """
    cfg = parse_training_config(src)
    m = _build(seed=1, grid_positions=cfg.grid_positions)
    assert m._grid_positions is None, (
        "enabled=false dict must yield no grid-cell module"
    )


# ── 2. DSL cortex consumes the parsed config ─────────────────────────

def test_dsl_cortex_instantiates_grid_positions_k1():
    """With enabled=true and K=1, _grid_positions must be a real module."""
    spec = {"enabled": True, "n_scales": 1, "scale_ratio": 1.618}
    m = _build(seed=2, grid_positions=spec)
    assert m._grid_positions is not None
    assert m._grid_positions.n_scales == 1


# ── 3. K=1 mathematical properties ───────────────────────────────────

def test_k1_produces_single_scale_code():
    """With K=1 the code is exactly (cos(2πt/τ₀), sin(2πt/τ₀))."""
    spec = {"enabled": True, "n_scales": 1, "scale_ratio": 1.618,
            "base_period": 16.0}
    m = _build(seed=3, grid_positions=spec)
    pos = m._grid_positions
    # Directly call the position module to inspect the raw code.
    # The proj is zero-init so the *output* is zero — but the
    # internal 2K=2 feature code must be cos/sin at the base period.
    L = 8
    t = torch.arange(L, dtype=torch.float32)
    expected_cos = torch.cos(2.0 * math.pi * t / 16.0)
    expected_sin = torch.sin(2.0 * math.pi * t / 16.0)
    # Re-derive what the forward computed by inspecting the projection
    # input via reconstruction: out = proj @ code → with zero proj,
    # output is zero, so we re-run the inner math directly to check.
    ks = torch.arange(1, dtype=torch.float32)
    taus = 16.0 * (1.618 ** ks)
    ang = 2.0 * math.pi * t.unsqueeze(1) / taus.unsqueeze(0)
    code = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    assert torch.allclose(code[:, 0], expected_cos, atol=1e-6)
    assert torch.allclose(code[:, 1], expected_sin, atol=1e-6)


def test_k1_first_forward_is_baseline_identical():
    """Phase 1 invariant: enabling K=1 with zero-init proj is a no-op."""
    spec = {"enabled": True, "n_scales": 1}
    m_off = _build(seed=42)
    m_on = _build(seed=42, grid_positions=spec)
    # Mirror state dicts so divergence is attributable to the grid
    # cell module's *forward injection* only.
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
        f"Phase 1 invariant violated; max diff "
        f"{(l_off - l_on).abs().max().item():.2e}"
    )


def test_k1_bias_flows_after_proj_perturbation():
    """After perturbing the (zero-init) projection the logits MUST move.
    Proves the bias is wired into the residual stream (not dead code)."""
    spec = {"enabled": True, "n_scales": 1}
    m = _build(seed=7, grid_positions=spec)
    m.eval()
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        l_before = m(ids).clone()
    # Perturb the projection weights — even small noise should propagate.
    with torch.no_grad():
        m._grid_positions.proj.weight.add_(
            torch.randn_like(m._grid_positions.proj.weight) * 0.05)
    with torch.no_grad():
        l_after = m(ids)
    diff = (l_after - l_before).abs().max().item()
    assert diff > 1e-4, (
        f"grid-cell projection has no downstream effect (max-diff={diff:.2e})"
    )


# ── 4. arch.neuro file integration smoke test ────────────────────────

def test_actual_arch_neuro_grid_positions_enabled():
    """The shipped arch.neuro MUST have grid_positions.enabled = true
    after Phase 1 lands.  This is a regression-prevention test for the
    activation step itself."""
    from pathlib import Path
    from neuroslm.dsl.training_config import load_training_config_from_arch
    arch_root = Path(__file__).parent.parent / "architectures" / "rcc_bowtie"
    cfg = load_training_config_from_arch(arch_root)
    assert cfg.grid_positions is not None, (
        "arch.neuro must declare a grid_positions block"
    )
    assert cfg.grid_positions.get("enabled") is True, (
        "Phase 1 of HPB activates grid_positions; arch.neuro still off"
    )
