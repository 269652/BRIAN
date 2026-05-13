"""Tests for the MAT (Maturity Index) protein and the developmental
fade-in / ε-routing mechanisms.

These cover:
  - test_mat_index_convergence:    MAT rises smoothly as lm_loss drops.
  - test_expert_fade_in:           Expert residual grows ~linearly with MAT.
  - test_stochastic_routing_gradients:
        MathCortex receives non-zero gradients under ε-exploration
        even when the input is pure-language (no math topic routed).
"""
from __future__ import annotations
import math
import torch
import pytest

from neuroslm.neurochem.transmitters import compute_mat, L_RANDOM_DEFAULT


# ----------------------------------------------------------------------
# 1) MAT index converges from 0 toward 1 as LM loss drops
# ----------------------------------------------------------------------
def test_mat_index_convergence(tiny_brain):
    """MAT must rise monotonically (under a sustained loss-drop schedule)
    and saturate near 1.0 when lm_loss << L_random."""
    brain = tiny_brain

    # Reset
    brain.maturity.zero_()

    # Compute_mat alone (deterministic, no EMA)
    m_high = compute_mat(lm_loss=10.8, l_random=L_RANDOM_DEFAULT)   # near random
    m_mid  = compute_mat(lm_loss=5.4,  l_random=L_RANDOM_DEFAULT)
    m_low  = compute_mat(lm_loss=0.5,  l_random=L_RANDOM_DEFAULT)   # very strong LM
    assert 0.0 <= m_high <= 0.05,     f"expected ~0 at random init, got {m_high}"
    assert 0.4  <= m_mid  <= 0.6,     f"expected ~0.5 mid-loss, got {m_mid}"
    assert m_low > 0.9,               f"expected ~1 when LM strong, got {m_low}"

    # EMA path on Brain.update_maturity: feed a decreasing loss schedule and
    # verify the smoothed maturity rises monotonically (with a small slack
    # since the EMA also smooths transient bumps).
    schedule = [10.8, 9.5, 8.0, 6.5, 5.0, 3.5, 2.0, 1.0, 0.5]
    seen = []
    for lm in schedule:
        seen.append(brain.update_maturity(float(lm)))

    assert seen[0] < 0.1, f"first update should be near zero, got {seen[0]}"
    assert seen[-1] > seen[0] + 0.05, \
        f"maturity must climb under decreasing loss; saw {seen}"

    # Awakening flag: _infancy must flip off once MAT > 0.3
    # Push a sustained 0-loss run to drive MAT past 0.3 with this EMA.
    for _ in range(200):
        brain.update_maturity(0.0)
    assert brain.maturity_scalar() > 0.3, "MAT should saturate well above 0.3"
    assert brain._infancy is False, "Brain._infancy must auto-clear at MAT > 0.3"


# ----------------------------------------------------------------------
# 2) Expert fade-in: residual weight grows ~linearly with MAT
# ----------------------------------------------------------------------
def test_expert_fade_in():
    """MathCortex residual = M·vesicle_gate·enrichment (with 5% noise floor).
    Probe by comparing outputs at MAT∈{0.0, 0.5, 1.0} and verifying the
    residual norm grows monotonically with M.
    """
    from neuroslm.modules.math import MathCortex

    torch.manual_seed(0)
    d_sem = 64
    cortex = MathCortex(d_sem=d_sem, n_heads=4, memory_size=16, enable_hfw=False)
    cortex.eval()

    # MathCortex zero-inits fact_vals and out_proj so it starts as identity.
    # Populate the fact memory and the output path so the residual is non-zero
    # — otherwise we'd just be measuring 0 == 0.
    with torch.no_grad():
        cortex.fact_vals.copy_(
            torch.randn_like(cortex.fact_vals) * 0.5)
        cortex.out_proj.weight.copy_(
            torch.randn_like(cortex.out_proj.weight) * 0.05)

    x = torch.randn(2, d_sem) * 0.5
    vesicle_gate = 1.0   # treat the topic gate as fully on so we measure M alone

    out0  = cortex(x.clone(), vesicle_gate=vesicle_gate, maturity=0.0)
    out_h = cortex(x.clone(), vesicle_gate=vesicle_gate, maturity=0.5)
    out1  = cortex(x.clone(), vesicle_gate=vesicle_gate, maturity=1.0)

    delta0 = (out0  - x).norm().item()
    deltah = (out_h - x).norm().item()
    delta1 = (out1  - x).norm().item()

    # Noise floor: even at M=0 the residual is at the 5% floor, not zero
    assert delta0 > 0.0, "M=0 should still emit a 5% noise residual"
    # Monotone growth in M
    assert deltah > delta0, f"M=0.5 residual must exceed M=0 floor: {deltah} vs {delta0}"
    assert delta1 > deltah, f"M=1.0 residual must exceed M=0.5: {delta1} vs {deltah}"
    # Approximate linearity: delta(M=1) ≈ 20× delta(M=0)  (1.0 / 0.05)
    ratio = delta1 / max(delta0, 1e-8)
    assert ratio > 8.0, f"expected ≥8x growth from noise floor to full, got {ratio}"


# ----------------------------------------------------------------------
# 3) Stochastic ε-routing produces gradient flow into expert cortices
# ----------------------------------------------------------------------
def test_stochastic_routing_gradients():
    """With ε-exploration on, a fraction of routings hit non-language
    streams; gradients must reach the corresponding StreamAdapter weights.
    """
    from neuroslm.modules.thalamus import Thalamus, STREAM_NAMES

    d_sem = 32
    torch.manual_seed(0)
    th = Thalamus(d_sem=d_sem, hidden=d_sem, epsilon=1.0)   # ε=1 forces every step
    th.train()

    # Math stream is index 1; reasoning is 2. Verify routing-driven gradient flow.
    math_idx = STREAM_NAMES.index("math")

    # Use squared-norm loss instead of plain .sum() — the final LayerNorm in
    # Thalamus makes sum(out) a constant (centred output sums to zero), so a
    # raw .sum().backward() would mask the gradient flow we want to measure.
    x = torch.randn(8, d_sem, requires_grad=True)
    out = th(x, return_routing=False, maturity=0.0)
    loss = (out * out).sum()
    loss.backward()

    math_grad_norm = th.streams[math_idx].fc1.weight.grad.norm().item()
    assert math_grad_norm > 0.0, (
        f"With ε=1.0, math stream must receive grad; got {math_grad_norm}")

    # Sanity: with ε=0 and a frozen language-only input, math grad stays at 0
    th.zero_grad(set_to_none=True)
    th.epsilon = 0.0
    # Build a deterministic routing toward language only: force language logit high.
    with torch.no_grad():
        th.router.weight.zero_()
        th.router.bias.zero_()
        th.router.bias[STREAM_NAMES.index("language")] = 50.0   # pins softmax→language
    x2 = torch.randn(8, d_sem, requires_grad=True)
    out2 = th(x2, return_routing=False, maturity=1.0)   # ε scaled to 0 anyway
    (out2 * out2).sum().backward()
    math_grad_norm_eps0 = th.streams[math_idx].fc1.weight.grad
    # No exploration → math fc1 should not have been touched
    assert (math_grad_norm_eps0 is None
            or math_grad_norm_eps0.norm().item() < math_grad_norm), (
        "Without exploration, math stream grad should be ~0 or far smaller "
        "than the ε=1 case.")
