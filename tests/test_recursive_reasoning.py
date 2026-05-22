"""Tests for the recursive reasoning cortex (§5.4).

`ReasoningCortex.forward_tokens` loops its `expert_blocks` `recursive_iters`
times with weight-sharing. Output flows through the existing bowtie / thought
/ from_sem path (ReZero-gated by λ_thought) — no new gradient paths into the
LM trunk. Depth-multiplies the reasoning expert at constant parameter count,
trading FLOPs for representational depth.
"""
from __future__ import annotations
import torch

from neuroslm.config import tiny, BrainConfig
from neuroslm.modules.reasoning import ReasoningCortex
from neuroslm.brain import Brain


def _cortex(recursive_iters: int) -> ReasoningCortex:
    return ReasoningCortex(
        d_sem=64, n_attractors=16, base_beta=4.0,
        d_hidden=48, n_blocks=2, max_ctx=64, expert_n_heads=4,
        recursive_iters=recursive_iters,
    )


# ── Config defaults ─────────────────────────────────────────────────────────

def test_recursive_defaults():
    c = BrainConfig()
    assert c.recursive_reasoning is True
    assert c.recursive_iters == 4


# ── Cortex behaviour ────────────────────────────────────────────────────────

def test_recursive_iters_stored_on_cortex():
    rc = _cortex(recursive_iters=4)
    assert rc.recursive_iters == 4


def test_recursive_iters_clamps_to_at_least_one():
    """Pathological 0 or negative collapses to 1 (legacy single pass)."""
    assert _cortex(recursive_iters=0).recursive_iters == 1
    assert _cortex(recursive_iters=-1).recursive_iters == 1


def test_recursion_changes_forward_tokens_output():
    """Looping the expert blocks N times must change the output vs N=1 —
    proof that the recursion actually runs."""
    torch.manual_seed(0); rc1 = _cortex(recursive_iters=1)
    torch.manual_seed(0); rcN = _cortex(recursive_iters=4)
    # Identical weights (same seed); only the loop count differs.
    assert torch.equal(rc1.attractors_h, rcN.attractors_h)
    x = torch.randn(2, 8, rc1.d_hidden)
    rc1.eval(); rcN.eval()
    with torch.no_grad():
        y1 = rc1.forward_tokens(x.clone(), maturity=1.0)
        yN = rcN.forward_tokens(x.clone(), maturity=1.0)
    assert not torch.allclose(y1, yN, atol=1e-5)


def test_recursion_is_weight_sharing_not_param_growth():
    """Param count must be invariant to recursive_iters — recursion shares
    the same expert_blocks across iterations rather than stacking more."""
    rc1 = _cortex(recursive_iters=1)
    rcN = _cortex(recursive_iters=8)
    p1 = sum(p.numel() for p in rc1.parameters())
    pN = sum(p.numel() for p in rcN.parameters())
    assert p1 == pN, f"recursion must not add params: {p1} vs {pN}"


def test_recursive_forward_tokens_preserves_shape_and_is_finite():
    rc = _cortex(recursive_iters=4); rc.eval()
    x = torch.randn(3, 12, rc.d_hidden)
    with torch.no_grad():
        y = rc.forward_tokens(x, maturity=1.0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ── End-to-end (Brain.forward_lm) ───────────────────────────────────────────

def test_brain_threads_recursive_iters_to_cortex():
    c = tiny(); c.vocab_size = 256
    c.recursive_reasoning = True; c.recursive_iters = 4
    torch.manual_seed(0); b = Brain(c)
    assert b.reasoning_cortex.recursive_iters == 4
    # Toggle off → single pass.
    c2 = tiny(); c2.vocab_size = 256
    c2.recursive_reasoning = False; c2.recursive_iters = 4
    torch.manual_seed(0); b2 = Brain(c2)
    assert b2.reasoning_cortex.recursive_iters == 1


def test_full_brain_forward_backward_with_recursive_reasoning():
    c = tiny(); c.vocab_size = 256
    c.recursive_reasoning = True; c.recursive_iters = 4
    torch.manual_seed(0); b = Brain(c); b.train()
    ids = torch.randint(0, 256, (2, 16))
    tgt = torch.randint(0, 256, (2, 16))
    loss = b.forward_lm(ids, tgt)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
