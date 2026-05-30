# -*- coding: utf-8 -*-
"""Forward-path Predictive Coding Trunk (PCT) tests.

PCT inserts a top-down prediction loop at every block boundary:

    pred_i = TopDown(h_{i+1})
    err_i  = h_i - pred_i
    h_i   <- h_i - alpha * pct_trunk * err_i

Properties to verify:
  1. With pct_trunk=0, forward is identical to the no-PCT baseline
     (parameter set differs by topdown_w which is unused when off).
  2. With pct_trunk>0, topdown_w is zero-initialised so the FIRST
     forward pass after init is also identical (residual identity
     preserved at start — the PCT mechanism turns on only as training
     pushes topdown_w away from zero).
  3. Once topdown_w is non-zero, the trunk output DOES change — the
     mechanism is wired into the forward path, not just an aux loss.
"""
import pytest
import torch

from neuroslm.dsl.nn_lang import build_dsl_language_cortex


VOCAB = 256
D_MODEL = 64
DEPTH = 4
N_HEADS = 4
MAX_CTX = 32


def _build(pct: float, seed: int = 0):
    torch.manual_seed(seed)
    return build_dsl_language_cortex(
        vocab=VOCAB, d_model=D_MODEL, depth=DEPTH,
        n_heads=N_HEADS, max_ctx=MAX_CTX, pct_trunk=pct)


def test_pct_off_matches_legacy_when_topdown_disabled():
    """pct_trunk=0 should produce identical logits to a model built
    without the topdown_w parameters at all (zero parameter delta)."""
    m_off = _build(pct=0.0)
    m_on = _build(pct=0.5)
    # Sync ALL shared params so any logit diff is due to PCT alone.
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
    # PCT is zero-init so the first-forward outputs must match
    # bit-for-bit (residual identity preserved at start).
    assert torch.allclose(l_off, l_on, atol=1e-6), \
        f"PCT changes the first forward despite zero-init topdown_w " \
        f"(max diff {(l_off - l_on).abs().max().item()})"


def test_pct_changes_output_when_topdown_nonzero():
    """After perturbing topdown_w the trunk output MUST change — proves
    the mechanism is in the forward path, not just an aux loss."""
    m = _build(pct=0.5)
    m.eval()
    ids = torch.randint(0, VOCAB, (2, 16))
    with torch.no_grad():
        l_before = m(ids)
        # Perturb topdown weights — should now influence forward
        for p in m.topdown_w:
            p.data.normal_(mean=0.0, std=0.01)
        l_after = m(ids)
    delta = (l_before - l_after).abs().max().item()
    assert delta > 1e-4, \
        f"Logits unchanged after perturbing topdown_w (delta {delta})" \
        " — PCT is not wired into the forward path"


def test_pct_off_means_topdown_w_is_none():
    """No PCT topdown params allocated when pct_trunk=0 — keeps the
    no-PCT baseline parameter-count identical to the legacy DSL trunk."""
    m = _build(pct=0.0)
    assert m.topdown_w is None
    n_pct_params = sum(p.numel() for n, p in m.named_parameters()
                       if "topdown_w" in n)
    assert n_pct_params == 0


def test_pct_on_allocates_n_minus_one_topdown_layers():
    """pct_trunk>0 must allocate `depth-1` topdown layers (one per pair)."""
    m = _build(pct=0.5)
    assert m.topdown_w is not None
    assert len(m.topdown_w) == DEPTH - 1
    for w in m.topdown_w:
        assert w.shape == (D_MODEL, D_MODEL)
        # Zero-init: norm should be effectively zero at construction
        assert w.abs().max().item() < 1e-9


def test_pct_gradient_flows_into_topdown_w():
    """A backward pass with any loss should produce non-zero gradients
    on the topdown_w parameters (otherwise they can never learn)."""
    m = _build(pct=0.5)
    m.train()
    # Perturb topdown so it's not exactly identity-residual; gradient
    # should flow even from zero-init via the prediction-error path.
    for p in m.topdown_w:
        p.data.normal_(mean=0.0, std=0.01)
    ids = torch.randint(0, VOCAB, (2, 16))
    targets = torch.randint(0, VOCAB, (2, 16))
    logits = m(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1))
    loss.backward()
    for i, w in enumerate(m.topdown_w):
        assert w.grad is not None, f"topdown_w[{i}] has no grad"
        assert w.grad.abs().max().item() > 1e-9, \
            f"topdown_w[{i}] grad is identically zero ({w.grad.abs().max()})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
