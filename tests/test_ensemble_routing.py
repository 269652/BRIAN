# -*- coding: utf-8 -*-
"""Multi-Cortex Thalamic Routing — TDD acceptance suite.

Validates the ``neuroslm.cortex.MultiCortexEnsemble`` design:

  • A pool of N specialist sub-cortices (math / code / chat / general)
    each implementing the same forward signature ``(B, T) -> (B, T, D)``.
  • A ``ThalamicRouter`` that maps the input token sequence to per-token
    routing weights ``(B, T, N)``, with two additive sources:
        – a hard-coded *lexical bias*  (domain-keyword prior, on at init)
        – a learnable MLP head         (zero-init, trains under LM loss)
    plus optional BEMA biological damping (running EMA over batches) to
    suppress per-token routing oscillations.
  • A weighted mixture of the sub-cortex hidden states ⇒ a single
    ``(B, T, d_target)`` output that drops into the existing trunk.

The headline acceptance test is :func:`test_math_question_routes_to_math_cortex`
— the user's explicit success criterion: feeding a math-heavy sequence
must yield the highest mean thalamic routing weight on the
math-specialist cortex, *at initialisation*, before any LM gradient has
moved the learnable router head off zero.

Synthetic vocabulary partition
------------------------------
To keep the tests deterministic and free of any HuggingFace download,
we use a 200-token vocab partitioned by domain::

    0–29   → math    (digits, operators, math nouns)
    30–59  → code    (def, class, import, …)
    60–89  → chat    (you, please, ?, …)
    90–199 → general (everything else / fallback)

The lexical bias is therefore a deterministic function of token id,
which means at step 0 the routing distribution is *exactly* the
softmax of the per-token one-hot bias, and the keystone test passes
analytically.
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.cortex import (
    SubCortex,
    StubSubCortex,
    ThalamicRouter,
    MultiCortexEnsemble,
    DomainLexicon,
    build_default_ensemble,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic test vocabulary
# ──────────────────────────────────────────────────────────────────────

VOCAB = 200
MATH_TOKENS    = list(range(0,   30))
CODE_TOKENS    = list(range(30,  60))
CHAT_TOKENS    = list(range(60,  90))
GENERAL_TOKENS = list(range(90,  VOCAB))

DOMAIN_TOKEN_MAP = {
    "math":    MATH_TOKENS,
    "code":    CODE_TOKENS,
    "chat":    CHAT_TOKENS,
    "general": GENERAL_TOKENS,
}
DOMAINS = ["math", "code", "chat", "general"]


def make_question(domain: str, length: int = 32,
                  noise_ratio: float = 0.1,
                  seed: int = 0) -> torch.Tensor:
    """Build a ``(1, length)`` sequence biased toward ``domain``.

    ``noise_ratio`` fraction of positions hold tokens drawn from the
    *other* domains, in deterministic round-robin order, so the test is
    fully reproducible.  The final positions are shuffled by a per-domain
    seed so the bias is not confined to a contiguous prefix (would let
    the router cheat with a position-only heuristic).
    """
    g = torch.Generator().manual_seed(hash((domain, seed)) & 0x7FFFFFFF)
    domain_tokens = DOMAIN_TOKEN_MAP[domain]
    noise_pool = [t for d, toks in DOMAIN_TOKEN_MAP.items() if d != domain
                  for t in toks]
    n_noise = int(round(length * noise_ratio))
    n_domain = length - n_noise
    domain_part = torch.tensor(
        [domain_tokens[i % len(domain_tokens)] for i in range(n_domain)],
        dtype=torch.long)
    noise_part = torch.tensor(
        [noise_pool[i % len(noise_pool)] for i in range(n_noise)],
        dtype=torch.long)
    seq = torch.cat([domain_part, noise_part])
    perm = torch.randperm(length, generator=g)
    return seq[perm].unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def lexicon():
    return DomainLexicon(domain_token_map=DOMAIN_TOKEN_MAP)


@pytest.fixture
def ensemble(lexicon):
    """4-cortex ensemble, one stub per domain, no BEMA damping."""
    torch.manual_seed(0)
    sub_cortices = [
        StubSubCortex(name=f"cortex_{d}", domain=d, vocab=VOCAB,
                      d_model=64, n_layers=2, n_heads=4, max_ctx=64)
        for d in DOMAINS
    ]
    router = ThalamicRouter(
        vocab_size=VOCAB, d_model=64, domains=DOMAINS,
        lexicon=lexicon, lexical_bias_weight=2.0, bema_tau=0.0,
    )
    return MultiCortexEnsemble(
        sub_cortices=sub_cortices, router=router, d_target=64,
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Construction & shape contracts
# ──────────────────────────────────────────────────────────────────────

def test_ensemble_construction(ensemble):
    """4 sub-cortices, parameters declared, forward shape correct."""
    assert len(ensemble.sub_cortices) == 4
    assert sum(p.numel() for p in ensemble.parameters()) > 0
    ids = torch.randint(0, VOCAB, (2, 16))
    out = ensemble(ids)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_routing_weights_are_a_probability_simplex(ensemble):
    """Per-token routing weights are non-negative and sum to 1 over the
    cortex axis (standard probability-simplex contract)."""
    ids = torch.randint(0, VOCAB, (2, 16))
    ensemble.eval()
    with torch.no_grad():
        ensemble(ids)
    w = ensemble.last_routing_weights
    assert w.shape == (2, 16, 4)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert (w >= 0).all()


def test_sub_cortices_expose_domain_label(ensemble):
    """Each sub-cortex carries its domain label for telemetry / routing
    inspection."""
    domains = [c.domain for c in ensemble.sub_cortices]
    assert set(domains) == set(DOMAINS)
    names = [c.name for c in ensemble.sub_cortices]
    assert all(n.startswith("cortex_") for n in names)


# ──────────────────────────────────────────────────────────────────────
# 2. THE KEYSTONE TEST — math question routes to math cortex
# ──────────────────────────────────────────────────────────────────────

def test_math_question_routes_to_math_cortex(ensemble):
    """The user's headline acceptance criterion: when a math-heavy
    sequence is fed in, the thalamic router awards the highest mean
    routing weight to the math-specialist cortex *at initialisation*.

    Mathematics of the assertion
    ----------------------------
    With ``lexical_bias_weight = 2.0`` and zero-init learnable logits,
    a token belonging to domain ``d`` produces the per-token logit
    vector with a single 2.0 entry at position ``d``. Softmax gives::

        p(d | token∈d) = e² / (e² + 3) ≈ 0.730
        p(d'| token∈d) = 1   / (e² + 3) ≈ 0.090   for d' ≠ d

    A length-32 sequence with 29 math tokens + 3 non-math tokens
    therefore yields mean routing weight ≈ 0.67 on the math cortex,
    which the test below requires to dominate the runner-up by ≥ 1.5×.
    """
    math_ids = make_question("math", length=32, noise_ratio=0.1)
    ensemble.eval()
    with torch.no_grad():
        ensemble(math_ids)
    w = ensemble.last_routing_weights.mean(dim=(0, 1))  # (N,)
    weights_dict = dict(zip(DOMAINS, w.tolist()))
    math_idx = DOMAINS.index("math")
    assert w.argmax().item() == math_idx, (
        f"Math question failed to route to math cortex. "
        f"Weights = {weights_dict}"
    )
    other_max = torch.cat([w[:math_idx], w[math_idx + 1:]]).max()
    assert w[math_idx] > 1.5 * other_max, (
        f"Math cortex weight ({w[math_idx]:.3f}) does not dominate the "
        f"runner-up ({other_max:.3f}) by the required 1.5× margin. "
        f"Full weights = {weights_dict}"
    )


def test_code_question_routes_to_code_cortex(ensemble):
    """Symmetry check: a code-heavy sequence must route to the code
    cortex by the same lexical-bias mechanism."""
    code_ids = make_question("code", length=32, noise_ratio=0.1)
    ensemble.eval()
    with torch.no_grad():
        ensemble(code_ids)
    w = ensemble.last_routing_weights.mean(dim=(0, 1))
    assert w.argmax().item() == DOMAINS.index("code"), (
        f"Code question routing failed: {dict(zip(DOMAINS, w.tolist()))}"
    )


def test_chat_question_routes_to_chat_cortex(ensemble):
    chat_ids = make_question("chat", length=32, noise_ratio=0.1)
    ensemble.eval()
    with torch.no_grad():
        ensemble(chat_ids)
    w = ensemble.last_routing_weights.mean(dim=(0, 1))
    assert w.argmax().item() == DOMAINS.index("chat"), (
        f"Chat question routing failed: {dict(zip(DOMAINS, w.tolist()))}"
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Lexical-bias-only-at-init invariant
# ──────────────────────────────────────────────────────────────────────

def test_learnable_logits_are_zero_init(ensemble):
    """The learnable router head must be zero-initialised so that the
    first forward is deterministically the lexical-bias softmax.  This
    is the ReZero contract — it lets us merge the routing extension into
    a live training run without disturbing baseline behaviour."""
    w = ensemble.router.learnable_logits.weight
    assert torch.allclose(w, torch.zeros_like(w)), (
        f"Router learnable_logits is not zero-initialised: "
        f"|w|_max = {w.abs().max().item()}"
    )


def test_pure_math_sequence_gives_pure_lexical_routing(ensemble):
    """Sanity contrapositive: a length-8 sequence with ZERO noise gives
    routing weights that exactly match the analytic softmax of the
    per-token bias.  Confirms there is no hidden source of randomness."""
    math_ids = make_question("math", length=8, noise_ratio=0.0)
    ensemble.eval()
    with torch.no_grad():
        ensemble(math_ids)
    w = ensemble.last_routing_weights.mean(dim=(0, 1))
    # Analytic: p(math) = e²/(e²+3), p(other) = 1/(e²+3)
    import math
    e2 = math.exp(2.0)
    expected_math = e2 / (e2 + 3.0)
    expected_other = 1.0 / (e2 + 3.0)
    math_idx = DOMAINS.index("math")
    assert abs(w[math_idx].item() - expected_math) < 1e-3, (
        f"Math weight {w[math_idx]:.4f} ≠ analytic {expected_math:.4f}"
    )
    for i, d in enumerate(DOMAINS):
        if i == math_idx:
            continue
        assert abs(w[i].item() - expected_other) < 1e-3, (
            f"Cortex {d} weight {w[i]:.4f} ≠ analytic {expected_other:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────
# 4. Differentiability & gradient flow
# ──────────────────────────────────────────────────────────────────────

def test_forward_is_differentiable(ensemble):
    """A backward pass on the ensemble output must deposit non-zero
    gradients in every sub-cortex AND in the router's learnable head."""
    ensemble.train()
    ids = torch.randint(0, VOCAB, (2, 16))
    out = ensemble(ids)
    out.sum().backward()
    for cortex in ensemble.sub_cortices:
        grads = [p.grad for p in cortex.parameters() if p.grad is not None]
        assert grads, f"No grad-tracked params reached cortex '{cortex.name}'"
        total = sum(g.abs().sum().item() for g in grads)
        assert total > 0.0, f"All gradients are zero in cortex '{cortex.name}'"
    # Router's learnable head must also receive gradient (it's the only
    # routing knob the LM loss can move; the lexical table is a buffer)
    learn_grad = ensemble.router.learnable_logits.weight.grad
    assert learn_grad is not None and learn_grad.abs().sum().item() > 0, \
        "Router learnable head received no gradient"


# ──────────────────────────────────────────────────────────────────────
# 5. BEMA biological damping
# ──────────────────────────────────────────────────────────────────────

def test_bema_damping_smooths_routing_across_batches(lexicon):
    """With ``bema_tau > 0`` the router's per-batch weights are blended
    with a running EMA over previous batches.  Concretely, after a
    math-priming batch is followed by a code batch, the math cortex's
    weight must NOT collapse to its instantaneous lexical value — the
    running average drags it up.  This is the 'biological inertia'
    constraint that prevents per-token routing oscillation.
    """
    torch.manual_seed(0)
    sub_cortices = [
        StubSubCortex(name=f"cortex_{d}", domain=d, vocab=VOCAB,
                      d_model=32, n_layers=1, n_heads=4, max_ctx=64)
        for d in DOMAINS
    ]
    router_damped = ThalamicRouter(
        vocab_size=VOCAB, d_model=32, domains=DOMAINS,
        lexicon=lexicon, lexical_bias_weight=2.0, bema_tau=0.9,
    )
    ens = MultiCortexEnsemble(
        sub_cortices=sub_cortices, router=router_damped, d_target=32,
    )
    ens.eval()
    math_ids = make_question("math", length=16, noise_ratio=0.0)
    code_ids = make_question("code", length=16, noise_ratio=0.0)
    with torch.no_grad():
        # Prime EMA with pure math
        ens(math_ids)
        # Then feed code — math weight should stay non-negligible
        ens(code_ids)
        w_after = ens.last_routing_weights.mean(dim=(0, 1))
    math_idx = DOMAINS.index("math")
    code_idx = DOMAINS.index("code")
    # With τ=0.9 the math weight should still be ≥ 40% of code's,
    # because the EMA still carries the priming batch's bias.
    assert w_after[math_idx] > 0.4 * w_after[code_idx], (
        f"BEMA damping ineffective — math weight collapsed: "
        f"{dict(zip(DOMAINS, w_after.tolist()))}"
    )


def test_bema_off_means_pure_instantaneous_routing(lexicon):
    """``bema_tau = 0.0`` is the identity case: no inertia, weights are
    exactly the per-batch softmax.  Two back-to-back forwards on the
    same input must give bit-identical routing weights."""
    torch.manual_seed(0)
    sub_cortices = [
        StubSubCortex(name=f"cortex_{d}", domain=d, vocab=VOCAB,
                      d_model=32, n_layers=1, n_heads=4, max_ctx=64)
        for d in DOMAINS
    ]
    router = ThalamicRouter(
        vocab_size=VOCAB, d_model=32, domains=DOMAINS,
        lexicon=lexicon, lexical_bias_weight=2.0, bema_tau=0.0,
    )
    ens = MultiCortexEnsemble(
        sub_cortices=sub_cortices, router=router, d_target=32)
    ens.eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        ens(ids); w1 = ens.last_routing_weights.clone()
        ens(ids); w2 = ens.last_routing_weights.clone()
    assert torch.allclose(w1, w2, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 6. Default factory
# ──────────────────────────────────────────────────────────────────────

def test_build_default_ensemble_runs():
    """The ``build_default_ensemble`` factory produces a runnable
    ensemble sized like the arch.neuro 30m_p4 cortices (smaller, for
    speed). Acts as the end-to-end smoke test."""
    ens = build_default_ensemble(
        vocab=VOCAB, d_model=64, n_layers=2, n_heads=4, max_ctx=64,
    )
    assert len(ens.sub_cortices) == 4
    ids = torch.randint(0, VOCAB, (1, 16))
    out = ens(ids)
    assert out.shape == (1, 16, 64)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
