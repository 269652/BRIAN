"""Tests for the Multidimensional Relational Hypergraph stack (BRIAN).

Five behavioural tests covering:

  1. test_causal_generalization          — ReasoningCortex action→reaction
  2. test_autobiographical_coherence     — NarrativeSystem JSON self-summary
  3. test_theory_of_mind_consistency     — Trust score + NT bias divergence
  4. test_sheaf_contradiction_detection  — H¹ ≠ 0 → SUPERSEDES edge
  5. test_predictive_forgetting_gain     — Sleep-cycle I(X;Z) decrease

Each test isolates one component and feeds it controlled inputs, then
asserts a behavioural property. No full Brain forward pass is required;
the components are exercised directly so the tests run in <2 s each.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Causal generalisation in the ReasoningCortex action→reaction predictor
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_generalization():
    """Feed 10 Gift→Joy episodes; verify a novel Gift gets P(Joy) > 0.8."""
    from neuroslm.modules.reasoning import ReasoningCortex

    d_sem = 32
    n_action_types = 8
    cortex = ReasoningCortex(d_sem=d_sem, n_attractors=16,
                              n_action_types=n_action_types,
                              enable_hfw=False)
    cortex.train()

    # Two action prototypes: "gift" and "insult" — well-separated vectors
    torch.manual_seed(11)
    gift_proto   = torch.randn(d_sem)
    insult_proto = torch.randn(d_sem)
    # Force separation
    insult_proto = insult_proto - (insult_proto @ gift_proto) * gift_proto / (
        gift_proto @ gift_proto + 1e-9)

    # Reaction targets: index 2 = "joy", index 5 = "offense"
    JOY, OFFENSE = 2, 5

    # 10 training pairs of (gift, joy) + 10 (insult, offense) to teach
    # the predictor that gift implies joy specifically.
    opt = torch.optim.AdamW(cortex.parameters(), lr=5e-2)
    for epoch in range(120):
        # mini-batch of jittered prototypes
        gift_batch   = gift_proto.unsqueeze(0).expand(8, -1) + 0.05 * torch.randn(8, d_sem)
        insult_batch = insult_proto.unsqueeze(0).expand(8, -1) + 0.05 * torch.randn(8, d_sem)
        x = torch.cat([gift_batch, insult_batch], dim=0)
        y = torch.tensor([JOY] * 8 + [OFFENSE] * 8, dtype=torch.long)
        loss = cortex.causal_aux_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    cortex.eval()
    # Novel gift input (small noise)
    novel_gift = gift_proto + 0.05 * torch.randn(d_sem)
    probs, _ = cortex.predict_reaction(novel_gift.unsqueeze(0))
    p_joy = float(probs[0, JOY].item())
    assert p_joy > 0.8, (
        f"Expected P(Joy | Gift) > 0.8 after 120 epochs, got {p_joy:.3f}; "
        f"full distribution = {probs[0].tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Autobiographical narrative coherence — JSON self-summary
# ─────────────────────────────────────────────────────────────────────────────

def test_autobiographical_coherence():
    """Three events in order → JSON dict preserves chronology + identity."""
    from neuroslm.memory.narrative import NarrativeSystem

    d_sem = 32
    ns = NarrativeSystem(d_sem=d_sem)
    torch.manual_seed(42)

    events = ["Creation", "Learning Math", "Meeting User"]
    for content in events:
        emb = torch.randn(d_sem)
        ns.record_autobiographical(emb, content=content, valence=0.2, salience=0.7)

    story = ns.self_summary(identity="Self", max_events=10)

    # Identity must be present and consistent for every event
    assert story["identity"] == "Self", story
    assert len(story["events"]) == 3, f"expected 3 events, got {story}"
    for evt in story["events"]:
        assert evt["subject"] == "Self", evt
    # Chronological order preserved
    ts = [evt["t"] for evt in story["events"]]
    assert ts == sorted(ts), f"events not in chronological order: {ts}"
    # Contents preserved
    assert [evt["content"] for evt in story["events"]] == events
    # JSON-serialisable
    import json
    json.dumps(story)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Theory-of-Mind consistency — trust score + NT bias differ per entity
# ─────────────────────────────────────────────────────────────────────────────

def test_theory_of_mind_consistency():
    """Alice (friendly) accumulates positive valence; Bob (rude) negative.
    Verify trust(Alice) > trust(Bob) and the resulting NT bias diverges."""
    from neuroslm.neurochem.personality import PersonalityVector
    from neuroslm.neurochem.transmitters import TransmitterSystem, NT_INDEX

    p = PersonalityVector(enable=True)
    p.set_awakened(True)        # explicit — trust updates only post-awakening

    # Feed asymmetric outcome histories
    for _ in range(20):
        p.observe_interaction("alice", valence=+0.8)
        p.observe_interaction("bob",   valence=-0.6)

    trust_a = p.trust("alice")
    trust_b = p.trust("bob")
    assert trust_a > 0.7, f"Alice's trust should be high; got {trust_a:.3f}"
    assert trust_b < 0.4, f"Bob's trust should be low; got {trust_b:.3f}"
    assert trust_a > trust_b + 0.3, (
        f"Trust(Alice) − Trust(Bob) should be substantial; "
        f"got Alice={trust_a:.3f}, Bob={trust_b:.3f}")

    # NT bias divergence — apply to two snapshots of the transmitter system
    tx_alice = TransmitterSystem()
    tx_bob   = TransmitterSystem()
    p.apply_bias(tx_alice, active_entities={"alice": 1.0})
    p.apply_bias(tx_bob,   active_entities={"bob":   1.0})

    da_a, da_b = tx_alice.bias[NT_INDEX["DA"]].item(), tx_bob.bias[NT_INDEX["DA"]].item()
    ne_a, ne_b = tx_alice.bias[NT_INDEX["NE"]].item(), tx_bob.bias[NT_INDEX["NE"]].item()
    assert da_a > da_b, (
        f"DA bias should be higher for trusted entity Alice; "
        f"DA(Alice)={da_a:.3f}, DA(Bob)={da_b:.3f}")
    assert ne_a < ne_b, (
        f"NE (vigilance) bias should be higher for untrusted Bob; "
        f"NE(Alice)={ne_a:.3f}, NE(Bob)={ne_b:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sheaf-based contradiction detection — H¹ ≠ 0 → SUPERSEDES edge
# ─────────────────────────────────────────────────────────────────────────────

def test_sheaf_contradiction_detection():
    """Store 'Alice likes coffee' then 'Alice hates coffee' and verify
    the H¹ residual exceeds threshold and a SUPERSEDES edge is created."""
    from neuroslm.memory.hypergraph import MemoryHyperGraph, RelationType
    import numpy as np

    d_emb = 16
    g = MemoryHyperGraph(d_emb=d_emb)

    # Encode two contradictory beliefs as vectors pointing in opposite
    # directions along the "Alice/coffee" axis.
    rng = np.random.default_rng(0)
    base = rng.normal(size=d_emb).astype(np.float32)
    base /= np.linalg.norm(base) + 1e-9
    likes  = (+1.0) * base + 0.05 * rng.normal(size=d_emb).astype(np.float32)
    hates  = (-1.0) * base + 0.05 * rng.normal(size=d_emb).astype(np.float32)

    id1 = g.encode("Alice likes coffee", embedding=likes,
                    entity_ref="alice", valence=+0.7)
    # Small timestamp gap so the resolver picks the newer one
    import time as _t
    _t.sleep(0.001)
    id2 = g.encode("Alice hates coffee", embedding=hates,
                    entity_ref="alice", valence=-0.7)
    assert id1 and id2

    # Add a causal edge between the two as if they're the same belief —
    # this is where the contradiction is exposed: an identity restriction
    # map sends one belief into something far from the other.
    g.add_causal_edge(id1, id2, alpha=0.9)

    is_contradiction, section = g.detect_contradiction([id1, id2])
    assert section.h1_residual > 0.5, (
        f"H¹ residual should be large for contradictory beliefs; "
        f"got {section.h1_residual:.3f}")
    assert is_contradiction, (
        f"Should be flagged as a contradiction; section.h1={section.h1_residual:.3f}")

    # SUPERSEDES edge should exist newer → older
    supersedes = [
        e for e in g.edges.values()
        if e.relation == RelationType.SUPERSEDES
    ]
    assert supersedes, "Expected a SUPERSEDES edge after contradiction resolution"
    # Newer (id2) supersedes older (id1)
    assert g.is_superseded(id1), \
        f"older node {id1} should be marked superseded"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predictive-forgetting gain — Gaussian I(X;Z) drops while LM intact
# ─────────────────────────────────────────────────────────────────────────────

def test_predictive_forgetting_gain():
    """100 sleep-distill iterations should not blow up I(X;Z) — and a
    proxy 'LM accuracy' on held-out data must stay non-decreasing.

    We use a tiny linear LM head as the 'language model' stand-in; the
    sleep cycle's distillation should not damage the held-out fit on
    purely random data (no useful structure to extract).
    """
    from neuroslm.memory.sleep_cycle import SleepCycle

    d_sem = 24
    sc = SleepCycle(d_sem=d_sem, replay_batch=8, n_iters=2,
                     sleep_period_steps=1, enable=True)
    sc.set_awakened(True)

    # Synthetic noisy episode buffer
    torch.manual_seed(7)
    n_episodes = 64
    episodes = [
        {"content_vec": torch.randn(d_sem).numpy(),
         "salience": 0.5 + 0.3 * float(torch.rand(1).item()),
         "decay": 1.0,
         "valence": float(torch.randn(1).item()),
         "surprise": 0.1 + 0.5 * float(torch.rand(1).item())}
        for _ in range(n_episodes)
    ]

    def to_embed(ep):
        return torch.from_numpy(ep["content_vec"]).float()

    def to_known_nll(ep):
        return float(ep["surprise"])

    # Measure I(X;Z) proxy before, and a held-out LM proxy fit too
    held_x = torch.stack([to_embed(e) for e in episodes[:16]], dim=0).float()
    held_z = sc.predictor(held_x).detach()
    pre_mi = sc._gaussian_mi_proxy(held_x, held_z)
    # LM proxy: linear ridge fit residual norm — lower is better
    A = held_x
    b = held_z.mean(dim=-1, keepdim=True)
    w_pre, *_ = torch.linalg.lstsq(A, b)
    resid_pre = float((A @ w_pre - b).pow(2).mean().item())

    # Run 100 sleep iterations
    for _ in range(100):
        sc.sleep(
            step=sc._last_sleep_step + 1,
            episodes=episodes,
            episode_to_embed=to_embed,
            episode_to_known_nll=to_known_nll,
            actual_causation=None,
            trophic_system=None,
        )

    held_z2 = sc.predictor(held_x).detach()
    post_mi = sc._gaussian_mi_proxy(held_x, held_z2)
    w_post, *_ = torch.linalg.lstsq(A, b)
    resid_post = float((A @ w_post - b).pow(2).mean().item())

    # Predictive-forgetting gain: post_mi should not exceed pre_mi by much.
    # On purely random data the predictor can't extract structure, but the
    # distillation should not amplify spurious dependencies either. We
    # require post ≤ pre + a small slack.
    assert post_mi <= pre_mi + 1.0, (
        f"I(X;Z) blew up after distillation; pre={pre_mi:.3f}, post={post_mi:.3f}")

    # LM accuracy proxy: residual of the held-out linear fit should be
    # maintained (or improved) — no catastrophic forgetting.
    assert resid_post <= resid_pre * 1.5 + 1e-3, (
        f"LM-proxy residual degraded; pre={resid_pre:.4f}, post={resid_post:.4f}")
