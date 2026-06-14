"""Regression contract: VocabBridge abstain semantics must NOT poison
``cortex_loss_ema``.

Background
----------
The original implementation set ``_ABSTAIN_LOGIT = -1e4`` so that the
softmax of the bridged logits gave essentially zero probability for any
trunk id that the expert tokenizer couldn't produce as a single token.

For the **fusion** path (``logits_mixture``) this is harmless: the
mixture takes a softmax of each branch and combines them; an
``exp(-1e4)`` contribution to the mixture is just ~0.

For the **cortex_loss_ema** EMA — which the harness recomputes inside
``_cortex_fusion_aux_step`` via
``F.cross_entropy(cortex_logits.detach().float(), targets)`` — the
exact same logits are used as a *standalone* prediction. When the
target token happens to be one of the unmapped trunk ids, the CE for
that position is roughly ``-log(softmax(-1e4)) ≈ |_ABSTAIN_LOGIT|``,
i.e. ~10000 nats per such position.

Concretely, with ~10 % of the trunk vocab unmapped on a bridge like
``gpt2 → microsoft/CodeGPT-small-py``, ``cortex_loss_ema`` was
observed at ~491 nats in production (deploy 40923107) while the
trunk's own ``lm_loss_ema`` was ~10 nats. The harness's
``_effective_alpha`` then concluded "cortex is catastrophic" and
crushed ``α_eff → 0``, fully disabling the pretrained ensemble — the
opposite of what the multi-cortex feature is supposed to do.

This file pins the contract: bridged logits, viewed as a standalone
prediction, must yield a cross-entropy in the same order of magnitude
as the uniform baseline (``ln V``), even when ~10 % of trunk ids are
unmapped. Specifically: with a uniform (zero) expert input, the
bridged cross-entropy on random targets must be ≲ 2 × ln V — not ≫ V.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from neuroslm import experts as _experts_module
from neuroslm.experts import VocabBridge


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _make_bridge(v_trunk: int, mapping: dict[int, int]) -> VocabBridge:
    """Directly construct a VocabBridge with a known mapping.

    `mapping` is ``{trunk_id: expert_id}``. Any trunk id not in
    `mapping` is treated as unmapped (``-1``). Maximum expert id + 1
    becomes ``vocab_size_expert``.
    """
    table = torch.full((v_trunk,), -1, dtype=torch.long)
    for tid, eid in mapping.items():
        table[tid] = int(eid)
    n_mapped = sum(1 for v in table.tolist() if v >= 0)
    v_expert = (max(mapping.values()) + 1) if mapping else 1
    return VocabBridge(
        trunk_to_expert=table,
        is_identity=False,
        coverage=n_mapped / max(1, v_trunk),
        vocab_size_trunk=v_trunk,
        vocab_size_expert=int(v_expert),
    )


# ─────────────────────────────────────────────────────────────────
# Contract A — softmax of bridged logits stays well-behaved
# ─────────────────────────────────────────────────────────────────


def test_bridged_softmax_does_not_collapse_to_zero_for_mapped_target():
    """Mapped-target softmax probability must dominate the uniform mass.

    Sanity: if the expert outputs a clear peak at trunk id 3 (one of
    the mapped ids), the bridged-logits softmax must still place
    *substantially* more probability on id 3 than on a uniform random
    pick. The bug shape would be: the abstain logic dominates the
    partition function and crushes the peak.

    Contract: ``p(peak) > 10 × uniform`` — i.e. the peak carries at
    least an order of magnitude more mass than a uniform tile. With
    the per-position abstain (``max_mapped - ln V``) the contribution
    of N_unmapped abstain slots scales as ``N_unmapped / V_trunk``,
    so for small mapped-fraction setups the peak isn't *absolute*
    but it must still be a clear signal.
    """
    v_trunk = 100
    # Map 10 trunk ids → expert ids 0..9; 90 % unmapped
    mapping = {i: i for i in range(10)}
    bridge = _make_bridge(v_trunk, mapping)
    # Expert outputs a peak at expert id 3
    expert_logits = torch.zeros(1, 1, 10)
    expert_logits[0, 0, 3] = 5.0
    bridged = bridge.apply(expert_logits)  # (1, 1, 100)
    probs = F.softmax(bridged, dim=-1)
    uniform = 1.0 / v_trunk
    p_peak = probs[0, 0, 3].item()
    assert p_peak > 10.0 * uniform, (
        f"Mapped peak collapsed: p={p_peak:.3f}, uniform={uniform:.3f}"
    )


def test_bridged_softmax_unmapped_target_ce_is_bounded_by_uniform_baseline():
    """CE on an unmapped target must be O(ln V_trunk), not O(|abstain|).

    The principled semantic of "expert abstains" is "I have no
    information → fall back to uniform baseline". After the fix, the
    per-position abstain value is ``max(mapped_logits) - ln(V_trunk)``,
    so an unmapped slot's softmax contribution is roughly
    ``1/V_trunk × p(peak)``. The CE for an unmapped target is then
    bounded by ``ln(V_trunk) + small`` — never the catastrophic
    ``~10000 nats`` of the original ``-1e4`` constant.

    Contract: ``CE_unmapped_target < 2 · ln(V_trunk)`` — comfortably
    in the uniform baseline regime, NOT in the abstain-magnitude
    regime that crashed deploy 40923107.
    """
    v_trunk = 100
    mapping = {i: i for i in range(10)}
    bridge = _make_bridge(v_trunk, mapping)
    # Uniform expert logits (no preference)
    expert_logits = torch.zeros(1, 1, 10)
    bridged = bridge.apply(expert_logits)
    # Target is an unmapped trunk id
    target = torch.tensor([50])
    ce = F.cross_entropy(bridged.reshape(1, v_trunk), target).item()
    ln_v = math.log(v_trunk)
    assert ce < 2.0 * ln_v, (
        f"CE on unmapped target = {ce:.3f} nats — must be in the "
        f"uniform baseline regime ≲ {2.0 * ln_v:.3f}. The fix to "
        "_ABSTAIN_LOGIT failed to prevent CE blow-up on unmapped "
        "targets, the exact regression that crashed deploy 40923107."
    )


# ─────────────────────────────────────────────────────────────────
# Contract B — cross-entropy on random targets stays sane (THE BUG)
# ─────────────────────────────────────────────────────────────────


def test_bridged_cross_entropy_is_bounded_when_10pct_targets_unmapped():
    """Standalone CE of bridged logits ≲ 2·ln V — not orders larger.

    This is the deploy-blocker. ``_cortex_fusion_aux_step`` in
    harness.py computes
        ``ce_cx = F.cross_entropy(cortex_logits.float(), targets)``
    against the actual training targets. If even 10 % of those targets
    fall on unmapped trunk ids and abstain = -1e4, then
        ``ce_cx ≈ 0.9·CE_mapped + 0.1·|abstain| ≈ 1000 nats``,
    which is ~50× the uniform baseline ``ln V ≈ ln 50257 ≈ 10.82``.
    The fix lowers the abstain logit so the partition function gives
    each unmapped slot roughly ``1/V`` mass, keeping CE on unmapped
    targets ≲ ``ln V``.
    """
    torch.manual_seed(0)
    v_trunk = 200
    # Map 180 trunk ids; leave 20 (10 %) unmapped
    mapping = {i: i for i in range(180)}
    bridge = _make_bridge(v_trunk, mapping)
    # Realistic expert logits: small random spread around 0
    expert_logits = torch.randn(4, 16, 180) * 0.5
    bridged = bridge.apply(expert_logits)  # (4, 16, 200)
    # Random targets — by construction ~10 % land on unmapped ids
    targets = torch.randint(0, v_trunk, (4 * 16,))
    flat = bridged.reshape(-1, v_trunk)
    ce = F.cross_entropy(flat, targets).item()
    uniform_baseline = math.log(v_trunk)  # ≈ 5.30
    assert math.isfinite(ce), f"CE not finite: {ce}"
    assert ce < 2.0 * uniform_baseline, (
        f"Bridged CE = {ce:.2f} nats but uniform baseline is "
        f"{uniform_baseline:.2f}. With _ABSTAIN_LOGIT = -1e4 the bug "
        "produces ~1000 nats — this contract pins the fix."
    )


def test_bridged_cross_entropy_on_only_mapped_targets_matches_expert_quality():
    """When the target is always a mapped id, bridged CE should match
    the expert's native CE — the bridge must add at most a small
    constant inflation from the abstain mass in the partition function.

    This is the "expert quality preserved" contract: the fix to
    ``_ABSTAIN_LOGIT`` should not unduly penalize the bridged
    predictions on the tokens the expert actually knows about.
    """
    torch.manual_seed(1)
    v_trunk = 200
    mapping = {i: i for i in range(180)}
    bridge = _make_bridge(v_trunk, mapping)
    # Strong expert prediction: peaks at the correct token id
    expert_logits = torch.full((4, 16, 180), -2.0)
    targets = torch.randint(0, 180, (4, 16))  # only mapped targets
    # Make the expert "right" on every target
    for b in range(4):
        for t in range(16):
            expert_logits[b, t, targets[b, t]] = 5.0
    # Expert's own CE (in its own vocab)
    expert_flat = expert_logits.reshape(-1, 180)
    ce_expert = F.cross_entropy(expert_flat, targets.reshape(-1)).item()
    # Bridged version
    bridged = bridge.apply(expert_logits)  # (4, 16, 200)
    ce_bridged = F.cross_entropy(
        bridged.reshape(-1, v_trunk), targets.reshape(-1)
    ).item()
    # Bridged CE must not be more than ~1 nat worse than expert CE.
    # The unmapped slots add a small extra term to the partition
    # function; with the fix this is a tiny constant.
    assert ce_bridged < ce_expert + 1.0, (
        f"Bridge inflated CE catastrophically: expert={ce_expert:.3f}, "
        f"bridged={ce_bridged:.3f} (delta {ce_bridged - ce_expert:.3f})."
    )


# ─────────────────────────────────────────────────────────────────
# Contract C — abstain is robust to expert baseline shift
# ─────────────────────────────────────────────────────────────────


def test_bridged_cross_entropy_invariant_to_expert_baseline_shift():
    """Same expert distribution, different additive baseline → same CE.

    The bug: gpt2's pretrained head sits around -65 baseline (logits
    in ``[-122, -30]``). A global constant abstain (e.g. ``-12``) was
    ABOVE these mapped logits and dominated the softmax, blowing CE
    up to ~17 nats on plain English. The per-position-relative fix
    makes abstain proportional to ``max(mapped)``, so any additive
    constant shift in the expert's logits leaves the bridged CE
    invariant. This contract pins that invariant.
    """
    torch.manual_seed(2)
    v_trunk = 200
    mapping = {i: i for i in range(180)}
    bridge = _make_bridge(v_trunk, mapping)
    # Expert outputs at baseline 0
    base = torch.randn(2, 8, 180)
    targets = torch.randint(0, v_trunk, (16,))
    ce_at_0 = F.cross_entropy(
        bridge.apply(base).reshape(-1, v_trunk), targets,
    ).item()
    # Same distribution, baseline -100 (gpt2-like)
    shifted = base - 100.0
    ce_at_neg100 = F.cross_entropy(
        bridge.apply(shifted).reshape(-1, v_trunk), targets,
    ).item()
    # Same distribution, baseline +50 (random-init head)
    shifted_pos = base + 50.0
    ce_at_pos50 = F.cross_entropy(
        bridge.apply(shifted_pos).reshape(-1, v_trunk), targets,
    ).item()
    # All three CEs must be within 0.1 nat of each other.
    assert abs(ce_at_0 - ce_at_neg100) < 0.1, (
        f"CE shifted by baseline: ce@0={ce_at_0:.3f}, "
        f"ce@-100={ce_at_neg100:.3f} — abstain logic isn't "
        "baseline-invariant!"
    )
    assert abs(ce_at_0 - ce_at_pos50) < 0.1, (
        f"CE shifted by baseline: ce@0={ce_at_0:.3f}, "
        f"ce@+50={ce_at_pos50:.3f} — abstain logic isn't "
        "baseline-invariant!"
    )
