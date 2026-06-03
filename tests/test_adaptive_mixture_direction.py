# -*- coding: utf-8 -*-
"""TDD spec for the corrected AdaptiveMixtureController direction.

Background — 2026-06-03 trace
─────────────────────────────
In the 1000-step run with `chat_ratio` initialised at 0.60 and target
entropy 4.5, the controller drove chat_ratio to its max cap of 0.80
within the first 100 steps and kept it pinned there while wikitext
ppl barely improved (7700 → 6400) and train ppl dropped 4× (7700 → 1900,
gap_ratio 4.0 → 7.4).

Diagnosis: the original control law
    chat_ratio_{t+1} = chat_ratio_t * (H_t / H_target)^gamma
is BACKWARDS. When the model is uncertain on prose (H_t > H_target),
it needs MORE prose exposure, but this law sends it MORE chat.

Fix: default direction is now "balance":
    chat_ratio_{t+1} = chat_ratio_t * (H_target / H_t)^gamma
Old behaviour kept available behind `direction="amplify"` for
ablation.
"""
import pytest
import torch

from neuroslm.dsl.regularization import AdaptiveMixtureConfig
from neuroslm.regularizers import AdaptiveMixtureController


def _logits_with_entropy(target_entropy: float, vocab: int = 256,
                          shape=(4, 8)) -> torch.Tensor:
    """Construct logits whose softmax has approximately the requested entropy.

    We use a two-mass distribution: prob p on one token, (1-p)/(V-1) on
    the rest. Solve p numerically for the desired H, then convert back
    to logits.
    """
    # Binary search on p ∈ (1/V, 1) for the requested entropy.
    lo, hi = 1.0 / vocab + 1e-6, 1.0 - 1e-6
    for _ in range(60):
        p = 0.5 * (lo + hi)
        q = (1.0 - p) / (vocab - 1)
        H = -(p * torch.log(torch.tensor(p)).item()
              + (vocab - 1) * q * torch.log(torch.tensor(q)).item())
        if H > target_entropy:
            lo = p
        else:
            hi = p
    p = 0.5 * (lo + hi)
    q = (1.0 - p) / (vocab - 1)
    # Convert to logits (any monotone is fine; use log p).
    log_p = torch.log(torch.tensor(p))
    log_q = torch.log(torch.tensor(q))
    base = torch.full((*shape, vocab), log_q.item())
    base[..., 0] = log_p.item()
    return base


class TestAdaptiveMixtureBalanceDirection:
    """The corrected (default) `direction="balance"` rule."""

    def _make(self, initial=0.60, target_H=4.5, gamma=2.0,
              direction="balance", r_min=0.10, r_max=0.80):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=target_H,
            probe_interval=1, gamma=gamma,
            min_ratio=r_min, max_ratio=r_max,
            direction=direction)
        return AdaptiveMixtureController(cfg, initial_ratio=initial)

    def test_default_direction_is_balance(self):
        cfg = AdaptiveMixtureConfig(enabled=True)
        assert cfg.direction == "balance"

    def test_default_max_ratio_is_capped_at_half(self):
        """The 2026-06-03 run hit max_ratio=0.80 and stayed there. The
        new safe default caps chat at 0.50 so prose always gets at
        least half of the training tokens."""
        cfg = AdaptiveMixtureConfig(enabled=True)
        assert cfg.max_ratio == 0.50

    def test_high_prose_entropy_shrinks_chat(self):
        """If the held-out prose probe shows HIGH entropy (model is
        uncertain on prose), the controller must DECREASE chat ratio
        to expose the model to more prose."""
        m = self._make(initial=0.60, target_H=3.0, gamma=2.0)
        # Logits with entropy ≈ 5.0 > 3.0
        m.observe_logits(_logits_with_entropy(5.0))
        assert m.ratio() < 0.60, (
            f"high prose entropy should shrink chat (was 0.60, now {m.ratio():.4f})"
        )

    def test_low_prose_entropy_grows_chat(self):
        """If the model is fluent on prose (low entropy on the probe),
        there is spare capacity for more chat — ratio grows toward
        max_ratio."""
        m = self._make(initial=0.40, target_H=5.0, gamma=2.0)
        # Logits with entropy ≈ 2.0 < 5.0
        m.observe_logits(_logits_with_entropy(2.0))
        assert m.ratio() > 0.40, (
            f"low prose entropy should grow chat (was 0.40, now {m.ratio():.4f})"
        )

    def test_balance_inverts_amplify(self):
        """For the same observed H, balance and amplify should move
        the ratio in OPPOSITE directions."""
        m_bal = self._make(initial=0.60, target_H=3.0, direction="balance")
        m_amp = self._make(initial=0.60, target_H=3.0, direction="amplify")
        logits = _logits_with_entropy(5.0)
        m_bal.observe_logits(logits.clone())
        m_amp.observe_logits(logits.clone())
        # H_t > H_target: balance shrinks, amplify grows
        assert m_bal.ratio() < 0.60
        assert m_amp.ratio() > 0.60

    def test_ratio_clipped_to_max(self):
        """Even with explosive gain, the new default 0.50 max holds."""
        m = self._make(initial=0.45, target_H=10.0, gamma=4.0,
                       r_max=0.50)
        for _ in range(50):
            m.observe_logits(_logits_with_entropy(0.5))  # very low H
        # direction=balance + low H → controller wants to GROW chat
        # (model is overconfident on prose, can afford more chat)
        assert m.ratio() == pytest.approx(0.50)

    def test_ratio_clipped_to_min(self):
        m = self._make(initial=0.30, target_H=0.5, gamma=4.0,
                       r_min=0.10)
        for _ in range(50):
            m.observe_logits(_logits_with_entropy(5.0))  # very high H
        # direction=balance + high H → controller wants to SHRINK chat
        assert m.ratio() == pytest.approx(0.10)


class TestAdaptiveMixtureAmplifyDirectionLegacy:
    """The `direction="amplify"` setting reproduces the original buggy
    behaviour for back-compat ablations. Existing tests in
    tests/test_regularizers.py exercise this path."""

    def _make(self, initial=0.60, target_H=4.5, gamma=2.0):
        cfg = AdaptiveMixtureConfig(
            enabled=True, target_entropy=target_H,
            probe_interval=1, gamma=gamma,
            min_ratio=0.05, max_ratio=0.95,
            direction="amplify")
        return AdaptiveMixtureController(cfg, initial_ratio=initial)

    def test_amplify_high_entropy_grows_ratio(self):
        m = self._make(target_H=3.0)
        m.observe_logits(_logits_with_entropy(5.0))
        assert m.ratio() > 0.60

    def test_amplify_low_entropy_shrinks_ratio(self):
        m = self._make(target_H=5.0)
        m.observe_logits(_logits_with_entropy(2.0))
        assert m.ratio() < 0.60


class TestDSLParsing:
    def test_parser_accepts_direction_balance(self):
        from neuroslm.dsl.regularization import parse_regularization_block
        body = """adaptive_mixture: { enabled: true, direction: "balance" }"""
        cfg = parse_regularization_block(body)
        assert cfg.adaptive_mixture.direction == "balance"

    def test_parser_accepts_direction_amplify(self):
        from neuroslm.dsl.regularization import parse_regularization_block
        body = """adaptive_mixture: { enabled: true, direction: "amplify" }"""
        cfg = parse_regularization_block(body)
        assert cfg.adaptive_mixture.direction == "amplify"

    def test_parser_rejects_unknown_direction(self):
        from neuroslm.dsl.regularization import parse_regularization_block
        body = """adaptive_mixture: { enabled: true, direction: "invert" }"""
        with pytest.raises(ValueError, match="direction"):
            parse_regularization_block(body)

    def test_parser_max_ratio_default(self):
        from neuroslm.dsl.regularization import parse_regularization_block
        body = """adaptive_mixture: { enabled: true }"""
        cfg = parse_regularization_block(body)
        assert cfg.adaptive_mixture.max_ratio == 0.50
