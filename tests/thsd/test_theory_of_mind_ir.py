# -*- coding: utf-8 -*-
"""TDD: ``TheoryOfMindIR`` — formal IR for Theory-of-Mind semantics
in the THSD framework.

Background
----------
The codebase already has *fragments* of Theory of Mind:
``test_theory_of_mind_consistency`` (trust-score + NT-bias divergence
per entity) and ``HomeostatMonitor.observe_theory_of_mind`` (a
telemetry counter). These pin behaviour at the harness layer but do
**not** give the THSD framework a formal IR class to declare ToM
constraints in a ``.neuro`` file.

This module pins the contract for the IR class. The semantics:

  * A ToM stalk ``F(σ_ToM(agent_id))`` holds the model's *belief
    state about agent X*: a vector in ``ℝ^{d_belief}`` encoding what
    X is believed to know, want, and feel.
  * The agent set is finite and indexed; the ToM section over the
    indexed family is the model's full first-order belief model.
  * Second-order ToM (X believes Y believes Z) corresponds to a
    nested ToM IR with depth > 1.

The IR carries the data the parser will extract from a DSL block of
the form::

    theory_of_mind agent_model {
        d_belief: 64,
        max_agents: 16,
        belief_decay: 0.95,
        false_belief: { enabled: true, gating_threshold: 0.5 },
        order: 1
    }

Pin only the dataclass shape + validation rules here. Wiring into
the parser (``thsd_parser.py``) lands in P3; this commit ships the
formal IR + validation contract.
"""
from __future__ import annotations

import pytest

from neuroslm.dsl.thsd_ir import TheoryOfMindIR


class TestTheoryOfMindIRConstruction:
    """The IR dataclass holds the parsed ``theory_of_mind`` DSL block."""

    def test_defaults_are_sane(self):
        ir = TheoryOfMindIR()
        # The model should have *some* belief capacity by default,
        # not zero — a ToM block with no fields is still a request
        # for "give me a usable ToM layer".
        assert ir.d_belief > 0
        assert ir.max_agents > 0
        assert 0.0 < ir.belief_decay <= 1.0
        assert ir.order >= 1

    def test_custom_fields_round_trip(self):
        ir = TheoryOfMindIR(
            d_belief=128, max_agents=32,
            belief_decay=0.99, order=2,
            false_belief_enabled=True,
            false_belief_threshold=0.6,
        )
        assert ir.d_belief == 128
        assert ir.max_agents == 32
        assert ir.belief_decay == 0.99
        assert ir.order == 2
        assert ir.false_belief_enabled is True
        assert ir.false_belief_threshold == 0.6


class TestTheoryOfMindIRValidation:
    """Validation rules pin the contract that prevents silently-broken
    ToM configs from compiling."""

    def test_d_belief_must_be_positive(self):
        with pytest.raises(ValueError, match="d_belief"):
            TheoryOfMindIR(d_belief=0)
        with pytest.raises(ValueError, match="d_belief"):
            TheoryOfMindIR(d_belief=-1)

    def test_max_agents_must_be_positive(self):
        with pytest.raises(ValueError, match="max_agents"):
            TheoryOfMindIR(max_agents=0)

    def test_belief_decay_must_be_in_unit_interval(self):
        # 0 means "forget instantly" (degenerate); 1 means "never forget"
        # (allowed); >1 is unstable.
        with pytest.raises(ValueError, match="belief_decay"):
            TheoryOfMindIR(belief_decay=0.0)
        with pytest.raises(ValueError, match="belief_decay"):
            TheoryOfMindIR(belief_decay=1.5)
        # The endpoints are: open at 0, closed at 1.
        TheoryOfMindIR(belief_decay=1.0)   # must not raise

    def test_order_must_be_at_least_1(self):
        """Order 0 is "no ToM" (== disabled), which should be expressed
        by *not having* a ``theory_of_mind`` block, not by ``order=0``.
        """
        with pytest.raises(ValueError, match="order"):
            TheoryOfMindIR(order=0)
        # Order ≥ 1 is fine; deep nesting (≥ 4) the test suite does not
        # forbid but the linter will warn — that's a future concern.
        TheoryOfMindIR(order=1)
        TheoryOfMindIR(order=3)

    def test_false_belief_threshold_in_unit_interval(self):
        with pytest.raises(ValueError, match="false_belief_threshold"):
            TheoryOfMindIR(false_belief_enabled=True,
                           false_belief_threshold=-0.1)
        with pytest.raises(ValueError, match="false_belief_threshold"):
            TheoryOfMindIR(false_belief_enabled=True,
                           false_belief_threshold=1.5)


class TestTheoryOfMindIRStalkDim:
    """The IR computes the per-agent belief stalk dimension so the
    sheaf-stalk plumbing in :mod:`thsd.engine` can use it directly."""

    def test_stalk_dim_matches_d_belief_for_first_order(self):
        ir = TheoryOfMindIR(d_belief=64, order=1)
        assert ir.stalk_dim() == 64

    def test_stalk_dim_scales_with_order(self):
        """Second-order ToM (X believes Y believes Z) needs a belief
        cube — the stalk dim grows with order. We pin a simple
        polynomial law here so the codegen can size buffers."""
        ir1 = TheoryOfMindIR(d_belief=32, order=1)
        ir2 = TheoryOfMindIR(d_belief=32, order=2)
        # Strictly larger; the exact formula is contract-internal but
        # callers may inspect it for buffer sizing.
        assert ir2.stalk_dim() > ir1.stalk_dim()
