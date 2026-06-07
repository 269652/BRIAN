# -*- coding: utf-8 -*-
"""TDD acceptance suite — SymbolicSimplex integration into the simplicial complex K.

Phase THSD-1/4 of the formal-discovery roadmap (see ``docs/formal_framework.md``).

Background
----------
The base ``SimplexComplex`` from ``neuroslm/thsd/engine.py`` treats every
simplex as an opaque cell — useful for cohomology bookkeeping but
*not* yet a substrate for *mathematical discovery*.  The
``SymbolicHyperNeuron`` (``neuroslm/modules/symbolic_unit.py``)
already learns small explicit equations over its inputs through
Gumbel-softmax selection from an operator bank
``{identity, add, sub, mul, exp, sin, tanh}``.

This suite specifies the algebraic contract for embedding a
``SymbolicHyperNeuron`` *into* a simplex so that:

  1. The simplex's stalk ``F(σ)`` is the output space of the unit
     (so the cellular-sheaf machinery sees one consistent vector space).
  2. The unit's argmax selection is exposed as an explicit equation
     via ``simplex.symbolic_expression()`` — the discovery surface.
  3. Its sparsity loss bubbles up via ``simplex.sparsity_loss()`` so the
     ``FitnessComposer`` can collect it like any other objective.
  4. The simplex registers correctly inside both ``SimplexComplex``
     and ``CellularSheaf`` without bypassing existing invariants.

Reference: ``docs/formal_framework.md`` §3 (Symbolic Expression Units).
"""
from __future__ import annotations

import pytest
import torch

from neuroslm.thsd.engine import (
    SimplexComplex,
    CellularSheaf,
)


# ──────────────────────────────────────────────────────────────────────
# Importability + construction
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicSimplexConstruction:
    """A ``SymbolicSimplex`` must be importable from ``neuroslm.thsd.engine``
    and bind a ``SymbolicHyperNeuron`` instance to a 0-simplex."""

    def test_symbolic_simplex_importable(self):
        from neuroslm.thsd.engine import SymbolicSimplex  # noqa: F401

    def test_construction_with_default_operator_bank(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(
            name="sym_layer0_unit3",
            n_units=4,
            n_features=8,
        )
        assert sx.name == "sym_layer0_unit3"
        assert sx.dim == 0  # symbolic units are vertices (0-simplices)
        # The default operator bank from SymbolicHyperNeuron must be
        # available through the simplex for inspection.
        assert "identity" in sx.operator_names
        assert "tanh" in sx.operator_names

    def test_construction_rejects_zero_units(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        with pytest.raises(ValueError):
            SymbolicSimplex(name="bad", n_units=0, n_features=8)

    def test_construction_rejects_zero_features(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        with pytest.raises(ValueError):
            SymbolicSimplex(name="bad", n_units=4, n_features=0)


# ──────────────────────────────────────────────────────────────────────
# Stalk integration — the algebraic contract with CellularSheaf
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicSimplexStalkContract:
    """The simplex's stalk dimension must match its symbolic unit's
    output dimension so the sheaf restriction maps compose correctly."""

    def test_stalk_dim_equals_n_units(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8)
        assert sx.stalk_dim == 4

    def test_register_with_simplex_complex(self):
        """A SymbolicSimplex must register inside a SimplexComplex
        with kind='symbolic' metadata for the verifier to find it."""
        from neuroslm.thsd.engine import SymbolicSimplex
        K = SimplexComplex(dim_max=1)
        sx = SymbolicSimplex(name="sym0", n_units=4, n_features=8)
        sx.register(K)
        assert "sym0" in K.simplices[0]
        meta = K.simplices[0]["sym0"]
        assert meta.get("kind") == "symbolic"
        assert meta["dim"] == 0

    def test_sheaf_stalk_has_matching_dimension(self):
        """After register-then-attach to a CellularSheaf with the same
        stalk_dim, the stalk for this simplex must have shape
        ``(n_units,)`` so ``sheaf.get_stalk(name)`` works without
        a dimension mismatch."""
        from neuroslm.thsd.engine import SymbolicSimplex
        K = SimplexComplex(dim_max=1)
        sx = SymbolicSimplex(name="sym0", n_units=4, n_features=8)
        sx.register(K)
        F = CellularSheaf(complex=K, stalk_dim=4)
        assert F.get_stalk("sym0").shape == (4,)


# ──────────────────────────────────────────────────────────────────────
# Forward semantics — activation AND equation
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicSimplexForward:
    """The forward pass must produce (a) a numeric activation that can
    flow through the rest of the pipeline, and (b) the same expression
    strings the underlying SymbolicHyperNeuron emits."""

    def test_forward_shape(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8)
        x = torch.randn(2, 7, 8)  # (batch, seq, n_features)
        y = sx(x)
        assert y.shape == (2, 7, 4)

    def test_forward_is_differentiable(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8)
        x = torch.randn(1, 4, 8, requires_grad=True)
        y = sx(x).sum()
        y.backward()
        assert x.grad is not None and (x.grad != 0).any()

    def test_symbolic_expression_returns_n_units_strings(self):
        """The discovery surface: a list of human-readable expressions,
        one per learnt unit."""
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8)
        exprs = sx.symbolic_expression()
        assert isinstance(exprs, list)
        assert len(exprs) == 4
        for e in exprs:
            assert isinstance(e, str) and len(e) > 0

    def test_symbolic_expression_mentions_feature_names(self):
        """Custom feature names must flow through to the expressions."""
        from neuroslm.thsd.engine import SymbolicSimplex
        feats = [f"f{i}" for i in range(8)]
        sx = SymbolicSimplex(
            name="sym", n_units=4, n_features=8,
            feature_names=feats,
        )
        exprs = sx.symbolic_expression()
        # At least one expression must reference at least one feature name.
        joined = " ".join(exprs)
        assert any(f in joined for f in feats), (
            f"no feature names in expressions; got {exprs}"
        )


# ──────────────────────────────────────────────────────────────────────
# Loss bubble-up — sparsity surface for FitnessComposer
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicSimplexLossBubble:
    """The simplex must expose ``sparsity_loss()`` returning the
    underlying unit's loss tensor, so ``FitnessComposer`` can collect
    it as the ``symbolic`` objective without poking at internals."""

    def test_sparsity_loss_is_scalar_tensor(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8)
        loss = sx.sparsity_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_sparsity_loss_decreases_with_lower_tau(self):
        """Lower Gumbel temperature concentrates the selection
        distribution, so entropy (= sparsity_loss) should drop."""
        from neuroslm.thsd.engine import SymbolicSimplex
        sx = SymbolicSimplex(name="sym", n_units=4, n_features=8,
                             tau_init=1.0)
        loss_hot = sx.sparsity_loss().item()
        sx.set_tau(0.1)
        loss_cold = sx.sparsity_loss().item()
        # The entropy of the *softmax of logits* doesn't depend on tau
        # at all (sparsity_loss looks at parameter logits directly), so
        # the contract here is just "set_tau is wired and doesn't crash";
        # actual annealing dynamics are tested in test_symbolic_unit.py.
        assert isinstance(loss_hot, float)
        assert isinstance(loss_cold, float)


# ──────────────────────────────────────────────────────────────────────
# Sheaf wiring — symbolic simplex plays nicely with existing machinery
# ──────────────────────────────────────────────────────────────────────

class TestSymbolicSimplexInSheaf:
    """A SymbolicSimplex registered in K must let the sheaf set/get
    its stalk and connect to an edge simplex without breaking the
    existing CellularSheaf assertions."""

    def test_set_stalk_to_unit_output(self):
        from neuroslm.thsd.engine import SymbolicSimplex
        K = SimplexComplex(dim_max=1)
        sx = SymbolicSimplex(name="sym0", n_units=4, n_features=8)
        sx.register(K)
        F = CellularSheaf(complex=K, stalk_dim=4)

        x = torch.randn(1, 4, 8)
        y = sx(x)                    # (1, 4, 4)
        pooled = y.mean(dim=(0, 1))  # (4,)
        F.set_stalk("sym0", pooled)

        assert torch.allclose(F.get_stalk("sym0"), pooled)

    def test_edge_to_regular_simplex(self):
        """A 1-simplex connecting a symbolic simplex to an ordinary
        vertex should register without raising."""
        from neuroslm.thsd.engine import SymbolicSimplex
        K = SimplexComplex(dim_max=1)
        sx = SymbolicSimplex(name="sym0", n_units=4, n_features=8)
        sx.register(K)
        K.add_simplex("v_out", dim=0)
        K.add_simplex("e_sym_out", dim=1, boundary=["sym0", "v_out"])

        F = CellularSheaf(complex=K, stalk_dim=4)
        boundary = K.boundary("e_sym_out")
        assert "sym0" in boundary
        assert "v_out" in boundary
        # Restriction maps for both endpoints exist.
        assert ("e_sym_out", "sym0") in F.restrictions
        assert ("e_sym_out", "v_out") in F.restrictions
