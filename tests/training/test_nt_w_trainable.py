# -*- coding: utf-8 -*-
"""TDD contract — Item 6: trainable W coupling matrix in DrivenNTSystem.

Goal
====
Right now the 7×5 NT coupling matrix ``W`` is hand-coded floats
(``_DEFAULT_W`` in ``driven_nt.py``). That captures the biology of
"DA loves +surprise, NE loves +gnorm, GABA loves +ignition" but the
exact magnitudes are guessed. With Items 2/3/4 the NT levels now
*drive trainable parameters of the system* (router temperature,
distillation λ, lateral inhibition κ), so there's a real gradient
signal that could refine W if we expose it as an ``nn.Parameter``.

Reparam contract (straight-through, OU stays in float)
======================================================
The OU state is stochastic and step-coupled across thousands of
training steps — backprop-through-time through the whole trajectory
would be a memory/compute disaster and isn't what we want anyway
(``W`` should learn from *instantaneous* sensitivity, not from a
chain through history).

So the design is hybrid:

  * OU state ``y, level`` evolves in **float** every step using
    ``W.detach()`` — no grad flows through history, identical numerics
    to today's float-only code path.
  * A NEW method ``predict_nt_tensor(drivers) -> Tensor(7,)`` runs
    the *same* readout formula but with ``W`` as a live ``Parameter``:

        ŷ = μ + W · z(drivers)             # logit space
        n̂t = σ(ŷ)                          # readout in [0,1]

    The returned tensor has ``requires_grad = True`` (transitively
    via the Parameter) when ``trainable_W=True``, so any downstream
    differentiable use of this tensor lets the optimizer move ``W``.
  * Identity-when-disabled: ``trainable_W=False`` (the default) gives
    a system that is **bit-identical** to the legacy DrivenNTSystem
    — no Parameter, no extra optimizer state, no Tensor output path.

Pinned contracts
================
A. ``DrivenNTSystem`` is now an ``nn.Module``. Construction without
   ``trainable_W`` (or with ``trainable_W=False``) behaves exactly as
   before; ``list(self.parameters())`` is empty.
B. With ``trainable_W=True``:
     - ``self.W_param`` is an ``nn.Parameter`` of shape ``(7, 5)``.
     - ``list(self.parameters())`` contains exactly that parameter.
     - The initial values match ``_DEFAULT_W`` row by row, channel
       order ``_CHANNELS``, driver order ``_DRIVERS``.
C. ``step_full(...)`` continues to update float state via the float
   ``W`` even when ``trainable_W=True`` — and the resulting
   ``levels()`` is numerically identical to the trainable-False path
   when given the same driver sequence. (Detached → no grad leak.)
D. ``predict_nt_tensor(drivers)`` returns a ``(7,)`` tensor:
     - With ``trainable_W=True``: ``requires_grad=True``; the gradient
       w.r.t. ``W_param`` from ``predict_nt_tensor(...).sum().backward()``
       is finite and non-trivially non-zero for at least one driver
       that was actually supplied.
     - With ``trainable_W=False``: returns a detached tensor (no
       grad), so callers can be polymorphic.
E. ``predict_nt_tensor`` agrees with ``levels()`` numerically when
   called with the *same* drivers immediately after a ``step_full``
   (within tight tol). This pins that the two readout paths share the
   same μ/W math — not two divergent implementations.
F. The DSL parser reads ``nt_w_trainable: true`` at the top level of
   the training-config block.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# A) Default (back-compat) path — nn.Module, no parameters
# ──────────────────────────────────────────────────────────────────────


class TestDefaultBackCompat:

    def test_driven_nt_is_an_nn_module(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem()
        assert isinstance(nt, nn.Module)

    def test_default_has_no_parameters(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem()
        params = list(nt.parameters())
        assert params == [], (
            f"default DrivenNTSystem must have no trainable params; "
            f"got {[p.shape for p in params]}"
        )

    def test_levels_after_init_is_baseline(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem()
        levels = nt.levels()
        for k, v in nt.baselines.items():
            assert math.isclose(levels[k], v, abs_tol=1e-9), (
                f"channel {k}: expected baseline {v}, got {levels[k]}"
            )

    def test_no_w_param_when_trainable_false(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=False)
        assert not hasattr(nt, "W_param") or nt.W_param is None, (
            "trainable_W=False must NOT expose a W_param parameter"
        )


# ──────────────────────────────────────────────────────────────────────
# B) Trainable-W path — Parameter exposed, init from _DEFAULT_W
# ──────────────────────────────────────────────────────────────────────


class TestTrainableParameter:

    def test_trainable_creates_w_parameter(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        assert hasattr(nt, "W_param"), "must expose `W_param` attribute"
        assert isinstance(nt.W_param, nn.Parameter)
        assert nt.W_param.shape == (7, 5)

    def test_parameters_lists_exactly_w(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        params = list(nt.parameters())
        assert len(params) == 1
        assert params[0] is nt.W_param

    def test_initial_values_match_default_w(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        from neuroslm.emergent.driven_nt import _DEFAULT_W, _CHANNELS, _DRIVERS
        nt = DrivenNTSystem(trainable_W=True)
        for i, ch in enumerate(_CHANNELS):
            for j, _drv in enumerate(_DRIVERS):
                expected = float(_DEFAULT_W[ch][j])
                got = float(nt.W_param[i, j].item())
                assert math.isclose(got, expected, abs_tol=1e-7), (
                    f"W_param[{ch}, {_drv}]: expected {expected}, got {got}"
                )


# ──────────────────────────────────────────────────────────────────────
# C) OU state evolves in float, identical to the legacy path
# ──────────────────────────────────────────────────────────────────────


class TestFloatPathIdentical:

    def test_step_full_identical_with_and_without_trainable(self):
        """The float OU state must be numerically identical regardless
        of whether W is a Parameter, so enabling trainable_W never
        changes the observable behaviour on its own."""
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt_a = DrivenNTSystem(trainable_W=False)
        nt_b = DrivenNTSystem(trainable_W=True)

        torch.manual_seed(0)
        # Run a deterministic stream of drivers through both systems.
        for t in range(40):
            kwargs = {
                "loss":             5.0 + 0.1 * math.sin(t * 0.3),
                "grad_norm":        1.0 + 0.2 * math.cos(t * 0.4),
                "activation":       0.5 + 0.1 * math.sin(t * 0.5),
                "ignition_rate":    0.2 + 0.05 * math.cos(t * 0.6),
                "attn_entropy_norm":0.4 + 0.05 * math.sin(t * 0.7),
            }
            nt_a.step_full(**kwargs)
            nt_b.step_full(**kwargs)

        la = nt_a.levels()
        lb = nt_b.levels()
        for k in la:
            assert math.isclose(la[k], lb[k], abs_tol=1e-7), (
                f"channel {k}: trainable-False={la[k]} vs "
                f"trainable-True={lb[k]} — float OU must be identical"
            )

    def test_w_param_grad_unset_after_step_full(self):
        """``step_full`` must NOT contaminate ``W.grad`` — the float
        path must use ``W.detach()`` so no gradient flows from the
        non-differentiable EMA standardisation through the OU chain.
        """
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        # Run a few steps.
        for _ in range(10):
            nt.step_full(loss=5.0, grad_norm=1.0, activation=0.5,
                         ignition_rate=0.2, attn_entropy_norm=0.4)
        assert nt.W_param.grad is None, (
            "step_full() leaked gradient into W_param.grad — OU update "
            "must use W.detach()"
        )


# ──────────────────────────────────────────────────────────────────────
# D) predict_nt_tensor — differentiable readout
# ──────────────────────────────────────────────────────────────────────


class TestPredictNtTensor:

    def _drivers(self):
        return dict(
            loss=4.0, grad_norm=1.5, activation=0.6,
            ignition_rate=0.25, attn_entropy_norm=0.35,
        )

    def test_returns_tensor_of_shape_seven(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        # Warm up running stats so z != 0.
        for _ in range(10):
            nt.step_full(**self._drivers())
        out = nt.predict_nt_tensor(self._drivers())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (7,)

    def test_values_in_zero_one(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        for _ in range(10):
            nt.step_full(**self._drivers())
        out = nt.predict_nt_tensor(self._drivers())
        assert (out >= 0.0).all() and (out <= 1.0).all(), (
            f"sigmoid readout must be in [0,1], got min={out.min()} "
            f"max={out.max()}"
        )

    def test_requires_grad_when_trainable(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        for _ in range(10):
            nt.step_full(**self._drivers())
        out = nt.predict_nt_tensor(self._drivers())
        assert out.requires_grad, (
            "predict_nt_tensor must return a tensor with requires_grad "
            "when trainable_W=True"
        )

    def test_grad_flows_to_w_param(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        # Warm up so at least one driver has a non-zero z-score.
        for _ in range(10):
            nt.step_full(**self._drivers())
        # Take a fresh sample with drivers that produce a non-zero z.
        spike = dict(self._drivers())
        spike["loss"] = 1.0   # well below the EMA mean → big surprise
        out = nt.predict_nt_tensor(spike)
        loss = out.sum()
        loss.backward()
        assert nt.W_param.grad is not None
        assert torch.isfinite(nt.W_param.grad).all()
        # Some entries must be non-zero (the columns whose driver had z≠0).
        assert nt.W_param.grad.abs().sum().item() > 0.0, (
            "W_param.grad is identically zero — gradient not flowing"
        )

    def test_detached_when_trainable_false(self):
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=False)
        for _ in range(10):
            nt.step_full(**self._drivers())
        out = nt.predict_nt_tensor(self._drivers())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (7,)
        assert not out.requires_grad, (
            "with trainable_W=False, the tensor readout must be "
            "detached so callers can be polymorphic"
        )


# ──────────────────────────────────────────────────────────────────────
# E) The two readouts agree numerically
# ──────────────────────────────────────────────────────────────────────


class TestReadoutsAgree:

    def test_predict_matches_levels_after_step(self):
        """After step_full(drivers), `levels()` and
        `predict_nt_tensor(drivers)` differ only by:
           - levels() is the *post-OU* sigmoid
           - predict_nt_tensor() is the *instantaneous* sigmoid:
             σ(μ + W·z(drivers))

        We pin a relaxed version: when the system is at rest (μ = y at
        baseline) and we supply the all-zero driver, both must return
        the baseline."""
        from neuroslm.emergent.driven_nt import DrivenNTSystem
        nt = DrivenNTSystem(trainable_W=True)
        # No drivers at all → z=0 for every column → readout = σ(μ)
        # = baseline.
        out = nt.predict_nt_tensor({})
        lev = nt.levels()
        for i, ch in enumerate(("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")):
            expected = lev[ch]
            got = float(out[i].item())
            assert math.isclose(got, expected, abs_tol=1e-6), (
                f"channel {ch}: predict={got} vs levels={expected}"
            )


# ──────────────────────────────────────────────────────────────────────
# F) DSL parses `nt_w_trainable: true`
# ──────────────────────────────────────────────────────────────────────


class TestDSLParse:

    def test_parser_reads_nt_w_trainable_true(self):
        from neuroslm.dsl.training_config import parse_training_config
        # parse_training_config expects the BODY of the training block,
        # not the wrapper.
        body = "steps: 100, seq_len: 32, nt_w_trainable: true"
        cfg = parse_training_config(body)
        assert hasattr(cfg, "nt_w_trainable")
        assert cfg.nt_w_trainable is True

    def test_parser_default_is_false(self):
        from neuroslm.dsl.training_config import parse_training_config
        body = "steps: 100, seq_len: 32"
        cfg = parse_training_config(body)
        assert cfg.nt_w_trainable is False
