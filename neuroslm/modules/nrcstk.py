# -*- coding: utf-8 -*-
"""NRCSTK Controller — metabolic-market selection through pruning.

Phase B of the C → A → B Multi-Objective-Fitness implementation.

The NRCSTK ("Neuronal-Resource-Constrained Selection Through Killing")
controller implements the metabolic-market selection pressure from the
work order:

    "Wir können ein System implementieren, in dem Neuronen in einem
     'metabolischen Markt' um begrenzte Energie konkurrieren.  Neuronen,
     die ihre Oszillationen nicht effizient mit den Eingangsmustern
     synchronisieren, 'verhungern' und werden gelöscht.  Dies erzwingt
     die Entdeckung von Least-Action-Prinzipien und extrem sparsamen
     Topologien."

Mathematical contract
---------------------
For a target layer producing activations :math:`\\mathbf{a} \\in \\mathbb{R}^{B \\times T \\times D}`:

1. **Demand** (per-neuron mean activation magnitude, EMA-smoothed):

   .. math::
       d_i(t)         &= \\overline{|\\mathbf{a}_{:,:,i}|} \\\\
       \\tilde{d}_i(t) &= \\alpha\\, d_i(t) + (1-\\alpha)\\, \\tilde{d}_i(t-1)

2. **Budget loss** (Hinge-squared overshoot of the population mean):

   .. math::
       \\mathcal{L}_\\text{met} = \\bigl[\\mathrm{ReLU}\\bigl(
           \\tfrac{1}{D}\\sum_i \\tilde{d}_i - B\\bigr)\\bigr]^2

   where :math:`B \\in [0, 1]` is the budget fraction.  Driven by the
   optimizer through ``a.grad`` it pushes the weakest-used neurons
   toward zero output, after which their EMA collapses below
   ``prune_threshold`` and they get masked out.

3. **Pruning mask** (hard one-hot kill signal):

   .. math::
       m_i = \\mathbb{1}[\\tilde{d}_i > \\tau_\\text{prune}]

4. **Live-neuron count** (telemetry):

   .. math::
       N_\\text{live} = \\sum_i m_i

Usage pattern
-------------
.. code-block:: python

    # 1. Build from config
    ctrl = NRCSTKController.from_fitness_config(
        target_dim=trunk.d_model, config=cfg.fitness)

    # 2. Inside forward, observe + optionally apply mask
    a = trunk(x)
    ctrl.observe(a.detach())          # update EMA (no grad)
    a = ctrl.apply_mask(a)            # forward kill on pruned neurons
    loss_met = ctrl.metabolic_loss(a) # backprop signal

    # 3. Inject into LossBundle
    bundle.metabolic = loss_met
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from neuroslm.dsl.training_config import FitnessConfig


class NRCSTKController(nn.Module):
    """Metabolic-budget enforcer with EMA-driven neuron pruning.

    Parameters
    ----------
    target_dim : int
        Number of neurons (last-axis dimensionality) the controller
        monitors.  Each neuron gets its own EMA channel.
    budget : float in [0, 1]
        Fraction of mean-magnitude activity allowed before the
        ``metabolic_loss`` penalty kicks in.  0.7 default ⇒ population
        mean ``|a|`` is allowed up to 0.7 before any pressure.
    prune_threshold : float in [0, 0.5)
        EMA value below which a neuron is considered "starved" and
        gets zeroed in :meth:`apply_mask`.
    ema_alpha : float in (0, 1]
        Single-sided EMA decay constant.  0.05 default ⇒ half-life
        of ~14 observations, matching the time-scale of neuro-modulator
        traces elsewhere in the system.

    Attributes
    ----------
    demand_ema : torch.Tensor, buffer of shape ``(target_dim,)``
        The per-neuron exponentially-smoothed demand signal.  Not a
        :class:`Parameter` (no gradient training of the EMA itself).
    """

    def __init__(
        self,
        target_dim: int,
        budget: float = 0.7,
        prune_threshold: float = 0.05,
        ema_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        if not (0.0 <= budget <= 1.0):
            raise ValueError(f"budget must be in [0, 1], got {budget}")
        if not (0.0 <= prune_threshold < 0.5):
            raise ValueError(
                f"prune_threshold must be in [0, 0.5), got {prune_threshold}"
            )
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {ema_alpha}"
            )

        self.target_dim = int(target_dim)
        self.budget = float(budget)
        self.prune_threshold = float(prune_threshold)
        self.ema_alpha = float(ema_alpha)

        # Buffer (not a Parameter): persists across train/eval and gets
        # serialised with the model, but never receives gradient.
        self.register_buffer(
            "demand_ema",
            torch.zeros(target_dim, dtype=torch.float32),
        )
        # Count of how many observations have been recorded.  Before
        # the first observation no neuron has been "judged" so the
        # pruning mask is all-ones; after the first observation, the
        # EMA is what decides who lives.
        self.register_buffer(
            "observation_count",
            torch.zeros((), dtype=torch.long),
        )

    # ── alternate constructor ──────────────────────────────────────

    @classmethod
    def from_fitness_config(
        cls, target_dim: int, config: "FitnessConfig"
    ) -> "NRCSTKController":
        """Build an NRCSTKController from a parsed `FitnessConfig`.

        This is the canonical wiring used by `FitnessComposer` and the
        harness: it sources `budget` and `prune_threshold` from the
        DSL-declared values so a single arch.neuro line controls both
        pieces of the metabolic loop.
        """
        return cls(
            target_dim=target_dim,
            budget=config.metabolic_budget,
            prune_threshold=config.metabolic_prune_threshold,
        )

    # ── EMA observation ────────────────────────────────────────────

    @torch.no_grad()
    def observe(self, activations: torch.Tensor) -> None:
        """Update the per-neuron demand EMA from a batch of activations.

        Parameters
        ----------
        activations : torch.Tensor of shape ``(..., target_dim)``
            Any leading-dim shape is accepted; mean magnitude is taken
            over every axis except the last.

        Notes
        -----
        Wrapped in ``@torch.no_grad`` because the EMA is a *statistic*,
        not a function to differentiate through.  The gradient signal
        for selection comes from :meth:`metabolic_loss`, not this method.
        """
        if activations.shape[-1] != self.target_dim:
            raise ValueError(
                f"NRCSTKController: activations last dim "
                f"{activations.shape[-1]} != target_dim {self.target_dim}"
            )
        # Mean over every axis except the last (=neurons).
        reduce_dims = tuple(range(activations.dim() - 1))
        per_neuron = activations.abs().mean(dim=reduce_dims)  # (D,)
        # Single-sided EMA: new = alpha * obs + (1 - alpha) * old
        self.demand_ema.mul_(1.0 - self.ema_alpha).add_(
            per_neuron * self.ema_alpha
        )
        # Flag that the EMA has now been "judged" at least once.
        self.observation_count.add_(1)

    # ── pruning mask ───────────────────────────────────────────────

    def pruning_mask(self) -> torch.Tensor:
        """Return a ``(target_dim,)`` 0/1 mask: 1 = alive, 0 = pruned.

        Pruning is "hard" (one-hot) so the apply step zeros the entire
        neuron output rather than just scaling it down.  This matches
        the biological semantic of a starved neuron going silent.

        Before any :meth:`observe` call the mask is all-ones — no
        observation means no judgement, so every neuron starts alive.
        After the first observation, the EMA drives the gating.
        """
        if self.observation_count.item() == 0:
            return torch.ones_like(self.demand_ema)
        return (self.demand_ema > self.prune_threshold).to(
            self.demand_ema.dtype
        )

    def n_live_neurons(self) -> int:
        """Number of neurons currently above the prune threshold."""
        return int(self.pruning_mask().sum().item())

    def apply_mask(self, activations: torch.Tensor) -> torch.Tensor:
        """Element-wise gate ``activations * pruning_mask``.

        Broadcasts the ``(target_dim,)`` mask across all leading dims.
        Differentiable: a multiplicative gate just zeros the gradient
        through pruned neurons, which is the desired behaviour (no
        forward signal ⇒ no backward signal).
        """
        mask = self.pruning_mask()
        # Build broadcast shape (1, ..., 1, D) matching activations.
        broadcast_shape = (1,) * (activations.dim() - 1) + (self.target_dim,)
        return activations * mask.view(broadcast_shape)

    # ── metabolic loss ─────────────────────────────────────────────

    def metabolic_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """Hinge-squared overshoot of the population mean magnitude.

        .. math::
            \\mathcal{L}_\\text{met} = \\bigl[\\mathrm{ReLU}\\bigl(
                \\overline{|\\mathbf{a}|} - B\\bigr)\\bigr]^2

        Returns a scalar autograd-traced tensor.  Backprop through
        ``activations`` is the selection pressure: gradient pushes
        every neuron toward zero until the population mean drops
        below the budget.
        """
        if activations.shape[-1] != self.target_dim:
            raise ValueError(
                f"NRCSTKController.metabolic_loss: activations last dim "
                f"{activations.shape[-1]} != target_dim {self.target_dim}"
            )
        mean_mag = activations.abs().mean()
        overshoot = torch.relu(mean_mag - self.budget)
        return overshoot * overshoot

    # ── repr ───────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"target_dim={self.target_dim}, budget={self.budget}, "
            f"prune_threshold={self.prune_threshold}, "
            f"ema_alpha={self.ema_alpha}, n_live={self.n_live_neurons()}"
        )
