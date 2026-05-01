"""Cortical column with dendritic predictive processing.

Implements the Larkum (2013) / Sacramento et al. (2018) two-compartment
neuron model: apical dendrites carry top-down predictions, basal dendrites
carry bottom-up evidence. When prediction matches input → suppression.
When mismatch → burst firing → learning signal.

Each column has:
  - Basal compartment: feedforward input (sensory / lower cortex)
  - Apical compartment: top-down prediction (PFC / higher cortex)
  - Lateral inhibition: winner-take-most within minicolumn
  - Burst detection: mismatch → strong signal → plasticity
  - Layer 2/3 output: lateral associations
  - Layer 5 output: motor / subcortical drive
  - Layer 6 output: thalamocortical feedback

This is the fundamental computational unit repeated across cortex.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticCompartment(nn.Module):
    """Single dendritic compartment with nonlinear integration."""

    def __init__(self, d_input: int, d_state: int, n_segments: int = 4):
        super().__init__()
        self.n_segments = n_segments
        # Each dendritic segment has its own receptive field
        self.segment_proj = nn.ModuleList([
            nn.Linear(d_input, d_state) for _ in range(n_segments)
        ])
        # Dendritic nonlinearity: NMDA-like spike
        self.threshold = nn.Parameter(torch.zeros(n_segments, d_state))
        # Segment gating (which segments are active)
        self.gate = nn.Linear(d_input, n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_input) → (B, d_state) integrated dendritic signal."""
        gates = torch.sigmoid(self.gate(x))  # (B, n_segments)
        outputs = []
        for i, proj in enumerate(self.segment_proj):
            seg_out = proj(x)  # (B, d_state)
            # NMDA-like plateau potential: sigmoid with learnable threshold
            plateau = torch.sigmoid(5.0 * (seg_out - self.threshold[i]))
            seg_out = seg_out * plateau
            outputs.append(seg_out * gates[:, i:i+1])
        # Superlinear integration: segments cooperate nonlinearly
        stacked = torch.stack(outputs, dim=0)  # (n_seg, B, d_state)
        integrated = stacked.sum(0) + 0.1 * stacked.prod(0).sign() * stacked.abs().prod(0).pow(1.0 / self.n_segments)
        return integrated


class CorticalColumn(nn.Module):
    """A cortical minicolumn implementing predictive processing.

    Computes prediction error between top-down (apical) and bottom-up (basal).
    Burst firing on mismatch drives learning and routing.
    """

    def __init__(self, d_sem: int, n_minicolumns: int = 8):
        super().__init__()
        self.d_sem = d_sem
        self.n_mini = n_minicolumns
        d_mini = d_sem // n_minicolumns

        # Basal dendrites: bottom-up (feedforward)
        self.basal = DendriticCompartment(d_sem, d_mini, n_segments=4)
        # Apical dendrites: top-down (feedback/prediction)
        self.apical = DendriticCompartment(d_sem, d_mini, n_segments=3)

        # Soma integration: combines basal + apical
        self.soma = nn.Sequential(
            nn.Linear(d_mini * 2, d_mini),
            nn.GELU(),
        )

        # Burst detector: large prediction error triggers burst
        self.burst_detector = nn.Sequential(
            nn.Linear(d_mini, 1),
            nn.Sigmoid(),
        )

        # Lateral inhibition across minicolumns (winner-take-most)
        self.lateral_inhibition = nn.Linear(d_mini * n_minicolumns, n_minicolumns)

        # Layer outputs
        self.layer23_out = nn.Linear(d_mini, d_sem)   # lateral / associative
        self.layer5_out = nn.Linear(d_mini, d_sem)     # subcortical drive
        self.layer6_out = nn.Linear(d_mini, d_sem)     # thalamocortical feedback

        # Prediction error projection
        self.error_proj = nn.Linear(d_mini, d_sem)

    def forward(self, bottom_up: torch.Tensor,
                top_down: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            bottom_up: (B, d_sem) feedforward sensory/lower input
            top_down: (B, d_sem) feedback prediction from higher areas

        Returns dict with layer outputs, prediction error, burst signal.
        """
        B = bottom_up.size(0)

        # Dendritic processing
        basal_signal = self.basal(bottom_up)    # (B, d_mini)
        apical_signal = self.apical(top_down)   # (B, d_mini)

        # Prediction error = basal - apical (mismatch)
        pred_error = basal_signal - apical_signal

        # Soma: integrate both compartments
        soma_in = torch.cat([basal_signal, apical_signal], dim=-1)
        soma_out = self.soma(soma_in)  # (B, d_mini)

        # Burst detection: large prediction error → burst firing
        burst = self.burst_detector(pred_error.abs())  # (B, 1)

        # Burst amplifies output (calcium spike analog)
        amplified = soma_out * (1.0 + 2.0 * burst)

        # Layer-specific outputs
        layer23 = self.layer23_out(amplified)   # lateral associations
        layer5 = self.layer5_out(amplified)     # subcortical
        layer6 = self.layer6_out(soma_out)      # thalamic feedback (not burst-amplified)
        error = self.error_proj(pred_error)

        return {
            "layer23": layer23,
            "layer5": layer5,
            "layer6": layer6,
            "prediction_error": error,
            "burst": burst.squeeze(-1),
            "soma": amplified,
        }


class CorticalSheet(nn.Module):
    """A sheet of cortical columns — repeated across a cortical area.

    Multiple columns with lateral connections and shared lateral inhibition.
    Implements the canonical microcircuit pattern seen across cortex.
    """

    def __init__(self, d_sem: int, n_columns: int = 4, n_minicolumns: int = 8):
        super().__init__()
        self.columns = nn.ModuleList([
            CorticalColumn(d_sem, n_minicolumns) for _ in range(n_columns)
        ])
        # Lateral connections between columns (layer 2/3 horizontal fibers)
        self.lateral = nn.Sequential(
            nn.Linear(d_sem * n_columns, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )
        # Aggregate layer 5 outputs for subcortical drive
        self.subcortical_agg = nn.Linear(d_sem * n_columns, d_sem)
        self.n_columns = n_columns

    def forward(self, bottom_up: torch.Tensor,
                top_down: torch.Tensor) -> dict[str, torch.Tensor]:
        col_outs = [col(bottom_up, top_down) for col in self.columns]

        # Lateral integration (layer 2/3 horizontal connections)
        l23_all = torch.cat([c["layer23"] for c in col_outs], dim=-1)
        lateral = self.lateral(l23_all)

        # Subcortical drive (layer 5 aggregate)
        l5_all = torch.cat([c["layer5"] for c in col_outs], dim=-1)
        subcortical = self.subcortical_agg(l5_all)

        # Aggregate prediction error and burst
        mean_error = torch.stack([c["prediction_error"] for c in col_outs]).mean(0)
        mean_burst = torch.stack([c["burst"] for c in col_outs]).mean(0)

        # Thalamocortical feedback (layer 6 mean)
        thalamic_fb = torch.stack([c["layer6"] for c in col_outs]).mean(0)

        return {
            "output": lateral,             # main cortical output
            "subcortical": subcortical,    # drive to BG/brainstem
            "prediction_error": mean_error,
            "burst": mean_burst,
            "thalamic_feedback": thalamic_fb,
        }
