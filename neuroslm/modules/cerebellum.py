"""Cerebellum: predictive error-driven forward model.

The cerebellum contains >50% of the brain's neurons and implements:
  - Ultra-fast forward models (predict sensory consequences of actions)
  - Error-driven learning via climbing fiber signals
  - Timing and sequencing (olivocerebellar loop)
  - Granule cell expansion: sparse high-dimensional encoding
  - Purkinje cell integration: massive fan-in, single output

The cerebellar microcircuit:
  Mossy fibers → Granule cells (expansion) → Parallel fibers →
  Purkinje cells (integration) → Deep nuclei (output)
  Climbing fibers (from inferior olive) → Purkinje cells (error signal)

For language: cerebellum predicts next-token probability distributions
and learns from prediction errors. This is analogous to how it predicts
sensory consequences of motor commands.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GranuleLayer(nn.Module):
    """Granule cell expansion: sparse high-dimensional encoding.

    Granule cells are the most numerous neurons in the brain.
    They expand input dimensionality and create sparse, decorrelated
    representations — like a learned random projection.
    """

    def __init__(self, d_input: int, expansion: int = 4, sparsity: float = 0.9):
        super().__init__()
        d_expanded = d_input * expansion
        self.expand = nn.Linear(d_input, d_expanded)
        self.sparsity = sparsity
        # Golgi cell inhibition: controls sparsity
        self.golgi = nn.Linear(d_expanded, d_expanded)

    def forward(self, mossy: torch.Tensor) -> torch.Tensor:
        """mossy: (B, d_input) → (B, d_expanded) sparse expansion."""
        expanded = F.relu(self.expand(mossy))
        # Golgi inhibition: enforce sparsity via top-k
        inhibition = torch.sigmoid(self.golgi(expanded))
        sparse = expanded * inhibition
        # Hard sparsity: zero out bottom fraction
        if self.training:
            k = max(1, int(sparse.size(-1) * (1 - self.sparsity)))
            topk_vals, _ = sparse.topk(k, dim=-1)
            threshold = topk_vals[:, -1:].detach()
            sparse = sparse * (sparse >= threshold).float()
        return sparse


class PurkinjeCell(nn.Module):
    """Purkinje cell: massive integration + climbing fiber error modulation.

    Each Purkinje cell receives ~200,000 parallel fiber inputs (granule cells)
    and ONE climbing fiber (error signal from inferior olive).
    """

    def __init__(self, d_parallel: int, d_output: int):
        super().__init__()
        # Parallel fiber weights (learned via LTD from climbing fibers)
        self.parallel_weights = nn.Linear(d_parallel, d_output)
        # Climbing fiber modulation: scales learning rate / output
        self.climbing_gate = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.Sigmoid(),
        )

    def forward(self, parallel: torch.Tensor,
                climbing_error: torch.Tensor | None = None) -> torch.Tensor:
        """
        parallel: (B, d_parallel) from granule cells
        climbing_error: (B, d_output) error signal (optional)
        """
        output = self.parallel_weights(parallel)
        if climbing_error is not None:
            # Climbing fiber suppresses Purkinje output on error
            gate = self.climbing_gate(climbing_error)
            output = output * (1 - 0.5 * gate)
        return output


class Cerebellum(nn.Module):
    """Full cerebellar circuit: prediction, error computation, timing.

    Predicts the next state and learns from prediction errors.
    Operates as an auxiliary forward model that's faster than
    the main cortical forward model.
    """

    def __init__(self, d_sem: int, expansion: int = 4):
        super().__init__()
        self.d_sem = d_sem
        d_expanded = d_sem * expansion

        # Mossy fiber input
        self.mossy_proj = nn.Linear(d_sem * 2, d_sem)  # state + action → mossy

        # Granule layer: sparse expansion
        self.granule = GranuleLayer(d_sem, expansion=expansion, sparsity=0.85)

        # Purkinje cells: integrate expanded representation
        self.purkinje = PurkinjeCell(d_expanded, d_sem)

        # Deep cerebellar nuclei: final output
        self.deep_nuclei = nn.Sequential(
            nn.Linear(d_sem, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

        # Inferior olive: computes climbing fiber error
        self.olive = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.Tanh(),
        )

        # Timing circuit: oscillatory sequencing
        self.timing = nn.GRUCell(d_sem, d_sem)
        self.register_buffer("timing_state", torch.zeros(1, d_sem))

        # Prediction head
        self.predictor = nn.Linear(d_sem, d_sem)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                actual_next: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            state: (B, d_sem) current brain state
            action: (B, d_sem) current action/motor command
            actual_next: (B, d_sem) actual next state (for error computation)

        Returns dict with prediction, error, cerebellar output.
        """
        B = state.size(0)

        # Mossy fiber input: state + action
        mossy = self.mossy_proj(torch.cat([state, action], dim=-1))

        # Granule cell expansion
        granule_out = self.granule(mossy)

        # Compute climbing fiber error (if we have ground truth)
        climbing_error = None
        if actual_next is not None:
            predicted = self.predictor(state)
            error_signal = actual_next - predicted
            climbing_error = self.olive(
                torch.cat([error_signal, state], dim=-1))

        # Purkinje integration
        purkinje_out = self.purkinje(granule_out, climbing_error)

        # Deep nuclei output
        output = self.deep_nuclei(purkinje_out)

        # Timing update
        ts = self.timing_state.expand(B, -1) if self.timing_state.size(0) != B else self.timing_state
        new_ts = self.timing(output, ts)
        self.timing_state = new_ts.detach().mean(0, keepdim=True)

        # Prediction for next tick
        prediction = self.predictor(output)

        # Prediction error magnitude (for upstream learning signals)
        pred_error = torch.zeros(B, device=state.device)
        if actual_next is not None:
            pred_error = (prediction - actual_next).pow(2).mean(-1)

        return {
            "output": output,              # (B, D) cerebellar output
            "prediction": prediction,      # (B, D) predicted next state
            "error": pred_error,           # (B,) prediction error magnitude
            "timing": new_ts,              # (B, D) timing/sequencing state
            "climbing_error": climbing_error,  # (B, D) or None
        }
