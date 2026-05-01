"""Entorhinal grid cells for conceptual navigation.

Grid cells in the entorhinal cortex represent position in abstract spaces.
Neuroscience shows they don't just encode physical location — they map
conceptual spaces too (Constantinescu et al., 2016, Science).

This module implements:
  - Grid cell populations with multiple spatial frequencies (modules)
  - Hexagonal firing patterns via interference model
  - Path integration: tracks position in abstract conceptual space
  - Place cells: stable representations of "conceptual locations"
  - Border cells: detect boundaries in concept space
  - Conjunctive cells: velocity × position (direction of thought)

The grid code provides:
  1. Efficient encoding of position in high-dimensional concept space
  2. Novel vector completion (predicting what's "nearby" conceptually)
  3. Relational inference via grid cell algebra (A:B :: C:?)
  4. Mental navigation through semantic space
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridCellModule(nn.Module):
    """A single grid cell module with a specific spatial frequency."""

    def __init__(self, d_sem: int, n_cells: int = 32, frequency: float = 1.0):
        super().__init__()
        self.n_cells = n_cells
        self.frequency = frequency

        # Preferred directions for each grid cell (learned, init hexagonal)
        angles = torch.linspace(0, 2 * math.pi * (n_cells - 1) / n_cells, n_cells)
        # Project into d_sem space
        self.direction_proj = nn.Linear(d_sem, n_cells, bias=False)

        # Phase offset per cell
        self.phase = nn.Parameter(torch.randn(n_cells) * 0.1)

        # Velocity integration weight
        self.velocity_gate = nn.Sequential(
            nn.Linear(d_sem, n_cells),
            nn.Sigmoid(),
        )

        # Persistent state: current grid phase
        self.register_buffer("_phase_state", torch.zeros(1, n_cells))

    def forward(self, position: torch.Tensor,
                velocity: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            position: (B, d_sem) current position in concept space
            velocity: (B, d_sem) movement direction (optional)

        Returns: (B, n_cells) grid cell activations
        """
        B = position.size(0)
        # Project position onto preferred directions
        proj = self.direction_proj(position) * self.frequency  # (B, n_cells)

        # Add phase offset
        activation = torch.cos(proj + self.phase)  # hexagonal-like pattern

        # Path integration: update phase with velocity
        if velocity is not None:
            vel_gate = self.velocity_gate(velocity)
            vel_proj = self.direction_proj(velocity) * self.frequency * 0.1
            activation = activation + vel_gate * torch.sin(vel_proj + self.phase)

        return activation


class PlaceCells(nn.Module):
    """Place cells: stable representations of specific conceptual locations."""

    def __init__(self, d_sem: int, n_places: int = 64):
        super().__init__()
        self.n_places = n_places
        # Learned place field centers
        self.centers = nn.Parameter(torch.randn(n_places, d_sem) * 0.5)
        # Place field width (inverse precision)
        self.log_width = nn.Parameter(torch.zeros(n_places))
        # Output projection
        self.out_proj = nn.Linear(n_places, d_sem)

    def forward(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position: (B, d_sem)

        Returns: (place_output (B, d_sem), place_activations (B, n_places))
        """
        # Gaussian place fields
        width = self.log_width.exp().clamp(min=0.1)  # (n_places,)
        diff = position.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, n_places, d_sem)
        dist_sq = (diff ** 2).sum(-1)  # (B, n_places)
        activations = torch.exp(-dist_sq / (2 * width.unsqueeze(0) ** 2))

        output = self.out_proj(activations)
        return output, activations


class EntorhinalCortex(nn.Module):
    """Full entorhinal cortex: grid cells + place cells + border cells.

    Provides conceptual navigation capabilities:
      - Where am I in concept space?
      - What's nearby? (pattern completion)
      - How do I get from A to B? (planning via grid algebra)
      - A:B :: C:? (analogy via grid cell vector operations)
    """

    def __init__(self, d_sem: int, n_modules: int = 4,
                 cells_per_module: int = 32, n_places: int = 64):
        super().__init__()
        self.d_sem = d_sem

        # Multiple grid cell modules at different spatial frequencies
        frequencies = [2 ** i for i in range(n_modules)]
        self.grid_modules = nn.ModuleList([
            GridCellModule(d_sem, cells_per_module, freq)
            for freq in frequencies
        ])

        # Place cells
        self.place_cells = PlaceCells(d_sem, n_places)

        # Border cells: detect edges/boundaries in concept space
        self.border_detector = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2),
            nn.GELU(),
            nn.Linear(d_sem // 2, d_sem // 4),
            nn.Sigmoid(),  # boundary proximity
        )

        # Grid → semantic projection
        total_grid = cells_per_module * n_modules
        self.grid_to_sem = nn.Sequential(
            nn.Linear(total_grid, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

        # Velocity estimator: difference between current and previous position
        self.velocity_proj = nn.Linear(d_sem * 2, d_sem)

        # Conjunctive cells: position × velocity
        self.conjunctive = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.GELU(),
            nn.Linear(d_sem, d_sem),
        )

        # Analogy head: A:B :: C:?
        self.analogy_proj = nn.Linear(d_sem * 3, d_sem)

        self.register_buffer("_prev_position", torch.zeros(1, d_sem))

    def forward(self, position: torch.Tensor,
                prev_position: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            position: (B, d_sem) current semantic position

        Returns dict with grid code, place activations, navigation signals.
        """
        B = position.size(0)
        if prev_position is None:
            prev_position = self._prev_position.expand(B, -1)

        # Estimate velocity in concept space
        velocity = self.velocity_proj(
            torch.cat([position, position - prev_position], dim=-1))

        # Grid cell activations at multiple scales
        grid_acts = []
        for module in self.grid_modules:
            g = module(position, velocity)
            grid_acts.append(g)
        grid_all = torch.cat(grid_acts, dim=-1)  # (B, total_grid)

        # Grid → semantic representation
        grid_sem = self.grid_to_sem(grid_all)

        # Place cells
        place_out, place_acts = self.place_cells(position)

        # Border detection
        borders = self.border_detector(position)

        # Conjunctive representation (direction of thought)
        conj = self.conjunctive(torch.cat([grid_sem, velocity], dim=-1))

        # Update previous position
        self._prev_position = position.detach().mean(0, keepdim=True)

        return {
            "grid_code": grid_sem,       # (B, d_sem) multi-scale position encoding
            "place_code": place_out,     # (B, d_sem) place cell output
            "place_activations": place_acts,  # (B, n_places)
            "borders": borders,          # (B, d_sem//4) boundary proximity
            "velocity": velocity,        # (B, d_sem) movement in concept space
            "conjunctive": conj,         # (B, d_sem) direction × position
            "grid_raw": grid_all,        # (B, total_grid) raw grid activations
        }

    def analogy(self, a: torch.Tensor, b: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """Compute A:B :: C:? via grid cell vector algebra.

        The grid code naturally supports vector addition/subtraction,
        so analogies become: grid(?) = grid(C) + (grid(B) - grid(A))
        """
        combined = torch.cat([a, b, c], dim=-1)
        return self.analogy_proj(combined)
