# -*- coding: utf-8 -*-
"""THSD Integration for Training Pipeline

Bridges DSL/THSD architecture to training loop.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any

from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.hypergraph_ir import HypergraphBuilder
from neuroslm.dsl.thsd_codegen import THSDCodeGenerator
from neuroslm.dsl.thsd_plasticity import (
    StructuralPlasticityController,
    HebbianFastWeights,
    NEMORIConsolidator,
)


class THSDTrainingWrapper(nn.Module):
    """Wraps THSD modules for training with constraint tracking."""

    def __init__(
        self,
        thsd_module: nn.Module,
        enable_fast_weights: bool = False,
        enable_plasticity: bool = False,
    ):
        super().__init__()
        self.thsd_module = thsd_module
        self.enable_fast_weights = enable_fast_weights
        self.enable_plasticity = enable_plasticity

        # Optional: Hebbian fast weights
        if enable_fast_weights and hasattr(thsd_module, "stalk_dim"):
            self.fast_weights = HebbianFastWeights(
                dim=thsd_module.stalk_dim, eta=0.05
            )
        else:
            self.fast_weights = None

        # Optional: Structural plasticity
        if enable_plasticity:
            self.plasticity_controller = StructuralPlasticityController()
            self.nemori_consolidator = NEMORIConsolidator()
        else:
            self.plasticity_controller = None
            self.nemori_consolidator = None

        # Constraint tracking
        self.phi_history = []
        self.spectral_gap_history = []

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through THSD module.

        Args:
            x: Input tensor
            h_prev: Previous hidden state (for fast weights)

        Returns:
            Output tensor with THSD constraints applied
        """
        # Core forward pass
        output = self.thsd_module(x)

        # Apply fast weights if enabled
        if self.fast_weights is not None and h_prev is not None:
            output = self.fast_weights(output, h_prev)

        # Track Φ
        if hasattr(self.thsd_module, "compute_phi_proxy"):
            phi = self.thsd_module.compute_phi_proxy(output)
            self.phi_history.append(phi)

        return output

    def consolidate(self) -> None:
        """Apply NEMORI consolidation (predictive forgetting)."""
        if self.nemori_consolidator is not None:
            # Would apply consolidation logic here
            pass


class THSDArchitectureLoader:
    """Loads THSD architecture from DSL and compiles to modules."""

    @staticmethod
    def from_dsl_file(
        dsl_path: str,
        enable_fast_weights: bool = False,
        enable_plasticity: bool = False,
    ) -> THSDTrainingWrapper:
        """Load THSD architecture from arch.neuro file.

        Args:
            dsl_path: Path to arch.neuro file
            enable_fast_weights: Enable Hebbian fast weights
            enable_plasticity: Enable structural plasticity

        Returns:
            THSDTrainingWrapper ready for training
        """
        # Read DSL file
        with open(dsl_path, "r") as f:
            dsl_code = f.read()

        return THSDArchitectureLoader.from_dsl_code(
            dsl_code,
            enable_fast_weights=enable_fast_weights,
            enable_plasticity=enable_plasticity,
        )

    @staticmethod
    def from_dsl_code(
        dsl_code: str,
        enable_fast_weights: bool = False,
        enable_plasticity: bool = False,
    ) -> THSDTrainingWrapper:
        """Load THSD architecture from DSL code string.

        Args:
            dsl_code: DSL code as string
            enable_fast_weights: Enable Hebbian fast weights
            enable_plasticity: Enable structural plasticity

        Returns:
            THSDTrainingWrapper ready for training
        """
        # Parse DSL
        ir = NeuroMLCompiler.compile(dsl_code)

        if not ir.thsd_complexes:
            raise ValueError("No THSD complexes found in DSL")

        # Build hypergraph for first complex
        builder = HypergraphBuilder()
        complex_ir = ir.thsd_complexes[0]
        hypergraph = builder.from_complex_ir(complex_ir)

        # Generate PyTorch module
        generator = THSDCodeGenerator()
        thsd_module = generator.generate_module(hypergraph)

        # Wrap for training
        wrapper = THSDTrainingWrapper(
            thsd_module,
            enable_fast_weights=enable_fast_weights,
            enable_plasticity=enable_plasticity,
        )

        return wrapper

    @staticmethod
    def create_tiny_brain() -> THSDTrainingWrapper:
        """Create minimal THSD brain for testing."""
        dsl = """
        complex TinyBrain {
            stalk {
                representation_dim: 128,
                fisher_information_metric: "information_geometry"
            },
            topology {
                kind: "Tonnetz",
                spectral_gap: 0.3,
                dimension: 4
            },
            formal_spec {
                cohomology_floor: 0.01,
                phi_target: 0.75
            }
        }
        """
        return THSDArchitectureLoader.from_dsl_code(dsl, enable_fast_weights=True)


def train_with_thsd(
    thsd_wrapper: THSDTrainingWrapper,
    dataloader,
    loss_fn,
    optimizer,
    steps: int = 100,
    log_every: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train a THSD-based model.

    Args:
        thsd_wrapper: THSDTrainingWrapper instance
        dataloader: Training dataloader
        loss_fn: Loss function
        optimizer: Optimizer
        steps: Number of training steps
        log_every: Logging frequency
        device: Device to train on

    Returns:
        Metrics dictionary
    """
    thsd_wrapper = thsd_wrapper.to(device)
    thsd_wrapper.train()

    metrics = {
        "losses": [],
        "phi_values": [],
        "steps": [],
    }

    step = 0
    for epoch in range(steps // len(dataloader) + 1):
        for batch_idx, batch in enumerate(dataloader):
            if step >= steps:
                break

            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                batch = batch.to(device)

            # Forward pass
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, batch

            output = thsd_wrapper(x)

            # Compute loss
            loss = loss_fn(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            if step % log_every == 0:
                metrics["losses"].append(loss.item())
                metrics["steps"].append(step)

                # Track Φ
                if thsd_wrapper.phi_history:
                    phi = thsd_wrapper.phi_history[-1]
                    metrics["phi_values"].append(phi)
                    print(
                        f"Step {step:5d} | Loss: {loss.item():.4f} | "
                        f"Phi: {phi:.3f}"
                    )
                else:
                    print(f"Step {step:5d} | Loss: {loss.item():.4f}")

            step += 1
            if step >= steps:
                break

    return metrics
