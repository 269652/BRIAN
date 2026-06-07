# -*- coding: utf-8 -*-
"""Colab/Jupyter utilities for evolutionary training.

Provides high-level API for:
- Loading evolved DNA and unfolding to architecture
- Resuming from saved checkpoints (base DNA + patch stack)
- Train-and-evolve workflow integration
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List

from neuroslm.compiler.ribosome import LatentDNA, DNAPatch, RibosomeCompiler


def init_evolution(dna_path: str, patch_dir: Optional[str] = None) -> dict:
    """Initialize evolutionary training from saved DNA and patches.

    Workflow:
    1. Load base DNA from dna_path
    2. If patch_dir provided, discover and apply all patches in order
    3. Unfold DNA to architecture.neuro
    4. Load fitness config from DNA or fitness.neuro
    5. Return config dict for harness initialization

    Args:
        dna_path: Path to base.dna file
        patch_dir: Optional directory containing step_XXXXX.patch.dna files

    Returns:
        dict with keys:
        - "arch_path": path to unfolded architecture
        - "arch_neuro": path to unfolded arch.neuro
        - "dna_path": path to base DNA
        - "patches_applied": list of DNAPatch objects
        - "resume_step": last training step from patches
        - "fitness_config": FitnessConfig object (from DNA or fitness.neuro)
        - "base_dna": the evolved LatentDNA after applying patches
    """
    from neuroslm.fitness import FitnessConfig

    dna_path = Path(dna_path)
    if not dna_path.exists():
        raise FileNotFoundError(f"DNA file not found: {dna_path}")

    # Load base DNA
    base_dna = LatentDNA.load(str(dna_path))

    # Load and apply patches if directory provided
    patches_applied = []
    resume_step = 0

    if patch_dir:
        patch_dir = Path(patch_dir)
        patch_files = sorted(patch_dir.glob("step_*.patch.dna"))

        for patch_file in patch_files:
            patch = DNAPatch.load(str(patch_file))
            patches_applied.append(patch)
            resume_step = max(resume_step, patch.step)

            # Apply patch to DNA (for reconstruction purposes)
            # In practice, patches are applied at inference time via layer-wise gating
            for i in range(min(len(patch.delta), len(base_dna.data))):
                base_dna.data[i] += patch.delta[i]

    # Unfold DNA to DSL
    compiler = RibosomeCompiler()
    arch_root = dna_path.parent / "evolution_arch"
    arch_root.mkdir(parents=True, exist_ok=True)

    arch_neuro_path = arch_root / "arch.neuro"
    dsl_code = compiler.dna_translator.translate(base_dna)
    arch_neuro_path.write_text(dsl_code, encoding='utf-8')

    # Load fitness config: from DNA if present, else from fitness.neuro sidefile
    fitness_config = None
    if "fitness_config" in base_dna.invariants:
        fitness_config = FitnessConfig.from_dict(base_dna.invariants["fitness_config"])
    else:
        # Try to load fitness.neuro from the architecture's parent
        fitness_neuro = dna_path.parent / "fitness.neuro"
        if fitness_neuro.exists():
            fitness_config = FitnessConfig.load(str(fitness_neuro))

    # Fallback to default fitness config
    if fitness_config is None:
        fitness_config = FitnessConfig.load_or_default("")

    return {
        "arch_path": str(arch_root),
        "arch_neuro": str(arch_neuro_path),
        "dna_path": str(dna_path),
        "patches_applied": patches_applied,
        "resume_step": resume_step,
        "fitness_config": fitness_config,
        "base_dna": base_dna,
    }


def apply_patch_stack(base_dna: LatentDNA, patches: List[DNAPatch]) -> LatentDNA:
    """Apply a sequence of patches to base DNA.

    Args:
        base_dna: Base LatentDNA
        patches: List of DNAPatch objects (should be sorted by step)

    Returns:
        Modified DNA with all patches applied
    """
    # Sort patches by step
    sorted_patches = sorted(patches, key=lambda p: p.step)

    # Apply each patch
    for patch in sorted_patches:
        for i in range(min(len(patch.delta), len(base_dna.data))):
            base_dna.data[i] += patch.delta[i]

    return base_dna


def discover_patches(patch_dir: str, up_to_step: Optional[int] = None) -> List[DNAPatch]:
    """Discover all patches in a directory, optionally up to a specific step.

    Args:
        patch_dir: Directory containing step_XXXXX.patch.dna files
        up_to_step: Optional step limit (inclusive)

    Returns:
        List of DNAPatch objects sorted by step
    """
    patch_dir = Path(patch_dir)
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    patch_files = sorted(patch_dir.glob("step_*.patch.dna"))
    patches = []

    for patch_file in patch_files:
        patch = DNAPatch.load(str(patch_file))

        if up_to_step is None or patch.step <= up_to_step:
            patches.append(patch)

    return sorted(patches, key=lambda p: p.step)


def get_last_checkpoint_step(patch_dir: str) -> int:
    """Get the last training step with a checkpoint.

    Args:
        patch_dir: Directory containing step_XXXXX.patch.dna files

    Returns:
        Last step number, or 0 if no checkpoints
    """
    patch_dir = Path(patch_dir)
    if not patch_dir.exists():
        return 0

    patch_files = list(patch_dir.glob("step_*.patch.dna"))
    if not patch_files:
        return 0

    patches = [DNAPatch.load(str(f)) for f in patch_files]
    return max(p.step for p in patches) if patches else 0


class EvolutionaryTrainingContext:
    """Context manager for evolutionary training sessions.

    Usage:
        with EvolutionaryTrainingContext("dna/base.dna", "checkpoints/") as ctx:
            # ctx.arch_path, ctx.dna, ctx.patches, ctx.resume_step available
            # harness = BRIANHarness(ctx.arch_path, ...)
            # harness.train_and_evolve(steps=10000)
            # Checkpoints saved to ctx.checkpoint_dir automatically
    """

    def __init__(self, dna_path: str, checkpoint_dir: Optional[str] = None):
        self.dna_path = Path(dna_path)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.dna_path.parent / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = None
        self.arch_path = None
        self.dna = None
        self.patches = None
        self.resume_step = None

    def __enter__(self):
        """Load DNA and patches on context entry."""
        self.config = init_evolution(str(self.dna_path), str(self.checkpoint_dir))
        self.arch_path = self.config["arch_path"]
        self.dna = self.config["base_dna"]
        self.patches = self.config["patches_applied"]
        self.resume_step = self.config["resume_step"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on context exit."""
        pass

    def save_checkpoint(self, step: int, mutations: List[dict]) -> None:
        """Save mutations as a checkpoint patch.

        Args:
            step: Training step
            mutations: List of mutation dicts (with kind, target, delta, metadata)
        """
        for mutation in mutations:
            patch = DNAPatch(
                version="1.0",
                step=step,
                kind=mutation.get("kind", "node_mutation"),
                target=mutation.get("target", "unknown"),
                delta=mutation.get("delta", []),
                metadata=mutation.get("metadata", {}),
            )
            patch_file = self.checkpoint_dir / f"step_{step:05d}.patch.dna"
            patch.save(str(patch_file))
