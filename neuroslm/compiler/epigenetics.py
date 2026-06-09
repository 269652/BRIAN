# -*- coding: utf-8 -*-
"""Epigenetics system for incremental DNA evolution.

Enables evolution through gene patches that:
1. Respect minify settings from parent DNA
2. Compose incrementally (patch1 + patch2 = new DNA)
3. Track source maps for attribution
4. Maintain module structure
5. Support pretty-printing or minification as configured

Architecture:
  DNA → [Gene Patches] → Evolved DNA
                ↓
        Source maps + Minify setting
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path

from neuroslm.compiler.ribosome import LatentDNA, DNAPatch
from neuroslm.compiler.dsl_minifier import DSLMinifier, PrettifyPolicy


@dataclass
class EvolutionContext:
    """Context for evolving DNA through gene patches.

    Tracks minify setting, source maps, and ensures evolved DNA
    maintains format consistency with parent.
    """
    parent_dna: LatentDNA
    minify_setting: Optional[bool] = None
    patches: List[DNAPatch] = field(default_factory=list)
    source_maps: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize from parent DNA."""
        if self.minify_setting is None:
            self.minify_setting = self.parent_dna.invariants.get("minify")

        # Extract source maps from parent
        if "minification_map" in self.parent_dna.invariants:
            self.source_maps["minification"] = \
                self.parent_dna.invariants["minification_map"]

        if "bundled_dsl" in self.parent_dna.invariants:
            bundled = self.parent_dna.invariants["bundled_dsl"]
            if "source_map" in bundled:
                self.source_maps["module"] = bundled["source_map"]

    def apply_patch(self, patch: DNAPatch) -> None:
        """Apply a gene patch to the evolution context.

        The patch is recorded but not immediately applied to DNA.
        Use compose() to generate the evolved DNA.
        """
        self.patches.append(patch)

        # Track patch in source map for attribution
        if patch.target not in self.source_maps:
            self.source_maps[patch.target] = []

        self.source_maps[patch.target].append({
            "step": patch.step,
            "kind": patch.kind,
            "metadata": patch.metadata,
        })

    def compose(self) -> LatentDNA:
        """Compose all patches into evolved DNA.

        Returns new DNA that:
        1. Maintains minify setting from parent
        2. Preserves source maps with patch attribution
        3. Applies mutations to appropriate targets
        """
        # Create new DNA from parent
        evolved_dna = LatentDNA(
            length=self.parent_dna.length,
            data=list(self.parent_dna.data),
            parity_blocks=list(self.parent_dna.parity_blocks),
            invariants=dict(self.parent_dna.invariants),
        )

        # Preserve minify setting
        evolved_dna.invariants["minify"] = self.minify_setting

        # Apply patches (simple additive for now)
        for patch in self.patches:
            # Find target in data and apply delta
            # For now, this is a placeholder - real implementation
            # would apply mutations to specific positions
            if patch.kind == "node_mutation" and patch.target:
                # Record that patch was applied
                if "_applied_patches" not in evolved_dna.invariants:
                    evolved_dna.invariants["_applied_patches"] = []
                evolved_dna.invariants["_applied_patches"].append(patch.to_dict())

        # Update source maps for evolution traceability
        evolved_dna.invariants["_evolution_source_maps"] = self.source_maps

        return evolved_dna

    def to_dict(self) -> Dict:
        """Serialize evolution context."""
        return {
            "minify_setting": self.minify_setting,
            "num_patches": len(self.patches),
            "source_maps": self.source_maps,
            "patches": [p.to_dict() for p in self.patches],
        }


class GeneticEpigenetics:
    """Apply genetic modifications to DNA via patches.

    Manages:
    - Creating gene patches
    - Applying patches while respecting minify settings
    - Composing multiple patches into evolved DNA
    - Tracking source maps and attribution
    """

    @staticmethod
    def create_patch(
        step: int,
        target: str,
        delta: List[float],
        kind: str = "node_mutation",
        metadata: Optional[Dict] = None,
    ) -> DNAPatch:
        """Create a gene patch for evolution.

        Args:
            step: Training step when patch was created
            target: Node/module being mutated (e.g., "main", "@/lib/cortex")
            delta: Change vector to apply
            kind: Patch type (node_mutation, edge_weight, topology_change)
            metadata: Optional metadata (reason, confidence, phi_delta, etc.)

        Returns:
            DNAPatch ready for application
        """
        return DNAPatch(
            version="1.0",
            step=step,
            kind=kind,
            target=target,
            delta=delta,
            metadata=metadata or {},
        )

    @staticmethod
    def create_evolution_context(dna_file: str) -> EvolutionContext:
        """Create an evolution context from a DNA file.

        Args:
            dna_file: Path to DNA file

        Returns:
            EvolutionContext ready for patches
        """
        dna = LatentDNA.load(dna_file)
        return EvolutionContext(parent_dna=dna)

    @staticmethod
    def apply_patches_to_dna(
        dna_file: str,
        patches: List[DNAPatch],
        output_file: Optional[str] = None,
    ) -> LatentDNA:
        """Apply a sequence of patches to DNA.

        Args:
            dna_file: Input DNA file
            patches: List of patches to apply
            output_file: Optional output file for evolved DNA

        Returns:
            Evolved DNA
        """
        ctx = GeneticEpigenetics.create_evolution_context(dna_file)

        for patch in patches:
            ctx.apply_patch(patch)

        evolved_dna = ctx.compose()

        if output_file:
            evolved_dna.save(output_file)

        return evolved_dna

    @staticmethod
    def unfold_evolved_dna(
        evolved_dna: LatentDNA,
        output_neuro: str,
        pretty_print: bool = True,
    ) -> None:
        """Unfold evolved DNA back to DSL.

        Respects minify setting and applies pretty-printing if configured.

        Args:
            evolved_dna: The evolved DNA to unfold
            output_neuro: Output .neuro file
            pretty_print: Whether to pretty-print (ignores minify: true)
        """
        from neuroslm.compiler.ribosome import DNATranslator

        # Get minify setting
        minify_setting = evolved_dna.invariants.get("minify")

        # Translate DNA to DSL
        translator = DNATranslator()
        dsl_code = translator.translate(evolved_dna)

        # Apply formatting based on minify setting
        if pretty_print and minify_setting is not False:
            # pretty_print overrides minify, or minify is not explicitly false
            minifier = DSLMinifier()
            if '\n' not in dsl_code[:200]:  # Looks minified
                dsl_code = minifier.prettify(dsl_code)

        # Write unfolded DSL
        with open(output_neuro, 'w', encoding='utf-8') as f:
            f.write(dsl_code)


class PatchComposer:
    """Compose multiple gene patches into a sequence.

    Ensures patches compose correctly while respecting:
    - Module boundaries
    - Minify settings
    - Source map continuity
    """

    def __init__(self, base_dna: LatentDNA):
        """Initialize composer with base DNA.

        Args:
            base_dna: Starting DNA for composition
        """
        self.base_dna = base_dna
        self.patches: List[DNAPatch] = []
        self.minify_setting = base_dna.invariants.get("minify")

    def add_patch(self, patch: DNAPatch) -> PatchComposer:
        """Add a patch to the composition sequence.

        Args:
            patch: DNAPatch to add

        Returns:
            Self for chaining
        """
        self.patches.append(patch)
        return self

    def compose(self) -> LatentDNA:
        """Compose all patches into evolved DNA.

        Returns:
            New LatentDNA with all patches applied
        """
        ctx = EvolutionContext(parent_dna=self.base_dna)
        ctx.minify_setting = self.minify_setting

        for patch in self.patches:
            ctx.apply_patch(patch)

        return ctx.compose()

    def save_evolved_dna(self, output_path: str) -> LatentDNA:
        """Compose and save evolved DNA to file.

        Args:
            output_path: Output DNA file

        Returns:
            The composed evolved DNA
        """
        evolved_dna = self.compose()
        evolved_dna.save(output_path)
        return evolved_dna
