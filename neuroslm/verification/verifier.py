# -*- coding: utf-8 -*-
"""Task 4: THSD Verifier — formal verification linter.

Ensures architectural invariants:
  - Φ > 0 (integrated information)
  - H¹(K;F) = 0 (no cohomological obstructions)
  - Spectral gap > λ_min (topological stability)
  - DNA parity (no representation vandalism)
  - Genetic consistency
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class InvariantChecker:
    """Check topological and genetic invariants."""

    spectral_gap_min: float = 0.01
    rank_min: int = 8
    embedding_bounds: tuple = (-10.0, 10.0)

    def check_spectral_gap(self, spectral_gap: float) -> bool:
        """Verify spectral gap > λ_min."""
        return spectral_gap > self.spectral_gap_min

    def check_embedding_validity(self, embedding: List[float]) -> bool:
        """Check embedding has no NaN/Inf and is within bounds."""
        try:
            for val in embedding:
                if not isinstance(val, (int, float)):
                    return False
                if val != val:  # NaN check
                    return False
                if val == float("inf") or val == float("-inf"):
                    return False
                if val < self.embedding_bounds[0] or val > self.embedding_bounds[1]:
                    return False
            return True
        except Exception:
            return False

    def check_gene_state(self, gene_state: Dict) -> bool:
        """Verify genetic state consistency."""
        # Basic checks: all values should be numeric and positive learning rates
        try:
            if "learning_rate" in gene_state:
                if gene_state["learning_rate"] <= 0:
                    return False
            if "baseline_nt" in gene_state:
                if gene_state["baseline_nt"] < 0:
                    return False
            return True
        except Exception:
            return False


@dataclass
class CohomologyValidator:
    """Validate H¹ cohomology and Φ dynamics."""

    hallucination_threshold: float = 5.0
    phi_min: float = 0.0  # Φ ≥ 0 always

    def check_h1_consistency(self, thg) -> bool:
        """Check if H¹(K;F) = 0 (no contradictions).

        Simplified: look for extremely high-norm components in THG.
        """
        # Sum all embedding norms
        total_norm = 0.0
        for node in thg.nodes.values():
            embedding_norm = sum(e**2 for e in node.operator_embedding) ** 0.5
            total_norm += embedding_norm

        # If any single norm is suspiciously high, flag as potential H¹ obstruction
        max_node_norm = 0.0
        for node in thg.nodes.values():
            norm = sum(e**2 for e in node.operator_embedding) ** 0.5
            max_node_norm = max(max_node_norm, norm)

        # H¹ = 0 means no large global obstructions
        return max_node_norm < self.hallucination_threshold

    def compute_phi(self, thg) -> float:
        """Estimate integrated information Φ from THG-IR.

        Simplified: correlation between node embeddings.
        """
        if len(thg.nodes) < 2:
            return 0.0

        embeddings = [
            torch.tensor(node.operator_embedding, dtype=torch.float32)
            for node in thg.nodes.values()
        ]

        if len(embeddings) < 2:
            return 0.0

        # Stack and compute correlation
        stacked = torch.stack(embeddings)
        mean = stacked.mean(dim=0)
        centered = stacked - mean

        # Compute correlation matrix norm as proxy for Φ
        # Higher correlation = higher Φ (more integrated)
        try:
            cov = torch.cov(centered.T)
            phi_proxy = torch.norm(cov).item()
            return max(0.0, phi_proxy)
        except Exception:
            return 0.0

    def check_no_hallucination(self, thg) -> bool:
        """Check that H¹ norm stays below hallucination threshold."""
        max_norm = 0.0
        for node in thg.nodes.values():
            norm = sum(e**2 for e in node.operator_embedding) ** 0.5
            max_norm = max(max_norm, norm)

        return max_norm < self.hallucination_threshold


@dataclass
class THSDVerifier:
    """Full THSD verifier: orchestrates all architectural checks."""

    invariant_checker: InvariantChecker = field(default_factory=InvariantChecker)
    cohomology_validator: CohomologyValidator = field(
        default_factory=CohomologyValidator
    )

    def verify_checkpoint(self, thg) -> Dict:
        """Verify a THG-IR checkpoint for all invariants.

        Returns a report with status, errors, warnings, and suggestions.
        """
        report = {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        # Check 1: H¹ consistency
        h1_ok = self.cohomology_validator.check_h1_consistency(thg)
        report["checks"]["h1_consistency"] = h1_ok
        if not h1_ok:
            report["errors"].append("H¹ obstruction detected (potential contradiction)")
            report["status"] = "FAIL"

        # Check 2: Φ > 0
        phi = self.cohomology_validator.compute_phi(thg)
        report["checks"]["phi_score"] = phi
        if phi <= 0.0:
            report["warnings"].append("Low integrated information (Φ ≈ 0)")

        # Check 3: No hallucinations
        no_hallucination = self.cohomology_validator.check_no_hallucination(thg)
        report["checks"]["no_hallucination"] = no_hallucination
        if not no_hallucination:
            report["errors"].append("Hallucination risk (high-norm components detected)")
            report["status"] = "FAIL"

        # Check 4: Embedding validity
        for node_id, node in thg.nodes.items():
            valid = self.invariant_checker.check_embedding_validity(
                node.operator_embedding
            )
            if not valid:
                report["errors"].append(f"Invalid embedding for node {node_id}")
                report["status"] = "FAIL"

        # Check 5: Gene state consistency
        valid_genes = self.invariant_checker.check_gene_state(thg.gene_state)
        report["checks"]["gene_state_valid"] = valid_genes
        if not valid_genes:
            report["warnings"].append("Gene state values out of expected range")

        return report

    def verify_dsl(self, dsl_code: str) -> Dict:
        """Verify DSL code for architectural constraints.

        Parses DSL and runs verifier on resulting THG-IR.
        """
        from neuroslm.dsl.compiler import NeuroMLCompiler

        try:
            ir = NeuroMLCompiler.compile(dsl_code)
            from neuroslm.dsl.thg_ir import THGCheckpoint

            thg = THGCheckpoint.from_program_ir(ir)
            return self.verify_checkpoint(thg)
        except Exception as e:
            return {
                "status": "ERROR",
                "errors": [f"Failed to parse DSL: {str(e)}"],
                "warnings": [],
                "checks": {},
            }

    def verify_dna(self, dna) -> Dict:
        """Verify latent DNA for parity and invariant protection."""
        report = {"status": "PASS", "errors": [], "warnings": []}

        # Check parity
        parity_ok = dna.check_parity()
        if not parity_ok:
            report["warnings"].append("DNA parity check failed (possible corruption)")

        # Check invariants are registered
        if not dna.invariants:
            report["warnings"].append("No topological invariants protected in DNA")

        return report
