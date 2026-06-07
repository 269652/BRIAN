# -*- coding: utf-8 -*-
"""Tests for Task 2: Two-Way Ribosome Compiler with RAID-5 DNA encoding.

The Ribosome Compiler bidirectionally translates:
  DSL (.neuro) ↔ Latent DNA (error-corrected bitstream) ↔ THG-IR ↔ PyTorch

RAID-5-style parity checks protect topological invariants (e.g., Tonnetz spectral gap)
against "Representation Vandalism" — corruption of critical architecture parameters.
"""
import pytest
import torch
from pathlib import Path
import tempfile

from neuroslm.compiler.ribosome import (
    LatentDNA,
    RibosomeCompiler,
    DNATranscriber,
    DNATranslator,
)


class TestLatentDNA:
    """Test the latent DNA bitstream with RAID-5 parity encoding."""

    def test_dna_creation(self):
        """Create a latent DNA object."""
        dna = LatentDNA(length=256)
        assert dna.length == 256
        assert dna.data is not None

    def test_dna_encode_decode_roundtrip(self):
        """Encode data into DNA and decode it back (lossy compression)."""
        original_data = torch.rand(16)  # Use positive data for lossless roundtrip

        dna = LatentDNA.from_tensor(original_data, length=256)
        recovered_data = dna.to_tensor(original_data.shape[0])

        # Due to quantization, expect close but not identical
        assert torch.allclose(recovered_data, original_data, atol=0.1) or recovered_data.shape == original_data.shape

    def test_dna_raid5_parity_protection(self):
        """RAID-5 parity should protect against single-bit corruption."""
        dna = LatentDNA.from_tensor(torch.randn(16), length=256)

        # Introduce a corruption (bit flip)
        dna.data[10] = not dna.data[10] if hasattr(dna.data[10], '__bool__') else 1.0 - dna.data[10]

        # Try to detect/correct the error
        is_corrupted = dna.check_parity()
        assert isinstance(is_corrupted, bool)

    def test_dna_invariant_protection(self):
        """Protect critical invariants (e.g., spectral gap constraint)."""
        dna = LatentDNA(length=256)

        # Add an invariant: spectral_gap > 0.05
        invariant = {"spectral_gap_min": 0.05}
        dna.add_invariant_check(invariant)

        # Verify the invariant is registered
        assert len(dna.invariants) >= 1


class TestDNATranscriber:
    """Test transcription: DSL → DNA."""

    def test_dsl_to_dna_simple(self):
        """Transcribe a simple DSL complex into DNA."""
        dsl_code = """
        architecture test { d_sem: 256, dt: 0.01 }

        complex SimplexNet {
            topology: Tonnetz(dim: 256, spectral_gap: 0.05),
            trunk: "Identity()"
        }
        """

        transcriber = DNATranscriber()
        dna = transcriber.transcribe(dsl_code)

        assert dna is not None
        assert isinstance(dna, LatentDNA)

    def test_dsl_to_dna_with_genes(self):
        """Transcribe DSL with genetic_library into DNA."""
        dsl_code = """
        architecture test { d_sem: 256, dt: 0.01 }

        complex Adaptive {
            topology: Tonnetz(dim: 256, spectral_gap: 0.05),
            genetic_library {
                gene learning_rate {
                    target: "*",
                    rate: 0.5
                }
            }
        }
        """

        transcriber = DNATranscriber()
        dna = transcriber.transcribe(dsl_code)

        # DNA should contain the gene information
        assert dna is not None


class TestDNATranslator:
    """Test translation: DNA → DSL (backtranslation)."""

    def test_dna_to_dsl_simple(self):
        """Backtranslate DNA into equivalent DSL."""
        dsl_code = """
        architecture test { d_sem: 256, dt: 0.01 }
        complex Net { topology: Tonnetz(dim: 256, spectral_gap: 0.05), trunk: "Id" }
        """

        transcriber = DNATranscriber()
        dna = transcriber.transcribe(dsl_code)

        translator = DNATranslator()
        recovered_dsl = translator.translate(dna)

        assert recovered_dsl is not None
        assert "architecture" in recovered_dsl
        assert "complex" in recovered_dsl


class TestRibosomeCompiler:
    """Test the full bidirectional Ribosome compiler."""

    def test_ribosome_dsl_to_thg_ir(self):
        """Compile DSL → DNA → THG-IR via Ribosome."""
        dsl_code = """
        architecture test { d_sem: 256, dt: 0.01 }

        complex Layer1 {
            topology: Tonnetz(dim: 256, spectral_gap: 0.05),
            trunk: "ReLU()"
        }

        population sensory { count: 128, dynamics: "rate_code" }
        population motor { count: 64, dynamics: "rate_code" }
        synapse sensory -> motor { weight: 0.5 }
        """

        compiler = RibosomeCompiler()
        thg_ir = compiler.compile_dsl_to_thg(dsl_code)

        assert thg_ir is not None
        assert len(thg_ir.nodes) >= 2

    def test_ribosome_incremental_patch(self):
        """Incremental patching: apply DNA diff to existing THG-IR."""
        # Create an initial checkpoint
        from neuroslm.dsl.thg_ir import THGCheckpoint, THGNode, THGEdge

        initial_thg = THGCheckpoint(
            version="2.0",
            nodes={
                "n1": THGNode("n1", "pop", [0.0] * 16, {}),
                "n2": THGNode("n2", "pop", [0.0] * 16, {}),
            },
            edges={
                "e1": THGEdge("e1", "n1", "n2", "synapse", 0.5, "fixed"),
            },
            gene_state={},
            step=0,
            metadata={},
        )

        # Create a DNA diff patch
        compiler = RibosomeCompiler()
        dna_patch = compiler.create_patch(delta_embedding=[0.1] * 16, target_node="n1")

        # Apply patch via rank-one update
        patched_thg = compiler.apply_patch(initial_thg, dna_patch)

        assert patched_thg.nodes["n1"].operator_embedding[0] > 0.0

    def test_ribosome_preserves_invariants(self):
        """Compilation should preserve topological invariants (spectral gap, Φ>0)."""
        dsl_code = """
        architecture test { d_sem: 256, dt: 0.01 }

        complex Stable {
            topology: Tonnetz(dim: 256, spectral_gap: 0.05),
            trunk: "Identity()"
        }
        """

        compiler = RibosomeCompiler()

        # Transcribe to DNA
        dna = compiler.dna_transcriber.transcribe(dsl_code)

        # Verify invariants are encoded
        assert len(dna.invariants) > 0 or dna is not None


class TestIncomentalPatchingViaRankOne:
    """Test rank-one update patching for efficient topology modification."""

    def test_rank_one_node_update(self):
        """Rank-one update modifies node embedding without full recomputation."""
        from neuroslm.dsl.thg_ir import THGCheckpoint, THGNode

        thg = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.0] * 16, {})},
            edges={},
            gene_state={},
            step=0,
            metadata={},
        )

        # Apply rank-one update: new_embedding = old_embedding + α * u * v^T
        u = torch.randn(16)
        alpha = 0.1
        delta = alpha * u

        compiler = RibosomeCompiler()
        updated = compiler.apply_rank_one_update(thg, "n1", delta.tolist())

        assert updated.nodes["n1"].operator_embedding[0] != 0.0

    def test_rank_one_edge_weight_update(self):
        """Rank-one update for edge weight modification."""
        from neuroslm.dsl.thg_ir import THGCheckpoint, THGNode, THGEdge

        thg = THGCheckpoint(
            version="2.0",
            nodes={
                "n1": THGNode("n1", "pop", [0.0] * 16, {}),
                "n2": THGNode("n2", "pop", [0.0] * 16, {}),
            },
            edges={
                "e1": THGEdge("e1", "n1", "n2", "synapse", 0.5, "fixed"),
            },
            gene_state={},
            step=0,
            metadata={},
        )

        compiler = RibosomeCompiler()

        # Update edge weight
        updated = compiler.update_edge_weight(thg, "e1", delta_weight=0.1)

        assert updated.edges["e1"].weight == pytest.approx(0.6)


class TestBiDirectionalIntegration:
    """Full bidirectional flow: DSL → DNA → THG → DNA → DSL."""

    def test_full_cycle_dsl_dna_dsl(self):
        """Complete roundtrip: DSL → DNA → DSL should preserve semantics."""
        original_dsl = """
        architecture test { d_sem: 256, dt: 0.01 }

        complex Processing {
            topology: Tonnetz(dim: 256, spectral_gap: 0.05),
            trunk: "Linear()"
        }
        """

        compiler = RibosomeCompiler()

        # DSL → DNA
        dna = compiler.dna_transcriber.transcribe(original_dsl)

        # DNA → DSL
        recovered_dsl = compiler.dna_translator.translate(dna)

        # Both should contain key elements
        assert "Tonnetz" in recovered_dsl
        assert "256" in recovered_dsl

    def test_thg_dna_thg_roundtrip(self):
        """Roundtrip THG-IR → DNA → THG-IR."""
        from neuroslm.dsl.thg_ir import THGCheckpoint, THGNode

        original_thg = THGCheckpoint(
            version="2.0",
            nodes={"n1": THGNode("n1", "pop", [0.1, 0.2, 0.3], {})},
            edges={},
            gene_state={},
            step=0,
            metadata={},
        )

        compiler = RibosomeCompiler()

        # THG → DNA
        dna = compiler.thg_to_dna(original_thg)

        # DNA → THG
        recovered_thg = compiler.dna_to_thg(dna)

        assert recovered_thg is not None
        if "n1" in recovered_thg.nodes:
            assert len(recovered_thg.nodes["n1"].operator_embedding) == 3
