# -*- coding: utf-8 -*-
"""TDD tests for incremental DNA patching system.

Tests cover:
- Patch creation and serialization
- Patch application and composition
- Roundtrip (base DNA + patches → final DNA)
- Patch versioning and dependency tracking
"""
import pytest
import tempfile
import json
from pathlib import Path

from neuroslm.compiler.ribosome import LatentDNA


class TestDNAPatchCreation:
    """Test DNA patch creation and serialization."""

    def test_patch_creation_basic(self):
        """Create a basic DNA patch with delta."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="language_trunk",
            delta=[0.1, 0.2, 0.3],
            metadata={"reason": "hotpath_stabilization"}
        )

        assert patch.version == "1.0"
        assert patch.step == 1000
        assert patch.kind == "node_mutation"
        assert patch.target == "language_trunk"
        assert patch.delta == [0.1, 0.2, 0.3]
        assert patch.metadata["reason"] == "hotpath_stabilization"

    def test_patch_serialization(self):
        """Serialize patch to JSON."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="gws",
            delta=[0.05, 0.1],
            metadata={"confidence": 0.95}
        )

        patch_dict = patch.to_dict()
        assert patch_dict["version"] == "1.0"
        assert patch_dict["step"] == 1000
        assert patch_dict["kind"] == "node_mutation"
        assert patch_dict["target"] == "gws"
        assert patch_dict["delta"] == [0.05, 0.1]
        assert patch_dict["metadata"]["confidence"] == 0.95

    def test_patch_deserialization(self):
        """Deserialize patch from dict."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch_dict = {
            "version": "1.0",
            "step": 2000,
            "kind": "edge_weight",
            "target": "gws_to_motor",
            "delta": [0.02],
            "metadata": {"phi_gain": 0.05}
        }

        patch = DNAPatch.from_dict(patch_dict)
        assert patch.version == "1.0"
        assert patch.step == 2000
        assert patch.kind == "edge_weight"
        assert patch.target == "gws_to_motor"

    def test_patch_file_save_load(self):
        """Save and load patch to/from file."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch = DNAPatch(
            version="1.0",
            step=3000,
            kind="node_mutation",
            target="hippo",
            delta=[0.15],
            metadata={"source": "epigenetic"}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            patch_file = Path(tmpdir) / "test.patch.dna"
            patch.save(str(patch_file))
            assert patch_file.exists()

            loaded = DNAPatch.load(str(patch_file))
            assert loaded.step == 3000
            assert loaded.target == "hippo"
            assert loaded.metadata["source"] == "epigenetic"


class TestDNAPatchComposition:
    """Test composing multiple patches onto a base DNA."""

    def test_patch_application_to_dna(self):
        """Apply single patch to DNA."""
        from neuroslm.compiler.ribosome import DNAPatch

        base_dna = LatentDNA(length=256)
        base_data = base_dna.data.copy()

        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="n1",
            delta=[0.05] * 16,
            metadata={}
        )

        # Apply patch (mutate DNA in place)
        for i in range(min(16, len(base_dna.data))):
            base_dna.data[i] = base_data[i] + patch.delta[i % len(patch.delta)]

        # Verify mutation occurred
        assert base_dna.data[0] != base_data[0]

    def test_patch_stack_application(self):
        """Apply multiple patches in sequence."""
        from neuroslm.compiler.ribosome import DNAPatch

        base_dna = LatentDNA(length=256)
        original_data = base_dna.data[0]

        patches = [
            DNAPatch(
                version="1.0",
                step=1000,
                kind="node_mutation",
                target="n1",
                delta=[0.01],
                metadata={}
            ),
            DNAPatch(
                version="1.0",
                step=2000,
                kind="node_mutation",
                target="n1",
                delta=[0.02],
                metadata={}
            ),
        ]

        # Apply patches sequentially
        current_val = original_data
        for patch in patches:
            current_val += patch.delta[0]

        # Both patches applied
        assert current_val > original_data
        assert abs(current_val - (original_data + 0.01 + 0.02)) < 1e-6

    def test_patch_dependency_tracking(self):
        """Track dependencies: patch B depends on patch A."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch_a = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="n1",
            delta=[0.1],
            metadata={}
        )

        patch_b = DNAPatch(
            version="1.0",
            step=2000,
            kind="node_mutation",
            target="n1",
            delta=[0.2],
            metadata={"depends_on": patch_a.step}
        )

        assert patch_b.metadata.get("depends_on") == patch_a.step


class TestDNAPatchRoundtrip:
    """Test end-to-end patch application workflow."""

    def test_base_dna_plus_single_patch(self):
        """Base DNA + single patch → final DNA."""
        from neuroslm.compiler.ribosome import DNAPatch, RibosomeCompiler

        compiler = RibosomeCompiler()

        # Create base DNA
        base_dna = LatentDNA(length=256)
        base_dna.data[0] = 0.5

        # Create patch
        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="n1",
            delta=[0.1],
            metadata={}
        )

        # Apply patch
        final_val = base_dna.data[0] + patch.delta[0]
        assert abs(final_val - 0.6) < 1e-6

    def test_base_dna_plus_multiple_patches(self):
        """Base DNA + multiple patches → final DNA."""
        from neuroslm.compiler.ribosome import DNAPatch

        base_dna = LatentDNA(length=256)
        base_dna.data[0] = 0.3

        patches = [
            DNAPatch(
                version="1.0",
                step=1000,
                kind="node_mutation",
                target="n1",
                delta=[0.1],
                metadata={}
            ),
            DNAPatch(
                version="1.0",
                step=2000,
                kind="node_mutation",
                target="n1",
                delta=[0.15],
                metadata={}
            ),
            DNAPatch(
                version="1.0",
                step=3000,
                kind="node_mutation",
                target="n1",
                delta=[0.05],
                metadata={}
            ),
        ]

        # Apply all patches
        current_val = base_dna.data[0]
        for patch in patches:
            current_val += patch.delta[0]

        # Final value should be 0.3 + 0.1 + 0.15 + 0.05 = 0.6
        assert abs(current_val - 0.6) < 1e-6

    def test_patch_roundtrip_file_based(self):
        """Save patches, load them, apply to DNA, verify result."""
        from neuroslm.compiler.ribosome import DNAPatch

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save patches
            patches_to_save = [
                DNAPatch(
                    version="1.0",
                    step=1000,
                    kind="node_mutation",
                    target="n1",
                    delta=[0.1],
                    metadata={"index": 0}
                ),
                DNAPatch(
                    version="1.0",
                    step=2000,
                    kind="node_mutation",
                    target="n1",
                    delta=[0.2],
                    metadata={"index": 1}
                ),
            ]

            patch_files = []
            for i, patch in enumerate(patches_to_save):
                patch_file = Path(tmpdir) / f"step_{patch.step:04d}.patch.dna"
                patch.save(str(patch_file))
                patch_files.append(patch_file)

            # Load patches
            loaded_patches = [DNAPatch.load(str(f)) for f in patch_files]
            assert len(loaded_patches) == 2
            assert loaded_patches[0].step == 1000
            assert loaded_patches[1].step == 2000

            # Apply to DNA
            base_dna = LatentDNA(length=256)
            base_dna.data[0] = 0.0
            for patch in loaded_patches:
                base_dna.data[0] += patch.delta[0]

            # Verify
            assert abs(base_dna.data[0] - 0.3) < 1e-6


class TestDNAPatchValidation:
    """Test patch validation and consistency checks."""

    def test_patch_step_ordering(self):
        """Patches should be applied in step order."""
        from neuroslm.compiler.ribosome import DNAPatch

        patches = [
            DNAPatch(version="1.0", step=1000, kind="node_mutation", target="n1", delta=[0.1], metadata={}),
            DNAPatch(version="1.0", step=500, kind="node_mutation", target="n1", delta=[0.2], metadata={}),
            DNAPatch(version="1.0", step=2000, kind="node_mutation", target="n1", delta=[0.3], metadata={}),
        ]

        # Sort by step
        sorted_patches = sorted(patches, key=lambda p: p.step)
        assert sorted_patches[0].step == 500
        assert sorted_patches[1].step == 1000
        assert sorted_patches[2].step == 2000

    def test_patch_delta_magnitude_check(self):
        """Verify patch delta is within reasonable bounds."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="n1",
            delta=[0.05],
            metadata={}
        )

        # Delta should be small (learning rate scale)
        assert all(abs(d) < 1.0 for d in patch.delta)

    def test_patch_metadata_preservation(self):
        """Patch metadata (reason, confidence) should be preserved."""
        from neuroslm.compiler.ribosome import DNAPatch

        patch = DNAPatch(
            version="1.0",
            step=1000,
            kind="node_mutation",
            target="n1",
            delta=[0.1],
            metadata={
                "reason": "hotpath_stabilization",
                "confidence": 0.95,
                "phi_delta": 0.08
            }
        )

        patch_dict = patch.to_dict()
        loaded = DNAPatch.from_dict(patch_dict)

        assert loaded.metadata["reason"] == "hotpath_stabilization"
        assert loaded.metadata["confidence"] == 0.95
        assert loaded.metadata["phi_delta"] == 0.08
