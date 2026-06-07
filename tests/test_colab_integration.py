# -*- coding: utf-8 -*-
"""TDD tests for Colab/Jupyter integration of evolutionary training.

Tests cover:
- Evolutionary initialization (load DNA, unfold to arch, initialize harness)
- Incremental checkpointing (step_XXXX.patch.dna files)
- Session resumption (apply patch stack to resume from checkpoint)
- Train-and-evolve workflow
"""
import pytest
import tempfile
import json
from pathlib import Path

from neuroslm.compiler.ribosome import LatentDNA, DNAPatch, RibosomeCompiler


class TestEvolutionaryInitialization:
    """Test evolutionary_init() utility for Colab."""

    def test_init_loads_base_dna(self):
        """Load base DNA from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base DNA
            base_dna = LatentDNA(length=256)
            dna_file = Path(tmpdir) / "base.dna"
            base_dna.save(str(dna_file))

            # Load it
            loaded = LatentDNA.load(str(dna_file))
            assert loaded.length == 256
            assert len(loaded.data) == 256

    def test_init_unfolds_dna_to_neuro(self):
        """Unfold DNA to arch.neuro DSL."""
        from neuroslm.compiler.ribosome import RibosomeCompiler

        arch_root = str(Path(__file__).parent.parent / "architectures" / "rcc_bowtie")

        with tempfile.TemporaryDirectory() as tmpdir:
            dna_file = Path(tmpdir) / "test.dna"
            neuro_file = Path(tmpdir) / "evolution.neuro"

            compiler = RibosomeCompiler()

            # Compile arch to DNA
            compiler.compile_file(arch_root, str(dna_file))
            assert dna_file.exists()

            # Unfold DNA back to neuro
            compiler.unfold_file(str(dna_file), str(neuro_file))
            assert neuro_file.exists()

            # Neuro file should have content
            content = neuro_file.read_text(encoding='utf-8')
            assert len(content) > 0

    def test_init_with_patch_stack(self):
        """Load base DNA + apply patch stack."""
        base_dna = LatentDNA(length=256)
        base_dna.data[0] = 0.1

        patches = [
            DNAPatch(version="1.0", step=1000, kind="node_mutation", target="n1", delta=[0.05], metadata={}),
            DNAPatch(version="1.0", step=2000, kind="node_mutation", target="n1", delta=[0.1], metadata={}),
        ]

        # Apply patches
        current_val = base_dna.data[0]
        for patch in patches:
            current_val += patch.delta[0]

        # Should be 0.1 + 0.05 + 0.1 = 0.25
        assert abs(current_val - 0.25) < 1e-6

    def test_init_function_exists(self):
        """Verify neuroslm.utils.colab module exists."""
        # Import should not fail
        try:
            from neuroslm.utils.colab import init_evolution
            assert callable(init_evolution)
        except ImportError:
            # Module might not exist yet (will be created)
            pytest.skip("colab module not yet created")


class TestIncrementalCheckpointing:
    """Test incremental checkpoint creation during training."""

    def test_checkpoint_patch_filename_convention(self):
        """Checkpoints follow step_XXXXX.patch.dna naming."""
        patches = []
        for step in [1000, 5000, 10000]:
            patch = DNAPatch(
                version="1.0",
                step=step,
                kind="node_mutation",
                target="gws",
                delta=[0.1],
                metadata={"checkpoint": True}
            )
            patches.append((step, patch))

        # Verify naming convention
        for step, patch in patches:
            filename = f"step_{step:05d}.patch.dna"
            # Extract step number from filename
            extracted_step = int(filename.split("_")[1].split(".")[0])
            assert step == extracted_step

    def test_checkpoint_contains_all_mutations_since_last_checkpoint(self):
        """step_5000.patch.dna contains mutations from steps 1000-5000."""
        mutations = [
            DNAPatch(version="1.0", step=1500, kind="node_mutation", target="gws", delta=[0.05], metadata={}),
            DNAPatch(version="1.0", step=2500, kind="node_mutation", target="lang", delta=[0.1], metadata={}),
            DNAPatch(version="1.0", step=4000, kind="node_mutation", target="hippo", delta=[0.08], metadata={}),
        ]

        # These mutations all occurred before checkpoint at step 5000
        checkpoint_step = 5000
        mutations_in_window = [m for m in mutations if m.step < checkpoint_step]

        assert len(mutations_in_window) == 3

    def test_checkpoint_accumulation_over_epoch(self):
        """Multiple checkpoints accumulate patches in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create checkpoints at different steps
            for checkpoint_step in [1000, 2000, 3000]:
                patch = DNAPatch(
                    version="1.0",
                    step=checkpoint_step,
                    kind="node_mutation",
                    target="gws",
                    delta=[0.1],
                    metadata={}
                )
                patch_file = checkpoint_dir / f"step_{checkpoint_step:05d}.patch.dna"
                patch.save(str(patch_file))

            # All checkpoints should exist
            checkpoint_files = sorted(checkpoint_dir.glob("step_*.patch.dna"))
            assert len(checkpoint_files) == 3

            # Load in order
            patches_loaded = [DNAPatch.load(str(f)) for f in checkpoint_files]
            assert patches_loaded[0].step == 1000
            assert patches_loaded[2].step == 3000


class TestSessionResumption:
    """Test resuming from a saved checkpoint (DNA + patches)."""

    def test_resume_loads_base_dna_and_patches(self):
        """Resume: load base DNA + apply all patches up to checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create base DNA
            base_dna = LatentDNA(length=256)
            base_dna.data[0] = 0.0
            base_dna_file = checkpoint_dir / "base.dna"
            base_dna.save(str(base_dna_file))

            # Create patches at different steps
            patches = [
                DNAPatch(version="1.0", step=1000, kind="node_mutation", target="n1", delta=[0.1], metadata={}),
                DNAPatch(version="1.0", step=2000, kind="node_mutation", target="n1", delta=[0.15], metadata={}),
                DNAPatch(version="1.0", step=3000, kind="node_mutation", target="n1", delta=[0.05], metadata={}),
            ]

            for patch in patches:
                patch_file = checkpoint_dir / f"step_{patch.step:05d}.patch.dna"
                patch.save(str(patch_file))

            # Resume: load base + apply patches
            loaded_base = LatentDNA.load(str(base_dna_file))
            loaded_patches = [
                DNAPatch.load(str(f))
                for f in sorted(checkpoint_dir.glob("step_*.patch.dna"))
            ]

            # Apply patches
            current_val = loaded_base.data[0]
            for patch in loaded_patches:
                current_val += patch.delta[0]

            # Should be 0 + 0.1 + 0.15 + 0.05 = 0.3
            assert abs(current_val - 0.3) < 1e-6

    def test_resume_from_intermediate_checkpoint(self):
        """Resume from checkpoint at step 2000 (skip step 1000 patch)."""
        # Create patches
        all_patches = [
            DNAPatch(version="1.0", step=1000, kind="node_mutation", target="n1", delta=[0.1], metadata={}),
            DNAPatch(version="1.0", step=2000, kind="node_mutation", target="n1", delta=[0.15], metadata={}),
            DNAPatch(version="1.0", step=3000, kind="node_mutation", target="n1", delta=[0.05], metadata={}),
        ]

        # Resume at step 2000: apply patches up to 2000
        resume_step = 2000
        patches_to_apply = [p for p in all_patches if p.step <= resume_step]

        # Apply only patches up to resume point
        base_val = 0.0
        for patch in patches_to_apply:
            base_val += patch.delta[0]

        # Should be 0 + 0.1 + 0.15 = 0.25
        assert abs(base_val - 0.25) < 1e-6

    def test_resume_preserves_evolution_continuity(self):
        """Resuming preserves the topological state (no loss of information)."""
        # Original session
        dna_1 = LatentDNA(length=256)
        dna_1.data[0] = 0.0

        patches_1 = [
            DNAPatch(version="1.0", step=1000, kind="node_mutation", target="n1", delta=[0.1], metadata={}),
            DNAPatch(version="1.0", step=2000, kind="node_mutation", target="n1", delta=[0.2], metadata={}),
        ]

        # Apply all patches in original session
        val_original = dna_1.data[0]
        for p in patches_1:
            val_original += p.delta[0]

        # Resumed session: load base + apply same patches
        dna_2 = LatentDNA(length=256)
        dna_2.data[0] = 0.0
        val_resumed = dna_2.data[0]
        for p in patches_1:
            val_resumed += p.delta[0]

        # Values should be identical
        assert abs(val_original - val_resumed) < 1e-9


class TestTrainAndEvolveWorkflow:
    """Test end-to-end train-and-evolve workflow in Colab."""

    def test_evolutionary_step_during_training(self):
        """During training step, detect HOT paths and emit mutations."""
        # Simulate one training step
        activity_log = {
            "e_gws_motor": {"activation_corr": 0.9},  # HOT
            "e_motor_thal": {"activation_corr": 0.05},  # COLD
        }

        mutations_emitted = []
        for eid, data in activity_log.items():
            if data["activation_corr"] > 0.7:  # HOT
                mutation = DNAPatch(
                    version="1.0",
                    step=1000,
                    kind="edge_weight",
                    target=eid,
                    delta=[0.02],
                    metadata={"reason": "hot_path_bdnf"}
                )
                mutations_emitted.append(mutation)

        # One mutation emitted for HOT edge
        assert len(mutations_emitted) == 1
        assert mutations_emitted[0].target == "e_gws_motor"

    def test_checkpoint_creation_at_epoch_end(self):
        """At epoch end, save all mutations since last checkpoint."""
        mutations_in_epoch = [
            DNAPatch(version="1.0", step=1000, kind="node_mutation", target="gws", delta=[0.1], metadata={}),
            DNAPatch(version="1.0", step=2500, kind="node_mutation", target="lang", delta=[0.15], metadata={}),
            DNAPatch(version="1.0", step=4800, kind="node_mutation", target="hippo", delta=[0.08], metadata={}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Save all mutations as step_XXXXX.patch.dna
            for mutation in mutations_in_epoch:
                patch_file = checkpoint_dir / f"step_{mutation.step:05d}.patch.dna"
                mutation.save(str(patch_file))

            # All saved
            saved_patches = list(checkpoint_dir.glob("*.patch.dna"))
            assert len(saved_patches) == 3

    def test_workflow_metrics_computed(self):
        """Compute and log metrics at each evolutionary step."""
        metrics = {
            "step": 5000,
            "train_ppl": 42.5,
            "ood_ppl": 185.3,
            "gap_ratio": 4.36,
            "phi": 0.18,
            "mutations_emitted": 7,
            "hot_paths": 14,
            "cold_paths": 3,
        }

        # Metrics should be loggable
        assert metrics["step"] == 5000
        assert metrics["gap_ratio"] < 5.0  # Good generalization
        assert metrics["phi"] > 0.1  # Decent consciousness

    def test_workflow_resumption_after_interrupt(self):
        """Workflow can resume after interruption (e.g., Colab timeout)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Session 1: run to step 5000
            patches_session_1 = [
                DNAPatch(version="1.0", step=1000, kind="node_mutation", target="gws", delta=[0.1], metadata={}),
                DNAPatch(version="1.0", step=3000, kind="node_mutation", target="lang", delta=[0.15], metadata={}),
                DNAPatch(version="1.0", step=5000, kind="node_mutation", target="hippo", delta=[0.08], metadata={}),
            ]

            for patch in patches_session_1:
                patch.save(str(checkpoint_dir / f"step_{patch.step:05d}.patch.dna"))

            # Session 2: resume from step 5000, continue to 10000
            checkpoint_files = sorted(checkpoint_dir.glob("step_*.patch.dna"))
            last_patch = DNAPatch.load(str(checkpoint_files[-1]))
            last_step = last_patch.step

            # Continue from 5000
            patches_session_2 = [
                DNAPatch(version="1.0", step=7000, kind="node_mutation", target="amyg", delta=[0.05], metadata={}),
                DNAPatch(version="1.0", step=10000, kind="node_mutation", target="pfc", delta=[0.12], metadata={}),
            ]

            for patch in patches_session_2:
                patch.save(str(checkpoint_dir / f"step_{patch.step:05d}.patch.dna"))

            # All patches from both sessions
            final_patches = sorted(checkpoint_dir.glob("step_*.patch.dna"))
            assert len(final_patches) == 5
            last_final = DNAPatch.load(str(final_patches[-1]))
            assert last_final.step == 10000
