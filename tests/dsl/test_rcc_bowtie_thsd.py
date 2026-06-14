# -*- coding: utf-8 -*-
"""TDD Tests for RCC Bowtie THSD Integration

Tests that arch.neuro can contain both v2.0 declarations and THSD complexes.
"""
import pytest
from pathlib import Path
from neuroslm.dsl.compiler import NeuroMLCompiler
from neuroslm.dsl.multifile import compile_folder


class TestRCCBowtieThsdParsing:
    """Test that RCC Bowtie arch.neuro with THSD complexes parses correctly."""

    def test_rcc_bowtie_compiles_with_thsd_complexes(self):
        """RCC Bowtie arch.neuro should compile despite having THSD complex blocks."""
        arch_root = Path("architectures/master")

        # Should not raise an error
        ir = compile_folder(arch_root)

        assert ir is not None

    def test_rcc_bowtie_extracts_thsd_complexes(self):
        """RCC Bowtie THSD complexes should be extracted correctly."""
        arch_root = Path("architectures/master")
        ir = compile_folder(arch_root)

        # Should have extracted THSD complexes
        assert len(ir.thsd_complexes) == 4

        names = [c.name for c in ir.thsd_complexes]
        assert "Thalamus" in names
        assert "GlobalWorkspace" in names
        assert "PrefrontalCortex" in names
        assert "MotorCortex" in names

    def test_rcc_bowtie_thalamus_complex(self):
        """Thalamus complex should have correct specifications."""
        arch_root = Path("architectures/master")
        ir = compile_folder(arch_root)

        thalamus = next((c for c in ir.thsd_complexes if c.name == "Thalamus"), None)
        assert thalamus is not None

        assert thalamus.stalk.representation_dim == 256
        assert thalamus.topology.spectral_gap == 0.32
        assert thalamus.topology.dimension == 6
        assert thalamus.formal_spec.phi_target == 0.8
        assert thalamus.dynamics.emission is not None
        assert thalamus.dynamics.nemori is not None

    def test_rcc_bowtie_global_workspace_complex(self):
        """GlobalWorkspace complex should have correct specifications."""
        arch_root = Path("architectures/master")
        ir = compile_folder(arch_root)

        gws = next((c for c in ir.thsd_complexes if c.name == "GlobalWorkspace"), None)
        assert gws is not None

        assert gws.stalk.representation_dim == 512
        assert gws.topology.spectral_gap == 0.35
        assert gws.topology.dimension == 8
        assert gws.formal_spec.phi_target == 0.85
        assert gws.dynamics.emission is not None
        assert gws.dynamics.release is not None  # Has release operator
        assert gws.dynamics.nemori is not None

    def test_rcc_bowtie_preserves_v2_declarations(self):
        """v2.0 declarations should still be accessible."""
        arch_root = Path("architectures/master")
        ir = compile_folder(arch_root)

        # Should have architecture declaration
        assert ir.architecture is not None
        assert ir.architecture['name'] == 'master'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
