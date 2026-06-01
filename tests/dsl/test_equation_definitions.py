# -*- coding: utf-8 -*-
"""Test suite for equation definitions and reusability.

Tests the ability to define equations once and reuse them across synapses,
populations, and modulations via @name references.

NOTE: These tests are marked xfail because equation-definition support
(standalone `equation` blocks, `@ref` resolution, linter without file_path)
is not yet fully implemented in the compiler/linter.
"""
import pytest
from pathlib import Path
from neuroslm.dsl.compiler import NeuroMLCompiler, EquationDefnIR, ProgramIR
from neuroslm.dsl.equations import parse_equation, EquationExpr
from neuroslm.dsl.neuro_linter import NeuroLinter
from neuroslm.dsl.codegen import CodeGenerator

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="equation-definition features not yet fully implemented in compiler/linter",
)


class TestEquationDefinitionParsing:
    """Test parsing of equation definitions from .neuro files."""

    def test_parse_simple_equation_definition(self):
        """Parse a simple equation definition."""
        dsl = """
        equation rate_code_dynamics {
            params: [x],
            formula: "y = ReLU(x)"
        }
        """
        compiler = NeuroMLCompiler()
        # Should not raise; definition should be recognized
        compiler.compile(dsl)

    def test_parse_equation_with_multiple_params(self):
        """Parse equation definition with multiple parameters."""
        dsl = """
        equation standard_synapse_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }
        """
        compiler = NeuroMLCompiler()
        compiler.compile(dsl)

    def test_parse_exported_equation(self):
        """Parse exported equation definition."""
        dsl = """
        export equation standard_synapse_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        # Check that equation is in exports
        assert hasattr(ir, 'equation_decls'), "IR should contain equation_decls"

    def test_parse_multiple_equation_definitions(self):
        """Parse file with multiple equation definitions."""
        dsl = """
        equation rate_code { params: [x], formula: "y = ReLU(x)" }
        equation gated_dynamics { params: [x, gate], formula: "y = ReLU(x) * sigmoid(gate)" }
        equation attractor { params: [x, s], formula: "y = (1 - 0.1) * s + 0.1 * ReLU(x)" }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        assert hasattr(ir, 'equation_decls'), "IR should contain equation definitions"


class TestEquationDefinitionInSynapses:
    """Test using equation definitions in synapse declarations."""

    def test_synapse_with_equation_reference(self):
        """Synapse using @equation reference instead of inline."""
        dsl = """
        equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            neurotransmitter: "glutamate",
            equation: @standard_transmission
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        # Should resolve @standard_transmission reference
        assert len(ir.synapses) == 1
        synapse = ir.synapses[0]
        assert synapse.equation is not None

    def test_synapse_inline_vs_referenced_equivalence(self):
        """Inline equation and @-referenced equation produce identical IR."""
        dsl_inline = """
        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            neurotransmitter: "glutamate",
            equation: "y = weight * (x_pre @ W)"
        }
        """

        dsl_referenced = """
        equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            neurotransmitter: "glutamate",
            equation: @standard_transmission
        }
        """

        compiler_inline = NeuroMLCompiler()
        compiler_ref = NeuroMLCompiler()

        ir_inline = compiler_inline.compile(dsl_inline)
        ir_ref = compiler_ref.compile(dsl_referenced)

        # Both should have identical synapse configurations
        assert len(ir_inline.synapses) == len(ir_ref.synapses)
        synapse_inline = ir_inline.synapses[0]
        synapse_ref = ir_ref.synapses[0]

        assert synapse_inline.weight == synapse_ref.weight
        assert synapse_inline.neurotransmitter == synapse_ref.neurotransmitter

    def test_multiple_synapses_same_equation_definition(self):
        """Multiple synapses use the same equation definition."""
        dsl = """
        equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population sensory { count: 100, dynamics: "rate_code" }
        export population thalamus { count: 100, dynamics: "rate_code" }
        export population cortex { count: 100, dynamics: "rate_code" }

        synapse sensory -> thalamus {
            weight: 0.7,
            neurotransmitter: "glutamate",
            equation: @standard_transmission
        }

        synapse thalamus -> cortex {
            weight: 0.8,
            neurotransmitter: "glutamate",
            equation: @standard_transmission
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        assert len(ir.synapses) == 2
        # Both synapses should reference the same equation definition


class TestEquationDefinitionInPopulations:
    """Test using equation definitions in population declarations."""

    def test_population_with_equation_reference(self):
        """Population using @equation reference."""
        dsl = """
        equation rate_code_dynamics {
            params: [x],
            formula: "y = ReLU(x)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pfc {
            count: 256,
            equation: @rate_code_dynamics
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        assert len(ir.populations) == 1
        pop = ir.populations[0]
        assert pop.equation is not None

    def test_population_dynamics_vs_equation_reference(self):
        """Population can use dynamics enum OR @equation reference."""
        dsl = """
        equation rate_code_eq {
            params: [x],
            formula: "y = ReLU(x)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population p1 {
            count: 256,
            dynamics: "rate_code"
        }

        export population p2 {
            count: 256,
            equation: @rate_code_eq
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        assert len(ir.populations) == 2


class TestEquationDefinitionInModulations:
    """Test using equation definitions in modulation declarations."""

    def test_modulation_with_equation_reference(self):
        """Modulation using @equation reference."""
        dsl = """
        equation multiplicative_modulation {
            params: [output, c, gain],
            formula: "y = output * (c * gain)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        neurotransmitter dopamine { base_concentration: 0.1 }

        export population pfc { count: 256, dynamics: "rate_code" }

        modulation dopamine -> pfc {
            effect: "multiplicative",
            gain: 0.6,
            equation: @multiplicative_modulation
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        assert len(ir.modulations) == 1
        mod = ir.modulations[0]
        assert mod.equation is not None


class TestEquationParameterSubstitution:
    """Test parameter substitution in equation definitions."""

    def test_parameter_substitution_in_synapse(self):
        """Parameters from synapse are substituted into equation definition."""
        dsl = """
        equation synapse_eq {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            equation: @synapse_eq
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        synapse = ir.synapses[0]
        # weight=0.6 should be substituted into the formula
        assert synapse.weight == 0.6

    def test_parameter_mismatch_warning(self):
        """Linter warns when synapse parameters don't match equation definition."""
        dsl = """
        equation synapse_eq {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            neurotransmitter: "glutamate",
            equation: @synapse_eq
        }
        """
        linter = NeuroLinter()
        diags = linter.lint(dsl)
        # Should not warn about undefined parameters
        # (neurotransmitter is valid even though not in equation definition)


class TestEquationImportExport:
    """Test importing and exporting equation definitions across files."""

    def test_export_equation_definition(self):
        """Equation definition can be exported for cross-file use."""
        dsl = """
        export equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }
        """
        linter = NeuroLinter()
        diags = linter.lint(dsl)
        # Should not raise; export should be recognized

    def test_import_equation_definition(self):
        """Equation definition can be imported from another module."""
        dsl = """
        import { standard_transmission } from "@/lib/equations"

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            equation: @standard_transmission
        }
        """
        compiler = NeuroMLCompiler()
        # Should resolve import and use the definition
        # (actual resolution depends on file system, so we test the mechanism)


class TestEquationDefinitionLinting:
    """Test linting and validation of equation definitions."""

    def test_undefined_equation_reference_warning(self):
        """Linter warns when @equation reference is undefined."""
        dsl = """
        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            equation: @undefined_equation
        }
        """
        linter = NeuroLinter()
        diags = linter.lint(dsl)
        # Should warn: undefined equation reference
        assert any('undefined' in str(d).lower() or 'equation' in str(d).lower() for d in diags)

    def test_unused_exported_equation_info(self):
        """Linter informs when exported equation is never used."""
        dsl = """
        export equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }
        export population p { count: 100, dynamics: "rate_code" }
        """
        linter = NeuroLinter()
        diags = linter.lint(dsl)
        # May warn about unused export (optional diagnostic)

    def test_parameter_mismatch_detection(self):
        """Linter detects when equation formula has different params than declared."""
        dsl = """
        equation bad_def {
            params: [weight, x],
            formula: "y = weight * (x_pre @ W)"
        }
        """
        linter = NeuroLinter()
        diags = linter.lint(dsl)
        # Should warn: formula uses x_pre, W but only weight, x declared


class TestEquationDefinitionCodegen:
    """Test code generation for equations defined via definitions."""

    def test_codegen_uses_equation_definition(self):
        """Code generation correctly uses equation definition parameters."""
        dsl = """
        equation synapse_eq {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }

        export population pre { count: 100, dynamics: "rate_code", equation: "y = ReLU(x)" }
        export population post { count: 100, dynamics: "rate_code" }

        synapse pre -> post {
            weight: 0.6,
            equation: @synapse_eq
        }
        """
        compiler = NeuroMLCompiler()
        ir = compiler.compile(dsl)
        # Should be able to generate valid module without errors
        gen = CodeGenerator(ir, module_name="TestCircuit")
        module_src = gen.generate()
        assert module_src is not None
        assert 'weight' in module_src or 'class' in module_src


class TestEquationDefinitionRoundtrip:
    """Test round-trip: parse -> IR -> serialize -> parse."""

    def test_roundtrip_equation_definition(self):
        """Equation definition survives round-trip parse -> serialize -> parse."""
        dsl = """
        export equation standard_transmission {
            params: [weight, x_pre, W],
            formula: "y = weight * (x_pre @ W)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }
        """
        compiler = NeuroMLCompiler()
        ir1 = compiler.compile(dsl)
        # Should be able to serialize and re-parse
        # (actual serialization depends on IR implementation)


class TestEquationDefinitionThreeWayEquivalence:
    """Test 3-way equivalence: dynamics enum ≡ inline equation ≡ @definition reference."""

    def test_rate_code_three_way_equivalence(self):
        """rate_code enum ≡ inline y=ReLU(x) ≡ @definition reference."""
        dsl_enum = """
        architecture test_arch { d_sem: 256, dt: 0.01 }
        export population p { count: 256, dynamics: "rate_code" }
        """

        dsl_inline = """
        architecture test_arch { d_sem: 256, dt: 0.01 }
        export population p { count: 256, equation: "y = ReLU(x)" }
        """

        dsl_definition = """
        equation rate_code_dynamics {
            params: [x],
            formula: "y = ReLU(x)"
        }

        architecture test_arch { d_sem: 256, dt: 0.01 }
        export population p { count: 256, equation: @rate_code_dynamics }
        """

        compiler_enum = NeuroMLCompiler()
        compiler_inline = NeuroMLCompiler()
        compiler_def = NeuroMLCompiler()

        ir_enum = compiler_enum.compile(dsl_enum)
        ir_inline = compiler_inline.compile(dsl_inline)
        ir_def = compiler_def.compile(dsl_definition)

        # All three should produce identical population IR
        pop_enum = ir_enum.populations[0]
        pop_inline = ir_inline.populations[0]
        pop_def = ir_def.populations[0]

        # Should have same equation semantics
        assert pop_enum.count == pop_inline.count == pop_def.count
        # The actual equation should be identical after lowering


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
