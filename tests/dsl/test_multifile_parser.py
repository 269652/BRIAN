# -*- coding: utf-8 -*-
"""Stage 2 — `module` / `import` / `export` parsing + symbol tables.

A .neuro file is implicitly its own module (mjs convention — no outer
`module {}` block needed). Declarations marked `export` are public;
without `export` they're private to the file.

Imports are JavaScript-style:
    import { foo, bar }            from "@/lib/dyn"     # named
    import { foo as alias_name }   from "./other"      # alias
    import "./setup"                                   # side-effect only

The parser this stage delivers:
    parse_module(source, path) -> ModuleAST
        .imports: List[ImportDecl]
        .exports: Dict[name -> raw declaration text]
        .private: Dict[name -> raw declaration text]

It does NOT compile the declarations into the final ProgramIR yet — that's
Stage 3 (reference resolution + symbol qualification). For now the raw
declaration text is enough to validate parsing.
"""
import pytest
from pathlib import Path

from neuroslm.dsl.multifile import parse_module, ImportDecl


# ── Export marker on declarations ──────────────────────────────────────

class TestExportKeyword:
    def test_exported_population_in_exports(self):
        src = '''
            export population foo { count: 256, dynamics: "rate_code" }
        '''
        m = parse_module(src, path=Path("a.neuro"))
        assert "foo" in m.exports
        assert "foo" not in m.private

    def test_unmarked_declaration_is_private(self):
        src = '''
            population helper { count: 64, dynamics: "rate_code" }
        '''
        m = parse_module(src, path=Path("a.neuro"))
        assert "helper" not in m.exports
        assert "helper" in m.private

    def test_mixed_public_and_private(self):
        src = '''
            export population public_pop { count: 256 }
            population private_pop { count: 64 }
            export dynamics custom_dyn { equation: "y = ReLU(x)" }
        '''
        m = parse_module(src, path=Path("a.neuro"))
        assert set(m.exports.keys()) == {"public_pop", "custom_dyn"}
        assert "private_pop" in m.private

    def test_export_synapse(self):
        # Synapses can also be exported (for cross-module reuse).
        src = '''
            population src { count: 256 }
            population tgt { count: 256 }
            export synapse src -> tgt { weight: 0.5 }
        '''
        m = parse_module(src, path=Path("a.neuro"))
        # Synapses are keyed `<src>__<tgt>`.
        assert "src__tgt" in m.exports


# ── Import statements ──────────────────────────────────────────────────

class TestImports:
    def test_named_import(self):
        src = 'import { foo, bar } from "@/lib/dyn"'
        m = parse_module(src, path=Path("a.neuro"))
        assert len(m.imports) == 1
        imp = m.imports[0]
        assert isinstance(imp, ImportDecl)
        assert imp.specifier == "@/lib/dyn"
        assert imp.names == ["foo", "bar"]
        assert imp.aliases == {}

    def test_aliased_import(self):
        src = 'import { foo as bar, baz as qux } from "./other"'
        m = parse_module(src, path=Path("a.neuro"))
        assert m.imports[0].names == ["foo", "baz"]
        assert m.imports[0].aliases == {"foo": "bar", "baz": "qux"}

    def test_mixed_aliased_and_named(self):
        src = 'import { foo, bar as alias_bar } from "@/lib/x"'
        m = parse_module(src, path=Path("a.neuro"))
        imp = m.imports[0]
        assert imp.names == ["foo", "bar"]
        assert imp.aliases == {"bar": "alias_bar"}

    def test_side_effect_import(self):
        src = 'import "./setup"'
        m = parse_module(src, path=Path("a.neuro"))
        assert m.imports[0].specifier == "./setup"
        assert m.imports[0].names == []
        assert m.imports[0].aliases == {}

    def test_multiple_imports(self):
        src = '''
            import { foo } from "@/lib/a"
            import { bar, baz } from "@/lib/b"
            import "./setup"
        '''
        m = parse_module(src, path=Path("a.neuro"))
        assert len(m.imports) == 3
        assert m.imports[0].specifier == "@/lib/a"
        assert m.imports[1].specifier == "@/lib/b"
        assert m.imports[2].specifier == "./setup"


# ── Combined module (imports + exports + private) ─────────────────────

class TestFullModule:
    def test_realistic_module(self):
        src = '''
            import { lif_neuron } from "@/lib/dynamics"
            import { hebbian as plast } from "@/lib/plasticity"

            population core { count: 256, dynamics: "lif_neuron" }
            export population output { count: 256, equation: "y = ReLU(x)" }
            export synapse core -> output { weight: 0.5 }
        '''
        m = parse_module(src, path=Path("modules/pfc/index.neuro"))

        # Imports
        assert len(m.imports) == 2
        assert m.imports[0].names == ["lif_neuron"]
        assert m.imports[1].aliases == {"hebbian": "plast"}

        # Exports vs private
        assert set(m.exports.keys()) == {"output", "core__output"}
        assert "core" in m.private

    def test_arch_neuro_with_runtime_block(self):
        # `arch.neuro` uses the architecture-config block at the top.
        src = '''
            architecture rcc_bowtie { d_sem: 256, dt: 0.01 }

            neurotransmitter dopamine { base_concentration: 0.10 }

            import { output as pfc_out } from "@/modules/pfc"
        '''
        m = parse_module(src, path=Path("arch.neuro"))
        assert m.architecture is not None
        assert m.architecture["name"] == "rcc_bowtie"
        assert m.architecture["properties"]["d_sem"] == "256"
        # NTs become private declarations (they're globally visible by
        # virtue of being declared in arch.neuro, but the export marker
        # isn't required for them).
        assert "dopamine" in m.private
        # Import recorded
        assert m.imports[0].aliases == {"output": "pfc_out"}


# ── Error handling ─────────────────────────────────────────────────────

class TestParserErrors:
    def test_export_without_declaration_rejected(self):
        with pytest.raises(ValueError, match="export"):
            parse_module("export", path=Path("a.neuro"))

    def test_malformed_import_rejected(self):
        with pytest.raises(ValueError, match="import"):
            parse_module('import { foo from "x"', path=Path("a.neuro"))

    def test_duplicate_export_rejected(self):
        src = '''
            export population foo { count: 256 }
            export population foo { count: 128 }
        '''
        with pytest.raises(ValueError, match="duplicate"):
            parse_module(src, path=Path("a.neuro"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
