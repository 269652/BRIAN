# -*- coding: utf-8 -*-
"""Stage 3 — Reference resolver: turn per-file ASTs into a linked program.

The resolver:
  1. Walks the architecture folder, parses every .neuro file into a
     ModuleAST.
  2. Locates the arch.neuro (package config); its `architecture { ... }`
     block becomes the program's metadata.
  3. For each import, resolves the specifier to a target file via
     PathResolver, then validates that every imported name is actually
     `export`ed by the target.
  4. Returns a `ResolvedProgram` with:
        - per-file ASTs
        - import_map[file] = {local_alias → (target_file, source_name)}
        - architecture config

This is enough for Stage 4 (lib-defined dynamics/function lookup) and
Stage 5 (synapse/modulation equation codegen) to consume.
"""
import pytest
from pathlib import Path

from neuroslm.dsl.multifile import Resolver, ResolverError


# ── Helpers ────────────────────────────────────────────────────────────

def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ── Discovery + parsing ────────────────────────────────────────────────

class TestResolverDiscovery:
    def test_loads_all_files(self, tmp_path):
        _write(tmp_path, "arch.neuro", 'architecture rcc_bowtie { d_sem: 256 }')
        _write(tmp_path, "modules/pfc.neuro",
               'export population output { count: 256 }')
        _write(tmp_path, "lib/dyn.neuro",
               'export dynamics lif { equation: "y = ReLU(x)" }')

        program = Resolver(tmp_path).resolve()
        assert len(program.modules) == 3
        assert program.architecture["name"] == "rcc_bowtie"
        assert program.architecture["properties"]["d_sem"] == "256"


# ── Cross-file imports resolve to target files ────────────────────────

class TestImportResolution:
    def test_resolves_named_import(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { output } from "@/modules/pfc"')
        pfc = _write(tmp_path, "modules/pfc.neuro",
                     'export population output { count: 256 }')

        program = Resolver(tmp_path).resolve()
        arch_path = (tmp_path / "arch.neuro").resolve()
        imports = program.import_map[arch_path]
        assert "output" in imports
        target_file, src_name = imports["output"]
        assert target_file == pfc.resolve()
        assert src_name == "output"

    def test_resolves_aliased_import(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { output as pfc_out } from "@/modules/pfc"')
        _write(tmp_path, "modules/pfc.neuro",
               'export population output { count: 256 }')

        program = Resolver(tmp_path).resolve()
        arch_path = (tmp_path / "arch.neuro").resolve()
        imports = program.import_map[arch_path]
        assert "pfc_out" in imports
        # The original is still recorded as `output`
        assert imports["pfc_out"][1] == "output"

    def test_resolves_folder_module_index(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { output } from "@/modules/pfc"')
        idx = _write(tmp_path, "modules/pfc/index.neuro",
                     'export population output { count: 256 }')

        program = Resolver(tmp_path).resolve()
        arch_path = (tmp_path / "arch.neuro").resolve()
        target_file = program.import_map[arch_path]["output"][0]
        assert target_file == idx.resolve()

    def test_resolves_relative_import(self, tmp_path):
        _write(tmp_path, "arch.neuro", 'architecture x { d_sem: 256 }')
        idx = _write(tmp_path, "modules/pfc/index.neuro",
                     'import { core } from "./layers"')
        layers = _write(tmp_path, "modules/pfc/layers.neuro",
                        'export population core { count: 256 }')

        program = Resolver(tmp_path).resolve()
        idx_path = idx.resolve()
        assert program.import_map[idx_path]["core"][0] == layers.resolve()


# ── Error reporting ────────────────────────────────────────────────────

class TestResolverErrors:
    def test_missing_arch_neuro(self, tmp_path):
        _write(tmp_path, "modules/foo.neuro",
               'export population x { count: 256 }')
        with pytest.raises(ResolverError, match="arch.neuro"):
            Resolver(tmp_path).resolve()

    def test_unresolved_import_target(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { foo } from "@/does_not_exist"')
        with pytest.raises(ResolverError, match="does_not_exist"):
            Resolver(tmp_path).resolve()

    def test_imported_name_not_exported(self, tmp_path):
        # Target file exists, but `foo` isn't exported there.
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { foo } from "@/modules/pfc"')
        _write(tmp_path, "modules/pfc.neuro",
               'population foo { count: 256 }')   # private, not exported
        with pytest.raises(ResolverError, match="not exported"):
            Resolver(tmp_path).resolve()

    def test_imported_name_unknown(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { ghost } from "@/modules/pfc"')
        _write(tmp_path, "modules/pfc.neuro",
               'export population output { count: 256 }')
        with pytest.raises(ResolverError, match="ghost"):
            Resolver(tmp_path).resolve()


# ── Symbol lookup across modules ──────────────────────────────────────

class TestSymbolLookup:
    def test_lookup_imported_symbol(self, tmp_path):
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }\n'
               'import { output as pfc_out } from "@/modules/pfc"')
        _write(tmp_path, "modules/pfc.neuro",
               'export population output { count: 256, dynamics: "rate_code" }')

        program = Resolver(tmp_path).resolve()
        arch_path = (tmp_path / "arch.neuro").resolve()

        # `pfc_out` in arch.neuro's scope should look up the target file's
        # `output` declaration text.
        decl_text = program.lookup(arch_path, "pfc_out")
        assert "count: 256" in decl_text
        assert "rate_code" in decl_text

    def test_lookup_local_private_symbol(self, tmp_path):
        # Each module can resolve its own private symbols too.
        _write(tmp_path, "arch.neuro",
               'architecture x { d_sem: 256 }')
        idx = _write(tmp_path, "modules/pfc/index.neuro",
                     'population helper { count: 64, dynamics: "rate_code" }')

        program = Resolver(tmp_path).resolve()
        decl_text = program.lookup(idx.resolve(), "helper")
        assert "count: 64" in decl_text

    def test_lookup_unknown_symbol_raises(self, tmp_path):
        _write(tmp_path, "arch.neuro", 'architecture x { d_sem: 256 }')
        program = Resolver(tmp_path).resolve()
        arch_path = (tmp_path / "arch.neuro").resolve()
        with pytest.raises(ResolverError, match="not found"):
            program.lookup(arch_path, "ghost")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
