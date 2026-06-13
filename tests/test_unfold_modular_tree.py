# -*- coding: utf-8 -*-
"""TDD: unfold reconstructs the FULL modular tree from DNA.

Compiling a multi-file architecture and unfolding it must regenerate
arch.neuro AND every module file (modules/*.neuro, lib/*.neuro),
byte-for-byte — so an evolved arch can live in modules/libs.
"""
import tempfile
from pathlib import Path

import pytest

from neuroslm.compiler.ribosome import RibosomeCompiler


def _make_arch(root: Path) -> dict:
    (root / "lib").mkdir(parents=True)
    (root / "modules").mkdir(parents=True)
    files = {
        "arch.neuro": (
            "architecture main { d_sem: 256, dt: 0.01 }\n"
            'import { x } from "@/lib/equations"\n'
            'import { cortex } from "@/modules/cortex"\n'
            'population gws { count: 512, dynamics: "rate_code" }\n'
        ),
        "lib/equations.neuro": (
            "# shared equations\n"
            'export equation x { params: [a], formula: "a * 2" }\n'
        ),
        "modules/cortex.neuro": (
            'export population cortex { count: 1024, dynamics: "rate_code" }\n'
        ),
    }
    for rel, src in files.items():
        (root / rel).write_text(src, encoding="utf-8")
    return files


class TestModularUnfold:
    def test_unfold_reconstructs_module_files(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_root = td / "src"
            src_root.mkdir()
            files = _make_arch(src_root)

            dna_file = td / "arch.dna"
            out_root = td / "evol"
            out_main = out_root / "arch.neuro"

            comp = RibosomeCompiler()
            comp.compile_file(str(src_root), str(dna_file))
            comp.unfold_file(str(dna_file), str(out_main))

            # Every original file must be reconstructed under out_root.
            for rel, original in files.items():
                regenerated = out_root / rel
                assert regenerated.exists(), f"missing reconstructed file: {rel}"
                assert regenerated.read_text(encoding="utf-8") == original

    def test_unfold_creates_lib_and_modules_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_root = td / "src"
            src_root.mkdir()
            _make_arch(src_root)

            dna_file = td / "arch.dna"
            out_root = td / "evol"

            comp = RibosomeCompiler()
            comp.compile_file(str(src_root), str(dna_file))
            comp.unfold_file(str(dna_file), str(out_root / "arch.neuro"))

            assert (out_root / "lib").is_dir()
            assert (out_root / "modules").is_dir()


class TestRccBowtieModularUnfold:
    def test_rcc_bowtie_unfold_regenerates_all_modules(self):
        arch_root = Path(__file__).parent.parent / "architectures" / "rcc_bowtie"
        if not (arch_root / "arch.neuro").exists():
            pytest.skip("rcc_bowtie not found")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dna_file = td / "rcc.dna"
            out_root = td / "evol"

            comp = RibosomeCompiler()
            comp.compile_file(str(arch_root), str(dna_file))
            comp.unfold_file(str(dna_file), str(out_root / "arch.neuro"))

            # arch.neuro bit-identical
            assert (out_root / "arch.neuro").read_text(encoding="utf-8") == \
                (arch_root / "arch.neuro").read_text(encoding="utf-8")

            # At least the module + lib trees exist with several files each.
            assert (out_root / "modules").is_dir()
            assert (out_root / "lib").is_dir()
            n_modules = len(list((out_root / "modules").glob("*.neuro")))
            n_lib = len(list((out_root / "lib").glob("*.neuro")))
            assert n_modules >= 5, f"expected several module files, got {n_modules}"
            assert n_lib >= 3, f"expected several lib files, got {n_lib}"

            # Spot-check one reconstructed module is bit-identical to source.
            # `modules/gws.neuro` is arch-local (`@/modules/...`) and
            # roundtrips against its arch-relative source.
            # `lib/equations.neuro` now comes from the canonical
            # `@brian/equations` (under <repo>/architectures/lib/), not
            # from the local `<arch>/lib/equations.neuro` shadow — the
            # arch.neuro switched to `@brian/equations` so its 5
            # feature equations are picked up. Compare against the
            # canonical path to keep the bit-identical guarantee where
            # it matters (the source of truth on disk).
            local_checks = [
                ("modules/gws.neuro", arch_root / "modules/gws.neuro"),
            ]
            canonical_lib = (
                arch_root.parent / "lib" / "equations.neuro"
            )
            if canonical_lib.is_file():
                local_checks.append(("lib/equations.neuro", canonical_lib))
            for rel, src_file in local_checks:
                out_file = out_root / rel
                if src_file.exists():
                    assert out_file.exists(), f"missing {rel}"
                    assert out_file.read_text(encoding="utf-8") == \
                        src_file.read_text(encoding="utf-8"), (
                            f"unfold of {rel} drifted from its source "
                            f"of truth {src_file}"
                        )
