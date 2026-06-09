#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compile rcc_bowtie architecture to DNA and unfold it back.

Usage:
    python scripts/compile_rcc_bowtie.py
"""
from __future__ import print_function
from pathlib import Path
import os
import sys

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuroslm.compiler.ribosome import RibosomeCompiler


def main():
    compiler = RibosomeCompiler()

    # Compile rcc_bowtie to DNA
    arch_root = str(Path("architectures/rcc_bowtie").resolve())
    dna_output = str(Path("dna/evol/arch.dna").resolve())

    # Create output directory if needed
    Path(dna_output).parent.mkdir(parents=True, exist_ok=True)

    print("Compiling {} to {}...".format(arch_root, dna_output))
    compiler.compile_file(arch_root, dna_output)
    print("[OK] Compiled to {}".format(dna_output))

    # Check if source map was created
    source_map_path = dna_output.replace('.dna', '.sourcemap.json')
    if os.path.exists(source_map_path):
        print("[OK] Source map created: {}".format(source_map_path))
    else:
        print("[WARN] No source map file (modules may be empty)")

    # Unfold back to neuro
    neuro_output = str(Path("architectures/evol/arch.neuro").resolve())
    Path(neuro_output).parent.mkdir(parents=True, exist_ok=True)

    print("Unfolding {} to {}...".format(dna_output, neuro_output))
    compiler.unfold_file(dna_output, neuro_output)
    print("[OK] Unfolded to {}".format(neuro_output))

    # Verify roundtrip
    output_size = os.path.getsize(neuro_output)
    print("[OK] Output file size: {} bytes".format(output_size))

    # Show first few lines
    try:
        with open(neuro_output, encoding='utf-8') as f:
            lines = f.readlines()[:5]
            print("\nFirst few lines of unfolded arch:")
            for line in lines:
                print("  {}".format(line.rstrip()))
    except Exception as e:
        print("  (Could not read file: {})".format(str(e)))


if __name__ == "__main__":
    main()
