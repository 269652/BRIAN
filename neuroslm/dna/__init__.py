"""DNA subsystem — evolvable algorithms and latent program encoding.

The compilation pipeline for each brain module:
    ModuleGenome (alleles = opcode program) → decompile → Lisp Source → DSL execute
"""
from .latent_program import (
    LatentProgramSystem, LatentProgramEncoder, LatentProgramDecoder,
    LatentProgramDecompiler,
)
from .compiler import (
    GenomeCompiler, ModuleGenome, ModuleGenomePool, ALL_REGIONS,
)
