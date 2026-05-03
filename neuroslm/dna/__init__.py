"""DNA subsystem — evolvable algorithms and latent program encoding.

The compilation pipeline for each brain module:
    ModuleGenome → compile → Latent Embedding → decompile → Lisp Source → DSL execute
"""
from .latent_program import (
    LatentProgramSystem, LatentProgramEncoder, LatentProgramDecoder,
    LatentProgramDecompiler,
)
from .compiler import (
    GenomeCompiler, ModuleGenome, ModuleGenomePool,
    MODULE_GENE_SPECS, MODULE_OPCODE_SPECS,
)
