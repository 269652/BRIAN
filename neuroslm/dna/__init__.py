"""DNA subsystem — evolvable algorithms, structural genomes, and NT wiring.

Three genome types per module:
  1. Algorithmic Genome: opcode program (what the module DOES)
  2. Structural Genome: architecture + receptors + NT modifiers (what the module IS)
  3. Projection Genome: NT wiring topology (how modules are connected)

Pipeline:
  ModuleGenome (alleles = opcode program) → decompile → Lisp → DSL execute → env
  StructuralGenome → receptors, modifiers, architecture params
  ProjectionGenome → NT flow topology between all regions

NT modifiers activate at runtime: e.g. high 5HT → PFC selects safer actions.
"""
from .latent_program import (
    LatentProgramSystem, LatentProgramEncoder, LatentProgramDecoder,
    LatentProgramDecompiler,
)
from .compiler import (
    GenomeCompiler, ModuleGenome, ModuleGenomePool, ALL_REGIONS,
)
from .structural_genome import (
    BrainDNA, StructuralGenome, ProjectionGenome, ProjectionGene,
    NTModifier, ReceptorSpec,
    default_projection_genome, STRUCTURAL_INIT_REGISTRY,
)
