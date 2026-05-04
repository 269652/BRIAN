"""Legacy DNA compatibility layer (stubs).

The original DNA / genome subsystem was removed. This module provides
small, inert placeholders so older scripts that import symbols from
``neuroslm.dna`` won't crash. All functionality is effectively a no-op.
"""
from typing import Dict, List

ALL_REGIONS: List[str] = []


class BrainDNA:
  """Compatibility stub for BrainDNA. Use Brain.projections and config instead."""
  def __init__(self):
    self.structural = []
    self.projection = type("P", (), {"projections": []})()

  @classmethod
  def default(cls, regions=None):
    return cls()


class GenomeCompiler:
  def __init__(self, *args, **kwargs):
    pass

  def compile_batch(self, genomes: Dict[str, object]):
    return {}

  def get_lisp(self, region: str) -> str:
    return ""

  def get_env(self, region: str) -> dict:
    return {}

  def save_all_lisp(self, out_dir: str) -> str:
    return out_dir


class ModuleGenomePool:
  def __init__(self, *args, **kwargs):
    pass

  def active_all(self):
    return {}

  @classmethod
  def from_state(cls, state):
    return cls()


__all__ = ["ALL_REGIONS", "BrainDNA", "GenomeCompiler", "ModuleGenomePool"]
