"""GenomeConfigurable — mixin that lets nn.Modules read compiled genome envs
AND respond to real-time NT levels via modifier rules.

Two sources of configuration:
  1. Compiled genome env (opcode-derived params like recall_gate, attend_gate)
  2. Structural genome NT modifiers (e.g. high 5HT → PFC safer selection)

At init time, configure_from_genome() sets base parameter values.
Each forward tick, apply_nt_modifiers() adjusts them based on NT state.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dna.structural_genome import StructuralGenome, NTModifier


class GenomeConfigurable:
    """Mixin: genome-driven + NT-modulated module behavior."""

    _genome_env: dict
    _base_env: dict          # snapshot of env BEFORE NT modifiers
    _nt_modifiers: list
    _structural: object

    def configure_from_genome(self, env: dict,
                              structural: "StructuralGenome | None" = None):
        """Apply compiled genome environment to this module.

        Args:
            env: compiled env dict from GenomeCompiler (opcode-derived params)
            structural: StructuralGenome with receptors, modifiers, architecture
        """
        self._genome_env = dict(env)
        self._base_env = dict(env)  # save original for NT modifier reset
        self._structural = structural
        if structural is not None:
            self._nt_modifiers = list(structural.nt_modifiers)
        else:
            self._nt_modifiers = getattr(self, '_nt_modifiers', [])

    def apply_nt_modifiers(self, nt_levels: dict[str, float]):
        """Apply NT modifier rules — shift genome params based on NT state.

        Called each forward tick. Resets to base env first, then applies
        all matching modifiers. This is how NTs control behavior:

          high 5HT → PFC select_gate *= 0.5 (safer selection)
          high NE  → PFC attend_gate *= 1.5 (urgent tasks prioritized)
          high DA  → BG go_gate *= 1.5 (approach behavior)
          low DA   → BG nogo_gate *= 1.5 (avoidance)
        """
        # Reset to base values (so modifiers don't compound across ticks)
        base = getattr(self, '_base_env', {})
        for key, val in base.items():
            if isinstance(val, (int, float)):
                self._genome_env[key] = val

        for mod in getattr(self, '_nt_modifiers', []):
            level = nt_levels.get(mod.nt, 0.0)
            param = mod.target_param
            current = self._genome_env.get(param)
            if current is not None and isinstance(current, (int, float)):
                new_val = mod.apply(float(current), level)
                self._genome_env[param] = new_val
                # Also set the private attribute if the module has one
                attr = f'_{param}'
                if hasattr(self, attr):
                    setattr(self, attr, new_val)

    def genv(self, key: str, default=None):
        """Read a value from the genome environment."""
        return getattr(self, '_genome_env', {}).get(key, default)

    def genv_float(self, key: str, default: float = 0.0) -> float:
        v = getattr(self, '_genome_env', {}).get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def genv_int(self, key: str, default: int = 0) -> int:
        v = getattr(self, '_genome_env', {}).get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def genv_str(self, key: str, default: str = '') -> str:
        v = getattr(self, '_genome_env', {}).get(key, default)
        return str(v) if v is not None else default

    @property
    def has_genome(self) -> bool:
        return bool(getattr(self, '_genome_env', {}))

    @property
    def structural_genome(self):
        return getattr(self, '_structural', None)
