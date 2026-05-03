"""GenomeConfigurable — mixin that lets nn.Modules read compiled genome envs
AND respond to real-time NT levels via modifier rules.

Two sources of configuration:
  1. Compiled genome env (opcode-derived params like recall_gate, attend_gate)
  2. Structural genome NT modifiers (e.g. high 5HT → PFC safer selection)

NT modifiers now support **latent protein-shape matching**:
  - When a NTShapeRegistry is bound, each modifier computes its effective NT
    level as an affinity-weighted sum across ALL NTs (not just the named one).
  - This allows cross-reactivity: a "5HT modifier" can partially activate
    when another NT has a similar latent shape.

At init time, configure_from_genome() sets base parameter values.
Each forward tick, apply_nt_modifiers() adjusts them based on NT state.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dna.structural_genome import StructuralGenome, NTModifier
    from ..neurochem.receptors import NTShapeRegistry


class GenomeConfigurable:
    """Mixin: genome-driven + NT-modulated module behavior."""

    _genome_env: dict
    _base_env: dict          # snapshot of env BEFORE NT modifiers
    _nt_modifiers: list
    _structural: object
    _nt_shape_registry: object  # NTShapeRegistry for protein-shape matching

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

    def bind_nt_shapes(self, registry: "NTShapeRegistry"):
        """Bind shared NT shape registry for protein-shape modifier matching."""
        self._nt_shape_registry = registry

    def _effective_nt_level(self, modifier_nt: str,
                            nt_levels: dict[str, float]) -> float:
        """Compute effective NT level for a modifier using shape matching.

        If NTShapeRegistry is bound: effective_level = Σ affinity(modifier_nt, nt_j) * level_j
        This means a "5HT modifier" partially responds to any NT with a similar shape.
        Without registry: falls back to direct name lookup (legacy).
        """
        registry = getattr(self, '_nt_shape_registry', None)
        if registry is None:
            return nt_levels.get(modifier_nt, 0.0)

        # Shape-based: compute affinity between modifier's canonical NT and all NTs
        import torch
        import torch.nn.functional as F
        from ..neurochem.transmitters import NT_NAMES, NT_INDEX

        all_shapes = registry.all_shapes()  # (N_NT, SHAPE_DIM)
        idx = NT_INDEX[modifier_nt]
        modifier_shape = all_shapes[idx]    # (SHAPE_DIM,)

        # Cosine similarity → affinity (already normalised)
        cos_sim = (modifier_shape @ all_shapes.T)     # (N_NT,)
        affinity = torch.sigmoid(5.0 * cos_sim)       # (N_NT,) in [0,1]

        # Affinity-weighted sum of NT levels
        effective = 0.0
        for i, name in enumerate(NT_NAMES):
            effective += float(affinity[i]) * nt_levels.get(name, 0.0)
        return effective

    def apply_nt_modifiers(self, nt_levels: dict[str, float]):
        """Apply NT modifier rules — shift genome params based on NT state.

        Uses protein-shape matching when NTShapeRegistry is bound:
        each modifier responds to an affinity-weighted blend of ALL NTs,
        not just the named one.  Cross-reactivity is emergent.

        Called each forward tick. Resets to base env first, then applies
        all matching modifiers.

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
            level = self._effective_nt_level(mod.nt, nt_levels)
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
