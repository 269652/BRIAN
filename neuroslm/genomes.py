from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class BuildGenome:
    name: str
    layer_scale: float = 1.0
    d_sem_scale: float = 1.0
    d_hidden_scale: float = 1.0
    lang_ctx_scale: float = 1.0

    def to_dict(self):
        return {
            'name': self.name,
            'layer_scale': self.layer_scale,
            'd_sem_scale': self.d_sem_scale,
            'd_hidden_scale': self.d_hidden_scale,
            'lang_ctx_scale': self.lang_ctx_scale,
        }


def select_build_genome() -> BuildGenome:
    """Detect hardware and return a suitable BuildGenome profile.

    NOTE: When the user selects a preset (tiny/small/large/xl/xxl),
    the preset already encodes the correct sizes for the target hardware.
    The build genome should only scale when using the default 'small' preset
    without an explicit choice.  For safety, we now default to NO scaling
    (cpu profile) — the preset IS the scaling.
    """
    return BuildGenome('preset', layer_scale=1.0, d_sem_scale=1.0,
                       d_hidden_scale=1.0, lang_ctx_scale=1.0)


BUILTIN_BUILDS = {
    'cpu': BuildGenome('cpu', 1.0, 1.0, 1.0, 1.0),
    'gpu_small': BuildGenome('gpu_small', 1.3, 1.2, 1.2, 1.0),
    'gpu_medium': BuildGenome('gpu_medium', 1.8, 1.5, 1.5, 1.2),
    'gpu_large': BuildGenome('gpu_large', 2.5, 2.0, 2.0, 1.5),
}


def apply_build_genome(cfg, build: BuildGenome):
    """Return a new BrainConfig with sizes scaled according to the build genome."""
    # Create a shallow copy
    new = type(cfg)()  # new BrainConfig
    # Copy over all fields first
    for k, v in cfg.__dict__.items():
        setattr(new, k, v)
    # Scale integer layer counts (round and at least 1)
    new.lang_layers = max(1, int(round(cfg.lang_layers * build.layer_scale)))
    new.dmn_layers = max(1, int(round(cfg.dmn_layers * build.layer_scale)))
    new.pfc_layers = max(1, int(round(cfg.pfc_layers * build.layer_scale)))
    # world/self/forward layers
    new.world_layers = max(1, int(round(cfg.world_layers * build.layer_scale)))
    new.self_layers = max(1, int(round(cfg.self_layers * build.layer_scale)))
    new.forward_layers = max(1, int(round(cfg.forward_layers * build.layer_scale)))

    # Scale dims
    new.d_sem = max(16, int(round(cfg.d_sem * build.d_sem_scale)))
    new.d_hidden = max(16, int(round(cfg.d_hidden * build.d_hidden_scale)))

    # Scale context
    new.lang_ctx = max(32, int(round(cfg.lang_ctx * build.lang_ctx_scale)))

    return new
