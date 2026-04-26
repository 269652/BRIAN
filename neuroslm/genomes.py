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

    Logic:
      - If CUDA is available, choose a GPU profile depending on available memory.
      - Otherwise return the CPU profile.
    """
    try:
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            mem_gb = prop.total_memory / (1024 ** 3)
            # Heuristics
            if mem_gb >= 16:
                # Big GPU: scale layers and dims up
                return BuildGenome('gpu_large', layer_scale=2.5, d_sem_scale=2.0, d_hidden_scale=2.0, lang_ctx_scale=1.5)
            elif mem_gb >= 10:
                return BuildGenome('gpu_medium', layer_scale=1.8, d_sem_scale=1.5, d_hidden_scale=1.5, lang_ctx_scale=1.2)
            else:
                return BuildGenome('gpu_small', layer_scale=1.3, d_sem_scale=1.2, d_hidden_scale=1.2, lang_ctx_scale=1.0)
    except Exception:
        # Any failure -> fall back to CPU profile
        pass
    return BuildGenome('cpu', layer_scale=1.0, d_sem_scale=1.0, d_hidden_scale=1.0, lang_ctx_scale=1.0)


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
