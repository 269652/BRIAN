# -*- coding: utf-8 -*-
"""Per-arch/preset run heatmap store — the latest run's heat map, kept per config.

Every training run records where its gradient heat concentrated, namespaced by
``(arch, preset)`` so ``heatmaps/<arch>/<preset>.json`` always holds the *latest*
run of that configuration. This is the map that answers "where does the wild gnorm
live" and the signal that steers the exploration search toward hot pathways.

Reuses ``neuroslm.evolution.grad_heat.parameter_grad_norms`` (the existing
grad-norm extractor) so it composes with the training loop's per-parameter grads.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def _arch_name(arch: str) -> str:
    # accept a folder path ("architectures/SmolLM") or a bare name
    return Path(str(arch)).name


@dataclass
class RunHeatmap:
    arch: str
    preset: str
    step: int
    entries: Dict[str, float] = field(default_factory=dict)
    git_commit: str = ""
    summary: dict = field(default_factory=dict)


def _summarise(entries: Dict[str, float], top_k: int) -> dict:
    if not entries:
        return {"max": 0.0, "mean": 0.0, "n": 0, "hot": []}
    vals = list(entries.values())
    hot = sorted(entries.items(), key=lambda kv: -kv[1])[:top_k]
    return {
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "n": len(vals),
        "hot": [[k, v] for k, v in hot],
    }


def heatmap_from_grad_norms(arch: str, preset: str, grad_norms: Dict[str, float],
                            step: int, git_commit: str = "", top_k: int = 15) -> RunHeatmap:
    entries = {k: float(v) for k, v in grad_norms.items()}
    return RunHeatmap(
        arch=_arch_name(arch), preset=preset, step=step, entries=entries,
        git_commit=git_commit, summary=_summarise(entries, top_k),
    )


class HeatmapStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, arch: str, preset: str) -> Path:
        return self.root / _arch_name(arch) / f"{preset}.json"

    def record(self, rh: RunHeatmap) -> Path:
        p = self.path(rh.arch, rh.preset)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(rh), indent=1), encoding="utf-8")
        return p

    def load(self, arch: str, preset: str) -> RunHeatmap:
        p = self.path(arch, preset)
        if not p.exists():
            raise KeyError(f"no heatmap for ({_arch_name(arch)}, {preset})")
        d = json.loads(p.read_text(encoding="utf-8"))
        return RunHeatmap(**{k: d.get(k) for k in RunHeatmap.__dataclass_fields__})

    def list_all(self) -> List[Tuple[str, str]]:
        out = []
        if not self.root.exists():
            return out
        for arch_dir in sorted(self.root.iterdir()):
            if arch_dir.is_dir():
                for f in sorted(arch_dir.glob("*.json")):
                    out.append((arch_dir.name, f.stem))
        return out


def record_training_run(store: HeatmapStore, arch: str, preset: str, model,
                        step: int, git_commit: str = "", top_k: int = 15) -> RunHeatmap:
    """Collect per-parameter grad heat from a model and record it (latest wins)."""
    from neuroslm.evolution.grad_heat import parameter_grad_norms
    grad_norms = parameter_grad_norms(model.named_parameters())
    rh = heatmap_from_grad_norms(arch, preset, grad_norms, step=step,
                                 git_commit=git_commit, top_k=top_k)
    store.record(rh)
    return rh
