"""Honest accounting of every preset / scale in the codebase.

Instantiates each `scales: {}` variant declared in `architectures/SmolLM/arch.neuro`
using the *exact same code path* that `train_dsl.py` uses at deploy time (which is
`build_dsl_language_cortex(...)` with the live `cfg.nfo / cfg.grid_positions /
cfg.episodic_memory / cfg.surprise_head` etc.), then reports:

  - total trainable param count (the thing the optimizer steps over)
  - module-by-module breakdown so the labels reflect *all* trained modules
    (embedding, attention, MLP, lm_head, NFO, grid_positions, …)
  - delta vs the `approx_params:` label declared in arch.neuro

Run:  `python scripts/audit_presets.py`
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make `neuroslm` importable when run from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from neuroslm.dsl.nn_lang import build_dsl_language_cortex  # noqa: E402
from neuroslm.dsl.training_config import load_training_config_from_arch  # noqa: E402


VOCAB = 50257   # GPT-2 BPE — what production actually uses


def _fmt_m(n: int) -> str:
    """Format param count as 'XX.XM' (millions, 1 decimal)."""
    return f"{n / 1e6:.1f}M"


def _module_breakdown(model: torch.nn.Module) -> Dict[str, int]:
    """Group trainable params by top-level module name.

    Buckets we care about (for honest labeling):
        embedding   — token embeddings (vocab × d_model)
        lm_head     — output projection (often tied to embedding)
        attention   — Q/K/V/O projections across every block
        mlp         — feed-forward MLPs (largest portion at depth)
        norms       — LayerNorm / RMSNorm (tiny but counted)
        nfo         — Neural Field Oscillator (H015..H018)
        grid_pos    — multi-scale grid-cell positional bias (H16)
        episodic    — kNN episodic memory (H15)
        surprise    — local-NLL surprise head (H19)
        other       — anything not classified above
    """
    buckets: Dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        lname = name.lower()
        if "embed" in lname and "pos" not in lname and "grid" not in lname:
            buckets["embedding"] += n
        elif "lm_head" in lname or "to_logits" in lname or "unembed" in lname:
            buckets["lm_head"] += n
        elif "nfo" in lname or "oscillator" in lname:
            buckets["nfo"] += n
        elif "grid" in lname:
            buckets["grid_pos"] += n
        elif "episod" in lname or "memory" in lname:
            buckets["episodic"] += n
        elif "surprise" in lname:
            buckets["surprise"] += n
        elif any(k in lname for k in (".q.", ".k.", ".v.", ".o.", "wq", "wk", "wv", "wo",
                                       "qkv", "attn", "attention")):
            buckets["attention"] += n
        elif any(k in lname for k in ("mlp", "ff", "feed", "gate", "up_proj",
                                       "down_proj", "w1", "w2", "w3")):
            buckets["mlp"] += n
        elif "norm" in lname or "ln" in lname:
            buckets["norms"] += n
        else:
            buckets["other"] += n
    return dict(buckets)


def _build_with_cfg(d_model: int, depth: int, n_heads: int, max_ctx: int,
                    cfg) -> torch.nn.Module:
    """Build the cortex *exactly* as `train_dsl.py` does (with NFO/grid/etc).

    `cfg` is the parsed `TrainingConfig` from `arch.neuro` — it carries the
    feature-flag objects (`cfg.nfo`, `cfg.grid_positions`, …) that determine
    whether those modules are instantiated.
    """
    return build_dsl_language_cortex(
        vocab=VOCAB,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        max_ctx=max_ctx,
        dropout=cfg.dropout,
        n_kv_heads=None,        # default → MHA (n_kv_heads = n_heads)
        pct_trunk=cfg.pct_trunk,
        tonnetz_period=cfg.tonnetz_period,
        stochastic_depth=cfg.stochastic_depth,
        grid_positions=cfg.grid_positions,
        episodic_memory=cfg.episodic_memory,
        surprise_head=cfg.surprise_head,
        nfo=cfg.nfo,
        cosine_head=cfg.cosine_head,
        rope_base=float(getattr(cfg, "rope_base", 10000.0)),
    )


def audit_arch(arch_root: Path) -> List[Tuple[str, dict]]:
    """Audit every scale declared in `arch_root/arch.neuro`.

    Returns a list of (scale_name, info_dict) tuples sorted by trainable
    param count (ascending).
    """
    cfg = load_training_config_from_arch(arch_root)
    rows: List[Tuple[str, dict]] = []
    for name, sv in cfg.scales.variants.items():
        try:
            model = _build_with_cfg(sv.d_model, sv.depth, sv.n_heads,
                                    sv.max_ctx, cfg)
        except Exception as exc:  # noqa: BLE001
            rows.append((name, {"error": repr(exc), "label": sv.approx_params}))
            continue
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buckets = _module_breakdown(model)
        rows.append((name, {
            "d_model":   sv.d_model,
            "depth":     sv.depth,
            "n_heads":   sv.n_heads,
            "max_ctx":   sv.max_ctx,
            "label":     sv.approx_params,
            "trainable": trainable,
            "buckets":   buckets,
        }))
        del model  # free memory between iterations
    rows.sort(key=lambda r: r[1].get("trainable", 0))
    return rows


def print_table(arch_name: str, rows: List[Tuple[str, dict]]) -> None:
    print(f"\n┏━━ Architecture: {arch_name} ".ljust(85, "━"))
    print("┃")
    print("┃  scale           label    actual    Δ       d_model depth heads  ctx")
    print("┃  ──────────────  ───────  ────────  ──────  ─────── ───── ─────  ────")
    for name, info in rows:
        if "error" in info:
            print(f"┃  {name:<14}  {info['label']:<7}  ERROR: {info['error']}")
            continue
        label = info["label"] or "(none)"
        actual = _fmt_m(info["trainable"])
        # Δ — qualitative gap between declared label and actual count
        delta = "—"
        try:
            label_m = float(label.replace("~", "").replace("M", "").strip())
            actual_m = info["trainable"] / 1e6
            ratio = actual_m / max(label_m, 1.0)
            if ratio > 1.5:
                delta = f"×{ratio:.1f} ✗"
            elif ratio > 1.15:
                delta = f"×{ratio:.2f} ⚠"
            else:
                delta = f"×{ratio:.2f} ✓"
        except Exception:
            pass
        print(f"┃  {name:<14}  {label:<7}  {actual:<8}  {delta:<6}  "
              f"{info['d_model']:<7} {info['depth']:<5} {info['n_heads']:<5}  "
              f"{info['max_ctx']}")
    print("┃")
    print("┃  Module breakdown (per scale):")
    for name, info in rows:
        if "error" in info:
            continue
        b = info["buckets"]
        # Sort buckets by size descending so the dominant cost is first.
        items = sorted(b.items(), key=lambda kv: -kv[1])
        parts = [f"{k}={_fmt_m(v)}" for k, v in items if v > 0]
        print(f"┃    {name:<14}  " + "  ".join(parts))
    print("┗" + "━" * 84)


def main() -> int:
    archs = [
        _REPO / "architectures" / "SmolLM",
        _REPO / "architectures" / "master",
        _REPO / "architectures" / "gpt2",
    ]
    for arch_root in archs:
        if not (arch_root / "arch.neuro").exists():
            print(f"[skip] {arch_root} — no arch.neuro")
            continue
        rows = audit_arch(arch_root)
        print_table(arch_root.name, rows)

    print("\nNOTE: 'actual' counts ONLY the DSL trunk that the optimizer trains.")
    print("      The bio-Brain (subcortical units + neuromodulators) is loaded")
    print("      but frozen at deploy, so its params do NOT contribute to gradient steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
