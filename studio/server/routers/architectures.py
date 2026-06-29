# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from studio.server.neuro_parser import parse_arch

router = APIRouter(tags=["architectures"])

_REPO = Path(__file__).parent.parent.parent.parent
_ARCH_ROOT = _REPO / "architectures"

_NT_COLORS: dict[str, str] = {
    "glutamate": "#00bcd4",
    "gaba": "#ef5350",
    "dopamine": "#ab47bc",
    "serotonin": "#66bb6a",
    "norepinephrine": "#ffa726",
    "acetylcholine": "#42a5f5",
    "endocannabinoid": "#ffee58",
}
_COLS = 7
_COL_W = 200
_ROW_H = 120

# Pretty labels for the brain-region modules (lib/modules/*.neuro file stems).
_REGION_LABEL: dict[str, str] = {
    "sensory": "Sensory cortex",
    "thalamus": "Thalamus (relay)",
    "world": "World model",
    "amygdala": "Amygdala / insula",
    "qualia": "Qualia / affect",
    "gws": "Global workspace",
    "hippocampus": "Hippocampus",
    "pfc": "Prefrontal cortex",
    "bg": "Basal ganglia",
    "dmn": "Default-mode network",
    "motor": "Motor cortex",
    "nuclei": "Brainstem nuclei",
    "reasoning_cortex": "Reasoning cortex",
    "cortex_specialists": "Cortex experts",
}

_POP_DECL_RE = re.compile(r"^\s*(?:export\s+)?population\s+(\w+)", re.M)


def _population_module_map(arch_dir: Path) -> dict[str, str]:
    """Map population name → the module (lib/modules/*.neuro file stem) it is
    declared in. This is the brain's functional-region decomposition, taken
    straight from the DSL's own module split — arch-local lib wins over the
    shared repo-root lib.
    """
    name2mod: dict[str, str] = {}
    seen: set[Path] = set()
    for libroot in (arch_dir / "lib", _REPO / "lib"):
        if not libroot.exists():
            continue
        for f in sorted(libroot.rglob("*.neuro")):
            if f in seen:
                continue
            seen.add(f)
            try:
                txt = f.read_text(encoding="utf-8")
            except Exception:
                continue
            for m in _POP_DECL_RE.finditer(txt):
                name2mod.setdefault(m.group(1), f.stem)
    return name2mod


def _nfg_to_react_flow(arch_dir: Path, name: str) -> dict | None:
    """Try compile_nfg; convert result to React Flow nodes/edges. Returns None on failure."""
    try:
        sys.path.insert(0, str(_REPO))
        from neuroslm.dsl import compile_nfg  # type: ignore
        nfg = compile_nfg(str(arch_dir))
    except Exception:
        return None

    # Deduplicate nodes by name (NT nodes can appear twice in some archs)
    _seen_names: set[str] = set()
    nfg_nodes = []
    for n in (nfg.nodes if hasattr(nfg, "nodes") else []):
        if n.name not in _seen_names:
            _seen_names.add(n.name)
            nfg_nodes.append(n)
    nfg_edges = list(nfg.edges) if hasattr(nfg, "edges") else []
    if not nfg_nodes:
        return None

    # Topological-ish layout: BFS layers from no-in-edge roots
    in_deg: dict[str, int] = {n.name: 0 for n in nfg_nodes}
    adj: dict[str, list[str]] = {n.name: [] for n in nfg_nodes}
    for e in nfg_edges:
        if e.tgt in in_deg:
            in_deg[e.tgt] += 1
        if e.src in adj:
            adj[e.src].append(e.tgt)

    from collections import deque
    layer: dict[str, int] = {n.name: 0 for n in nfg_nodes}
    visit_budget: dict[str, int] = {n.name: len(nfg_nodes) for n in nfg_nodes}
    queue: deque[str] = deque(n for n, d in in_deg.items() if d == 0)
    while queue:
        node = queue.popleft()
        if visit_budget[node] <= 0:
            continue
        visit_budget[node] -= 1
        cur = layer[node]
        for child in adj.get(node, []):
            if child not in layer:
                continue
            if cur + 1 > layer[child]:
                layer[child] = cur + 1
                queue.append(child)

    # ------------------------------------------------------------------
    # Group populations by their declared param_scope (trunk / bio / ...).
    # This is the brain's actual functional division as stated in the DSL:
    #   trunk = cortex (normal LM gradient), bio = limbic / neuromodulatory
    #   nuclei (detached). Neurotransmitters get their own panel.
    # ------------------------------------------------------------------
    pop_scope: dict[str, str] = {}
    scope_order: list[str] = []
    for ps in (getattr(nfg, "param_scopes", None) or []):
        sname = ps.get("name", "scope")
        if sname not in scope_order:
            scope_order.append(sname)
        for p in ps.get("populations", []):
            pop_scope[p] = sname

    rf_nodes: list[dict] = []
    rf_edges: list[dict] = []
    nt_nodes = [n for n in nfg_nodes if n.kind == "nt"]
    pop_nodes = [n for n in nfg_nodes if n.kind != "nt"]

    # Any population without a declared scope lands in an "other" panel.
    for n in pop_nodes:
        pop_scope.setdefault(n.name, "other")
    if any(s == "other" for s in pop_scope.values()) and "other" not in scope_order:
        scope_order.append("other")

    # ------------------------------------------------------------------
    # Nested grouping: gradient-scope container (trunk/bio) → brain-region
    # module panel (sensory, pfc, hippocampus, …) → population leaf.
    # The module split is the DSL's own functional decomposition; the scope
    # is its gradient-flow division (cortex learns from LM loss, bio nuclei
    # are detached). Neurotransmitters get their own panel below.
    # ------------------------------------------------------------------
    pop_module = _population_module_map(arch_dir)

    _SCOPE_LABEL = {
        "trunk": "CORTEX  -  trunk  (learns from LM loss)",
        "bio": "NEUROMODULATORY  -  bio  (detached gradient)",
        "other": "OTHER",
    }
    POP_W, POP_H = 158, 86
    MOD_HEAD, MOD_PAD = 30, 12
    SCOPE_HEAD, SCOPE_PAD = 34, 18
    MAX_MOD_ROW_W = 1500
    mod_gap, scope_gap = 34, 72

    def _mod_cols(n: int) -> int:
        return 1 if n <= 1 else (2 if n <= 4 else 3)

    def _avg_layer(members: list) -> float:
        return sum(layer.get(m.name, 0) for m in members) / max(len(members), 1)

    from collections import defaultdict
    scope_mods: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for n in pop_nodes:
        scope_mods[pop_scope.get(n.name, "other")][pop_module.get(n.name, "other")].append(n)

    ordered_scopes = [s for s in scope_order if s in scope_mods]
    for s in scope_mods:
        if s not in ordered_scopes:
            ordered_scopes.append(s)

    y_cursor = 0
    for scope in ordered_scopes:
        mods = scope_mods[scope]
        if not mods:
            continue
        scope_gid = f"grp_{scope}"
        container_idx = len(rf_nodes)
        rf_nodes.append({
            "id": scope_gid,
            "type": "group",
            "position": {"x": 0, "y": y_cursor},
            "style": {"width": 100, "height": 100},  # finalized after packing
            "data": {"label": _SCOPE_LABEL.get(scope, scope.upper()),
                     "kind": scope, "tier": "scope"},
        })

        # Module panels flow left→right by average topological layer, wrapping rows.
        mod_items = sorted(mods.items(), key=lambda kv: (_avg_layer(kv[1]), kv[0]))
        cx, cy, row_h, max_right = SCOPE_PAD, SCOPE_HEAD, 0, SCOPE_PAD
        for mod_name, members in mod_items:
            n = len(members)
            cols = _mod_cols(n)
            rows = (n + cols - 1) // cols
            panel_w = cols * POP_W + 2 * MOD_PAD
            panel_h = MOD_HEAD + rows * POP_H + MOD_PAD
            if cx > SCOPE_PAD and cx + panel_w > MAX_MOD_ROW_W:
                cx = SCOPE_PAD
                cy += row_h + mod_gap
                row_h = 0
            # Scope-qualify the id: a region (e.g. basal ganglia) can have part of
            # its populations in trunk and part in bio, so it may appear in both
            # containers — the ids must still be unique.
            mod_gid = f"grp_mod_{scope}_{mod_name}"
            rf_nodes.append({
                "id": mod_gid,
                "type": "group",
                "parentId": scope_gid,
                "extent": "parent",
                "position": {"x": cx, "y": cy},
                "style": {"width": panel_w, "height": panel_h},
                "data": {
                    "label": _REGION_LABEL.get(mod_name, mod_name.replace("_", " ").title()),
                    "kind": scope, "tier": "module", "module": mod_name,
                },
            })
            for i, m in enumerate(sorted(members, key=lambda m: (layer.get(m.name, 0), m.name))):
                r, c = divmod(i, cols)
                props = m.properties or {}
                rf_nodes.append({
                    "id": m.name,
                    "type": "population",
                    "parentId": mod_gid,
                    "extent": "parent",
                    "position": {"x": MOD_PAD + c * POP_W, "y": MOD_HEAD + r * POP_H},
                    "data": {
                        "label": m.name,
                        "name": m.name,
                        "scope": scope,
                        "region": mod_name,
                        "dynamics": props.get("dynamics", "rate_code") or "rate_code",
                        "equation": m.equation or "",
                        "count": props.get("count"),
                        "timescale": props.get("timescale"),
                        "output_dim": props.get("output_dim"),
                    },
                })
            max_right = max(max_right, cx + panel_w)
            cx += panel_w + mod_gap
            row_h = max(row_h, panel_h)

        rf_nodes[container_idx]["style"] = {
            "width": max_right + SCOPE_PAD,
            "height": cy + row_h + SCOPE_PAD,
        }
        y_cursor += rf_nodes[container_idx]["style"]["height"] + scope_gap

    # Neurotransmitter panel — one colored group, each NT in its own colour.
    if nt_nodes:
        nt_cell_w, nt_cell_h = 140, 78
        per_row = 7
        rows = (len(nt_nodes) + per_row - 1) // per_row
        grp_w = min(len(nt_nodes), per_row) * nt_cell_w + 30
        grp_h = rows * nt_cell_h + SCOPE_HEAD
        rf_nodes.append({
            "id": "grp_nt",
            "type": "group",
            "position": {"x": 0, "y": y_cursor},
            "style": {"width": grp_w, "height": grp_h},
            "data": {"label": "NEUROTRANSMITTERS", "kind": "nt", "tier": "scope"},
        })
        for i, n in enumerate(nt_nodes):
            r, c = divmod(i, per_row)
            rf_nodes.append({
                "id": n.name,
                "type": "neurotransmitter",
                "parentId": "grp_nt",
                "extent": "parent",
                "position": {"x": 15 + c * nt_cell_w, "y": SCOPE_HEAD - 4 + r * nt_cell_h},
                "data": {
                    "label": n.name,
                    "name": n.name,
                    "dynamics": n.name,  # header label shows the transmitter name
                    "color": _NT_COLORS.get(n.name.lower()),
                    "equation": n.equation or "",
                },
            })

    # Edges: synapses are the solid feedforward flow; modulations are dashed,
    # dimmer diffuse neuromodulation. Width scales with synaptic weight.
    for i, e in enumerate(nfg_edges):
        nt = (e.nt or "").lower()
        color = _NT_COLORS.get(nt, "#7a7a8c")
        is_mod = e.kind == "modulation"
        weight = getattr(e, "weight", None)
        try:
            sw = max(0.8, min(3.2, float(weight) * 2.6)) if weight is not None else 1.3
        except (TypeError, ValueError):
            sw = 1.3
        style: dict[str, Any] = {"stroke": color, "strokeWidth": sw}
        if is_mod:
            style["strokeDasharray"] = "5 4"
            style["opacity"] = 0.45
        rf_edges.append({
            "id": f"e{i}",
            "source": e.src,
            "target": e.tgt,
            "label": e.nt or "",
            "style": style,
            "data": {
                "kind": e.kind,
                "nt": e.nt,
                "weight": weight,
                "effect": getattr(e, "effect", None),
                "equation": getattr(e, "equation", None),
            },
        })

    source = (arch_dir / "arch.neuro").read_text(encoding="utf-8")
    return {"name": name, "source": source, "nodes": rf_nodes, "edges": rf_edges}


def _list_arch_dirs() -> list[Path]:
    if not _ARCH_ROOT.exists():
        return []
    return sorted(
        p for p in _ARCH_ROOT.iterdir()
        if p.is_dir() and (p / "arch.neuro").exists()
    )


class ArchSummary(BaseModel):
    name: str
    has_config: bool
    has_fitness: bool
    kind: str


class ArchDetail(BaseModel):
    name: str
    source: str
    nodes: list[dict]
    edges: list[dict]


class SaveRequest(BaseModel):
    source: str


@router.get("", response_model=list[ArchSummary])
def list_architectures() -> list[ArchSummary]:
    out = []
    for p in _list_arch_dirs():
        src = (p / "arch.neuro").read_text(encoding="utf-8")
        # Quick kind extraction
        import re
        m = re.search(r"kind\s*:\s*(\w+)", src)
        kind = m.group(1) if m else "custom"
        out.append(ArchSummary(
            name=p.name,
            has_config=(p / "config.neuro").exists(),
            has_fitness=(p / "fitness.neuro").exists(),
            kind=kind,
        ))
    return out


@router.get("/{name}", response_model=ArchDetail)
def get_architecture(name: str) -> ArchDetail:
    arch_dir = _ARCH_ROOT / name
    neuro_file = arch_dir / "arch.neuro"
    if not neuro_file.exists():
        raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")

    # Try full NFG pipeline first (population-based archs like SmolLM)
    nfg_result = _nfg_to_react_flow(arch_dir, name)
    if nfg_result:
        return ArchDetail(**nfg_result)

    # Fall back to lightweight parser (model{}-based archs like gpt2/llama/qwen)
    source = neuro_file.read_text(encoding="utf-8")
    parsed = parse_arch(source, name)
    return ArchDetail(**parsed)


@router.put("/{name}")
def save_architecture(name: str, body: SaveRequest) -> dict:
    arch_dir = _ARCH_ROOT / name
    arch_dir.mkdir(parents=True, exist_ok=True)
    (arch_dir / "arch.neuro").write_text(body.source, encoding="utf-8")
    return {"saved": True, "name": name}


@router.post("/{name}/compile")
def compile_architecture(name: str) -> dict:
    """Compile arch.neuro → DNA via brian dna compile."""
    result = subprocess.run(
        [sys.executable, "-m", "neuroslm.cli", "dna", "compile", f"architectures/{name}"],
        capture_output=True, text=True, cwd=_REPO
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
