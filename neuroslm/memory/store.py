"""Portable memory checkpoint store.

Memory is stored independently from model weights in `.mem` files. This
makes hierarchical memory transferable to a fresh model, shippable via
Git LFS, and human-inspectable (a small JSON header + a binary numpy
payload).

Format
------
A `.mem` file is a single Python pickle of a dict:

    {
      "version":           int,
      "format":            "neuroslm.memory.v1",
      "created_unix":      float,
      "consolidated_nodes": [{ "id": int, "vec": list[float], "meta": {...} }],
      "consolidated_edges": [{ "u": int, "v": int, "weights": {...} }],
      "narratives": {
          "autobiographical": {"summary": list[float], "events": [...]},
          "world":            {"summary": list[float], "events": [...]},
          "entities":         {entity_id: {...}, ...},
      },
      "causal_rules":      [...],   # CausalRuleStore.to_state()["rules"]
      "stats":             {...},
    }

The reason for pickle (not JSON): preserves numpy dtypes faithfully and
is dramatically smaller than JSON for embedding-heavy payloads. Files
remain ~5-50 MB even with thousands of nodes.

A plain `.json` sidecar is also written so humans can inspect counts /
labels without unpickling.
"""
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
import numpy as np


VERSION = 1
FORMAT = "neuroslm.memory.v1"


# ────────────────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────────────────

def save_memory(path: str | Path, brain) -> dict:
    """Save brain.consolidated + brain.narrative_system + brain.causal
    to `path` (a `.mem` file). Returns the stats dict written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = []
    for nid, data in brain.consolidated.graph.nodes(data=True):
        cv = data.get("content_vec")
        meta = {k: v for k, v in data.items() if k != "content_vec"}
        # Sanitize numpy types in meta to plain python
        meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in meta.items()}
        nodes.append({
            "id": int(nid),
            "vec": (np.asarray(cv, dtype=np.float32).flatten().tolist()
                    if cv is not None else None),
            "meta": meta,
        })

    edges = []
    for u, v, data in brain.consolidated.graph.edges(data=True):
        edges.append({"u": int(u), "v": int(v),
                      "weights": {k: float(val) for k, val in data.items()}})

    def _stream_state(stream):
        return {
            "summary": stream.summary.detach().cpu().float().tolist(),
            "tone": float(stream.tone),
            "coherence": float(stream.coherence),
            "tick": int(stream._tick),
            "events": [
                {"content": e.content,
                 "embedding": e.embedding.float().tolist(),
                 "valence": e.valence, "salience": e.salience,
                 "timestamp": e.timestamp}
                for e in stream.events
            ],
        }

    narratives = {
        "autobiographical": _stream_state(brain.narrative_system.autobiographical),
        "world":            _stream_state(brain.narrative_system.world),
        "entities": {eid: _stream_state(s)
                     for eid, s in brain.narrative_system.entities.items()},
    }

    causal_state = brain.causal.to_state() if hasattr(brain, "causal") else {}

    payload = {
        "version": VERSION,
        "format":  FORMAT,
        "created_unix": time.time(),
        "consolidated_nodes": nodes,
        "consolidated_edges": edges,
        "narratives":         narratives,
        "causal_rules":       causal_state,
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_causal_rules": len(causal_state.get("rules", [])),
            "n_entities": len(narratives["entities"]),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # human-readable sidecar
    sidecar = {
        "version": VERSION,
        "format":  FORMAT,
        "created_unix": payload["created_unix"],
        "stats": payload["stats"],
        "entity_ids": list(narratives["entities"].keys()),
        "causal_labels": [r.get("label", "") for r in
                          causal_state.get("rules", [])][:50],
    }
    with open(str(path) + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return payload["stats"]


# ────────────────────────────────────────────────────────────────────────
# Load / transfer
# ────────────────────────────────────────────────────────────────────────

def load_memory(path: str | Path, brain) -> dict:
    """Restore consolidated graph, narratives, and causal rules into
    `brain`. The brain's *weights* are untouched; only memory state is
    overwritten. Returns the stats dict from the file.

    Compatible across model architectures as long as `d_sem` matches;
    embeddings shorter than current d_sem are zero-padded, longer are
    truncated.
    """
    import torch
    import networkx as nx
    from .causal import CausalRuleStore

    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("format") != FORMAT:
        raise ValueError(f"Unrecognized memory format: {payload.get('format')}")

    d_sem = brain.cfg.d_sem
    device = next(brain.parameters()).device

    def _fit(vec_list):
        v = np.asarray(vec_list, dtype=np.float32)
        if v.size < d_sem:
            v = np.pad(v, (0, d_sem - v.size))
        elif v.size > d_sem:
            v = v[:d_sem]
        return v

    # ── Consolidated graph ──
    g = nx.Graph()
    next_id = 0
    for n in payload["consolidated_nodes"]:
        meta = n.get("meta", {})
        cv = _fit(n["vec"]) if n["vec"] is not None else None
        g.add_node(int(n["id"]), content_vec=cv, **meta)
        next_id = max(next_id, int(n["id"]) + 1)
    for e in payload["consolidated_edges"]:
        g.add_edge(int(e["u"]), int(e["v"]), **e.get("weights", {}))
    brain.consolidated.graph = g
    brain.consolidated.next_id = next_id

    # ── Narratives ──
    def _restore_stream(stream, state):
        from .narrative import NarrativeEntry
        s = torch.tensor(_fit(state["summary"]), device=device,
                         dtype=stream.summary.dtype)
        stream.summary.copy_(s)
        stream.tone.fill_(state.get("tone", 0.0))
        stream.coherence.fill_(state.get("coherence", 1.0))
        stream._tick = int(state.get("tick", 0))
        stream.events = []
        for ev in state.get("events", []):
            emb = torch.tensor(_fit(ev["embedding"]), dtype=torch.float32)
            stream.events.append(NarrativeEntry(
                content=ev["content"], embedding=emb,
                valence=ev["valence"], salience=ev["salience"],
                timestamp=ev["timestamp"]))

    n = payload["narratives"]
    _restore_stream(brain.narrative_system.autobiographical, n["autobiographical"])
    _restore_stream(brain.narrative_system.world, n["world"])
    brain.narrative_system.entities.clear()
    for eid, st in n.get("entities", {}).items():
        ent_stream = brain.narrative_system.get_or_create_entity(eid)
        _restore_stream(ent_stream, st)

    # ── Causal rules ──
    if hasattr(brain, "causal") and payload.get("causal_rules"):
        brain.causal = CausalRuleStore.from_state(payload["causal_rules"])

    return payload.get("stats", {})


# ────────────────────────────────────────────────────────────────────────
# Find latest .mem in a directory
# ────────────────────────────────────────────────────────────────────────

def latest_memory(directory: str | Path) -> Path | None:
    p = Path(directory)
    if not p.exists():
        return None
    cands = sorted(p.glob("*.mem"), key=lambda f: f.stat().st_mtime)
    return cands[-1] if cands else None
