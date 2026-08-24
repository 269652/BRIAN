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

MIND_VERSION = 1
MIND_FORMAT = "neuroslm.memory.mind.v1"
"""§14.14: CognitiveRuntime's checkpoint tag — deliberately distinct
from FORMAT (the older Brain class's shape) so a wrong-shaped file is
rejected with a clear error at load time instead of being silently
mis-interpreted (mirrors the existing FORMAT guard in load_memory)."""


# ────────────────────────────────────────────────────────────────────────
# Shared narrative-stream (de)serialization — used by both the Brain-
# shaped (save_memory/load_memory) and Mind-shaped (save_mind_memory/
# load_mind_memory, §14.14) checkpoint paths. Extracted from what were
# previously two private closures inside save_memory/load_memory so
# there is exactly one implementation of "how a NarrativeStream
# (de)serializes", not two that could silently diverge.
# ────────────────────────────────────────────────────────────────────────

def _stream_state(stream) -> dict:
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


def _fit(vec_list, d_sem: int):
    """Zero-pad/truncate a stored vector to the CURRENT run's
    embedding width — the same portability property save/load has
    always had for the Brain-shaped format, generalised to take
    ``d_sem`` explicitly (the Mind-shaped path's width comes from
    ``mind.embed_dim()``, not ``brain.cfg.d_sem``)."""
    v = np.asarray(vec_list, dtype=np.float32)
    if v.size < d_sem:
        v = np.pad(v, (0, d_sem - v.size))
    elif v.size > d_sem:
        v = v[:d_sem]
    return v


def _restore_stream(stream, state: dict, device, d_sem: int) -> None:
    import torch
    from .narrative import NarrativeEntry

    s = torch.tensor(_fit(state["summary"], d_sem), device=device,
                     dtype=stream.summary.dtype)
    stream.summary.copy_(s)
    stream.tone.fill_(state.get("tone", 0.0))
    stream.coherence.fill_(state.get("coherence", 1.0))
    stream._tick = int(state.get("tick", 0))
    stream.events = []
    for ev in state.get("events", []):
        emb = torch.tensor(_fit(ev["embedding"], d_sem), dtype=torch.float32)
        stream.events.append(NarrativeEntry(
            content=ev["content"], embedding=emb,
            valence=ev["valence"], salience=ev["salience"],
            timestamp=ev["timestamp"]))


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
    import networkx as nx
    from .causal import CausalRuleStore

    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("format") != FORMAT:
        raise ValueError(f"Unrecognized memory format: {payload.get('format')}")

    d_sem = brain.cfg.d_sem
    device = next(brain.parameters()).device

    def _fit_local(vec_list):
        return _fit(vec_list, d_sem)

    # ── Consolidated graph ──
    g = nx.Graph()
    next_id = 0
    for n in payload["consolidated_nodes"]:
        meta = n.get("meta", {})
        cv = _fit_local(n["vec"]) if n["vec"] is not None else None
        g.add_node(int(n["id"]), content_vec=cv, **meta)
        next_id = max(next_id, int(n["id"]) + 1)
    for e in payload["consolidated_edges"]:
        g.add_edge(int(e["u"]), int(e["v"]), **e.get("weights", {}))
    brain.consolidated.graph = g
    brain.consolidated.next_id = next_id

    # ── Narratives ──
    n = payload["narratives"]
    _restore_stream(brain.narrative_system.autobiographical,
                    n["autobiographical"], device, d_sem)
    _restore_stream(brain.narrative_system.world, n["world"], device, d_sem)
    brain.narrative_system.entities.clear()
    for eid, st in n.get("entities", {}).items():
        ent_stream = brain.narrative_system.get_or_create_entity(eid)
        _restore_stream(ent_stream, st, device, d_sem)

    # ── Causal rules ──
    if hasattr(brain, "causal") and payload.get("causal_rules"):
        brain.causal = CausalRuleStore.from_state(payload["causal_rules"])

    return payload.get("stats", {})


# ────────────────────────────────────────────────────────────────────────
# Mind (CognitiveRuntime) checkpoint — §14.14
# ────────────────────────────────────────────────────────────────────────
#
# Before this, CognitiveRuntime (the runtime chat_daemon.py actually
# drives) had NO persistence at all: EpisodicMemory + NarrativeSystem
# (§14.12) + mined association rules (§14.13) all lived only in
# process memory, lost on every restart. This reuses the SAME format
# pattern as the Brain-shaped checkpoint above (pickle + JSON sidecar,
# the shared _stream_state/_restore_stream helpers) rather than a
# second scheme — MIND_FORMAT keeps the two files from being confused
# for each other at load time.
#
# Survives a chat_daemon PROCESS RESTART on the same box (the
# git-fetch-and-restart live-patch pattern already used this session).
# Does NOT by itself survive destroying/recreating the vast.ai
# instance — that additionally needs the operator to copy the .mem
# file off-box first.

def save_mind_memory(path: str | Path, mind) -> dict:
    """Save ``mind.memory`` (already plain Python — no torch tensors),
    ``mind.narrative`` if attached, and ``mind._mined_rules`` to
    ``path``. Returns the stats dict written.
    """
    import dataclasses

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    episodes = mind.memory.all()

    narrative = getattr(mind, "narrative", None)
    if narrative is not None:
        narratives = {
            "autobiographical": _stream_state(narrative.autobiographical),
            "world": _stream_state(narrative.world),
            "entities": {eid: _stream_state(s)
                        for eid, s in narrative.entities.items()},
        }
    else:
        narratives = None

    mined_rules = [dataclasses.asdict(r)
                   for r in getattr(mind, "_mined_rules", []) or []]

    payload = {
        "version": MIND_VERSION,
        "format": MIND_FORMAT,
        "created_unix": time.time(),
        "episodes": episodes,
        "narratives": narratives,
        "mined_rules": mined_rules,
        "counters": {
            "tick_n": mind._tick_n,
            "boredom": mind._boredom,
            "wander_idx": mind._wander_idx,
        },
        "stats": {
            "n_episodes": len(episodes),
            "n_narrative_events": (
                len(narratives["autobiographical"]["events"])
                + len(narratives["world"]["events"])
                + sum(len(s["events"]) for s in narratives["entities"].values())
                if narratives is not None else 0),
            "n_mined_rules": len(mined_rules),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    sidecar = {
        "version": MIND_VERSION,
        "format": MIND_FORMAT,
        "created_unix": payload["created_unix"],
        "stats": payload["stats"],
    }
    with open(str(path) + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return payload["stats"]


def load_mind_memory(path: str | Path, mind) -> dict:
    """Restore episodic memory, narrative streams, and mined
    association rules into ``mind``. Returns the stats dict from the
    file. Raises ``ValueError`` on a file written in the Brain-shaped
    (or any other unrecognised) format.

    The episodic buffer respects ``mind.memory``'s OWN ``maxlen`` —
    ring-buffer semantics are preserved on load, not silently ignored:
    if the file holds more episodes than the current buffer can hold,
    only the most recent survive.
    """
    import dataclasses

    from neuroslm.cognition.patterns import AssociationRule

    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("format") != MIND_FORMAT:
        raise ValueError(
            f"Unrecognized mind-memory format: {payload.get('format')!r} "
            f"(expected {MIND_FORMAT!r} — a Brain-shaped {FORMAT!r} file "
            f"needs load_memory, not load_mind_memory)")

    # ── Episodic buffer ──
    episodes = payload.get("episodes", [])
    maxlen = mind.memory.buffer.maxlen
    if maxlen is not None and len(episodes) > maxlen:
        episodes = episodes[-maxlen:]
    with mind.memory.lock:
        mind.memory.buffer.clear()
        mind.memory.buffer.extend(episodes)

    # ── Narratives ── only if THIS run has narrative available (either
    # already constructed, or cfg.enable_narrative can lazily build
    # one) — a run with narrative disabled silently skips this, same
    # as any other config-gated capability.
    narratives = payload.get("narratives")
    if narratives is not None:
        ensure = getattr(mind, "_ensure_narrative", None)
        narrative = ensure() if ensure is not None else getattr(mind, "narrative", None)
        if narrative is not None:
            device = narrative.autobiographical.summary.device
            d_sem = narrative.d_sem
            _restore_stream(narrative.autobiographical,
                            narratives["autobiographical"], device, d_sem)
            _restore_stream(narrative.world, narratives["world"], device, d_sem)
            narrative.entities.clear()
            for eid, st in narratives.get("entities", {}).items():
                ent_stream = narrative.get_or_create_entity(eid)
                _restore_stream(ent_stream, st, device, d_sem)

    # ── Mined association rules ──
    mined_state = payload.get("mined_rules") or []
    if mined_state and hasattr(mind, "_mined_rules"):
        mind._mined_rules = [AssociationRule(**r) for r in mined_state]

    # ── Tick/boredom/wander counters — so a resumed mind's curiosity
    # homeostat doesn't silently reset to zero. ──
    counters = payload.get("counters", {})
    if hasattr(mind, "_tick_n") and "tick_n" in counters:
        mind._tick_n = int(counters["tick_n"])
    if hasattr(mind, "_boredom") and "boredom" in counters:
        mind._boredom = float(counters["boredom"])
    if hasattr(mind, "_wander_idx") and "wander_idx" in counters:
        mind._wander_idx = int(counters["wander_idx"])

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
