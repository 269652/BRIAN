"""Narrative memory: autobiographical, world, and entity identity narratives.

Maintains coherent ongoing narratives that the brain uses to:
  1. Ground its identity (autobiographical narrative)
  2. Model the external world (world narrative)
  3. Track entities it communicates with (entity narratives)

Each narrative is a compressed, evolving representation updated via
gated writes (LSTM-cell style). During mind wandering, narratives
cross-pollinate (associative drift). Narratives provide top-down
context to the DMN and PFC.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F


# Legacy interface (kept for backward compatibility)
class NarrativeBuffer:
    def __init__(self, maxlen=4096):
        self.buffer = []
        self.maxlen = maxlen
        self.lock = threading.Lock()

    def update(self, text):
        with self.lock:
            self.buffer.append(text)
            if len(self.buffer) > self.maxlen:
                self.buffer = self.buffer[-self.maxlen:]

    def get(self, n=32):
        with self.lock:
            return self.buffer[-n:]

    def all(self):
        with self.lock:
            return list(self.buffer)


@dataclass
class NarrativeEntry:
    """A single event in a narrative."""
    content: str
    embedding: torch.Tensor  # (D,)
    valence: float = 0.0
    salience: float = 0.0
    timestamp: int = 0
    causal_links: list[int] = field(default_factory=list)


class NarrativeStream(nn.Module):
    """A single narrative stream (autobiographical, world, or entity)."""

    def __init__(self, d_sem: int, max_events: int = 256):
        super().__init__()
        self.d_sem = d_sem
        self.max_events = max_events
        self.register_buffer("summary", torch.zeros(d_sem))
        self.register_buffer("tone", torch.zeros(1))
        self.register_buffer("coherence", torch.ones(1))
        self.write_gate = nn.Sequential(nn.Linear(d_sem * 2, d_sem), nn.Sigmoid())
        self.write_transform = nn.Sequential(nn.Linear(d_sem * 2, d_sem), nn.Tanh())
        self.forget_gate = nn.Sequential(nn.Linear(d_sem * 2, d_sem), nn.Sigmoid())
        self.events: list[NarrativeEntry] = []
        self._tick = 0

    @torch.no_grad()
    def write(self, event_embedding: torch.Tensor, content: str = "",
              valence: float = 0.0, salience: float = 0.5):
        device = self.summary.device
        event_embedding = event_embedding.to(device)
        if event_embedding.dim() > 1:
            event_embedding = event_embedding.mean(0)
        combined = torch.cat([self.summary, event_embedding])
        forget = self.forget_gate(combined.unsqueeze(0)).squeeze(0)
        write_g = self.write_gate(combined.unsqueeze(0)).squeeze(0)
        candidate = self.write_transform(combined.unsqueeze(0)).squeeze(0)
        self.summary = forget * self.summary + write_g * candidate
        self.tone = 0.9 * self.tone + 0.1 * valence
        cos = F.cosine_similarity(
            event_embedding.unsqueeze(0), self.summary.unsqueeze(0)).item()
        self.coherence = 0.9 * self.coherence + 0.1 * cos
        entry = NarrativeEntry(content=content, embedding=event_embedding.cpu(),
                               valence=valence, salience=salience, timestamp=self._tick)
        self.events.append(entry)
        if len(self.events) > self.max_events:
            self.events.sort(key=lambda e: e.salience, reverse=True)
            self.events = self.events[:self.max_events]
        self._tick += 1

    def read(self) -> torch.Tensor:
        return self.summary

    def query(self, query_vec: torch.Tensor, topk: int = 3) -> list[NarrativeEntry]:
        if not self.events:
            return []
        query_cpu = query_vec.detach().cpu().flatten()
        scored = []
        for e in self.events:
            emb = e.embedding.flatten()[:query_cpu.size(0)]
            if emb.size(0) < query_cpu.size(0):
                continue
            sim = F.cosine_similarity(query_cpu.unsqueeze(0), emb.unsqueeze(0)).item()
            scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:topk]]

    def info(self) -> dict:
        return {"n_events": len(self.events), "tone": float(self.tone),
                "coherence": float(self.coherence), "tick": self._tick}


class NarrativeSystem(nn.Module):
    """All narrative streams: autobiographical, world, and entities."""

    def __init__(self, d_sem: int, max_entities: int = 16):
        super().__init__()
        self.d_sem = d_sem
        self.max_entities = max_entities
        self.autobiographical = NarrativeStream(d_sem, max_events=512)
        self.world = NarrativeStream(d_sem, max_events=256)
        self.entities: dict[str, NarrativeStream] = {}
        self.integrate = nn.Sequential(
            nn.Linear(d_sem * 3, d_sem * 2), nn.GELU(),
            nn.Linear(d_sem * 2, d_sem),
        )
        self.to_thought = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem), nn.Tanh(),
        )

    def get_or_create_entity(self, entity_id: str) -> NarrativeStream:
        if entity_id not in self.entities:
            if len(self.entities) >= self.max_entities:
                least_active = min(self.entities.items(), key=lambda x: x[1]._tick)
                del self.entities[least_active[0]]
            self.entities[entity_id] = NarrativeStream(
                self.d_sem, max_events=128).to(self.autobiographical.summary.device)
        return self.entities[entity_id]

    @torch.no_grad()
    def record_autobiographical(self, embedding: torch.Tensor,
                                content: str = "", valence: float = 0.0,
                                salience: float = 0.5):
        self.autobiographical.write(embedding, content, valence, salience)

    @torch.no_grad()
    def record_world(self, embedding: torch.Tensor, content: str = "",
                     valence: float = 0.0, salience: float = 0.5):
        self.world.write(embedding, content, valence, salience)

    @torch.no_grad()
    def record_entity(self, entity_id: str, embedding: torch.Tensor,
                      content: str = "", valence: float = 0.0):
        stream = self.get_or_create_entity(entity_id)
        stream.write(embedding, content, valence)

    def narrative_context(self) -> torch.Tensor:
        """(D,) combined auto + world narrative context."""
        auto = self.autobiographical.read()
        world = self.world.read()
        combined = torch.cat([auto, world])
        return self.to_thought(combined.unsqueeze(0)).squeeze(0)

    def full_context(self, active_entity: str | None = None) -> torch.Tensor:
        """(D,) full context including active entity."""
        auto = self.autobiographical.read()
        world = self.world.read()
        if active_entity and active_entity in self.entities:
            entity = self.entities[active_entity].read()
        else:
            entity = torch.zeros_like(auto)
        combined = torch.cat([auto, world, entity])
        return self.integrate(combined.unsqueeze(0)).squeeze(0)

    def info(self) -> dict:
        return {
            "autobiographical": self.autobiographical.info(),
            "world": self.world.info(),
            "entities": {k: v.info() for k, v in self.entities.items()},
        }

