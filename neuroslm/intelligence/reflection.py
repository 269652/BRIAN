"""Spontaneous self-reflection.

When the brain is idle (not processing input), this module triggers a
self-reflection loop that:

  1. Reads the autobiographical narrative summary.
  2. Reads the world narrative + active entity summaries (theory of mind).
  3. Composes a "reflection prompt" embedding via the integrate head.
  4. Runs the language cortex in generation mode to produce reflective text.
  5. Writes the resulting thought back into the autobiographical stream
     (a recursive identity-formation loop).

This is the operational implementation of "the self is the story the
brain keeps telling itself" (Dennett's narrative-self model).

It also detects entity interactions and produces theory-of-mind
predictions: given an entity's narrative + a candidate action, it
predicts the entity's likely valence response. These predictions feed
into the IntelligenceMetrics ToM accuracy counter when ground truth is
later observed (e.g., the entity actually says they are pleased/upset).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpontaneousReflection(nn.Module):
    """Generates reflective thoughts during idle compute.

    Args:
        d_sem: shared semantic embedding dim.
        valence_dim: dim of the small ToM head's output (default 1).
    """

    def __init__(self, d_sem: int):
        super().__init__()
        self.d_sem = d_sem
        # Compose autobiographical + world + entity into a thought seed
        self.thought_seed = nn.Sequential(
            nn.Linear(d_sem * 3, d_sem * 2), nn.GELU(),
            nn.Linear(d_sem * 2, d_sem), nn.Tanh(),
        )
        # Theory of mind head: (entity_summary, action_emb) -> predicted valence
        self.tom_head = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem), nn.GELU(),
            nn.Linear(d_sem, 1), nn.Tanh(),
        )
        # Identity coherence head: how "self-like" is a given embedding?
        self.identity_head = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem), nn.GELU(),
            nn.Linear(d_sem, 1), nn.Sigmoid(),
        )

    @torch.no_grad()
    def reflect(self, narrative_system,
                entity_id: str | None = None) -> torch.Tensor:
        """Produce a reflection-thought embedding (d_sem,).

        This embedding can be fed to the language cortex's generate path
        as the initial semantic state to produce coherent reflective text.
        """
        device = next(self.parameters()).device
        auto = narrative_system.autobiographical.summary.to(device)
        world = narrative_system.world.summary.to(device)
        if entity_id and entity_id in narrative_system.entities:
            ent = narrative_system.entities[entity_id].summary.to(device)
        else:
            ent = torch.zeros_like(auto)
        seed = self.thought_seed(torch.cat([auto, world, ent]).unsqueeze(0)).squeeze(0)
        return seed

    @torch.no_grad()
    def predict_entity_response(self, narrative_system, entity_id: str,
                                action_embedding: torch.Tensor) -> float:
        """Predict the valence of an entity's response to a candidate action.

        Returns a scalar in [-1, 1]. Used to prime social planning *and*
        to be evaluated for theory-of-mind accuracy.
        """
        device = next(self.parameters()).device
        if entity_id not in narrative_system.entities:
            return 0.0
        ent = narrative_system.entities[entity_id].summary.to(device)
        action = action_embedding.to(device).flatten()
        if action.size(0) < ent.size(0):
            action = F.pad(action, (0, ent.size(0) - action.size(0)))
        elif action.size(0) > ent.size(0):
            action = action[:ent.size(0)]
        x = torch.cat([ent, action]).unsqueeze(0)
        return float(self.tom_head(x).squeeze().item())

    @torch.no_grad()
    def identity_score(self, candidate: torch.Tensor,
                       narrative_system) -> float:
        """How consistent is this embedding with current self-narrative?

        Used as a brake on identity drift: if score drops below a
        threshold, the autobiographical write is skipped (the model
        rejects experiences it cannot integrate).
        """
        device = next(self.parameters()).device
        cand = candidate.to(device).flatten()
        auto = narrative_system.autobiographical.summary.to(device)
        if cand.size(0) < auto.size(0):
            cand = F.pad(cand, (0, auto.size(0) - cand.size(0)))
        else:
            cand = cand[:auto.size(0)]
        x = torch.cat([auto, cand]).unsqueeze(0)
        return float(self.identity_head(x).squeeze().item())
