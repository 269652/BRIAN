"""Virtual environments: simulated worlds that feed sensory input to cortices.

The brain doesn't exist in a vacuum — it processes a continuous stream of
sensory experience. These virtual environments generate rich, structured
sensory input that grounds the model's cognition in embodied experience.

Each environment defines:
  - Visual field: what the agent "sees" (described as text → embedded)
  - Auditory field: ambient sounds and events
  - Proprioception: body state, posture, comfort
  - Interoception: hunger, temperature, fatigue, arousal
  - Temporal flow: time of day, passage of time, events unfolding
  - Emotional valence: the mood/feeling the environment induces
  - Narrative arc: how the scene evolves over time

Environments run as generators, yielding a new sensory frame each tick.
"""
from __future__ import annotations
