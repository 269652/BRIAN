"""World and self models (§14.16): short texts the mind keeps about
where it is and who it is. The trunk rewrites each from its previous
version plus the evidence since (percepts and speech for the world;
thoughts, chosen movements and the body's neuromodulator state for the
self), at event boundaries — the event model is updated when the event
changes (Zacks et al. 2007). Every version is kept in ``history``."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

_NO_MODEL = "nothing yet"


@dataclass
class SituationModel:
    kind: str
    """"world" or "self"."""
    text: str = ""
    history: List[Tuple[int, str]] = field(default_factory=list)

    def update(self, tick: int, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        self.text = text
        self.history.append((tick, text))
        return True


def trim_to_sentences(text: str, max_chars: int) -> str:
    """Whitespace collapsed, cut at the last sentence end within
    ``max_chars`` (a hard cut only when there is none)."""
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    head = t[:max_chars]
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", head)]
    return head[:ends[-1]].rstrip() if ends else head.rstrip()


def _bullets(lines: Sequence[str]) -> str:
    return "\n".join(f"- {' '.join(l.split())}" for l in lines) or "- nothing"


def world_prompt(persona: str, old: str, noticed: Sequence[str]) -> str:
    return (f"{persona}\n\n"
            f"What I believed about my surroundings: {old or _NO_MODEL}\n"
            f"What I have noticed since:\n{_bullets(noticed)}\n\n"
            f"In a few sentences, what I now believe about my surroundings:")


def body_state(levels: Dict[str, float], baselines: Dict[str, float]) -> str:
    """Each neuromodulator against its baseline, in words the trunk can
    read — interoception for the self model."""
    parts = []
    for ch, v in levels.items():
        b = baselines.get(ch)
        if b is None:
            continue
        rel = ("above" if v > b + 1e-9 else "below" if v < b - 1e-9
               else "at")
        parts.append(f"{ch} {v:.2f} ({rel} baseline {b:.2f})")
    return ", ".join(parts)


def self_prompt(persona: str, old: str, thoughts: Sequence[str],
                actions: Sequence[str], body: str) -> str:
    return (f"{persona}\n\n"
            f"What I believed about myself: {old or _NO_MODEL}\n"
            f"What I thought since:\n{_bullets(thoughts)}\n"
            f"What I did:\n{_bullets(actions)}\n"
            f"How my body is: {body or 'unremarkable'}\n\n"
            f"In a few sentences, what I now believe about myself — who I "
            f"am and what I am doing:")
