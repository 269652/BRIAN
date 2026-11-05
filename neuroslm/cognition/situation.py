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


WORLD_STEM = "Where I am now: I am"
SELF_STEM = "Who I am now: I am"
"""Update prompts end on a first-person stem the trunk simply completes.
Run 59311386: bullet lists + "In a few sentences, what I now believe…"
read as a writing exercise ("I need to write a story…", "**Title:**"),
and the story frame leaked into the thoughts through the context
frame."""


def _prose(lines: Sequence[str], each: int = 200) -> str:
    return " ".join(" ".join(l.split())[:each] for l in lines if l.strip())


def complete_model(completion: str, max_chars: int) -> str:
    """The model text: the stem's "I am" + the completion's first
    paragraph (later paragraphs are ramble — "Wait, so I…"), trimmed to
    whole sentences."""
    first = (completion or "").strip().split("\n\n")[0].strip()
    if not first:
        return ""
    return trim_to_sentences(f"I am {first}", max_chars)


def world_prompt(seen: Sequence[str], just_now: Sequence[str]) -> str:
    """Built from what was perceived — never from the previous world
    model (run 59313101: from "Earlier I thought: …" it invented a
    crossroads with signposts and every later version built on it).
    No persona line either: it was parroted back as the model."""
    earlier = f"What I have seen here: {_prose(seen)}\n" if seen else ""
    return (f"{earlier}Just now: {_prose(just_now) or 'nothing new.'}\n\n"
            f"{WORLD_STEM}")


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


def self_prompt(old: str, thoughts: Sequence[str],
                actions: Sequence[str], body: str) -> str:
    earlier = f"Earlier I thought of myself: {old}\n" if old else ""
    did = ", ".join(actions) if actions else "stayed where I was"
    return (f"{earlier}"
            f"Since then I was thinking: {_prose(thoughts) or 'nothing much.'}"
            f"\nI chose to: {did}.\n"
            f"My body: {body or 'unremarkable'}.\n\n"
            f"{SELF_STEM}")
