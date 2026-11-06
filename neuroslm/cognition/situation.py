"""World and self models (§14.16): short texts the mind keeps about
where it is and who it is. The trunk rewrites each from its previous
version plus the evidence since (percepts and speech for the world;
thoughts, chosen movements and the body's neuromodulator state for the
self), at event boundaries — the event model is updated when the event
changes (Zacks et al. 2007). Every version is kept in ``history``."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple


@dataclass
class SituationModel:
    kind: str
    """"world" or "self"."""
    text: str = ""
    history: List[Tuple[int, str]] = field(default_factory=list)
    raw_history: List[Tuple[int, str]] = field(default_factory=list)
    """Every completion as generated, before the grounding filter — what
    is measured (the filtered text would score grounded by
    construction)."""

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


SCENE_NOUNS = (
    # the park's own things
    "bench", "fountain", "pond", "duck", "kiosk", "ice cream", "tree",
    "oak", "birch", "willow", "chestnut", "pine", "maple", "lamp",
    "flower bed", "tulip", "rose", "daisy", "lavender", "dog", "person",
    "coat", "pram", "book", "phone", "fence",
    # what the trunk invents (runs 59313101, 59316881)
    "street", "road", "path", "crossroads", "crossroad", "intersection",
    "signpost", "sign", "traffic light", "building", "house", "car",
    "bus", "train", "station", "bridge", "river", "lake", "sea", "beach",
    "mountain", "hill", "forest", "woods", "gate", "entrance", "statue",
    "playground", "shop", "cafe", "restaurant", "office", "room", "desk",
    "table", "chair", "window", "door", "wall", "computer", "laptop",
    "camera", "image", "photo", "platform", "bicycle", "bike", "cat",
    "bird", "child", "children", "man", "woman", "sky", "sun", "cloud",
    "rain", "snow", "fog", "grass", "sunny", "weather", "temperature",
    "air pressure", "wind", "north", "south", "east", "west",
    # a life the mind never lived (run 59316881's self model)
    "meeting", "study session", "exam", "school", "apartment")
"""Concrete things a text can claim are there. A scene noun in a world
model, self model or thought is grounded only if a percept has named it."""

_SCENE_RES = [(n, re.compile(rf"\b{re.escape(n)}(?:e?s)?\b", re.I))
              for n in sorted(SCENE_NOUNS, key=len, reverse=True)]


def scene_nouns(text: str) -> List[str]:
    """Scene nouns in ``text``, in order, each once; a compound ("traffic
    light") is not also counted as a noun inside it."""
    t, found = text or "", []
    for n, rx in _SCENE_RES:
        for m in rx.finditer(t):
            found.append((m.start(), n))
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    out: List[str] = []
    for _, n in sorted(found):
        if n not in out:
            out.append(n)
    return out


def ungrounded_nouns(text: str, perceived: str) -> List[str]:
    have = set(scene_nouns(perceived))
    return [n for n in scene_nouns(text) if n not in have]


def keep_grounded(text: str, perceived: Set[str]) -> str:
    """``text`` without the sentences that name a scene noun outside
    ``perceived`` (run 59316881: traffic lights, streets, a camera)."""
    keep = [x for x in re.split(r"(?<=[.!?])\s+", (text or "").strip())
            if x and all(n in perceived for n in scene_nouns(x))]
    return " ".join(keep)


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


def self_prompt(thoughts: Sequence[str], actions: Sequence[str],
                body: str) -> str:
    """From what was lived since — never the previous self model (run
    59316881: "I am a 26-year-old woman…" carried on, via "Earlier I
    thought of myself", into study sessions and a meeting)."""
    did = ", ".join(actions) if actions else "stayed where I was"
    return (f"Since then I was thinking: {_prose(thoughts) or 'nothing much.'}"
            f"\nI chose to: {did}.\n"
            f"My body: {body or 'unremarkable'}.\n\n"
            f"{SELF_STEM}")
