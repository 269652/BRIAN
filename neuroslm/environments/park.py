"""Grounded park: a persistent world the mind walks through.

Unlike the template scenes in ``virtual_world``, every percept here is
derived from world state: objects have positions, the agent has a pose,
and a ``PerceptChannel`` decides what reaches the mind. The default
``GeometryChannel`` computes field of view, distance and occlusion; a
rendered-image channel (frames → vision model) can replace it without
touching the world or the action loop.

Conventions: heading θ in radians, forward = (cos θ, sin θ); bearing is
the signed angle from forward, positive = left.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from neuroslm.environments.virtual_world import Environment, SensoryFrame

PARK_ACTIONS = ("walk", "turn_left", "turn_right", "look", "approach")

_OPAQUE = {"tree", "kiosk"}
_MOVERS = {"dog", "person"}


@dataclass
class ParkObject:
    id: str
    kind: str
    x: float
    y: float
    radius: float = 0.5
    solid: bool = True
    attrs: Dict[str, str] = field(default_factory=dict)
    sound: Optional[str] = None
    speed: float = 0.0
    heading: float = 0.0

    def describe(self, detailed: bool) -> str:
        if not detailed:
            return self.kind
        adj = " ".join(self.attrs[k] for k in ("colour", "species")
                       if k in self.attrs)
        text = f"{adj} {self.kind}".strip()
        if "wearing" in self.attrs:
            text += f" in a {self.attrs['wearing']}"
        if "doing" in self.attrs:
            text += f" {self.attrs['doing']}"
        return text


@dataclass
class Percept:
    text: str
    object_ids: Tuple[str, ...]
    modality: str
    distance: Optional[float] = None


class PerceptChannel(Protocol):
    def perceive(self, park: "Park") -> List[Percept]: ...


def _wrap(a: float) -> float:
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0.0:
        a += 2 * math.pi
    return a - math.pi


def _article(text: str) -> str:
    return ("an " if text[:1].lower() in "aeiou" else "a ") + text


class GeometryChannel:
    """What a viewer at the agent's pose can see and hear: objects in
    the field of view, within the light-limited view distance, not
    hidden behind a nearer opaque object; sounds within earshot from any
    direction. Detail falls off with distance."""

    def __init__(self, fov_deg: float = 120.0, view_distance: float = 25.0,
                 detail_distance: float = 10.0,
                 hearing_distance: float = 30.0) -> None:
        self.half_fov = math.radians(fov_deg) / 2.0
        self.view_distance = float(view_distance)
        self.detail_distance = float(detail_distance)
        self.hearing_distance = float(hearing_distance)

    def perceive(self, park: "Park") -> List[Percept]:
        ax, ay, th = park.agent_x, park.agent_y, park.heading
        reach = self.view_distance * park.light
        polar = []
        for o in park.objects:
            dx, dy = o.x - ax, o.y - ay
            d = math.hypot(dx, dy)
            polar.append((o, d, _wrap(math.atan2(dy, dx) - th)))
        out: List[Percept] = []
        for o, d, b in sorted(polar, key=lambda t: t[1]):
            if d > reach or abs(b) > self.half_fov:
                continue
            if self._occluded(o, d, ax, ay, polar):
                continue
            desc = _article(o.describe(d <= self.detail_distance))
            out.append(Percept(_sentence(f"{desc}, {_distance_word(d)} "
                                         f"{_bearing_word(b)}"),
                               (o.id,), "visual", d))
        for o, d, b in polar:
            if o.sound and d <= self.hearing_distance:
                out.append(Percept(_sentence(f"{_article(o.kind)} {o.sound} "
                                             f"{_direction_word(b)}"),
                                   (o.id,), "auditory", d))
        return out

    @staticmethod
    def _occluded(o, d, ax, ay, polar) -> bool:
        ux, uy = (o.x - ax) / max(d, 1e-9), (o.y - ay) / max(d, 1e-9)
        for p, dp, _ in polar:
            if p is o or p.kind not in _OPAQUE or dp >= d:
                continue
            t = (p.x - ax) * ux + (p.y - ay) * uy
            if t <= 0.0:
                continue
            perp = abs((p.x - ax) * uy - (p.y - ay) * ux)
            if perp < p.radius:
                return True
        return False


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def _distance_word(d: float) -> str:
    if d < 4.0:
        return "close"
    if d < 10.0:
        return "a few metres"
    if d < 20.0:
        return "some way off"
    return "far off"


def _bearing_word(b: float) -> str:
    if abs(b) < math.radians(15):
        return "ahead"
    return "to the left" if b > 0 else "to the right"


def _direction_word(b: float) -> str:
    if abs(b) < math.radians(45):
        return "ahead"
    if abs(b) > math.radians(135):
        return "behind you"
    return "to the left" if b > 0 else "to the right"


def _time_of_day(minute: float) -> Tuple[str, float]:
    h = (minute / 60.0) % 24.0
    if 5 <= h < 8:
        return "dawn", 0.7
    if 8 <= h < 12:
        return "morning", 1.0
    if 12 <= h < 17:
        return "afternoon", 1.0
    if 17 <= h < 20:
        return "evening", 0.7
    return "night", 0.35


_SPECIES = ("oak", "birch", "willow", "chestnut", "pine", "maple")
_COLOURS = ("red", "green", "blue", "yellow", "white", "grey")
_FLOWERS = ("tulip", "rose", "daisy", "lavender")
_ACTIVITIES = ("reading a book", "jogging", "feeding the ducks",
               "talking on the phone", "pushing a pram", "walking slowly")


class Park(Environment):
    """A procedural park: trees, benches, a pond, lamps, flower beds, a
    kiosk, a fountain, a wandering dog and passers-by. ``step(action)``
    applies one of ``PARK_ACTIONS`` (``approach:<object id>`` needs the
    object to be perceived), advances the movers and the clock, and
    returns the channel's percepts as a ``SensoryFrame``."""

    name = "park"
    n_actions = len(PARK_ACTIONS)

    def __init__(self, seed: int = 42, size: float = 60.0,
                 channel: Optional[PerceptChannel] = None,
                 start_minute: float = 14 * 60, minutes_per_tick: float = 2.0,
                 stride: float = 1.5, turn_deg: float = 45.0,
                 _objects: Optional[Sequence[ParkObject]] = None,
                 _agent: Optional[Tuple[float, float, float]] = None) -> None:
        super().__init__(seed)
        self.size = float(size)
        self.channel = channel if channel is not None else GeometryChannel()
        self.minute = float(start_minute)
        self.minutes_per_tick = float(minutes_per_tick)
        self.stride = float(stride)
        self.turn = math.radians(turn_deg)
        self.agent_radius = 0.3
        self.objects: List[ParkObject] = (list(_objects) if _objects is not None
                                          else self._layout())
        if _agent is not None:
            self.agent_x, self.agent_y, h = map(float, _agent)
            self.heading = _wrap(h)
        else:
            self.agent_x, self.agent_y = self._free_spot(1.0)
            self.heading = _wrap(self.rng.uniform(-math.pi, math.pi))
        self.seen_ids: set = set()
        self.last_percepts: List[Percept] = []

    @classmethod
    def from_objects(cls, objects: Sequence[ParkObject],
                     agent: Tuple[float, float, float], **kw) -> "Park":
        return cls(_objects=objects, _agent=agent, **kw)

    # ---- world --------------------------------------------------------

    @property
    def time_of_day(self) -> str:
        return _time_of_day(self.minute)[0]

    @property
    def light(self) -> float:
        return _time_of_day(self.minute)[1]

    def _fits(self, x: float, y: float, r: float,
              placed: Sequence[ParkObject]) -> bool:
        if not (r <= x <= self.size - r and r <= y <= self.size - r):
            return False
        return all(math.hypot(x - o.x, y - o.y) >= r + o.radius + 0.5
                   for o in placed if o.solid)

    def _place(self, placed: List[ParkObject], oid: str, kind: str,
               radius: float, **kw) -> None:
        for _ in range(500):
            x = self.rng.uniform(radius, self.size - radius)
            y = self.rng.uniform(radius, self.size - radius)
            if self._fits(x, y, radius, placed):
                placed.append(ParkObject(oid, kind, x, y, radius, **kw))
                return

    def _layout(self) -> List[ParkObject]:
        r = self.rng
        objs: List[ParkObject] = []
        self._place(objs, "pond", "pond", r.uniform(4.0, 6.0),
                    attrs={"doing": "with ducks on it"})
        self._place(objs, "fountain", "fountain", 1.5,
                    attrs={"colour": "stone"}, sound="splashing")
        self._place(objs, "kiosk", "kiosk", 1.5,
                    attrs={"colour": r.choice(_COLOURS),
                           "doing": "selling ice cream"})
        for i in range(14):
            self._place(objs, f"tree{i}", "tree", r.uniform(0.8, 1.5),
                        attrs={"species": r.choice(_SPECIES)})
        for i in range(6):
            self._place(objs, f"bench{i}", "bench", 0.8,
                        attrs={"colour": r.choice(_COLOURS)})
        for i in range(5):
            self._place(objs, f"lamp{i}", "lamp", 0.3)
        for i in range(4):
            self._place(objs, f"flowers{i}", "flower bed", 1.2,
                        attrs={"colour": f"{r.choice(_COLOURS)} "
                                         f"{r.choice(_FLOWERS)}"})
        self._place(objs, "dog", "dog", 0.4, solid=False, speed=2.5,
                    heading=r.uniform(-math.pi, math.pi),
                    attrs={"colour": r.choice(("brown", "black",
                                               "white", "spotted"))})
        for i in range(3):
            self._place(objs, f"person{i}", "person", 0.4, solid=False,
                        speed=1.0, heading=r.uniform(-math.pi, math.pi),
                        attrs={"wearing": f"{r.choice(_COLOURS)} coat",
                               "doing": r.choice(_ACTIVITIES)})
        return objs

    def _free_spot(self, margin: float) -> Tuple[float, float]:
        for _ in range(1000):
            x = self.rng.uniform(margin, self.size - margin)
            y = self.rng.uniform(margin, self.size - margin)
            if not self._blocked(x, y):
                return x, y
        raise RuntimeError("no free ground in the park")

    def _blocked(self, x: float, y: float) -> bool:
        return self._obstacle(x, y) is not None or not (
            0.0 <= x <= self.size and 0.0 <= y <= self.size)

    def _obstacle(self, x: float, y: float) -> Optional[ParkObject]:
        for o in self.objects:
            if o.solid and math.hypot(x - o.x, y - o.y) < (
                    o.radius + self.agent_radius):
                return o
        return None

    def _move_movers(self) -> None:
        for o in self.objects:
            if o.kind not in _MOVERS or o.speed <= 0.0:
                continue
            o.heading = _wrap(o.heading + self.rng.gauss(0.0, 0.6))
            for _ in range(8):
                nx = o.x + o.speed * math.cos(o.heading)
                ny = o.y + o.speed * math.sin(o.heading)
                inside = (o.radius <= nx <= self.size - o.radius
                          and o.radius <= ny <= self.size - o.radius)
                if inside and self._obstacle(nx, ny) is None:
                    o.x, o.y = nx, ny
                    break
                o.heading = _wrap(o.heading + self.rng.uniform(1.0, 3.0))
            if o.kind == "dog":
                o.sound = "barking" if self.rng.random() < 0.25 else None

    # ---- agent --------------------------------------------------------

    def perceive(self) -> List[Percept]:
        return self.channel.perceive(self)

    def step(self, action: Optional[str] = None) -> SensoryFrame:  # type: ignore[override]
        act = action or "look"
        target = None
        if act.startswith("approach"):
            _, _, target = act.partition(":")
            if not target:
                raise ValueError("approach needs an object id: "
                                 "'approach:<id>'")
            act = "approach"
        if act not in PARK_ACTIONS:
            raise ValueError(f"unknown park action {action!r}; "
                             f"expected one of {PARK_ACTIONS}")
        self.tick += 1
        bump: Optional[str] = None
        if act == "turn_left":
            self.heading = _wrap(self.heading + self.turn)
        elif act == "turn_right":
            self.heading = _wrap(self.heading - self.turn)
        elif act == "walk":
            bump = self._try_move(self.agent_x + self.stride * math.cos(self.heading),
                                  self.agent_y + self.stride * math.sin(self.heading))
        elif act == "approach":
            bump = self._approach(target)
        self._move_movers()
        self.minute += self.minutes_per_tick
        percepts = self.perceive()
        self.last_percepts = percepts
        return self._frame(percepts, bump)

    def _try_move(self, nx: float, ny: float) -> Optional[str]:
        if not (0.0 <= nx <= self.size and 0.0 <= ny <= self.size):
            return "the park fence"
        o = self._obstacle(nx, ny)
        if o is not None:
            return o.describe(True)
        self.agent_x, self.agent_y = nx, ny
        return None

    def _approach(self, oid: str) -> Optional[str]:
        visible = {i for p in self.perceive() if p.modality == "visual"
                   for i in p.object_ids}
        if oid not in visible:
            return None
        o = next(o for o in self.objects if o.id == oid)
        dx, dy = o.x - self.agent_x, o.y - self.agent_y
        d = math.hypot(dx, dy)
        contact = o.radius + self.agent_radius + 0.1
        step = min(self.stride, d - contact)
        if step <= 0.0:
            return None
        self.heading = _wrap(math.atan2(dy, dx))
        return self._try_move(self.agent_x + step * dx / d,
                              self.agent_y + step * dy / d)

    def _frame(self, percepts: List[Percept],
               bump: Optional[str]) -> SensoryFrame:
        seen = [p for p in percepts if p.modality == "visual"]
        ids = {i for p in seen for i in p.object_ids}
        new = ids - self.seen_ids
        self.seen_ids |= ids
        tactile = (f"You bump into {_article(bump)}." if bump
                   else "Grass and gravel underfoot.")
        return SensoryFrame(
            visual=" ".join(p.text for p in seen),
            auditory=" ".join(p.text for p in percepts
                              if p.modality == "auditory"),
            tactile=tactile,
            interoceptive=f"You have been in the park for "
                          f"{self.tick} steps.",
            novelty=len(new) / len(ids) if ids else 0.0,
            arousal=0.3 + 0.4 * (len(new) / len(ids) if ids else 0.0),
            social_presence=min(1.0, sum(
                1 for p in seen for i in p.object_ids
                if i.startswith("person")) * 0.3),
            environment=self.name, tick=self.tick,
            time_of_day=self.time_of_day,
            elapsed_minutes=self.tick * self.minutes_per_tick)
