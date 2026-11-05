"""The embodied loop: a CognitiveRuntime walking a grounded park.

Each step: the park's percepts are sensed (sights and sounds, ambient —
not speech addressed to the mind), the movement just taken learns from
the novelty it produced, the mind thinks one tick, and the BG chooses
the next movement from what is actually in view. ``grounding_report``
scores the walk against the world state.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neuroslm.environments.park import GeometryChannel, Park

_HEADINGS = (0, 45, 90, 135, 180, -135, -90, -45)
"""Relative directions the mind can head off in (degrees, left +)."""


@dataclass
class WalkRecord:
    tick: int
    action: str
    """The movement executed this step."""
    next_action: str
    """The movement the mind chose after thinking."""
    pose: Tuple[float, float, float]
    seen_ids: Tuple[str, ...]
    percepts: Tuple[str, ...]
    thought: Optional[str]


class PlaceMemory:
    """Which ground the mind has viewed, on a grid of ``cell`` metres —
    built from its own pose and field of view (path integration), never
    from the world's objects. ``novelty_ahead`` is the never-viewed share
    of the in-park ground inside the view cone at a pose: hippocampal
    place novelty as the exploration value of a move."""

    def __init__(self, size: float, cell: float = 2.0,
                 reach: float = 25.0, fov_deg: float = 120.0) -> None:
        self.size, self.cell = float(size), float(cell)
        self.reach, self.half_fov = float(reach), math.radians(fov_deg) / 2
        n = max(1, int(math.ceil(self.size / self.cell)))
        c = (np.arange(n) + 0.5) * self.cell
        self._cx, self._cy = np.meshgrid(c, c, indexing="ij")
        self._viewed = np.zeros((n, n), dtype=bool)
        self._obstacle = np.zeros((n, n), dtype=bool)
        self.n = n

    def _cell_of(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not (0.0 <= x <= self.size and 0.0 <= y <= self.size):
            return None
        i = min(self.n - 1, int(x // self.cell))
        j = min(self.n - 1, int(y // self.cell))
        return i, j

    def mark_obstacle(self, x: float, y: float, heading: float,
                      distance: float, bearing: float, extent: float,
                      body_radius: float = 0.3) -> None:
        """Map a seen solid object from its percept (distance, bearing,
        angular half-width) — or a felt one, with extent 0."""
        a = heading + bearing
        ox, oy = x + distance * math.cos(a), y + distance * math.sin(a)
        r = distance * math.sin(extent) + body_radius
        self._obstacle |= np.hypot(self._cx - ox, self._cy - oy) <= r
        c = self._cell_of(ox, oy)
        if c is not None:
            self._obstacle[c] = True

    def value_toward(self, x: float, y: float, angle: float,
                     step: float) -> float:
        """Exploration value of heading ``angle``: 1/(1 + walking
        distance, in cells, from the point one cell ahead to the nearest
        unviewed reachable ground) — frontier exploration on the map:
        routes go round mapped obstacles, sealed-off ground is worth
        nothing, and a step toward the frontier raises its value. (A sum
        of 1/distance over all unviewed ground barely changed per step,
        and the choice dithered.)"""
        look = max(step, 1.5 * self.cell)
        start = None
        for k in range(1, int(math.ceil(look / (self.cell / 2))) + 1):
            t = min(look, k * self.cell / 2)
            c = self._cell_of(x + t * math.cos(angle), y + t * math.sin(angle))
            if c is None or self._obstacle[c]:
                return 0.0
            start = c
        dist = self._walking_distance(start)
        target = (~self._viewed) & (~self._obstacle) & (dist >= 0)
        if not target.any():
            return 0.0
        return 1.0 / (1.0 + float(dist[target].min()))

    def _walking_distance(self, start: Tuple[int, int]) -> np.ndarray:
        """BFS steps over free cells (8-connected); −1 = unreachable."""
        dist = np.full((self.n, self.n), -1, dtype=np.int32)
        dist[start] = 0
        q = deque([start])
        free = ~self._obstacle
        while q:
            i, j = q.popleft()
            d = dist[i, j] + 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (0 <= a < self.n and 0 <= b < self.n
                            and dist[a, b] < 0 and free[a, b]):
                        dist[a, b] = d
                        q.append((a, b))
        return dist

    def _cone(self, x: float, y: float, heading: float,
              reach: Optional[float] = None) -> np.ndarray:
        dx, dy = self._cx - x, self._cy - y
        d = np.hypot(dx, dy)
        b = np.angle(np.exp(1j * (np.arctan2(dy, dx) - heading)))
        r = self.reach if reach is None else reach
        return (d <= r) & ((np.abs(b) <= self.half_fov) | (d < self.cell))

    def view(self, x: float, y: float, heading: float,
             reach: Optional[float] = None,
             occluders: Sequence[Tuple[float, float, float]] = ()) -> None:
        """Mark the ground in view. ``occluders``: (distance, bearing,
        angular half-width) of seen obstructions — the ground behind
        them was not seen."""
        seen = self._cone(x, y, heading, reach)
        if occluders:
            dx, dy = self._cx - x, self._cy - y
            d = np.hypot(dx, dy)
            ang = np.arctan2(dy, dx) - heading
            for od, ob, ext in occluders:
                off = np.abs(np.angle(np.exp(1j * (ang - ob))))
                # from the obstruction's far edge (radius ≈ d·sin(extent));
                # its own footprint must not stay "unseen" forever
                far = od * (1.0 + math.sin(ext)) + self.cell
                seen &= ~((d > far) & (off < ext))
        self._viewed |= seen

    def novelty_ahead(self, x: float, y: float, heading: float,
                      reach: Optional[float] = None) -> float:
        """Never-viewed share of the in-park ground in a direction —
        out to the fence by default, or within ``reach``."""
        cone = self._cone(x, y, heading, reach=(
            self.size * math.sqrt(2) if reach is None else reach))
        n = int(cone.sum())
        return float((cone & ~self._viewed).sum()) / n if n else 0.0

    def unexplored_ahead(self, x: float, y: float, heading: float,
                         reach: Optional[float] = None) -> float:
        """Never-viewed in-park ground in a direction — out to the fence
        by default, or within ``reach`` — each cell discounted by its
        distance (travel cost). Undiscounted, the widening cone behind
        the mind pulled it back after every step (it oscillated)."""
        cone = self._cone(x, y, heading, reach=(
            self.size * math.sqrt(2) if reach is None else reach))
        d = np.maximum(np.hypot(self._cx - x, self._cy - y), self.cell)
        return float(((cone & ~self._viewed) / d).sum())

    def coverage(self) -> float:
        return float(self._viewed.mean())


def movement_options(park: Park) -> List[Tuple[str, Optional[str]]]:
    """Look, head off in one of ``_HEADINGS``, or approach an object
    currently in view."""
    opts: List[Tuple[str, Optional[str]]] = [("look", None)]
    opts += [(f"head:{d}", None) for d in _HEADINGS]
    for p in park.last_percepts:
        if p.modality == "visual":
            for oid in p.object_ids:
                opts.append((f"approach:{oid}", p.text))
    return opts


def _stuck_heads(park: Park, blocked: Sequence[float]) -> set:
    """Heading options that point (±22.5°) where the body just bumped."""
    return {f"head:{d}" for d in _HEADINGS
            if any(abs(math.remainder(park.heading + math.radians(d) - b,
                                      2 * math.pi)) < math.pi / 8
                   for b in blocked)}


def _place_values(places: PlaceMemory, park: Park,
                  blocked: Sequence[float] = ()) -> Dict[str, float]:
    """Exploration value of each move: the unexplored ground it heads
    toward (a count, not a share — a fence-facing cone has few cells),
    relative to the best move. ``blocked``: absolute headings that just
    bumped; worth nothing until the body has moved."""
    x, y, th = park.agent_x, park.agent_y, park.heading
    # this view was just recorded: whatever is still unviewed inside it
    # is hidden, so looking again reveals nothing (it froze the mind
    # behind a tree); heading off goes toward everything that lies that way
    vals = {"look": 0.0}
    stuck = _stuck_heads(park, blocked)
    for d in _HEADINGS:
        a = f"head:{d}"
        vals[a] = 0.0 if a in stuck else places.value_toward(
            x, y, th + math.radians(d), park.stride)
    # relative to the best option: the best remaining direction stays
    # salient however little ground is left. (Normalising by the pool
    # MEAN flattened all directions to ~0.5 and the walk diffused.)
    top = max(vals.values())
    return {a: v / top if top > 0.0 else 0.0 for a, v in vals.items()}


def walk(rt, park: Park, ticks: int, action: str = "look",
         places: Optional[PlaceMemory] = None,
         on_tick=None) -> List[WalkRecord]:
    """Run the loop for ``ticks`` steps. ``places`` carries the place
    memory across calls; ``on_tick(record, tick_result)`` observes."""
    if places is None:
        ch = park.channel
        places = PlaceMemory(park.size, reach=getattr(
            ch, "view_distance", 25.0), fov_deg=math.degrees(2 * getattr(
                ch, "half_fov", math.radians(60.0))))
    recs: List[WalkRecord] = []
    blocked: List[float] = []
    for _ in range(int(ticks)):
        here = (park.agent_x, park.agent_y)
        frame = park.step(action)
        if (park.agent_x, park.agent_y) != here:
            blocked.clear()
        if frame.tactile.startswith("You bump"):
            blocked.append(park.heading)        # touch: not that way
            places.mark_obstacle(park.agent_x, park.agent_y, park.heading,
                                 park.stride, 0.0, 0.0)
        for p in park.last_percepts:
            if (p.solid and p.distance is not None and p.bearing is not None
                    and p.extent is not None):
                places.mark_obstacle(park.agent_x, park.agent_y,
                                     park.heading, p.distance, p.bearing,
                                     p.extent)
        places.view(park.agent_x, park.agent_y, park.heading,
                    reach=places.reach * park.light,
                    occluders=[(p.distance, p.bearing, p.extent)
                               for p in park.last_percepts
                               if p.occludes and p.distance is not None
                               and p.bearing is not None
                               and p.extent is not None])
        for p in park.last_percepts:
            rt.sense(p.text, "sight" if p.modality == "visual" else "sound")
        result = rt.tick()
        # touch rules a way out entirely, not just to value 0 (which tied
        # with every explored direction and kept getting picked)
        stuck = _stuck_heads(park, blocked)
        nxt, _ = rt.choose_movement(
            [o for o in movement_options(park) if o[0] not in stuck],
            expected=_place_values(places, park, blocked))
        rt.note_movement(nxt)
        recs.append(WalkRecord(
            tick=park.tick, action=action, next_action=nxt,
            pose=(park.agent_x, park.agent_y, park.heading),
            seen_ids=tuple(i for p in park.last_percepts
                           if p.modality == "visual" for i in p.object_ids),
            percepts=tuple(p.text for p in park.last_percepts),
            thought=result.thought))
        if on_tick is not None:
            on_tick(recs[-1], result)
        action = nxt
    return recs


def _mentions(text: str, kinds: Sequence[str]) -> List[str]:
    low = (text or "").lower()
    return [k for k in kinds if re.search(rf"\b{re.escape(k)}", low)]


def grounding_report(recs: Sequence[WalkRecord], park: Park) -> Dict:
    """Coverage = share of the park's objects ever seen; static coverage
    the same over objects that stay put (movers are met by chance). A thought's
    mention of a park kind (bench, fountain, …) is grounded when that
    kind appeared in a percept at or before that tick; otherwise it is
    confabulated."""
    kinds = sorted({o.kind for o in park.objects})
    seen = {i for r in recs for i in r.seen_ids}
    perceived: set = set()
    mentions = grounded = 0
    confab: List[str] = []
    for r in recs:
        for t in r.percepts:
            perceived.update(_mentions(t, kinds))
        for k in _mentions(r.thought or "", kinds):
            mentions += 1
            if k in perceived:
                grounded += 1
            else:
                confab.append(k)
    static = [o.id for o in park.objects if o.speed <= 0.0]
    return {"coverage": len(seen) / len(park.objects) if park.objects else 0.0,
            "static_coverage": (sum(i in seen for i in static) / len(static)
                                if static else 0.0),
            "objects_seen": len(seen),
            "mentions": mentions,
            "grounded_mentions": grounded,
            "grounded_rate": grounded / mentions if mentions else None,
            "confabulated": confab}
