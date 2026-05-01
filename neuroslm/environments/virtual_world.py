"""Virtual world simulation: generates embodied sensory experience.

Provides a continuous stream of sensory frames to ground the brain's
cognition in simulated embodied experience. The brain processes these
frames through its sensory cortex just like real sensory input.

Environments:
  - BusStop: waiting for a bus that never comes
  - MeadowTree: lying in grass under a tree in nature
  - RainyWindow: sitting by a window watching rain
  - OceanCliff: standing on a cliff overlooking the ocean
  - Library: sitting in a quiet library surrounded by books
  - Campfire: sitting by a campfire at night

Each environment has:
  - A scene graph of objects with properties
  - Temporal dynamics (things change over time)
  - Sensory channels (visual, auditory, tactile, olfactory, interoceptive)
  - Emotional affordances (the feeling the scene evokes)
  - Random micro-events (bird flies by, leaf falls, etc.)
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class SensoryFrame:
    """One tick of sensory experience from the virtual world."""
    # Rich text descriptions of each channel
    visual: str = ""
    auditory: str = ""
    tactile: str = ""
    olfactory: str = ""
    interoceptive: str = ""

    # Numeric signals
    valence: float = 0.0          # emotional tone [-1, 1]
    arousal: float = 0.0          # activation level [0, 1]
    novelty: float = 0.0          # how surprising this frame is [0, 1]
    comfort: float = 0.5          # physical comfort [0, 1]
    time_pressure: float = 0.0    # urgency [0, 1]
    social_presence: float = 0.0  # other agents nearby [0, 1]

    # Metadata
    environment: str = ""
    tick: int = 0
    time_of_day: str = "afternoon"
    elapsed_minutes: float = 0.0

    def to_text(self) -> str:
        """Flatten to a text string for the sensory cortex."""
        parts = []
        if self.visual:
            parts.append(f"[see] {self.visual}")
        if self.auditory:
            parts.append(f"[hear] {self.auditory}")
        if self.tactile:
            parts.append(f"[feel] {self.tactile}")
        if self.olfactory:
            parts.append(f"[smell] {self.olfactory}")
        if self.interoceptive:
            parts.append(f"[body] {self.interoceptive}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "text": self.to_text(),
            "valence": self.valence,
            "arousal": self.arousal,
            "novelty": self.novelty,
            "comfort": self.comfort,
            "time_pressure": self.time_pressure,
            "social_presence": self.social_presence,
            "environment": self.environment,
            "tick": self.tick,
        }


class Environment:
    """Base class for virtual environments."""

    name: str = "abstract"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tick = 0

    def step(self) -> SensoryFrame:
        raise NotImplementedError

    def stream(self, max_ticks: int = 10000) -> Iterator[SensoryFrame]:
        for _ in range(max_ticks):
            yield self.step()

    def _maybe(self, prob: float, text: str) -> str:
        return text if self.rng.random() < prob else ""


class BusStop(Environment):
    """Sitting at a bus stop, waiting for a bus that never comes.

    Captures: patience, anticipation, boredom, people-watching,
    urban ambient sounds, passage of time, mild frustration.
    """
    name = "bus_stop"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.waited_minutes = 0.0
        self.people_at_stop = self.rng.randint(0, 3)
        self.weather = self.rng.choice(["overcast", "sunny", "drizzling", "windy"])
        self.time = self.rng.choice(["morning", "afternoon", "evening"])

    def step(self) -> SensoryFrame:
        self.tick += 1
        self.waited_minutes += self.rng.uniform(0.5, 2.0)
        t = self.tick

        # Visual scene
        road = self.rng.choice([
            "An empty road stretches ahead, no bus in sight.",
            "Cars pass by intermittently. None of them is the bus.",
            "A taxi slows down near the stop, then accelerates away.",
            "The road is quiet. A pigeon pecks at crumbs on the sidewalk.",
            "You glance down the road again. Still no bus.",
        ])
        people_desc = ""
        if self.people_at_stop > 0:
            actions = [
                "A woman checks her phone impatiently.",
                "An old man reads a folded newspaper.",
                "A teenager listens to music through earbuds, nodding slightly.",
                "A mother holds her child's hand, looking down the road.",
                "Someone sighs and shifts their weight.",
            ]
            people_desc = " " + self.rng.choice(actions)

        # Random micro-events
        micro = self._maybe(0.15, self.rng.choice([
            " A leaf skitters across the pavement.",
            " A dog trots by on the opposite sidewalk.",
            " Someone coughs nearby.",
            " A cyclist whizzes past.",
            " A bird lands on the bus stop shelter briefly.",
            " An ambulance siren wails in the distance.",
        ]))

        visual = road + people_desc + micro

        # Auditory
        ambient = self.rng.choice([
            "Traffic hums in the background.",
            "A distant horn honks.",
            "The wind rustles a discarded plastic bag.",
            "Footsteps approach and then pass by.",
            "Muffled music from a nearby shop.",
        ])
        auditory = ambient

        # Tactile
        tactile_opts = {
            "overcast": "Cool air on your skin. The bench is hard under you.",
            "sunny": "Warm sun on your face. The metal bench is hot.",
            "drizzling": "Fine mist settles on your hands. The bench is damp.",
            "windy": "Gusts tug at your clothes. Hair blows across your face.",
        }
        tactile = tactile_opts[self.weather]

        # Olfactory
        smell = self.rng.choice([
            "Exhaust fumes hang in the air.",
            "A faint smell of coffee from a nearby cafe.",
            "Wet asphalt after recent rain.",
            "Nothing particular — just city air.",
        ])

        # Interoceptive
        boredom = min(1.0, self.waited_minutes / 30.0)
        body_states = [
            f"You've been waiting for {int(self.waited_minutes)} minutes.",
        ]
        if boredom > 0.5:
            body_states.append("A restless feeling grows.")
        if boredom > 0.8:
            body_states.append("Your patience is wearing thin.")
        if self.waited_minutes > 10:
            body_states.append(self._maybe(0.3, "Your legs feel stiff from sitting."))

        # Occasionally someone arrives or leaves
        if self.rng.random() < 0.1:
            self.people_at_stop = max(0, self.people_at_stop + self.rng.choice([-1, 1]))

        return SensoryFrame(
            visual=visual,
            auditory=auditory,
            tactile=tactile,
            olfactory=smell,
            interoceptive=" ".join(s for s in body_states if s),
            valence=-0.1 - 0.3 * boredom,  # increasingly negative
            arousal=0.2 + 0.1 * (1 - boredom),
            novelty=0.1 + self.rng.random() * 0.15,
            comfort=0.5 - 0.2 * boredom,
            time_pressure=0.3 + 0.4 * boredom,
            social_presence=min(1.0, self.people_at_stop * 0.3),
            environment=self.name,
            tick=self.tick,
            time_of_day=self.time,
            elapsed_minutes=self.waited_minutes,
        )


class MeadowTree(Environment):
    """Lying in grass under a tree, dwelling in nature's beauty.

    Captures: peace, awe, sensory richness, mind wandering,
    connectedness with nature, gentle temporal flow.
    """
    name = "meadow_tree"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.tree_type = self.rng.choice(["oak", "maple", "willow", "birch", "cherry"])
        self.season = self.rng.choice(["spring", "summer", "early_autumn"])
        self.time = "afternoon"

    def step(self) -> SensoryFrame:
        self.tick += 1
        elapsed = self.tick * 1.5  # ~1.5 min per tick

        # Visual
        canopy = {
            "spring": f"Fresh green leaves of the {self.tree_type} filter golden sunlight into dancing patterns on the grass.",
            "summer": f"The thick canopy of the {self.tree_type} creates a cool dappled shade. Sunbeams pierce through gaps.",
            "early_autumn": f"The {self.tree_type}'s leaves are turning amber and gold. A few drift down in slow spirals.",
        }[self.season]

        sky_bits = self.rng.choice([
            "Patches of blue sky are visible through the branches.",
            "White clouds drift slowly overhead, visible between leaves.",
            "A hawk circles high above in the clear sky.",
            "Thin cirrus clouds stretch like brushstrokes across the blue.",
        ])

        micro = self._maybe(0.2, self.rng.choice([
            " A butterfly lands on a wildflower nearby.",
            " A squirrel scurries up the trunk above you.",
            " A ladybug crawls across a blade of grass near your hand.",
            " A dragonfly hovers briefly, then darts away.",
            " A spider web between two low branches catches the light.",
            " A caterpillar inches along a leaf overhead.",
        ]))

        visual = canopy + " " + sky_bits + micro

        # Auditory
        sounds = self.rng.choice([
            "Birdsong fills the air — a thrush, perhaps, and the distant call of a cuckoo.",
            "The gentle rustle of leaves in the breeze, like soft applause.",
            "A stream babbles somewhere nearby, its sound mixing with insect hum.",
            "The drone of bees visiting wildflowers in the meadow.",
            "Wind sighs through the branches above, a slow exhalation.",
            "A woodpecker taps rhythmically in a distant tree.",
        ])

        # Tactile
        tactile = self.rng.choice([
            "Cool grass presses against your back. A gentle breeze caresses your skin.",
            "The earth is warm beneath you. Grass tickles the backs of your arms.",
            "Sun-warmed air on your face. The ground is soft and yielding.",
            "A light breeze carries the warmth of the sun. The grass is slightly dewy.",
        ])

        # Olfactory
        smell = self.rng.choice([
            "Fresh-cut grass and wildflowers — clover and chamomile.",
            "The earthy scent of warm soil and tree bark.",
            "Sweet honeysuckle drifts from somewhere in the hedgerow.",
            "Clean air carrying the faint fragrance of pine from the hillside.",
            f"The distinctive smell of {self.tree_type} leaves warming in the sun.",
        ])

        # Interoceptive
        peace = min(1.0, self.tick * 0.1)
        body = self.rng.choice([
            "Your body feels heavy and relaxed against the earth.",
            "Each breath is slow and deep. Your heartbeat is calm.",
            "A profound stillness settles in your chest.",
            "Your muscles have released their tension. You feel held by the ground.",
            "Time seems to slow. You are deeply present.",
        ])

        return SensoryFrame(
            visual=visual,
            auditory=sounds,
            tactile=tactile,
            olfactory=smell,
            interoceptive=body,
            valence=0.5 + 0.3 * peace,
            arousal=0.15 + self.rng.random() * 0.1,
            novelty=0.05 + self.rng.random() * 0.15,
            comfort=0.8 + 0.15 * peace,
            time_pressure=0.0,
            social_presence=0.0,
            environment=self.name,
            tick=self.tick,
            time_of_day=self.time,
            elapsed_minutes=elapsed,
        )


class RainyWindow(Environment):
    """Sitting by a window watching rain. Contemplative, cozy, introspective."""
    name = "rainy_window"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.intensity = self.rng.choice(["light", "steady", "heavy"])
        self.time = self.rng.choice(["morning", "afternoon", "evening"])
        self.has_tea = self.rng.random() > 0.3

    def step(self) -> SensoryFrame:
        self.tick += 1
        elapsed = self.tick * 2.0

        rain_visual = {
            "light": "Fine rain streaks the window glass. The world outside is soft and grey.",
            "steady": "Rain streams down the windowpane in rivulets. Puddles form on the street.",
            "heavy": "Sheets of rain blur the view. The street below is a river of silver.",
        }[self.intensity]

        scene = self.rng.choice([
            " A person hurries past with an umbrella.",
            " Car headlights reflect in the wet road.",
            " The neighbor's cat sits under an awning, watching the rain.",
            " A delivery van splashes through a puddle.",
            " The trees across the street sway in the wind.",
            " Raindrops bounce off a parked car's roof.",
        ])

        visual = rain_visual + self._maybe(0.25, scene)

        auditory = self.rng.choice([
            "The steady patter of rain on the window. The hiss of tires on wet road.",
            "Rain drums softly on the roof. A gutter gurgles.",
            "Thunder rumbles distantly. The rain intensifies momentarily.",
            "The rhythmic tapping of rain. Somewhere, a wind chime sings.",
        ])

        tactile = "The room is warm."
        if self.has_tea:
            tactile += self.rng.choice([
                " A warm mug rests between your palms.",
                " You take a sip of tea. It's the perfect temperature.",
                " Steam rises from your cup, carrying warmth to your face.",
            ])

        smell = self.rng.choice([
            "Petrichor — the smell of rain on dry earth — drifts through a cracked window.",
            "The aroma of tea mingles with the damp, clean air.",
            "Warm wood and the faint scent of rain.",
        ])

        body = self.rng.choice([
            "A deep contentment in being sheltered while the world is washed clean.",
            "Your breathing matches the rhythm of the rain.",
            "A gentle melancholy, not unpleasant, settles over you.",
            "You feel safe and warm. Nowhere to be. Nothing to do.",
        ])

        return SensoryFrame(
            visual=visual, auditory=auditory, tactile=tactile,
            olfactory=smell, interoceptive=body,
            valence=0.3, arousal=0.15, novelty=0.05 + self.rng.random() * 0.1,
            comfort=0.85, time_pressure=0.0, social_presence=0.05,
            environment=self.name, tick=self.tick,
            time_of_day=self.time, elapsed_minutes=elapsed,
        )


class OceanCliff(Environment):
    """Standing on a cliff overlooking the ocean. Awe, vastness, sublimity."""
    name = "ocean_cliff"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.time = self.rng.choice(["sunrise", "midday", "sunset"])

    def step(self) -> SensoryFrame:
        self.tick += 1
        elapsed = self.tick * 1.0

        time_desc = {
            "sunrise": "The sun rises from the horizon, painting the ocean in golds and pinks.",
            "midday": "The sun blazes overhead. The ocean is a vast sheet of glittering blue.",
            "sunset": "The sun sinks toward the water, turning the sky into layers of amber and violet.",
        }[self.time]

        wave = self.rng.choice([
            " Waves crash against the rocks far below, sending up white spray.",
            " The ocean swells and falls rhythmically, an endless breathing.",
            " Whitecaps dot the surface as far as you can see.",
            " A wave breaks against the cliff face with a deep boom.",
        ])

        micro = self._maybe(0.2, self.rng.choice([
            " A seagull glides on the updraft, effortless.",
            " A distant ship moves slowly across the horizon line.",
            " Dolphins arc through the water below.",
            " Seabirds wheel and dive into the waves.",
        ]))

        visual = time_desc + wave + micro

        auditory = self.rng.choice([
            "The roar and hiss of surf. Wind rushes past your ears.",
            "Seagulls cry. The ocean's voice is constant and ancient.",
            "Wind whips across the cliff. Far below, waves thunder.",
            "The deep, resonant boom of waves in a sea cave below.",
        ])

        tactile = self.rng.choice([
            "Strong wind pushes against your chest. Salt spray mists your face.",
            "The rock under your feet is solid and sun-warm. Wind tugs at your hair.",
            "Cool ocean air fills your lungs. The ground trembles faintly with each wave.",
        ])

        smell = "Salt air and seaweed. The clean, mineral scent of open ocean."

        body = self.rng.choice([
            "A sense of vastness expands in your chest. You feel small and alive.",
            "Vertigo tickles the edges of awareness. The height is exhilarating.",
            "Awe fills you — the ocean has been doing this for millions of years.",
            "Your heart beats stronger. The wind and water and light feel infinite.",
        ])

        return SensoryFrame(
            visual=visual, auditory=auditory, tactile=tactile,
            olfactory=smell, interoceptive=body,
            valence=0.7, arousal=0.6, novelty=0.15 + self.rng.random() * 0.15,
            comfort=0.6, time_pressure=0.0, social_presence=0.0,
            environment=self.name, tick=self.tick,
            time_of_day=self.time, elapsed_minutes=elapsed,
        )


class Library(Environment):
    """Sitting in a quiet library. Focus, knowledge, sanctuary."""
    name = "library"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.time = "afternoon"

    def step(self) -> SensoryFrame:
        self.tick += 1
        elapsed = self.tick * 3.0

        visual = self.rng.choice([
            "Tall shelves of books line the walls, their spines a mosaic of colors and gold lettering.",
            "Dust motes drift in a shaft of light from a tall arched window.",
            "An open book lies before you. The pages are cream-colored, the text dense and inviting.",
            "The long reading room stretches ahead, polished wood tables gleaming softly.",
        ])

        micro = self._maybe(0.2, self.rng.choice([
            " Someone turns a page with a soft rustle.",
            " A librarian wheels a cart of books past, nodding.",
            " A student taps a pen thoughtfully against their chin.",
            " Shadows of branches play across the floor from the window.",
        ]))

        auditory = self.rng.choice([
            "Silence, except for the whisper of turning pages.",
            "The faint tick of a clock. A chair creaks distantly.",
            "Almost perfect quiet. Your own breathing is the loudest sound.",
            "The soft scratch of a pen on paper somewhere nearby.",
        ])

        tactile = "The chair is firm and well-worn. Cool air circulates gently."
        smell = "Old paper, wood polish, and the faint mustiness of aged books."

        body = self.rng.choice([
            "Your mind feels sharp and clear. Focus comes easily here.",
            "A deep sense of being surrounded by accumulated human thought.",
            "Calm concentration. The world outside has faded away.",
            "Knowledge feels almost tangible in this room — it saturates the air.",
        ])

        return SensoryFrame(
            visual=visual + micro, auditory=auditory, tactile=tactile,
            olfactory=smell, interoceptive=body,
            valence=0.4, arousal=0.25, novelty=0.05 + self.rng.random() * 0.1,
            comfort=0.75, time_pressure=0.1, social_presence=0.15,
            environment=self.name, tick=self.tick,
            time_of_day=self.time, elapsed_minutes=elapsed,
        )


class Campfire(Environment):
    """Sitting by a campfire at night. Warmth, darkness, primal comfort."""
    name = "campfire"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.time = "night"
        self.wood_remaining = 1.0

    def step(self) -> SensoryFrame:
        self.tick += 1
        elapsed = self.tick * 2.0
        self.wood_remaining = max(0.1, self.wood_remaining - 0.005)

        fire_state = "roaring" if self.wood_remaining > 0.7 else "steady" if self.wood_remaining > 0.3 else "glowing embers"

        visual = self.rng.choice([
            f"The {fire_state} fire sends sparks spiraling upward into the dark sky.",
            f"Flames dance and flicker, casting shifting shadows on the nearby trees.",
            f"Orange light from the {fire_state} fire illuminates a circle of warmth in the darkness.",
            "You stare into the flames. Shapes form and dissolve — faces, landscapes, memories.",
        ])

        sky = self._maybe(0.3, self.rng.choice([
            " Above, the Milky Way stretches across the sky in a river of faint light.",
            " Stars are impossibly bright out here, away from city lights.",
            " A shooting star traces a brief line across the darkness.",
            " The moon hangs low, orange near the horizon.",
        ]))

        auditory = self.rng.choice([
            "The fire crackles and pops. Somewhere, an owl calls.",
            "Wood shifts in the fire with a soft thump. Crickets chirp.",
            "The snap of burning sap. Wind sighs through pine branches.",
            "Embers hiss. The forest is alive with small, distant sounds.",
        ])

        warmth = 0.5 + 0.4 * self.wood_remaining
        tactile = self.rng.choice([
            f"Warm firelight on your face and hands. Cool night air on your back.",
            f"The heat radiates in waves. You shift to warm a different side.",
            f"Smoke drifts your way briefly, making your eyes water.",
        ])

        smell = self.rng.choice([
            "Woodsmoke — the most ancient of human scents. Pine resin and cedar.",
            "The sweet, acrid smell of burning wood. Cool forest air beyond the firelight.",
            "Smoke and earth and the clean smell of the night forest.",
        ])

        body = self.rng.choice([
            "Something primal and ancient stirs — fire, night, the circle of light against darkness.",
            "Deep relaxation. The fire holds your gaze. Thoughts drift like smoke.",
            "You feel connected to every human who has ever sat by a fire and looked up at stars.",
            "Warmth and safety in a small circle of light. The darkness is vast but not threatening.",
        ])

        return SensoryFrame(
            visual=visual + sky, auditory=auditory, tactile=tactile,
            olfactory=smell, interoceptive=body,
            valence=0.6, arousal=0.2, novelty=0.05 + self.rng.random() * 0.1,
            comfort=warmth, time_pressure=0.0, social_presence=0.0,
            environment=self.name, tick=self.tick,
            time_of_day=self.time, elapsed_minutes=elapsed,
        )


# Registry of all environments
ENVIRONMENTS = {
    "bus_stop": BusStop,
    "meadow_tree": MeadowTree,
    "rainy_window": RainyWindow,
    "ocean_cliff": OceanCliff,
    "library": Library,
    "campfire": Campfire,
}


def create_environment(name: str = "random", seed: int = 42) -> Environment:
    """Create a virtual environment by name, or random."""
    if name == "random":
        rng = random.Random(seed)
        name = rng.choice(list(ENVIRONMENTS.keys()))
    cls = ENVIRONMENTS.get(name)
    if cls is None:
        raise ValueError(f"Unknown environment: {name}. "
                         f"Available: {list(ENVIRONMENTS.keys())}")
    return cls(seed=seed)


def environment_stream(seed: int = 42,
                       switch_every: int = 50) -> Iterator[SensoryFrame]:
    """Infinite stream cycling through environments.

    Every `switch_every` ticks, transitions to a new environment,
    simulating the passage through different life contexts.
    """
    rng = random.Random(seed)
    names = list(ENVIRONMENTS.keys())
    while True:
        name = rng.choice(names)
        env = create_environment(name, seed=rng.randint(0, 99999))
        for frame in env.stream(max_ticks=switch_every):
            yield frame
