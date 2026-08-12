# -*- coding: utf-8 -*-
"""CognitiveRuntime — the always-on cognition loop (architecture.md §14).

The two-layer doctrine: the LM trunk is the language cortex (trained on
LM data, isolated); the neuroanatomical machinery is the mind that USES
it at inference. One :meth:`CognitiveRuntime.tick` is one cognitive
cycle:

    SENSE   drain the sensory queue (user text, environment events);
            percepts are always written to episodic memory (salient)
    RECALL  cosine retrieval from ``EpisodicMemory`` (hippocampus) —
            related past episodes enter the thinking context
    THINK   the trunk generates K candidate thoughts from
            persona (PFC goal state) + recall + recent context
    GATE    basal-ganglia selection over candidates:
              * DA raises the selection temperature —
                dopaminergic exploration over greedy exploitation
              * GABA above threshold inhibits the whole act: no
                generation compute, no thought, no write (silence)
    STORE   surprise-gated episodic write: a thought whose trunk-NLL
            sits far from the running EMA is novel → remembered;
            repetitive thoughts converge to the EMA → not written
    DRIVE   ``DrivenNTSystem.step_full`` integrates the tick's
            signals (selected-thought NLL as the loss/surprise
            driver) so NT state carries across ticks

Every collaborator is dependency-injected (the test battery runs the
whole loop with deterministic fakes on CPU); production wiring from a
checkpointed harness lives in :func:`build_runtime_from_harness`.
"""
from __future__ import annotations

import math
import random as _random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from neuroslm.memory.episodic import EpisodicMemory
from neuroslm.cognition.patterns import ActionClassification  # noqa: F401 (re-export)

GenerateFn = Callable[[str, int], str]
EmbedFn = Callable[[str], Sequence[float]]


@dataclass
class ThoughtScore:
    """Trunk-eye view of one candidate thought.

    mean_nll      — mean per-token cross-entropy (nats) of the thought
                    under the trunk. Selection utility (lower = more
                    plausible to the trunk) and the surprise signal.
    entropy_norm  — mean per-token softmax entropy normalised by
                    ln(vocab): the SAME formula the training harness
                    publishes as its runtime Φ proxy
                    (harness.compute_loss, "Cheap runtime Phi proxy"),
                    so training and cognition report one quantity.
    """
    mean_nll: float
    entropy_norm: float


ScoreFn = Callable[[str], ThoughtScore]


@dataclass
class MindConfig:
    """All cognition-loop knobs in one place."""

    n_candidates: int = 3
    """Candidate thoughts generated per tick (basal-ganglia menu size)."""

    thought_n_tok: int = 96
    """Token budget per candidate. 2026-08-12: raised from 32 — live
    thoughts on a verbose instruct model consistently ran out of
    budget mid-sentence ("...complement human"), which read as a
    rendering bug but was actually the generation being cut short."""

    recall_k: int = 3
    """Episodes retrieved per tick (hippocampal read width)."""

    persona: str = ("I am BRIAN, an always-on mind. I think in language, "
                    "remember what happens, and act when it is worth "
                    "acting.")
    """PFC-analog goal/identity header prepended to every thinking
    prompt — the persistent context the trunk thinks within."""

    selection_temp_base: float = 0.3
    """Selection softmax temperature at baseline DA (near-greedy)."""

    da_temp_gain: float = 2.0
    """How strongly DA above its baseline raises selection temperature
    (dopaminergic exploration)."""

    gaba_silence_threshold: float = 0.55
    """GABA level at/above which the tick is inhibited outright."""

    novelty_write_threshold: float = 0.3
    """Semantic novelty (1 − max cosine vs stored episodes) required
    to write a thought to episodic memory (hippocampal novelty gate).
    2026-08-12: replaced the NLL-EMA gate — live ticks showed NLL flat
    in a 3.0-3.9 band while CONTENT orbited one topic; the old gate
    wrote nothing for nine straight ticks and recall served a single
    episode forever."""

    novelty_ema_alpha: float = 0.25
    """EMA rate of the boredom trace: boredom ← (1−α)·boredom +
    α·(1−novelty). Falling novelty accumulates into boredom."""

    curiosity_gain: float = 2.0
    """How strongly boredom raises the basal ganglia's exploration
    temperature: T_eff = T_DA · (1 + curiosity_gain·boredom). Wires
    boredom into the SAME temperature path DA already modulates."""

    replay_boredom_threshold: float = 0.6
    """Boredom level at which idle wandering anchors RECALL on a
    randomly sampled stored episode (hippocampal replay — the
    associative-jump mechanism) instead of the last thought."""

    ior_window: int = 3
    """Inhibition of return: episodes recalled within the last N ticks
    are transiently suppressed from retrieval, so one memory can't
    dominate every tick. IOR yields when it would empty the result."""

    second_person_penalty: float = 1.5
    """BG prior (nats, added to a candidate's NLL) against
    second-person address in WANDERING candidates — inner speech has
    no addressee. Never applied on respond ticks."""

    inferred_recall_penalty: float = 0.15
    """Provenance-aware retrieval (control fix 1): cosine-scale
    penalty on kind="inferred" episodes during recall. Observed
    reality outcompetes the mind's own prior prose at equal
    relevance — a thumb on the scale, not a veto (a strongly more
    relevant inferred episode still wins)."""

    boredom_relief: float = 0.3
    """Control fixes 2+3: a replay jump or a reorient CONSUMES
    boredom (multiplied by this factor) — live trace showed replay
    locking on and firing every tick once boredom crossed the
    threshold and stayed."""

    replay_refractory: int = 5
    """Minimum ticks between replay jumps — replay is a pulse, not a
    state."""

    loop_novelty_threshold: float = 0.25
    """A tick whose semantic novelty falls below this counts toward
    the LOOPING streak."""

    reorient_after: int = 3
    """LOOPING condition (control fix 3): after this many consecutive
    low-novelty ticks, the next wandering tick is a forced state
    transition (reorient) instead of another continuation."""

    trivial_percept_classes: Tuple[str, ...] = ("greeting", "farewell")
    """Control fix 4: percept action classes gated by triviality —
    short greetings/farewells are processed but not committed to
    episodic memory."""

    trivial_percept_max_words: int = 4
    """A trivial-class percept at or under this word count is not
    stored (a contentful long greeting still is)."""

    percept_min_words: int = 3
    """A bare 'statement'-class percept under this word count carries
    ~no autobiographical content and is not stored. Salient classes
    (insult, question, …) are stored regardless of length."""

    max_prompt_chars: int = 1200
    """Hard cap on the composed thinking prompt (CPU latency guard)."""

    wander_prompts: Tuple[str, ...] = (
        "Let my mind wander from what I remember toward something new.",
        "I notice a thread connecting my last thought to what came "
        "before it.",
        "Stepping back for a moment to reflect on the pattern in what "
        "I've seen.",
        "Following a tangent that feels worth pursuing.",
        "Something about this still nags at me — sitting with it a "
        "moment.",
        "Drifting associatively, one idea suggesting the next.",
    )
    """DMN framing for idle ticks (no sensory input this cycle): biases
    THINK toward associative, reflective continuation instead of a
    flat completion of nothing. Rotated round-robin, one per idle
    tick, so the wandering doesn't feel scripted. Never used when
    responding to real input — that's task-positive, not DMN."""


@dataclass
class TickResult:
    """Everything one cognitive cycle produced (telemetry included)."""
    thought: Optional[str]
    candidates: List[str] = field(default_factory=list)
    scores: List[ThoughtScore] = field(default_factory=list)
    recalled: List[dict] = field(default_factory=list)
    stored: bool = False
    inhibited: bool = False
    nt_levels: Dict[str, float] = field(default_factory=dict)
    phi_proxy: float = 0.0
    tick_n: int = 0
    """Monotonically increasing cycle count — the sequence number that
    lets a debug log show how thoughts evolve over time."""
    prior_thought: Optional[str] = None
    """What ``_last_thought`` was BEFORE this tick — the lineage a
    thought evolved from."""
    wandering: bool = False
    """True when this tick had no sensory input (pure DMN idle cycle)
    — as opposed to responding to real input (task-positive)."""
    action: str = "think"
    """The basal-ganglia act this tick resolved to — exactly one of:
      'respond' — external input was present; always surfaces.
      'speak'   — idle tick, but the thought passed the hippocampal
                  novelty gate (``stored``) — voiced spontaneously.
      'think'   — idle tick, thought converged/repetitive — stays
                  internal (logged, never posted as a spoken thought).
    Reuses ``wandering``/``stored`` — no separate gate invented."""
    selection_entropy: float = 0.0
    """IIT-flavored proxy (NOT a rigorous Φ): normalised entropy of
    the basal ganglia's softmax(-NLL/T) choice distribution — low
    means a confident, decisive exclusion of the alternatives; high
    means the pick was close to arbitrary among near-equal options."""
    differentiation: float = 0.0
    """IIT-flavored proxy: population stdev of candidate NLLs (nats)
    — how distinguishable THINK's repertoire of options was this
    tick. A richer, more differentiated repertoire scores higher."""
    novelty: float = 0.0
    """Semantic novelty of the selected thought: 1 − max cosine vs
    every stored episode (1.0 into an empty memory). The write gate,
    the NT activation driver, and the boredom trace all consume this
    one signal."""
    boredom: float = 0.0
    """EMA of (1 − novelty) — the curiosity homeostat's state. Rises
    while thinking stays in one basin; drives exploration temperature
    and replay."""
    replay: bool = False
    """True when this idle tick anchored RECALL on a randomly sampled
    stored episode (hippocampal replay) because boredom crossed the
    threshold — the mechanism behind associative jumps."""
    reorient: bool = False
    """True when the LOOPING condition (a low-novelty streak) forced
    a state transition: recall anchored on the last OBSERVED input
    (return-to-user), or suppressed entirely (clean slate) — instead
    of another continuation of the loop."""
    sensory_modality: Optional[str] = None
    """§15: which modality (if any) anchored THIS tick's RECALL —
    'visual' / 'acoustic' / 'proprioceptive' when a sensory percept
    (``observe_sensory``) drove the tick, ``None`` for a text-
    triggered respond, a replay/reorient jump, or a plain DMN idle
    tick. The debug-observability half of "how does sensory input
    influence thought generation" — see also ``prompt``."""
    prompt: Optional[str] = None
    """The actual composed prompt THINK generated from this tick —
    persona + RECALL's episodes + the SENSE trigger (a sensory
    modality marker, the real user text, or the wander prompt). The
    literal causal link between what was sensed/recalled and what the
    trunk produced; rendered in full by ``format_debug_trace``."""


_NT_ORDER = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")


def format_introspection(result: TickResult) -> str:
    """One-line inner-state summary of a tick — the basal-ganglia
    action, the hippocampal recall/write decision, the NT snapshot,
    and Φ. Same bracket style (``NT[...]``) the training harness's own
    log line uses, so the vocabulary is consistent across layers.

    This is the answer to "let me see the inner processes when
    talking to it": every tick, spoken or silent, gets one of these.
    """
    nt = result.nt_levels or {}
    nt_str = "NT[" + " ".join(
        f"{k}={float(nt.get(k, 0.0)):.2f}" for k in _NT_ORDER) + "]"

    if result.inhibited:
        return (f"{nt_str} BG[inhibited pending={result.action} — silence]")

    n_cand = len(result.candidates)
    pick = None
    if result.thought is not None and result.thought in result.candidates:
        pick = result.candidates.index(result.thought)
    bg = (f"BG[action={result.action} n={n_cand}"
         + (f" pick={pick}" if pick is not None else "")
         + f" H={result.selection_entropy:.2f}]")
    hc = (f"HC[recall={len(result.recalled)} "
         f"write={'yes' if result.stored else 'no'}"
         + (" replay" if result.replay else "") + "]")
    cur = f"CUR[nov={result.novelty:.2f} bore={result.boredom:.2f}]"
    sense = f" SENSE[{result.sensory_modality}]" if result.sensory_modality else ""
    return f"Φ={result.phi_proxy:.2f} {nt_str} {bg} {hc} {cur}{sense}"


def _truncate(text: str, n: int = 90) -> str:
    text = str(text).replace("\n", " ⏎ ")
    return text if len(text) <= n else text[: n - 1] + "…"


def format_debug_trace(result: TickResult) -> str:
    """Multi-line basal-ganglia deliberation trace — "debug log the
    actions generated by basal ganglia": the act (respond/speak/think),
    every candidate thought + its trunk NLL + which one won (rendered
    in FULL, never truncated — only the rejected alternatives in the
    menu are shortened), the NT snapshot, IIT-flavored differentiation/
    decisiveness proxies, the hippocampal episodes that shaped THINK,
    and the thought's lineage from the prior tick. Meant for a raw
    debug stream (``log_stream`` / a connected client), not the
    compact wire-protocol summary (:func:`format_introspection`).
    """
    mode = "DMN wandering" if result.wandering else "responding"
    if result.replay:
        mode += " (replay)"
    if result.reorient:
        mode += " (reorient)"
    lines = [f"[tick {result.tick_n}] {mode} action={result.action}"]

    nt = result.nt_levels or {}
    lines.append("  NT[" + " ".join(
        f"{k}={float(nt.get(k, 0.0)):.2f}" for k in _NT_ORDER) + "]")

    # §15: which sensory percept (if any) drove SENSE this tick — the
    # first link in "how does sensory input influence thought
    # generation" (the second is PROMPT below, the third is HC
    # recalled, since a sensory anchor changes what RECALL retrieves).
    if result.sensory_modality:
        lines.append(f"  SENSE: {result.sensory_modality} percept "
                     f"anchored RECALL this tick")

    if result.inhibited:
        lines.append("  BG: inhibited — no deliberation this tick "
                     "(GABA gated the act before THINK ran)")
        return "\n".join(lines)

    # The exact prompt THINK conditioned on — persona + RECALL's
    # episodes + the SENSE trigger, rendered in full (not truncated)
    # so the causal chain from percept to prompt to thought is
    # actually inspectable, not just asserted.
    if result.prompt:
        lines.append("  PROMPT:")
        lines.append(result.prompt)

    if not result.candidates:
        lines.append("  BG: no candidates generated")
        return "\n".join(lines)

    lines.append(f"  BG deliberation (H={result.selection_entropy:.2f} "
                 f"decisiveness, diff={result.differentiation:.2f} nats "
                 f"repertoire spread — IIT-flavored proxies, not a "
                 f"rigorous Φ):")
    for cand, sc in zip(result.candidates, result.scores):
        selected = cand == result.thought
        mark = "SELECTED" if selected else "        "
        # The winning thought renders in full — only rejected
        # alternatives are shortened for a scannable menu.
        text = cand if selected else _truncate(cand)
        lines.append(f"    [{mark}] nll={sc.mean_nll:5.2f} \"{text}\"")

    if result.recalled:
        lines.append(f"  HC recalled {len(result.recalled)}:")
        for e in result.recalled:
            kind = (e.get("context") or {}).get("kind")
            label = f"[{kind}] " if kind else ""
            lines.append(f"    - {label}{_truncate(e.get('content', ''))}")

    if result.prior_thought:
        lines.append(f"  evolved from: \"{_truncate(result.prior_thought)}\"")

    return "\n".join(lines)


def _population_stdev(values: List[float]) -> float:
    """IIT-flavored differentiation proxy: population stdev of the
    candidate repertoire's NLLs. 0 for <2 values (nothing to
    differentiate)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return var ** 0.5


def selection_temperature(da_level: float, da_baseline: float,
                          cfg: MindConfig) -> float:
    """Basal-ganglia exploration knob: T = T₀ · (1 + g·max(0, DA−DA₀)).

    At baseline DA the selection softmax runs at ``selection_temp_base``
    (near-greedy on trunk NLL); DA above baseline raises T monotonically
    — dopamine trades exploitation for exploration.
    """
    excess = max(0.0, float(da_level) - float(da_baseline))
    return cfg.selection_temp_base * (1.0 + cfg.da_temp_gain * excess)


def curious_selection_temperature(da_level: float, da_baseline: float,
                                  boredom: float, cfg: MindConfig) -> float:
    """Boredom-augmented exploration temperature: the DA-modulated
    temperature (:func:`selection_temperature`) further scaled by
    ``1 + curiosity_gain·boredom``. Bored → hotter → more exploratory
    selection; engaged (boredom≈0) → identical to the DA path alone.
    This is the curiosity homeostat's actuator — the SAME knob DA
    already turns, now also driven by declining semantic novelty."""
    T = selection_temperature(da_level, da_baseline, cfg)
    return T * (1.0 + cfg.curiosity_gain * max(0.0, min(1.0, boredom)))


_SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|yourself)\b|^brian\s*[,:]",
                               re.IGNORECASE)


def _is_second_person(text: str) -> bool:
    """Does this candidate address someone? Inner speech shouldn't."""
    return bool(_SECOND_PERSON_RE.search(text or ""))


class CognitiveRuntime:
    """The always-on mind around a trained trunk. See module docstring.

    Parameters
    ----------
    generate_fn : ``(prompt, max_new_tokens) -> str``
        The trunk's generation seam (same contract as ChatDaemon's).
    score_fn : ``text -> ThoughtScore``
        Trunk NLL + normalised entropy of a candidate thought.
    embed_fn : ``text -> vector``
        Semantic embedding for episodic storage/retrieval.
    nt : DrivenNTSystem-compatible, optional
        Needs ``levels() / baselines() / step_full(**drivers)``.
        Defaults to a real :class:`DrivenNTSystem`.
    memory : EpisodicMemory, optional
    cfg : MindConfig, optional
    rng : random.Random, optional — selection sampling (seedable).
    classify_fn : ``text -> ActionClassification``, optional
        Generalized-action classifier (§14.8). Defaults to the
        deterministic regex lexicon (:func:`patterns.classify_action`)
        — cheap, dependency-free, what the test suite uses. Real
        deployments (:func:`build_runtime_from_hf_lm`) inject a
        generation-based classifier built from the mind's OWN
        trunk/expert instead — the lexicon is a fallback, not the
        production path.
    """

    def __init__(self,
                 generate_fn: GenerateFn,
                 score_fn: ScoreFn,
                 embed_fn: EmbedFn,
                 nt: Optional[Any] = None,
                 memory: Optional[EpisodicMemory] = None,
                 cfg: Optional[MindConfig] = None,
                 rng: Optional[_random.Random] = None,
                 classify_fn: Optional[Callable[[str], "ActionClassification"]] = None,
                 generate_wander_fn: Optional[GenerateFn] = None,
                 ) -> None:
        if nt is None:
            from neuroslm.emergent.driven_nt import DrivenNTSystem
            nt = DrivenNTSystem()
        self._gen = generate_fn
        # Inner-speech register (D of the curiosity loop): DMN ticks
        # generate through this seam — raw completion, no chat
        # template, no imaginary addressee. Defaults to the main seam
        # for injected/test setups; build_runtime_from_hf_lm wires a
        # real completion-mode closure.
        self._gen_wander = generate_wander_fn or generate_fn
        self._score = score_fn
        self._embed = embed_fn
        if classify_fn is None:
            from neuroslm.cognition.patterns import classify_action
            classify_fn = classify_action
        self._classify = classify_fn
        self.nt = nt
        self.memory = memory if memory is not None else EpisodicMemory(512)
        self.cfg = cfg or MindConfig()
        self.rng = rng or _random.Random(0)
        self._sensory: Deque[str] = deque(maxlen=32)
        # Lockstep with `_sensory`: (vector, modality) behind a
        # non-text percept (§15 observe_sensory), or None for a plain
        # text percept (anchor derived from embed_fn as before). This
        # is what lets RECALL anchor on the actual sensory latent
        # instead of re-embedding the percept's text marker, and lets
        # a tick record WHICH modality drove it (debug observability).
        self._sensory_vecs: Deque[Optional[Tuple[List[float], str]]] = deque(
            maxlen=32)
        # Debug/telemetry readback for the last observe_sensory() call
        # — the actual novelty score behind its bool return, useful
        # even when a percept was rejected as habituated.
        self.last_sensory_novelty: Optional[float] = None
        self._last_thought: Optional[str] = None
        self._tick_n: int = 0
        self._wander_idx: int = 0
        # Curiosity homeostat state: EMA of (1 − semantic novelty).
        self._boredom: float = 0.0
        # Inhibition of return: contents recalled in the last few
        # ticks, transiently suppressed from retrieval.
        self._recent_recall: Deque[str] = deque(
            maxlen=max(1, self.cfg.ior_window) * max(1, self.cfg.recall_k))
        # Replay pulse state: tick of the last replay jump (refractory
        # window) — far in the past so the first jump is allowed.
        self._last_replay_tick: int = -(10 ** 9)
        # LOOPING streak: consecutive ticks with novelty below
        # cfg.loop_novelty_threshold.
        self._low_nov_streak: int = 0

    def embed_dim(self) -> int:
        """Dimensionality of ``embed_fn``'s output. Sensory cortices
        (§15, :mod:`neuroslm.sensory.cortices`) probe this once to
        size their projection heads so a percept's ``content_vec``
        lands in the SAME cosine-similarity space as text thoughts —
        the one thing that makes cross-modal RECALL (§14.10 fix 1)
        meaningful."""
        return len(self._embed(self.cfg.persona or " "))

    # ── SENSE ────────────────────────────────────────────────────────

    def observe(self, text: str, source: str = "user") -> None:
        """Sensory input. Percepts are always stored (external events
        are salient by default — the novelty gate applies to the
        mind's own thoughts, not to the world) and drive the NT
        activation channel.

        Stored as an OBSERVED episode (``context.kind == "observed"``)
        — reality that actually happened, as opposed to something the
        mind inferred/generated. This distinction is what lets
        ``format_debug_trace`` (and any future retrieval logic) tell
        real input apart from the mind's own prior thoughts instead
        of treating both as equivalent text.
        """
        text = (text or "").strip()
        if not text:
            return
        # Generalized action class alongside the literal text — "You
        # suck" stores BOTH the literal utterance AND action_class=
        # "insult" — classified by self._classify (the mind's own
        # trunk/expert in production; the regex lexicon by default).
        action_class = self._classify(text).primary

        # Utility gate (control fix 4): trivial percepts are PROCESSED
        # (queued below, replied to) but not committed to episodic
        # memory — live evidence: a stray "Hellohello" became a
        # permanent autobiographical anchor that replay resurfaced.
        # Length is not salience: a two-word insult IS stored; a
        # two-word bare statement or short greeting is not.
        n_words = len(text.split())
        trivial = (
            (action_class in self.cfg.trivial_percept_classes
             and n_words <= self.cfg.trivial_percept_max_words)
            or (action_class == "statement"
                and n_words < self.cfg.percept_min_words)
        )
        if not trivial:
            self.memory.add(text,
                            content_vec=list(self._embed(text)),
                            nt_state=self.nt.levels(),
                            tags=[source, "percept", "kind=observed",
                                 f"action_class={action_class}"],
                            context={"kind": "observed", "source": source,
                                    "action_class": action_class})
        self._sensory.append(text)
        self._sensory_vecs.append(None)
        self.nt.step_full(activation=1.0)

    def observe_sensory(self, modality: str, content_vec: Sequence[float],
                        source: str = "sensor") -> bool:
        """Non-text SENSE (§15): a percept that arrives already
        embedded — a sensory cortex's own latent vector (vision,
        acoustic, proprioceptive) — never captioned into text. The
        stored/retrieved representation IS ``content_vec``; the only
        text involved is a fixed, content-INDEPENDENT modality marker
        used solely to cue THINK's prompt ("Just now: [visual
        percept]") — never a description derived from what was
        actually sensed.

        Gated by the SAME semantic-novelty signal that gates the
        mind's own thoughts (§14.9) rather than by word count (there
        are none): a habituated, unchanged stream — the same camera
        frame every tick — is filtered at the door, the orienting-
        response habituation (Sokolov 1963) this loop's boredom trace
        already models. A novel percept both interrupts wandering
        (queues the marker, anchoring the next RECALL on THIS vector
        — see ``tick()``'s SENSE section) and is written to episodic
        memory as an OBSERVED episode, so it out-competes the mind's
        own inferred prose in retrieval exactly like an observed text
        percept does (§14.10 fix 1).

        Returns whether the percept was novel enough to be attended to.
        """
        vec = list(content_vec)
        top = self.memory.retrieve_scored(vec, k=1)
        novelty = 1.0 - top[0][0] if top else 1.0
        novelty = max(0.0, min(1.0, novelty))
        self.last_sensory_novelty = novelty
        if novelty < self.cfg.novelty_write_threshold:
            return False
        label = f"[{modality} percept]"
        self.memory.add(label, content_vec=vec, nt_state=self.nt.levels(),
                        tags=[source, "percept", "kind=observed",
                             f"modality={modality}"],
                        context={"kind": "observed", "source": source,
                                "modality": modality, "novelty": novelty})
        self._sensory.append(label)
        self._sensory_vecs.append((vec, modality))
        self.nt.step_full(activation=novelty)
        return True

    # ── The cognitive cycle ──────────────────────────────────────────

    def tick(self) -> TickResult:
        """One SENSE→RECALL→THINK→GATE→STORE→DRIVE cycle."""
        self._tick_n += 1
        tick_n = self._tick_n
        prior_thought = self._last_thought
        levels = self.nt.levels()

        # GATE (pre-emptive): GABA inhibition suppresses the act
        # itself — an inhibited tick spends no generation compute.
        if levels.get("GABA", 0.0) >= self.cfg.gaba_silence_threshold:
            self.nt.step_full()          # pure leak toward baselines
            # Debug-trace honesty: report WHAT was suppressed, not
            # generic silence — a pending respond is worth knowing
            # about even though inhibition blocked it.
            pending_action = "respond" if self._sensory else "think"
            return TickResult(thought=None, inhibited=True,
                              nt_levels=levels, tick_n=tick_n,
                              prior_thought=prior_thought,
                              action=pending_action)

        # SENSE: drain the queue (newest percept anchors the tick).
        sensory = list(self._sensory)
        sensory_vecs = list(self._sensory_vecs)
        self._sensory.clear()
        self._sensory_vecs.clear()
        # DMN condition: nothing external arrived this cycle — the
        # tick is pure mind-wandering rather than a response.
        wandering = not bool(sensory)

        # REORIENT (control fix 3): the LOOPING condition — novelty
        # below threshold for `reorient_after` consecutive ticks —
        # forces a state transition instead of another continuation:
        # anchor RECALL on the last OBSERVED input (return-to-user),
        # or suppress recall entirely (clean slate) when no observed
        # episode exists to return to. Wandering only — respond is
        # already anchored on real input.
        reorient = (wandering
                    and self._low_nov_streak >= self.cfg.reorient_after)

        # Hippocampal replay (C2 of the curiosity loop): when bored,
        # an idle tick anchors RECALL on a randomly sampled stored
        # episode instead of the last thought — the associative-jump
        # mechanism that breaks the last_thought→similarity→same-basin
        # chain. Never during respond, never during reorient, and a
        # PULSE, not a state (control fix 2): refractory-gated here,
        # boredom-consumed below — live trace showed replay locking on
        # and firing every tick once boredom saturated.
        replay = False
        anchor_vec = None
        suppress_recall = False
        # Which modality (if any) actually anchored THIS tick's RECALL
        # — debug observability into the sensory->thought causal
        # chain (format_debug_trace's SENSE line). None for a
        # text-triggered respond, a replay/reorient jump, or a plain
        # DMN idle tick.
        sensory_modality: Optional[str] = None
        if reorient:
            last_observed = next(
                (e for e in reversed(self.memory.all())
                 if (e.get("context") or {}).get("kind") == "observed"
                 and e.get("content_vec") is not None), None)
            if last_observed is not None:
                anchor_vec = list(last_observed["content_vec"])
            else:
                suppress_recall = True
        elif (wandering
              and self._boredom >= self.cfg.replay_boredom_threshold
              and (tick_n - self._last_replay_tick)
              > self.cfg.replay_refractory):
            candidates_for_replay = [e for e in self.memory.all()
                                     if e.get("content_vec") is not None]
            if candidates_for_replay:
                replay = True
                self._last_replay_tick = tick_n
                anchor_vec = list(
                    self.rng.choice(candidates_for_replay)["content_vec"])
        if anchor_vec is None and not suppress_recall:
            if sensory and sensory_vecs and sensory_vecs[-1] is not None:
                # §15: the newest percept arrived pre-embedded (a
                # sensory cortex's own latent) — anchor on THAT vector
                # directly, never on a re-embedding of its text
                # marker, which carries none of the percept's content.
                anchor_vec, sensory_modality = sensory_vecs[-1]
                anchor_vec = list(anchor_vec)
            else:
                anchor = (sensory[-1] if sensory
                          else self._last_thought or self.cfg.persona)
                anchor_vec = list(self._embed(anchor))

        # RECALL: hippocampal similarity read with provenance-aware
        # scoring (control fix 1: kind="inferred" episodes carry a
        # cosine-scale penalty, so observed reality outcompetes the
        # mind's own prose at equal relevance — a thumb on the scale,
        # not a veto) and inhibition of return (C1: episodes recalled
        # in the last few ticks are transiently suppressed). IOR
        # yields rather than blinds — when suppression would empty
        # the result, the unsuppressed top-k is used anyway.
        if suppress_recall:
            recalled = []
        else:
            # Over-fetch past the raw-cosine top-k: the provenance
            # penalty and IOR re-rank AFTER retrieve_scored's own
            # top-k cut, so a penalized/suppressed episode that would
            # otherwise place just outside a narrow k never gets a
            # chance to be out-ranked correctly. Bounded by the total
            # episode count (pure-python cosine, cheap at memory's
            # capped maxlen).
            over_fetch = max(self.cfg.recall_k + len(self._recent_recall),
                             len(self.memory.all()))
            scored = self.memory.retrieve_scored(anchor_vec, k=over_fetch)
            pen = self.cfg.inferred_recall_penalty
            adjusted = sorted(
                ((sim - (pen if (e.get("context") or {}).get("kind")
                         == "inferred" else 0.0), e)
                 for sim, e in scored),
                key=lambda t: t[0], reverse=True)
            fresh = [e for _, e in adjusted
                     if e.get("content") not in self._recent_recall]
            recalled = fresh[: self.cfg.recall_k] if fresh else [
                e for _, e in adjusted[: self.cfg.recall_k]]
            for e in recalled:
                self._recent_recall.append(e.get("content"))

        # THINK: K candidates from persona + recall + context. The
        # trigger — what this tick is actually ABOUT — is the real
        # user text for a respond, or the specific wander prompt
        # chosen for an idle tick; captured here (not hidden inside
        # _compose_prompt) so the episode we may store below can
        # record it (Layer 2 — Context: "what happened around it").
        # Inner-speech register (D): idle ticks generate through the
        # wander seam (raw completion, no chat template) — an
        # addressee only exists on respond ticks.
        trigger_text = sensory[-1] if sensory else self._next_wander_prompt()
        prompt = self._compose_prompt(sensory, recalled, trigger_text)
        gen = self._gen if not wandering else self._gen_wander
        candidates: List[str] = []
        for _ in range(max(1, self.cfg.n_candidates)):
            out = (gen(prompt, self.cfg.thought_n_tok) or "").strip()
            if out:
                candidates.append(out)
        if not candidates:
            self.nt.step_full()
            return TickResult(thought=None, recalled=recalled,
                              nt_levels=levels, tick_n=tick_n,
                              prior_thought=prior_thought,
                              wandering=wandering,
                              action="respond" if not wandering else "think",
                              sensory_modality=sensory_modality,
                              prompt=prompt)

        # GATE: boredom-and-DA-tempered softmax over −NLL, with the
        # inner-speech prior (D2) penalizing second-person address in
        # WANDERING candidates only.
        scores = [self._score(c) for c in candidates]
        idx, selection_entropy = self._select(scores, levels, candidates,
                                              wandering)
        thought, sc = candidates[idx], scores[idx]
        differentiation = _population_stdev([s.mean_nll for s in scores])

        # NOVELTY (A of the curiosity loop): semantic novelty of the
        # selected thought — 1 − max cosine vs every stored episode.
        # Computed BEFORE storing (a thought can't be compared against
        # itself). One signal, three consumers: the write gate below,
        # the NT activation driver, and the boredom trace.
        thought_vec = list(self._embed(thought))
        top = self.memory.retrieve_scored(thought_vec, k=1)
        novelty = 1.0 - top[0][0] if top else 1.0
        novelty = max(0.0, min(1.0, novelty))

        # ACTION: the basal ganglia's actual act this tick — exactly
        # one of respond/reorient/speak/think. External input always
        # surfaces (respond); a forced state transition reports itself
        # (reorient — surfaces, so the shift is visible); otherwise
        # the SAME novelty gate that decided whether to remember the
        # thought also decides whether to voice it (speak) or keep it
        # internal (think) — one signal, two consequences.
        stored = novelty >= self.cfg.novelty_write_threshold
        if not wandering:
            action = "respond"
        elif reorient:
            action = "reorient"
        else:
            action = "speak" if stored else "think"

        # STORE: novelty-gated episodic write — as a full episode,
        # not bare text. Layer 1 (event) is `thought` itself; the
        # rest lives in `context`, reusing EpisodicMemory's existing
        # (previously-unused) slot rather than inventing a parallel
        # structure. `associations` names which prior episodes fed
        # RECALL — real IIT-flavored state (confidence/phi/selection
        # entropy/differentiation), never a fabricated mood label.
        if stored:
            action_class = self._classify(thought).primary
            self.memory.add(
                thought,
                content_vec=thought_vec,
                nt_state=levels,
                tags=["thought", f"action={action}",
                     "wandering" if wandering else "responding",
                     f"action_class={action_class}"],
                context={
                    "kind": "inferred",
                    "action": action,
                    "action_class": action_class,
                    "wandering": wandering,
                    "trigger": trigger_text,
                    "tick_n": tick_n,
                    "associations": [e.get("content", "") for e in recalled],
                    "confidence": max(0.0, min(1.0, 1.0 - sc.entropy_norm)),
                    "phi_proxy": max(0.0, min(1.0, sc.entropy_norm)),
                    "selection_entropy": selection_entropy,
                    "differentiation": differentiation,
                    "novelty": novelty,
                })

        # DRIVE (B): the tick's signals advance the NT dynamics — the
        # selected thought's NLL as the loss/surprise driver, and
        # SEMANTIC NOVELTY as the activation driver (novel content is
        # arousing; the old entropy proxy barely moved, which is why
        # NT sat pinned at baseline for nine straight live ticks).
        # Boredom integrates 1−novelty and feeds the next tick's
        # selection temperature via curious_selection_temperature.
        self.nt.step_full(loss=sc.mean_nll, activation=novelty)
        a = self.cfg.novelty_ema_alpha
        self._boredom = (1.0 - a) * self._boredom + a * (1.0 - novelty)

        # LOOPING streak (control fix 3) + pulse relief (fix 2/3): a
        # replay jump or a reorient CONSUMES boredom — the correction
        # IS the relief; without this, replay locked on and fired
        # every tick once boredom saturated (live trace, tick 10+).
        if novelty < self.cfg.loop_novelty_threshold:
            self._low_nov_streak += 1
        else:
            self._low_nov_streak = 0
        if replay or reorient:
            self._boredom *= self.cfg.boredom_relief
        if reorient:
            self._low_nov_streak = 0

        self._last_thought = thought
        return TickResult(thought=thought, candidates=candidates,
                          scores=scores, recalled=recalled,
                          stored=stored, inhibited=False,
                          nt_levels=levels,
                          phi_proxy=max(0.0, min(1.0, sc.entropy_norm)),
                          tick_n=tick_n, prior_thought=prior_thought,
                          wandering=wandering, action=action,
                          selection_entropy=selection_entropy,
                          differentiation=differentiation,
                          novelty=novelty, boredom=self._boredom,
                          replay=replay, reorient=reorient,
                          sensory_modality=sensory_modality, prompt=prompt)

    # ── Internals ────────────────────────────────────────────────────

    def _compose_prompt(self, sensory: List[str],
                        recalled: List[dict], trigger_text: str) -> str:
        lines: List[str] = [self.cfg.persona, ""]
        if recalled:
            lines.append("I remember:")
            for e in recalled:
                lines.append(f"- {e['content']}")
            lines.append("")
        if self._last_thought:
            lines.append(f"My last thought: {self._last_thought}")
        if sensory:
            for s in sensory:
                lines.append(f"Just now: {s}")
            # 'respond' action: cue a reply, not a free-floating
            # thought — this is what THINK produces for the basal
            # ganglia to surface as the actual answer to real input.
            lines.append("My reply:")
        else:
            # DMN: nothing external arrived — bias THINK toward
            # associative, reflective continuation instead of a flat
            # completion of nothing. `trigger_text` is the wander
            # prompt already chosen by the caller (tick()), so it can
            # be recorded as this episode's Layer-2 trigger too.
            lines.append(trigger_text)
            lines.append("Thought:")
        return "\n".join(lines)[-self.cfg.max_prompt_chars:]

    def _next_wander_prompt(self) -> str:
        """Round-robin over ``cfg.wander_prompts`` so idle mind-
        wandering doesn't repeat the exact same framing every tick."""
        prompts = self.cfg.wander_prompts or ("",)
        p = prompts[self._wander_idx % len(prompts)]
        self._wander_idx += 1
        return p

    def _select(self, scores: List[ThoughtScore],
                levels: Dict[str, float],
                candidates: Optional[List[str]] = None,
                wandering: bool = False) -> Tuple[int, float]:
        """Basal-ganglia selection. Returns ``(chosen_index,
        selection_entropy)`` — the latter an IIT-flavored decisiveness
        proxy: normalised entropy of the softmax(utility/T) choice
        distribution (0 = one option clearly excluded the rest, 1 =
        arbitrary among near-equal options).

        Utility per candidate = −(NLL + prior), where the prior adds
        ``second_person_penalty`` nats to WANDERING candidates that
        address someone (inner speech has no addressee — D2 of the
        curiosity loop; how action priors enter BG models generally).
        Temperature = :func:`curious_selection_temperature` — the DA
        path scaled up by boredom (B of the loop).
        """
        if len(scores) == 1:
            return 0, 0.0
        # DrivenNTSystem exposes `baselines` as a property while
        # `levels()` is a method — accept both shapes so duck-typed
        # NT systems keep working.
        baselines = self.nt.baselines
        if callable(baselines):
            baselines = baselines()
        T = curious_selection_temperature(
            levels.get("DA", 0.0), baselines.get("DA", 0.15),
            self._boredom, self.cfg)
        penalties = [0.0] * len(scores)
        if wandering and candidates is not None:
            penalties = [self.cfg.second_person_penalty
                        if _is_second_person(c) else 0.0
                        for c in candidates]
        # softmax(−(NLL+prior) / T), computed stably.
        utils = [-(s.mean_nll + p) / max(T, 1e-6)
                for s, p in zip(scores, penalties)]
        m = max(utils)
        exps = [math.exp(u - m) for u in utils]
        z = sum(exps)
        probs = [e / z for e in exps]
        ent = -sum(p * math.log(p) for p in probs if p > 0.0)
        selection_entropy = ent / math.log(len(probs))
        r = self.rng.random() * z
        acc = 0.0
        idx = len(exps) - 1
        for i, e in enumerate(exps):
            acc += e
            if r <= acc:
                idx = i
                break
        return idx, selection_entropy

    # ── Knowledge extraction: cross-episode pattern mining ────────────

    def detect_patterns(self, window: int = 1, min_support: float = 0.0,
                        min_confidence: float = 0.3) -> List["AssociationRule"]:
        """Mine temporal association rules over the whole episode
        history (see ``neuroslm/cognition/patterns.py`` — Apriori-
        derived, statistical association, NOT causation; every rule
        reports whether its evidence is externally grounded or pure
        self-talk). On-demand, not run automatically per tick — the
        buffer needs enough history for the statistics to mean
        anything, and this is an explicit analysis step, not part of
        the SENSE→RECALL→THINK→GATE→STORE→DRIVE cycle itself.
        """
        from neuroslm.cognition.patterns import mine_temporal_associations
        # EpisodicMemory stores action_class/kind nested under
        # `context` (§14.7); mine_temporal_associations wants the flat
        # {"action_class", "kind"} shape it's independently tested
        # against — adapt here rather than coupling the storage shape
        # into the storage-agnostic mining function.
        flat = [{"action_class":
                (e.get("context") or {}).get("action_class") or "statement",
                "kind": (e.get("context") or {}).get("kind", "inferred")}
               for e in self.memory.all()]
        return mine_temporal_associations(
            flat, window=window, min_support=min_support,
            min_confidence=min_confidence)


# ── Production wiring ────────────────────────────────────────────────

def build_runtime_from_harness(harness: Any, tokenizer: Any,
                               device: str = "cpu",
                               cfg: Optional[MindConfig] = None,
                               temperature: float = 0.8,
                               top_k: int = 40) -> CognitiveRuntime:
    """Wire a checkpointed harness into a CognitiveRuntime.

    Reuses the daemon's generate seam for THINK; builds score/embed
    from the trunk itself:
      * score — one forward over the thought's tokens: mean shifted CE
        (nats) + mean softmax entropy / ln(V) (the harness's Φ-proxy
        formula).
      * embed — mean of the trunk's token-embedding rows (the DSL LM's
        ``embed`` parameter). Raises for checkpoints without an
        embedding surface rather than degrading to a non-semantic hash.
    """
    import torch
    import torch.nn.functional as F

    from neuroslm.chat_daemon import _build_generate_fn_from_harness

    lm = getattr(harness, "language_model", None) or harness
    embed_table = getattr(lm, "embed", None)
    if embed_table is None:
        raise ValueError(
            "build_runtime_from_harness: the trunk exposes no `embed` "
            "token-embedding surface — episodic recall needs semantic "
            "vectors (use a DSL LM checkpoint).")

    generate_fn = _build_generate_fn_from_harness(
        harness, tokenizer, device=device,
        temperature=temperature, top_k=top_k)

    @torch.no_grad()
    def score_fn(text: str) -> ThoughtScore:
        ids = tokenizer.encode(text or " ")
        if len(ids) < 2:
            ids = ids + [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = lm(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        V = logits.shape[-1]
        lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = x[0, 1:]
        nll = float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())
        ent = float(-(lp.exp() * lp).sum(-1).mean()) / math.log(V)
        return ThoughtScore(mean_nll=nll, entropy_norm=ent)

    @torch.no_grad()
    def embed_fn(text: str) -> Sequence[float]:
        ids = tokenizer.encode(text or " ") or [0]
        rows = embed_table[torch.tensor(ids, dtype=torch.long,
                                        device=device)]
        return rows.float().mean(dim=0).cpu().tolist()

    # The mind classifies with its OWN trunk — reuses the SAME
    # generate_fn closure THINK uses (one model, two jobs), never the
    # bare regex lexicon by default (§14.8).
    from neuroslm.cognition.patterns import classify_action_via_generation

    def classify_fn(text: str) -> "ActionClassification":
        return classify_action_via_generation(text, generate_fn)

    return CognitiveRuntime(generate_fn=generate_fn, score_fn=score_fn,
                            embed_fn=embed_fn, cfg=cfg,
                            classify_fn=classify_fn)


def build_runtime_from_hf_lm(model_id: str = "smollm2_360m",
                             device: str = "cpu",
                             cfg: Optional[MindConfig] = None,
                             temperature: float = 0.8,
                             top_k: int = 40,
                             model_factory: Optional[Callable[[], Any]] = None,
                             tokenizer_factory: Optional[Callable[[], Any]] = None,
                             ) -> CognitiveRuntime:
    """Wire the mind to a frozen pretrained HF expert — no trunk needed.

    The doctrine's escape hatch for using BRIAN before full trunk
    training: THINK/score/embed all come from one of the LM experts
    (default the `general` roster slot, ``smollm2_360m``) instead of a
    checkpointed trunk. ``model_id`` accepts a roster alias, a full
    ``owner/repo`` id, or an ``hf://`` URL (resolved via
    :func:`neuroslm.experts.resolve_expert_alias` — pure, typo-safe).

    ``model_factory`` / ``tokenizer_factory`` are injection points: the
    test battery passes fakes; ``run_chat_daemon`` passes closures over
    an already-loaded model so the daemon and the mind share one copy.
    """
    import torch
    import torch.nn as nn

    from neuroslm.experts import resolve_expert_alias

    resolved = resolve_expert_alias(model_id)

    if model_factory is not None:
        model = model_factory()
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(resolved)
    model.eval()
    # 2026-08-12 live incident: from_pretrained() loads on CPU by
    # default. Without this, an A100 mind deploy silently ran the
    # whole expert on CPU — each DMN tick took ~55-60s instead of a
    # couple seconds, and the daemon's non-blocking inference lock
    # stayed held that whole time, so /think calls landing mid-tick
    # returned None. Unconditional (not "if device != cpu") so a
    # future default change can't silently reintroduce the bug.
    model = model.to(device)
    if tokenizer_factory is not None:
        tokenizer = tokenizer_factory()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolved)

    mcfg = getattr(model, "config", None)
    max_ctx = int(getattr(mcfg, "n_positions", None)
                  or getattr(mcfg, "max_position_embeddings", 1024) or 1024)

    class _HFLMWrapper(nn.Module):
        """ids → logits; unwraps the HF output object so the daemon's
        generate seam and the scorer see a plain tensor."""

        def __init__(self, m):
            super().__init__()
            self.model = m
            self.max_ctx = max_ctx

        def forward(self, ids):
            out = self.model(input_ids=ids)
            return getattr(out, "logits", out)

    wrapper = _HFLMWrapper(model)
    embed_table = model.get_input_embeddings().weight

    if hasattr(model, "generate"):
        # KV-cache fast path. The daemon's naive seam re-runs a FULL
        # forward over the growing sequence per generated token —
        # written for the small DSL trunk, it turns a 360M HF expert
        # on CPU into minutes-per-reply (user-visible "hang", found
        # live 2026-08-11). HF's generate() with use_cache=True is the
        # correct sampler for HF models.
        eos_id = getattr(tokenizer, "eos_token_id", None)
        # 2026-08-11 live incident: unconstrained sampling on a BASE
        # (non-instruct) model hallucinated the next "USER:" turn and
        # then fell into n-gram repetition ("Well, I do." x5). A base
        # model has no learned turn-end token, so it never stops on
        # its own — an instruct model's chat template supplies one.
        has_chat_template = bool(getattr(tokenizer, "chat_template", None))

        def _make_generate(use_chat_template: bool) -> GenerateFn:
            """One closure factory, two seams (D of the curiosity
            loop): the chat seam (respond — an addressee exists) and
            the raw-completion wander seam (inner speech — no chat
            template, so the instruct model continues the thinker's
            own first-person text instead of replying TO someone,
            which is what produced 'Brian, ...' on every live tick)."""

            @torch.no_grad()
            def _generate(prompt: str, max_new_tokens: int) -> str:
                if use_chat_template:
                    ids = list(tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True, tokenize=True))
                else:
                    ids = list(tokenizer.encode(prompt) or [0])
                ids = ids[-(max_ctx - max(1, int(max_new_tokens))):]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                gen_kwargs = dict(
                    input_ids=x,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(max(temperature, 1e-6)),
                    top_k=int(top_k) if top_k else 0,
                    use_cache=True,
                    pad_token_id=eos_id,
                    # Repetition controls: always on. Cheap,
                    # model-agnostic, and the second half of the fix
                    # for the observed degenerate loop.
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
                if not use_chat_template:
                    # No template turn-end token applies in raw
                    # completion — stop before the model drifts into
                    # hallucinating a conversation.
                    gen_kwargs["stop_strings"] = ["\nUSER:", "\nUser:",
                                                 "\n\n"]
                    gen_kwargs["tokenizer"] = tokenizer
                out = model.generate(**gen_kwargs)
                new_ids = out[0, x.shape[1]:].tolist()
                return tokenizer.decode(new_ids, skip_special_tokens=True)

            return _generate

        generate_fn = _make_generate(has_chat_template)
        generate_wander_fn = _make_generate(False)
    else:
        from neuroslm.chat_daemon import _build_generate_fn_from_harness
        from types import SimpleNamespace
        generate_fn = _build_generate_fn_from_harness(
            SimpleNamespace(language_model=wrapper), tokenizer,
            device=device, temperature=temperature, top_k=top_k)
        generate_wander_fn = generate_fn

    @torch.no_grad()
    def score_fn(text: str) -> ThoughtScore:
        ids = tokenizer.encode(text or " ")
        if len(ids) < 2:
            ids = list(ids) + [0]
        ids = ids[-max_ctx:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = wrapper(x)
        V = logits.shape[-1]
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = x[0, 1:]
        nll = float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())
        ent = float(-(lp.exp() * lp).sum(-1).mean()) / math.log(V)
        return ThoughtScore(mean_nll=nll, entropy_norm=ent)

    @torch.no_grad()
    def embed_fn(text: str) -> Sequence[float]:
        ids = tokenizer.encode(text or " ") or [0]
        rows = embed_table[torch.tensor(ids, dtype=torch.long,
                                        device=device)]
        return rows.float().mean(dim=0).cpu().tolist()

    # The mind classifies with its OWN trunk/expert — reuses the SAME
    # generate_fn closure THINK uses (one model, two jobs), never the
    # bare regex lexicon by default (§14.8).
    from neuroslm.cognition.patterns import classify_action_via_generation

    def classify_fn(text: str) -> "ActionClassification":
        return classify_action_via_generation(text, generate_fn)

    return CognitiveRuntime(generate_fn=generate_fn, score_fn=score_fn,
                            embed_fn=embed_fn, cfg=cfg,
                            classify_fn=classify_fn,
                            generate_wander_fn=generate_wander_fn)
