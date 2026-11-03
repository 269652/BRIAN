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

DEFAULT_MIND_EXPERT = "deepseek_r1_qwen_7b"
"""The mind's default trunk (§14.12). SmolLM2-360M set the thought-
quality floor in run 58670004; a DeepSeek-R1 distill thinks in a
visible <think> trace and is the default for every mind entry point
(chat --expert, deploy-mind, eval mind, deploy-mind-eval). Use
``deepseek_r1_qwen_1_5b`` on CPU."""


def _hf_load_kwargs(device: str) -> dict:
    """GPU: load in the checkpoint's dtype (bf16 for R1) — transformers
    4.x otherwise defaults to fp32. CPU: fp32 (bf16 matmuls are slow)."""
    return {} if str(device) == "cpu" else {"dtype": "auto"}


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

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def split_reasoning(text: str) -> Tuple[str, str]:
    """``(answer, reasoning)`` from a reasoning model's output. Handles
    an explicit ``<think>`` and a template-opened block (R1 chat
    templates put ``<think>`` in the prompt). An unterminated block
    has no answer yet."""
    t = text or ""
    if _THINK_CLOSE in t:
        head, _, tail = t.partition(_THINK_CLOSE)
        return tail.strip(), head.replace(_THINK_OPEN, "").strip()
    if _THINK_OPEN in t:
        return "", t.split(_THINK_OPEN, 1)[1].strip()
    return t.strip(), ""


@dataclass
class MindSeams:
    """The trunk-facing seams a :class:`CognitiveRuntime` is built from,
    separated from any one runtime so a caller (the §14.11 indicator
    battery) can build many fresh runtimes over one loaded model.

    ``reseed`` re-seeds the generator's sampling RNG (torch for HF
    experts) so each probe item is reproducible."""
    generate_fn: GenerateFn
    score_fn: ScoreFn
    embed_fn: EmbedFn
    generate_wander_fn: Optional[GenerateFn] = None
    classify_fn: Optional[Callable[[str], "ActionClassification"]] = None
    reseed: Optional[Callable[[int], None]] = None
    reasoning_fn: Optional[Callable[[], Optional[str]]] = None
    """Think trace of the most recent ``generate_fn`` call (reasoning
    experts only; None otherwise)."""

    def runtime(self, cfg: Optional["MindConfig"] = None,
                nt: Optional[Any] = None,
                memory: Optional[EpisodicMemory] = None,
                rng: Optional[_random.Random] = None) -> "CognitiveRuntime":
        return CognitiveRuntime(generate_fn=self.generate_fn,
                                score_fn=self.score_fn,
                                embed_fn=self.embed_fn, nt=nt,
                                memory=memory, cfg=cfg, rng=rng,
                                classify_fn=self.classify_fn,
                                generate_wander_fn=self.generate_wander_fn,
                                reasoning_fn=self.reasoning_fn)


@dataclass
class MindConfig:
    """All cognition-loop knobs in one place."""

    n_candidates: int = 3
    """Candidate thoughts generated per tick (basal-ganglia menu size)."""

    thought_n_tok: int = 192
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
    """Selection softmax temperature at baseline NE, zero boredom."""

    ne_gain: float = 2.0
    """Adaptive gain (Aston-Jones & Cohen 2005): selection temperature
    is divided by 1 + ne_gain·max(0, NE−NE₀)/NE₀ — urgency makes the
    choice decisive."""

    goal_weight: float = 2.0
    """Phasic-DA goal pull in the BG utility, nats per unit cosine per
    DA₀ of release: U = −NLL + goal_weight·max(0, DA−DA₀)/DA₀·cos(
    candidate, goal). Zero at tonic DA. The goal itself is prefrontal
    working memory gated by the same phasic DA (Braver & Cohen 2000):
    goal ← goal + g·(content − goal), g = clip((DA−DA₀)/DA₀, 0, 1)."""

    hc_vta_coupling: float = 1.0
    """Hippocampus → VTA synapse (Lisman & Grace 2005): DA's coupling to
    the semantic-novelty driver in the runtime's NT system. The driver
    is z-scored against its running stats, so only unusual novelty
    releases DA and sustained novelty habituates."""

    appraisal: Dict[str, float] = field(default_factory=lambda: {
        "insult": 1.0, "disagreement": 0.5, "request": 0.5})
    """Amygdala→LC: arousal of a percept by its action class (the
    mind's own classifier — an LM in production). Feeds NE's driver;
    never touches text."""

    curiosity_value: float = 2.0
    """Intrinsic motivation as VALUE (Schmidhuber/Oudeyer): BG utility
    gains curiosity_value · boredom · novelty(candidate), novelty vs
    working memory ∪ stored episodes. Smoke run 58698239: likelihood
    alone rewarded copying the last thought (novelty 0.04)."""

    gen_boredom_gain: float = 0.67
    """Boredom also heats GENERATION: the trunk samples at its base
    temperature × (1 + gen_boredom_gain·boredom) (R1: 0.6 → 1.0), so
    the candidates themselves diversify, not just the choice."""

    mmr_lambda: float = 0.3
    """Hippocampal retrieval diversity at baseline ACh: recall picks
    episodes by maximal marginal relevance, relevance − λ·redundancy
    with episodes already recalled. See :func:`recall_diversity`."""

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

    max_prompt_chars: int = 4000
    """Hard cap on the composed thinking prompt, truncated from the
    LEFT. 2026-11-01: raised from 1200 — with R1's long thoughts the
    persona and recalled memories at the top fell off the prompt."""

    wm_size: int = 8
    """Working memory: novelty is measured against the last ``wm_size``
    SELECTED thoughts as well as stored episodes (0 = stored only).
    Run 58680199: a ~30-tick basin never registered as repetitive —
    with ~9% ignition almost nothing was stored to compare against."""

    global_workspace: bool = True
    """§14.13 workspace CONTENT: percepts, recalled memories and
    selected thoughts compete for ``ws_capacity`` slots and the
    survivors are the context of idle thought generation. Off → the
    prompt is recalled memories + the last thought, as before."""
    ws_capacity: int = 4
    """Workspace slots (Cowan's ~4-chunk working-memory limit)."""
    ws_decay: float = 0.7
    """Per-tick salience retention of every workspace item."""

    control_thresholds: str = "relative"
    """'relative' (H67): LOOPING and replay are judged against the run's
    own slow novelty baseline (``loop_ratio``, ``replay_margin``);
    'absolute': ``loop_novelty_threshold`` / ``replay_boredom_threshold``
    (tuned on SmolLM2; never fired on R1-7B's centred novelty)."""

    loop_ratio: float = 0.75
    """Relative LOOPING: novelty < loop_ratio × baseline. At SmolLM2's
    baseline (~0.3) this is ≈ the old absolute 0.25."""

    replay_margin: float = 0.1
    """Relative replay: boredom ≥ (1 − baseline) + replay_margin, i.e.
    boredom above what the run's own novelty level implies."""

    nov_baseline_alpha: float = 0.05
    """EWMA rate of the novelty baseline (~20-tick memory), seeded with
    the first novelty."""

    nt_driver_var_init: str = "warmup"
    """Variance init of the default NT's driver z-scores: 'warmup'
    takes it from the first samples ([0,1]-bounded runtime drivers were
    shrunk several-fold by the training-scale prior of 1.0 — run
    58678026: NE/ACh/GABA per-run sd 0.002-0.005)."""

    respond_persona: str = (
        "You are an always-on mind: you think in language and remember "
        "what happens to you.")
    """Second-person framing for REPLY prompts. Run 58689719: a
    first-person prompt ("I am BRIAN…") was read by R1 as a document
    to analyse ("Brian appears focused on…"). Reply prompts now
    address the mind; idle (DMN) prompts keep first-person inner
    speech (``persona``). No name: "You are BRIAN…" made R1 treat
    Brian as a third party (run 58692548)."""

    respond_instruction: str = (
        "Reply directly to what was just said, in the first person, as "
        "yourself. Do not describe or analyse this text.")
    """Last line of a reply prompt. R1-family models want instructions
    in the user turn, not a system prompt."""

    nt_driver_alpha: float = 0.1
    """EMA rate of the default DrivenNTSystem's driver z-scoring. Its
    own default (0.005, ~200 steps) suits training; over a dozen ticks
    it pins the running mean at the FIRST sample, so one early event
    shifts a channel for good (run 58673415: DA 0.15→0.05)."""

    workspace_oscillator: bool = True
    """§14.13 oscillatory global workspace: a Kuramoto gamma-band phase
    model over the runtime's modules decides whether the selected
    thought IGNITES (is broadcast: stored + spoken). Off → every
    novel thought broadcasts, as before §14.13."""

    osc_nt_feedback: bool = True
    """ACh scales the oscillator's coupling and GABA its frequency.
    Off → the ignition↔NT loop is cut (the H65 ablation)."""

    osc_coupling_ratio: float = 2.0
    """Coupling as a multiple of the Kuramoto critical coupling."""

    osc_ignition_threshold: float = 0.25
    """Fixed mode: the R that counts as ignition. Adaptive mode: the
    seed of the running baseline μ_R. See oscillator.py."""

    osc_ignition_mode: str = "adaptive"
    """'adaptive' (H66): ignite when R exceeds its own recent baseline
    by osc_adapt_k sd; 'fixed': R ≥ osc_ignition_threshold (H65)."""

    osc_ignition_floor: float = 0.1
    osc_adapt_k: float = 1.0

    osc_n_per_module: int = 8
    osc_f0_hz: float = 40.0
    osc_f_sd_hz: float = 4.0
    osc_noise: float = 2.0
    osc_ms_per_tick: int = 250
    osc_seed: int = 0

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
    ignited: Optional[bool] = None
    """§14.13: did the selected thought ignite the oscillatory
    workspace (global phase-locking ≥ threshold)? None when the
    oscillator is off."""
    sync_r: Optional[float] = None
    """Global Kuramoto order parameter R over the ignition window."""
    gamma_peak_hz: Optional[float] = None
    ignition_threshold: Optional[float] = None
    workspace_inputs: Optional[Dict[str, float]] = None
    """Module activations fed to the workspace this tick (sensory,
    memory, language, selection, novelty)."""
    workspace: List[Tuple[str, float, str]] = field(default_factory=list)
    """Global workspace contents after this tick as (source, salience,
    text), most salient last; empty when ``global_workspace`` is off."""
    gamma_power: Optional[float] = None
    """Fraction of the population signal's power in 30-100 Hz."""
    goal_relevance: float = 0.0
    """Cosine of the selected thought to the current goal (most recent
    observed input); 0 when there is no goal yet."""
    reasoning: Optional[str] = None
    """Reasoning experts only: the think trace that produced the
    SELECTED thought — the model's own visible deliberation, kept as
    an introspection channel (§14.12)."""
    reorient: bool = False
    """True when the LOOPING condition (a low-novelty streak) forced
    a state transition: recall anchored on the last OBSERVED input
    (return-to-user), or suppressed entirely (clean slate) — instead
    of another continuation of the loop."""


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
    return f"Φ={result.phi_proxy:.2f} {nt_str} {bg} {hc} {cur}"


def _one_line(text: str) -> str:
    """Full text on one trace line (2026-11-01: no more truncation —
    the 90-char cut read as thoughts being 'cut off')."""
    return str(text).replace("\n", " ⏎ ")


def format_debug_trace(result: TickResult) -> str:
    """Multi-line basal-ganglia deliberation trace — "debug log the
    actions generated by basal ganglia": the act (respond/speak/think),
    every candidate thought + its trunk NLL + which one won (all text
    in full, one line each), the NT snapshot, IIT-flavored differentiation/
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

    if result.candidates and not result.inhibited:
        lines.append(f"  CUR[nov={result.novelty:.2f} "
                     f"bore={result.boredom:.2f}]")
    if result.sync_r is not None:
        wi = result.workspace_inputs or {}
        ins = " ".join(f"{k[0]}={wi.get(k, 0.0):.2f}" for k in
                       ("sensory", "memory", "language", "selection",
                        "novelty"))
        lines.append(
            f"  WS[R={result.sync_r:.2f} θ={result.ignition_threshold:.2f} "
            f"ignited={'yes' if result.ignited else 'no'} "
            f"{result.gamma_peak_hz:.0f}Hz γ={result.gamma_power:.2f} "
            f"in[{ins}]]")

    if result.inhibited:
        lines.append("  BG: inhibited — no deliberation this tick "
                     "(GABA gated the act before THINK ran)")
        return "\n".join(lines)
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
        text = _one_line(cand)
        lines.append(f"    [{mark}] nll={sc.mean_nll:5.2f} \"{text}\"")

    if result.reasoning:
        lines.append(f"  think trace (SELECTED): \"{result.reasoning}\"")

    if result.recalled:
        lines.append(f"  HC recalled {len(result.recalled)}:")
        for e in result.recalled:
            kind = (e.get("context") or {}).get("kind")
            label = f"[{kind}] " if kind else ""
            lines.append(f"    - {label}{_one_line(e.get('content', ''))}")

    if result.workspace:
        lines.append(f"  GW in mind {len(result.workspace)}:")
        for src, sal, text in result.workspace:
            lines.append(f"    - [{src} {sal:.2f}] {_one_line(text)}")

    if result.prior_thought:
        lines.append(f"  evolved from: \"{_one_line(result.prior_thought)}\"")

    return "\n".join(lines)


_WS_LABEL = {"percept": "Earlier: ", "memory": "I remember: ",
             "thought": "I thought: "}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _mmr(pool: List[Tuple[float, dict]], k: int,
         lam: float) -> List[dict]:
    """Greedy maximal-marginal-relevance pick of ``k`` episodes from
    (relevance, episode) pairs: relevance − λ·max cos to those chosen."""
    chosen: List[dict] = []
    left = list(pool)
    while left and len(chosen) < k:
        def mmr(t):
            v = t[1].get("content_vec")
            red = max((_cosine(v, c["content_vec"]) for c in chosen
                       if v is not None and c.get("content_vec") is not None),
                      default=0.0)
            return t[0] - lam * red
        best = max(left, key=mmr)
        chosen.append(best[1])
        left.remove(best)
    return chosen


def _population_stdev(values: List[float]) -> float:
    """IIT-flavored differentiation proxy: population stdev of the
    candidate repertoire's NLLs. 0 for <2 values (nothing to
    differentiate)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return var ** 0.5


def selection_temperature(ne_level: float, ne_baseline: float,
                          boredom: float, cfg: MindConfig) -> float:
    """BG softmax temperature: T₀ · (1 + curiosity_gain·boredom)
    / (1 + ne_gain·max(0, NE−NE₀)/NE₀). Boredom heats selection
    (exploration); NE above baseline cools it (adaptive gain: urgency
    → decisive choice)."""
    heat = 1.0 + cfg.curiosity_gain * max(0.0, min(1.0, boredom))
    excess = max(0.0, float(ne_level) - float(ne_baseline))
    gain = 1.0 + cfg.ne_gain * excess / max(float(ne_baseline), 1e-6)
    return cfg.selection_temp_base * heat / gain


def recall_diversity(ach_level: float, ach_baseline: float,
                     cfg: MindConfig) -> float:
    """MMR redundancy weight λ, set by ACh (Hasselmo): λ₀·ACh₀/ACh,
    clipped to [λ₀/2, 2λ₀] — low ACh = diverse associative recall,
    high ACh = focused recall."""
    ratio = float(ach_baseline) / max(float(ach_level), 1e-6)
    return cfg.mmr_lambda * max(0.5, min(2.0, ratio))


def loop_threshold(baseline: Optional[float], cfg: "MindConfig") -> float:
    """Novelty below which a tick counts toward the LOOPING streak."""
    if cfg.control_thresholds == "absolute" or baseline is None:
        return cfg.loop_novelty_threshold
    return cfg.loop_ratio * baseline


def replay_threshold(baseline: Optional[float], cfg: "MindConfig") -> float:
    """Boredom at which an idle tick may replay."""
    if cfg.control_thresholds == "absolute" or baseline is None:
        return cfg.replay_boredom_threshold
    return (1.0 - baseline) + cfg.replay_margin


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
                 reasoning_fn: Optional[Callable[[], Optional[str]]] = None,
                 ) -> None:
        cfg = cfg or MindConfig()
        if nt is None:
            from neuroslm.emergent.driven_nt import (
                _DEFAULT_W, _DRIVERS, DrivenNTSystem)
            da_row = list(_DEFAULT_W["DA"])
            da_row[_DRIVERS.index("activation")] = cfg.hc_vta_coupling
            nt = DrivenNTSystem(driver_alpha=cfg.nt_driver_alpha,
                                driver_var_init=cfg.nt_driver_var_init,
                                W={"DA": tuple(da_row)})
        self._gen = generate_fn
        # Inner-speech register (D of the curiosity loop): DMN ticks
        # generate through this seam — raw completion, no chat
        # template, no imaginary addressee. Defaults to the main seam
        # for injected/test setups; build_runtime_from_hf_lm wires a
        # real completion-mode closure.
        self._gen_wander = generate_wander_fn or generate_fn
        self._reasoning = reasoning_fn
        self._score = score_fn
        self._embed = embed_fn
        if classify_fn is None:
            from neuroslm.cognition.patterns import classify_action
            classify_fn = classify_action
        self._classify = classify_fn
        self.nt = nt
        self.memory = memory if memory is not None else EpisodicMemory(512)
        self.cfg = cfg
        self.rng = rng or _random.Random(0)
        self._sensory: Deque[str] = deque(maxlen=32)
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
        # NE's novelty expectation, seeded with the first observation
        # (boredom starts at 0 by design, which made it a biased
        # expectation for ~10 ticks — run 58676023).
        self._nov_expect: Optional[float] = None
        # H67: working memory of selected-thought vectors + the run's
        # own slow novelty baseline for relative control thresholds.
        self._wm: Deque[List[float]] = deque(maxlen=max(1, self.cfg.wm_size))
        self._nov_base: Optional[float] = None
        self._goal_vec: Optional[List[float]] = None
        self.workspace = None
        if self.cfg.global_workspace:
            from neuroslm.cognition.workspace import GlobalWorkspace
            self.workspace = GlobalWorkspace(capacity=self.cfg.ws_capacity,
                                             decay=self.cfg.ws_decay)
        self._osc = None
        if self.cfg.workspace_oscillator:
            from neuroslm.cognition.oscillator import GammaWorkspace
            base = nt.baselines() if callable(nt.baselines) else nt.baselines
            c = self.cfg
            self._osc = GammaWorkspace(
                n_per_module=c.osc_n_per_module, f0_hz=c.osc_f0_hz,
                f_sd_hz=c.osc_f_sd_hz, coupling_ratio=c.osc_coupling_ratio,
                noise=c.osc_noise, ms_per_tick=c.osc_ms_per_tick,
                ignition_threshold=c.osc_ignition_threshold,
                nt_feedback=c.osc_nt_feedback,
                ignition_mode=c.osc_ignition_mode,
                ignition_floor=c.osc_ignition_floor, adapt_k=c.osc_adapt_k,
                ach0=base.get("ACh", 0.30), gaba0=base.get("GABA", 0.15),
                seed=c.osc_seed)

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
        text_vec = list(self._embed(text))
        n_words = len(text.split())
        trivial = (
            (action_class in self.cfg.trivial_percept_classes
             and n_words <= self.cfg.trivial_percept_max_words)
            or (action_class == "statement"
                and n_words < self.cfg.percept_min_words)
        )
        if not trivial:
            self.memory.add(text,
                            content_vec=text_vec,
                            nt_state=self.nt.levels(),
                            tags=[source, "percept", "kind=observed",
                                 f"action_class={action_class}"],
                            context={"kind": "observed", "source": source,
                                    "action_class": action_class})
        self._sensory.append(text)
        self.nt.step_full(
            activation=1.0,
            grad_norm=float(self.cfg.appraisal.get(action_class, 0.0)))
        self._gate_goal(text_vec)

    def remember(self, text: str, source: str = "autobiographical") -> None:
        """Lay down prior autobiographical memory: an OBSERVED episode
        that is not a percept of the current moment — no sensory queue,
        no goal, no NT drive. Gives wandering recall and replay real
        associations to follow (smoke run 58700866: an amnesic mind
        could only recall its seed sentence)."""
        text = (text or "").strip()
        if not text:
            return
        self.memory.add(text,
                        content_vec=list(self._embed(text)),
                        nt_state=self.nt.levels(),
                        tags=[source, "memory", "kind=observed"],
                        context={"kind": "observed", "source": source})

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
        self._sensory.clear()
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

        # WORKSPACE (§14.13): contents fade; percepts enter at full
        # salience; a reorient drops the train of thought.
        if self.workspace is not None:
            self.workspace.decay()
            if reorient:
                self.workspace.drop("thought")
            for s in sensory:
                self.workspace.offer(s, "percept", list(self._embed(s)), 1.0)

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
              and self._boredom >= replay_threshold(self._nov_base,
                                                    self.cfg)
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
        recall_strength = 0.0
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
            # Content already IN the prompt is never recalled: the
            # current percepts ("Just now:", stored by observe() before
            # this tick — §14.11) and the last thought ("My last
            # thought:"). Recalled, each is its own nearest neighbour
            # (cos 1); the last thought's self-recall also pinned the
            # workspace's memory input at 1.0 after every broadcast
            # (run 58676023: 4/5 ignitions at m=1.00).
            current = set(sensory)
            if self._last_thought:
                current.add(self._last_thought)
            adjusted = sorted(
                ((sim - (pen if (e.get("context") or {}).get("kind")
                         == "inferred" else 0.0), e)
                 for sim, e in scored
                 if e.get("content") not in current),
                key=lambda t: t[0], reverse=True)
            # IOR is a DMN anti-fixation device: on a respond tick the
            # most relevant memory wins even if it was recalled lately
            # (§14.11 episodic_recall probe).
            fresh = [e for _, e in adjusted
                     if not wandering
                     or e.get("content") not in self._recent_recall]
            pool = ([t for t in adjusted if t[1] in fresh] if fresh
                    else adjusted)
            base = self.nt.baselines
            base = base() if callable(base) else base
            lam = recall_diversity(levels.get("ACh", 0.30),
                                   base.get("ACh", 0.30), self.cfg)
            recalled = _mmr(pool, self.cfg.recall_k, lam)
            for e in recalled:
                self._recent_recall.append(e.get("content"))
            sims = {id(e): sim for sim, e in scored}
            recall_strength = max((sims.get(id(e), 0.0) for e in recalled),
                                  default=0.0)
            if self.workspace is not None:
                for e in recalled:
                    inferred = ((e.get("context") or {}).get("kind")
                                == "inferred")
                    self.workspace.offer(
                        e.get("content", ""),
                        "thought" if inferred else "memory",
                        e.get("content_vec") or [],
                        max(0.0, sims.get(id(e), 0.0)))

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
        # REORIENT is a state transition: the train of thought is
        # dropped from the prompt, not just re-anchored in recall.
        prompt = self._compose_prompt(sensory, recalled, trigger_text,
                                      keep_last_thought=not reorient)
        gen_scale = 1.0 + self.cfg.gen_boredom_gain * max(
            0.0, min(1.0, self._boredom))
        gen = self._gen if not wandering else self._gen_wander
        candidates: List[str] = []
        reasonings: List[Optional[str]] = []
        for _ in range(max(1, self.cfg.n_candidates)):
            if getattr(gen, "accepts_temperature_scale", False):
                out = gen(prompt, self.cfg.thought_n_tok,
                          temperature_scale=gen_scale)
            else:
                out = gen(prompt, self.cfg.thought_n_tok)
            out = (out or "").strip()
            if out:
                candidates.append(out)
                reasonings.append(self._reasoning() if self._reasoning
                                  else None)
        if not candidates:
            self.nt.step_full()
            return TickResult(thought=None, recalled=recalled,
                              nt_levels=levels, tick_n=tick_n,
                              prior_thought=prior_thought,
                              wandering=wandering,
                              action="respond" if not wandering else "think")

        # GATE: BG softmax over likelihood + DA goal pull + curiosity
        # value, at a boredom-heated / NE-cooled temperature.
        scores = [self._score(c) for c in candidates]
        cand_vecs = [list(self._embed(c)) for c in candidates]
        goal_rel = [(_cosine(v, self._goal_vec) if self._goal_vec else 0.0)
                    for v in cand_vecs]
        cand_nov = [self._novelty_of(v) for v in cand_vecs]
        idx, selection_entropy = self._select(scores, levels, goal_rel,
                                              cand_nov)
        thought, sc = candidates[idx], scores[idx]
        differentiation = _population_stdev([s.mean_nll for s in scores])

        # NOVELTY (A of the curiosity loop): semantic novelty of the
        # selected thought — 1 − max cosine vs every stored episode.
        # Computed BEFORE storing (a thought can't be compared against
        # itself). One signal, three consumers: the write gate below,
        # the NT activation driver, and the boredom trace.
        thought_vec = cand_vecs[idx]
        top = self.memory.retrieve_scored(thought_vec, k=1)
        best = top[0][0] if top else None
        if self.cfg.wm_size > 0:
            for v in self._wm:
                c = _cosine(thought_vec, v)
                best = c if best is None else max(best, c)
        novelty = 1.0 - best if best is not None else 1.0
        novelty = max(0.0, min(1.0, novelty))

        # ACTION: the basal ganglia's actual act this tick — exactly
        # one of respond/reorient/speak/think. External input always
        # surfaces (respond); a forced state transition reports itself
        # (reorient — surfaces, so the shift is visible); otherwise
        # the SAME novelty gate that decided whether to remember the
        # thought also decides whether to voice it (speak) or keep it
        # internal (think) — one signal, two consequences.
        # WORKSPACE (§14.13): the selected thought is broadcast only if
        # the module oscillators phase-lock globally this tick.
        ws_inputs = {
            "sensory": 1.0 if sensory else 0.0,
            "memory": recall_strength,
            "language": 1.0 - sc.entropy_norm,
            "selection": 1.0 - selection_entropy,
            "novelty": novelty,
        }
        osc = None
        if self._osc is not None:
            osc = self._osc.step(ws_inputs, levels,
                                 update_baseline=wandering)
        stored = (novelty >= self.cfg.novelty_write_threshold
                  and (osc is None or osc.ignited))
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
        # §14.13: NE ← novelty surprise against the curiosity
        # homeostat's expectation (1 − boredom); GABA/−ACh ← workspace
        # ignition (or broadcast, without the oscillator); ACh ← BG
        # decisiveness. Before this, NE/ACh/GABA never moved.
        ignition = float(osc.ignited) if osc is not None else float(stored)
        if self.workspace is not None:
            from neuroslm.cognition.workspace import thought_salience
            self.workspace.offer(thought, "thought", thought_vec,
                                 thought_salience(novelty, self._nov_base,
                                                  ignited=bool(ignition)))
        ne_drive = (0.0 if self._nov_expect is None
                    else abs(novelty - self._nov_expect))
        a_n = self.cfg.novelty_ema_alpha
        self._nov_expect = (novelty if self._nov_expect is None else
                            (1 - a_n) * self._nov_expect + a_n * novelty)
        self.nt.step_full(loss=sc.mean_nll, activation=novelty,
                          grad_norm=ne_drive,
                          ignition_rate=ignition,
                          attn_entropy_norm=selection_entropy)
        self._gate_goal(thought_vec)
        a = self.cfg.novelty_ema_alpha
        self._boredom = (1.0 - a) * self._boredom + a * (1.0 - novelty)

        # LOOPING streak (control fix 3) + pulse relief (fix 2/3): a
        # replay jump or a reorient CONSUMES boredom — the correction
        # IS the relief; without this, replay locked on and fired
        # every tick once boredom saturated (live trace, tick 10+).
        self._update_control(novelty)
        if self.cfg.wm_size > 0:
            self._wm.append(thought_vec)
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
                          goal_relevance=goal_rel[idx],
                          differentiation=differentiation,
                          novelty=novelty, boredom=self._boredom,
                          replay=replay, reorient=reorient,
                          reasoning=reasonings[idx],
                          ignited=osc.ignited if osc else None,
                          sync_r=osc.sync_r if osc else None,
                          gamma_peak_hz=osc.peak_hz if osc else None,
                          gamma_power=osc.gamma_power if osc else None,
                          workspace_inputs=ws_inputs,
                          workspace=(self.workspace.items_view()
                                     if self.workspace is not None else []),
                          ignition_threshold=osc.threshold if osc else None)

    # ── Internals ────────────────────────────────────────────────────

    def _novelty_of(self, vec: List[float]) -> float:
        """1 − max cosine vs stored episodes ∪ working memory."""
        top = self.memory.retrieve_scored(vec, k=1)
        best = top[0][0] if top else None
        if self.cfg.wm_size > 0:
            for v in self._wm:
                c = _cosine(vec, v)
                best = c if best is None else max(best, c)
        return max(0.0, min(1.0, 1.0 - best)) if best is not None else 1.0

    def _gate_goal(self, vec: List[float]) -> None:
        """PFC goal update gated by phasic DA: the content active
        during a burst is loaded in proportion to the burst."""
        base = self.nt.baselines
        da0 = (base() if callable(base) else base).get("DA", 0.15)
        g = max(0.0, min(1.0, (self.nt.levels().get("DA", 0.0) - da0)
                         / max(da0, 1e-6)))
        if g <= 0.0:
            return
        if self._goal_vec is None:
            self._goal_vec = list(vec)
        else:
            self._goal_vec = [x + g * (v - x)
                              for x, v in zip(self._goal_vec, vec)]

    def _update_control(self, novelty: float) -> None:
        """LOOPING streak against the (relative or absolute) threshold,
        then fold this tick's novelty into the slow baseline."""
        if self._nov_base is None:
            self._nov_base = novelty
        thr = loop_threshold(self._nov_base, self.cfg)
        if novelty < thr:
            self._low_nov_streak += 1
        else:
            self._low_nov_streak = 0
        a = self.cfg.nov_baseline_alpha
        self._nov_base = (1 - a) * self._nov_base + a * novelty

    def _compose_prompt(self, sensory: List[str],
                        recalled: List[dict], trigger_text: str,
                        keep_last_thought: bool = True) -> str:
        if sensory:
            return self._compose_reply_prompt(sensory, recalled)
        lines: List[str] = [self.cfg.persona, ""]
        if self.workspace is not None:
            items = self.workspace.items()
            if items:
                lines.append("On my mind:")
                for it in items:
                    lines.append(f"- {_WS_LABEL.get(it.source, '')}{it.text}")
                lines.append("")
            lines.append(trigger_text)
            lines.append("Thought:")
            return "\n".join(lines)[-self.cfg.max_prompt_chars:]
        if recalled:
            lines.append("I remember:")
            for e in recalled:
                lines.append(f"- {e['content']}")
            lines.append("")
        if self._last_thought and keep_last_thought:
            lines.append(f"My last thought: {self._last_thought}")
        # DMN: nothing external arrived — bias THINK toward
        # associative, reflective continuation instead of a flat
        # completion of nothing. `trigger_text` is the wander prompt
        # already chosen by the caller (tick()), so it can be recorded
        # as this episode's Layer-2 trigger too.
        lines.append(trigger_text)
        lines.append("Thought:")
        return "\n".join(lines)[-self.cfg.max_prompt_chars:]

    def _compose_reply_prompt(self, sensory: List[str],
                              recalled: List[dict]) -> str:
        """A conversation turn ADDRESSED to the mind (second person):
        context, then what was just said, then the reply instruction.
        Truncation keeps the head (persona) and the tail (input +
        instruction); the middle (memories) gives way first."""
        head = [self.cfg.respond_persona, ""]
        mid: List[str] = []
        if recalled:
            mid.append("Things you remember:")
            mid.extend(f"- {e['content']}" for e in recalled)
            mid.append("")
        if self._last_thought:
            mid.append(f"Your last thought: {self._last_thought}")
            mid.append("")
        tail = [f"Just now: {s}" for s in sensory]
        tail.append(self.cfg.respond_instruction)
        head_s, tail_s = "\n".join(head), "\n".join(tail)
        room = self.cfg.max_prompt_chars - len(head_s) - len(tail_s) - 2
        mid_s = "\n".join(mid)
        if room <= 0:
            mid_s = ""
        elif len(mid_s) > room:
            mid_s = mid_s[-room:]
        return "\n".join(x for x in (head_s, mid_s, tail_s) if x)

    def _next_wander_prompt(self) -> str:
        """Round-robin over ``cfg.wander_prompts`` so idle mind-
        wandering doesn't repeat the exact same framing every tick."""
        prompts = self.cfg.wander_prompts or ("",)
        p = prompts[self._wander_idx % len(prompts)]
        self._wander_idx += 1
        return p

    def _select(self, scores: List[ThoughtScore],
                levels: Dict[str, float],
                goal_rel: Optional[List[float]] = None,
                novelty: Optional[List[float]] = None
                ) -> Tuple[int, float]:
        """Basal-ganglia selection. Returns ``(chosen_index,
        selection_entropy)`` — the latter an IIT-flavored decisiveness
        proxy: normalised entropy of the softmax(utility/T) choice
        distribution (0 = one option clearly excluded the rest, 1 =
        arbitrary among near-equal options).

        Utility = −NLL + goal_weight·max(0, DA−DA₀)/DA₀·cos(candidate,
        goal): the trunk's own likelihood plus a pull toward the current
        goal that only phasic (released) dopamine switches on. Temperature = :func:`selection_temperature`
        (boredom heats, NE cools). No content rules (2026-11-01).
        """
        if len(scores) == 1:
            return 0, 0.0
        # DrivenNTSystem exposes `baselines` as a property while
        # `levels()` is a method — accept both shapes so duck-typed
        # NT systems keep working.
        baselines = self.nt.baselines
        if callable(baselines):
            baselines = baselines()
        T = selection_temperature(
            levels.get("NE", 0.20), baselines.get("NE", 0.20),
            self._boredom, self.cfg)
        da0 = baselines.get("DA", 0.15)
        da_gain = max(0.0, levels.get("DA", 0.0) - da0) / max(da0, 1e-6)
        rel = goal_rel or [0.0] * len(scores)
        nov = novelty or [0.0] * len(scores)
        cur = self.cfg.curiosity_value * max(0.0, min(1.0, self._boredom))
        utils = [(-s.mean_nll + self.cfg.goal_weight * da_gain * g + cur * n)
                 / max(T, 1e-6) for s, g, n in zip(scores, rel, nov)]
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

def _mean_centred_embed_fn(embed_table: Any, tokenizer: Any,
                           device: str) -> EmbedFn:
    """Mean-pooled token embedding MINUS the embedding-table mean.

    Input embeddings share a large common component (anisotropy), so
    raw mean-pooled cosines between unrelated texts sit high and an
    absolute novelty threshold stops meaning the same thing across
    trunks (run 58673415, R1-7B: 16/16 ticks failed the 0.3 novelty
    gate). Removing the common mean is the standard correction
    (Mu & Viswanath 2018, "All-but-the-Top")."""
    import torch

    table_mean = embed_table.detach().float().mean(dim=0)

    @torch.no_grad()
    def embed_fn(text: str) -> Sequence[float]:
        ids = tokenizer.encode(text or " ") or [0]
        rows = embed_table[torch.tensor(ids, dtype=torch.long,
                                        device=device)]
        return (rows.float().mean(dim=0) - table_mean).cpu().tolist()

    return embed_fn


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

    embed_fn = _mean_centred_embed_fn(embed_table, tokenizer, device)

    # The mind classifies with its OWN trunk — reuses the SAME
    # generate_fn closure THINK uses (one model, two jobs), never the
    # bare regex lexicon by default (§14.8).
    from neuroslm.cognition.patterns import classify_action_via_generation

    def classify_fn(text: str) -> "ActionClassification":
        return classify_action_via_generation(text, generate_fn)

    return CognitiveRuntime(generate_fn=generate_fn, score_fn=score_fn,
                            embed_fn=embed_fn, cfg=cfg,
                            classify_fn=classify_fn)


def build_runtime_from_hf_lm(model_id: str = DEFAULT_MIND_EXPERT,
                             device: str = "cpu",
                             cfg: Optional[MindConfig] = None,
                             temperature: float = 0.8,
                             top_k: int = 40,
                             model_factory: Optional[Callable[[], Any]] = None,
                             tokenizer_factory: Optional[Callable[[], Any]] = None,
                             ) -> CognitiveRuntime:
    """Wire the mind to a frozen pretrained HF expert — no trunk needed.
    See :func:`build_hf_seams` for the wiring itself."""
    return build_hf_seams(model_id, device=device, temperature=temperature,
                          top_k=top_k, model_factory=model_factory,
                          tokenizer_factory=tokenizer_factory,
                          ).runtime(cfg=cfg)


def build_hf_seams(model_id: str = DEFAULT_MIND_EXPERT,
                   device: str = "cpu",
                   temperature: float = 0.8,
                   top_k: int = 40,
                   model_factory: Optional[Callable[[], Any]] = None,
                   tokenizer_factory: Optional[Callable[[], Any]] = None,
                   reasoning_budget: int = 1024,
                   ) -> MindSeams:
    """Build the mind's seams from a frozen pretrained HF expert.

    The doctrine's escape hatch for using BRIAN before full trunk
    training: THINK/score/embed all come from one of the LM experts
    (default :data:`DEFAULT_MIND_EXPERT`) instead of a
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
        model = AutoModelForCausalLM.from_pretrained(
            resolved, **_hf_load_kwargs(device))
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
    last: Dict[str, Optional[str]] = {"reasoning": None}

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
        # Reasoning experts (DeepSeek-R1 family) open a <think> block
        # from their chat template.
        reasoning_model = _THINK_OPEN in str(
            getattr(tokenizer, "chat_template", "") or "")
        eos_text = getattr(tokenizer, "eos_token", None)

        def _decode_raw(new_ids) -> str:
            # Keep <think> markers (skip_special_tokens may strip them).
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            return text.replace(eos_text, "") if eos_text else text

        def _make_generate(use_chat_template: bool) -> GenerateFn:
            """One closure factory, two seams (D of the curiosity
            loop): the chat seam (respond — an addressee exists) and
            the raw-completion wander seam (inner speech — no chat
            template, so the instruct model continues the thinker's
            own first-person text instead of replying TO someone,
            which is what produced 'Brian, ...' on every live tick)."""

            @torch.no_grad()
            def _sample(ids: List[int], n: int,
                        scale: float = 1.0) -> List[int]:
                ids = ids[-(max_ctx - max(1, int(n))):]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                gen_kwargs = dict(
                    input_ids=x,
                    # single unpadded sequence; explicit because R1's
                    # pad token == eos (run 58673415 warning)
                    attention_mask=torch.ones_like(x),
                    max_new_tokens=int(n),
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=eos_id,
                )
                if reasoning_model:
                    # DeepSeek's recommended R1 sampling.
                    gen_kwargs.update(temperature=0.6 * scale, top_p=0.95,
                                      top_k=0)
                else:
                    # No repetition bans or stop strings for any trunk
                    # (2026-11-01 design principle: no output-shaping
                    # rules; the 2026-08-11 guards against base-SmolLM2
                    # loops are gone with it).
                    gen_kwargs.update(
                        temperature=float(max(temperature, 1e-6)) * scale,
                        top_k=int(top_k) if top_k else 0)
                out = model.generate(**gen_kwargs)
                return out[0, x.shape[1]:].tolist()

            @torch.no_grad()
            def _generate(prompt: str, max_new_tokens: int,
                          temperature_scale: float = 1.0) -> str:
                n = int(max_new_tokens)
                if use_chat_template:
                    enc = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True, tokenize=True)
                    # transformers >= 5 returns a dict-like
                    # BatchEncoding here, < 5 a plain id list.
                    if hasattr(enc, "keys"):
                        enc = enc["input_ids"]
                    ids = list(enc)
                else:
                    ids = list(tokenizer.encode(prompt) or [0])
                last["reasoning"] = None
                if use_chat_template and reasoning_model:
                    new = _sample(ids, reasoning_budget + n,
                                  temperature_scale)
                    text = _decode_raw(new)
                    if _THINK_CLOSE in text:
                        answer, last["reasoning"] = split_reasoning(text)
                        return answer
                    # Budget forcing (Muennighoff et al. 2025, s1):
                    # close an overlong think block and generate the
                    # answer as its continuation.
                    think = new[:reasoning_budget]
                    last["reasoning"] = _decode_raw(think).replace(
                        _THINK_OPEN, "").strip()
                    close = list(tokenizer.encode(
                        "\n" + _THINK_CLOSE + "\n\n"))
                    return split_reasoning(
                        _decode_raw(_sample(ids + think + close, n,
                                            temperature_scale)))[0]
                text = tokenizer.decode(_sample(ids, n, temperature_scale),
                                        skip_special_tokens=True)
                if reasoning_model:
                    answer, reasoning = split_reasoning(text)
                    text = answer or reasoning
                return text

            # the runtime passes a boredom-driven temperature scale
            _generate.accepts_temperature_scale = True
            return _generate

        generate_fn = _make_generate(has_chat_template)
        generate_wander_fn = _make_generate(False)
    else:
        reasoning_model = False
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

    embed_fn = _mean_centred_embed_fn(embed_table, tokenizer, device)

    # The mind classifies with its OWN trunk/expert — reuses the SAME
    # generate_fn closure THINK uses (one model, two jobs), never the
    # bare regex lexicon by default (§14.8).
    from neuroslm.cognition.patterns import classify_action_via_generation

    def classify_fn(text: str) -> "ActionClassification":
        return classify_action_via_generation(text, generate_fn)

    def reseed(seed: int) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    return MindSeams(generate_fn=generate_fn, score_fn=score_fn,
                     embed_fn=embed_fn, classify_fn=classify_fn,
                     generate_wander_fn=generate_wander_fn, reseed=reseed,
                     reasoning_fn=lambda: (last["reasoning"]
                                           if reasoning_model else None))
