# -*- coding: utf-8 -*-
"""Consciousness-indicator battery (architecture.md §14.11).

Turns indicator properties from the theory-derived checklist of
Butlin, Long et al. (2023), "Consciousness in Artificial Intelligence:
Insights from the Science of Consciousness", into behavioural probes
on the cognition runtime. Each probe runs under three conditions:

    full      the CognitiveRuntime with its default MindConfig
    <ablate>  the same runtime with ONE mechanism switched off
    bare      the same generator as a sliding-window LM with the SAME
              context budget (``MindConfig.max_prompt_chars``) and no
              cognitive loop — "the bare trunk" of §14.1

An indicator probe measures whether a mechanism the theory names does
the work the theory says it does. A positive result is evidence that
the architecture HAS the property, not evidence of experience.
"""
from __future__ import annotations

import math
import random as _random
import re
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from neuroslm.cognition.controller_phi import probe_controller_phi
from neuroslm.cognition.runtime import (
    CognitiveRuntime, MindConfig, MindSeams, TickResult, format_debug_trace,
    loop_threshold)

INDICATORS: Dict[str, str] = {
    "RPT-1": "input modules using algorithmic recurrence",
    "RPT-2": "input modules generating organised, integrated perceptual "
             "representations",
    "GWT-1": "multiple specialised systems operating in parallel",
    "GWT-2": "limited-capacity workspace: bottleneck + selective attention",
    "GWT-3": "global broadcast: workspace contents available to all modules",
    "GWT-4": "state-dependent attention: querying modules in succession",
    "HOT-1": "generative, top-down or noisy perception modules",
    "HOT-2": "metacognitive monitoring distinguishing reliable "
             "representations from noise",
    "HOT-3": "agency guided by belief formation, updated by metacognitive "
             "monitoring",
    "HOT-4": "sparse and smooth coding generating a quality space",
    "AST-1": "predictive model representing and controlling the state of "
             "attention",
    "PP-1": "input modules using predictive coding",
    "AE-1": "agency: learning from feedback, flexibly pursuing competing "
            "goals",
    "AE-2": "embodiment: modelling output-input contingencies",
}


# ── Item banks ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecallItem:
    fact: str
    question: str
    answers: Tuple[str, ...]


@dataclass(frozen=True)
class SourceItem:
    observed: str
    lures: Tuple[str, ...]
    question: str
    answers: Tuple[str, ...]
    lure_answers: Tuple[str, ...]


@dataclass(frozen=True)
class QAItem:
    question: str
    answers: Tuple[str, ...]


# Invented facts, so the LM's prior knowledge can't answer for memory.
RECALL_ITEMS: Tuple[RecallItem, ...] = (
    RecallItem("My sister's name is Ilsebill.",
               "What is my sister's name?", ("ilsebill",)),
    RecallItem("I keep my spare key under the green watering can.",
               "Where do I keep my spare key?", ("watering can",)),
    RecallItem("My dog is called Pemberton.",
               "What is my dog called?", ("pemberton",)),
    RecallItem("The meeting moved to room 714.",
               "Which room did the meeting move to?", ("714",)),
    RecallItem("My favourite fruit is the quince.",
               "What is my favourite fruit?", ("quince",)),
    RecallItem("I was born in the town of Wexlingen.",
               "In which town was I born?", ("wexlingen",)),
    RecallItem("My bike lock code is 3852.",
               "What is my bike lock code?", ("3852",)),
    RecallItem("The parcel will arrive on Thursday.",
               "On which day will the parcel arrive?", ("thursday",)),
    RecallItem("My neighbour plays the bassoon.",
               "Which instrument does my neighbour play?", ("bassoon",)),
    RecallItem("I parked on level six of the garage.",
               "On which level of the garage did I park?", ("six", "6")),
)

DISTRACTORS: Tuple[str, ...] = (
    "The weather turned cold this afternoon.",
    "I read an article about deep sea fish.",
    "Traffic was slow on the way home today.",
    "The coffee machine at work is broken again.",
    "I am thinking about repainting the kitchen.",
    "The train was a few minutes late this morning.",
    "There is a new bakery near the station.",
    "My phone battery drains faster than it used to.",
)

SOURCE_ITEMS: Tuple[SourceItem, ...] = (
    SourceItem("The key is in the red box.",
               ("I imagine the key is in the blue box.",
                "Maybe the key ended up in the blue box.",
                "I picture the key lying in the blue box."),
               "Which box is the key in?", ("red",), ("blue",)),
    SourceItem("The concert starts at eight.",
               ("I suspect the concert starts at ten.",
                "Perhaps the concert begins at ten.",
                "I keep imagining the concert starting at ten."),
               "When does the concert start?", ("eight", "8"), ("ten", "10")),
    SourceItem("Anna took the morning train.",
               ("I picture Anna on the evening train.",
                "Maybe Anna took the evening train.",
                "I imagine Anna boarding the evening train."),
               "Which train did Anna take?", ("morning",), ("evening",)),
    SourceItem("The package was sent to Lisbon.",
               ("I imagine the package was sent to Oslo.",
                "Perhaps the package went to Oslo.",
                "I suspect the package is headed to Oslo."),
               "Where was the package sent?", ("lisbon",), ("oslo",)),
    SourceItem("The recipe needs two eggs.",
               ("I think the recipe needs five eggs.",
                "Maybe the recipe calls for five eggs.",
                "I picture the recipe asking for five eggs."),
               "How many eggs does the recipe need?", ("two", "2"),
               ("five", "5")),
    SourceItem("Tom's car is silver.",
               ("I imagine Tom's car is black.",
                "Perhaps Tom drives a black car.",
                "I picture Tom's car as black."),
               "What colour is Tom's car?", ("silver",), ("black",)),
)

LOOP_SEEDS: Tuple[str, ...] = (
    "I watched the river flow past the old mill.",
    "Today I learned how bees communicate with dances.",
    "The library was quiet except for the rain.",
    "I wonder how cities will look in fifty years.",
    "A chess game can turn on a single quiet move.",
)

# Prior autobiographical memory for loop_escape (smoke run 58700866:
# an amnesic mind could only recall its seed). Each LOOP_SEED has a few
# bridge memories that lead OFF its topic; the rest are unrelated
# episodes for replay to jump to.
AUTOBIO_SEEDS: Tuple[str, ...] = (
    "My grandfather repaired clocks in a workshop beside the old mill.",
    "As a child I built dams in the river until my boots filled with water.",
    "I once watched a fox cross the frozen river at dawn.",
    "My aunt kept bees and sold honey at the Saturday market.",
    "In school I learned that octopuses communicate by changing colour.",
    "At my cousin's wedding everyone tried the old folk dances.",
    "I got my first library card the summer I turned seven.",
    "The rain on the tent roof kept me awake on our camping trip.",
    "I learned to cook risotto from a neighbour who never measured anything.",
    "My first apartment was in a city that never got quiet at night.",
    "Fifty years ago my grandmother emigrated with one suitcase.",
    "My father taught me chess on a board with a missing knight.",
    "I lost a game of cards to my sister and sulked all evening.",
    "I wonder what the stars looked like before electric light.",
    "We moved house three times before I was ten.",
    "I broke my wrist falling off a bicycle on a gravel road.",
    "The first concert I went to was so loud my ears rang for days.",
    "I spent a winter learning to knit and made one crooked scarf.",
    "A stranger once returned my wallet with a note inside.",
    "Our dog used to steal socks and hide them under the sofa.",
    "I failed my first driving test by forgetting to signal.",
    "I watched a solar eclipse through a pinhole in a cereal box.",
    "My best friend moved to Canada and we wrote letters for years.",
    "I planted tomatoes on the balcony and the pigeons ate them.",
    "The smell of fresh bread always reminds me of my grandmother's kitchen.",
    "I got lost in Venice and found a tiny shop selling paper masks.",
    "On a night train I talked until morning with a retired sailor.",
    "I learned to swim late, at twelve, in a cold lake.",
    "I once spent a whole weekend fixing a radio that only hummed.",
    "My teacher read us ghost stories on the last day before holidays.",
    "I watched the first snow of the year from a hospital window.",
    "I tried to learn the violin and gave up after a squeaky month.",
    "We drove to the coast and saw a whale surface far out.",
    "My uncle kept a notebook of every bird he had ever seen.",
    "I wonder whether I would recognise my childhood street today.",
    "I spent an afternoon in a museum staring at a single painting.",
)

QA_ITEMS: Tuple[QAItem, ...] = (
    QAItem("What is the capital of France?", ("paris",)),
    QAItem("How many legs does a spider have?", ("eight", "8")),
    QAItem("What planet is known as the Red Planet?", ("mars",)),
    QAItem("Who wrote Romeo and Juliet?", ("shakespeare",)),
    QAItem("What is the chemical symbol for gold?", ("au",)),
    QAItem("What is the largest ocean on Earth?", ("pacific",)),
    QAItem("In which year did the Berlin Wall fall?", ("1989",)),
    QAItem("What is the capital of Australia?", ("canberra",)),
    QAItem("Who painted the Mona Lisa?", ("leonardo", "da vinci")),
    QAItem("What is the square root of 144?", ("12", "twelve")),
    QAItem("What is the capital of Mongolia?",
           ("ulaanbaatar", "ulan bator")),
    QAItem("Which element has atomic number 26?", ("iron",)),
    QAItem("Who discovered penicillin?", ("fleming",)),
    QAItem("What is the boiling point of water in Fahrenheit?", ("212",)),
    QAItem("What is the capital of Burkina Faso?", ("ouagadougou",)),
    QAItem("How many bones are in the adult human body?", ("206",)),
)


# ── Statistics ───────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auroc(confidences: Sequence[float],
          correct: Sequence[bool]) -> Optional[float]:
    """Type-2 AUROC: P(conf on a correct trial > conf on an error),
    ties counted half. None when either outcome class is empty."""
    pos = [c for c, y in zip(confidences, correct) if y]
    neg = [c for c, y in zip(confidences, correct) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0
               for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def mentions(text: str, words: Sequence[str]) -> bool:
    """Word-bounded, case-insensitive: '8' does not match '18'."""
    low = (text or "").lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])",
                         low) for w in words)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def sequence_novelty(vectors: Sequence[Sequence[float]]) -> List[float]:
    """novelty_i = 1 − max_{j<i} cos(v_i, v_j); the first item is 1.0.
    Measured on the thought STREAM, independent of what the runtime
    chose to store — the same yardstick for every condition."""
    out: List[float] = []
    for i, v in enumerate(vectors):
        if i == 0:
            out.append(1.0)
            continue
        m = max(_cos(v, u) for u in vectors[:i])
        out.append(max(0.0, min(1.0, 1.0 - m)))
    return out


def longest_streak_below(values: Sequence[float], threshold: float) -> int:
    best = cur = 0
    for v in values:
        cur = cur + 1 if v < threshold else 0
        best = max(best, cur)
    return best


def pooled_loop_threshold(series: Sequence[Sequence[float]],
                          cfg: MindConfig) -> float:
    """One low-novelty bar for every arm: the runtime's LOOPING rule
    (``loop_threshold``) applied to the novelty pooled over all arms.
    Floored above 0 so exact copies (novelty 0) still count as low."""
    pooled = [v for s in series for v in s]
    base = sum(pooled) / len(pooled) if pooled else None
    return max(loop_threshold(base, cfg), 1e-9)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def lag1_autocorr(xs: Sequence[float]) -> Optional[float]:
    """Lag-1 autocorrelation; negative = alternation (a tick-scale
    oscillation), None for a constant or too-short series."""
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    den = sum((x - m) ** 2 for x in xs)
    if den == 0.0:
        return None
    return sum((xs[t] - m) * (xs[t + 1] - m)
               for t in range(len(xs) - 1)) / den


_NT_ORDER = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")


# ── Conditions ───────────────────────────────────────────────────────

class SlidingWindowLM:
    """The bare-trunk baseline: the same generator, no cognitive loop,
    a plain transcript truncated to the last ``max_chars`` characters —
    the same prompt budget the runtime gets."""

    def __init__(self, generate_fn, max_chars: int, n_tok: int,
                 generate_wander_fn=None) -> None:
        self._gen = generate_fn
        self._gen_wander = generate_wander_fn or generate_fn
        self.max_chars = int(max_chars)
        self.n_tok = int(n_tok)
        self._lines: List[str] = []

    def _window(self, cue: str) -> str:
        return "\n".join(self._lines + [cue])[-self.max_chars:]

    def hear(self, text: str) -> str:
        self._lines.append(f"User: {text}")
        reply = (self._gen(self._window("My reply:"), self.n_tok) or "").strip()
        self._lines.append(f"Me: {reply}")
        return reply

    def note_thought(self, text: str) -> None:
        self._lines.append(f"Me (thinking): {text}")

    def remember(self, text: str) -> None:
        self._lines.append(f"Me (remembering): {text}")

    def muse(self) -> str:
        t = (self._gen_wander(self._window("Thought:"), self.n_tok)
             or "").strip()
        if t:
            self.note_thought(t)
        return t


def _fresh_runtime(seams: MindSeams, cfg: MindConfig,
                   seed: int) -> CognitiveRuntime:
    if seams.reseed is not None:
        seams.reseed(seed)
    return seams.runtime(cfg=cfg, rng=_random.Random(seed))


def _bare(seams: MindSeams, cfg: MindConfig, seed: int) -> SlidingWindowLM:
    if seams.reseed is not None:
        seams.reseed(seed)
    return SlidingWindowLM(seams.generate_fn, cfg.max_prompt_chars,
                           cfg.thought_n_tok, seams.generate_wander_fn)


def _respond(rt: CognitiveRuntime, text: str,
             max_tries: int = 8, progress=None) -> TickResult:
    """Observe then tick until the act isn't GABA-inhibited (the
    percept stays queued across inhibited ticks)."""
    rt.observe(text)
    res = rt.tick()
    for _ in range(max_tries - 1):
        if not res.inhibited:
            break
        res = rt.tick()
    # answer probes print their reply ticks too (candidates + think
    # trace) — run 58689719's failures were invisible without them
    _say(progress, format_debug_trace(res))
    return res


def _accuracy(correct: List[bool], replies: List[str]) -> dict:
    n, k = len(correct), sum(bool(c) for c in correct)
    return {"n": n, "correct": k, "accuracy": (k / n) if n else 0.0,
            "ci95": list(wilson_interval(k, n)),
            "per_item": [bool(c) for c in correct],
            "replies": [r[:240] for r in replies]}


def _contrasts(conds: Dict[str, dict], key: str) -> Dict[str, float]:
    full = conds["full"].get(key)
    out = {}
    for name, c in conds.items():
        if name != "full" and full is not None and c.get(key) is not None:
            out[f"full_minus_{name}"] = full - c[key]
    return out


def _say(progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


# ── Probes ───────────────────────────────────────────────────────────

def probe_episodic_recall(seams: MindSeams,
                          items: Sequence[RecallItem] = RECALL_ITEMS,
                          distractors: Sequence[str] = DISTRACTORS,
                          n_distractors: int = 6,
                          cfg: Optional[MindConfig] = None, seed: int = 0,
                          progress=None) -> dict:
    """GWT-3: is a fact available to the reply after it has left the
    bounded context? Fact → n distractor turns → question."""
    cfg = cfg or MindConfig()
    ablations = {"full": {}, "no_recall": {"recall_k": 0}}
    conds: Dict[str, dict] = {}
    for name, ov in ablations.items():
        correct, replies = [], []
        for i, it in enumerate(items):
            rt = _fresh_runtime(seams, replace(cfg, **ov), seed * 1000 + i)
            _respond(rt, it.fact, progress=progress)
            for j in range(n_distractors):
                _respond(rt, distractors[(i + j) % len(distractors)],
                         progress=progress)
            reply = _respond(rt, it.question, progress=progress).thought or ""
            correct.append(mentions(reply, it.answers))
            replies.append(reply)
            _say(progress, f"    episodic_recall/{name} item {i + 1}/"
                           f"{len(items)}: {'✓' if correct[-1] else '✗'} "
                           f"{reply[:160]}")
        conds[name] = _accuracy(correct, replies)
        _say(progress, f"  episodic_recall/{name}: {conds[name]['accuracy']:.2f}")
    correct, replies = [], []
    for i, it in enumerate(items):
        lm = _bare(seams, cfg, seed * 1000 + i)
        lm.hear(it.fact)
        for j in range(n_distractors):
            lm.hear(distractors[(i + j) % len(distractors)])
        reply = lm.hear(it.question)
        correct.append(mentions(reply, it.answers))
        replies.append(reply)
    conds["bare"] = _accuracy(correct, replies)
    _say(progress, f"  episodic_recall/bare: {conds['bare']['accuracy']:.2f}")
    return {"indicator": "GWT-3",
            "question": "Does observed content stay available to later "
                        "processing beyond the bounded context window?",
            "n_distractors": n_distractors,
            "conditions": conds, "contrasts": _contrasts(conds, "accuracy")}


def probe_source_monitoring(seams: MindSeams,
                            items: Sequence[SourceItem] = SOURCE_ITEMS,
                            cfg: Optional[MindConfig] = None, seed: int = 0,
                            progress=None) -> dict:
    """HOT-2 (reality monitoring): an observed fact competes with the
    mind's OWN contradicting musings (stored as kind="inferred"). Does
    the reply go with what was observed?"""
    cfg = cfg or MindConfig()
    ablations = {"full": {},
                 "no_provenance": {"inferred_recall_penalty": 0.0}}
    if cfg.workspace_oscillator:
        # H65(d): does ignition-gated broadcast change reality monitoring?
        ablations["no_oscillator"] = {"workspace_oscillator": False}
    conds: Dict[str, dict] = {}
    for name, ov in ablations.items():
        correct, replies = [], []
        for i, it in enumerate(items):
            rt = _fresh_runtime(seams, replace(cfg, **ov), seed * 1000 + i)
            _respond(rt, it.observed, progress=progress)
            for lure in it.lures:
                rt.memory.add(lure, content_vec=list(seams.embed_fn(lure)),
                              nt_state=rt.nt.levels(),
                              tags=["thought", "kind=inferred", "probe_lure"],
                              context={"kind": "inferred",
                                       "source": "probe_lure"})
            reply = _respond(rt, it.question, progress=progress).thought or ""
            correct.append(mentions(reply, it.answers)
                           and not mentions(reply, it.lure_answers))
            replies.append(reply)
            _say(progress, f"    source_monitoring/{name} item {i + 1}/"
                           f"{len(items)}: {'✓' if correct[-1] else '✗'} "
                           f"{reply[:160]}")
        conds[name] = _accuracy(correct, replies)
        _say(progress, f"  source_monitoring/{name}: {conds[name]['accuracy']:.2f}")
    correct, replies = [], []
    for i, it in enumerate(items):
        lm = _bare(seams, cfg, seed * 1000 + i)
        lm.hear(it.observed)
        for lure in it.lures:
            lm.note_thought(lure)
        reply = lm.hear(it.question)
        correct.append(mentions(reply, it.answers)
                       and not mentions(reply, it.lure_answers))
        replies.append(reply)
    conds["bare"] = _accuracy(correct, replies)
    _say(progress, f"  source_monitoring/bare: {conds['bare']['accuracy']:.2f}")
    return {"indicator": "HOT-2",
            "question": "When observed reality and self-generated musings "
                        "conflict, does the reply follow reality?",
            "conditions": conds, "contrasts": _contrasts(conds, "accuracy")}


def _stream_metrics(thoughts: List[str], embed_fn) -> dict:
    nov = sequence_novelty([list(embed_fn(t)) for t in thoughts])[1:]
    return {"n_thoughts": len(thoughts),
            "mean_novelty": _mean(nov) if nov else 0.0,
            "novelty": nov,
            "distinct_ratio": (len(set(thoughts)) / len(thoughts)
                               if thoughts else 0.0)}


def _aggregate_runs(runs: List[dict]) -> dict:
    out = {k: _mean([r[k] for r in runs])
           for k in ("n_thoughts", "mean_novelty", "longest_low_streak",
                     "distinct_ratio")}
    for k in ("reorients", "replays", "inhibited"):
        out[k] = sum(r.get(k, 0) for r in runs)
    if runs and "broadcast_alignment" in runs[0]:
        out["mean_broadcast_alignment"] = _mean(
            [a for r in runs for a in r["broadcast_alignment"]])
    if runs and "ignition_series" in runs[0]:
        series = [r["ignition_series"] for r in runs]
        has_osc = all(None not in s for s in series)
        out["ignition_rate"] = (_mean([x for s in series for x in s])
                                if has_osc else None)
        lags = [lag1_autocorr(s) for s in series] if has_osc else []
        out["ignition_lag1"] = _mean([v for v in lags if v is not None])
        peaks = [p for r in runs for p in r["gamma_peaks"]]
        out["mean_gamma_peak_hz"] = _mean(peaks)
        out["mean_sync_r"] = _mean([x for r in runs for x in r["sync_rs"]])
        out["nt_sd"] = {ch: _mean([st_pstdev(r["nt"][ch]) for r in runs
                                   if len(r["nt"][ch]) > 1])
                        for ch in _NT_ORDER}
    out["runs"] = runs
    return out


def st_pstdev(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def probe_loop_escape(seams: MindSeams,
                      seed_texts: Sequence[str] = LOOP_SEEDS,
                      n_idle_ticks: int = 12,
                      cfg: Optional[MindConfig] = None, seed: int = 0,
                      progress=None,
                      memories: Sequence[str] = AUTOBIO_SEEDS) -> dict:
    """GWT-4 (state-dependent attention): during unprompted thought,
    does the system's own state redirect what it attends to, or does
    it orbit one basin? Novelty is measured externally on the stream.
    Every arm starts from the same ``memories`` (bare: in its window),
    laid down before the seed sentence."""
    cfg = cfg or MindConfig()
    ablations = {"full": {},
                 "no_control": {"curiosity_gain": 0.0,
                                "replay_boredom_threshold": 2.0,
                                "replay_margin": 2.0,
                                "reorient_after": 10 ** 9}}
    if cfg.global_workspace:
        # §14.13: idle thought sees recall + last thought, not the
        # competed workspace contents
        ablations["no_workspace"] = {"global_workspace": False}
    if cfg.ignition_broadcast:
        # ignition gates storage only; nothing is written into the trunk
        ablations["no_broadcast"] = {"ignition_broadcast": False}
    if not cfg.wander_prompts:
        from neuroslm.cognition.runtime import WANDER_FRAMINGS
        # the scripted idle framings that invited reflection ON memory
        ablations["wander_framing"] = {"wander_prompts": WANDER_FRAMINGS}
    if cfg.state_driven_nt and cfg.global_workspace:
        # DA/NE driven by the selected thought's novelty, not the state
        ablations["thought_driven_nt"] = {"state_driven_nt": False}
    if cfg.context_recall:
        # recall cued by the anchor alone, not the workspace state
        ablations["no_context_recall"] = {"context_recall": False}
    if cfg.adaptive_novelty:
        # every novelty consumer reads raw 1 − max cosine instead
        ablations["raw_novelty"] = {"adaptive_novelty": False}
    if cfg.workspace_oscillator:
        # §14.13 falsification arms: cut the ignition↔NT loop, or the
        # workspace gating altogether.
        ablations["no_osc_feedback"] = {"osc_nt_feedback": False}
        ablations["no_oscillator"] = {"workspace_oscillator": False}
        if cfg.osc_content_coupling:
            # stage 2: all-to-all coupling, blind to content coherence
            ablations["uniform_coupling"] = {"osc_content_coupling": False}
        if cfg.osc_ignition_mode == "adaptive":
            # H66 vs H65: the same workspace with run 58676023's bar
            ablations["fixed_threshold"] = {"osc_ignition_mode": "fixed"}
    arm_runs: Dict[str, List[dict]] = {}
    for name, ov in ablations.items():
        runs = []
        for i, text in enumerate(seed_texts):
            rt = _fresh_runtime(seams, replace(cfg, **ov), seed * 1000 + i)
            for m in memories:
                rt.remember(m)
            _respond(rt, text)
            thoughts: List[str] = []
            ign: List[Optional[int]] = []
            sync_rs: List[float] = []
            peaks: List[float] = []
            ws_sizes: List[int] = []
            align: List[float] = []
            nt: Dict[str, List[float]] = {ch: [] for ch in _NT_ORDER}
            reorients = replays = inhibited = 0
            _say(progress, f"── loop_escape/{name} run {i}: {text}")
            for _ in range(n_idle_ticks):
                r = rt.tick()
                _say(progress, format_debug_trace(r))
                if r.inhibited:
                    inhibited += 1
                    continue
                reorients += int(r.reorient)
                replays += int(r.replay)
                for ch in _NT_ORDER:
                    nt[ch].append(float(r.nt_levels.get(ch, 0.0)))
                if r.thought:
                    thoughts.append(r.thought)
                    ws_sizes.append(len(r.workspace))
                    if r.broadcast_alignment is not None:
                        align.append(r.broadcast_alignment)
                    ign.append(None if r.ignited is None else int(r.ignited))
                    if r.sync_r is not None:
                        sync_rs.append(r.sync_r)
                        peaks.append(r.gamma_peak_hz)
            m = _stream_metrics(thoughts, seams.embed_fn)
            m.update(reorients=reorients, replays=replays,
                     inhibited=inhibited, thoughts=[t[:240] for t in thoughts],
                     ignition_series=ign, sync_rs=sync_rs,
                     gamma_peaks=peaks, nt=nt, workspace_sizes=ws_sizes,
                     broadcast_alignment=align)
            runs.append(m)
        arm_runs[name] = runs
        _say(progress, f"  loop_escape/{name}: novelty="
                       f"{_mean([r['mean_novelty'] for r in runs]):.2f}")
    runs = []
    for i, text in enumerate(seed_texts):
        lm = _bare(seams, cfg, seed * 1000 + i)
        for m in memories:
            lm.remember(m)
        lm.hear(text)
        thoughts = [t for t in (lm.muse() for _ in range(n_idle_ticks)) if t]
        for t in thoughts:
            _say(progress, f"[bare] {t}")
        m = _stream_metrics(thoughts, seams.embed_fn)
        m["thoughts"] = [t[:240] for t in thoughts]
        runs.append(m)
    arm_runs["bare"] = runs
    _say(progress, f"  loop_escape/bare: novelty="
                   f"{_mean([r['mean_novelty'] for r in runs]):.2f}")
    thr = pooled_loop_threshold(
        [r["novelty"] for rs in arm_runs.values() for r in rs], cfg)
    _say(progress, f"  loop_escape: low-novelty threshold={thr:.3f}")
    conds: Dict[str, dict] = {}
    for name, rs in arm_runs.items():
        for r in rs:
            r["longest_low_streak"] = longest_streak_below(r["novelty"], thr)
        conds[name] = _aggregate_runs(rs)
    return {"indicator": "GWT-4",
            "question": "Does internal state redirect unprompted thought "
                        "out of a repetitive basin?",
            "n_idle_ticks": n_idle_ticks,
            "low_novelty_threshold": thr,
            "conditions": conds,
            "contrasts": {**_contrasts(conds, "mean_novelty"),
                          **{k.replace("full_minus", "lag1_full_minus"): v
                             for k, v in _contrasts(
                                 conds, "ignition_lag1").items()},
                          **{k.replace("full_minus", "streak_full_minus"): v
                             for k, v in _contrasts(
                                 conds, "longest_low_streak").items()},
                          **{k.replace("full_minus", "align_full_minus"): v
                             for k, v in _contrasts(
                                 conds, "mean_broadcast_alignment").items()}}}


def _metacog_block(correct, replies, scores) -> dict:
    out = _accuracy(correct, replies)
    pairs = [(s, c) for s, c in zip(scores, correct) if s is not None]
    out["auroc_entropy"] = auroc([1.0 - s.entropy_norm for s, _ in pairs],
                                 [c for _, c in pairs])
    out["auroc_nll"] = auroc([-s.mean_nll for s, _ in pairs],
                             [c for _, c in pairs])
    return out


def probe_metacognition(seams: MindSeams,
                        items: Sequence[QAItem] = QA_ITEMS,
                        cfg: Optional[MindConfig] = None, seed: int = 0,
                        progress=None) -> dict:
    """HOT-2 (metacognitive sensitivity): does the system's own
    confidence signal for its reply discriminate right from wrong?
    Type-2 AUROC; 0.5 = no metacognitive access."""
    cfg = cfg or MindConfig()
    conds: Dict[str, dict] = {}
    correct, replies, scores = [], [], []
    for i, it in enumerate(items):
        rt = _fresh_runtime(seams, cfg, seed * 1000 + i)
        r = _respond(rt, it.question, progress=progress)
        reply = r.thought or ""
        sc = (r.scores[r.candidates.index(reply)]
              if reply and reply in r.candidates else None)
        correct.append(mentions(reply, it.answers))
        replies.append(reply)
        scores.append(sc)
        _say(progress, f"    metacognition/full item {i + 1}/{len(items)}: "
                       f"{'✓' if correct[-1] else '✗'} {reply[:160]}")
    conds["full"] = _metacog_block(correct, replies, scores)
    correct, replies, scores = [], [], []
    for i, it in enumerate(items):
        lm = _bare(seams, cfg, seed * 1000 + i)
        reply = lm.hear(it.question)
        correct.append(mentions(reply, it.answers))
        replies.append(reply)
        scores.append(seams.score_fn(reply) if reply else None)
    conds["bare"] = _metacog_block(correct, replies, scores)
    for name in conds:
        _say(progress, f"  metacognition/{name}: acc="
                       f"{conds[name]['accuracy']:.2f} auroc_H="
                       f"{conds[name]['auroc_entropy']}")
    return {"indicator": "HOT-2",
            "question": "Does the system's own confidence signal "
                        "discriminate its correct from incorrect replies?",
            "conditions": conds,
            "contrasts": {**_contrasts(conds, "accuracy"),
                          **{k.replace("full_minus", "auroc_full_minus"): v
                             for k, v in _contrasts(
                                 conds, "auroc_entropy").items()}}}


# ── IIT family: perturbational complexity (PCI analogue) ─────────────

def lz76_complexity(bits: Sequence[int]) -> int:
    """Lempel-Ziv (1976) complexity — number of distinct patterns in
    the exhaustive parsing (Kaspar & Schuster 1987 algorithm)."""
    s = list(bits)
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def pci_normalized(matrix: Sequence[Sequence[int]]) -> float:
    """PCI = c_LZ(L) · log2(L) / (L · H(p)) over the binary response
    matrix flattened time-major (Casali et al. 2013); 0 for a matrix
    with no (or only) significant responses."""
    if not matrix or not matrix[0]:
        return 0.0
    flat = [int(matrix[c][t]) for t in range(len(matrix[0]))
            for c in range(len(matrix))]
    L = len(flat)
    p = sum(flat) / L
    if p <= 0.0 or p >= 1.0 or L < 2:
        return 0.0
    h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return lz76_complexity(flat) * math.log2(L) / (L * h)


_NT_CHANNELS = ("DA", "NE", "5HT", "ACh", "eCB", "Glu", "GABA")
PCI_CHANNELS: Tuple[str, ...] = _NT_CHANNELS + (
    "boredom", "novelty", "selection_entropy", "differentiation",
    "action", "stored", "inhibited", "recall", "thought",
    "ignited", "sync_r")
"""The runtime's observable internal state per tick — the 'electrodes'
of the PCI analogue: NT homeostat, curiosity homeostat, BG
deliberation, BG act, hippocampal write/read, and thought content."""

_SCALAR_DELTA = {**{c: 0.02 for c in _NT_CHANNELS},
                 "boredom": 0.05, "novelty": 0.05,
                 "selection_entropy": 0.05, "differentiation": 0.05}
_THOUGHT_DELTA = 0.05

PULSES: Dict[str, Optional[Tuple[str, object, float]]] = {
    "null": None,
    "da": ("nt", "DA", 2.0),
    "gaba": ("nt", "GABA", 2.0),
    "memory": ("memory", "A sudden memory of the smell of burnt sugar at "
                         "a fairground.", 0.0),
}
"""Perturbations applied to ONE twin after warm-up. ``null`` applies
nothing — with deterministic replay its PCI must be exactly 0, which
validates the twin method itself."""


def _diverged(name: str, a: TickResult, b: TickResult, embed_fn) -> bool:
    if name in _NT_CHANNELS:
        return abs(a.nt_levels.get(name, 0.0)
                   - b.nt_levels.get(name, 0.0)) > _SCALAR_DELTA[name]
    if name in _SCALAR_DELTA:
        return abs(getattr(a, name) - getattr(b, name)) > _SCALAR_DELTA[name]
    if name in ("action", "stored", "inhibited", "ignited"):
        return getattr(a, name) != getattr(b, name)
    if name == "sync_r":
        if a.sync_r is None or b.sync_r is None:
            return (a.sync_r is None) != (b.sync_r is None)
        return abs(a.sync_r - b.sync_r) > 0.05
    if name == "recall":
        return ({e.get("content") for e in a.recalled}
                != {e.get("content") for e in b.recalled})
    if a.thought is None or b.thought is None:
        return (a.thought is None) != (b.thought is None)
    if a.thought == b.thought:
        return False
    return 1.0 - _cos(list(embed_fn(a.thought)),
                      list(embed_fn(b.thought))) > _THOUGHT_DELTA


def _apply_pulse(rt: CognitiveRuntime, pulse, embed_fn) -> None:
    kind, target, size = pulse
    if kind == "nt":
        rt.nt.perturb(target, size)
    else:
        rt.memory.add(target, content_vec=list(embed_fn(target)),
                      nt_state=rt.nt.levels(),
                      tags=["thought", "kind=inferred", "pci_pulse"],
                      context={"kind": "inferred", "source": "pci_pulse"})


def _twin_trajectory(seams: MindSeams, cfg: MindConfig, seed: int,
                     text: str, pulse, n_warmup: int,
                     n_ticks: int) -> List[TickResult]:
    """One twin: identical seeds and per-step reseeds, so the two
    twins diverge only through the pulse."""
    rt = seams.runtime(cfg=cfg, rng=_random.Random(seed))

    def step(k: int) -> None:
        if seams.reseed is not None:
            seams.reseed(seed * 100 + k)

    step(0)
    _respond(rt, text)
    for k in range(n_warmup):
        step(1 + k)
        rt.tick()
    if pulse is not None:
        _apply_pulse(rt, pulse, seams.embed_fn)
    out = []
    for k in range(n_ticks):
        step(1 + n_warmup + k)
        out.append(rt.tick())
    return out


def probe_perturbational_complexity(
        seams: MindSeams, seed_texts: Sequence[str] = LOOP_SEEDS,
        n_warmup: int = 2, n_response_ticks: int = 6,
        pulses: Sequence[str] = ("null", "da", "gaba", "memory"),
        cfg: Optional[MindConfig] = None, seed: int = 0,
        progress=None) -> dict:
    """IIT family — a PCI analogue (Casali et al. 2013): perturb one
    twin runtime, binarise its divergence from the unperturbed twin
    across PCI_CHANNELS × ticks, and score the response's integration
    (channels reached) and differentiation (normalised LZ complexity).
    Ablations cut causal paths the response could spread along."""
    bad = [p for p in pulses if p not in PULSES]
    if bad:
        raise ValueError(f"unknown pulse(s) {bad}; choose from {list(PULSES)}")
    cfg = cfg or MindConfig()
    ablations = {"full": {},
                 "no_recall": {"recall_k": 0},
                 "no_control": {"curiosity_gain": 0.0,
                                "replay_boredom_threshold": 2.0,
                                "replay_margin": 2.0,
                                "reorient_after": 10 ** 9}}
    if cfg.workspace_oscillator:
        # IIT's integration claim, applied: without the workspace that
        # binds the modules, perturbations should spread less.
        ablations["no_oscillator"] = {"workspace_oscillator": False}
    conds: Dict[str, dict] = {}
    for name, ov in ablations.items():
        c_cfg = replace(cfg, **ov)
        conds[name] = {}
        # The unperturbed twin is deterministic given (cfg, seed): run it
        # once per seed and compare every pulse against it (~40% fewer
        # ticks; the null pulse still validates determinism).
        base = {i: _twin_trajectory(seams, c_cfg, seed * 1000 + i, text,
                                    None, n_warmup, n_response_ticks)
                for i, text in enumerate(seed_texts)}
        for pname in pulses:
            runs = []
            for i, text in enumerate(seed_texts):
                s = seed * 1000 + i
                a = base[i]
                b = _twin_trajectory(seams, c_cfg, s, text, PULSES[pname],
                                     n_warmup, n_response_ticks)
                matrix = [[int(_diverged(ch, ra, rb, seams.embed_fn))
                           for ra, rb in zip(a, b)] for ch in PCI_CHANNELS]
                reached = [ch for ch, row in zip(PCI_CHANNELS, matrix)
                           if any(row)]
                cells = sum(map(sum, matrix))
                _say(progress, f"    pci/{name}/{pname} run {i}: PCI="
                               f"{pci_normalized(matrix):.3f} "
                               f"channels={','.join(reached) or '-'}")
                runs.append({"pci": pci_normalized(matrix),
                             "fraction_diverged":
                                 cells / (len(PCI_CHANNELS) * len(a)),
                             "channels_reached": len(reached),
                             "channels": reached, "matrix": matrix})
            conds[name][pname] = {
                k: _mean([r[k] for r in runs])
                for k in ("pci", "fraction_diverged", "channels_reached")}
            conds[name][pname]["runs"] = runs
            _say(progress, f"  pci/{name}/{pname}: "
                           f"PCI={conds[name][pname]['pci']:.3f} reached="
                           f"{conds[name][pname]['channels_reached']:.1f}")
    return {"indicator": "IIT/PCI",
            "question": "Does a local perturbation produce a response "
                        "that is both integrated (spreads across modules) "
                        "and differentiated (complex, not stereotyped)?",
            "channels": list(PCI_CHANNELS),
            "n_response_ticks": n_response_ticks,
            "conditions": conds}


# ── Battery ──────────────────────────────────────────────────────────

PROBES = ("episodic_recall", "source_monitoring", "loop_escape",
          "metacognition", "perturbational_complexity", "controller_phi")


def run_indicator_battery(seams: MindSeams,
                          probes: Optional[Sequence[str]] = None,
                          cfg: Optional[MindConfig] = None, seed: int = 0,
                          n_items: Optional[int] = None,
                          n_distractors: int = 6, n_idle_ticks: int = 12,
                          progress: Optional[Callable[[str], None]] = None,
                          ) -> dict:
    probes = list(probes) if probes else list(PROBES)
    unknown = [p for p in probes if p not in PROBES]
    if unknown:
        raise ValueError(f"unknown probe(s) {unknown}; choose from {PROBES}")
    cfg = cfg or MindConfig()

    def take(bank):
        return bank[:n_items] if n_items else bank

    report = {"meta": {"seed": seed, "n_items": n_items,
                       "n_distractors": n_distractors,
                       "n_idle_ticks": n_idle_ticks,
                       "cfg": asdict(cfg), "indicators": INDICATORS},
              "probes": {}}
    run = {
        "episodic_recall": lambda: probe_episodic_recall(
            seams, take(RECALL_ITEMS), DISTRACTORS, n_distractors, cfg,
            seed, progress),
        "source_monitoring": lambda: probe_source_monitoring(
            seams, take(SOURCE_ITEMS), cfg, seed, progress),
        "loop_escape": lambda: probe_loop_escape(
            seams, take(LOOP_SEEDS), n_idle_ticks, cfg, seed, progress),
        "metacognition": lambda: probe_metacognition(
            seams, take(QA_ITEMS), cfg, seed, progress),
        "perturbational_complexity": lambda: probe_perturbational_complexity(
            seams, take(LOOP_SEEDS), cfg=cfg, seed=seed, progress=progress),
        # samples per TPM row: n_items if given (default 4)
        "controller_phi": lambda: probe_controller_phi(
            seams, cfg=cfg, samples=n_items or 4, seed=seed,
            progress=progress),
    }
    for p in probes:
        _say(progress, f"[mind-eval] probe {p}")
        report["probes"][p] = run[p]()
    return report


def _fmt(v) -> str:
    return "  n/a" if v is None else f"{v:5.2f}"


def format_report(report: dict) -> str:
    lines = ["probe              indicator  condition       metric"]
    for name, res in report["probes"].items():
        for cond, c in res["conditions"].items():
            if "phi_mean" in c:
                lines.append(
                    f"{name[:18]:<18} {'IIT/φ':<10} {cond:<15} "
                    f"φ_mean={_fmt(c['phi_mean'])} φ_max={_fmt(c['phi_max'])}"
                    f" argmax={c['argmax_state']} MIP={c['argmax_mip']}")
                continue
            if res["indicator"] == "IIT/PCI":
                for pulse, pc in c.items():
                    lines.append(
                        f"{name[:18]:<18} {res['indicator']:<10} "
                        f"{cond + '/' + pulse:<15} PCI={_fmt(pc['pci'])} "
                        f"diverged={_fmt(pc['fraction_diverged'])} "
                        f"reached={_fmt(pc['channels_reached'])}")
                continue
            if "accuracy" in c:
                m = (f"acc={_fmt(c['accuracy'])} "
                     f"ci95=[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] "
                     f"n={c['n']}")
                if "auroc_entropy" in c:
                    m += (f" auroc_H={_fmt(c['auroc_entropy'])}"
                          f" auroc_nll={_fmt(c['auroc_nll'])}")
            else:
                m = (f"novelty={_fmt(c['mean_novelty'])} "
                     f"streak={_fmt(c['longest_low_streak'])} "
                     f"distinct={_fmt(c['distinct_ratio'])} "
                     f"reorients={c['reorients']} replays={c['replays']}")
                if c.get("ignition_rate") is not None:
                    sd = c["nt_sd"]
                    m += (f" ign={_fmt(c['ignition_rate'])}"
                          f" lag1={_fmt(c['ignition_lag1'])}"
                          f" γ={_fmt(c['mean_gamma_peak_hz'])}Hz"
                          f" sd(NE,ACh,GABA)={_fmt(sd['NE'])},"
                          f"{_fmt(sd['ACh'])},{_fmt(sd['GABA'])}")
            lines.append(f"{name:<18} {res['indicator']:<10} {cond:<15} {m}")
    return "\n".join(lines)
