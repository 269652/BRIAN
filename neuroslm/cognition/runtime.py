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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from neuroslm.memory.episodic import EpisodicMemory

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

    surprise_write_z: float = 0.35
    """Relative NLL distance from the running EMA required to write a
    thought to episodic memory (hippocampal novelty gate)."""

    surprise_ema_alpha: float = 0.2
    """EMA rate of the thought-NLL trace the novelty gate compares to."""

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
    hc = f"HC[recall={len(result.recalled)} write={'yes' if result.stored else 'no'}]"
    return f"Φ={result.phi_proxy:.2f} {nt_str} {bg} {hc}"


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
    lines = [f"[tick {result.tick_n}] {mode} action={result.action}"]

    nt = result.nt_levels or {}
    lines.append("  NT[" + " ".join(
        f"{k}={float(nt.get(k, 0.0)):.2f}" for k in _NT_ORDER) + "]")

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
    """

    def __init__(self,
                 generate_fn: GenerateFn,
                 score_fn: ScoreFn,
                 embed_fn: EmbedFn,
                 nt: Optional[Any] = None,
                 memory: Optional[EpisodicMemory] = None,
                 cfg: Optional[MindConfig] = None,
                 rng: Optional[_random.Random] = None) -> None:
        if nt is None:
            from neuroslm.emergent.driven_nt import DrivenNTSystem
            nt = DrivenNTSystem()
        self._gen = generate_fn
        self._score = score_fn
        self._embed = embed_fn
        self.nt = nt
        self.memory = memory if memory is not None else EpisodicMemory(512)
        self.cfg = cfg or MindConfig()
        self.rng = rng or _random.Random(0)
        self._sensory: Deque[str] = deque(maxlen=32)
        self._last_thought: Optional[str] = None
        # Running NLL trace for the hippocampal novelty gate. None until
        # the first thought seeds it (bootstrap: first thought is novel).
        self._nll_ema: Optional[float] = None
        self._tick_n: int = 0
        self._wander_idx: int = 0

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
        self.memory.add(text,
                        content_vec=list(self._embed(text)),
                        nt_state=self.nt.levels(),
                        tags=[source, "percept", "kind=observed"],
                        context={"kind": "observed", "source": source})
        self._sensory.append(text)
        self.nt.step_full(activation=1.0)

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
        anchor = (sensory[-1] if sensory
                  else self._last_thought or self.cfg.persona)

        # RECALL: hippocampal similarity read.
        recalled = self.memory.retrieve(
            list(self._embed(anchor)), k=self.cfg.recall_k)

        # THINK: K candidates from persona + recall + context. The
        # trigger — what this tick is actually ABOUT — is the real
        # user text for a respond, or the specific wander prompt
        # chosen for an idle tick; captured here (not hidden inside
        # _compose_prompt) so the episode we may store below can
        # record it (Layer 2 — Context: "what happened around it").
        trigger_text = sensory[-1] if sensory else self._next_wander_prompt()
        prompt = self._compose_prompt(sensory, recalled, trigger_text)
        candidates: List[str] = []
        for _ in range(max(1, self.cfg.n_candidates)):
            out = (self._gen(prompt, self.cfg.thought_n_tok) or "").strip()
            if out:
                candidates.append(out)
        if not candidates:
            self.nt.step_full()
            return TickResult(thought=None, recalled=recalled,
                              nt_levels=levels, tick_n=tick_n,
                              prior_thought=prior_thought,
                              wandering=wandering,
                              action="respond" if not wandering else "think")

        # GATE: DA-tempered softmax over −NLL (basal-ganglia selection).
        scores = [self._score(c) for c in candidates]
        idx, selection_entropy = self._select(scores, levels)
        thought, sc = candidates[idx], scores[idx]
        differentiation = _population_stdev([s.mean_nll for s in scores])

        # ACTION: the basal ganglia's actual act this tick — exactly
        # one of respond/speak/think. External input always surfaces
        # (respond); otherwise the SAME novelty gate that decided
        # whether to remember the thought also decides whether to
        # voice it (speak) or keep it internal (think) — one signal,
        # two consequences, not a second invented gate.
        stored = self._novelty_gate(sc.mean_nll)
        action = "respond" if not wandering else ("speak" if stored else "think")

        # STORE: surprise-gated episodic write — as a full episode,
        # not bare text. Layer 1 (event) is `thought` itself; the
        # rest lives in `context`, reusing EpisodicMemory's existing
        # (previously-unused) slot rather than inventing a parallel
        # structure. `associations` names which prior episodes fed
        # RECALL — real IIT-flavored state (confidence/phi/selection
        # entropy/differentiation), never a fabricated mood label.
        if stored:
            self.memory.add(
                thought,
                content_vec=list(self._embed(thought)),
                nt_state=levels,
                tags=["thought", f"action={action}",
                     "wandering" if wandering else "responding"],
                context={
                    "kind": "inferred",
                    "action": action,
                    "wandering": wandering,
                    "trigger": trigger_text,
                    "tick_n": tick_n,
                    "associations": [e.get("content", "") for e in recalled],
                    "confidence": max(0.0, min(1.0, 1.0 - sc.entropy_norm)),
                    "phi_proxy": max(0.0, min(1.0, sc.entropy_norm)),
                    "selection_entropy": selection_entropy,
                    "differentiation": differentiation,
                })

        # DRIVE: the tick's signals advance the NT dynamics. The
        # selected thought's NLL is the loss/surprise driver (an
        # unusually easy thought reads as reward); its inverse
        # normalised entropy is the activation proxy.
        self.nt.step_full(loss=sc.mean_nll,
                          activation=1.0 - sc.entropy_norm)

        self._last_thought = thought
        return TickResult(thought=thought, candidates=candidates,
                          scores=scores, recalled=recalled,
                          stored=stored, inhibited=False,
                          nt_levels=levels,
                          phi_proxy=max(0.0, min(1.0, sc.entropy_norm)),
                          tick_n=tick_n, prior_thought=prior_thought,
                          wandering=wandering, action=action,
                          selection_entropy=selection_entropy,
                          differentiation=differentiation)

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
                levels: Dict[str, float]) -> Tuple[int, float]:
        """Basal-ganglia selection. Returns ``(chosen_index,
        selection_entropy)`` — the latter an IIT-flavored decisiveness
        proxy: normalised entropy of the softmax(-NLL/T) choice
        distribution (0 = one option clearly excluded the rest, 1 =
        arbitrary among near-equal options)."""
        if len(scores) == 1:
            return 0, 0.0
        # DrivenNTSystem exposes `baselines` as a property while
        # `levels()` is a method — accept both shapes so duck-typed
        # NT systems keep working.
        baselines = self.nt.baselines
        if callable(baselines):
            baselines = baselines()
        T = selection_temperature(levels.get("DA", 0.0),
                                  baselines.get("DA", 0.15), self.cfg)
        # softmax(−NLL / T), computed stably.
        utils = [-s.mean_nll / max(T, 1e-6) for s in scores]
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

    def _novelty_gate(self, nll: float) -> bool:
        """Hippocampal write gate: store when the thought's NLL sits
        far (relatively) from the running EMA. Bootstrap: the first
        thought seeds the EMA and is stored."""
        if self._nll_ema is None:
            self._nll_ema = float(nll)
            return True
        z = abs(float(nll) - self._nll_ema) / max(abs(self._nll_ema), 1e-6)
        a = self.cfg.surprise_ema_alpha
        self._nll_ema = (1.0 - a) * self._nll_ema + a * float(nll)
        return z >= self.cfg.surprise_write_z


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

    return CognitiveRuntime(generate_fn=generate_fn, score_fn=score_fn,
                            embed_fn=embed_fn, cfg=cfg)


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

        @torch.no_grad()
        def generate_fn(prompt: str, max_new_tokens: int) -> str:
            if has_chat_template:
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
                # Repetition controls: always on. Cheap, model-agnostic,
                # and the second half of the fix for the observed loop.
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
            if not has_chat_template:
                # Base-model safety net: stop before it hallucinates
                # the next USER turn instead of burning the whole
                # token budget on a degenerate continuation. Instruct
                # models don't need this — their chat template's own
                # turn-end token already stops generation cleanly.
                gen_kwargs["stop_strings"] = ["\nUSER:", "\nUser:", "\n\n"]
                gen_kwargs["tokenizer"] = tokenizer
            out = model.generate(**gen_kwargs)
            new_ids = out[0, x.shape[1]:].tolist()
            return tokenizer.decode(new_ids, skip_special_tokens=True)
    else:
        from neuroslm.chat_daemon import _build_generate_fn_from_harness
        from types import SimpleNamespace
        generate_fn = _build_generate_fn_from_harness(
            SimpleNamespace(language_model=wrapper), tokenizer,
            device=device, temperature=temperature, top_k=top_k)

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

    return CognitiveRuntime(generate_fn=generate_fn, score_fn=score_fn,
                            embed_fn=embed_fn, cfg=cfg)
