# -*- coding: utf-8 -*-
"""Knowledge extraction over episodic memory — generalized action
classification + temporal association mining (architecture.md §14.8).

Two real, established techniques, chosen deliberately over inventing
ad-hoc heuristics:

1. :func:`classify_action` — rule-based dialogue-act classification.
   Lexicon/pattern matching against a fixed taxonomy is standard
   practice for dialogue-act tagging in the absence of labeled
   training data (in the tradition of DAMSL-style schemes — Core &
   Allen, "Coding Dialogs with the DAMSL Annotation Scheme", 1997).
   "You suck" is still stored verbatim (the literal event) — this
   adds the GENERALIZED action class ("insult") alongside it, so the
   episode carries both.

2. :func:`mine_temporal_associations` — Apriori-derived sequential
   association-rule mining (Agrawal & Srikant, "Fast Algorithms for
   Mining Association Rules", VLDB 1994; sequential extension in
   "Mining Sequential Patterns", ICDE 1995) over an ORDERED sequence
   of classified actions. Reports support / confidence / lift.

   This is deliberately named "association", never "causal": passive
   observation alone cannot establish causation (Pearl's causal
   hierarchy places association below intervention and counterfactual
   reasoning — no amount of correlational mining over observed
   sequences crosses that gap without an actual intervention). Every
   :class:`AssociationRule` also reports whether its evidence is
   externally GROUNDED (at least one supporting instance touched a
   ``kind="observed"`` episode) or SELF_REFERENTIAL_ONLY (every
   supporting instance was the mind's own ``kind="inferred"`` output
   talking to itself) — the exact contamination risk flagged in the
   2026-08-12 log analysis, now visible instead of silently trusted.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


# ── 1. Generalized action classification ──────────────────────────────

ACTION_TAXONOMY: Dict[str, Tuple[str, ...]] = {
    "insult": (
        r"\byou\s+suck\b",
        r"\byou'?re\s+(an?\s+)?(idiot|stupid|dumb|moron|loser|fool|worthless|pathetic|useless)\b",
        r"\byou\s+are\s+(an?\s+)?(idiot|stupid|dumb|moron|loser|fool|worthless|pathetic|useless)\b",
        r"\bshut\s+up\b",
        r"\bi\s+hate\s+you\b",
        r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\bpathetic\b",
    ),
    "compliment": (
        r"\byou'?re\s+(amazing|great|awesome|brilliant|wonderful|fantastic)\b",
        r"\byou\s+are\s+(amazing|great|awesome|brilliant|wonderful|fantastic)\b",
        r"\bwell\s+done\b", r"\bgood\s+job\b", r"\bnicely\s+done\b",
        r"\bimpressive\b",
    ),
    "disagreement": (
        r"^no\b", r"\bno,",
        r"\bdisagree\b",
        r"\bthat'?s\s+(wrong|incorrect|false|not\s+right)\b",
        r"\bi\s+don'?t\s+think\s+so\b",
    ),
    "agreement": (
        r"^yes\b", r"\byes,", r"\byeah\b",
        r"\bagreed?\b", r"\bexactly\b",
        r"\bthat'?s\s+(right|correct|true)\b",
    ),
    "apology": (
        r"\bsorry\b", r"\bapologi[sz]e\b", r"\bmy\s+(bad|mistake|fault)\b",
        r"\bforgive\s+me\b",
    ),
    "gratitude": (
        r"\bthanks?\b", r"\bthank\s+you\b", r"\bappreciate\s+(it|that|you)\b",
        r"\bmuch\s+obliged\b",
    ),
    "request": (
        r"\bplease\b", r"\bcould\s+you\b", r"\bcan\s+you\b",
        r"\bwould\s+you\b", r"\bi\s+need\b",
    ),
    "greeting": (
        r"\bhello\b", r"\bhi\b", r"\bhey\b",
        r"\bgood\s+(morning|afternoon|evening)\b", r"\bgreetings\b",
    ),
    "farewell": (
        r"\bgoodbye\b", r"\bbye\b", r"\bsee\s+you\b", r"\btake\s+care\b",
        r"\bfarewell\b",
    ),
    "question": (
        r"\?\s*$",
        r"^(what|why|how|when|where|who|which|is|are|do|does|did|can|could|would|will)\b",
    ),
}

# Multiple classes can match the same utterance ("No, you're an idiot"
# is both disagreement and insult) — priority resolves which becomes
# `.primary`. Most socially salient / specific first; `question`'s
# broad ends-with-"?" pattern sits last so more specific classes win.
_PRIORITY: Tuple[str, ...] = (
    "insult", "compliment", "disagreement", "agreement", "apology",
    "gratitude", "request", "greeting", "farewell", "question",
)

_STATEMENT = "statement"  # fallback — not a lexicon entry, nothing matched


@dataclass
class ActionClassification:
    primary: str
    candidates: List[str]


def classify_action(text: str) -> ActionClassification:
    """Classify one utterance's generalized dialogue act.

    Checks every class in :data:`ACTION_TAXONOMY`; returns ALL matches
    as ``candidates`` (an utterance can genuinely be more than one
    thing) and the highest-priority match as ``primary``. Falls back
    to ``"statement"`` when nothing matches — a plain declarative with
    no detected act, not an error.
    """
    t = (text or "").strip().lower()
    if not t:
        return ActionClassification(primary=_STATEMENT, candidates=[_STATEMENT])
    matched = [cls for cls in _PRIORITY
              if any(re.search(pat, t) for pat in ACTION_TAXONOMY[cls])]
    if not matched:
        return ActionClassification(primary=_STATEMENT, candidates=[_STATEMENT])
    return ActionClassification(primary=matched[0], candidates=matched)


_CLASSIFY_LABELS: Tuple[str, ...] = tuple(_PRIORITY) + (_STATEMENT,)


def _build_classify_prompt(text: str) -> str:
    labels = ", ".join(_CLASSIFY_LABELS)
    return (
        "Classify the following message's action into exactly ONE of "
        f"these categories: {labels}.\n"
        f"Message: \"{text}\"\n"
        "Category:"
    )


def classify_action_via_generation(
    text: str,
    generate_fn: Callable[[str, int], str],
    max_new_tokens: int = 8,
) -> ActionClassification:
    """Classify using the MIND'S OWN trunk/expert — a real zero-shot
    generation-based classification, not a second model and not the
    lexicon. ``generate_fn`` is the SAME ``(prompt, max_new_tokens) ->
    str`` seam THINK already uses (see ``build_runtime_from_hf_lm``,
    which reuses its own already-built ``generate_fn`` closure for
    this — one model, two jobs, no parallel generation path).

    Prompts the model with the fixed taxonomy and parses its reply
    (case-insensitive substring match against every known label —
    robust to "This is clearly an insult." as well as a bare
    "insult"). Falls back to the deterministic lexicon
    (:func:`classify_action`) whenever the model's reply doesn't
    parse to a known category, or the call itself fails — never
    silently returns garbage or raises out of a STORE step.
    """
    try:
        reply = (generate_fn(_build_classify_prompt(text), max_new_tokens)
                 or "").strip().lower()
    except Exception as exc:  # model unavailable, OOM, etc.
        print(f"[patterns] classify_action_via_generation fell back to "
              f"the lexicon: {type(exc).__name__}: {exc}", file=sys.stderr)
        return classify_action(text)

    for label in _CLASSIFY_LABELS:
        if label in reply:
            return ActionClassification(primary=label, candidates=[label])
    return classify_action(text)


# ── 2. Temporal association mining ─────────────────────────────────────

@dataclass
class AssociationRule:
    """One mined (antecedent -> consequent) pattern. Statistical
    association ONLY — see module docstring. Never treat as a
    validated causal claim; ``grounded``/``self_referential_only``
    report how much (if any) external reality backs the evidence."""
    antecedent: str
    consequent: str
    support: float
    confidence: float
    lift: float
    evidence_count: int
    antecedent_count: int
    grounded: bool
    self_referential_only: bool


def mine_temporal_associations(
    episodes: Sequence[dict],
    window: int = 1,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
) -> List[AssociationRule]:
    """Mine (A -> B) sequential association rules from a CHRONOLOGICAL
    list of ``{"action_class": str, "kind": "observed"|"inferred"}``
    dicts (as stored per-episode by the cognition runtime).

    For each occurrence of action A at position i (with at least one
    possible successor, i.e. i < n-1), A "supports" A->B when B occurs
    anywhere in the window ``episodes[i+1 : i+1+window]``. Standard
    Apriori-family metrics, computed over the whole sequence:

        confidence(A->B) = P(B within window | A) = evidence / antecedent_count
        support(A->B)    = evidence / total_valid_antecedent_positions
        lift(A->B)       = confidence(A->B) / P(B)   [marginal frequency of B]

    ``grounded`` is True when at least one supporting (A, B) instance
    involved a ``kind="observed"`` episode on either side — i.e. the
    pattern touched external reality somewhere, not pure self-talk.
    ``self_referential_only`` is the complement: every supporting
    instance was inferred->inferred (the mind's own output chained to
    itself with no outside grounding at all) — flag this, don't hide it.

    Results are sorted by confidence, descending. Rules below
    ``min_support``/``min_confidence`` are omitted (the standard
    Apriori pruning thresholds).
    """
    n = len(episodes)
    if n < 2:
        return []

    classes = [e.get("action_class", _STATEMENT) for e in episodes]
    kinds = [e.get("kind", "inferred") for e in episodes]

    # Marginal frequency of every class (support of the single itemset).
    marginal: Dict[str, float] = {}
    for c in classes:
        marginal[c] = marginal.get(c, 0.0) + 1.0
    for c in marginal:
        marginal[c] /= n

    valid_positions = [i for i in range(n) if i < n - 1]
    total_valid = len(valid_positions)

    antecedent_counts: Dict[str, int] = {}
    # (A, B) -> [evidence_count, grounded_flag_seen, non_grounded_seen]
    pair_stats: Dict[Tuple[str, str], List] = {}

    for i in valid_positions:
        a = classes[i]
        antecedent_counts[a] = antecedent_counts.get(a, 0) + 1
        window_end = min(i + 1 + window, n)
        seen_in_window = set()
        for j in range(i + 1, window_end):
            b = classes[j]
            if b in seen_in_window:
                continue  # count each (A,B) at most once per A-occurrence
            seen_in_window.add(b)
            key = (a, b)
            grounded_here = (kinds[i] == "observed") or (kinds[j] == "observed")
            if key not in pair_stats:
                pair_stats[key] = [0, False]  # [evidence_count, any_grounded]
            pair_stats[key][0] += 1
            if grounded_here:
                pair_stats[key][1] = True

    rules: List[AssociationRule] = []
    for (a, b), (evidence, any_grounded) in pair_stats.items():
        a_count = antecedent_counts.get(a, 0)
        if a_count == 0 or evidence == 0:
            continue
        confidence = evidence / a_count
        support = evidence / total_valid if total_valid else 0.0
        p_b = marginal.get(b, 0.0)
        lift = (confidence / p_b) if p_b > 0 else 0.0
        if support < min_support or confidence < min_confidence:
            continue
        rules.append(AssociationRule(
            antecedent=a, consequent=b,
            support=support, confidence=confidence, lift=lift,
            evidence_count=evidence, antecedent_count=a_count,
            grounded=any_grounded, self_referential_only=not any_grounded,
        ))

    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules
