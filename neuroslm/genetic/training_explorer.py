# -*- coding: utf-8 -*-
"""Exploration wired into training — search every N steps, keep-if-better, ledger.

Every ``explore_every`` steps the explorer runs a short NGL modulation search on
the *current* model, A/B-tests the winner (baseline = identity modulation), and
keeps it only if the metric improves — installing it into the running model
before training resumes. Every attempt is recorded to a persistent
``SearchLedger`` keyed by semantic signature, and prior-run **duds are skipped**
so a fresh run doesn't re-search the same space.

The explorer is model-agnostic: it takes a ``score_fn(program) -> metric`` (lower
is better). ``run_training_with_exploration`` wires it to a tiny CPU LM as the
runnable miniature; the same explorer attaches to the real trunk by supplying a
``score_fn`` that applies a modulation to the trunk's residual stream and returns
a validation metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import hashlib
import math

import numpy as np

from neuroslm.genetic.language import Program


def _stable_seed(run_id: str, step: int) -> int:
    """Process-independent RNG seed — reproducible across runs.

    Python's builtin ``hash`` is salted per process (``PYTHONHASHSEED``), so
    seeding the search with it made runs non-reproducible. A blake2b digest of
    ``(run_id, step)`` is deterministic, which is what a *measurement* needs.
    """
    h = hashlib.blake2b(f"{run_id}:{step}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2**32)


def probe_hidden_modulation(hidden, head_fn, targets, *, ledger,
                            store=None, config=None, step: int = 0,
                            run_id: str = "probe"):
    """Read-only probe: does a residual modulation of the trunk's final hidden
    lower next-token CE on this batch? Searches one, records/persists the winner.

    This is the *safe* way to gather first discovery data on a real trunk: it
    never touches the training forward or weights — it re-projects the LM head on
    a modulated copy of the (detached) final hidden state, so it cannot perturb
    the run. ``hidden`` is ``(B, T, D)``; ``head_fn`` maps hidden→logits;
    ``targets`` is ``(B, T)`` next-token ids. Returns a summary dict with the
    baseline CE, the best CE found, the Δ, and the persisted winner (if improved).
    """
    import torch
    import torch.nn.functional as F
    from neuroslm.genetic.neuro_evolve import _make_modulator

    def ce(logits) -> float:
        return float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))

    cfg = config or ExploreConfig()
    with torch.no_grad():                 # read-only — never build an autograd graph
        baseline = ce(head_fn(hidden))
    explorer = TrainingExplorer(ledger, cfg, run_id=run_id)

    def score_fn(prog: Program) -> float:
        try:
            with torch.no_grad():
                return ce(head_fn(_make_modulator(prog)(hidden)))
        except Exception:
            return baseline * 10.0        # ill-typed candidate → strongly dispreferred

    res = explorer.explore(step, score_fn)
    improved = res.improved and math.isfinite(res.best_score)
    delta = baseline - res.best_score
    saved = None
    if improved and store is not None:
        from neuroslm.genetic.modulation_store import ModulationRecord
        winner = _minimal_equivalent(res.best_program, seed=step)
        saved = f"{run_id.replace('-', '_')}_step{step}"
        store.save(ModulationRecord(
            name=saved, program=winner,
            metrics={"step": step, "baseline_ce": round(baseline, 4),
                     "best_ce": round(res.best_score, 4),
                     "delta_ce": round(delta, 4)}))
    return {"step": step, "baseline_ce": baseline, "best_ce": res.best_score,
            "delta_ce": delta, "improved": improved, "saved": saved,
            "evaluated": res.n_evaluated, "skipped_duds": res.n_skipped_duds}


def _minimal_equivalent(program: Program, seed: int = 0):
    """Verified-minimal form of a discovered winner — the mechanism, not the salad.

    The GA emits bloated programs (a real discovery was 12 instructions, 8 dead);
    DCE + the probe-verified peephole superoptimizer strip that to the essential
    computation so the persisted ``modulations/*.neuro`` *is* the mechanism. Falls
    back to the raw program if simplification isn't probe-equivalent.
    """
    from neuroslm.genetic.simplify import simplify, programs_equivalent
    try:
        minimal = simplify(program, n_probes=12, seed=seed)
        if programs_equivalent(program, minimal, n_probes=12, seed=seed + 1):
            return minimal
    except Exception:
        pass
    return program
from neuroslm.genetic.evolve import Objective, auto_evolve
from neuroslm.genetic.ledger import SearchLedger
from neuroslm.genetic.neuro_evolve import identity_modulation


@dataclass
class ExploreConfig:
    explore_every: int = 500
    pop_size: int = 16
    generations: int = 8
    length: int = 5
    n_scalar: int = 4
    n_tensor: int = 8
    tol: float = 1e-4          # metric must improve by more than this to "keep"
    dud_penalty: float = 1e9
    normalize: bool = True     # canonicalize candidates before the ledger sees them
    wellformed_penalty: float = 0.05    # fitness bump per read of an undefined register
    inputs: tuple = ("t0",)             # registers pre-bound by the harness (h)


@dataclass
class ExploreResult:
    step: int
    improved: bool
    baseline: float
    best_score: float
    best_program: Program
    n_evaluated: int
    n_skipped_duds: int


class TrainingExplorer:
    def __init__(self, ledger: SearchLedger, config: ExploreConfig = None,
                 run_id: str = "run"):
        self.ledger = ledger
        self.cfg = config or ExploreConfig()
        self.run_id = run_id

    def maybe_explore(self, step: int, score_fn: Callable[[Program], float],
                      progress: Optional[Callable[[str], None]] = None) -> Optional[ExploreResult]:
        if step > 0 and step % self.cfg.explore_every == 0:
            return self.explore(step, score_fn, progress=progress)
        return None

    def _canon(self, prog: Program) -> Program:
        """Canonical (normalized) form of a candidate, if normalization is on.

        Collapsing syntactic variants to one normal form here means the ledger's
        dud-skip and signature dedup operate on *semantics*, not syntax — so the
        search never re-explores a rewrite of something already seen.
        """
        if not self.cfg.normalize:
            return prog
        try:
            from neuroslm.genetic.normalize import canonical_form
            return canonical_form(prog, n_probes=6, seed=0)
        except Exception:
            return prog

    def _canonical_sig(self, prog: Program) -> str:
        return self.ledger.signature(self._canon(prog))

    def _fitness_penalty(self, prog: Program) -> float:
        """Multiplicative fitness penalty for ill-formed programs (undefined reads).

        Applied to the search *objective* only — the reported ppl stays the true
        value — so the GA prefers clean mechanics without distorting measurements.
        """
        from neuroslm.genetic.evolve import undefined_reads
        return 1.0 + self.cfg.wellformed_penalty * undefined_reads(prog, self.cfg.inputs)

    def explore(self, step: int, score_fn: Callable[[Program], float],
                progress: Optional[Callable[[str], None]] = None) -> ExploreResult:
        cfg = self.cfg
        rng = np.random.default_rng(_stable_seed(self.run_id, step))
        baseline = float(score_fn(identity_modulation()))

        evaluated: dict = {}     # signature -> (program, score)
        skipped = [0]

        if progress:
            progress(f"[explore @ step {step}] searching  baseline_ppl={baseline:.2f}  "
                     f"(pop={cfg.pop_size}, gens={cfg.generations})")

        def _on_gen(g: int, total: int, best_obj, primary_obj) -> None:
            if not progress:
                return
            # throttle: first, last, and ~10 evenly-spaced generations
            if g != 0 and g != total and g % max(1, total // 10) != 0:
                return
            try:
                best = -float(primary_obj.values[0])
            except Exception:
                best = float("nan")
            progress(f"[explore @ step {step}] gen {g}/{total}  best_ppl={best:.2f}  "
                     f"evaluated={len(evaluated)} skipped_duds={skipped[0]}")

        # the true clean-best RAW program, floored at identity/baseline. We score
        # the *raw* candidate (what would actually install), never the canonical
        # form — canonicalization is only for dedup and can misbehave as a live
        # modulator. Identity is a seed, so best_score ≤ baseline always.
        best_raw = [baseline, identity_modulation()]

        def evaluate(prog: Program) -> Objective:
            # canon is used only for dedup / ledger — the winner is a raw program
            canon = self._canon(prog)
            if self.ledger.is_dud(canon):
                skipped[0] += 1
                return Objective((-cfg.dud_penalty,))
            sig = self.ledger.signature(canon)
            if sig in evaluated:
                s = evaluated[sig][1]
            else:
                s = float(score_fn(prog))          # score the RAW candidate
                evaluated[sig] = (canon, s)         # store CANON (ledger/dedup key)
                if math.isfinite(s) and s < best_raw[0]:
                    best_raw[0], best_raw[1] = s, prog
            # penalize ill-formed programs (undefined reads) in the *fitness* only
            return Objective((-(s * self._fitness_penalty(prog)),))

        result = auto_evolve(
            evaluate, rng,
            pop_size=cfg.pop_size, generations=cfg.generations,
            length=cfg.length, n_scalar=cfg.n_scalar, n_tensor=cfg.n_tensor,
            seeds=[identity_modulation()],
            elite_frac=0.3, crossover_rate=0.5,
            on_generation=_on_gen,
        )
        best_score, best_prog = best_raw[0], best_raw[1]
        improved = best_score < baseline - cfg.tol

        # record every distinct pattern we searched (so future runs skip duds)…
        for sig, (prog, s) in evaluated.items():
            self.ledger.record(prog, outcome="searched", delta=s - baseline,
                               step=step, run_id=self.run_id)
        # …then the verdict on the winner (canon key, so next run's dud-gate
        # recognises it — consistent with the is_dud(canon) check above)
        self.ledger.record(self._canon(best_prog),
                           outcome="kept" if improved else "rejected",
                           delta=best_score - baseline, metric_before=baseline,
                           metric_after=best_score, step=step, run_id=self.run_id)

        return ExploreResult(
            step=step, improved=improved, baseline=baseline, best_score=best_score,
            best_program=best_prog, n_evaluated=len(evaluated),
            n_skipped_duds=skipped[0],
        )


# ---------------------------------------------------------------------------
# Runnable miniature: a tiny-LM training loop with exploration wired in.
# ---------------------------------------------------------------------------
def run_training_with_exploration(*, total_steps: int = 2000, explore_every: int = 500,
                                  seed: int = 0, ledger: SearchLedger = None,
                                  pop_size: int = 12, generations: int = 4,
                                  inner_steps: int = 20,
                                  progress: Callable[[str], None] = None,
                                  store=None, wellformed_penalty: float = 0.05) -> dict:
    """Train a tiny CPU LM; every ``explore_every`` steps search + keep-if-better.

    This is the miniature of "wire exploration into training": the residual-stream
    modulation is A/B-tested on the live model and installed only if it lowers
    validation perplexity. The same loop attaches to the real trunk on GPU.

    ``store`` (a ``ModulationStore``): when given, each *installed* winner is
    persisted as ``modulations/<run>-step<N>.neuro`` with its measured Δ, and an
    install that is later reverted (destabilized training) is dropped again — so
    the store ends with only the durable survivors, each carrying an honest,
    healthy-baseline Δ.
    """
    import torch
    import torch.nn as nn
    from neuroslm.genetic.neuro_evolve import _TinyLM, _markov_corpus, _make_modulator

    ledger = ledger or SearchLedger(":memory:")
    train, val, vocab = _markov_corpus(seed)
    torch.manual_seed(seed + 3)
    model = _TinyLM(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    current_mod = identity_modulation()

    def val_ppl(prog: Program) -> float:
        modulate = _make_modulator(prog)
        with torch.no_grad():
            logits = model(val[:, :-1], modulate=modulate)
            vloss = loss_fn(logits.reshape(-1, vocab), val[:, 1:].reshape(-1))
        return float(np.exp(min(float(vloss), 20.0)))

    explorer = TrainingExplorer(
        ledger, ExploreConfig(explore_every=explore_every, pop_size=pop_size,
                              generations=generations,
                              wellformed_penalty=wellformed_penalty),
        run_id=f"run-{seed}")

    import copy
    import math
    identity = identity_modulation()
    DIVERGE_FACTOR = 3.0          # revert an install once the no-mod ppl blows up
    best_ref = [math.inf]         # best unmodulated val ppl seen (the model's health)
    mod_is_identity = [True]
    reverts = [0]
    good_state = [copy.deepcopy(model.state_dict())]   # last healthy checkpoint
    saved_names: set = set()      # modulations persisted-and-still-alive this run
    current_record = [None]       # name of the currently-installed modulation's file

    def _drop_current():
        # a reverted install is proven bad → remove its persisted file
        if store is not None and current_record[0] is not None:
            store.drop(current_record[0])
            saved_names.discard(current_record[0])
        current_record[0] = None

    def _guard(step: int) -> float:
        """Restore the last healthy model when an install destabilizes training.

        The A/B gate installs whatever eval'd best on the *current* model, but a
        modulation that eval's well can still wreck training dynamics (an 8x gain
        compounds over steps). We watch the unmodulated ("identity") val ppl — the
        model's true health — and if an install pushes it past DIVERGE_FACTOR× the
        best seen, we **restore the model weights** to the last healthy checkpoint,
        drop the modulation, and reset the optimizer moments. Swapping the
        modulation alone isn't enough — the damage is already baked into the
        weights, so we roll them back too. Without this the baseline diverges and
        every 'KEPT' is a win over a collapsing reference (meaningless).
        """
        nonlocal current_mod
        ref = val_ppl(identity)
        if (not mod_is_identity[0] and math.isfinite(best_ref[0])
                and ref > best_ref[0] * DIVERGE_FACTOR):
            model.load_state_dict(good_state[0])   # roll the weights back
            opt.state.clear()                      # drop stale Adam moments
            current_mod = identity
            mod_is_identity[0] = True
            reverts[0] += 1
            _drop_current()                        # the reverted install was bad → un-persist it
            if progress:
                progress(f"[guard @ step {step}] modulation destabilized training "
                         f"(ppl {ref:.1f} > {best_ref[0] * DIVERGE_FACTOR:.1f}) → "
                         f"restored last healthy model")
            ref = val_ppl(identity)
        if math.isfinite(ref) and ref <= best_ref[0]:
            best_ref[0] = ref
            good_state[0] = copy.deepcopy(model.state_dict())   # checkpoint a new best
        return ref

    explorations: List[dict] = []
    heartbeat = max(1, explore_every // 5)   # ~5 training pulses between searches
    for step in range(1, total_steps + 1):
        modulate = _make_modulator(current_mod)
        opt.zero_grad()
        logits = model(train[:, :-1], modulate=modulate)
        loss = loss_fn(logits.reshape(-1, vocab), train[:, 1:].reshape(-1))
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if step % heartbeat == 0 and step % explore_every != 0:
            if progress:
                progress(f"[train] step {step}/{total_steps}  train_loss={float(loss):.4f}")
            _guard(step)             # catch divergence mid-window, not only at search time

        if step % explore_every == 0:
            _guard(step)             # revert first so the search sees a healthy model
            res = explorer.explore(step, val_ppl, progress=progress)
            installed = res.improved and math.isfinite(res.best_score)
            winner = res.best_program
            if installed:
                if store is not None:
                    winner = _minimal_equivalent(res.best_program, seed=step)
                current_mod = winner   # install the (minimal) winner
                mod_is_identity[0] = False
                if store is not None:
                    # persist the minimal mechanism with its healthy-baseline Δ
                    # (res.baseline is measured after _guard, on the restored
                    # healthy model; the minimal form is verified-equivalent).
                    from neuroslm.genetic.modulation_store import ModulationRecord
                    # name must be a bare identifier (the store parser is \w+) —
                    # run_id is "run-<seed>", so map hyphens to underscores
                    name = f"{explorer.run_id.replace('-', '_')}_step{step}"
                    store.save(ModulationRecord(
                        name=name, program=winner,
                        metrics={"step": step,
                                 "baseline_ppl": round(res.baseline, 2),
                                 "best_ppl": round(res.best_score, 2),
                                 "delta_ppl": round(res.baseline - res.best_score, 2)}))
                    saved_names.add(name)
                    current_record[0] = name
            explorations.append({
                "step": step, "improved": installed,
                "baseline_ppl": round(res.baseline, 4),
                "best_ppl": round(res.best_score, 4),
                "evaluated": res.n_evaluated, "skipped_duds": res.n_skipped_duds,
                "program": winner.to_source() if installed else None,
            })

    return {
        "explorations": explorations,
        "final_val_ppl": val_ppl(current_mod),
        "installed_modulation": current_mod.to_source(),
        "reverts": reverts[0],
        "persisted": len(saved_names),
        "persisted_names": sorted(saved_names),
        "ledger_stats": ledger.stats(),
    }
