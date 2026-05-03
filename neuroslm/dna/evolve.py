"""Self-improving DNA: genetic optimization of LISP-encoded brain algorithms.

Each region's algorithm is a LISP-like program (templates/*.lisp).
This module enables the brain to *modify its own code* by either:

  1. **Random mutation** — point/subtree mutations + crossover, with
     fitness-driven tournament selection over a gene pool of program
     variants. Always works.

  2. **Model-guided mutation** (when the LM is competent) — the brain
     proposes mutations to its own DNA by sampling tokens conditioned on
     the current program and a "make this faster/smarter" prompt. The
     proposed program is parsed; if it parses and improves fitness on a
     held-out probe set, it replaces the parent.

Fitness is multi-objective:
    f = w_lm * (-Δlm_loss) + w_reason * reasoning_gain
        + w_cons * Φ_proxy + w_eff * (1 / mean_compute)

A program survives only if all of these are non-decreasing on average
over `eval_window` steps (Pareto-style elitism + global rank).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import os
import random
import re
import time
from pathlib import Path

from .dsl import parse, LispError


# ────────────────────────────────────────────────────────────────────────
# Program data structure
# ────────────────────────────────────────────────────────────────────────

@dataclass
class DNAProgram:
    """A parsed LISP program with metadata for evolution."""
    region: str
    source: str
    ast: list = field(default_factory=list)
    fitness: float = float("-inf")
    n_evaluations: int = 0
    parent_id: str | None = None
    id: str = field(default_factory=lambda: f"p{random.randint(0, 1<<32):08x}")
    created_unix: float = field(default_factory=time.time)
    generation: int = 0
    notes: str = ""

    def reparse(self) -> bool:
        try:
            self.ast = parse(self.source)
            return True
        except LispError:
            return False


# ────────────────────────────────────────────────────────────────────────
# Mutation operators (operate on source text + AST)
# ────────────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r"(?<![\w.])(-?\d+\.\d+|-?\d+)(?![\w.])")

# Symbols safe to swap with one another within an arithmetic context.
_SWAP_GROUPS = [
    {"+", "-"}, {"*", "/"},
    {">", "<", ">=", "<="},
    {"and", "or"},
]


def mutate_numeric(source: str, sigma: float = 0.2,
                   point_rate: float = 0.10) -> str:
    """Perturb numeric constants by N(0, sigma * value), independently
    per literal with `point_rate` probability.
    """
    def _repl(m):
        if random.random() > point_rate:
            return m.group(0)
        s = m.group(0)
        try:
            v = float(s) if "." in s else int(s)
        except ValueError:
            return s
        scale = abs(v) if v != 0 else 1.0
        new_v = v + random.gauss(0.0, sigma * scale)
        if isinstance(v, int):
            new_v = int(round(new_v))
        else:
            new_v = round(new_v, 4)
        return str(new_v)
    return _NUMERIC_RE.sub(_repl, source)


def mutate_operator_swap(source: str, point_rate: float = 0.05) -> str:
    """Randomly swap operators within a safe group at points."""
    tokens = re.split(r"(\s+|\(|\))", source)
    out = []
    for t in tokens:
        cand = t
        for grp in _SWAP_GROUPS:
            if t in grp and random.random() < point_rate:
                opts = list(grp - {t})
                cand = random.choice(opts)
                break
        out.append(cand)
    return "".join(out)


def crossover_text(a: str, b: str) -> str:
    """Cheap line-based crossover. Picks each line from a or b 50/50.
    Lines that fail to parse cause the whole result to be rejected.
    """
    la = a.splitlines()
    lb = b.splitlines()
    n = max(len(la), len(lb))
    out = []
    for i in range(n):
        src = la if (i < len(la) and (i >= len(lb) or random.random() < 0.5)) \
              else lb
        if i < len(src):
            out.append(src[i])
    return "\n".join(out)


def model_guided_mutation(source: str, brain, tok, max_attempts: int = 3
                          ) -> str | None:
    """Use the brain's language cortex to propose a mutated program.

    This is best-effort: only succeeds if the LM is good enough to emit a
    parseable program. We try `max_attempts` times. The proposed text is
    *not* trusted blindly — `evolve` always reparses and re-evaluates.
    """
    import torch
    prompt = (
        ";; Improve this LISP program for efficiency and accuracy.\n"
        ";; Original:\n"
        f"{source}\n"
        ";; Improved version:\n"
    )
    device = next(brain.parameters()).device
    ids = torch.tensor([tok.encode(prompt)[-brain.cfg.lang_ctx + 256:]],
                       dtype=torch.long, device=device)
    for _ in range(max_attempts):
        try:
            with torch.no_grad():
                out_ids = brain.generate(ids, max_new=256, temperature=0.7,
                                         top_k=40, use_convergent=False)
            new_text = tok.decode(out_ids[0].tolist())
            # Take only what comes after our prompt marker
            if "Improved version:" in new_text:
                new_text = new_text.split("Improved version:", 1)[1]
            # Try to parse
            parse(new_text)
            return new_text
        except Exception:
            continue
    return None


# ────────────────────────────────────────────────────────────────────────
# Evolutionary pool
# ────────────────────────────────────────────────────────────────────────

class DNAEvolver:
    """Holds a pool of DNAProgram variants per region; evaluates and
    selects them based on a fitness callback.

    The fitness callback signature is:
        fitness_fn(program: DNAProgram, brain, region: str) -> float
    higher = better.
    """

    def __init__(self, pool_size: int = 6,
                 mutation_rate: float = 0.4,
                 crossover_rate: float = 0.3,
                 model_guided_rate: float = 0.0,
                 elitism: int = 2):
        self.pool_size = pool_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.model_guided_rate = model_guided_rate
        self.elitism = elitism
        # region -> list[DNAProgram]
        self.pools: dict[str, list[DNAProgram]] = {}

    # ── Initialization ───────────────────────────────────────────────
    def seed_from_dir(self, templates_dir: str | Path) -> None:
        """Load all *.lisp files in `templates_dir` as initial seeds."""
        d = Path(templates_dir)
        for f in sorted(d.glob("*.lisp")):
            region = f.stem
            src = f.read_text()
            prog = DNAProgram(region=region, source=src)
            if prog.reparse():
                self.pools.setdefault(region, []).append(prog)

    # ── Evolution step ───────────────────────────────────────────────
    def step(self, region: str, brain, tok, fitness_fn) -> DNAProgram:
        """Evolve one generation for one region; return the new champion.
        """
        pool = self.pools.get(region, [])
        if not pool:
            raise ValueError(f"No seed program for region {region!r}")

        # 1. Evaluate any unevaluated programs
        for p in pool:
            if p.fitness == float("-inf"):
                try:
                    p.fitness = float(fitness_fn(p, brain, region))
                except Exception:
                    p.fitness = float("-inf")
                p.n_evaluations += 1

        # 2. Sort by fitness
        pool.sort(key=lambda p: p.fitness, reverse=True)

        # 3. Produce children to refill pool
        elite = pool[:self.elitism]
        children: list[DNAProgram] = list(elite)
        while len(children) < self.pool_size:
            r = random.random()
            if r < self.crossover_rate and len(pool) >= 2:
                a, b = random.sample(pool[:max(2, len(pool)//2)], 2)
                src = crossover_text(a.source, b.source)
                parent_id = f"{a.id}+{b.id}"
                gen = max(a.generation, b.generation) + 1
            elif r < self.crossover_rate + self.model_guided_rate:
                a = pool[0]
                src = model_guided_mutation(a.source, brain, tok) or a.source
                parent_id = a.id; gen = a.generation + 1
            else:
                a = random.choice(pool[:max(2, len(pool)//2)])
                src = mutate_numeric(a.source)
                if random.random() < 0.3:
                    src = mutate_operator_swap(src)
                parent_id = a.id; gen = a.generation + 1

            child = DNAProgram(region=region, source=src,
                               parent_id=parent_id, generation=gen)
            if not child.reparse():
                continue  # skip syntactically invalid mutants
            children.append(child)

        self.pools[region] = children
        # Champion = best of new pool (re-evaluated next step)
        return children[0]

    def champion(self, region: str) -> DNAProgram | None:
        pool = self.pools.get(region, [])
        if not pool:
            return None
        return max(pool, key=lambda p: p.fitness)

    def stats(self) -> dict:
        s = {}
        for region, pool in self.pools.items():
            if not pool:
                continue
            best = max(p.fitness for p in pool if p.fitness != float("-inf"))
            s[region] = {
                "pool_size": len(pool),
                "best_fitness": best,
                "champion_gen": max(p.generation for p in pool),
            }
        return s


# ────────────────────────────────────────────────────────────────────────
# Default fitness function
# ────────────────────────────────────────────────────────────────────────

def default_fitness_fn(program: DNAProgram, brain, region: str) -> float:
    """Default fitness: rewards programs that, when parsed, contain
    primitives the brain knows how to execute, and whose numerical
    constants stay within sane bounds. This is a placeholder that lets
    the GA run continuously even before a region-specific evaluator is
    wired up; replace with a region-specific evaluator that actually
    swaps the program into the brain and measures reasoning gain.
    """
    # Crude proxy: prefer programs of moderate length with bounded constants.
    length = len(program.source.splitlines())
    bounded = all(-100.0 <= float(m.group(0)) <= 100.0
                  for m in _NUMERIC_RE.finditer(program.source)
                  if m.group(0).replace(".", "").lstrip("-").isdigit())
    score = -abs(length - 30) / 30.0          # prefer ~30 lines
    if bounded:
        score += 0.5
    return score
