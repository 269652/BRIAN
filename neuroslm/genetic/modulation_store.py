# -*- coding: utf-8 -*-
"""Persist NGL neuromodulations as ``modulations/*.neuro`` — mergeable, droppable.

A discovered residual-stream modulation (see ``neuro_evolve.py``) is a first-class
artifact, not a throwaway log line. This module serializes an NGL ``Program`` to a
``.neuro`` ``modulation { … }`` block, parses it back exactly, and gives the store
its lifecycle: save / list / show / drop / **merge** (compose gains sequentially,
``g2(g1(h))``). Driven from ``brian modulation …``.

The `.neuro` block format:

    modulation <name> {
        # metric: value        (one per recorded metric)
        program {
            <reg> = <op>(<ins…>[, c=<const>][, k=v…])
            …
            return <reg>
        }
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from neuroslm.genetic.language import Instruction, Program, REGISTRY


# ---------------------------------------------------------------------------
# NGL source <-> Program.
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(r"^\s*(?P<out>[a-zA-Z]\w*)\s*=\s*(?P<op>\w+)\s*\((?P<args>.*)\)\s*$")
_RET_RE = re.compile(r"^\s*return\s+(?P<reg>[a-zA-Z]\w*)\s*$")


def parse_program(text: str, n_scalar: int = 8, n_tensor: int = 16,
                  meta: dict = None) -> Program:
    """Parse ``Program.to_source()`` text back into a ``Program``.

    Bank sizes are inferred from the highest register index actually used (with a
    small buffer), so a program serialized with high register numbers (e.g. after
    a merge) round-trips without silently dropping out-of-range writes. The
    ``n_scalar``/``n_tensor`` args act as floors.
    """
    instrs: List[Instruction] = []
    out_reg = None
    max_t = n_tensor - 1
    max_s = n_scalar - 1

    def _track(reg: str):
        nonlocal max_t, max_s
        if reg and reg[0] == "t":
            try:
                max_t = max(max_t, int(reg[1:]))
            except ValueError:
                pass
        elif reg and reg[0] == "s":
            try:
                max_s = max(max_s, int(reg[1:]))
            except ValueError:
                pass
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line in ("{", "}"):
            continue
        rm = _RET_RE.match(line)
        if rm:
            out_reg = rm.group("reg")
            _track(out_reg)
            continue
        m = _LINE_RE.match(line)
        if not m:
            raise ValueError(f"unparseable NGL line: {line!r}")
        op = m.group("op")
        if op not in REGISTRY:
            raise ValueError(f"unknown op {op!r}")
        ins: List[str] = []
        const = None
        config: List = []
        for tok in _split_args(m.group("args")):
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith("c=") and "=" in tok:
                const = float(tok[2:])
            elif "=" in tok:
                k, v = tok.split("=", 1)
                config.append((k.strip(), float(v)))
            else:
                ins.append(tok)
        _track(m.group("out"))
        for r in ins:
            _track(r)
        instrs.append(Instruction(op, m.group("out"), tuple(ins), const,
                                  tuple(config)))
    if out_reg is None:
        raise ValueError("NGL program has no `return`")
    return Program(instrs, n_scalar=max_s + 1, n_tensor=max_t + 4, out_reg=out_reg,
                   meta=dict(meta or {}))


def _split_args(s: str) -> List[str]:
    return [p for p in s.split(",")] if s.strip() else []


# ---------------------------------------------------------------------------
# Modulation record + .neuro block.
# ---------------------------------------------------------------------------
@dataclass
class ModulationRecord:
    name: str
    program: Program
    metrics: Dict[str, float] = field(default_factory=dict)


def program_to_neuro(rec: ModulationRecord) -> str:
    lines = [f"modulation {rec.name} {{"]
    for k, v in rec.metrics.items():
        lines.append(f"    # {k}: {v}")
    lines.append("    program {")
    for src_line in rec.program.to_source().splitlines():
        lines.append(f"        {src_line}")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def neuro_to_record(text: str) -> ModulationRecord:
    name_m = re.search(r"modulation\s+(\w+)\s*\{", text)
    if not name_m:
        raise ValueError("no `modulation <name> {` header")
    name = name_m.group(1)
    metrics: Dict[str, float] = {}
    for mm in re.finditer(r"#\s*(\w+):\s*([-\d.eE+]+)", text):
        try:
            metrics[mm.group(1)] = float(mm.group(2))
        except ValueError:
            pass
    prog_m = re.search(r"program\s*\{(.*)\}", text, re.DOTALL)
    if not prog_m:
        raise ValueError("no `program { … }` block")
    program = parse_program(prog_m.group(1))
    return ModulationRecord(name=name, program=program, metrics=metrics)


# ---------------------------------------------------------------------------
# Merge — compose gains sequentially: merged(h) = g_n(…g_2(g_1(h))).
# ---------------------------------------------------------------------------
def merge_programs(programs: List[Program]) -> Program:
    """Compose modulation programs so the output of one feeds the next.

    Each program reads ``t0`` and writes ``out_reg``; the merged program threads
    the first's output into the second's ``t0``, and so on, renaming registers to
    avoid collisions.
    """
    if not programs:
        raise ValueError("merge_programs needs at least one program")
    merged_instrs: List[Instruction] = []
    counter = [0]
    n_scalar = max(p.n_scalar for p in programs)

    def fresh() -> str:
        r = f"t{100 + counter[0]}"
        counter[0] += 1
        return r

    cur_input = "t0"
    for prog in programs:
        remap: Dict[str, str] = {"t0": cur_input}

        def rr(reg: str) -> str:
            if reg == "t0":
                return cur_input
            if reg.startswith("s"):
                return reg
            if reg not in remap:
                remap[reg] = fresh()
            return remap[reg]

        for ins in prog.instructions:
            new_ins = tuple(rr(r) for r in ins.ins)
            new_out = rr(ins.out)
            merged_instrs.append(Instruction(ins.op, new_out, new_ins, ins.const, ins.config))
        cur_input = rr(prog.out_reg)

    n_tensor = 100 + counter[0] + 8
    return Program(merged_instrs, n_scalar=n_scalar, n_tensor=n_tensor, out_reg=cur_input)


# ---------------------------------------------------------------------------
# The store.
# ---------------------------------------------------------------------------
class ModulationStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.neuro"

    def save(self, rec: ModulationRecord) -> Path:
        path = self._path(rec.name)
        path.write_text(program_to_neuro(rec), encoding="utf-8")
        return path

    def get(self, name: str) -> ModulationRecord:
        path = self._path(name)
        if not path.exists():
            raise KeyError(f"no modulation {name!r} in {self.root}")
        return neuro_to_record(path.read_text(encoding="utf-8"))

    def list_all(self) -> List[ModulationRecord]:
        out = []
        for p in sorted(self.root.glob("*.neuro")):
            try:
                out.append(neuro_to_record(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def drop(self, name: str) -> bool:
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def merge(self, names: List[str], out_name: str) -> ModulationRecord:
        progs = [self.get(n).program for n in names]
        merged = merge_programs(progs)
        rec = ModulationRecord(name=out_name, program=merged,
                               metrics={"merged_from": len(names)})
        self.save(rec)
        return rec
