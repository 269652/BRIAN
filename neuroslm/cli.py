    # -*- coding: utf-8 -*-
"""brian — unified CLI for the NeuroSLM / BRIAN project.

One entry point for the operations you actually do day-to-day:
compiling architectures, emitting math formulations, analyzing dynamics,
launching vast.ai training, running OOD eval, and managing instances.

Usage:
    py -m neuroslm.cli <command> [args...]

Or via the wrapper scripts:
    bash scripts/brian.sh <command> [args...]      # Linux / git-bash
    .\\scripts\\brian.ps1 <command> [args...]      # PowerShell

Commands (run `brian <cmd> -h` for per-command help):

  Architecture
    compile <arch>            Compile arch.neuro to a runnable nn.Module
    wolfram <arch> [--full]   Emit Mathematica/Wolfram code
    analyze <arch> [--all]    SymPy analysis (fixed points, Jacobian, ...)

  Training (local or remote)
    train [--preset=tiny]     Local training (tiny=CPU minimal, default=30M DSL)
    train --dna=path/to.dna   Train from evolved DNA with fitness config
    deploy [arch] [--steps N] Launch DSL training run on vast (default 10k)
    deploy-100k               Long DSL run (100k steps)
    deploy-brain [...]        Launch a Brain (non-DSL) training run
    deploy-discover <mode>    Run `discover` (experts/trunk/explore) on vast —
                              pushes logs+modulations+ledger while running
    logs <id>                 Tail container logs
    status                    List active vast instances
    destroy <id> | --all      Tear down instance(s)

  Evaluation
    ood <ckpt> [--branch B]   Spin a throwaway OOD-eval instance

  Project
    test [pattern] [--slow]   pytest tests/dsl (skips slow tests by default)
    push                      Commit + push current branch (PAT from .env)

All vast commands use --ssh-less create + self-destroying onstart-cmd
(see fix(vast) commits) so instances cannot bill while idle after a
training run finishes.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Locate the repo root so commands work from any cwd ────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_arch(arg: str) -> str:
    """Accept either `architectures/foo` or just `foo` → resolve to a real path.

    Lookup order:
      1. `<arg>` as given (relative to cwd) — if it contains an arch.neuro,
         use it.
      2. `architectures/<arg>` under the repo root.
      3. The raw `<arg>` (lets the caller see the resolver error directly).
    """
    if arg is None:
        raise ValueError("Architecture argument is required")
    p = Path(arg)
    # If the user passed a file (e.g. arch.neuro or config.neuro),
    # resolve to the containing directory.
    if p.is_file() and p.parent.is_dir() and (p.parent / "arch.neuro").is_file():
        return str(p.parent)
    if p.is_dir() and (p / "arch.neuro").is_file():
        return str(p)
    short = REPO_ROOT / "architectures" / arg
    if short.is_dir() and (short / "arch.neuro").is_file():
        return str(short)
    return arg


def _run(cmd: List[str], **kw) -> int:
    """Run a subprocess and stream its output. Returns exit code."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT), **kw)


def _run_tee(cmd: List[str]) -> Tuple[int, str]:
    """Run ``cmd`` while teeing its combined stdout/stderr to our own
    stdout AND collecting it into a returned string.

    Used by :func:`cmd_test_full` to refresh the duration cache from
    the same pytest invocation the developer is watching live — no
    double-runs, no waste. Returns ``(exit_code, captured_text)``.
    """
    print(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError as e:
        print(f"[run_tee] cannot start {cmd[0]}: {e}")
        return 1, ""

    collected: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        collected.append(line)
    rc = proc.wait()
    return rc, "".join(collected)


def _bash() -> str:
    """Path to bash — git-bash on Windows, /bin/bash elsewhere."""
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    return "bash"


# ── compile ────────────────────────────────────────────────────────────

def _resolve_nfg_arch(arg: str) -> str:
    """Resolve an NFG-compile argument to an architecture root with arch.neuro.

    The user can pass several shapes:

      1. ``architectures/current/`` — a regular folder with arch.neuro
         (the active working-copy; renamed from rcc_bowtie/ on 2026-06-14).
      2. ``some/path/evol.dna``        — a self-contained DNA snapshot.
      3. ``some/path/evol/``           — a folder whose ``arch.neuro`` is
         absent but which contains exactly one ``*.dna`` snapshot
         (the result of ``brian dna unfold X.dna --output some/dir/``).

    For (2) and (3) we unfold the DNA in-memory, extract the
    ``architecture <name> { … }`` block from its embedded DSL, and
    return the live ``REPO_ROOT/architectures/<name>/`` path (which
    still has the ``modules/`` + ``lib/`` sub-trees needed to resolve
    the snapshot's ``@/…`` imports).  This is the same routing pattern
    ``init_evolution()`` in ``neuroslm/utils/colab.py`` already uses
    for training-from-DNA.

    Raises
    ------
    FileNotFoundError
        If the DNA references an architecture name that does not exist
        in ``REPO_ROOT/architectures/``.
    ValueError
        If the DNA's embedded DSL has no parseable
        ``architecture <name>`` declaration.
    """
    import re

    p = Path(arg)

    # ── (3) folder with no arch.neuro but exactly one .dna ─────────
    if p.is_dir() and not (p / "arch.neuro").is_file():
        dna_files = list(p.glob("*.dna"))
        if len(dna_files) == 1:
            return _resolve_nfg_arch(str(dna_files[0]))

    # ── (2) explicit .dna file ─────────────────────────────────────
    if p.is_file() and p.suffix == ".dna":
        from neuroslm.compiler.ribosome import RibosomeCompiler

        compiler = RibosomeCompiler()
        dsl_code = compiler.dna_translator.translate_from_file(str(p))
        match = re.search(r"\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
                          dsl_code)
        if not match:
            raise ValueError(
                f"DNA file {p}: cannot extract `architecture <name> {{ … }}` "
                f"block from its embedded DSL"
            )
        arch_name = match.group(1)
        arch_root = REPO_ROOT / "architectures" / arch_name
        if not (arch_root / "arch.neuro").is_file():
            raise FileNotFoundError(
                f"DNA file {p} references architecture {arch_name!r} but "
                f"{arch_root}/arch.neuro does not exist — cannot resolve "
                f"the snapshot's @/... imports without the source tree"
            )
        return str(arch_root)

    # ── (1) fall through to the canonical DSL path ─────────────────
    return _resolve_arch(arg)


def cmd_compile_nfg(args: argparse.Namespace) -> int:
    """Compile arch to a Neural Flow Graph.

    Default pipeline (since v0.10):
        Lift arch.neuro + all imported modules into the
        :class:`HypergraphIR` (single source of truth), then render via
        Graphviz with a layered ``dot`` layout. PNG/SVG/PDF output uses
        the ``dot`` binary on ``$PATH``; ``--format dot`` writes the DOT
        source verbatim and needs no external tool.

    Legacy pipeline (``--legacy``):
        Falls back to the older ``neuroslm.dsl.nfg`` matplotlib
        renderer plus its ``nfg.py`` dict-of-dicts emission. Kept for
        operators who have the old visual grammar baked into their
        workflow.

    Accepts:
      * a normal architecture folder with ``arch.neuro``,
      * a ``.dna`` snapshot file (auto-routed to its source arch),
      * a folder containing exactly one ``.dna`` (ditto),
      * no positional arg + ``--current`` — reads the configured
        arch / DNA from ``brian.toml`` and writes to the configured
        ``[nfg].output`` (with a ``.heat`` infix when ``--heat`` is
        *also* passed: ``.neuro/nfg.png`` → ``.neuro/nfg.heat.png``).

    ``--heat <heatmap.json>`` works with any input mode — it just
    passes the heatmap payload through to the renderer. The
    ``.heat`` filename infix is *only* applied when ``--heat`` is
    combined with ``--current`` (so the README's plain NFG and the
    operator's heat-overlay NFG live side by side without colliding).
    """
    use_current = getattr(args, "current", False)
    heat = getattr(args, "heat", None)  # str path or None

    # ── --current path: source from brian.toml ─────────────────────
    if use_current:
        if args.arch:
            print(
                "✗ --current and a positional arch are mutually exclusive "
                "(brian.toml owns the source in --current mode).",
                file=sys.stderr,
            )
            return 1
        from neuroslm.project_config import load_project_config
        cfg = load_project_config()
        # In DNA-mode point the renderer at the DNA file (it auto-routes
        # back to the source arch via _resolve_nfg_arch); otherwise the
        # arch folder.
        if cfg.is_dna_mode:
            source = str(cfg.resolve_dna_path())
            source_label = f"DNA {cfg.dna}"
        else:
            source = str(cfg.resolve_arch_path())
            source_label = f"arch {cfg.arch}"
        try:
            arch = _resolve_nfg_arch(source)
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Cannot resolve {source_label}: {e}", file=sys.stderr)
            return 1
        # `default_dir` is unused in this branch — we pass the resolved
        # output path explicitly via args.out below — but the renderer
        # still receives a reasonable fallback.
        default_dir = Path(arch)
        # Override the output path + format from brian.toml. CLI args
        # (--out / --format) win if the operator explicitly sets them.
        resolved_out = str(cfg.nfg_output_path(heat=bool(heat)))
        if not args.out and not args.png:
            args.out = resolved_out
        if not getattr(args, "format", None) or args.format == "png":
            args.format = cfg.nfg_format
        if not getattr(args, "engine", None) or args.engine == "dot":
            args.engine = cfg.nfg_engine
        # Multi-format: stash the full ``[nfg].formats`` list + dpi knob
        # on args so the renderer can loop over them. Only honoured when
        # the operator didn't pass an explicit --out / --png / --format
        # (which would mean "render this one specific thing"). When
        # any of those CLI escape hatches are used we fall back to
        # singular behaviour for orthogonality.
        cli_overrode_target = bool(
            getattr(args, "png", None)
            or (getattr(args, "out", None) and args.out != resolved_out)
        )
        cli_overrode_format = (
            getattr(args, "_format_explicit", False) is True
        )
        if not cli_overrode_target and not cli_overrode_format:
            args._cfg_formats = list(cfg.nfg_formats)
            args._cfg_output_paths = cfg.nfg_output_paths(heat=bool(heat))
        args._cfg_dpi = cfg.nfg_dpi
        args._cfg_spring_gain = cfg.nfg_spring_gain
        args._cfg_panel_opacity = cfg.nfg_panel_opacity
        args._cfg_show_panels = cfg.nfg_show_panels
        print(
            f"current: {source_label}  ->  {resolved_out}"
            + (" (heat overlay)" if heat else "")
        )
    else:
        # Positional-arch path. ``--heat`` may also be set here — it
        # just propagates to the renderer; no filename rewriting (only
        # the --current branch owns the .heat. infix convention).
        try:
            arch = _resolve_nfg_arch(args.arch)
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ Cannot resolve architecture: {e}", file=sys.stderr)
            return 1

        # Output paths default to the SOURCE folder unless the caller passed
        # a `.dna` file — in which case write next to the DNA, not next to
        # the source arch (so we don't clobber the live architecture's
        # nfg.py / nfg.png on a snapshot render).
        src_path = Path(args.arch)
        if src_path.is_file() and src_path.suffix == ".dna":
            default_dir = src_path.parent
        elif src_path.is_dir() and not (src_path / "arch.neuro").is_file() \
                and list(src_path.glob("*.dna")):
            default_dir = src_path
        else:
            default_dir = Path(arch)

        # Apply brian.toml [nfg] settings that the --current branch already
        # handles. Without this, the positional path silently uses "dot" even
        # when brian.toml configures engine = "neato" (or any other engine).
        from neuroslm.project_config import load_project_config
        try:
            cfg = load_project_config()
            if not getattr(args, "engine", None) or args.engine == "dot":
                args.engine = cfg.nfg_engine
            args._cfg_dpi = cfg.nfg_dpi
            args._cfg_spring_gain = cfg.nfg_spring_gain
            args._cfg_panel_opacity = cfg.nfg_panel_opacity
            args._cfg_show_panels = cfg.nfg_show_panels
        except Exception:
            pass  # no brian.toml or malformed — fall through to hard-coded defaults

    use_legacy = getattr(args, "legacy", False)
    if use_legacy:
        return _cmd_compile_nfg_legacy(args, arch, default_dir)
    return _cmd_compile_nfg_graphviz(args, arch, default_dir)


def _cmd_compile_nfg_graphviz(args: argparse.Namespace,
                              arch: str,
                              default_dir: Path) -> int:
    """New default: hypergraph IR → Graphviz."""
    try:
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.nfg_graphviz import render_hypergraph
    except ImportError as e:
        print(f"✗ Graphviz pipeline unavailable: {e}", file=sys.stderr)
        print("  Install the 'graphviz' Python package: pip install graphviz",
              file=sys.stderr)
        print("  Or fall back to the legacy renderer with --legacy",
              file=sys.stderr)
        return 1

    out_format = getattr(args, "format", "png") or "png"
    engine = getattr(args, "engine", "dot") or "dot"
    dpi = int(getattr(args, "_cfg_dpi", 96) or 96)
    spring_gain = float(getattr(args, "_cfg_spring_gain", 0.9) or 0.9)
    panel_opacity = float(getattr(args, "_cfg_panel_opacity", 1.0) or 1.0)
    show_panels = bool(getattr(args, "_cfg_show_panels", True))

    # --png overrides --format implicitly to png
    if args.png:
        out_path = args.png
        out_format = "png"
    elif args.out and args.out.lower().endswith(
            (".png", ".svg", ".pdf", ".dot")):
        out_path = args.out
        out_format = Path(args.out).suffix[1:].lower()
    elif args.out:
        # caller passed --out with a non-image extension → treat as a
        # render target with the requested --format
        out_path = args.out
    else:
        out_path = str(default_dir / f"nfg.{out_format}")

    # Multi-format render: cmd_compile_nfg's --current branch stashes the
    # ``[nfg].formats`` list (and matching per-format output paths) onto
    # args when no CLI override is present. We render each in a loop so a
    # single ``brian compile nfg --current`` produces both .png AND .svg
    # (or whatever the user configured).
    cfg_formats = getattr(args, "_cfg_formats", None)
    cfg_output_paths = getattr(args, "_cfg_output_paths", None)
    render_targets: List[Tuple[str, str]]
    if cfg_formats and cfg_output_paths:
        render_targets = [(fmt, str(p)) for fmt, p in cfg_output_paths]
    else:
        render_targets = [(out_format, out_path)]

    ir = lift_arch_to_hypergraph(arch)

    written_list: List[str] = []
    try:
        title = f"NFG · {Path(arch).name}"
        for fmt, target_path in render_targets:
            written = render_hypergraph(
                ir, target_path, format=fmt, engine=engine, title=title,
                heat=getattr(args, "heat", None), dpi=dpi,
                spring_gain=spring_gain, panel_opacity=panel_opacity,
                show_panels=show_panels,
            )
            written_list.append(written)
    except Exception as e:
        # Most likely the `dot` binary isn't installed; tell the user
        # exactly how to fix it.
        print(f"✗ Graphviz render failed: {e}", file=sys.stderr)
        print("  Make sure the Graphviz 'dot' binary is on your PATH.",
              file=sys.stderr)
        print("  Windows: winget install Graphviz.Graphviz", file=sys.stderr)
        print("  Linux:   sudo apt install graphviz", file=sys.stderr)
        print("  macOS:   brew install graphviz", file=sys.stderr)
        print("  Or fall back to the legacy renderer with --legacy",
              file=sys.stderr)
        return 1

    pops = sum(1 for n in ir.nodes if n.kind == "population")
    nts = sum(1 for n in ir.nodes if n.kind == "neurotransmitter")
    syns = sum(1 for e in ir.hyperedges if e.kind == "synapse")
    mods = sum(1 for e in ir.hyperedges if e.kind == "modulation")
    if len(written_list) == 1:
        print(f"wrote NFG render      -> {written_list[0]}  "
              f"(engine: {engine}, format: {render_targets[0][0]}"
              f"{', dpi=' + str(dpi) if dpi != 96 else ''})")
    else:
        formats_str = ", ".join(fmt for fmt, _ in render_targets)
        print(f"wrote NFG render ({len(written_list)} files, formats: "
              f"{formats_str}, engine: {engine}"
              f"{', dpi=' + str(dpi) if dpi != 96 else ''}):")
        for w in written_list:
            print(f"  - {w}")
    print(f"  hypergraph: {pops} populations · {nts} NT systems · "
          f"{syns} synapses · {mods} modulations")
    return 0


def _cmd_compile_nfg_legacy(args: argparse.Namespace,
                            arch: str,
                            default_dir: Path) -> int:
    """Legacy matplotlib NFG renderer (preserved for back-compat)."""
    from neuroslm.dsl.nfg import (
        compile_nfg, render_nfg, emit_python,
        RCC_BOWTIE_SPEC, SEMANTIC_SPEC,
    )

    out_py = args.out or str(default_dir / "nfg.py")
    out_png = args.png or str(default_dir / "nfg.png")

    g = compile_nfg(arch)
    emit_python(g, out_py)
    print(f"wrote NFG definition  -> {out_py}")
    try:
        spec = SEMANTIC_SPEC if getattr(args, 'semantic', False) else RCC_BOWTIE_SPEC
        render_nfg(g, out_png, spec=spec)
        layout_mode = "semantic" if spec.use_semantic_layout else "legacy"
        print(f"wrote NFG render      -> {out_png}  (layout: {layout_mode})")
    except ImportError as e:
        print(f"render skipped: {e}")
        print("  install matplotlib + networkx to enable PNG output")
    print(f"  stats: {g.stats()}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile an architecture .neuro folder to a runnable nn.Module class."""
    from neuroslm.dsl.codegen import CodeGenerator
    from neuroslm.dsl.multifile import compile_folder

    if args.arch is None:
        print("Error: architecture argument required", file=sys.stderr)
        print("Usage: brian compile <arch> [--out FILE]", file=sys.stderr)
        print("       brian compile nfg <arch> [--out FILE] [--png FILE]", file=sys.stderr)
        return 1

    try:
        arch = _resolve_arch(args.arch)
        ir = compile_folder(Path(arch))
        g = CodeGenerator(ir)
        src = g.generate()
        if args.out:
            Path(args.out).write_text(src, encoding="utf-8")
            print(f"wrote {args.out}  ({len(src)} chars)")
        else:
            print(f"--- generated nn.Module source ({len(src)} chars) ---")
            print(src[: args.head] if args.head else src)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ── hypothesis / discovery ledgers ────────────────────────────────────


def _hypothesis_root() -> Path:
    """Resolve ``hypothesis/`` under the repo root."""
    return REPO_ROOT / "hypothesis"


def _discovery_root() -> Path:
    """Resolve ``discoveries/`` under the repo root."""
    return REPO_ROOT / "discoveries"


def _modulations_root() -> Path:
    """Resolve ``modulations/`` (env ``BRIAN_MODULATIONS_DIR`` overrides)."""
    env = os.environ.get("BRIAN_MODULATIONS_DIR")
    return Path(env) if env else REPO_ROOT / "modulations"


def cmd_modulation(args: argparse.Namespace) -> int:
    """Manage NGL neuromodulations stored under ``modulations/*.neuro``.

    Subcommands: list | show <name> | drop <name> | merge <name…> --name <out>
    """
    from neuroslm.genetic.modulation_store import ModulationStore, program_to_neuro
    store = ModulationStore(_modulations_root())
    sub = args.mod_cmd

    if sub == "list":
        recs = store.list_all()
        if not recs:
            print(f"(no modulations in {_modulations_root()})")
            return 0
        print(f"{'NAME':20s}  METRICS")
        for r in recs:
            metrics = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
            print(f"{r.name:20s}  {metrics}")
        return 0

    if sub == "show":
        try:
            rec = store.get(args.name)
        except KeyError:
            print(f"Error: no modulation {args.name!r}", file=sys.stderr)
            return 1
        print(program_to_neuro(rec))
        return 0

    if sub == "drop":
        if store.drop(args.name):
            print(f"dropped {args.name}")
            return 0
        print(f"Error: no modulation {args.name!r}", file=sys.stderr)
        return 1

    if sub == "merge":
        try:
            rec = store.merge(args.names, args.name)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"merged {args.names} -> {rec.name} "
              f"({len(rec.program.instructions)} instructions)")
        return 0

    print("Usage: brian modulation {list|show|drop|merge}", file=sys.stderr)
    return 1


def cmd_hypothesis(args: argparse.Namespace) -> int:
    """Manage the formal hypothesis ledger under ``hypothesis/``.

    Subcommands:
        list                  — print every hypothesis as a table row
        show <id>             — dump a single hypothesis file
        emit-proofs           — regenerate any missing ``.lean`` stubs
        verify <id>|--all     — run ``lean --json`` against the .lean file(s)
    """
    from neuroslm.discoveries import (
        HypothesisStore, emit_hypothesis_proof, verify_lean_proof,
    )
    sub = args.hyp_cmd
    store = HypothesisStore(_hypothesis_root())

    if sub == "list":
        records = store.list_all()
        if not records:
            print("(no hypotheses yet — see hypothesis/README.md)")
            return 0
        print(f"{'ID':5s}  {'STATUS':10s} {'PROOF':10s}  TITLE")
        for r in records:
            print(f"{r.id:5s}  {r.status:10s} {r.proof_status:10s}  {r.title}")
        return 0

    if sub == "show":
        try:
            rec = store.get(args.id)
        except KeyError:
            print(f"Error: no hypothesis with id {args.id}", file=sys.stderr)
            return 1
        # Print the on-disk markdown verbatim (front-matter + body).
        md_path = _hypothesis_root() / rec.filename()
        print(md_path.read_text(encoding="utf-8"))
        return 0

    if sub == "emit-proofs":
        records = store.list_all()
        wrote = 0
        for r in records:
            if r.proof_status == "verified":
                continue                # don't clobber a verified proof
            out = emit_hypothesis_proof(r, _hypothesis_root())
            store.save(r)
            print(f"  emitted {Path(out).name}")
            wrote += 1
        print(f"emit-proofs: {wrote} file(s) written")
        return 0

    if sub == "verify":
        ids = ([args.id] if args.id else [r.id for r in store.list_all()]
               if args.all else [])
        if not ids:
            print("Usage: brian hypothesis verify <id> | --all", file=sys.stderr)
            return 1
        any_error = False
        for hid in ids:
            try:
                rec = store.get(hid)
            except KeyError:
                print(f"  {hid}: NOT FOUND", file=sys.stderr)
                any_error = True
                continue
            if not rec.proof_path:
                print(f"  {hid}: no proof file emitted yet")
                any_error = True
                continue
            lean_path = REPO_ROOT / rec.proof_path
            verdict = verify_lean_proof(str(lean_path))
            new_status = verdict.proof_status_for_record()
            rec.proof_status = new_status
            store.save(rec)
            print(f"  {hid}: lean={verdict.status:8s} "
                  f"→ proof_status={new_status:8s} "
                  f"({len(verdict.errors)} errors, {verdict.n_sorry} sorry)")
            if verdict.status == "error":
                any_error = True
        return 1 if any_error else 0

    print(f"Unknown hypothesis subcommand: {sub}", file=sys.stderr)
    return 1


def cmd_discover(args: argparse.Namespace) -> int:
    """Search NGL program space for an ML algorithm on CPU (see genetic/).

    Subcommands:
        optimizer  — evolve update-rule programs; score by training a tiny MLP
        flow       — evolve gradient/flow-modulation programs; score loss + EI
    """
    import json
    from neuroslm.genetic.discovery import (
        run_optimizer_discovery, run_flow_modulation_discovery,
    )

    mode = args.discover_cmd
    if mode == "optimizer":
        outcome = run_optimizer_discovery(
            seed=args.seed, pop_size=args.pop, generations=args.generations,
            steps=args.steps, task=args.task,
            include_sota_seeds=not args.from_scratch,
            novelty_weight=getattr(args, "novelty", 0.0),
            device=getattr(args, "device", "cpu"),
            avoid_known=getattr(args, "avoid_known", False),
            macros=getattr(args, "macros", False),
            seed_from=(args.seed_from.split(",") if getattr(args, "seed_from", None) else None),
            progress=True,
        )
        print(f"[discover:optimizer] task={args.task} pop={args.pop} "
              f"gens={args.generations} steps={args.steps}")
        print(f"  SGD baseline final_loss : {outcome.sgd_baseline_loss:.6f}")
        print(f"  discovered  final_loss  : {outcome.best_final_loss:.6f}")
        improv = outcome.sgd_baseline_loss - outcome.best_final_loss
        rel = improv / outcome.sgd_baseline_loss * 100 if outcome.sgd_baseline_loss else 0.0
        print(f"  improvement over SGD    : {improv:+.6f}  ({rel:+.1f}%)")
        print(f"  Pareto front (loss,cost):")
        for s in sorted(outcome.front_stats, key=lambda d: d["final_loss"])[:8]:
            print(f"    loss={s['final_loss']:.6f}  cost={s['cost']:2d}  {s['name']}")
        print("  best program (NGL):")
        for line in outcome.best_program.to_source().splitlines():
            print(f"    {line}")
        payload = {
            "mode": "optimizer",
            "task": args.task,
            "seed": args.seed,
            "pop": args.pop,
            "generations": args.generations,
            "steps": args.steps,
            "from_scratch": bool(args.from_scratch),
            "sgd_baseline_loss": outcome.sgd_baseline_loss,
            "best_final_loss": outcome.best_final_loss,
            "history": outcome.history,
            "front_stats": outcome.front_stats,
            "best_program": outcome.best_program.to_source(),
        }
    elif mode == "flow":
        outcome = run_flow_modulation_discovery(
            seed=args.seed, pop_size=args.pop, generations=args.generations,
            steps=args.steps, device=getattr(args, "device", "cpu"),
            progress=True,
        )
        print(f"[discover:flow] pop={args.pop} gens={args.generations} steps={args.steps}")
        print(f"  discovered final_loss : {outcome.best_final_loss:.6f}")
        print(f"  effective-info (synergy, bits): {outcome.best_ei:.4f}")
        print("  best modulation program (NGL):")
        for line in outcome.best_program.to_source().splitlines():
            print(f"    {line}")
        payload = {
            "mode": "flow",
            "seed": args.seed,
            "pop": args.pop,
            "generations": args.generations,
            "steps": args.steps,
            "best_final_loss": outcome.best_final_loss,
            "best_ei": outcome.best_ei,
            "history": outcome.history,
            "best_program": outcome.best_program.to_source(),
        }
    elif mode == "trunk":
        from neuroslm.genetic.neuro_evolve import run_trunk_evolution
        outcome = run_trunk_evolution(
            seed=args.seed, pop_size=args.pop, generations=args.generations,
            steps=args.steps, device=getattr(args, "device", "cpu"),
            progress=True)
        print(f"[discover:trunk] pop={args.pop} gens={args.generations} steps={args.steps}")
        print(f"  unmodulated trunk val PPL : {outcome.baseline_val_ppl:.4f}")
        print(f"  best modulated   val PPL  : {outcome.best_val_ppl:.4f}")
        print(f"  best neuroanatomic score  : {outcome.best_plausibility:.3f}")
        print("  Pareto front (val_ppl, plausibility):")
        for s in sorted(outcome.front_stats, key=lambda d: d["val_ppl"])[:8]:
            print(f"    ppl={s['val_ppl']:.4f}  plaus={s['plausibility']:.3f}  {s['name']}")
        print("  best modulation program (NGL):")
        for line in outcome.best_program.to_source().splitlines():
            print(f"    {line}")
        print("  note: this is a CPU tiny-LM search — a param-matched GPT-2 "
              "competitor comes ONLY from GPU exploration + extensive GPU "
              "training (brian deploy). Save the gain law (--save) to carry it over.")
        payload = {
            "mode": "trunk",
            "seed": args.seed, "pop": args.pop,
            "generations": args.generations, "steps": args.steps,
            "baseline_val_ppl": outcome.baseline_val_ppl,
            "best_val_ppl": outcome.best_val_ppl,
            "best_plausibility": outcome.best_plausibility,
            "history": outcome.history,
            "front_stats": outcome.front_stats,
            "best_program": outcome.best_program.to_source(),
        }
        if getattr(args, "save", None):
            from neuroslm.genetic.modulation_store import ModulationStore, ModulationRecord
            store = ModulationStore(_modulations_root())
            rec = ModulationRecord(
                name=args.save, program=outcome.best_program,
                metrics={"val_ppl": round(outcome.best_val_ppl, 4),
                         "baseline_ppl": round(outcome.baseline_val_ppl, 4),
                         "plausibility": round(outcome.best_plausibility, 3)})
            path = store.save(rec)
            print(f"  saved modulation -> {path}")
            if getattr(args, "push", False):
                from neuroslm.genetic.modulation_pusher import push_modulations
                res = push_modulations(REPO_ROOT, message=f"modulations: discovered {args.save}")
                print(f"  push: {'-> ' + res.get('branch','?') if res.get('pushed') else res.get('reason')}")
    elif mode == "simplify":
        from neuroslm.genetic.compile_arch import compile_layer_to_ngl
        from neuroslm.genetic.simplify import simplify, programs_equivalent
        src = Path(args.layer_file).read_text(encoding="utf-8")
        compiled = compile_layer_to_ngl(src)
        slim, stats = simplify(compiled.program, n_probes=12, seed=args.seed,
                               return_stats=True)
        equiv = programs_equivalent(compiled.program, slim, n_probes=16,
                                    seed=args.seed + 1)
        print(f"[discover:simplify] {args.layer_file}")
        print(f"  instructions: {stats['before']} -> {stats['after']} "
              f"(-{stats['removed']})")
        print(f"  behaviour preserved on probes: {equiv}")
        print("  simplified program (NGL):")
        for line in slim.to_source().splitlines():
            print(f"    {line}")
        payload = {
            "mode": "simplify",
            "layer_file": args.layer_file,
            "before": stats["before"],
            "after": stats["after"],
            "removed": stats["removed"],
            "equivalent": bool(equiv),
            "program": slim.to_source(),
        }
    elif mode == "qd":
        import numpy as _np
        from neuroslm.genetic.qd_search import map_elites
        from neuroslm.genetic.discovery import benchmark_optimizer
        from neuroslm.genetic.baselines import seeds_for
        rng = _np.random.default_rng(args.seed)
        seeds = seeds_for(args.seed_from.split(",")) if getattr(args, "seed_from", None) else None

        def _ev(prog):
            return -benchmark_optimizer(prog, steps=args.steps, seed=args.seed,
                                        task=args.task, device=args.device).final_loss

        print(f"[discover:qd] MAP-Elites over the semantic manifold "
              f"(task={args.task}, iters={args.iters}, init={args.init})")
        arch = map_elites(_ev, rng, n_iters=args.iters, init_size=args.init,
                          length=6, n_scalar=8, n_tensor=12, seeds=seeds)
        print(f"  illuminated {arch.coverage()} shape-cells of the manifold")
        print(f"  {'CELL(len,fams)':16s} {'loss':>10s}  best program (first op)")
        for cell, prog, fit in sorted(arch.elites(), key=lambda e: -e[2]):
            first = prog.to_source().splitlines()[0] if prog.instructions else "(empty)"
            print(f"  {str(cell):16s} {-fit:10.4f}  {first}")
        payload = {"mode": "qd", "task": args.task, **arch.to_dict()}
    elif mode == "optimize-mechanics":
        from neuroslm.genetic.mechanic_optimizer import (
            analyze_common_mechanics, shared_subexpressions, common_mechanics,
        )
        print("[discover:optimize-mechanics] CSE + superopt on commonly-used mechanics")
        reports = analyze_common_mechanics()
        print(f"  {'MECHANIC':18s} {'BEFORE':>6s} {'AFTER':>6s} {'Δ':>4s}  status")
        for r in reports:
            status = "REDUCIBLE" if r["reducible"] else "already minimal"
            print(f"  {r['name']:18s} {r['before']:>6d} {r['after']:>6d} "
                  f"{r['removed']:>4d}  {status}")
        shared = shared_subexpressions(common_mechanics())
        print(f"  shared subexpressions across mechanics ({len(shared)} — compute once):")
        for sub in shared[:8]:
            print(f"    x{sub['count']}  {sub['expr'][:60]}  in {', '.join(sub['mechanics'])}")
        payload = {"mode": "optimize-mechanics", "reports": reports, "shared": shared}
    elif mode == "baselines":
        from neuroslm.genetic.baselines import tradeoff_table
        print("[discover:baselines] seed the search from these with --seed-from NAME[,NAME]")
        print(f"  {'NAME':10s} {'COST':>4s} {'MEM':>3s}  {'STABILITY':10s} DESCRIPTION")
        for b in tradeoff_table():
            print(f"  {b['name']:10s} {b['cost']:>4d} {b['memory']:>3d}  "
                  f"{b['stability']:10s} {b['description']}")
        payload = {"mode": "baselines", "baselines": tradeoff_table()}
    elif mode == "experts":
        from neuroslm.genetic.discovery import _resolve_device
        from neuroslm.genetic.expert_probe import run_expert_discovery
        dev = _resolve_device(getattr(args, "device", "cpu"))
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"[discover:experts] device={dev.type} rounds={args.rounds} "
              f"batch={args.batch} seq_len={args.seq_len} pop={args.pop} "
              f"gens={args.generations}")
        print(f"[discover:experts] roster: {', '.join(models)}")
        results = run_expert_discovery(
            models=models, rounds=args.rounds, batch=args.batch,
            seq_len=args.seq_len, pop=args.pop, gens=args.generations,
            length=args.length, device=str(dev), push=args.push)
        kept = [r for r in results if r.get("saved")]
        print(f"[discover:experts] {len(results)} probes, {len(kept)} winners banked")
        by_model: dict = {}
        for r in results:
            by_model.setdefault(r["model_id"], []).append(r)
        for mid, rs in by_model.items():
            best = min(rs, key=lambda r: r["best_ce"])
            base = sum(r["baseline_ce"] for r in rs) / len(rs)
            print(f"  {mid}: mean_baseline_ce={base:.4f} best_ce={best['best_ce']:.4f} "
                  f"(best Δ={best['delta_ce']:.4f}, "
                  f"{sum(1 for r in rs if r.get('saved'))} kept)")
        payload = {"mode": "experts", "rounds": args.rounds,
                   "models": models,
                   "kept": [r["saved"] for r in kept],
                   "results": [{k: v for k, v in r.items()
                                if k in ("model_id", "baseline_ce", "best_ce",
                                         "delta_ce", "improved", "saved",
                                         "evaluated", "headroom")}
                               for r in results]}
    elif mode == "explore":
        from neuroslm.genetic.ledger import SearchLedger
        from neuroslm.genetic.training_explorer import run_training_with_exploration
        ledger_path = args.ledger or (REPO_ROOT / ".neuro" / "search_ledger.json")
        led = SearchLedger(ledger_path)
        if getattr(args, "seed_known", True):
            from neuroslm.genetic.known import seed_ledger_with_known
            seed_ledger_with_known(led)   # skip known algorithms → only novel mechanics
        before_stats = led.stats()
        print(f"[discover:explore] tiny-LM training, explore every {args.explore_every} "
              f"steps (ledger: {ledger_path}, {before_stats['total']} prior patterns"
              f"{' incl. seeded prior-art' if getattr(args, 'seed_known', True) else ''})")
        def _progress(msg: str) -> None:
            print("  " + msg, flush=True)   # live heartbeat during the long run
        from neuroslm.genetic.modulation_store import ModulationStore
        mod_store = ModulationStore(_modulations_root())   # persist durable winners
        result = run_training_with_exploration(
            total_steps=args.total_steps, explore_every=args.explore_every,
            seed=args.seed, ledger=led, pop_size=args.pop,
            generations=args.generations, inner_steps=args.inner_steps,
            progress=_progress, store=mod_store,
            wellformed_penalty=args.wellformed_penalty)
        led.save()
        for e in result["explorations"]:
            tag = "KEPT ✓" if e["improved"] else "rejected"
            print(f"  step {e['step']:5d}: {tag}  baseline_ppl={e['baseline_ppl']} "
                  f"best_ppl={e['best_ppl']}  evaluated={e['evaluated']} "
                  f"skipped_duds={e['skipped_duds']}")
        print(f"  final val ppl: {result['final_val_ppl']:.4f}"
              + (f"  (stability reverts: {result['reverts']})" if result.get("reverts") else ""))
        if result.get("persisted"):
            print(f"  persisted {result['persisted']} durable modulation(s) → modulations/: "
                  f"{', '.join(result.get('persisted_names', []))}")
        print(f"  ledger now holds {led.stats()['total']} searched patterns "
              f"({led.stats()['kept']} kept, {led.stats()['rejected']} rejected)")
        if getattr(args, "push", False):
            from neuroslm.genetic.modulation_pusher import push_artifacts
            arts = ["modulations"]
            try:
                arts.append(str(Path(ledger_path).resolve().relative_to(REPO_ROOT.resolve())))
            except ValueError:
                pass  # ledger outside the repo — push modulations only
            res = push_artifacts(REPO_ROOT, arts, message="artifacts: explore run")
            if res.get("pushed"):
                print(f"  push: -> {res.get('branch','?')}")
            else:
                detail = f" ({res['detail']})" if res.get("detail") else ""
                print(f"  push: {res.get('reason')}{detail}")
        payload = {"mode": "explore", **result, "ledger": str(ledger_path)}
    elif mode == "ledger":
        from neuroslm.genetic.ledger import SearchLedger
        ledger_path = args.ledger or (REPO_ROOT / ".neuro" / "search_ledger.json")
        led = SearchLedger(ledger_path)
        if getattr(args, "clear", False):
            led._by_sig.clear()
            led.save()
            print(f"[discover:ledger] cleared {ledger_path}")
            return 0
        if getattr(args, "seed_known", False):
            from neuroslm.genetic.known import seed_ledger_with_known
            n0 = led.stats()["total"]
            seed_ledger_with_known(led)
            led.save()
            print(f"[discover:ledger] seeded prior-art: {led.stats()['total'] - n0} "
                  f"known algorithms recorded (total {led.stats()['total']}) → {ledger_path}")
            return 0
        s = led.stats()
        print(f"[discover:ledger] {ledger_path}")
        print(f"  {s['total']} patterns: {s['kept']} kept, {s['rejected']} rejected, "
              f"{s.get('searched', 0)} searched")
        for r in sorted(led.all(), key=lambda r: r.delta)[:args.top]:
            print(f"    {r.outcome:8s} delta={r.delta:+.4f} step={r.step} "
                  f"[{r.run_id}]  {r.source.splitlines()[0] if r.source else ''}")
        payload = {"mode": "ledger", "stats": s}
    elif mode == "profile":
        import torch
        from neuroslm.dsl.nn_lang import compile_layer
        from neuroslm.genetic.compile_arch import compile_layer_to_ngl, make_probes
        from neuroslm.genetic.profile import profile_program
        from neuroslm.genetic.topology import analyze, propose_edits
        src = Path(args.layer_file).read_text(encoding="utf-8")
        def _num(v):
            f = float(v)
            return int(f) if f.is_integer() else f
        bindings = {k: _num(v) for k, v in (b.split("=", 1) for b in (args.binding or []))}
        compiled = compile_layer_to_ngl(src, bindings=bindings)
        probes = make_probes(compiled, bindings, n=1, seed=args.seed)
        prof = profile_program(compiled.program, probes[0])
        rep = analyze(prof)
        print(f"[discover:profile] {args.layer_file}  ({len(prof.nodes)} ops, "
              f"total_flops={prof.total_flops():.0f})")
        print("  heaviest compute:")
        for n in prof.heavy_compute(top=5):
            print(f"    #{n.index} {n.op:14s} flops={n.flops:.0f}  flow={n.flow:.2f}")
        print("  hottest information flow:")
        for n in prof.hot_flow(top=5):
            print(f"    #{n.index} {n.op:14s} flow={n.flow:.2f}  flops={n.flops:.0f}")
        print("  low-hanging fruit (high flow / low compute):")
        for n in prof.low_hanging(top=5):
            print(f"    #{n.index} {n.op:14s} flow={n.flow:.2f}  flops={n.flops:.0f}")
        print(f"  bottleneck nodes: {rep.bottleneck_nodes}  "
              f"min_cut={rep.min_cut_value:.2f}  connectivity={rep.algebraic_connectivity:.3f}")
        edits = propose_edits(prof)
        print(f"  proposed edits ({len(edits)}):")
        for e in edits[:6]:
            print(f"    [{e['kind']}] {e['reason']}")
        payload = {"mode": "profile", "layer_file": args.layer_file,
                   "profile": prof.to_dict(), "topology": rep.to_dict(),
                   "edits": edits}
    elif mode == "mechanics":
        from neuroslm.genetic.catalog import load_catalog
        cat = load_catalog()
        if getattr(args, "describe", None):
            print(cat.describe(args.describe))
            payload = {"mode": "mechanics", "describe": args.describe,
                       "text": cat.describe(args.describe)}
        else:
            grouped = cat.by_category()
            only = getattr(args, "category", None)
            print(f"[discover:mechanics] {len(cat)} research mechanics in "
                  f"{len(grouped)} categories")
            listed = []
            for category in sorted(grouped):
                if only and category != only:
                    continue
                specs = grouped[category]
                print(f"  {category} ({len(specs)}):")
                for s in specs:
                    print(f"    {s.name:26s} {s.summary[:60]}")
                    listed.append({"name": s.name, "category": category,
                                   "summary": s.summary})
            payload = {"mode": "mechanics", "count": len(cat), "mechanics": listed}
    elif mode == "semantics":
        from neuroslm.genetic.semantics import analyze, describe
        if getattr(args, "layer_file", None):
            from neuroslm.genetic.compile_arch import compile_layer_to_ngl
            src = Path(args.layer_file).read_text(encoding="utf-8")
            prog = compile_layer_to_ngl(src).program
            label = args.layer_file
        else:
            from neuroslm.genetic.known import known_programs
            name = getattr(args, "known", None) or "adam"
            progs = known_programs()
            if name not in progs:
                print(f"[discover:semantics] unknown program {name!r}; "
                      f"available: {', '.join(sorted(progs))}", file=sys.stderr)
                return 1
            prog, label = progs[name], name
        summary = analyze(prog)
        print(f"[discover:semantics] {label}")
        print("  " + describe(summary).replace(" Traits:", "\n  Traits:")
              .replace(" Families:", "\n  Families:").replace(" Notes:", "\n  Notes:"))
        payload = {"mode": "semantics", "target": label, "summary": summary.to_dict()}
    elif mode == "extract-shared":
        from neuroslm.genetic.mechanic_optimizer import common_mechanics
        from neuroslm.genetic.shared_macros import extract_shared_as_macros
        res = extract_shared_as_macros(common_mechanics())
        print(f"[discover:extract-shared] {len(res.extracted)} shared subexpression(s) "
              f"factored into reusable macros")
        for e in res.extracted:
            print(f"  {e['macro']}: {e['expr'][:56]}  ({e['ops']} ops) "
                  f"reused in {', '.join(e['mechanics'])}")
        if not res.extracted:
            print("  (no multi-op subexpression is shared across ≥2 mechanics yet)")
        payload = {"mode": "extract-shared",
                   "extracted": res.extracted,
                   "macros": res.library.names()}
    elif mode == "normalize":
        from neuroslm.genetic.mechanic_optimizer import common_mechanics
        from neuroslm.genetic.known import known_programs
        from neuroslm.genetic.normalize import normalize_semantics
        # the corpus to canonicalize: common mechanics + known prior-art programs
        progs = {**known_programs(), **common_mechanics()}
        res = normalize_semantics(progs, prefer=args.prefer)
        collapsed = [c for c in res.classes if len(c.members) > 1]
        print(f"[discover:normalize] {len(progs)} programs → {len(res.classes)} "
              f"semantic classes ({len(collapsed)} collapse ≥2 variants; "
              f"prefer={args.prefer})")
        for c in collapsed:
            others = [m for m in c.members if m != c.canonical]
            print(f"  {c.canonical}  ⇐  {', '.join(others)}   [{c.reason}]")
        if not collapsed:
            print("  (every program is already its own canonical form)")
        payload = {"mode": "normalize", **res.to_dict(), "prefer": args.prefer}
    else:
        print("Usage: brian discover {optimizer|flow|simplify} [...]", file=sys.stderr)
        return 1

    if getattr(args, "out", None):
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")
    return 0


def cmd_discovery(args: argparse.Namespace) -> int:
    """Manage the autodiscovered-mutation ledger under ``discoveries/``.

    Subcommands:
        list                       — print every discovery as a table row
        show <id>                  — dump a single discovery file
        verify <id>|--all          — run lean against the discovery's proof
        promote <id> <arch>        — splice a verified discovery into the
                                     genome under ``architectures/<arch>``
    """
    from neuroslm.discoveries import (
        DiscoveryStore, verify_lean_proof, splice_discovery_into_dna,
    )
    sub = args.disc_cmd
    store = DiscoveryStore(_discovery_root())

    if sub == "list":
        records = store.list_all()
        if not records:
            print("(no discoveries yet — see discoveries/README.md)")
            return 0
        print(f"{'ID':5s}  {'GEN':4s}  {'PROOF':10s}  {'INTEGRATED':10s}  TITLE")
        for r in records:
            integ = "yes" if r.dna_integrated else "no"
            print(f"{r.id:5s}  {r.generation:4d}  {r.proof_status:10s}  "
                  f"{integ:10s}  {r.title}")
        return 0

    if sub == "show":
        try:
            rec = store.get(args.id)
        except KeyError:
            print(f"Error: no discovery with id {args.id}", file=sys.stderr)
            return 1
        md_path = _discovery_root() / rec.filename()
        print(md_path.read_text(encoding="utf-8"))
        return 0

    if sub == "verify":
        ids = ([args.id] if args.id else [r.id for r in store.list_all()]
               if args.all else [])
        if not ids:
            print("Usage: brian discovery verify <id> | --all", file=sys.stderr)
            return 1
        any_error = False
        for did in ids:
            try:
                rec = store.get(did)
            except KeyError:
                print(f"  {did}: NOT FOUND", file=sys.stderr)
                any_error = True
                continue
            if not rec.proof_path:
                print(f"  {did}: no proof file emitted yet")
                any_error = True
                continue
            lean_path = REPO_ROOT / rec.proof_path
            verdict = verify_lean_proof(str(lean_path))
            new_status = verdict.proof_status_for_record()
            rec.proof_status = new_status
            store.save(rec)
            print(f"  {did}: lean={verdict.status:8s} "
                  f"→ proof_status={new_status:8s}")
            if verdict.status == "error":
                any_error = True
        return 1 if any_error else 0

    if sub == "promote":
        try:
            rec = store.get(args.id)
        except KeyError:
            print(f"Error: no discovery with id {args.id}", file=sys.stderr)
            return 1
        arch_root = Path(_resolve_arch(args.arch))
        try:
            result = splice_discovery_into_dna(rec, arch_root)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        store.save(rec)         # persist dna_integrated bit
        if result.success:
            print(f"promoted {rec.id} → {result.touched_files[0]}")
            return 0
        else:
            print(f"no-op: {result.reason}")
            return 0

    print(f"Unknown discovery subcommand: {sub}", file=sys.stderr)
    return 1


# ── bundle ─────────────────────────────────────────────────────────────

def cmd_dna(args: argparse.Namespace) -> int:
    """Dispatch DNA compile/unfold commands.

    ``brian dna compile`` (no positional arg) reads the architecture
    from ``brian.toml [current].arch`` and, when ``--output`` is also
    omitted, writes to ``brian.toml [current].dna`` (the path the
    deploy actually consumes). This closes the two-file split that
    caused wasted-compute deploy 40951692 on 2026-06-14: the legacy
    behaviour of writing to ``architectures/<arch>/evolution.dna``
    by default left the deploy-targeted ``dna/evol/arch.dna`` stale
    after every recompile.

    The legacy form ``brian dna compile <arch>`` (positional) still
    writes to ``architectures/<arch>/evolution.dna`` — that's the
    "compile some other arch, leave deploy alone" workflow.
    """
    from neuroslm.compiler.ribosome import RibosomeCompiler

    if args.dna_cmd is None:
        print("Error: dna subcommand required (compile or unfold)", file=sys.stderr)
        print("Usage: brian dna compile [<arch>] [--output FILE]", file=sys.stderr)
        print("       brian dna unfold <dna_file> [--output FILE]", file=sys.stderr)
        return 1

    if args.dna_cmd == "compile":
        # ── Resolve the source arch ────────────────────────────────
        # No positional → consult brian.toml [current].arch.
        arch_name = getattr(args, "arch", None)
        used_brian_toml_arch = False
        if not arch_name:
            from neuroslm.project_config import load_project_config
            cfg = load_project_config()
            if not cfg.arch:
                print(
                    "Error: architecture name required for 'brian dna compile'",
                    file=sys.stderr,
                )
                print(
                    "Usage: brian dna compile [<arch>] [--output FILE]",
                    file=sys.stderr,
                )
                print(
                    "       (no positional → reads brian.toml [current].arch)",
                    file=sys.stderr,
                )
                return 1
            arch_name = cfg.arch
            used_brian_toml_arch = True
            print(f"[dna compile] arch from brian.toml: {arch_name}")

        arch = _resolve_arch(arch_name)

        # ── Resolve the output path ────────────────────────────────
        # Precedence:
        #   1. --output flag                            (always wins)
        #   2. brian.toml [current].dna                 (only when arch
        #                                                came from
        #                                                brian.toml too —
        #                                                we'd silently
        #                                                retarget the
        #                                                deploy otherwise)
        #   3. architectures/<arch>/evolution.dna       (legacy default)
        if args.output:
            output = args.output
        elif used_brian_toml_arch:
            from neuroslm.project_config import load_project_config
            cfg = load_project_config()
            if cfg.dna:
                # brian.toml says deploy reads this exact path — write
                # there so a subsequent `brian deploy` picks up the
                # fresh DNA without the user re-typing -o.
                output = str(cfg.resolve_dna_path())
                print(
                    f"[dna compile] output from brian.toml: {output}"
                )
            else:
                output = str(Path(arch) / "evolution.dna")
        else:
            output = str(Path(arch) / "evolution.dna")
        output_path = Path(output)

        try:
            # Same directory-vs-file detection as the `unfold` branch
            # so `brian dna compile rcc_bowtie --output some/dir/` writes
            # `some/dir/<arch_name>.dna` instead of crashing.
            looks_like_dir = (
                output.endswith(os.sep)
                or output.endswith("/")
                or output_path.is_dir()
            )
            if looks_like_dir:
                arch_short = Path(arch).name or "evolution"
                output_path = output_path / f"{arch_short}.dna"
                output = str(output_path)

            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Compiling {arch} -> {output}...")
            compiler = RibosomeCompiler()
            compiler.compile_file(arch, str(output_path))
            print(f"[OK] DNA written to {output}")
            return 0
        except Exception as e:
            print(f"[ERROR] Compilation failed: {e}", file=sys.stderr)
            return 1

    elif args.dna_cmd == "unfold":
        if not hasattr(args, 'dna') or args.dna is None:
            print("Error: DNA file path required for 'brian dna unfold'", file=sys.stderr)
            print("Usage: brian dna unfold <dna_file> [--output FILE]", file=sys.stderr)
            return 1

        dna_path = args.dna
        output = args.output or str(Path(dna_path).with_suffix(".neuro"))
        output_path = Path(output)

        try:
            # If `--output` looks like a directory destination (trailing
            # separator, OR points to an existing directory), write the
            # unfolded DSL inside it as `<dna_stem>.neuro`. This makes
            # `brian dna unfold X.dna --output some/dir/` behave like
            # the user expects on Windows (where a trailing `\` would
            # otherwise fan out into `open()` with Errno 22).
            looks_like_dir = (
                output.endswith(os.sep)
                or output.endswith("/")
                or output_path.is_dir()
            )
            if looks_like_dir:
                output_path = output_path / (Path(dna_path).stem + ".neuro")
                output = str(output_path)

            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Unfolding {dna_path} -> {output}...")
            compiler = RibosomeCompiler()
            compiler.unfold_file(dna_path, output)
            print(f"[OK] DSL written to {output}")
            return 0
        except Exception as e:
            print(f"[ERROR] Unfold failed: {e}", file=sys.stderr)
            return 1

    else:
        print(f"Unknown DNA command: {args.dna_cmd}", file=sys.stderr)
        return 1


def cmd_bundle_arch(args: argparse.Namespace) -> int:
    """Bundle all .neuro files from an architecture into a single file for AI analysis."""
    from pathlib import Path

    arch = _resolve_arch(args.arch)
    arch_path = Path(arch)

    if not arch_path.is_dir():
        print(f"error: {arch} is not a directory", file=sys.stderr)
        return 1

    if not (arch_path / "arch.neuro").is_file():
        print(f"error: {arch}/arch.neuro not found", file=sys.stderr)
        return 1

    # Collect all .neuro files, sorted by path
    neuro_files = sorted(arch_path.rglob("*.neuro"))

    if not neuro_files:
        print(f"error: no .neuro files found in {arch}", file=sys.stderr)
        return 1

    # Build the bundle
    bundle = []
    bundle.append("=" * 78)
    bundle.append(f"ARCHITECTURE BUNDLE: {arch_path.name}")
    bundle.append(f"Generated: {arch_path}")
    bundle.append(f"Files: {len(neuro_files)}")
    bundle.append("=" * 78)
    bundle.append("")

    # Include arch.neuro first, then other files alphabetically
    arch_neuro = arch_path / "arch.neuro"
    other_files = [f for f in neuro_files if f != arch_neuro]

    for neuro_file in [arch_neuro] + other_files:
        rel_path = neuro_file.relative_to(arch_path)
        bundle.append("")
        bundle.append("─" * 78)
        bundle.append(f"FILE: {rel_path}")
        bundle.append("─" * 78)
        bundle.append("")

        try:
            content = neuro_file.read_text(encoding="utf-8")
            bundle.append(content)
        except Exception as e:
            print(f"warning: could not read {rel_path}: {e}", file=sys.stderr)
            bundle.append(f"[ERROR: could not read file: {e}]")

    # Write or print
    bundle_text = "\n".join(bundle)

    if args.out:
        Path(args.out).write_text(bundle_text, encoding="utf-8")
        print(f"wrote {args.out}  ({len(bundle_text)} chars, {len(neuro_files)} files)")
    else:
        # Write to stdout with UTF-8 encoding handling
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(bundle_text)

    return 0


# ── wolfram ────────────────────────────────────────────────────────────

def cmd_wolfram(args: argparse.Namespace) -> int:
    """Emit the architecture as Wolfram / Mathematica code."""
    from neuroslm.dsl.wolfram import (
        architecture_to_wolfram, architecture_to_wolfram_full, save_wolfram,
    )
    arch = _resolve_arch(args.arch)
    if args.full:
        code = architecture_to_wolfram_full(arch)
    else:
        code = architecture_to_wolfram(arch)
    if args.out:
        save_wolfram(arch, args.out, full=args.full)
        print(f"wrote {args.out}  ({len(code)} chars)")
    else:
        print(code)
    return 0


# ── analyze ────────────────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> int:
    """SymPy analysis: fixed points + Jacobian + stability + graph + WA queries."""
    from neuroslm.dsl import analyzer as A
    # Repackage namespace for analyzer.main()
    cli = []
    cli.append(_resolve_arch(args.arch))
    if args.all:
        cli.append("--all")
    if args.fixed_points:
        cli.append("--fixed-points")
    if args.jacobian:
        cli.append("--jacobian")
    if args.stability:
        cli.append("--stability")
    if args.wa_queries:
        cli.append("--wa-queries")
    if args.graph:
        cli.extend(["--graph", args.graph])
    if args.flow:
        cli.append("--flow")
    if args.phi:
        cli.append("--phi")
    if args.discover:
        cli.extend(["--discover", args.discover])
        if args.top_k:
            cli.extend(["--top-k", str(args.top_k)])
    if args.topo_10x:
        cli.append("--topo-10x")
    return A.main(cli)


# ── deploy / deploy-100k / deploy-brain ───────────────────────────────

def _run_hook(name: str, repo_root: Optional[Path] = None,
              env: Optional[dict] = None) -> int:
    """Thin shim around :func:`neuroslm.hooks.run_hook` so cmd_deploy
    can be monkey-patched in tests without touching the runner module.

    Returns 0 on success / skip / disabled; non-zero on fatal hook
    failure. ``cmd_deploy`` aborts on any non-zero value, propagating
    the exact exit code upstream.
    """
    from neuroslm.hooks import run_hook
    return run_hook(name, repo_root if repo_root is not None else REPO_ROOT,
                    env=env)


def _deploy_confirm_is_human() -> bool:
    """True when a confirmation ``input()`` will actually reach a human.

    Two human channels:
    - a real TTY (interactive terminal), or
    - an interactive notebook frontend — Colab / Kaggle / Jupyter — where
      ``input()`` renders a prompt box a person types into (this is the
      "deploy from your phone via Colab" path).

    Everything else — plain scripts, piped stdin, CI, and agent-driven
    subprocesses (``detect_environment() == "script"``) — returns False and is
    blocked. An agent cannot type into a live Colab input box, and a headless
    ``input()`` raises EOFError (caught below), so this does not weaken the
    anti-agent guarantee: it only admits a genuinely interactive human.
    """
    try:
        if sys.stdin.isatty():
            return True
    except Exception:
        pass
    try:
        from neuroslm.utils.secrets import detect_environment
        return detect_environment() in ("colab", "kaggle", "jupyter")
    except Exception:
        return False


def _require_human_confirmation(platform: str, steps: int) -> None:
    """Block until a human types 'deploy' at a real TTY or notebook prompt.

    Raises SystemExit(1) when:
    - the confirmation channel is not human (piped input, subprocess, agent, CI)
    - user types anything other than the word 'deploy'
    - stdin reaches EOF or user hits Ctrl-C

    No flag bypasses this gate. It exists specifically to prevent AI agents
    from autonomously launching paid cloud instances; the notebook path still
    requires a live human to type into the Colab/Jupyter input box.
    """
    if not _deploy_confirm_is_human():
        print(
            "[deploy] BLOCKED: no interactive human confirmation channel.\n"
            "  Deploying requires a human to confirm at a real TTY or in an\n"
            "  interactive Colab/Jupyter cell. Piped input, agent-driven calls,\n"
            "  and CI contexts are rejected here.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\n[deploy] About to launch a paid cloud instance: {platform}, {steps:,} steps."
        "\n[deploy] This costs real money and cannot be undone automatically."
        '\n[deploy] Type "deploy" to confirm, anything else to abort: ',
        end="", flush=True,
    )
    try:
        answer = input()
    except (EOFError, KeyboardInterrupt):
        print("\n[deploy] Aborted.", file=sys.stderr)
        sys.exit(1)

    if answer.strip() != "deploy":
        print(
            f'[deploy] Expected "deploy", got "{answer.strip()}". Aborted.',
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_deploy(args: argparse.Namespace) -> int:
    """Launch a DSL or DNA training run on a cloud platform.

    Source-of-truth precedence (highest wins)::

        1. CLI flag (``--steps`` / ``--branch`` / ``--dna`` / ``--platform``)
        2. ``brian.toml`` ([defaults].steps / [defaults].branch /
           [current].dna / [deploy].platform)
        3. Hardcoded fallback (10_000 steps, platform "vast")

    Examples::

        brian deploy                              # all from brian.toml
        brian deploy --steps 50000               # override steps
        brian deploy --platform lightning        # use Lightning AI
        brian deploy yolo                         # alias for --no-verify
    """
    # ── yolo alias: `brian deploy yolo` → deploy --no-verify ──
    if getattr(args, "arch", None) == "yolo":
        args.arch = None
        args.no_verify = True

    # ── Pre-deploy hook ──
    if getattr(args, "no_verify", False):
        print("[deploy] pre-deploy hook SKIPPED (--no-verify)")
    else:
        hook_rc = _run_hook("pre-deploy")
        if hook_rc != 0:
            print(f"[deploy] pre-deploy hook failed (exit {hook_rc}); aborting.",
                  file=sys.stderr)
            return hook_rc

    # ── Config load ──
    from neuroslm.project_config import load_project_config
    cfg = load_project_config()

    # ── Platform: CLI --platform > brian.toml [deploy].platform > "vast" ──
    platform = getattr(args, "platform", None) or cfg.default_platform or "vast"

    # ── Steps: CLI > brian.toml [defaults].steps > 10_000 ──
    steps = args.steps
    if steps is None:
        steps = cfg.default_steps if cfg.default_steps > 0 else 10_000

    # ── Branch: CLI > brian.toml [defaults].branch > leave unset ──
    branch = args.branch
    if branch is None and cfg.default_branch:
        branch = cfg.default_branch

    # ── DNA: CLI > brian.toml [current].dna (when file exists) > None ──
    dna_path = getattr(args, "dna", None)
    if not dna_path and cfg.is_dna_mode:
        dna_path = cfg.dna
        print(f"[deploy] brian.toml DNA mode: {dna_path}")

    # ── OOD probe cadence: CLI --ood > brian.toml [defaults].ood_every > 0 ──
    # When > 0 the on-box trainer fires `_mid_ood_eval` every N steps and
    # prints `[mid-ood] step N: wikitext ppl=...` — the line `brian ps`
    # parses to populate the PPL column. The per-step `gif[ood_ema=…]`
    # in normal log rows is a *training-batch* EMA, not held-out OOD.
    ood = args.ood if args.ood else cfg.default_ood_every
    log_every = cfg.default_log_every
    save_every = cfg.default_save_every
    push_every = cfg.default_push_every

    # ── Build DeployConfig ──
    from neuroslm.connectors import DeployConfig, get_connector
    config = DeployConfig(
        steps=steps,
        branch=branch,
        scale=args.scale if args.scale else (cfg.default_scale or None),
        label=getattr(args, "label", None),
        ood_every=ood,
        log_every=log_every,
        save_every=save_every,
        push_every=push_every,
        push_backend=cfg.default_push_backend,
        hf_repo_id=cfg.default_hf_repo_id,
        push_optimizer=cfg.default_push_optimizer,
    )

    # ── Machine: CLI --machine > brian.toml [deploy].machine > "" ──
    # Threaded through extra_env so any connector that understands
    # ``LIGHTNING_MACHINE`` (or future ``VAST_MACHINE``) picks it up
    # without needing a dedicated DeployConfig field per connector.
    machine = getattr(args, "machine", None) or cfg.default_machine
    if machine:
        config.extra_env["LIGHTNING_MACHINE"] = machine
        print(f"[deploy] machine: {machine}")

    # ── Teamspace: CLI --teamspace > brian.toml [deploy].teamspace > "" ──
    # Lightning-only; Vast ignores extra_env keys it doesn't recognise.
    teamspace = getattr(args, "teamspace", None) or cfg.default_teamspace
    if teamspace:
        config.extra_env["LIGHTNING_TEAMSPACE"] = teamspace
        print(f"[deploy] teamspace: {teamspace}")

    # ── Arch: CLI positional > brian.toml [current].arch ──
    # Previously: ``config.arch`` was left unset when no CLI ``--arch``
    # was passed, so the connector fell back to ``architectures/current``
    # (a stale working-copy folder). That meant deploys *silently
    # ignored* the configured ``[current].arch = "architectures/SmolLM"``
    # and trained on a different file with different (and out-of-date)
    # NFO / regularization settings. Now we always propagate ``cfg.arch``
    # when neither ``--arch`` nor DNA-mode is set.
    _arch_arg = getattr(args, "arch", None)
    if _arch_arg:
        arch_path = Path(_arch_arg)
        if arch_path.is_file():
            arch_path = arch_path.parent
        config.arch = str(arch_path)
    elif not dna_path and cfg.arch:
        # No CLI override and no DNA workspace: honour brian.toml.
        config.arch = cfg.arch
        print(f"[deploy] arch: {cfg.arch} (from brian.toml [current].arch)")

    # ── Resume ──
    resume_target = getattr(args, "resume", None)
    use_latest = getattr(args, "latest", False)
    if use_latest and not resume_target:
        try:
            from neuroslm.hf_checkpoints import find_latest_checkpoint
            latest = find_latest_checkpoint(
                repo_id=getattr(args, "hf_repo", None),
                prefix=getattr(args, "hf_prefix", "") or "",
            )
        except Exception as e:
            print(f"[deploy] --latest lookup failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        if latest is None:
            print("[deploy] --latest: no checkpoints found on HF Hub.",
                  file=sys.stderr)
            return 1
        repo = (getattr(args, "hf_repo", None)
                or os.environ.get("HF_REPO_ID", "")
                or "moritzroessler/BRIAN")
        resume_target = f"hf://{repo}/{latest.path_in_repo}"
        print(f"[deploy] --latest resolved to step {latest.step}: {resume_target}")
    if resume_target:
        config.resume_from = resume_target
        if getattr(args, "hf_repo", None):
            config.extra_env["HF_REPO_ID"] = args.hf_repo
        print(f"[deploy] resuming from: {resume_target}")

    # ── DNA workspace compilation (fail-fast before any provider call) ──
    if dna_path:
        try:
            from neuroslm.compiler.run_workspace import prepare_run_workspace
            workspace = prepare_run_workspace(dna=dna_path)
        except Exception as e:
            print(f"[deploy] workspace preparation failed: {e}", file=sys.stderr)
            print("[deploy] aborting — no cloud resources were provisioned.",
                  file=sys.stderr)
            return 1
        print(f"[deploy] workspace ready: {workspace.arch_root}")
        print(f"        source: {workspace.source_kind}={workspace.source_path}")
        print(f"        hypergraph IR: {len(workspace.hypergraph_ir.nodes)} nodes, "
              f"{len(workspace.hypergraph_ir.hyperedges)} edges")
        config.arch = str(workspace.arch_root)
        config.source_dna = dna_path

    # ── Load secrets from .env before the connector reads os.environ ──
    # VastConnector._build_env() does os.environ.copy(), so GH_TOKEN /
    # HF_TOKEN / VAST_API_KEY must be in os.environ at call time.
    # bootstrap_secrets() walks CWD upward for a .env file and writes
    # found values into os.environ — it is a no-op when values are
    # already exported or the .env is absent.
    try:
        from neuroslm.utils.secrets import bootstrap_secrets
        bootstrap_secrets(
            ["GH_TOKEN", "HF_TOKEN", "VAST_API_KEY"],
            aliases={
                "GH_TOKEN":     ["GITHUB_TOKEN", "GITHUB", "GITHUB_PAT"],
                "VAST_API_KEY": ["VAST_AI", "VASTAI_API_KEY"],
            },
            verbose=False,
        )
    except Exception:
        pass

    # ── Human confirmation gate (cannot be bypassed by any flag) ──
    _require_human_confirmation(platform, config.steps)

    # ── Dispatch to connector ──
    try:
        connector = get_connector(platform)
    except ValueError as e:
        print(f"[deploy] {e}", file=sys.stderr)
        return 1
    print(f"[deploy] platform: {platform} ({type(connector).__name__})")
    return connector.launch(config)


def cmd_deploy_100k(args: argparse.Namespace) -> int:
    """Shortcut for a long-horizon (100k steps) DSL run."""
    from neuroslm.project_config import load_project_config
    from neuroslm.connectors import DeployConfig, get_connector
    cfg = load_project_config()
    platform = cfg.default_platform or "vast"
    config = DeployConfig(
        steps=100_000,
        branch=args.branch or cfg.default_branch or None,
        log_every=cfg.default_log_every,
        save_every=cfg.default_save_every,
        push_every=cfg.default_push_every,
        push_backend=cfg.default_push_backend,
        hf_repo_id=cfg.default_hf_repo_id,
    )
    return get_connector(platform).launch(config)


def cmd_deploy_discover(args: argparse.Namespace) -> int:
    """Launch a `brian discover <mode>` run on vast.ai (not training).

    Same anti-agent human-confirmation gate as `brian deploy` — no separate,
    weaker path. Only experts/trunk/explore are deployable (see
    ``neuroslm.connectors.vast_discover.DEPLOYABLE_MODES``); the run pushes
    its log + modulations + search ledger to git on a background timer WHILE
    it runs, so an interrupted instance never loses more than one interval
    of progress.
    """
    from neuroslm.connectors.vast_discover import (
        DEPLOYABLE_MODES, DiscoverDeployConfig, VastDiscoverConnector,
    )

    mode = args.deploy_discover_mode
    if mode not in DEPLOYABLE_MODES:
        print(f"[deploy-discover] mode {mode!r} not deployable to vast.ai — "
              f"choose one of {DEPLOYABLE_MODES}. The other discover modes "
              f"finish in seconds/minutes on the free local Colab GPU "
              f"(see `brian discover {mode} --help`); no paid rental needed.",
              file=sys.stderr)
        return 1

    discover_args: List[str] = []
    if mode == "experts":
        if args.models:
            discover_args += ["--models", args.models]
        if args.rounds is not None:
            discover_args += ["--rounds", str(args.rounds)]
        if args.batch is not None:
            discover_args += ["--batch", str(args.batch)]
        if args.seq_len is not None:
            discover_args += ["--seq_len", str(args.seq_len)]
        if args.pop is not None:
            discover_args += ["--pop", str(args.pop)]
        if args.generations is not None:
            discover_args += ["--generations", str(args.generations)]
        discover_args += ["--device", "auto"]
    elif mode == "trunk":
        if args.pop is not None:
            discover_args += ["--pop", str(args.pop)]
        if args.generations is not None:
            discover_args += ["--generations", str(args.generations)]
        if args.steps is not None:
            discover_args += ["--steps", str(args.steps)]
        if args.seed is not None:
            discover_args += ["--seed", str(args.seed)]
        discover_args += ["--device", "auto"]
        if args.label:
            discover_args += ["--save", args.label]
    elif mode == "explore":
        if args.steps is not None:
            discover_args += ["--total-steps", str(args.steps)]
        if args.pop is not None:
            discover_args += ["--pop", str(args.pop)]
        if args.generations is not None:
            discover_args += ["--generations", str(args.generations)]
        if args.seed is not None:
            discover_args += ["--seed", str(args.seed)]
        discover_args += ["--seed-known"]

    config = DiscoverDeployConfig(
        mode=mode,
        discover_args=discover_args,
        branch=args.branch,
        label=args.label or "neuroslm-discover",
        push_interval=args.push_interval or 90,
        gpu_query=args.gpu_query or "",
    )

    # Same gate `brian deploy` uses — a human must type "deploy" at a real
    # TTY or in a live Colab/Jupyter cell. No flag bypasses this.
    _require_human_confirmation(f"vast (discover:{mode})", args.rounds or args.generations or 0)

    return VastDiscoverConnector().launch(config)


def cmd_deploy_brain(args: argparse.Namespace) -> int:
    """Launch a Brain (non-DSL) training run on vast.ai."""
    env = os.environ.copy()
    env["USE_DSL"] = "0"
    env["FRESH"] = "1"
    env["STEPS"] = str(args.steps)
    if args.preset:
        env["PRESET"] = args.preset
    if args.branch:
        env["BRANCH"] = args.branch
    env["PYTHONIOENCODING"] = "utf-8"
    return _run([_bash(), "scripts/vast_train.sh"], env=env)


# ── logs / status / destroy ───────────────────────────────────────────

def _find_latest_log_file(log_dir: Path) -> Optional[Path]:
    """Return the newest ``*.log`` file in ``log_dir`` by mtime, or None.

    Returns ``None`` if the directory is missing or contains no ``.log``
    files. Matches every ``.log`` file regardless of naming convention —
    both the legacy ``<id>__neuroslm-*.log`` format and the newer
    ``<utc>_<container>_<arch>_<params>_<label>_stepNofN.log`` format
    work. (Contrast with ``_scan_recent_destroyed`` which uses the
    narrower glob ``*__neuroslm-*.log`` and misses the new format.)
    """
    if not log_dir.is_dir():
        return None
    logs = [p for p in log_dir.glob("*.log") if p.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _find_local_log_for_instance(instance_id: str) -> Optional[Path]:
    """Find the locally-pushed log snapshot for a (possibly destroyed)
    vast instance. Returns the newest match or None.

    ``scripts/log_pusher.sh`` running on the instance periodically
    rsyncs the training log to ``logs/vast/<instance_id>__neuroslm-*.log``
    in this repo, then commits + pushes. So even after the instance is
    destroyed, the log lives on locally.
    """
    log_dir = Path("logs/vast")
    if not log_dir.is_dir():
        return None
    matches = [p for p in log_dir.glob(f"{instance_id}__*.log") if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _print_log_file(path: Path) -> None:
    """Print a log file's content to stdout (UTF-8, errors replaced)."""
    body = path.read_text(encoding="utf-8", errors="replace")
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_logs(args: argparse.Namespace) -> int:
    """Show training logs for a vast.ai instance.

    Three modes:

    1. ``brian logs --latest``
       Local-only. Print the newest ``*.log`` in ``logs/vast/`` by
       mtime. No vast API call. Useful when the last instance was
       destroyed and you just want to see what happened.

    2. ``brian logs <id>``
       Try ``scripts/vast.sh logs <id>`` first (live container).
       If that fails (e.g. the instance is destroyed), fall back to
       the locally-pushed snapshot ``logs/vast/<id>__neuroslm-*.log``.
       If even that's missing, ``git fetch && git pull`` once and
       retry (maybe a sibling workstation pushed the log).

    3. ``brian logs`` (no positional, no ``--latest``)
       User mistake — print a one-line hint and exit non-zero.
    """
    # Mode 1: --latest (strictly local, no vast API call)
    if getattr(args, "latest", False):
        log_dir = Path("logs/vast")
        latest = _find_latest_log_file(log_dir)
        if latest is None:
            print(f"brian logs --latest: no log files found in {log_dir}/")
            return 1
        print(f"# brian logs --latest → {latest.name}")
        _print_log_file(latest)
        return 0

    # Mode 3: nothing to do
    if not args.instance_id:
        print("brian logs: pass an instance id (e.g. `brian logs 40952126`) "
              "or use `--latest` to view the newest log in logs/vast/.")
        return 1

    # Mode 2a: try live container first
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    rc = _run([_bash(), "scripts/vast.sh", "logs", str(args.instance_id)],
              env=env)
    if rc == 0:
        return 0

    # Mode 2b: live call failed → look for locally-pushed snapshot
    local = _find_local_log_for_instance(str(args.instance_id))
    if local is not None:
        print(f"# vast.ai has no record of {args.instance_id}; "
              f"showing local snapshot {local.name}")
        _print_log_file(local)
        return 0

    # Mode 2c: no local file → maybe a sibling pushed it; pull & retry
    print(f"# No local log for {args.instance_id} either; "
          f"running git fetch && git pull...")
    _run(["git", "fetch", "--all"])
    _run(["git", "pull"])
    local = _find_local_log_for_instance(str(args.instance_id))
    if local is not None:
        print(f"# Found pulled log {local.name}")
        _print_log_file(local)
        return 0

    # Mode 2d: nothing worked → user-facing hint
    print(f"brian logs: no log found for instance {args.instance_id} "
          f"(not live on vast.ai, no local snapshot, no pulled snapshot). "
          f"Try `brian logs --latest` to see the most recent run.")
    return 1


def cmd_status(_: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return _run([_bash(), "scripts/vast.sh", "show", "instances"], env=env)


# ── ps (parsed-status table) ───────────────────────────────────────────

# Regex for the standard train_dsl log line:
#   step  9980 | loss 6.28 | lm 6.28 | ppl 536.1 | gnorm 0.91 | lr 3.00e-05 | 28891 tok/s
_STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+"
    r"loss\s+(?P<loss>[\d.]+)\s+\|\s+"
    r"lm\s+[\d.]+\s+\|\s+"
    r"ppl\s+(?P<ppl>[\d.]+).*?"
    r"(?P<tps>\d+)\s+tok/s",
    re.DOTALL,
)
# `[mid-ood] step 3000: wikitext ppl=1550.1`
_MID_OOD_RE = re.compile(
    r"\[mid-ood\]\s+step\s+(?P<step>\d+):\s+wikitext\s+ppl=(?P<ppl>[\d.]+)")


def _parse_phase(log: str) -> str:
    """Detect the run's current phase from log signatures."""
    if "── self-destroying" in log or "vastai destroy instance" in log:
        return "self-destroying"
    if "training reached target" in log or "[train_dsl] done" in log:
        return "training-done"
    if "PASS-MARK EARLY EXIT" in log:
        return "passmark-exit"
    if "─ pushing final checkpoints ─" in log:
        return "pushing-ckpts"
    if "Traceback" in log:
        return "ERROR"
    if "step " in log:
        return "training"
    if "[train_dsl] DSL-LM" in log:
        return "model-built"
    if "── bootstrap" in log or "pip install" in log:
        return "bootstrap"
    if "── cloning " in log:
        return "cloning"
    return "booting"


def _parse_status(log: str) -> dict:
    """Pull the latest training metric + last mid-OOD result from a log tail."""
    # Search in reverse for the last step line (regex on full text + take last)
    step_matches = list(_STEP_RE.finditer(log))
    mid_matches  = list(_MID_OOD_RE.finditer(log))
    out = {"step": None, "loss": None, "ppl": None, "tps": None,
           "mid_ood_step": None, "mid_ood_ppl": None,
           "phase": _parse_phase(log)}
    if step_matches:
        m = step_matches[-1]
        out["step"] = int(m.group("step"))
        out["loss"] = float(m.group("loss"))
        out["ppl"]  = float(m.group("ppl"))
        out["tps"]  = int(m.group("tps"))
    if mid_matches:
        m = mid_matches[-1]
        out["mid_ood_step"] = int(m.group("step"))
        out["mid_ood_ppl"]  = float(m.group("ppl"))
    return out


def cmd_ps(args: argparse.Namespace) -> int:
    """List active training runs across every cloud platform.

    Output columns (vast section):
      ID | LABEL | GPU | $/hr | UPTIME | PHASE | STEP | PPL | OOD-PPL | TOK/S

    Output columns (lightning section):
      JOB ID | LABEL | STATUS | MACHINE | UPTIME | LAST LOG LINE

    Modes:
      --it                interactive watch, redraws every --interval seconds
      --colab <url>       connect to a Colab log server
      --logs <job_id>     stream the last --tail N lines of one remote job's log
      --platform <name>   restrict the listing to one platform
    """
    # ── Single-job log streaming (no live list) ──
    job_id = getattr(args, "logs", None)
    if job_id:
        return _ps_show_logs(args, job_id)

    # ── Colab mode — connect to remote log server ──
    if getattr(args, "colab", None):
        if args.it:
            return _ps_colab_watch(args)
        return _ps_colab_once(args)
    if args.it:
        return _ps_watch(args)
    return _render_ps_once(args)


def _ps_show_logs(args: argparse.Namespace, job_id: str) -> int:
    """Stream the last N lines of one remote job's log.

    Looks up the job in ``.brian/jobs/<job_id>.json``, dispatches to
    the owning connector's :meth:`tail_logs`. With ``--it`` keeps
    refreshing.
    """
    from neuroslm.connectors import get_connector, load_job
    info = load_job(job_id)
    if info is None:
        print(f"brian ps --logs: no job {job_id!r} in .brian/jobs/",
              file=sys.stderr)
        return 1
    try:
        connector = get_connector(info.platform)
    except ValueError as e:
        print(f"brian ps --logs: {e}", file=sys.stderr)
        return 1
    tail_n = int(getattr(args, "tail", 200) or 200)

    def _fetch_and_print(initial: bool = False) -> bool:
        try:
            out = connector.tail_logs(job_id, n=tail_n)
        except NotImplementedError:
            print(f"brian ps --logs: {info.platform!r} connector does not "
                  f"support log tailing", file=sys.stderr)
            return False
        except Exception as e:
            print(f"brian ps --logs: tail failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return False
        if initial:
            print(f"=== {info.platform}/{job_id}  ({info.studio_name or info.host}, "
                  f"last {tail_n} lines) ===")
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        return True

    if not getattr(args, "it", False):
        ok = _fetch_and_print(initial=True)
        return 0 if ok else 1
    # Watch mode — redraw the tail every --interval seconds.
    import time
    interval = float(getattr(args, "interval", 1.0) or 1.0)
    is_tty = sys.stdout.isatty()
    if is_tty:
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.flush()
    try:
        while True:
            if is_tty:
                sys.stdout.write("\x1b[H")
            ok = _fetch_and_print(initial=True)
            if is_tty:
                sys.stdout.write("\x1b[J")
            sys.stdout.flush()
            if not ok:
                return 1
            time.sleep(interval)
    except KeyboardInterrupt:
        if is_tty:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()
        print("(stopped)")
        return 0
    finally:
        if is_tty:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


def _ps_colab_once(args: argparse.Namespace, out=None) -> int:
    """Fetch status from a Colab log server and display it."""
    import json
    import urllib.request
    import urllib.error
    
    sink = out if out is not None else sys.stdout
    def _say(msg: str = "") -> None:
        sink.write(msg + "\n")
    
    url = args.colab.rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/status", timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        _say(f"✗ Cannot connect to Colab: {e}")
        _say(f"  URL: {url}")
        return 1
    except json.JSONDecodeError as e:
        _say(f"✗ Invalid response from Colab: {e}")
        return 1
    
    if out is None:
        _say("Brian Task Manager — Colab monitor")
        _say(f"  URL: {url}")
        _say("")
    
    step = data.get("step")
    ppl = data.get("ppl")
    tps = data.get("tps")
    ood_step = data.get("ood_step")
    ood_ppl = data.get("ood_ppl")
    lines = data.get("lines", 0)
    
    hdr = f"{'PLATFORM':<10}  {'STEP':>8}  {'PPL':>10}  {'TOK/S':>8}  {'OOD-PPL':>12}  {'LOG LINES':>10}"
    _say(hdr)
    _say("-" * len(hdr))
    
    step_s = str(step) if step is not None else "-"
    ppl_s = f"{ppl:.1f}" if ppl is not None else "-"
    tps_s = f"{tps/1000:.0f}k" if tps else "-"
    ood_s = f"{ood_ppl:.0f}@{ood_step}" if ood_ppl is not None else "-"
    
    _say(f"{'Colab':<10}  {step_s:>8}  {ppl_s:>10}  {tps_s:>8}  {ood_s:>12}  {lines:>10}")
    
    # Show recent log tail if not in watch mode
    if out is None and not getattr(args, "it", False):
        _say("")
        _say("Recent logs (last 20 lines):")
        _say("-" * 60)
        try:
            with urllib.request.urlopen(f"{url}/logs", timeout=10) as resp:
                logs = resp.read().decode()
                for line in logs.strip().split("\n")[-20:]:
                    _say(line)
        except Exception:
            _say("(could not fetch logs)")
    
    return 0


def _ps_colab_watch(args: argparse.Namespace) -> int:
    """Watch mode for Colab log server — streams live logs."""
    import datetime
    import io
    import time
    import urllib.request
    import urllib.error
    
    url = args.colab.rstrip("/")
    is_tty = sys.stdout.isatty()
    
    if is_tty:
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.flush()
    
    try:
        while True:
            buf = io.StringIO()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            buf.write(f"Brian Task Manager — Colab monitor   "
                      f"(refresh every {args.interval}s · Ctrl-C to exit)   "
                      f"{ts}\n")
            buf.write(f"URL: {url}\n")
            buf.write("=" * 79 + "\n")
            
            rc = _ps_colab_once(args, out=buf)
            if rc != 0:
                if is_tty:
                    sys.stdout.write("\x1b[?25h")
                    sys.stdout.flush()
                return rc
            
            # Add live log tail
            buf.write("\n")
            buf.write("Live logs (last 30 lines):\n")
            buf.write("-" * 60 + "\n")
            try:
                with urllib.request.urlopen(f"{url}/logs", timeout=5) as resp:
                    logs = resp.read().decode()
                    for line in logs.strip().split("\n")[-30:]:
                        buf.write(line + "\n")
            except Exception as e:
                buf.write(f"(could not fetch logs: {e})\n")
            
            if is_tty:
                sys.stdout.write("\x1b[H" + buf.getvalue() + "\x1b[J")
            else:
                sys.stdout.write("\n\n" + buf.getvalue())
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if is_tty:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()
        print("(stopped)")
        return 0
    finally:
        if is_tty:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


def _ps_watch(args: argparse.Namespace) -> int:
    """Seamless watch mode for `brian ps --it`.

    Strategy:
      1. Hide cursor (\\e[?25l) for the duration of the watch.
      2. Each tick: build the entire render into a StringIO, write it
         in ONE go with cursor-home (\\e[H) + clear-to-end-of-screen
         (\\e[J). No `\\e[2J` between draws — that's what causes the
         black-screen flicker.
      3. On Ctrl-C: show cursor (\\e[?25h), print "(stopped)".

    Falls back to plain re-prints with newlines when stdout isn't a tty.
    """
    import datetime
    import io
    import time
    is_tty = sys.stdout.isatty()
    if is_tty:
        # Hide cursor + clear once at start so the first frame paints a
        # clean canvas. Subsequent frames only home + clear-to-end.
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.flush()
    try:
        while True:
            buf = io.StringIO()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            buf.write("Brian Task Manager — vast.ai monitor   "
                       f"(refresh every {args.interval}s · Ctrl-C to exit)   "
                       f"{ts}\n")
            buf.write("=" * 79 + "\n")
            # Render into the buffer so the actual terminal write is one
            # atomic operation — no partial frames visible to the user.
            rc = _render_ps_once(args, out=buf)
            if rc != 0:
                if is_tty:
                    sys.stdout.write("\x1b[?25h")
                    sys.stdout.flush()
                return rc
            if is_tty:
                # Cursor home + write + clear-to-end. The clear erases
                # any leftover characters past where the new frame ends
                # (e.g. if the previous frame had a longer table) without
                # blanking the screen first.
                sys.stdout.write("\x1b[H" + buf.getvalue() + "\x1b[J")
            else:
                sys.stdout.write("\n\n" + buf.getvalue())
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if is_tty:
            sys.stdout.write("\x1b[?25h\n")
            sys.stdout.flush()
        print("(stopped)")
        return 0
    finally:
        if is_tty:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


def _detect_exit_reason(log_tail: str) -> tuple:
    """From a log tail, infer (exit_reason: str, detail: str).

    Returns one of: "passmark-exit", "completed", "crashed", "destroyed",
    "stopped", "unknown" + a human-readable detail (e.g. the pass-mark name).

    Skips vast.ai's reverse-tunnel SSH chatter ("Error: remote port forwarding
    failed...") — those are sibling-process retries, not training failures.
    """
    import re as _re
    # Pass-mark early exit — pull out the rule name + reason
    m = _re.search(r"PASS-MARK EARLY EXIT @ step (\d+): (.+)", log_tail)
    if m:
        return ("passmark-exit", f"step {m.group(1)}: {m.group(2)[:80]}")
    if "training reached target" in log_tail:
        return ("completed", "all steps done")
    if "Traceback" in log_tail or any(
            _re.search(r"\b(Error|Exception)\b", l) and "port forwarding" not in l
            for l in log_tail.splitlines()):
        # Find a meaningful tail line — skip SSH chatter
        for line in reversed(log_tail.splitlines()):
            if "port forwarding" in line or "known hosts" in line:
                continue
            if ("Traceback" in line or
                    _re.search(r"\b(Error|Exception)\b", line)):
                # Skip pure-warning lines like "Warning: ..."
                if line.strip().startswith("Warning:"):
                    continue
                return ("crashed", line.strip()[:80])
        return ("crashed", "exception in log")
    if "vastai destroy instance" in log_tail or "── self-destroying" in log_tail:
        return ("destroyed", "instance self-destroyed")
    if "[train_dsl] done." in log_tail:
        return ("stopped", "train loop returned")
    return ("unknown", "")


def _scan_recent_destroyed(top_n: int = 3) -> list:
    """Walk logs/vast/*.log, group by instance id, return the most recent
    `top_n` that are no longer running (parsed status indicates exit).

    Returns a list of dicts {id, last_seen, exit_reason, detail, steps, ood_ppl}.
    """
    import re as _re
    from datetime import datetime
    log_dir = Path("logs/vast")
    if not log_dir.is_dir():
        return []
    # Group by instance id (filename prefix before "__")
    by_id: dict = {}
    for p in log_dir.glob("*__neuroslm-*.log"):
        iid = p.stem.split("__")[0]
        prev = by_id.get(iid)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            by_id[iid] = p
    rows = []
    for iid, path in by_id.items():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Only include if we can find an exit signal (otherwise it's still live)
        tail = text[-15000:]   # last ~15KB is enough for exit signals
        reason, detail = _detect_exit_reason(tail)
        if reason == "unknown":
            continue
        # Pull final step + last OOD ppl
        step_matches = list(_STEP_RE.finditer(text))
        mid_matches  = list(_MID_OOD_RE.finditer(text))
        final_step = int(step_matches[-1].group("step")) if step_matches else None
        last_ood   = float(mid_matches[-1].group("ppl")) if mid_matches else None
        rows.append({
            "id": iid,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime),
            "exit_reason": reason,
            "detail": detail,
            "steps": final_step,
            "ood_ppl": last_ood,
            "logfile": path.name,
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:top_n]


# ── Vast.ai "actual_status" values that mean the container is gone ──
_INACTIVE_VAST_STATUSES = frozenset(
    ("stopped", "destroyed", "offline", "exited", "failed")
)
# ── Lightning JobStatus values that mean the run is still live ──
_ACTIVE_LN_STATUSES = frozenset(("pending", "starting", "running", "stopping"))


def _collect_vast_rows(args: argparse.Namespace) -> "tuple[list, str | None]":
    """Query vast.ai and return (row_dicts, error_msg_or_None).

    Each row dict has: plat="v", id, label, gpu, cost, uptime_mins,
    is_active, and all keys from _parse_status() (step, ppl, phase, …).
    """
    import json
    vastai = _vastai_exe()
    raw, rc = _run_capture([vastai, "show", "instances", "--raw"])
    offline_marker = any(s in raw.lower() for s in
                         ("connection", "failed to resolve", "could not resolve",
                          "timed out", "getaddrinfo", "no route to host"))
    if offline_marker or (not raw.strip() and rc != 0):
        return [], "(vast.ai offline — destroyed-instance history may still show below)"
    if rc != 0 and "DEPRECATED" not in raw:
        return [], f"vastai error: {raw[:200]}"
    data: list = []
    start = raw.find("[")
    if start >= 0:
        depth = 0; end = start; in_str = False; esc = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            elif ch == '"': in_str = True
            elif ch == "[": depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0: end = i + 1; break
        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return [], "(can't parse vastai response)"
    rows = []
    show_all = getattr(args, "all", False)
    for inst in data:
        label = inst.get("label") or ""
        if not show_all and not label.startswith("neuroslm"):
            continue
        iid = inst.get("id")
        actual_status = (inst.get("actual_status") or "").lower()
        is_active = actual_status not in _INACTIVE_VAST_STATUSES
        log, _ = _run_capture([vastai, "logs", str(iid)])
        status = _parse_status(log)
        rows.append({
            "plat": "v",
            "id": str(iid),
            "label": label or "(no label)",
            "gpu": inst.get("gpu_name", "?"),
            "cost": inst.get("dph_total", 0),
            "uptime_mins": int(inst.get("uptime_mins", 0) or 0),
            "is_active": is_active,
            **status,
        })
    return rows, None


def _collect_lightning_rows(args: argparse.Namespace) -> list:
    """Query the Lightning job registry and return row dicts.

    Each row dict has: plat="l", id, label, gpu, cost=None,
    uptime_mins, phase, is_active, _step_s, _ppl_s, _ood_s, _tps_s
    (pre-formatted strings from the log tail).
    """
    try:
        from neuroslm.connectors import LightningConnector, JobStatus
    except Exception:
        return []
    try:
        connector = LightningConnector()
        jobs = connector.list_jobs()
    except Exception:
        return []
    if not jobs:
        return []
    import time as _time
    now = int(_time.time())
    rows = []
    for j in jobs:
        up_m = max(0, (now - (j.started_at or now)) // 60)
        is_active = (j.status or "").lower() in _ACTIVE_LN_STATUSES
        step_s = ppl_s = ood_s = tps_s = "-"
        if j.status in (JobStatus.RUNNING.value, JobStatus.STARTING.value):
            try:
                tail = connector.tail_logs(j.job_id, n=500)
                step_s, ppl_s, ood_s, tps_s = _summarise_lightning_tail(tail)
            except Exception:
                pass
        rows.append({
            "plat": "l",
            "id": str(j.job_id)[:10],
            "label": j.label or "(none)",
            "gpu": j.machine or "?",
            "cost": None,
            "uptime_mins": up_m,
            "phase": j.status or "?",
            "is_active": is_active,
            "_step_s": step_s, "_ppl_s": ppl_s, "_ood_s": ood_s, "_tps_s": tps_s,
        })
    return rows


def _format_unified_row(r: dict) -> str:
    """Format one row for the unified brian ps table.

    Handles both vast rows (raw numeric step/ppl/tps) and lightning rows
    (pre-formatted _step_s / _ppl_s / _ood_s / _tps_s strings).
    """
    if "_step_s" in r:
        step_s, ppl_s, ood_s, tps_s = r["_step_s"], r["_ppl_s"], r["_ood_s"], r["_tps_s"]
    else:
        step_s = str(r["step"]) if r.get("step") is not None else "-"
        ppl_s  = f"{r['ppl']:.1f}" if r.get("ppl") is not None else "-"
        ood_s  = (f"{r['mid_ood_ppl']:.0f}@{r['mid_ood_step']}"
                  if r.get("mid_ood_ppl") is not None else "-")
        tps_s  = f"{r['tps']/1000:.0f}k" if r.get("tps") else "-"
    cost_s = f"{r['cost']:.2f}" if r.get("cost") else "-"
    return (
        f"{r['plat']:>1}  {str(r['id'])[:10]:>10}  {r['label'][:28]:<28}  "
        f"{r['gpu'][:12]:<12}  {cost_s:>5}  {r.get('uptime_mins', 0):>6}  "
        f"{r.get('phase', '?')[:16]:<16}  {step_s:>6}  {ppl_s:>8}  "
        f"{ood_s:>9}  {tps_s:>7}"
    )


def _render_ps_once(args: argparse.Namespace, out=None) -> int:
    """Single ps render — extracted from cmd_ps so --it can call it in a loop.

    `out`: optional file-like (e.g. io.StringIO from the watch loop). When
    set, all output goes there instead of stdout; status/error messages
    that would normally hit stderr also route to `out` so the watch
    redraw stays atomic.

    Collects rows from all requested platforms, filters to active-only
    (unless ``--all``), then renders one unified table with a platform
    code column (v/l/c). Empty platforms produce no section header and
    no error message — the unified fallback fires only when every
    platform yields zero active rows.
    """
    sink = out if out is not None else sys.stdout
    def _say(msg: str = "") -> None:
        sink.write(msg + "\n")
    platform_filter = (getattr(args, "platform", None) or "all").lower()
    show_vast = platform_filter in ("all", "vast")
    show_lightning = platform_filter in ("all", "lightning")

    # Header (one-time; not re-emitted in watch mode)
    if out is None:
        _say("Brian Task Manager — training monitor (vast.ai + Lightning AI)")
        _say("  Interactive: brian ps --it --interval 1")
        _say("  Platform:    brian ps --platform lightning")
        _say("  Logs:        brian ps --logs <job_id> [--tail 200] [--it]")
        _say("")

    # ── Collect from every requested platform ─────────────────────────
    rows: list = []
    status_msgs: list = []
    if show_vast:
        vast_rows, vast_err = _collect_vast_rows(args)
        rows.extend(vast_rows)
        if vast_err:
            status_msgs.append(vast_err)
    if show_lightning:
        rows.extend(_collect_lightning_rows(args))

    for msg in status_msgs:
        _say(msg)

    # ── Filter: show only active unless --all ─────────────────────────
    show_all = getattr(args, "all", False)
    display = rows if show_all else [r for r in rows if r.get("is_active", True)]

    # ── Unified table ──────────────────────────────────────────────────
    if display:
        hdr = (f"{'P':>1}  {'ID':>10}  {'LABEL':<28}  {'GPU':<12}  {'$/hr':>5}  "
               f"{'UP(m)':>6}  {'PHASE':<16}  {'STEP':>6}  {'PPL':>8}  "
               f"{'OOD-PPL':>9}  {'TOK/S':>7}")
        _say(hdr)
        _say("-" * len(hdr))
        for r in display:
            _say(_format_unified_row(r))
        _say("")
        _say("Legend: v=vast.ai  l=lightning  c=colab")
    else:
        hint = (" (pass --all to include stopped/completed instances)"
                if not show_all else "")
        _say(f"No active instances{hint}.")

    # ── Recent destroyed (vast history, shown regardless of filters) ──
    if show_vast:
        destroyed = _scan_recent_destroyed(top_n=5)
        if destroyed:
            _say()
            _say("Recent destroyed (vast):")
            dhdr = (f"  {'ID':>10}  {'WHEN':<19}  {'REASON':<15}  "
                    f"{'STEPS':>6}  {'OOD-PPL':>9}  DETAIL")
            _say(dhdr)
            _say("  " + "-" * (len(dhdr) - 2))
            for r in destroyed:
                steps = str(r["steps"]) if r["steps"] is not None else "-"
                ood = f"{r['ood_ppl']:.0f}" if r["ood_ppl"] is not None else "-"
                when = r["mtime"].strftime("%Y-%m-%d %H:%M:%S")
                _say(f"  {r['id']:>10}  {when:<19}  {r['exit_reason']:<15}  "
                     f"{steps:>6}  {ood:>9}  {r['detail'][:80]}")
    return 0


def _render_vast_section(args: argparse.Namespace, _say) -> None:
    """Render the vast.ai instance table (extracted from old _render_ps_once)."""
    import json
    _say("── vast.ai ──")
    vastai = _vastai_exe()
    raw, rc = _run_capture([vastai, "show", "instances", "--raw"])
    # Failure modes: offline (rc!=0, raw contains 'connection'/'resolve'/empty),
    # CLI error (rc!=0, raw has actual error), or success with DEPRECATED notice.
    data = []
    parse_ok = True
    offline_marker = any(s in raw.lower() for s in
                         ("connection", "failed to resolve", "could not resolve",
                          "timed out", "getaddrinfo", "no route to host"))
    if rc != 0 and "DEPRECATED" not in raw and not offline_marker:
        _say(f"vastai show failed: {raw[:300]}")
        parse_ok = False
    elif offline_marker or not raw.strip():
        _say("(offline — can't reach vast.ai; showing destroyed-instance "
             "history from logs/vast/*)")
        parse_ok = False
    else:
        start = raw.find("[")
        if start < 0:
            data = []
        else:
            depth = 0
            end = start
            in_str = False
            esc = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if in_str:
                    if esc: esc = False
                    elif ch == "\\": esc = True
                    elif ch == '"': in_str = False
                elif ch == '"': in_str = True
                elif ch == "[": depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError as e:
                _say(f"(can't parse vastai response: {e}; falling back to "
                     "destroyed-instance history)")
                parse_ok = False
    rows = []
    for inst in data:
        label = inst.get("label") or ""
        if args.all or label.startswith("neuroslm"):
            iid = inst.get("id")
            log, _ = _run_capture([vastai, "logs", str(iid)])
            status = _parse_status(log)
            rows.append({
                "id": iid,
                "label": label or "(no label)",
                "gpu": inst.get("gpu_name", "?"),
                "cost": inst.get("dph_total", 0),
                "uptime_mins": int(inst.get("uptime_mins", 0)
                                    if inst.get("uptime_mins") else 0),
                "status": inst.get("actual_status", "?"),
                **status,
            })
    if rows:
        hdr = (f"{'ID':>10}  {'LABEL':<28}  {'GPU':<12}  {'$/hr':>5}  "
               f"{'UP(m)':>6}  {'PHASE':<16}  {'STEP':>6}  {'PPL':>8}  "
               f"{'OOD-PPL':>9}  {'TOK/S':>7}")
        _say(hdr)
        _say("-" * len(hdr))
        for r in rows:
            step  = str(r["step"]) if r["step"] is not None else "-"
            ppl   = f"{r['ppl']:.1f}" if r["ppl"] is not None else "-"
            ood   = (f"{r['mid_ood_ppl']:.0f}@{r['mid_ood_step']}"
                     if r["mid_ood_ppl"] is not None else "-")
            tps   = f"{r['tps']/1000:.0f}k" if r["tps"] else "-"
            cost  = f"{r['cost']:.2f}" if r["cost"] else "-"
            _say(f"{str(r['id']):>10}  {r['label'][:28]:<28}  "
                 f"{r['gpu'][:12]:<12}  {cost:>5}  {r['uptime_mins']:>6}  "
                 f"{r['phase']:<16}  {step:>6}  {ppl:>8}  {ood:>9}  {tps:>7}")
    elif parse_ok:
        live_hint = ""
        if not args.all:
            live_hint = " (pass --all to list non-neuroslm instances too)"
        _say(f"(no live instances{live_hint})")


def _render_lightning_section(args: argparse.Namespace, _say) -> None:
    """Render the Lightning AI jobs table.

    Reads ``.brian/jobs/*.json``, attaches to each Studio, queries live
    status, and tails the last 2 lines of the remote training log so
    the table column shows current step/PPL even though Lightning has
    no native ``vastai show``-style poll.

    Errors short-circuit to a friendly message — never blow up
    ``brian ps`` because the Lightning SDK can't reach the cloud.
    """
    _say("── Lightning AI ──")
    try:
        from neuroslm.connectors import LightningConnector, JobStatus
    except Exception as e:
        _say(f"(lightning import failed: {type(e).__name__}: {e})")
        return
    try:
        connector = LightningConnector()
        jobs = connector.list_jobs()
    except Exception as e:
        _say(f"(lightning list_jobs failed: {type(e).__name__}: {e})")
        return

    if not jobs:
        _say("(no Lightning jobs registered — `brian deploy --platform "
             "lightning` writes them to .brian/jobs/)")
        return

    import time as _time
    # Match the vast.ai column layout exactly so both sections are visually
    # identical — GPU = machine tier, $/hr = "-" (Lightning billing differs),
    # PHASE = job status. TOK/S parsed from the log tail.
    hdr = (f"{'ID':>10}  {'LABEL':<28}  {'GPU':<12}  {'$/hr':>5}  "
           f"{'UP(m)':>6}  {'PHASE':<16}  {'STEP':>6}  {'PPL':>8}  "
           f"{'OOD-PPL':>9}  {'TOK/S':>7}")
    _say(hdr)
    _say("-" * len(hdr))
    now = int(_time.time())
    for j in jobs:
        up_m = max(0, (now - (j.started_at or now)) // 60)
        label  = (j.label   or "(none)")[:28]
        phase  = (j.status  or "?")[:16]
        gpu    = (j.machine or "?")[:12]
        # Tail the log for the most recent step/ppl/ood/tps. n=500 so the
        # parser catches the latest [mid-ood] line (fired every ood_every
        # steps, which with log_every=20 is ≥25 lines back).
        step_s, ppl_s, ood_s, tps_s = "-", "-", "-", "-"
        if j.status in (JobStatus.RUNNING.value, JobStatus.STARTING.value):
            try:
                tail = connector.tail_logs(j.job_id, n=500)
                step_s, ppl_s, ood_s, tps_s = _summarise_lightning_tail(tail)
            except Exception:
                pass
        _say(f"{str(j.job_id)[:10]:>10}  {label:<28}  {gpu:<12}  {'  -':>5}  "
             f"{up_m:>6}  {phase:<16}  {step_s:>6}  {ppl_s:>8}  "
             f"{ood_s:>9}  {tps_s:>7}")


def _summarise_lightning_tail(tail: str) -> tuple:
    """Extract ``(step, ppl, ood, tps)`` from a tail of train.log.

    Uses the same regexes as the vast parser so the per-step
    ``step N | loss … | ppl …`` line and the periodic
    ``[mid-ood] step N: wikitext ppl=…`` line both surface in the
    Lightning ps table with the same columns as the vast section.

    Returns a 4-tuple matching the vast row layout:
      (step_s, ppl_s, ood_s, tps_s)
    """
    if not tail:
        return ("-", "-", "-", "-")
    step, ppl, tps = None, None, None
    ood_step, ood_ppl = None, None
    try:
        for m in _STEP_RE.finditer(tail):
            step = int(m.group("step"))
            ppl  = float(m.group("ppl"))
            _t   = m.group("tps")
            if _t:
                tps = int(_t)
    except NameError:
        pass
    try:
        for m in _MID_OOD_RE.finditer(tail):
            ood_step = int(m.group("step"))
            ood_ppl  = float(m.group("ppl"))
    except NameError:
        pass
    ood_s = (f"{ood_ppl:.0f}@{ood_step}"
             if ood_ppl is not None and ood_step is not None else "-")
    tps_s = f"{tps/1000:.0f}k" if tps else "-"
    return (
        str(step) if step is not None else "-",
        f"{ppl:.1f}" if ppl is not None else "-",
        ood_s,
        tps_s,
    )


def _vastai_exe() -> str:
    """Locate the vastai executable. Prefers the project's .venv."""
    candidates = [
        REPO_ROOT / ".venv-2" / "Scripts" / "vastai.exe",
        REPO_ROOT / ".venv"   / "Scripts" / "vastai.exe",
        REPO_ROOT / ".venv-2" / "bin" / "vastai",
        REPO_ROOT / ".venv"   / "bin" / "vastai",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return "vastai"   # rely on PATH


def _run_capture(cmd) -> Tuple[str, int]:
    """Run a command and return (combined output, rc)."""
    r = subprocess.run(cmd, capture_output=True, text=True,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                        encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def cmd_destroy(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if args.all:
        # Use the deploy script's --destroy machinery which targets every
        # neuroslm-labelled instance.
        return _run([_bash(), "scripts/vast_deploy.sh", "--destroy"], env=env)
    if not args.instance_id:
        print("destroy: pass an instance id, or --all")
        return 2
    return _run([_bash(), "scripts/vast.sh",
                 "destroy", "instance", str(args.instance_id), "-y"], env=env)


def cmd_nuke(args: argparse.Namespace) -> int:
    """Destroy ALL running neuroslm-labelled instances immediately.

    Requires an interactive TTY and the user to type 'nuke'. This prevents
    AI agents from invoking it autonomously.
    """
    if not sys.stdin.isatty():
        print(
            "[nuke] BLOCKED: stdin is not an interactive terminal.\n"
            "  Nuke requires a human to confirm at a real TTY.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\n[nuke] This will DESTROY ALL running neuroslm-labelled vast.ai instances."
        "\n[nuke] There is no undo."
        '\n[nuke] Type "nuke" to confirm, anything else to abort: ',
        end="", flush=True,
    )
    try:
        answer = input()
    except (EOFError, KeyboardInterrupt):
        print("\n[nuke] Aborted.", file=sys.stderr)
        sys.exit(1)

    if answer.strip() != "nuke":
        print(
            f'[nuke] Expected "nuke", got "{answer.strip()}". Aborted.',
            file=sys.stderr,
        )
        sys.exit(1)

    return cmd_destroy(argparse.Namespace(all=True, instance_id=None))


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop a remote training job by job_id (Lightning AI).

    Reads ``.brian/jobs/<job_id>.json`` to discover the platform and
    dispatches to that connector's :meth:`stop`. The job record is
    removed on success.
    """
    from neuroslm.connectors import get_connector, load_job
    info = load_job(args.job_id)
    if info is None:
        print(f"brian stop: no job {args.job_id!r} in .brian/jobs/",
              file=sys.stderr)
        return 1
    try:
        connector = get_connector(info.platform)
    except ValueError as e:
        print(f"brian stop: {e}", file=sys.stderr)
        return 1
    try:
        return connector.stop(args.job_id)
    except NotImplementedError:
        print(f"brian stop: {info.platform!r} connector does not support "
              f"stop(). Use `brian destroy {args.job_id}` for vast.",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"brian stop: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


# ── ood ────────────────────────────────────────────────────────────────

def cmd_ood(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CKPT"] = args.ckpt
    if args.branch:
        env["BRANCH"] = args.branch
    if args.tag:
        env["ROLE_TAG"] = args.tag
    if args.windows:
        env["MAX_OOD_WINDOWS"] = str(args.windows)
    return _run([_bash(), "scripts/vast_ood_eval.sh"], env=env)


# ── best ───────────────────────────────────────────────────────────────

def cmd_best_update(args: argparse.Namespace) -> int:
    """Scan logs, find best run, write .brian/best_run.ln."""
    from neuroslm.log_refs import update_best_run_pointer, score_log
    repo_root = Path(".")
    log_dir = Path(getattr(args, "log_dir", "logs") or "logs")
    metric = getattr(args, "metric", "gap_ratio") or "gap_ratio"
    result = update_best_run_pointer(root=repo_root, log_dir=log_dir,
                                     metric=metric)
    if result is None:
        print(f"✗ No qualifying logs found under {log_dir}", file=sys.stderr)
        return 1
    print(f"best  →  {result}  (metric: {metric})")
    score = score_log(result)
    if score:
        _parts: list[str] = []
        if score.gap_ratio is not None:
            _parts.append(f"gap_ratio={score.gap_ratio:.2f}")
        if score.ood_ppl is not None:
            _parts.append(f"ood_ppl={score.ood_ppl:.1f}")
        if score.train_ppl is not None:
            _parts.append(f"train_ppl={score.train_ppl:.1f}")
        if score.step:
            _parts.append(f"step={score.step}")
        if _parts:
            print("  " + "  |  ".join(_parts))
    return 0


# ── analyze-log ────────────────────────────────────────────────────────

def cmd_analyze_log(args: argparse.Namespace) -> int:
    """Parse a training/OOD log → upsert docs/metrics.md row → append finding."""
    from neuroslm.cli_metrics import (
        analyze_log_file, scan_ood_dir, METRICS_PATH, FINDINGS_PATH,
        claude_available, OOD_DIR,
    )
    if args.scan_ood:
        rows = scan_ood_dir()
        print(f"scanned {len(rows)} OOD JSON files; updated {METRICS_PATH}")
        return 0
    if args.latest:
        # Discover the newest training log + matching OOD JSONs and
        # analyze them as a group.
        logs_dir = REPO_ROOT / "logs" / "vast"
        train_logs = sorted(logs_dir.glob("*__neuroslm-full.log"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not train_logs:
            print("analyze-log --latest: no training logs in logs/vast/",
                  file=sys.stderr)
            return 1
        latest = train_logs[0]
        rid = latest.stem.split("__")[0]
        print(f"=== latest training log: {latest.name} (run_id={rid}) ===")
        metric = analyze_log_file(latest, run_id=rid,
                                   branch=args.branch,
                                   use_claude=not args.no_claude)
        print("--- parsed metrics ---")
        print(metric.md_row())
        # Plus any matching OOD JSONs
        ood_matches = sorted(OOD_DIR.glob(f"ood_*{rid}*.json")) + \
                      sorted(OOD_DIR.glob(f"ood_*step*.json"))
        ood_matches = list({p: None for p in ood_matches})   # dedupe
        if ood_matches:
            print(f"\nfound {len(ood_matches)} OOD JSON(s) — scanning...")
            rows = scan_ood_dir()
            print(f"upserted {len(rows)} OOD rows in {METRICS_PATH}")
        print(f"\nmetrics ledger: {METRICS_PATH}")
        if not args.no_claude:
            print(f"findings: {FINDINGS_PATH}"
                  if claude_available() else
                  "(claude CLI not on PATH — narrative insights skipped)")
        return 0
    if not args.logfile:
        print("analyze-log: pass a logfile path, --latest, or --scan-ood",
              file=sys.stderr)
        return 2
    p = Path(args.logfile)
    metric = analyze_log_file(p, run_id=args.run_id, branch=args.branch,
                              use_claude=not args.no_claude)
    print("--- parsed metrics ---")
    print(metric.md_row())
    print()
    print(f"upserted row in {METRICS_PATH}")
    if not args.no_claude:
        if claude_available():
            print(f"appended insight to {FINDINGS_PATH}")
        else:
            print("(claude CLI not found on PATH — skipped narrative insights)")
    return 0


# ── eval ───────────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> int:
    """Group command: `brian eval ood` etc."""
    if args.eval_kind == "ood":
        return _eval_ood(args)
    print(f"unknown eval kind: {args.eval_kind}")
    return 2


def _find_dsl_checkpoints() -> List[Tuple[int, str]]:
    """Return [(step, path), ...] sorted desc by step. Local first, then origin.

    Matches both filename schemes:
      - legacy:        dsl_arch_step<N>.pt
      - timestamped:   dsl_arch_<YYYYMMDD-HHMMSS>_step<N>.pt
    """
    ckpt_dir = REPO_ROOT / "lfs_checkpoints"
    items: List[Tuple[int, str]] = []
    seen: set = set()
    if ckpt_dir.is_dir():
        # Both `dsl_arch_step*.pt` and `dsl_arch_*_step*.pt` patterns.
        for p in list(ckpt_dir.glob("dsl_arch_*step*.pt")) + \
                  list(ckpt_dir.glob("dsl_arch_step*.pt")):
            m = re.search(r"_step(\d+)\.pt$", p.name)
            if m:
                rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                if rel not in seen:
                    seen.add(rel)
                    items.append((int(m.group(1)), rel))
    # Also list what's on origin (silent if origin/<branch> not fetched yet)
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
        out = subprocess.check_output(
            ["git", "ls-tree", f"origin/{branch}",
             "lfs_checkpoints/", "--name-only"],
            cwd=str(REPO_ROOT), text=True,
            stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "dsl_arch" not in line:
                continue
            m = re.search(r"_step(\d+)\.pt$", line)
            if not m:
                continue
            if line not in seen:
                seen.add(line)
                items.append((int(m.group(1)), line))
    except Exception:
        pass
    items.sort(key=lambda r: -r[0])
    return items


def _eval_ood(args: argparse.Namespace) -> int:
    """Deploy a vast.ai OOD-eval instance for a DSL checkpoint.

    Checkpoint selection (in priority order):
      1. positional path:          `brian eval ood path/to.pt`
      2. --checkpoint PATH
      3. --latest                  → highest-step dsl_arch_*.pt
      4. interactive picker        (default if none of the above)
    """
    ckpts = _find_dsl_checkpoints()
    if not ckpts:
        print("no DSL checkpoints found in lfs_checkpoints/ or on origin")
        return 1
    # Resolve the checkpoint.
    ckpt_path = None
    # Positional arg: `brian eval ood foo.pt` — and "--latest" allowed as the
    # positional placeholder too, so `brian eval ood --latest` works.
    pos = getattr(args, "ckpt_pos", None)
    if pos and pos != "--latest":
        ckpt_path = pos
    if ckpt_path is None and getattr(args, "checkpoint", None):
        ckpt_path = args.checkpoint
    if ckpt_path is None and (getattr(args, "latest", False) or pos == "--latest"):
        ckpt_path = ckpts[0][1]
        print(f"--latest → step {ckpts[0][0]}: {ckpt_path}")
    if ckpt_path is None:
        # Interactive picker (default to highest-step on Enter).
        print("=== DSL checkpoints (newest first) ===")
        for i, (step, path) in enumerate(ckpts):
            tag = " (default)" if i == 0 else ""
            print(f"  [{i+1:2d}] step {step:>6d}  {path}{tag}")
        sel = input(f"choose [1-{len(ckpts)}] (Enter for 1): ").strip()
        if not sel:
            sel = "1"
        try:
            idx = int(sel) - 1
            ckpt_path = ckpts[idx][1]
        except (ValueError, IndexError):
            print(f"invalid selection: {sel!r}")
            return 2
    # Verify the checkpoint exists locally or in the origin tree before
    # we spin up a vast.ai instance for nothing.
    p = Path(ckpt_path)
    if not p.is_absolute() and not (REPO_ROOT / ckpt_path).exists():
        # Check if it's on origin via git ls-tree (matches _find_dsl_checkpoints).
        known = any(c[1] == ckpt_path for c in ckpts)
        if not known:
            print(f"checkpoint not found locally or on origin: {ckpt_path}")
            print("known checkpoints (newest first):")
            for step, path in ckpts[:10]:
                print(f"  step {step:>6d}  {path}")
            return 1

    print(f"\n=== Deploying OOD eval ===")
    print(f"  checkpoint = {ckpt_path}")
    branch = args.branch or subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(REPO_ROOT), text=True).strip()
    print(f"  branch     = {branch}")
    tag = args.tag or f"dsl-{Path(ckpt_path).stem.replace('dsl_arch_', '')}"
    print(f"  role tag   = {tag}")

    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "BRANCH": branch,
        "CKPT": ckpt_path,
        "ROLE_TAG": tag,
    })
    if args.windows:
        env["MAX_OOD_WINDOWS"] = str(args.windows)
    # Python-based deploy (mirrors deploy/train_dsl.py) — bypasses the
    # bash script which can stall for minutes on Windows due to
    # vastai-import detection + pip-install in the wrong python.
    deploy_py = REPO_ROOT / "deploy" / "ood_eval.py"
    if not deploy_py.is_file():
        # Legacy fallback for older clones
        legacy = REPO_ROOT / "_deploy_ood.py"
        if legacy.is_file():
            deploy_py = legacy
        else:
            return _run([_bash(), "scripts/vast_ood_eval.sh"], env=env)
    return _run([sys.executable, str(deploy_py), ckpt_path], env=env)


# ── test / push ────────────────────────────────────────────────────────

_AGENT_SKILLS_DIR = "agents/skills"


def _list_ai_skills() -> List[str]:
    """Return every agents/skills/<name>/ with an INSTRUCTIONS.md."""
    skills_dir = REPO_ROOT / _AGENT_SKILLS_DIR
    if not skills_dir.is_dir():
        return []
    out = []
    for entry in sorted(skills_dir.iterdir()):
        if entry.is_dir() and (entry / "INSTRUCTIONS.md").is_file():
            out.append(entry.name)
    return out


def cmd_ai(args: argparse.Namespace) -> int:
    """Run a Claude-Code-backed skill: `brian ai <name> [--auto]`.

    Skills are folders under `agents/skills/<name>/INSTRUCTIONS.md`.
    The INSTRUCTIONS.md is passed as the prompt to `claude -p` with
    the repo root as cwd, so the skill can read any file in the repo.
    `--auto` adds `--dangerously-skip-permissions` for unattended runs.
    """
    skill_name = args.ai_kind
    available = _list_ai_skills()
    if skill_name not in available:
        print(f"unknown skill: {skill_name!r}. "
              f"Available: {', '.join(available) if available else '(none)'}",
              file=sys.stderr)
        return 2
    claude_path = shutil.which("claude")
    if claude_path is None:
        print("claude CLI not on PATH. Install it from "
              "https://github.com/anthropics/claude-code first.",
              file=sys.stderr)
        return 1
    instructions = REPO_ROOT / _AGENT_SKILLS_DIR / skill_name / "INSTRUCTIONS.md"
    body = instructions.read_text(encoding="utf-8")
    # Without an explicit imperative wrapper, `claude -p <long-markdown>`
    # treats the markdown as descriptive context and responds with a
    # conversational greeting ("What would you like me to do?"). Wrap
    # the skill body so claude knows to EXECUTE it rather than DISCUSS it.
    prompt_text = (
        f"Execute the following skill from "
        f"{instructions.relative_to(REPO_ROOT)} now. "
        f"Start working immediately, do not ask for confirmation, do not "
        f"summarise the skill back to me — perform the file edits the "
        f"skill describes. When the skill instructs you to stop, stop and "
        f"print the summary block.\n\n"
        f"--- SKILL: {skill_name} ---\n\n"
        f"{body}\n\n"
        f"--- BEGIN NOW ---"
    )
    print(f"=== brian ai {skill_name} — invoking claude ===")
    print(f"  skill:   {instructions.relative_to(REPO_ROOT)}")
    print(f"  repo:    {REPO_ROOT}")
    print()
    cli = [claude_path, "-p", prompt_text]
    if args.auto:
        cli.insert(1, "--dangerously-skip-permissions")
    return subprocess.call(
        cli, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})


# ─── Effective-preset resolution (CLI > arch > hardware-map > global > AUTO) ──

def _resolve_effective_preset(cli_preset, arch_cfg, project_cfg):
    """Fold the precedence layers into one decision.

    **2026-06-12 precedence (high → low):**

      1. CLI flag                    (``--preset``)
      2. Arch.neuro                  (any non-empty ``preset:`` value)
      3. ``[hardware.<DETECTED>]``   per-hardware map in ``brian.toml``
      4. ``[defaults] preset``       workspace-wide fallback
      5. AUTO                        detect hardware → look up map → else
                                     ``pick_preset_for_vram(detected_gib)``

    Hardware "detection" uses ``project_cfg.default_hardware`` when set,
    otherwise asks :func:`neuroslm.hardware.detect_hardware`.

    Returns
    -------
    (effective_preset, change_log) : (Optional[str], List[Tuple[str, str, str]])
        ``effective_preset`` is ``None`` only when every layer is empty
        (no CUDA, no config) so the caller can fall back to its own
        hardcoded default. ``change_log`` carries any merges done by
        :func:`apply_global_defaults` (empty when no merge ran).
    """
    # Layer 1: CLI flag wins outright.
    if cli_preset:
        return cli_preset, []

    # Layer 2: Arch.neuro non-empty value wins over everything below.
    arch_preset = (getattr(arch_cfg, "preset", "") or "").strip()
    if arch_preset:
        return arch_preset, []

    # Resolve the active hardware key once — used by layers 3 and 5.
    from neuroslm import hardware as _hw
    active_hw = (
        (getattr(project_cfg, "default_hardware", "") or "").strip()
        or _hw.detect_hardware()
    )

    # Layer 3: per-hardware map in brian.toml.
    hw_map = getattr(project_cfg, "hardware_presets", {}) or {}
    if active_hw and active_hw in hw_map and hw_map[active_hw]:
        return hw_map[active_hw], []

    # Layer 4: workspace-wide [defaults] preset (apply_global_defaults
    # writes it onto arch_cfg.preset so subsequent reads see it).
    from neuroslm.dsl.training_config import apply_global_defaults
    changes = apply_global_defaults(arch_cfg, project_cfg)
    eff = (getattr(arch_cfg, "preset", "") or "").strip()
    if eff:
        return eff, changes

    # Layer 5: AUTO — VRAM-based fallback.
    if active_hw == "CPU":
        return "tiny", changes
    return _hw.pick_preset_for_vram(_hw.detect_vram_gib()), changes


def _resolve_effective_steps(cli_steps, arch_cfg, project_cfg):
    """Mirror of :func:`_resolve_effective_preset` for the step count.

    Precedence: CLI > arch.neuro > ``brian.toml [defaults] steps``.
    Returns ``None`` when every layer is silent (caller picks default).
    """
    if cli_steps:
        return int(cli_steps)
    arch_steps = int(getattr(arch_cfg, "steps", 0) or 0)
    if arch_steps > 0:
        return arch_steps
    glb_steps = int(getattr(project_cfg, "default_steps", 0) or 0)
    if glb_steps > 0:
        return glb_steps
    return None


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model with configurable settings.

    Usage:
        brian train --preset=tiny              # Train tiny model for 40k steps on CPU
        brian train --preset=tiny --steps=100  # Train tiny model for 100 steps
        brian train --arch=rcc_bowtie --steps=10000
        brian train --dna=dna/evol/arch.dna    # Load from DNA with fitness config

    When ``--dna`` or ``--arch`` is given, the source is unpacked into
    ``./.neuro/arch/temp/`` (canonical run workspace) and the
    Hypergraph IR is compiled from there. This keeps every consumer
    (harness, NFG, evolution overlays) reading from a single,
    predictable location and avoids accidental writes to the source
    architecture tree. See ``neuroslm/compiler/run_workspace.py``.
    """
    # ── Unpack DNA/arch into .neuro/arch/temp/ (single source of truth) ──
    # When either flag is set (and we're NOT on the `tiny` preset path
    # which handles DNA loading itself via colab_train_minimal_cpu),
    # we centralise the unfolding so the run ALWAYS reads from the
    # workspace — never from the source tree — and so the lifted
    # HypergraphIR is built from exactly the bytes the run will use.
    # The workspace is rebuilt on every invocation (no stale modules
    # leaking between runs).
    #
    # Tiny preset is special-cased because it dispatches to
    # ``colab_train_minimal_cpu.main(dna_path=...)`` which runs its own
    # ``init_evolution(dna_path)`` — pre-unpacking the workspace from
    # the CLI would force the DNA file to exist on disk just to satisfy
    # the workspace prep step, breaking the mock-based unit tests in
    # ``tests/test_fitness_load_or_default.py::TestTinyCliDnaRouting``
    # and forcing every Colab/local CPU run to pre-cache a DNA file
    # that the inner path was perfectly capable of fetching itself.
    workspace = None
    _skip_workspace_prep = (args.preset == "tiny")
    if not _skip_workspace_prep and (
        args.dna or (args.arch and not args.arch.endswith('.dna'))
    ):
        try:
            from neuroslm.compiler.run_workspace import prepare_run_workspace
            workspace = prepare_run_workspace(
                dna=args.dna,
                arch=None if args.dna else args.arch,
            )
            print(f"[train] workspace ready: {workspace.arch_root}")
            print(f"        source: {workspace.source_kind}={workspace.source_path}")
            print(f"        hypergraph IR: {len(workspace.hypergraph_ir.nodes)} nodes, "
                  f"{len(workspace.hypergraph_ir.hyperedges)} edges")
        except Exception as e:
            print(f"[train] workspace preparation failed: {e}", file=sys.stderr)
            return 1
    elif (not _skip_workspace_prep) and args.arch and args.arch.endswith('.dna'):
        # Back-compat: `--arch=foo.dna` is treated as `--dna=foo.dna`.
        try:
            from neuroslm.compiler.run_workspace import prepare_run_workspace
            workspace = prepare_run_workspace(dna=args.arch)
            print(f"[train] workspace ready (via --arch=.dna): "
                  f"{workspace.arch_root}")
        except Exception as e:
            print(f"[train] workspace preparation failed: {e}", file=sys.stderr)
            return 1

    # If tiny preset requested, run minimal training
    if args.preset == "tiny":
        import importlib.util
        original_cwd = os.getcwd()
        try:
            os.chdir(REPO_ROOT)

            dna_path = None

            # Explicit --dna takes precedence.
            if args.dna:
                dna_path = args.dna
                print(f"[train] Loading from DNA: {dna_path}")

            # Support --arch pointing to .dna files
            if args.arch and args.arch.endswith('.dna'):
                if dna_path is None:
                    dna_path = args.arch
                print(f"[train] Loading from DNA: {args.arch}")

            # Import and run the minimal training
            spec = importlib.util.spec_from_file_location(
                "colab_train_minimal_cpu",
                REPO_ROOT / "colab_train_minimal_cpu.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run main with configurable steps and OOD eval
            steps = args.steps if args.steps else 40000
            ood_every = args.ood_every if args.ood_every else 500
            module.main(steps=steps, ood_every=ood_every,
                        dna_path=dna_path or "dna/evol/arch.dna")
            return 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1
        finally:
            os.chdir(original_cwd)

    # Build training command
    cmd = [sys.executable, "-m", "neuroslm.train_dsl"]

    # Workspace-based dispatch — both DNA and arch modes use the
    # unpacked .neuro/arch/temp/ tree.
    if workspace is not None:
        cmd.extend(["--arch", str(workspace.arch_root)])
    elif args.arch:
        # arch path that didn't go through the workspace (e.g. user
        # explicitly pointed at an existing flat folder) — keep legacy.
        arch = _resolve_arch(args.arch)
        cmd.extend(["--arch", arch])
    else:
        # Default: the active working-copy arch. Renamed from
        # architectures/rcc_bowtie on 2026-06-14; master/ holds the
        # canonical source-of-truth, current/ is the live arch the
        # trainer consumes by default.
        cmd.extend(["--arch", "architectures/master"])

    # ── Resolve effective preset + steps (CLI > arch > global > AUTO) ──
    # Load the arch's parsed training config + the workspace-level
    # brian.toml defaults. The CLI flag is the top-priority lever;
    # otherwise the arch wins over the global, which wins over the
    # auto-detected hardware bucket. See docs/CLI.md § "Global defaults"
    # and tests/test_hardware_aware_preset_selection.py for the spec.
    arch_cfg = None
    project_cfg = None
    try:
        from neuroslm.dsl.training_config import (
            load_training_config_from_arch,
        )
        from neuroslm.project_config import load_project_config
        arch_root_for_cfg = (
            workspace.arch_root if workspace is not None
            else (REPO_ROOT / (args.arch or "architectures/master"))
        )
        arch_cfg = load_training_config_from_arch(arch_root_for_cfg)
        project_cfg = load_project_config()
        effective_preset, preset_changes = _resolve_effective_preset(
            cli_preset=args.preset,
            arch_cfg=arch_cfg,
            project_cfg=project_cfg,
        )
        for field_name, old, new in preset_changes:
            print(f"[global] {field_name}: {old!r} → {new!r}  "
                  f"(from brian.toml [defaults])")
        # Announce the hardware/preset decision so deploys are traceable.
        from neuroslm import hardware as _hw
        active_hw = (project_cfg.default_hardware or _hw.detect_hardware())
        if not args.preset and not (arch_cfg and arch_cfg.preset):
            print(f"[train] hardware: {active_hw}  "
                  f"→ preset: {effective_preset!r}")
    except Exception as e:
        # Resolution must never break training. Fall back to CLI or
        # built-in default and warn the user.
        print(f"[train] preset resolution failed ({e}); "
              f"falling back to CLI/default", file=sys.stderr)
        effective_preset = args.preset

    if effective_preset:
        cmd.extend(["--preset", effective_preset])
    else:
        cmd.extend(["--preset", "rcc_bowtie_30m_p4"])

    # Resolve effective step count with the same precedence.
    effective_steps = args.steps
    if arch_cfg is not None and project_cfg is not None:
        try:
            effective_steps = _resolve_effective_steps(
                cli_steps=args.steps,
                arch_cfg=arch_cfg,
                project_cfg=project_cfg,
            )
        except Exception as e:
            print(f"[train] step resolution failed ({e}); "
                  f"using CLI value", file=sys.stderr)
            effective_steps = args.steps

    if effective_steps:
        cmd.extend(["--steps", str(effective_steps)])

    if args.batch:
        cmd.extend(["--batch", str(args.batch)])

    if args.seq_len:
        cmd.extend(["--seq_len", str(args.seq_len)])

    if args.d_sem:
        cmd.extend(["--d_sem", str(args.d_sem)])

    # Add standard flags
    cmd.extend(["--data", "real", "--device", "cpu", "--resume"])

    print(f"[train] {' '.join(cmd)}")
    return _run(cmd)


# ── hf : list / pull / latest checkpoints from HuggingFace Hub ────────

def cmd_hf(args: argparse.Namespace) -> int:
    """Top-level dispatcher for ``brian hf <subcommand>``.

    The HF subcommands wrap :mod:`neuroslm.hf_checkpoints` with
    a one-line CLI surface:

      brian hf list   [--repo R] [--prefix P] [--limit N]
      brian hf pull   <path-in-repo|hf://uri>  [--repo R] [--out DIR]
      brian hf latest [--repo R] [--prefix P]                # PRINT only
      brian hf pull --latest [--repo R] [--prefix P] [--out DIR]

    The default repo follows the same chain as the push side:
    ``--repo`` arg → ``HF_REPO_ID`` env →
    :data:`neuroslm.checkpoint_push._DEFAULT_HF_REPO_ID`.
    """
    sub = getattr(args, "hf_cmd", None)
    if sub == "list":
        return _hf_list(args)
    if sub == "pull":
        return _hf_pull(args)
    if sub == "latest":
        return _hf_latest(args)
    print(f"[hf] unknown subcommand: {sub!r}", file=sys.stderr)
    return 2


def _hf_list(args: argparse.Namespace) -> int:
    """``brian hf list`` — print every checkpoint on the repo."""
    from neuroslm.hf_checkpoints import list_repo_checkpoints
    entries = list_repo_checkpoints(
        repo_id=args.repo, prefix=args.prefix or "")
    if not entries:
        print("(no checkpoints found)")
        return 1
    limit = args.limit or len(entries)
    print(f"  {'step':>10}  {'sidecar':<8}  path_in_repo")
    print(f"  {'-' * 10}  {'-' * 8}  {'-' * 60}")
    for e in entries[:limit]:
        sidecar = "yes" if e.has_mem_sidecar else "no"
        print(f"  {e.step:>10}  {sidecar:<8}  {e.path_in_repo}")
    print(f"\n  total: {len(entries)} (showing {min(limit, len(entries))})")
    return 0


def _hf_pull(args: argparse.Namespace) -> int:
    """``brian hf pull <path>`` (or ``--latest``) — download into
    ``lfs_checkpoints/`` (or ``--out``)."""
    from neuroslm.hf_checkpoints import (
        download_checkpoint, find_latest_checkpoint, parse_hf_uri,
    )
    target = args.target
    repo = args.repo
    if args.latest:
        latest = find_latest_checkpoint(
            repo_id=repo, prefix=args.prefix or "")
        if latest is None:
            print("[hf pull] no checkpoints found on the repo",
                  file=sys.stderr)
            return 1
        target = latest.path_in_repo
        print(f"[hf pull] --latest resolved to step {latest.step} "
              f"({target})")
    if not target:
        print("[hf pull] either provide a path-in-repo positional or "
              "use --latest", file=sys.stderr)
        return 2
    # Accept ``hf://owner/repo/path`` shorthand
    if target.startswith("hf://"):
        try:
            uri_repo, uri_path = parse_hf_uri(target)
        except ValueError as e:
            print(f"[hf pull] {e}", file=sys.stderr)
            return 2
        repo = repo or uri_repo
        target = uri_path
    out = download_checkpoint(
        target, repo_id=repo,
        dest_dir=args.out, force_download=bool(args.force))
    return 0 if out is not None else 1


def _hf_latest(args: argparse.Namespace) -> int:
    """``brian hf latest`` — print the highest-step checkpoint URI."""
    from neuroslm.hf_checkpoints import find_latest_checkpoint
    latest = find_latest_checkpoint(
        repo_id=args.repo, prefix=args.prefix or "")
    if latest is None:
        print("(no checkpoints found)")
        return 1
    repo = args.repo or os.environ.get("HF_REPO_ID", "") \
        or "moritzroessler/BRIAN"
    print(f"step       : {latest.step}")
    print(f"run_dir    : {latest.run_dir or '(flat)'}")
    print(f"sidecar    : {'yes' if latest.has_mem_sidecar else 'no'}")
    print(f"path       : {latest.path_in_repo}")
    print(f"hf_uri     : hf://{repo}/{latest.path_in_repo}")
    return 0


# ── checkpoints : local checkpoint manager ─────────────────────────


def _read_neuro_checkpoint_ln() -> Optional[str]:
    """Return the contents of .neuro/checkpoint.ln, or None if absent/empty."""
    ln_file = REPO_ROOT / ".neuro" / "checkpoint.ln"
    if not ln_file.is_file():
        return None
    text = ln_file.read_text(encoding="utf-8").strip()
    return text or None


def _write_neuro_checkpoint_ln(path: Path) -> None:
    """Write a repo-relative path to .neuro/checkpoint.ln."""
    ln_file = REPO_ROOT / ".neuro" / "checkpoint.ln"
    ln_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = path.relative_to(REPO_ROOT)
        ln_file.write_text(str(rel).replace("\\", "/"), encoding="utf-8")
    except ValueError:
        ln_file.write_text(str(path), encoding="utf-8")


def _format_ckpt_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "—"
    mb = size_bytes / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _ckpts_list(args: argparse.Namespace) -> int:
    try:
        from neuroslm.utils.secrets import bootstrap_secrets
        bootstrap_secrets(["HF_TOKEN"])
    except Exception:
        pass
    from neuroslm.hf_checkpoints import list_repo_checkpoints
    entries = list_repo_checkpoints(
        repo_id=args.repo, prefix=args.prefix or "")
    if not entries:
        print("(no checkpoints found)")
        return 1
    limit = args.limit or len(entries)

    active_path = _read_neuro_checkpoint_ln() or ""

    def _find_local(entry) -> Optional[Path]:
        for root in ("checkpoints", "lfs_checkpoints"):
            if entry.run_dir:
                p = REPO_ROOT / root / entry.run_dir / f"step{entry.step}.pt"
                if p.is_file():
                    return p
        return None

    repo = (args.repo or os.environ.get("HF_REPO_ID", "")
            or "moritzroessler/BRIAN")
    print(f"\n  brian checkpoints  ({repo})  — {len(entries)} total\n")

    col_w = {"step": 8, "params": 8, "ppl": 7, "ood": 8, "hash": 12,
              "size": 7, "sidecar": 7}
    header = (f"  {'#':>3}  {'Step':>{col_w['step']}}  "
              f"{'Params':>{col_w['params']}}  {'PPL':>{col_w['ppl']}}  "
              f"{'OOD PPL':>{col_w['ood']}}  {'Hash':>{col_w['hash']}}  "
              f"{'Size':>{col_w['size']}}  {'Sidecar':>{col_w['sidecar']}}  "
              f"Run Dir")
    sep = "  " + "─" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    for i, entry in enumerate(entries[:limit], 1):
        local = _find_local(entry)
        params_s = ppl_s = ood_s = hash_s = "—"
        if local:
            try:
                from neuroslm.hf_checkpoints import inspect_checkpoint_metadata
                meta = inspect_checkpoint_metadata(local)
                p = meta.get("params") or 0
                if p:
                    params_s = f"{p / 1e6:.1f}M"
                if meta.get("ppl"):
                    ppl_s = f"{meta['ppl']:.1f}"
                if meta.get("ood_ppl"):
                    ood_s = f"{meta['ood_ppl']:.1f}"
                if meta.get("model_hash"):
                    hash_s = meta["model_hash"]
            except Exception:
                pass

        size_s = _format_ckpt_size(entry.size)
        sidecar_s = "yes" if entry.has_mem_sidecar else "no"
        run_dir = entry.run_dir or "(flat)"

        # Determine active marker
        norm = active_path.replace("\\", "/")
        rel_variants = {
            f"checkpoints/{entry.run_dir}/step{entry.step}.pt",
            f"lfs_checkpoints/{entry.run_dir}/step{entry.step}.pt",
        }
        marker = " ●" if norm in rel_variants else "  "

        print(f"  {i:>3}  {entry.step:>{col_w['step']}}  "
              f"{params_s:>{col_w['params']}}  {ppl_s:>{col_w['ppl']}}  "
              f"{ood_s:>{col_w['ood']}}  {hash_s:>{col_w['hash']}}  "
              f"{size_s:>{col_w['size']}}  {sidecar_s:>{col_w['sidecar']}}  "
              f"{run_dir}{marker}")

    print(sep)
    if active_path:
        print(f"\n  ● = active checkpoint  ({active_path})")
    print()
    return 0


def _ckpts_download(args: argparse.Namespace) -> int:
    try:
        from neuroslm.utils.secrets import bootstrap_secrets
        bootstrap_secrets(["HF_TOKEN"])
    except Exception:
        pass
    from neuroslm.hf_checkpoints import (
        download_checkpoint, find_latest_checkpoint,
        list_repo_checkpoints, parse_hf_uri,
    )

    repo = args.repo
    dest_dir = REPO_ROOT / "checkpoints"

    if args.latest:
        latest = find_latest_checkpoint(
            repo_id=repo, prefix=args.prefix or "")
        if latest is None:
            print("[checkpoints] no checkpoints found on the repo",
                  file=sys.stderr)
            return 1
        path_in_repo = latest.path_in_repo
        print(f"[checkpoints] --latest → step {latest.step} "
              f"({path_in_repo})")
    elif args.target:
        target = args.target
        if target.startswith("hf://"):
            try:
                uri_repo, uri_path = parse_hf_uri(target)
                repo = repo or uri_repo
                path_in_repo = uri_path
            except ValueError as e:
                print(f"[checkpoints] {e}", file=sys.stderr)
                return 2
        elif target.isdigit():
            step = int(target)
            entries = list_repo_checkpoints(
                repo_id=repo, prefix=args.prefix or "")
            match = next((e for e in entries if e.step == step), None)
            if match is None:
                print(f"[checkpoints] no checkpoint at step {step}",
                      file=sys.stderr)
                return 1
            path_in_repo = match.path_in_repo
        else:
            path_in_repo = target
    else:
        print("[checkpoints] provide a target (step, path, hf:// URI) "
              "or use --latest", file=sys.stderr)
        return 2

    dest_dir.mkdir(parents=True, exist_ok=True)
    local = download_checkpoint(
        path_in_repo, repo_id=repo, dest_dir=str(dest_dir))
    if local is None:
        return 1

    print(f"[checkpoints] downloaded → {local}")

    if getattr(args, "activate", True):
        _write_neuro_checkpoint_ln(local)
        print(f"[checkpoints] active checkpoint set to: {local}")

    return 0


def _ckpts_use(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        print(f"[checkpoints] file not found: {path}", file=sys.stderr)
        return 2
    _write_neuro_checkpoint_ln(path)
    print(f"[checkpoints] active checkpoint set to: {path}")
    return 0


def _ckpts_active(args: argparse.Namespace) -> int:
    active = _read_neuro_checkpoint_ln()
    if not active:
        print("(no active checkpoint — run `brian checkpoints download "
              "--latest` to set one)")
        return 1
    print(f"active: {active}")
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    """Top-level dispatcher for ``brian checkpoints <subcommand>``."""
    sub = getattr(args, "ckpts_cmd", None)
    if sub == "list":
        return _ckpts_list(args)
    if sub == "download":
        return _ckpts_download(args)
    if sub == "use":
        return _ckpts_use(args)
    if sub == "active":
        return _ckpts_active(args)
    print(f"[checkpoints] unknown subcommand: {sub!r}", file=sys.stderr)
    return 2


# ── chat : always-on inference daemon with dashboard ──────────────────

def cmd_chat(args: argparse.Namespace) -> int:
    """``brian chat`` — boot a checkpoint and run the always-on dashboard.

    Resolution order for the checkpoint (top wins):

      1. ``--pt PATH_OR_URI``           — explicit named flag (alias for positional).
      2. positional ``ckpt`` arg.
      3. ``--latest`` flag               — pull HF Hub highest-step (by raw step).
      3.5. ``.neuro/checkpoint.ln``      — local active checkpoint set by
                                            ``brian checkpoints download/use``.
      4. ``.brian/checkpoint.ln``        — auto-pulled best-run checkpoint
                                            (DEFAULT; written by ``brian best update``).
      5. local ``lfs_checkpoints/`` highest-step (fallback for offline use).

    ``--no-best`` disables hop 4 — use it when you want to chat with whatever
    is on disk regardless of remote state (laptop offline, slow network).

    The default behaviour (no flags, no positional) downloads the *best*
    HF checkpoint per ``.brian/checkpoint.ln`` so ``brian chat`` "just
    works" right after a fresh training run lands its first push.
    """
    # Hop 1 + 2: explicit overrides — ``--pt`` wins over positional, both
    # wins over every implicit lookup below. Accept hf:// URIs too so the
    # user can paste anything ``brian hf latest`` printed.
    ckpt: Optional[str] = getattr(args, "pt", None) or args.ckpt
    if ckpt and ckpt.startswith("hf://"):
        from neuroslm.hf_checkpoints import (
            parse_hf_uri, download_checkpoint,
        )
        try:
            repo_id, path_in_repo = parse_hf_uri(ckpt)
        except ValueError as e:
            print(f"[chat] invalid hf URI: {e}", file=sys.stderr)
            return 2
        print(f"[chat] pulling {ckpt}")
        local = download_checkpoint(path_in_repo, repo_id=repo_id)
        ckpt = str(local) if local is not None else None

    # Hop 3: --latest (explicit HF lookup by raw step, NOT by best-run score)
    if not ckpt and args.latest:
        from neuroslm.hf_checkpoints import (
            find_latest_checkpoint, download_checkpoint,
        )
        latest = find_latest_checkpoint(
            repo_id=args.repo, prefix=args.prefix or "")
        if latest is None:
            print("[chat] no remote checkpoint found; falling back to "
                  "local lfs_checkpoints/", file=sys.stderr)
        else:
            print(f"[chat] pulling latest step {latest.step} "
                  f"({latest.path_in_repo})")
            local = download_checkpoint(
                latest.path_in_repo, repo_id=args.repo)
            if local is not None:
                ckpt = str(local)

    # Hop 3.5: .neuro/checkpoint.ln — local active checkpoint
    # (written by `brian checkpoints download [--activate]` or `brian
    # checkpoints use`). Uses the file directly — no HF download required.
    if not ckpt:
        local_path_str = _read_neuro_checkpoint_ln()
        if local_path_str:
            local_p = Path(local_path_str)
            if not local_p.is_absolute():
                local_p = REPO_ROOT / local_p
            if local_p.is_file():
                print(f"[chat] using active local checkpoint: {local_p}")
                ckpt = str(local_p)

    # Hop 4: best-run pointer (DEFAULT when no explicit flag was given).
    # Skipped if --no-best is set or if --latest already resolved a ckpt.
    if not ckpt and not getattr(args, "no_best", False):
        from neuroslm.log_refs import read_checkpoint_url
        hf_url = read_checkpoint_url(REPO_ROOT)
        if hf_url:
            from neuroslm.hf_checkpoints import (
                parse_hf_uri, download_checkpoint,
            )
            try:
                repo_id, path_in_repo = parse_hf_uri(hf_url)
                print(f"[chat] using best-run checkpoint: {hf_url}")
                local = download_checkpoint(
                    path_in_repo, repo_id=repo_id)
                if local is not None:
                    ckpt = str(local)
            except ValueError as e:
                print(f"[chat] .brian/checkpoint.ln has invalid URI: {e}; "
                      f"falling back to local", file=sys.stderr)

    # Hop 5: local fallback — highest-step .pt under lfs_checkpoints/
    if not ckpt:
        ckpt = _pick_local_latest_ckpt()

    if not ckpt:
        print("[chat] no checkpoint found. Provide a path with --pt, "
              "use --latest, run `brian best update` to refresh the "
              "best-run pointer, or train one first.", file=sys.stderr)
        return 2
    if not Path(ckpt).is_file():
        print(f"[chat] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    from neuroslm.chat_daemon import run_chat_daemon
    return run_chat_daemon(
        ckpt_path=ckpt,
        arch_root=args.arch,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        thought_n_tok=args.thought_tokens,
        thought_period=args.thought_period,
        idle_threshold=args.idle_threshold,
        no_color=bool(args.no_color),
        no_thoughts=bool(args.no_thoughts),
    )


def _pick_local_latest_ckpt() -> Optional[str]:
    """Find the highest-step .pt under lfs_checkpoints/ — both legacy
    flat layout and H24+ per-run subdir."""
    ckpt_root = REPO_ROOT / "lfs_checkpoints"
    if not ckpt_root.is_dir():
        return None
    flat = list(ckpt_root.glob("dsl_arch_*.pt")) \
        + list(ckpt_root.glob("dsl_arch_step*.pt"))
    nested = list(ckpt_root.glob("*/step*.pt"))
    candidates: List[Tuple[int, Path]] = []
    for p in flat + nested:
        m = re.search(r"step(\d+)\.pt$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        # Last resort: any .pt file at all
        any_pt = sorted(
            list(ckpt_root.glob("*.pt")) + list(ckpt_root.glob("*/*.pt")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        return str(any_pt[0]) if any_pt else None
    candidates.sort(key=lambda x: -x[0])
    return str(candidates[0][1])


def cmd_test(args: argparse.Namespace) -> int:
    path = args.pattern if args.pattern else "tests/dsl/"
    cli = [sys.executable, "-m", "pytest", path, "-q"]
    if args.slow:
        # Override pyproject.toml addopts which globally applies `-m not slow`
        cli.extend(["-o", "addopts="])
    else:
        cli.extend(["-m", "not slow"])
    if args.verbose:
        cli.append("-v")
    return _run(cli)


# ──────────────────────────────────────────────────────────────────────
# `brian test {quick,fast,full}` — unified test driver
#
# Policy: contributors (human + AI) should NEVER invoke `pytest`
# directly. Instead use one of:
#   brian test quick   30 most-recently-modified test files (by mtime)
#   brian test fast    30 fastest tests by cached durations
#   brian test full    canonical full sweep (excludes the known-slow
#                      files; rewrites the duration cache on success)
#
# Centralising the invocation means the exclusion list, the duration
# cache, and the "what's hot right now" heuristic all live in one
# place — no more drift between contributors' shell histories.
#
# Contracts pinned by tests/test_brian_test_subcommands.py.
# ──────────────────────────────────────────────────────────────────────


# Persistent per-nodeid wall-time cache. Populated by `brian test full`,
# consumed by `brian test fast`. Lives under .neuro/ so it stays out
# of the repo root (and so the existing `.neuro/` gitignore line covers
# it without a new rule).
TEST_DURATIONS_CACHE: Path = REPO_ROOT / ".neuro" / "test_durations.json"

# Canonical exclusion list for `brian test full`. Anyone removing one
# of these must also update the corresponding test in
# tests/test_brian_test_subcommands.py — and prove the removed file
# is fast enough to belong in the default sweep.
_FULL_SWEEP_IGNORES: tuple[str, ...] = (
    "tests/test_feature_flag_ablation.py",
    "tests/test_brian_compile.py",
    "tests/training",
)

# Limits — tweak here, not in the command bodies.
_QUICK_TOP_N: int = 30
_FAST_TOP_N: int = 30


def _list_test_files(tests_root: Path) -> List[Path]:
    """Return every ``test_*.py`` file under ``tests_root`` (recursively).

    Files named ``conftest.py``, ``__init__.py``, or anything that
    doesn't match ``test_*.py`` are excluded — they don't carry
    test cases themselves and pytest collects them automatically when
    a sibling test file is passed.

    The canonical :data:`_FULL_SWEEP_IGNORES` files / dirs are also
    excluded so ``quick`` and ``fast`` modes never trip on the same
    pre-existing slow / broken files that ``full`` already skips.
    Without this, e.g. the broken ``test_feature_flag_ablation.py``
    file would block ``brian test quick`` even though it's been on
    the canonical ignore list for months.
    """
    if not tests_root.is_dir():
        return []
    # Resolve every exclusion to an absolute path so the membership
    # check is robust to relative-vs-absolute caller paths.
    ignored_abs: set[Path] = set()
    for entry in _FULL_SWEEP_IGNORES:
        ignored_abs.add((tests_root.parent / entry).resolve())
    out: List[Path] = []
    for path in tests_root.rglob("test_*.py"):
        if not path.is_file() or path.name == "__init__.py":
            continue
        resolved = path.resolve()
        # Skip if this file IS an ignored path, or lives under an
        # ignored directory.
        skip = False
        for ig in ignored_abs:
            if resolved == ig or ig in resolved.parents:
                skip = True
                break
        if skip:
            continue
        out.append(path)
    return out


def _most_recent_test_files(repo_root: Path, n: int) -> List[Path]:
    """Pick the ``n`` test files with the newest mtime under ``<repo>/tests/``."""
    tests_root = repo_root / "tests"
    files = _list_test_files(tests_root)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def _parse_pytest_durations(output: str) -> Dict[str, float]:
    """Parse pytest's ``--durations=0`` block into ``{nodeid: seconds}``.

    Each duration line has the shape:

        ``0.42s call     tests/foo.py::test_bar``

    pytest prints three lines per nodeid (``call`` + ``setup`` +
    ``teardown``); we keep the MAX so a slow fixture isn't masked by
    a fast call. Non-duration lines (headers, hidden-row footer,
    final ``N passed in ...``) are skipped silently.
    """
    durations: Dict[str, float] = {}
    # Compiled inline because the helper is rarely called.
    line_re = re.compile(
        r"^\s*([\d.]+)s\s+(call|setup|teardown)\s+(\S+::\S+)\s*$"
    )
    for line in output.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        secs = float(m.group(1))
        nodeid = m.group(3)
        prev = durations.get(nodeid)
        if prev is None or secs > prev:
            durations[nodeid] = secs
    return durations


def _save_durations_cache(durations: Dict[str, float]) -> None:
    """Persist ``durations`` to :data:`TEST_DURATIONS_CACHE` as JSON."""
    TEST_DURATIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TEST_DURATIONS_CACHE.write_text(
        json.dumps(durations, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_durations_cache() -> Dict[str, float]:
    """Read :data:`TEST_DURATIONS_CACHE`; return ``{}`` if missing/invalid."""
    if not TEST_DURATIONS_CACHE.is_file():
        return {}
    try:
        raw = TEST_DURATIONS_CACHE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {str(k): float(v) for k, v in data.items()}
    except (OSError, ValueError):
        return {}


def cmd_test_quick(args: argparse.Namespace) -> int:
    """Run pytest on the 30 most-recently-modified test files.

    Designed for "I just edited a couple of test files, re-check
    them" — much faster than the full sweep, and tracks active work
    automatically via mtime.
    """
    files = _most_recent_test_files(REPO_ROOT, _QUICK_TOP_N)
    if not files:
        print("[test quick] no test files found under tests/")
        return 1
    cli = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if getattr(args, "verbose", False):
        cli.append("-v")
    cli.extend(str(p) for p in files)
    print(f"[test quick] {len(files)} most-recently-modified test files")
    return _run(cli)


def cmd_test_fast(args: argparse.Namespace) -> int:
    """Run the 30 fastest individual tests (by cached duration).

    Falls back to ``cmd_test_quick`` when no duration cache exists
    yet — that way a fresh clone still gets a useful smoke check
    even before any ``brian test full`` has populated the cache.
    """
    durations = _load_durations_cache()
    if not durations:
        print(
            "[test fast] no duration cache yet "
            f"({TEST_DURATIONS_CACHE.relative_to(REPO_ROOT) if TEST_DURATIONS_CACHE.is_absolute() else TEST_DURATIONS_CACHE});"
            " falling back to quick mode. Run `brian test full` once "
            "to populate it."
        )
        return cmd_test_quick(args)
    # Sort ascending by duration; pick the fastest _FAST_TOP_N.
    fastest = sorted(durations.items(), key=lambda kv: kv[1])[:_FAST_TOP_N]
    cli = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if getattr(args, "verbose", False):
        cli.append("-v")
    cli.extend(nodeid for nodeid, _ in fastest)
    total_s = sum(s for _, s in fastest)
    print(
        f"[test fast] {len(fastest)} fastest tests "
        f"(total cached time ≈ {total_s:.2f}s)"
    )
    return _run(cli)


def cmd_test_full(args: argparse.Namespace) -> int:
    """Run the canonical full sweep + refresh the duration cache.

    Mirrors the invocation that emerged from the 2026-06-15
    NFG-cleanup sweep:

        pytest tests/ -q --ignore=<known-slow> -p no:cacheprovider

    plus ``--durations=0`` so we capture every nodeid's wall time
    and dump it into :data:`TEST_DURATIONS_CACHE` for the next
    ``brian test fast`` invocation.

    The execution goes through :func:`_run_tee` so we get a single
    pytest invocation that the developer sees live AND whose output
    we parse for the duration cache — no double-runs.
    """
    cli = [
        sys.executable, "-m", "pytest", "tests/", "-q",
        "-p", "no:cacheprovider", "--durations=0",
    ]
    for ignored in _FULL_SWEEP_IGNORES:
        cli.append(f"--ignore={ignored}")
    if getattr(args, "verbose", False):
        cli.append("-v")

    rc, captured = _run_tee(cli)

    # Refresh the cache on best-effort basis — parse failures are
    # silent (the cache is a perf hint, not a correctness contract).
    try:
        durations = _parse_pytest_durations(captured)
        if durations:
            _save_durations_cache(durations)
            print(
                f"[test full] cached {len(durations)} test durations → "
                f"{TEST_DURATIONS_CACHE}"
            )
    except (OSError, ValueError) as e:
        print(f"[test full] could not refresh duration cache: {e}")

    return rc



def cmd_push(args: argparse.Namespace) -> int:
    """Push the current branch using the PAT from .env (avoids credential helper)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        print(".env not found at repo root")
        return 1
    pat = None
    for line in env_path.read_text().splitlines():
        if line.startswith("GH_TOKEN="):
            pat = line.split("=", 1)[1].strip()
            break
        if line.startswith("GITHUB_PAT=") or line.startswith("GITHUB="):
            pat = line.split("=", 1)[1].strip()
    if not pat:
        print("no GH_TOKEN found in .env")
        return 1
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(REPO_ROOT), text=True).strip()
    url = f"https://x-access-token:{pat}@github.com/269652/BRIAN.git"
    print(f"push HEAD ({branch}) → origin/{branch}")
    rc = subprocess.call(
        ["git", "-c", "credential.helper=", "push", url, f"HEAD:{branch}"],
        cwd=str(REPO_ROOT))
    return rc


def cmd_clean(args: argparse.Namespace) -> int:
    """Find + delete unreferenced logs / checkpoints / docs / lfs.

    Default is dry-run; pass ``--force`` to actually delete. Files
    are protected if they appear in any markdown/code in the repo
    (so a checkpoint cited in docs/FINDINGS.md is never deleted),
    if they are anchor docs, if they are among the N most-recent in
    the bucket, or if they are currently staged / modified in git.

    The ``lfs`` bucket is special: it runs the per-run LFS pruner
    (``neuroslm.tools.clean_lfs``) which groups checkpoints by their
    parent folder, keeps the N most-recent steps per folder, and
    keeps ``*_best.*`` only when its run's log file is itself
    referenced anywhere in the repo.
    """
    buckets = list(args.bucket)
    rc = 0

    # `lfs` has different semantics (per-run-folder retention + log-
    # gated best protection) so it gets its own runner.
    if "lfs" in buckets:
        buckets.remove("lfs")
        from neuroslm.tools.clean_lfs import run as _lfs_run
        rc |= _lfs_run(
            root=REPO_ROOT,
            force=args.force,
            keep_recent=args.keep_recent,
            verbose=args.verbose,
            use_git=not args.no_git,
        )

    if buckets:
        from neuroslm.tools.clean import run as _clean_run
        rc |= _clean_run(
            buckets,
            force=args.force,
            verbose=args.verbose,
            keep_recent=args.keep_recent,
            use_git=not args.no_git,
            root=REPO_ROOT,
        )
    return rc


# ── migrate ────────────────────────────────────────────────────────────


def cmd_migrate(args: argparse.Namespace) -> int:
    """brian migrate — versioned, idempotent repo migrations.

    Default: dry-run a single migration by ID, or `--list` to inventory
    every migration with its status (APPLIED / PENDING / DRIFT /
    NOOP_PENDING). Pass ``--force`` to actually apply. Pass ``--rerun``
    to re-apply a migration that is already in the ledger.

    The ledger lives at ``.brian/migrations.json`` and is the source
    of truth for "has migration X been run?".
    """
    from neuroslm.migrations import _framework as fw
    from neuroslm.references import build_reference_index

    pkg_dir = Path(__file__).resolve().parent / "migrations"

    # Build a single reference index up-front. Migrations that need it
    # read `ctx.refs`; ones that don't, ignore it. One scan covers
    # --list, --all, and single-id runs equally — and --list MUST
    # have the real refs or its status output lies (NOOP_PENDING vs
    # PENDING flips on whether refs.references() returned True).
    refs = build_reference_index(REPO_ROOT, progress=args.verbose)
    ctx = fw.Context(
        root=REPO_ROOT, refs=refs,
        dry_run=not args.force, force=args.force,
    )

    if args.list:
        return fw.cli_list(pkg_dir, ctx)

    if not args.id and not args.all:
        print("[migrate] error: provide a migration id, --all, or --list",
              flush=True)
        return 2

    if args.all:
        return fw.run_all(pkg_dir, ctx)

    return fw.run_one(pkg_dir, args.id, ctx, rerun=args.rerun)


# ── lint ───────────────────────────────────────────────────────────────

def _infer_equation_name(formula: str) -> str:
    """Suggest equation name based on formula pattern."""
    if "weight * (x_pre @ W)" in formula or "weight *(x_pre @ W)" in formula:
        return "standard_synapse"
    elif "output * (c * gain)" in formula:
        return "multiplicative_modulation"
    elif "output + (c * gain)" in formula:
        return "additive_modulation"
    elif "weight" in formula and "x_pre" in formula:
        return "synapse_transmission"
    else:
        # Generate name based on first two operators/keywords
        parts = re.findall(r'\b[a-z]\w*\b', formula.lower())
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return "custom_equation"


def cmd_model(args: argparse.Namespace) -> int:
    """Load a standard model from a .neuro arch spec and run inference.

    Reads the ``model { }`` block from <arch>/arch.neuro, builds the model,
    loads HF weights (specified by ``weights: "hf:..."`` in the spec), then
    runs the requested sub-command (ppl | generate).

    Examples::

        brian model ppl architectures/gpt2
        brian model generate architectures/smollm2-135m --prompt "Hello world"
        brian model ppl architectures/gpt2 --text "Once upon a time"
    """
    import math
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    from neuroslm.dsl.model_spec import parse_model_block
    from neuroslm.models import build_model

    # ── Resolve arch path ────────────────────────────────────────────────
    arch_path = Path(args.arch).resolve()
    neuro_file = (arch_path / "arch.neuro") if arch_path.is_dir() else arch_path
    if not neuro_file.exists():
        print(f"[ERROR] not found: {neuro_file}", file=sys.stderr)
        return 1

    spec = parse_model_block(neuro_file.read_text(encoding="utf-8"))
    s = spec.sheaf
    print(f"[model] kind={spec.kind}  dim={s.dim}  depth={s.depth}  "
          f"heads={s.heads}/{s.kv_heads}  vocab={s.vocab}  pos={s.pos}")

    # ── Load weights ─────────────────────────────────────────────────────
    tokenizer = None
    if spec.weights and spec.weights.startswith("hf:"):
        hf_id = spec.weights[3:]
        print(f"[model] loading HF weights: {hf_id}")
        try:
            if spec.kind == "gpt2":
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                from neuroslm.models.gpt2 import GPT2Model, hf_to_model_state_dict
                hf_model = GPT2LMHeadModel.from_pretrained(hf_id)
                tokenizer = GPT2Tokenizer.from_pretrained(hf_id)
                model = GPT2Model(spec)
                model.load_state_dict(hf_to_model_state_dict(hf_model.state_dict(), spec))
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from neuroslm.models.llama import LlamaModel, hf_to_model_state_dict
                hf_model = AutoModelForCausalLM.from_pretrained(hf_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = LlamaModel(spec)
                model.load_state_dict(hf_to_model_state_dict(hf_model.state_dict(), spec))
            del hf_model  # free HF copy
        except Exception as exc:
            print(f"[ERROR] HF load failed: {exc}", file=sys.stderr)
            return 1
    else:
        print("[model] no weights specified — using random init")
        model = build_model(spec)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}")

    # ── Sub-commands ─────────────────────────────────────────────────────
    _WIKITEXT_SAMPLE = (
        " = Valkyria Chronicles III = \n\n Senjō no Valkyria 3 : Unrecorded Chronicles "
        "( Japanese : 戦場のヴァルキュリア3 ) , commonly referred to as Valkyria Chronicles III "
        "outside Japan , is a tactical role-playing game developed by Sega and Media.Vision "
        "for the PlayStation Portable . Released in January 2011 in Japan , it is the third "
        "game in the Valkyria series . Employing the same fusion of tactical and real-time "
        "gameplay as its predecessors , the story runs parallel to the first game ."
    )

    if args.model_cmd == "ppl":
        if tokenizer is None:
            print("[ERROR] PPL requires a tokenizer — set weights: in arch.neuro or use --hf",
                  file=sys.stderr)
            return 1
        text = getattr(args, "text", None) or _WIKITEXT_SAMPLE
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids[:, : s.context]
        with torch.no_grad():
            logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets)
        ppl = math.exp(loss.item())
        n_tok = input_ids.size(1) - 1
        print(f"[model] PPL = {ppl:.2f}  (tokens={n_tok})")
        return 0

    if args.model_cmd == "generate":
        if tokenizer is None:
            print("[ERROR] generate requires a tokenizer — set weights: in arch.neuro",
                  file=sys.stderr)
            return 1
        prompt = getattr(args, "prompt", None) or "The quick brown fox"
        n_tokens = getattr(args, "tokens", 50)
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc.input_ids
        with torch.no_grad():
            for _ in range(n_tokens):
                logits = model(ids)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if ids.size(1) >= s.context:
                    break
        generated = tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"[model] {generated}")
        return 0

    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    """Lint a .neuro file or architecture folder.

    With --autofix, automatically apply fixable diagnostics:
    - Extract repeated equations to lib/equations.neuro
    - Add import in arch.neuro for library equations
    - Replace inline equations with @references to definitions

    Iterates until all autofix-able issues are resolved.
    """
    from pathlib import Path
    from neuroslm.dsl.neuro_linter import NeuroLinter
    import re

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"[ERROR] Path not found: {args.path}", file=sys.stderr)
        return 1

    # Lint the file or architecture
    is_arch_dir = path.is_dir() and (path / "arch.neuro").is_file()
    if is_arch_dir:
        linter_file = path / "arch.neuro"
        arch_dir = path
    elif path.suffix == ".neuro":
        linter_file = path
        arch_dir = path.parent
    else:
        print(f"[ERROR] Not a .neuro file or architecture directory: {args.path}", file=sys.stderr)
        return 1

    # Ensure lib directory exists
    lib_dir = arch_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    lib_equations_file = lib_dir / "equations.neuro"

    total_fixes = 0
    iteration = 0
    max_iterations = 10  # Prevent infinite loops

    # Iterate until no more fixes needed
    while iteration < max_iterations:
        iteration += 1
        linter = NeuroLinter(linter_file)
        diags = linter.lint()

        if not diags:
            print("[OK] No issues found")
            break

        if iteration == 1:  # Only print on first iteration
            print(f"[LINT] Found {len(diags)} issue(s):")
            for d in diags:
                severity = d.severity.value.upper()
                print(f"  {severity} {d.file.name}:{d.line}:{d.col} [{d.code}] {d.message}")

        if not args.autofix:
            return 1 if any(d.severity.value == "error" for d in diags) else 0

        # Apply autofix for repeated equations
        iteration_fixes = 0
        equations_to_add = {}  # Track equations to add to lib

        for d in diags:
            if d.code == "repeated-equation":
                # Extract equation from file
                with open(linter_file, 'r') as f:
                    content = f.read()

                # Extract equation formula from the diagnostic message
                msg_match = re.search(r'formula: "([^"]*)"', d.message)
                if not msg_match:
                    continue

                formula = msg_match.group(1)

                # Infer name based on formula
                eq_name = _infer_equation_name(formula)

                # Check if this equation already exists in lib/equations.neuro
                lib_content = lib_equations_file.read_text(encoding='utf-8') if lib_equations_file.exists() else ""
                if re.search(rf'export equation {eq_name}\s*\{{\s*params:[^}}]*formula: "{re.escape(formula)}"', lib_content):
                    # Already in lib, just replace references in arch.neuro
                    print(f"[AUTOFIX] Use library equation '{eq_name}' (found in lib/equations.neuro)")
                    new_content = re.sub(
                        f'equation: "{re.escape(formula)}"',
                        f'equation: @{eq_name}',
                        content
                    )
                else:
                    # Need to extract to lib/equations.neuro
                    print(f"[AUTOFIX] Extract '{formula}' as '{eq_name}' -> lib/equations.neuro")

                    # Extract parameters
                    params = []
                    for param_match in re.finditer(r'\b([a-zA-Z_]\w*)\b', formula):
                        param = param_match.group(1)
                        builtins = {'ReLU', 'sigmoid', 'tanh', 'sin', 'cos', 'exp', 'log', 'sqrt',
                                   'matmul', 'x', 'y', 's', 'V', 'dt', 'pi', 'e', 'max', 'min', 'output', 'c', 'gain', 'weight', 'x_pre', 'W'}
                        if param not in builtins and param not in params:
                            params.append(param)

                    # Create equation definition
                    eq_def = f"export equation {eq_name} {{\n    params: [{', '.join(params)}],\n    formula: \"{formula}\"\n}}\n"
                    equations_to_add[eq_name] = eq_def

                    # Replace all inline equations with @references in arch.neuro
                    new_content = re.sub(
                        f'equation: "{re.escape(formula)}"',
                        f'equation: @{eq_name}',
                        content
                    )

                    # Add/update import statement
                    import_match = re.search(r'(import\s*\{([^}]*)\}\s*from\s*"@/lib/equations")', new_content)
                    if import_match:
                        # Update existing import
                        existing_imports = import_match.group(2)
                        if f'{eq_name}' not in existing_imports:
                            new_import = f'import {{ {existing_imports}, {eq_name} }} from "@/lib/equations"'
                            new_content = new_content.replace(import_match.group(1), new_import)
                    else:
                        # Create new import line at top
                        lines = new_content.split('\n')
                        insert_line = 0
                        # Skip shebang and comments at the very top
                        while insert_line < len(lines) and (lines[insert_line].startswith('#') or not lines[insert_line].strip()):
                            insert_line += 1
                        lines.insert(insert_line, f'import {{ {eq_name} }} from "@/lib/equations"')
                        new_content = '\n'.join(lines)

                # Write arch.neuro back
                with open(linter_file, 'w') as f:
                    f.write(new_content)

                iteration_fixes += 1
                total_fixes += 1

        # Autofix bare `key {` → `key: {` block syntax (file-wide, one pass)
        if args.autofix and any(d.code == "missing-block-colon" for d in diags):
            from neuroslm.dsl.neuro_linter import autofix_block_colon_syntax
            content = linter_file.read_text(encoding='utf-8')
            new_content = autofix_block_colon_syntax(content)
            if new_content != content:
                linter_file.write_text(new_content, encoding='utf-8')
                n_fixed = sum(1 for d in diags if d.code == "missing-block-colon")
                print(f"[AUTOFIX] {n_fixed} bare block(s) rewritten to `key: {{` in {linter_file.name}")
                iteration_fixes += n_fixed
                total_fixes += n_fixed

        # Write all equations to lib/equations.neuro
        if equations_to_add:
            lib_content = lib_equations_file.read_text(encoding='utf-8') if lib_equations_file.exists() else ""
            for eq_name, eq_def in equations_to_add.items():
                # Check if equation already exists
                if not re.search(rf'export equation {eq_name}\s*\{{', lib_content):
                    lib_content += eq_def
            lib_equations_file.write_text(lib_content, encoding='utf-8')

        if iteration_fixes == 0:
            break  # No more fixes in this iteration

    if total_fixes > 0:
        print(f"\n[AUTOFIX] Total: {total_fixes} fix(es) applied ({iteration} iteration(s))")

    return 0


# ---------------------------------------------------------------------------
# Studio helpers
# ---------------------------------------------------------------------------

def _find_npm() -> str:
    """Return the path to npm that ships with Node 18+ (Next.js requirement).

    Prefers nvm-managed versions (newest first) over the system default.
    Falls back to plain 'npm' if nothing better is found.
    """
    import re
    nvm_root = Path(os.environ.get("NVM_HOME", "")) or Path.home() / "AppData" / "Roaming" / "nvm"
    if nvm_root.exists():
        candidates = []
        for ver_dir in nvm_root.iterdir():
            m = re.match(r"v(\d+)\.(\d+)\.(\d+)", ver_dir.name)
            if m:
                major = int(m.group(1))
                if major >= 18:
                    npm_cmd = ver_dir / "npm.cmd"
                    if npm_cmd.exists():
                        candidates.append((major, int(m.group(2)), int(m.group(3)), str(npm_cmd)))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][3]
    return "npm"


# ---------------------------------------------------------------------------
# Studio command
# ---------------------------------------------------------------------------

def cmd_studio_start(args: argparse.Namespace) -> int:
    """Start Brian Studio: REST+MCP server on :1984, Next.js client on :2049."""
    import os
    import signal
    import time
    import webbrowser
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    studio_dir = repo_root / "studio"
    client_dir = studio_dir / "client"

    server_port = 1984
    client_port = 3141  # π — port 2049 (NFS) is blocked by Chrome

    procs = []

    def _shutdown(sig=None, frame=None):
        print("\n[studio] shutting down...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    # ── Python REST+MCP server ──────────────────────────────────────────────
    import subprocess
    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "studio.server.app:app",
        "--host", args.host,
        "--port", str(server_port),
        "--reload",
        "--reload-dir", str(studio_dir / "server"),
        "--log-level", "warning",
    ]
    server_proc = subprocess.Popen(server_cmd, cwd=repo_root, shell=False)
    procs.append(server_proc)

    print(f"\n  Brian Studio")
    print(f"  {'-' * 45}")
    print(f"  Server  (REST+MCP) : http://localhost:{server_port}")
    print(f"  API docs           : http://localhost:{server_port}/docs")
    print(f"  MCP endpoint       : http://localhost:{server_port}/mcp")

    if args.server_only:
        print(f"\n  --server-only: Next.js client not started")
        print(f"  Press Ctrl-C to stop\n")
        try:
            server_proc.wait()
        except (KeyboardInterrupt, SystemExit):
            _shutdown()
        return 0

    # ── Next.js client ──────────────────────────────────────────────────────
    npm_exe = _find_npm()
    node_modules = client_dir / "node_modules"
    if not node_modules.exists():
        print(f"\n  Installing npm dependencies (first run)...")
        subprocess.run(
            npm_exe + " install",
            cwd=client_dir,
            shell=True,
        )

    client_env = {**os.environ, "PORT": str(client_port)}
    # Use shell=True so .cmd wrappers resolve correctly on Windows;
    # pass port via PORT env-var (Next.js reads it) rather than CLI flag.
    client_proc = subprocess.Popen(
        npm_exe + " run dev",
        cwd=client_dir,
        env=client_env,
        shell=True,
    )
    procs.append(client_proc)

    print(f"  Studio  (Next.js)  : http://localhost:{client_port}")
    print(f"  {'-' * 45}")
    print(f"  Press Ctrl-C to stop\n")

    if not args.no_browser:
        time.sleep(4)  # wait for Next.js to compile
        webbrowser.open(f"http://localhost:{client_port}")

    try:
        server_proc.wait()
    except (KeyboardInterrupt, SystemExit):
        _shutdown()

    return 0


# ── update-readme ──────────────────────────────────────────────────────

def cmd_update_readme(args: argparse.Namespace) -> int:
    """Render README.template.md → README.md using docs/readme_metrics.toml
    merged with arch.neuro # @export values from .neuro/exports.toml.

    Workflow (in order):

      1. ``log_refs.update_best_run_pointer`` — refresh ``.brian/best_run.ln``
         + ``.brian/checkpoint.ln`` so the ``${LOG_TAIL:best:N}`` macro
         always points at the freshest qualifying training run. Skipped
         with ``--no-best-update``; failures are non-fatal.
      2. Collect arch exports from ``architectures/master/arch.neuro``
         and write ``.neuro/exports.toml``.
      3. Render the template and write ``README.md`` (or check freshness
         with ``--check``).

    ``brian update-readme``                  — full pipeline, in-place write.
    ``brian update-readme --check``          — compare only; exit 1 if stale (pre-commit).
    ``brian update-readme --no-best-update`` — skip step 1 (CI / tests).
    """
    # ── Step 1: refresh .brian/best_run.ln + .brian/checkpoint.ln ──
    # Auto-update keeps the ${LOG_TAIL:best:N} macro fresh so a render
    # right after a new training log lands picks up the new best run
    # without a manual ``brian best update`` invocation. Failures are
    # non-fatal: the renderer's macro falls back to "(log not available)"
    # gracefully when the .ln file is absent.
    if not getattr(args, "no_best_update", False):
        try:
            from neuroslm.log_refs import update_best_run_pointer
            best = update_best_run_pointer(root=REPO_ROOT)
            if best is not None:
                print(f"[update-readme] refreshed best-run pointer: "
                      f"{best.relative_to(REPO_ROOT) if best.is_absolute() else best}")
        except Exception as exc:  # noqa: BLE001
            print(f"[update-readme] warning: best-run pointer refresh "
                  f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    from neuroslm.readme_renderer_v2 import (
        ReadmeRenderError,
        render_readme,
    )
    from neuroslm.arch_exports import collect_arch_exports, write_neuro_exports

    template        = REPO_ROOT / "README.template.md"
    metrics         = REPO_ROOT / "docs" / "readme_metrics.toml"
    output          = REPO_ROOT / "README.md"
    neuro_dir       = REPO_ROOT / ".neuro"
    neuro_exports   = neuro_dir / "exports.toml"
    arch_neuro      = REPO_ROOT / "architectures" / "master" / "arch.neuro"

    # Collect arch exports from arch.neuro and write .neuro/exports.toml.
    # Best-effort: if the arch file is missing or unreadable, skip silently.
    if arch_neuro.exists():
        try:
            exports = collect_arch_exports(arch_neuro)
            write_neuro_exports(exports, neuro_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[update-readme] warning: could not collect arch exports: {exc}",
                  file=sys.stderr)

    try:
        rendered, is_clean = render_readme(
            template, metrics, output,
            check=args.check,
            repo_root=REPO_ROOT,
            neuro_exports_path=neuro_exports,
        )
    except FileNotFoundError as exc:
        print(f"[update-readme] error: {exc}", file=sys.stderr)
        return 1
    except ReadmeRenderError as exc:
        print(f"[update-readme] template error:\n{exc}", file=sys.stderr)
        return 1

    if args.check:
        if is_clean:
            print("[update-readme] README.md is up to date.")
            return 0
        print(
            "[update-readme] README.md is stale — run `brian update-readme` "
            "and stage the result.",
            file=sys.stderr,
        )
        return 1

    print(f"[update-readme] wrote {output.relative_to(REPO_ROOT)}")
    return 0


# ── arg parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brian",
        description="Unified CLI for the NeuroSLM / BRIAN project.")
    # Top-level escape hatch for pre-flight hooks (modelled after
    # ``git commit --no-verify``). Today only ``brian deploy`` calls a
    # hook (``pre-deploy``); future hook-calling commands inherit this
    # flag automatically because it lives on the parent parser.
    # Lives on BOTH parent and subparser so it works in either position:
    #   brian --no-verify deploy   (parent slot)
    #   brian deploy --no-verify   (subparser slot, see ``sd`` below)
    # The subparser uses ``default=SUPPRESS`` so an unset subparser
    # flag does NOT clobber the parent's value.
    p.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="Skip pre-flight hooks (currently: pre-deploy). "
             "Mirrors `git commit --no-verify`.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # compile (group: `brian compile <arch>` for nn.Module, or
    #                  `brian compile nfg <arch>` for Neural Flow Graph)
    sc = sub.add_parser("compile",
                        help="Compile an arch.neuro folder (nn.Module or NFG)")
    sc.add_argument("arch_or_subcmd", nargs="?",
                    help="architecture name OR 'nfg' for the NFG sub-command")
    sc.add_argument("arch", nargs="?",
                    help="architecture (only when first arg was 'nfg')")
    sc.add_argument("--out", help="write generated .py to this path")
    sc.add_argument("--png", help="(nfg only) write PNG render to this path")
    sc.add_argument("--semantic", action="store_true",
                    help="(nfg only, legacy) use semantic layout inference (data-driven)")
    sc.add_argument("--legacy", action="store_true",
                    help="(nfg only) use the legacy matplotlib renderer instead of Graphviz")
    sc.add_argument("--engine", default="dot",
                    choices=["dot", "neato", "sfdp", "fdp", "circo"],
                    help="(nfg only) Graphviz layout engine (default: dot)")
    sc.add_argument("--format", default="png",
                    choices=["png", "svg", "pdf", "dot"],
                    help="(nfg only) output format for the Graphviz render (default: png)")
    sc.add_argument("--heat",
                    help="(nfg only) overlay training-heatmap colors; "
                         "accepts a JSON file produced by HeatmapPublisher. "
                         "When combined with --current, output filename gets a "
                         "`.heat` infix (e.g. .neuro/nfg.heat.png).")
    sc.add_argument("--current", action="store_true",
                    help="(nfg only) read arch/DNA from brian.toml's [current] "
                         "section and write to its [nfg].output path. Allows "
                         "`brian compile nfg --current` with no positional arg.")
    sc.add_argument("--head", type=int, default=2000,
                    help="when printing to stdout, truncate after N chars")
    def _dispatch_compile(a):
        # The 'nfg' sub-command can be invoked two ways:
        #   brian compile nfg <arch>     → first positional is "nfg"
        #   brian compile nfg --current  → first positional is "nfg", arch=None
        is_nfg = (a.arch_or_subcmd == "nfg")
        if a.arch_or_subcmd is None and not (is_nfg or a.current):
            print("Error: architecture name required", file=sys.stderr)
            print("Usage: brian compile <arch> [--out FILE]", file=sys.stderr)
            print("       brian compile nfg <arch> [--out FILE] [--png FILE]", file=sys.stderr)
            print("       brian compile nfg --current [--heat HEATMAP.json]", file=sys.stderr)
            return 1
        if is_nfg:
            if a.arch is None and not a.current:
                print("Error: architecture name required after 'nfg' "
                      "(or pass --current to read from brian.toml)",
                      file=sys.stderr)
                print("Usage: brian compile nfg <arch> [--out FILE] [--png FILE]", file=sys.stderr)
                print("       brian compile nfg --current [--heat HEATMAP.json]", file=sys.stderr)
                return 1
            return cmd_compile_nfg(argparse.Namespace(
                arch=a.arch, out=a.out, png=a.png, semantic=a.semantic,
                legacy=a.legacy, engine=a.engine, format=a.format,
                heat=a.heat, current=a.current))
        else:
            return cmd_compile(argparse.Namespace(
                arch=a.arch_or_subcmd, out=a.out, head=a.head))

    sc.set_defaults(func=_dispatch_compile)

    # dna (DNA encoding/decoding)
    sdna = sub.add_parser("dna", help="DNA encoding/decoding for evolutionary architecture")
    sdna_sub = sdna.add_subparsers(dest="dna_cmd", required=True)

    sdna_compile = sdna_sub.add_parser("compile", help="Compile arch.neuro to DNA binary")
    # ``arch`` is OPTIONAL. When omitted, ``cmd_dna`` reads
    # ``[current].arch`` from brian.toml and writes to
    # ``[current].dna`` (so a subsequent ``brian deploy`` picks up
    # the fresh DNA without -o). When given, the legacy default
    # (``architectures/<arch>/evolution.dna``) applies.
    sdna_compile.add_argument(
        "arch", nargs="?", default=None,
        help="architecture name (e.g., rcc_bowtie). "
             "Default: brian.toml [current].arch.",
    )
    sdna_compile.add_argument(
        "--output", "-o",
        help="DNA output file. Default: brian.toml [current].dna when "
             "no positional arch is given, else "
             "architectures/<arch>/evolution.dna.",
    )
    sdna_compile.set_defaults(func=cmd_dna)

    sdna_unfold = sdna_sub.add_parser("unfold", help="Unfold DNA binary back to .neuro DSL")
    sdna_unfold.add_argument("dna", help="path to DNA binary file")
    sdna_unfold.add_argument("--output", "-o", help="DSL output file (default: <dna>.neuro)")
    sdna_unfold.set_defaults(func=cmd_dna)

    # hypothesis (human-authored formal ledger)
    shyp = sub.add_parser(
        "hypothesis",
        help="Manage hypothesis/ formal-claim ledger (with Lean proofs)")
    shyp_sub = shyp.add_subparsers(dest="hyp_cmd", required=True)

    shyp_list = shyp_sub.add_parser("list", help="List every hypothesis")
    shyp_list.set_defaults(func=cmd_hypothesis)

    shyp_show = shyp_sub.add_parser("show", help="Show one hypothesis by id")
    shyp_show.add_argument("id", help="hypothesis id (e.g. H001)")
    shyp_show.set_defaults(func=cmd_hypothesis)

    shyp_emit = shyp_sub.add_parser(
        "emit-proofs",
        help="(Re)generate Lean proof stubs for every hypothesis missing one")
    shyp_emit.set_defaults(func=cmd_hypothesis)

    shyp_verify = shyp_sub.add_parser(
        "verify",
        help="Run lean against a hypothesis's proof file and update its "
             "proof_status")
    shyp_verify.add_argument("id", nargs="?",
                             help="hypothesis id (omit + --all to verify every record)")
    shyp_verify.add_argument("--all", action="store_true")
    shyp_verify.set_defaults(func=cmd_hypothesis)

    # discovery (engine-authored ledger + DNA splice)
    sdisc = sub.add_parser(
        "discovery",
        help="Manage discoveries/ autodiscovered-mutation ledger + DNA splice")
    sdisc_sub = sdisc.add_subparsers(dest="disc_cmd", required=True)

    sdisc_list = sdisc_sub.add_parser("list", help="List every discovery")
    sdisc_list.set_defaults(func=cmd_discovery)

    sdisc_show = sdisc_sub.add_parser("show", help="Show one discovery by id")
    sdisc_show.add_argument("id", help="discovery id (e.g. D001)")
    sdisc_show.set_defaults(func=cmd_discovery)

    sdisc_verify = sdisc_sub.add_parser(
        "verify",
        help="Run lean against a discovery's proof file")
    sdisc_verify.add_argument("id", nargs="?")
    sdisc_verify.add_argument("--all", action="store_true")
    sdisc_verify.set_defaults(func=cmd_discovery)

    sdisc_prom = sdisc_sub.add_parser(
        "promote",
        help="Splice a verified discovery into the architecture's DNA")
    sdisc_prom.add_argument("id", help="discovery id (e.g. D001)")
    sdisc_prom.add_argument("arch", help="architecture name (e.g. rcc_bowtie)")
    sdisc_prom.set_defaults(func=cmd_discovery)

    # discover — CPU search over NGL program space for an ML algorithm
    sdiscover = sub.add_parser(
        "discover",
        help="Search NGL program space (optimizers / flow modulation) on CPU")
    sdiscover_sub = sdiscover.add_subparsers(dest="discover_cmd", required=True)

    sdo = sdiscover_sub.add_parser(
        "optimizer", help="Evolve update-rule programs; score on a tiny CPU MLP")
    sdo.add_argument("--pop", type=int, default=24)
    sdo.add_argument("--generations", type=int, default=12)
    sdo.add_argument("--steps", type=int, default=30, help="train steps per eval")
    sdo.add_argument("--seed", type=int, default=0)
    sdo.add_argument("--task", default="regression", choices=["regression", "parity"])
    sdo.add_argument("--from-scratch", action="store_true",
                     help="seed only SGD+random (no SOTA seeds) — genuine discovery")
    sdo.add_argument("--novelty", type=float, default=0.0,
                     help="novelty-search weight (semantic-space distance); >0 hunts novel rules")
    sdo.add_argument("--avoid-known", action="store_true",
                     help="penalize rediscovering known algorithms (SGD/Adam/Lion/...)")
    sdo.add_argument("--macros", action="store_true",
                     help="let the search graft reusable macro building blocks (ADFs)")
    sdo.add_argument("--seed-from",
                     help="start from baseline algorithm(s), comma-sep (e.g. adam or adam,lion)")
    sdo.add_argument("--device", default="cpu",
                     help="cpu | cuda | auto — scale the tiny-model training onto a T4")
    sdo.add_argument("--out", help="write the run summary JSON here")
    sdo.set_defaults(func=cmd_discover)

    sdf = sdiscover_sub.add_parser(
        "flow", help="Evolve gradient/flow-modulation programs; score loss + EI")
    sdf.add_argument("--pop", type=int, default=16)
    sdf.add_argument("--generations", type=int, default=8)
    sdf.add_argument("--steps", type=int, default=25)
    sdf.add_argument("--seed", type=int, default=0)
    sdf.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    sdf.add_argument("--out", help="write the run summary JSON here")
    sdf.set_defaults(func=cmd_discover)

    sdt = sdiscover_sub.add_parser(
        "trunk",
        help="Evolve a neuroanatomically-constrained residual-stream modulation "
             "for a tiny CPU LM (val PPL + realism prior)")
    sdt.add_argument("--pop", type=int, default=16)
    sdt.add_argument("--generations", type=int, default=8)
    sdt.add_argument("--steps", type=int, default=30)
    sdt.add_argument("--seed", type=int, default=0)
    sdt.add_argument("--device", default="cpu", help="cpu | cuda | auto — use a T4")
    sdt.add_argument("--out", help="write the run summary JSON here")
    sdt.add_argument("--save", metavar="NAME",
                     help="persist the discovered modulation as modulations/NAME.neuro")
    sdt.add_argument("--push", action="store_true",
                     help="git commit+push the saved modulation (during Colab/vast runs)")
    sdt.set_defaults(func=cmd_discover)

    sde = sdiscover_sub.add_parser(
        "experts",
        help="Probe the frozen pretrained expert cortices (SmolLM2/CodeGPT/"
             "Qwen) for domain-shift slack: search NGL modulations of each "
             "expert's final hidden, scored by ITS own next-token CE on "
             "real text; winners bank to modulations/")
    sde.add_argument("--models",
                     default="smollm2_360m,microsoft/CodeGPT-small-py,Qwen/Qwen2.5-0.5B",
                     help="comma-sep HF ids / aliases (default: the arch roster)")
    sde.add_argument("--rounds", type=int, default=10,
                     help="probe rounds (fresh text per round; recurrence "
                          "across rounds = install-grade evidence, since "
                          "frozen weights never move)")
    sde.add_argument("--batch", type=int, default=2)
    sde.add_argument("--seq_len", type=int, default=256)
    sde.add_argument("--pop", type=int, default=24)
    sde.add_argument("--generations", type=int, default=10)
    sde.add_argument("--length", type=int, default=8)
    sde.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    sde.add_argument("--push", action="store_true",
                     help="git commit+push banked winners each round")
    sde.add_argument("--out", help="write the run summary JSON here")
    sde.set_defaults(func=cmd_discover)

    sds = sdiscover_sub.add_parser(
        "simplify",
        help="Compile an nn_lang layer to NGL and discover a shorter equivalent")
    sds.add_argument("--layer-file", required=True,
                     help="path to an nn_lang `layer {...}` source file")
    sds.add_argument("--seed", type=int, default=0)
    sds.add_argument("--out", help="write the run summary JSON here")
    sds.set_defaults(func=cmd_discover)

    sdiscover_sub.add_parser(
        "baselines",
        help="List baseline algorithms + tradeoffs (use with --seed-from)"
    ).set_defaults(func=cmd_discover)

    sdiscover_sub.add_parser(
        "optimize-mechanics",
        help="CSE + superoptimize commonly-used mechanics; find shared subexprs"
    ).set_defaults(func=cmd_discover)

    sdm = sdiscover_sub.add_parser(
        "mechanics",
        help="List every known research mechanic (mechanics/dynamics/structures)")
    sdm.add_argument("--category", help="filter to one category (e.g. attention)")
    sdm.add_argument("--describe", metavar="NAME",
                     help="print the full description of one mechanic")
    sdm.add_argument("--out", help="write the catalog JSON here")
    sdm.set_defaults(func=cmd_discover)

    sdsem = sdiscover_sub.add_parser(
        "semantics",
        help="Static semantic analysis of a mechanic (abstract interpretation)")
    sdsem.add_argument("--known", metavar="NAME",
                       help="analyze a known NGL program (adam/lion/attention/...)")
    sdsem.add_argument("--layer-file",
                       help="analyze a compiled nn_lang layer instead")
    sdsem.add_argument("--out", help="write the summary JSON here")
    sdsem.set_defaults(func=cmd_discover)

    sdx = sdiscover_sub.add_parser(
        "extract-shared",
        help="Factor subexpressions shared across mechanics into reusable macros")
    sdx.add_argument("--out", help="write the extraction JSON here")
    sdx.set_defaults(func=cmd_discover)

    sdn = sdiscover_sub.add_parser(
        "normalize",
        help="Semantic normalization: collapse equivalent mechanics to a canonical form")
    sdn.add_argument("--prefer", default="frequency",
                     choices=["frequency", "simplest"],
                     help="canonical = most-used (frequency) or lowest-complexity (simplest)")
    sdn.add_argument("--out", help="write the normalization JSON here")
    sdn.set_defaults(func=cmd_discover)

    sdq = sdiscover_sub.add_parser(
        "qd",
        help="Quality-diversity (MAP-Elites) search: illuminate the semantic "
             "manifold — a diverse zoo of algorithms across shapes")
    sdq.add_argument("--iters", type=int, default=300)
    sdq.add_argument("--init", type=int, default=48)
    sdq.add_argument("--steps", type=int, default=25)
    sdq.add_argument("--seed", type=int, default=0)
    sdq.add_argument("--task", default="regression", choices=["regression", "parity"])
    sdq.add_argument("--seed-from", help="seed from baseline(s), comma-sep")
    sdq.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    sdq.add_argument("--out", help="write the archive JSON here")
    sdq.set_defaults(func=cmd_discover)

    sde = sdiscover_sub.add_parser(
        "explore",
        help="Wire exploration into (tiny-LM) training: search every N steps, "
             "keep-if-better, record to the persistent ledger")
    sde.add_argument("--total-steps", type=int, default=2000)
    sde.add_argument("--explore-every", type=int, default=500)
    sde.add_argument("--pop", type=int, default=12)
    sde.add_argument("--generations", type=int, default=4)
    sde.add_argument("--inner-steps", type=int, default=20)
    sde.add_argument("--seed", type=int, default=0)
    sde.add_argument("--wellformed-penalty", type=float, default=0.05,
                     help="fitness penalty per undefined-register read (steers the "
                          "search toward clean mechanics; 0 disables)")
    sde.add_argument("--ledger", help="path to the search ledger JSON "
                     "(default .neuro/search_ledger.json)")
    sde.add_argument("--push", action="store_true",
                     help="git commit+push ledger + modulations after the run")
    sde.add_argument("--seed-known", action=argparse.BooleanOptionalAction, default=True,
                     help="seed the ledger with known algorithms so only novel mechanics are searched (default on)")
    sde.add_argument("--out", help="write the run summary JSON here")
    sde.set_defaults(func=cmd_discover)

    sdl = sdiscover_sub.add_parser(
        "ledger", help="Inspect / clear the persistent search ledger")
    sdl.add_argument("--ledger", help="path (default .neuro/search_ledger.json)")
    sdl.add_argument("--top", type=int, default=20, help="how many records to show")
    sdl.add_argument("--clear", action="store_true", help="wipe the ledger")
    sdl.add_argument("--seed-known", action="store_true",
                     help="record every known ML algorithm as prior-art (skipped by the explorer)")
    sdl.set_defaults(func=cmd_discover)

    sdp = sdiscover_sub.add_parser(
        "profile",
        help="Profile an arch's NGL flow+compute heat + graph-topology bottlenecks")
    sdp.add_argument("--layer-file", required=True,
                     help="path to an nn_lang `layer {...}` source file")
    sdp.add_argument("--binding", action="append", metavar="K=V",
                     help="scalar bindings for the layer (e.g. --binding D=16 --binding H=32)")
    sdp.add_argument("--seed", type=int, default=0)
    sdp.add_argument("--out", help="write the profile+topology JSON here")
    sdp.set_defaults(func=cmd_discover)

    # modulation — manage stored NGL neuromodulations (modulations/*.neuro)
    smod = sub.add_parser(
        "modulation",
        help="Manage NGL neuromodulations (list/show/drop/merge modulations/*.neuro)")
    smod_sub = smod.add_subparsers(dest="mod_cmd", required=True)
    smod_sub.add_parser("list", help="list stored modulations").set_defaults(func=cmd_modulation)
    smod_show = smod_sub.add_parser("show", help="print a modulation's .neuro")
    smod_show.add_argument("name")
    smod_show.set_defaults(func=cmd_modulation)
    smod_drop = smod_sub.add_parser("drop", help="delete a modulation")
    smod_drop.add_argument("name")
    smod_drop.set_defaults(func=cmd_modulation)
    smod_merge = smod_sub.add_parser("merge", help="compose modulations into one")
    smod_merge.add_argument("names", nargs="+", help="modulations to compose (in order)")
    smod_merge.add_argument("--name", required=True, help="name for the merged modulation")
    smod_merge.set_defaults(func=cmd_modulation)

    # bundle
    sb = sub.add_parser("bundle",
                        help="Bundle all .neuro files for AI analysis")
    sb.add_argument("arch", help="architecture name (e.g., rcc_bowtie)")
    sb.add_argument("--out", help="write bundle to this file (default: stdout)")
    sb.set_defaults(func=cmd_bundle_arch)

    # wolfram
    sw = sub.add_parser("wolfram",
                        help="Emit Mathematica/Wolfram code for an arch")
    sw.add_argument("arch")
    sw.add_argument("--full", action="store_true",
                    help="IIT-grade: populations + synapses + modulations + NT dynamics")
    sw.add_argument("--out", help="write Wolfram code to this .m file")
    sw.set_defaults(func=cmd_wolfram)

    # model — standard-arch inference (DSL v2 model{} block)
    sm = sub.add_parser("model",
                        help="Load a standard model (GPT-2/LLaMA/Qwen) from arch.neuro and run PPL or generation")
    sm.add_argument("arch", help="Architecture directory (e.g. architectures/gpt2) or path to arch.neuro")
    sm_sub = sm.add_subparsers(dest="model_cmd", required=True)
    sm_ppl = sm_sub.add_parser("ppl", help="Compute WikiText-103 perplexity")
    sm_ppl.add_argument("--tokens", type=int, default=512, help="Max tokens to evaluate on")
    sm_gen = sm_sub.add_parser("generate", help="Greedy-decode N tokens from a prompt")
    sm_gen.add_argument("prompt", nargs="?", default="The quick brown fox",
                        help="Prompt text (default: 'The quick brown fox')")
    sm_gen.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    sm.set_defaults(func=cmd_model)

    # lint
    sl = sub.add_parser("lint",
                        help="Lint a .neuro file or architecture folder")
    sl.add_argument("path", help=".neuro file or architecture directory")
    sl.add_argument("--autofix", action="store_true",
                    help="Automatically apply fixable issues (extracts repeated equations)")
    sl.add_argument("-i", "--interactive", action="store_true",
                    help="Prompt for equation names when autofixing")
    sl.set_defaults(func=cmd_lint)

    # analyze
    sa = sub.add_parser("analyze",
                        help="SymPy analysis of an arch (Mathematica-style)")
    sa.add_argument("arch")
    sa.add_argument("--fixed-points", action="store_true")
    sa.add_argument("--jacobian", action="store_true")
    sa.add_argument("--stability", action="store_true")
    sa.add_argument("--wa-queries", action="store_true",
                    help="emit short Wolfram-Alpha-pasteable queries")
    sa.add_argument("--graph", metavar="PATH",
                    help="render topology graph to PATH (.png/.svg)")
    sa.add_argument("--flow", action="store_true",
                    help="dataflow analysis: paths, bottlenecks, bowtie waist")
    sa.add_argument("--phi", action="store_true",
                    help="IIT Φ proxy + per-module contribution")
    sa.add_argument(
        "--discover",
        choices=["phi", "modularity", "sparsity", "generalization", "ppl"],
        help="propose architecture mods maximising the metric. The "
             "'generalization' and 'ppl' choices use literature-grounded "
             "structural proxies (no training required)")
    sa.add_argument("--top-k", type=int, default=10,
                    help="top-K proposals for --discover")
    sa.add_argument("--topo-10x", action="store_true",
                    help="surface the hand-curated high-leverage topological "
                         "mutations targeting >10x OOD improvement")
    sa.add_argument("--all", action="store_true",
                    help="run every analysis above")
    sa.set_defaults(func=cmd_analyze)

    # deploy
    sd = sub.add_parser("deploy",
                        help="Launch a DSL or DNA training run on vast.ai")
    # Optional positional: path to an arch.neuro or architecture folder.
    # When given, sets ARCH env var so _deploy_train.py uses this arch
    # instead of brian.toml [current].arch. Accepts both folder paths
    # (e.g. architectures/gpt2) and file paths (architectures/gpt2/arch.neuro).
    sd.add_argument("arch", nargs="?", default=None,
                    help="Architecture path (folder or arch.neuro file). "
                         "Overrides brian.toml [current].arch.")
    # Default=None lets cmd_deploy distinguish "user didn't say" (fall
    # through to brian.toml [defaults].steps) from "user explicitly
    # asked for N steps" (always wins). The hardcoded final fallback
    # of 10_000 lives in cmd_deploy, not in the argparse default.
    sd.add_argument("--steps", type=int, default=None,
                    help="Training steps (default: brian.toml "
                         "[defaults].steps, then 10000)")
    sd.add_argument("--branch",
                    help="git branch to train (default: brian.toml "
                         "[defaults].branch, then current HEAD)")
    sd.add_argument("--scale", help="Scale variant from arch.neuro scales block "
                    "(e.g. 100m, 300m, 1b). Default: brian.toml [deploy].scale, "
                    "then the arch's preset: dims.")
    sd.add_argument("--machine", default=None,
                    help="GPU/machine tier for connectors that support it "
                         "(currently Lightning AI). Substring match against "
                         "the connector's enum, e.g. T4, A10G, A100, L4. "
                         "Default: brian.toml [deploy].machine, then "
                         "connector's own default.")
    sd.add_argument("--teamspace", default=None,
                    help="Lightning AI teamspace to host the Studio under. "
                         "Default: brian.toml [deploy].teamspace, then the "
                         "SDK's default (user's personal teamspace). "
                         "Ignored by non-Lightning connectors.")
    sd.add_argument("--dna", help="path to evolved DNA file for training "
                    "(e.g., dna/evol/arch.dna). If set, trains from DNA instead of DSL arch")
    sd.add_argument("--label", help="Label suffix for the vast.ai instance")
    sd.add_argument("--ood", type=int, nargs="?", const=3000,
                    help="Run mid-training OOD eval every N steps "
                         "(default 3000 if flag passed without value)")
    # ── Resume training from a checkpoint ──
    # Three flag shapes work together:
    #   --resume PATH          local path or hf://repo/path URI
    #   --latest               pull the highest-step ckpt from HF Hub
    #   --hf-repo R            override the HF repo for --latest
    #   --hf-prefix P          scope --latest to a run-dir prefix
    # On-box: ``RESUME_FROM`` env propagates through _deploy_train.py
    # → vast_train_dsl_loop.sh → ``--resume_from`` to neuroslm.train_dsl,
    # which downloads ``hf://...`` URIs into ``lfs_checkpoints/`` and
    # then resumes via the existing globber.
    sd.add_argument(
        "--resume", default=None, metavar="PATH_OR_URI",
        help="Resume training from a specific checkpoint. Accepts a "
             "local path (e.g. lfs_checkpoints/run-A/step5000.pt) or "
             "an hf:// URI (e.g. hf://moritzroessler/BRIAN/checkpoints/"
             "run-A/step5000.pt). Mutually exclusive with --latest.")
    sd.add_argument(
        "--latest", action="store_true",
        help="Resume from the highest-step checkpoint on HF Hub. Use "
             "with --hf-repo / --hf-prefix to scope the lookup. The "
             "step counter continues from where the prior run left "
             "off; use --steps to set a new target.")
    sd.add_argument(
        "--hf-repo", default=None, metavar="OWNER/REPO",
        help="HF Hub repo for --latest (default: HF_REPO_ID env, "
             "then moritzroessler/BRIAN).")
    sd.add_argument(
        "--hf-prefix", default=None, metavar="RUN_DIR",
        help="Scope --latest to checkpoints under this run-dir "
             "prefix (e.g. run-20260615_abc1234).")
    # Subparser-slot alias for the top-level --no-verify flag so
    # `brian deploy --no-verify` works as well as
    # `brian --no-verify deploy`. ``default=SUPPRESS`` is critical:
    # without it, parsing `brian --no-verify deploy` would have the
    # subparser silently overwrite the parent's True with this
    # subparser's False default, defeating the whole flag.
    sd.add_argument(
        "--no-verify",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip the pre-deploy hook (alias for top-level "
             "`brian --no-verify deploy`).",
    )
    sd.add_argument(
        "--platform",
        choices=["vast", "lightning"],
        default=None,
        help="Cloud provider to deploy on (default: brian.toml [deploy].platform, "
             "then 'vast'). Overrides the config for this run only.",
    )
    sd.set_defaults(func=cmd_deploy)

    # deploy-100k
    sd2 = sub.add_parser("deploy-100k",
                         help="Long-horizon DSL training run (100k steps)")
    sd2.add_argument("--branch")
    sd2.set_defaults(func=cmd_deploy_100k)

    # deploy-brain
    sdb = sub.add_parser("deploy-brain",
                         help="Launch a Brain (non-DSL) training run")
    sdb.add_argument("--steps", type=int, default=10_000)
    sdb.add_argument("--preset", default="rcc_bowtie_30m_p4")
    sdb.add_argument("--branch")
    sdb.set_defaults(func=cmd_deploy_brain)

    # deploy-discover — rent a vast.ai instance to run `brian discover <mode>`
    # (not training). Only experts/trunk/explore: the other discover modes
    # already finish in seconds/minutes on the free local Colab GPU.
    sdd = sub.add_parser(
        "deploy-discover",
        help="Launch a `brian discover <mode>` run on vast.ai (experts/trunk/"
             "explore only) — pushes logs+modulations+ledger while running")
    sdd.add_argument("deploy_discover_mode", choices=["experts", "trunk", "explore"],
                     metavar="mode", help="experts | trunk | explore")
    sdd.add_argument("--branch", help="git branch the vast.ai box checks out "
                     "(default: current branch)")
    sdd.add_argument("--label", help="vast.ai instance label / (trunk) saved-"
                     "modulation name")
    sdd.add_argument("--push-interval", type=int, default=None,
                     help="seconds between background artifact pushes while "
                          "running (default 90)")
    sdd.add_argument("--gpu-query", default=None,
                     help="override the vast.ai offer filter (default: a "
                          "single A100 — see scripts/vast_discover.sh)")
    # experts-only
    sdd.add_argument("--models", help="comma-sep HF ids/aliases (experts)")
    sdd.add_argument("--rounds", type=int, default=None, help="probe rounds (experts)")
    sdd.add_argument("--batch", type=int, default=None, help="batch size (experts)")
    sdd.add_argument("--seq_len", type=int, default=None, help="sequence length (experts)")
    # shared GA knobs (trunk/explore/experts)
    sdd.add_argument("--pop", type=int, default=None, help="population size")
    sdd.add_argument("--generations", type=int, default=None, help="GA generations")
    sdd.add_argument("--length", type=int, default=None, help="candidate program length")
    # trunk/explore
    sdd.add_argument("--steps", type=int, default=None,
                     help="train steps per eval (trunk) / total steps (explore)")
    sdd.add_argument("--seed", type=int, default=None)
    # unused by deploy-discover but accepted for a uniform Namespace shape
    sdd.add_argument("--task", default=None, help=argparse.SUPPRESS)
    sdd.add_argument("--from-scratch", action="store_true", help=argparse.SUPPRESS)
    sdd.add_argument("--novelty", type=float, default=None, help=argparse.SUPPRESS)
    sdd.add_argument("--avoid-known", action="store_true", help=argparse.SUPPRESS)
    sdd.add_argument("--macros", action="store_true", help=argparse.SUPPRESS)
    sdd.add_argument("--seed-from", default=None, help=argparse.SUPPRESS)
    sdd.set_defaults(func=cmd_deploy_discover)

    # logs
    sl = sub.add_parser(
        "logs",
        help="Tail container logs for a vast instance. "
             "Use --latest for the newest local log (no vast API call).",
        description=(
            "Show training logs for a vast.ai instance.\n\n"
            "Three modes:\n"
            "  brian logs <id>      Tail the live container log; if the\n"
            "                       instance is destroyed, fall back to the\n"
            "                       locally-pushed snapshot under logs/vast/.\n"
            "                       Runs `git fetch && git pull` once if the\n"
            "                       local file is missing (maybe a sibling\n"
            "                       workstation already pushed it).\n"
            "  brian logs --latest  Print the newest .log in logs/vast/ by\n"
            "                       mtime. Strictly local, no vast API call.\n"
            "                       Useful when the last instance was just\n"
            "                       destroyed and you want to know why."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sl.add_argument("instance_id", nargs="?", default=None,
                    help="vast.ai instance id (omit when using --latest)")
    sl.add_argument("--latest", action="store_true",
                    help="show the newest .log in logs/vast/ by mtime "
                         "(strictly local — no vast API call)")
    sl.set_defaults(func=cmd_logs)

    # status
    ss = sub.add_parser("status", help="List active vast instances (raw vastai view)")
    ss.set_defaults(func=cmd_status)

    # ps (parsed-status table — like `docker ps` for neuroslm runs across platforms)
    sps = sub.add_parser(
        "ps",
        help="List training jobs across all platforms (vast + Lightning) + "
             "parsed last metric line + phase")
    sps.add_argument("--all", action="store_true",
                     help="include non-neuroslm instances too (vast only)")
    sps.add_argument("-it", "--it", action="store_true",
                     help="interactive watch mode — redraw every --interval "
                          "seconds until Ctrl-C")
    sps.add_argument("--interval", type=float, default=1.0,
                     help="seconds between refreshes when --it is on (default 1)")
    sps.add_argument("--colab", metavar="URL",
                     help="connect to Colab log server URL (from cell 5b). "
                          "Shows training status from Colab notebook")
    sps.add_argument("--platform", choices=["all", "vast", "lightning"],
                     default="all",
                     help="Restrict the listing to one cloud platform "
                          "(default: all). Use `lightning` to skip the "
                          "vastai roundtrip when you only care about Studios.")
    sps.add_argument("--logs", metavar="JOB_ID", default=None,
                     help="Stream the remote training log for one job "
                          "(reads .brian/jobs/<JOB_ID>.json). Combine with "
                          "--it for live tail, --tail N for the line count "
                          "(default 200).")
    sps.add_argument("--tail", type=int, default=200,
                     help="Lines of log to pull when --logs is set (default 200)")
    sps.set_defaults(func=cmd_ps)

    # psi — alias for `ps --it --interval 1`
    spsi = sub.add_parser("psi", help="Interactive process watch (alias: ps --it --interval 1)")
    spsi.add_argument("--interval", type=float, default=1.0,
                      help="seconds between refreshes (default 1)")
    spsi.add_argument("--platform", choices=["all", "vast", "lightning"],
                      default="all",
                      help="restrict to one cloud platform (default: all)")
    spsi.add_argument("--all", action="store_true",
                      help="include non-neuroslm instances too (vast only)")
    spsi.set_defaults(func=lambda a: cmd_ps(argparse.Namespace(
        it=True, interval=a.interval, platform=a.platform,
        all=a.all, colab=None, logs=None, tail=200,
    )))

    # destroy
    sde = sub.add_parser("destroy", help="Tear down vast instance(s)")
    sde.add_argument("instance_id", nargs="?")
    sde.add_argument("--all", action="store_true",
                     help="destroy every neuroslm-* labelled instance")
    sde.set_defaults(func=cmd_destroy)

    # nuke — destroy ALL instances immediately (requires TTY + "nuke" confirmation)
    snuke = sub.add_parser(
        "nuke",
        help="Destroy ALL running neuroslm-labelled instances (requires TTY confirmation)",
    )
    snuke.set_defaults(func=cmd_nuke)

    # stop (per-job halt — currently Lightning-only; vast uses `destroy`)
    sstop = sub.add_parser(
        "stop",
        help="Stop a remote training job by job_id (Lightning AI). "
             "Reads .brian/jobs/<JOB_ID>.json and stops the underlying Studio.")
    sstop.add_argument("job_id",
                       help="Job id printed by `brian deploy` and listed "
                            "in `brian ps --platform lightning`.")
    sstop.set_defaults(func=cmd_stop)

    # ood (legacy — explicit ckpt path)
    so = sub.add_parser("ood", help="Run OOD eval on a checkpoint")
    so.add_argument("ckpt", help="ckpt path (e.g. lfs_checkpoints/dsl_arch_step10000.pt)")
    so.add_argument("--branch")
    so.add_argument("--tag", default="eval", help="role tag for the eval JSON")
    so.add_argument("--windows", type=int)
    so.set_defaults(func=cmd_ood)

    # ai (group: `brian ai <skill>` — every dir under agents/skills/
    # with an INSTRUCTIONS.md becomes a subcommand automatically.)
    sa_ai = sub.add_parser("ai", help="AI-assisted chores")
    esa_ai = sa_ai.add_subparsers(dest="ai_kind", required=True,
                                   help="skill name (auto-discovered from "
                                        "agents/skills/)")
    for _skill_name in _list_ai_skills():
        _sp = esa_ai.add_parser(
            _skill_name,
            help=f"Run agents/skills/{_skill_name}/INSTRUCTIONS.md via claude")
        _sp.add_argument("--auto", action="store_true",
                          help="--dangerously-skip-permissions for unattended")
    sa_ai.set_defaults(func=cmd_ai)

    # eval (group: `brian eval ood [<ckpt>|--latest|<picker>]`)
    se = sub.add_parser("eval",
                        help="Evaluate a checkpoint (ood/...) — by path, --latest, or picker")
    ese = se.add_subparsers(dest="eval_kind", required=True)
    ese_ood = ese.add_parser(
        "ood",
        help="OOD eval on a DSL checkpoint. Usage:\n"
             "  brian eval ood path/to.pt    — explicit path\n"
             "  brian eval ood --latest      — highest-step checkpoint\n"
             "  brian eval ood               — interactive picker")
    # Positional arg accepts either a checkpoint path or the literal
    # "--latest" token (so `brian eval ood --latest` reads naturally).
    ese_ood.add_argument("ckpt_pos", nargs="?",
                         help="checkpoint path or '--latest'")
    ese_ood.add_argument("--checkpoint",
                         help="alias for the positional checkpoint")
    ese_ood.add_argument("--latest", action="store_true",
                         help="pick the highest-step dsl_arch_*.pt")
    ese_ood.add_argument("--branch")
    ese_ood.add_argument("--tag",
                         help="role tag (defaults to step number)")
    ese_ood.add_argument("--windows", type=int)
    se.set_defaults(func=cmd_eval)

    # analyze-log
    sal = sub.add_parser("analyze-log",
                         help="Parse log file → docs/metrics.md + claude → docs/FINDINGS.md")
    sal.add_argument("logfile", nargs="?",
                     help="path to a training or OOD log file")
    sal.add_argument("--run-id", help="override the run id used as table key")
    sal.add_argument("--branch", help="override the branch column")
    sal.add_argument("--no-claude", action="store_true",
                     help="skip the claude CLI insight extraction")
    sal.add_argument("--scan-ood", action="store_true",
                     help="scan logs/vast/benchmarks/ood/ and upsert every JSON")
    sal.add_argument("--latest", action="store_true",
                     help="auto-discover the newest training log under "
                          "logs/vast/ + its matching OOD JSONs and analyze "
                          "them together")
    sal.set_defaults(func=cmd_analyze_log)

    # best update — scan logs, write .brian/best_run.ln
    sbest = sub.add_parser("best",
                           help="Detect best run and write .brian/best_run.ln")
    sbest_sub = sbest.add_subparsers(dest="best_cmd", required=True)
    sbest_update = sbest_sub.add_parser("update",
                                        help="Scan logs, pick best, write .brian/best_run.ln")
    sbest_update.add_argument("--metric", default="gap_ratio",
                              choices=["gap_ratio", "ppl", "ood_ppl"],
                              help="ranking metric (default: gap_ratio)")
    sbest_update.add_argument("--log-dir", default="logs",
                              help="directory to scan (default: logs)")
    sbest_update.set_defaults(func=cmd_best_update)

    # train (run training from command line)
    str_train = sub.add_parser("train",
                               help="Train from evol.dna or run minimal CPU training")
    str_train.add_argument("--preset", default="rcc_bowtie_30m_p4",
                          help="training preset (default: rcc_bowtie_30m_p4). "
                               "Use 'tiny' for minimal CPU training")
    str_train.add_argument("--arch", help="architecture name or path to .dna file (e.g., dna/evol/arch.dna)")
    str_train.add_argument("--dna", help="path to evolved DNA file (e.g., dna/evol/arch.dna)")
    str_train.add_argument("--steps", type=int, help="number of training steps (default: 40000 for tiny)")
    str_train.add_argument("--ood_every", type=int, help="OOD eval frequency (default: 500)")
    str_train.add_argument("--batch", type=int, help="batch size")
    str_train.add_argument("--seq_len", type=int, help="sequence length")
    str_train.add_argument("--d_sem", type=int, help="semantic dimension")
    str_train.set_defaults(func=cmd_train, ood_every=500)

    # test (group: `brian test {quick,fast,full}` plus the legacy
    # `brian test [pattern]` form for backward compatibility).
    #
    # Dispatch is manual (no add_subparsers) because the legacy form
    # accepts an arbitrary path/pattern as the first positional, which
    # argparse's subparsers won't tolerate.
    st = sub.add_parser(
        "test",
        help="Run pytest — use quick/fast/full instead of bare pytest. "
             "(See CLAUDE.md: contributors must NOT call pytest directly.)",
        description=(
            "Unified test driver. Use one of the subcommands below "
            "instead of running pytest directly:\n\n"
            "  brian test quick   run the 30 most-recently-modified test files\n"
            "  brian test fast    run the 30 fastest tests (cached durations)\n"
            "  brian test full    canonical full sweep + refresh duration cache\n"
            "  brian test [PATH]  (legacy) pytest on tests/dsl/ or PATH"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    st.add_argument(
        "kind_or_pattern", nargs="?", default=None,
        help="one of {quick, fast, full}, or a legacy pytest path/pattern",
    )
    st.add_argument("-v", "--verbose", action="store_true",
                    help="verbose pytest output")
    st.add_argument("--slow", action="store_true",
                    help="(legacy only) include slow tests "
                         "(by default slow tests are skipped)")

    # ``add_subparsers`` is intentionally NOT used; we still need to
    # advertise the magic words in --help (already done in
    # description=) and synthesise a Namespace with the legacy
    # ``pattern`` attr the legacy ``cmd_test`` expects.
    #
    # The dispatch table maps to function NAMES (not callables) so
    # ``monkeypatch.setattr(cli, 'cmd_test_full', ...)`` in tests
    # actually rebinds the call target — a callable lookup would
    # capture the pre-patch reference at parser-build time.
    _TEST_SUBCOMMANDS = {
        "quick": "cmd_test_quick",
        "fast":  "cmd_test_fast",
        "full":  "cmd_test_full",
    }

    def _dispatch_test(a):
        kop = a.kind_or_pattern
        # Magic word → dedicated subcommand. Look up the live module
        # attribute so monkeypatch works in tests.
        if kop in _TEST_SUBCOMMANDS:
            import neuroslm.cli as _self_mod
            func = getattr(_self_mod, _TEST_SUBCOMMANDS[kop])
            return func(argparse.Namespace(
                verbose=a.verbose, test_kind=kop,
            ))
        # Else: legacy form. ``cmd_test`` expects ``pattern`` + ``slow``
        # attrs; synthesise them from kind_or_pattern.
        return cmd_test(argparse.Namespace(
            pattern=kop, verbose=a.verbose, slow=a.slow,
        ))

    st.set_defaults(func=_dispatch_test)

    # push
    sp = sub.add_parser("push",
                        help="Push current branch via PAT (no credential helper)")
    sp.set_defaults(func=cmd_push)

    # ── hf : HuggingFace Hub checkpoint operations ────────────────────
    shf = sub.add_parser(
        "hf",
        help="List / download checkpoints from HuggingFace Hub.")
    shf_sub = shf.add_subparsers(
        dest="hf_cmd", required=True,
        title="hf subcommands",
        description=(
            "  brian hf list   [--repo R] [--prefix P] [--limit N]\n"
            "  brian hf pull   PATH_OR_URI  [--repo R] [--out DIR]\n"
            "  brian hf pull   --latest    [--repo R] [--prefix P]\n"
            "  brian hf latest [--repo R] [--prefix P]"
        ),
    )
    # hf list
    shf_l = shf_sub.add_parser(
        "list", help="List every checkpoint on the repo (newest first).")
    shf_l.add_argument(
        "--repo", default=None, metavar="OWNER/REPO",
        help="Target HF repo (default: HF_REPO_ID env, then "
             "moritzroessler/BRIAN).")
    shf_l.add_argument(
        "--prefix", default=None, metavar="RUN_DIR",
        help="Filter to checkpoints under this run-dir prefix.")
    shf_l.add_argument(
        "--limit", type=int, default=20,
        help="Cap rows printed (default 20). Pass 0 for no cap.")
    shf_l.set_defaults(func=cmd_hf, hf_cmd="list")
    # hf pull
    shf_p = shf_sub.add_parser(
        "pull",
        help="Download a checkpoint into ./lfs_checkpoints (or --out).")
    shf_p.add_argument(
        "target", nargs="?", default=None, metavar="PATH_IN_REPO",
        help="Path on the repo (e.g. checkpoints/run-A/step5000.pt) "
             "or hf://owner/repo/path URI. Omit when using --latest.")
    shf_p.add_argument(
        "--latest", action="store_true",
        help="Pull the highest-step checkpoint instead of a named path.")
    shf_p.add_argument(
        "--repo", default=None, metavar="OWNER/REPO",
        help="Target HF repo (default: HF_REPO_ID env, then "
             "moritzroessler/BRIAN).")
    shf_p.add_argument(
        "--prefix", default=None, metavar="RUN_DIR",
        help="(With --latest) scope to checkpoints under this prefix.")
    shf_p.add_argument(
        "--out", default=None, metavar="DIR",
        help="Override the destination root (default: ./lfs_checkpoints).")
    shf_p.add_argument(
        "--force", action="store_true",
        help="Bypass the huggingface_hub local cache and re-fetch.")
    shf_p.set_defaults(func=cmd_hf, hf_cmd="pull")
    # hf latest
    shf_t = shf_sub.add_parser(
        "latest",
        help="Print the highest-step checkpoint URI (no download).")
    shf_t.add_argument(
        "--repo", default=None, metavar="OWNER/REPO",
        help="Target HF repo (default: HF_REPO_ID env, then "
             "moritzroessler/BRIAN).")
    shf_t.add_argument(
        "--prefix", default=None, metavar="RUN_DIR",
        help="Scope to checkpoints under this run-dir prefix.")
    shf_t.set_defaults(func=cmd_hf, hf_cmd="latest")

    # ── chat : always-on inference daemon with dashboard ──────────────
    sc_chat = sub.add_parser(
        "chat",
        help="Boot a checkpoint into an always-on inference daemon "
             "with memory + idle thoughts on a CLI dashboard.")
    sc_chat.add_argument(
        "ckpt", nargs="?", default=None,
        help="Local checkpoint path (positional). Omit all args to "
             "auto-pull the best-run checkpoint from "
             ".brian/checkpoint.ln (default). Use --pt/--latest/"
             "--no-best to control resolution.")
    sc_chat.add_argument(
        "--pt", default=None, metavar="PATH_OR_URI",
        help="Explicit checkpoint override (named-flag alias for "
             "the positional). Accepts a local path or hf:// URI. "
             "Wins over every other resolution mode.")
    sc_chat.add_argument(
        "--latest", action="store_true",
        help="Download the highest-step checkpoint from HF Hub before "
             "booting (by step number, NOT by best-run score; for "
             "best-by-score use the default behaviour).")
    sc_chat.add_argument(
        "--no-best", dest="no_best", action="store_true",
        help="Disable the default best-run auto-pull (skip "
             ".brian/checkpoint.ln). Useful for offline laptops.")
    sc_chat.add_argument(
        "--repo", default=None, metavar="OWNER/REPO",
        help="HF repo for --latest (default: HF_REPO_ID env, then "
             "moritzroessler/BRIAN).")
    sc_chat.add_argument(
        "--prefix", default=None, metavar="RUN_DIR",
        help="(With --latest) scope to checkpoints under this prefix.")
    sc_chat.add_argument(
        "--arch", default=None, metavar="PATH",
        help="Path to the architecture folder (containing arch.neuro). "
             "Default: auto-detect from checkpoint dir, then "
             "architectures/SmolLM.")
    sc_chat.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default cpu — the daemon is built for "
             "laptop-resident always-on use).")
    sc_chat.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default 0.8).")
    sc_chat.add_argument(
        "--top-k", dest="top_k", type=int, default=40,
        help="Top-k filter for sampling (default 40, 0 = disabled).")
    sc_chat.add_argument(
        "--max-new-tokens", dest="max_new_tokens", type=int, default=96,
        help="User-turn token budget (default 96).")
    sc_chat.add_argument(
        "--thought-tokens", dest="thought_tokens", type=int, default=32,
        help="Idle-thought token budget (default 32).")
    sc_chat.add_argument(
        "--thought-period", dest="thought_period", type=float,
        default=12.0,
        help="Seconds between idle-thought ticks (default 12).")
    sc_chat.add_argument(
        "--idle-threshold", dest="idle_threshold", type=float,
        default=6.0,
        help="Seconds of user inactivity before thoughts start firing "
             "(default 6).")
    sc_chat.add_argument(
        "--no-color", dest="no_color", action="store_true",
        help="Disable ANSI colour in the dashboard.")
    sc_chat.add_argument(
        "--no-thoughts", dest="no_thoughts", action="store_true",
        help="Disable the idle-thought thread (chat-only mode).")
    sc_chat.set_defaults(func=cmd_chat)

    # clean — reference-aware repo janitor
    sc_clean = sub.add_parser(
        "clean",
        help="Find + delete unreferenced logs / checkpoints / docs / lfs (default dry-run)")
    sc_clean.add_argument(
        "bucket", nargs="+",
        choices=["logs", "checkpoints", "docs", "lfs", "all"],
        help="which bucket(s) to clean ('lfs' = per-run LFS pruner with "
             "log-gated best-protection)")
    sc_clean.add_argument(
        "--force", action="store_true",
        help="actually delete (default: dry-run only)")
    sc_clean.add_argument(
        "-v", "--verbose", action="store_true",
        help="also list every kept file with the reason it was kept")
    sc_clean.add_argument(
        "--keep-recent", type=int, default=3,
        help="number of most-recent files per bucket to always keep "
             "(default 3)")
    sc_clean.add_argument(
        "--no-git", action="store_true",
        help="don't stage deletions via `git rm` — plain unlink only")
    sc_clean.set_defaults(func=cmd_clean)

    # migrate — versioned, ledger-tracked repo migrations
    sc_mig = sub.add_parser(
        "migrate",
        help="Run versioned repo migrations (default dry-run; --force to apply)")
    sc_mig.add_argument(
        "id", nargs="?", default=None,
        help="migration id to run (omit + use --list/--all)")
    sc_mig.add_argument(
        "--list", action="store_true",
        help="list every discovered migration with its status")
    sc_mig.add_argument(
        "--all", action="store_true",
        help="run every PENDING migration in order")
    sc_mig.add_argument(
        "--force", action="store_true",
        help="actually apply (default: dry-run only)")
    sc_mig.add_argument(
        "--rerun", action="store_true",
        help="re-apply a migration that is already in the ledger "
             "(escape hatch — usually a no-op)")
    sc_mig.add_argument(
        "-v", "--verbose", action="store_true",
        help="show reference-scan progress")
    sc_mig.set_defaults(func=cmd_migrate)

    # checkpoints — local checkpoint manager
    sc_ckpts = sub.add_parser(
        "checkpoints",
        help="List, download, and manage local inference checkpoints")
    sc_ckpts_sub = sc_ckpts.add_subparsers(dest="ckpts_cmd", required=True)

    sc_ckpts_list = sc_ckpts_sub.add_parser(
        "list", help="List all HF Hub checkpoints (newest first)")
    sc_ckpts_list.add_argument(
        "--repo", default=None,
        help="HF repo id (default: moritzroessler/BRIAN)")
    sc_ckpts_list.add_argument(
        "--prefix", default=None,
        help="Filter by run-dir prefix")
    sc_ckpts_list.add_argument(
        "--limit", type=int, default=20,
        help="Max entries to show (default 20)")

    sc_ckpts_dl = sc_ckpts_sub.add_parser(
        "download",
        help="Download a checkpoint into local checkpoints/ and optionally "
             "set it as active")
    sc_ckpts_dl.add_argument(
        "target", nargs="?", default=None,
        help="Step number, path-in-repo, or hf:// URI")
    sc_ckpts_dl.add_argument(
        "--latest", action="store_true",
        help="Download the highest-step checkpoint")
    sc_ckpts_dl.add_argument(
        "--repo", default=None, help="HF repo id")
    sc_ckpts_dl.add_argument(
        "--prefix", default=None, help="Filter by run-dir prefix")
    sc_ckpts_dl.add_argument(
        "--no-activate", dest="activate", action="store_false",
        help="Don't write .neuro/checkpoint.ln after download")
    sc_ckpts_dl.set_defaults(activate=True)

    sc_ckpts_use = sc_ckpts_sub.add_parser(
        "use",
        help="Set an already-downloaded .pt file as the active local "
             "checkpoint (.neuro/checkpoint.ln)")
    sc_ckpts_use.add_argument(
        "path", help="Local path to the .pt file")

    sc_ckpts_active = sc_ckpts_sub.add_parser(
        "active",
        help="Show the current active local checkpoint "
             "(.neuro/checkpoint.ln)")
    sc_ckpts_active.set_defaults()

    sc_ckpts.set_defaults(func=cmd_checkpoints)

    # update-readme
    sur = sub.add_parser(
        "update-readme",
        help="Render docs/README.template.md → README.md (pre-commit: --check)")
    sur.add_argument(
        "--check", action="store_true",
        help="compare rendered output to README.md without writing; exit 1 if stale")
    sur.add_argument(
        "--no-best-update", dest="no_best_update", action="store_true",
        help="skip the auto-refresh of .brian/best_run.ln before "
             "rendering (CI / tests where logs/ may be unstable).")
    sur.set_defaults(func=cmd_update_readme)

    # ── help ──────────────────────────────────────────────────────────────────
    from neuroslm.cli_help import cmd_help, cmd_tease, cmd_cite

    shelp = sub.add_parser(
        "help",
        help="Browse docs and platform descriptions in the terminal.")
    shelp.add_argument(
        "topic", nargs="?", default=None,
        help="Topic to display: a doc name (cli, dsl, ste, runs, findings, …) "
             "or 'platforms'. Omit for the full topic index.")
    shelp.set_defaults(func=cmd_help)

    # ── tease ─────────────────────────────────────────────────────────────────
    stease = sub.add_parser(
        "tease",
        help="Peek at ledgers and logs: runs, log, findings.")
    stease_sub = stease.add_subparsers(dest="what", required=True)

    stease_runs = stease_sub.add_parser(
        "runs", help="Show N most-recent run ledger entries (docs/runs.md).")
    stease_runs.add_argument(
        "--n", type=int, default=5,
        help="number of entries to show (default: 5)")
    stease_runs.add_argument(
        "--format", choices=["terminal", "md"], default="terminal",
        help="output format: terminal table (default) or markdown")
    stease_runs.set_defaults(func=cmd_tease)

    stease_log = stease_sub.add_parser(
        "log",
        help="Tail N lines from a training log (defaults to best run).")
    stease_log.add_argument(
        "path", nargs="?", default=None,
        help="path to log file (relative to repo root). "
             "Omit to use .brian/best_run.ln.")
    stease_log.add_argument(
        "--tail", type=int, default=10,
        help="number of lines to tail (default: 10)")
    stease_log.add_argument(
        "--best", action="store_true",
        help="explicitly tail the best-run log (.brian/best_run.ln)")
    stease_log.set_defaults(func=cmd_tease)

    stease_findings = stease_sub.add_parser(
        "findings",
        help="Show N most-recent hypothesis entries from docs/findings.md.")
    stease_findings.add_argument(
        "--n", type=int, default=3,
        help="number of entries to show (default: 3)")
    stease_findings.set_defaults(func=cmd_tease)

    # ── cite ──────────────────────────────────────────────────────────────────
    scite = sub.add_parser(
        "cite",
        help="Format a citation for a run in the ledger (docs/runs.md).")
    scite.add_argument(
        "run_id", nargs="?", default=None,
        help="run ID to cite (e.g. 20260615-175931). "
             "Omit and use --list to browse.")
    scite.add_argument(
        "--list", action="store_true",
        help="list all citable run IDs from docs/runs.md")
    scite.add_argument(
        "--format", choices=["md", "text"], default="md",
        help="output format (default: md)")
    scite.set_defaults(func=cmd_cite)

    # ── studio ────────────────────────────────────────────────────────────────
    sstudio = sub.add_parser(
        "studio",
        help="Brian Studio - visual language model editor (REST+MCP server + Next.js UI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Brian Studio: build, compose, test and deploy language models "
            "using a visual drag-and-drop editor.\n\n"
            "  Server: http://localhost:1984  (port 1984 - Orwell)\n"
            "  Studio: http://localhost:3141  (port 3141 - pi)\n"
            "  MCP:    http://localhost:1984/mcp"
        ),
    )
    sstudio_sub = sstudio.add_subparsers(dest="studio_cmd", required=True)

    sstudio_start = sstudio_sub.add_parser(
        "start",
        help="Start the Brian Studio server (port 1984) and client (port 3141)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Launches the Brian Studio REST+MCP server on port 1984 and the\n"
            "Next.js visual editor on port 3141 (pi). Opens a browser automatically.\n\n"
            "  Server (REST+MCP): http://localhost:1984\n"
            "  Studio (Next.js):  http://localhost:3141\n"
            "  API docs:          http://localhost:1984/docs"
        ),
    )
    sstudio_start.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically")
    sstudio_start.add_argument(
        "--server-only", action="store_true",
        help="Start only the Python REST+MCP server, not the Next.js client")
    sstudio_start.add_argument(
        "--host", default="0.0.0.0",
        help="Server bind host (default: 0.0.0.0)")
    sstudio_start.set_defaults(func=cmd_studio_start)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
