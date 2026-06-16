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
    deploy [--steps N]        Launch DSL training run on vast (default 10k)
    deploy-100k               Long DSL run (100k steps)
    deploy-brain [...]        Launch a Brain (non-DSL) training run
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

def _deploy_dsl(steps: int, branch: Optional[str], extra_env: dict,
                 ood_every: int = 0,
                 log_every: int = 0, save_every: int = 0,
                 push_every: int = 0) -> int:
    """Run _deploy_train.py with the appropriate env vars for DSL training.

    The cadence triple ``(log_every, save_every, push_every)`` propagates
    via env vars ``LOG_EVERY`` / ``SAVE_EVERY`` / ``PUSH_EVERY`` →
    ``_deploy_train.py`` bakes them into the ONSTART script →
    ``vast_train_dsl_loop.sh`` reads them and forwards as ``--log_every``
    / ``--save_every`` / ``--push_every`` to ``python -m neuroslm.train_dsl``.
    Zero means "use the trainer's own default".
    """
    env = os.environ.copy()
    env["STEPS"] = str(steps)
    if ood_every > 0:
        env["OOD_EVERY"] = str(ood_every)
    if log_every > 0:
        env["LOG_EVERY"] = str(log_every)
    if save_every > 0:
        env["SAVE_EVERY"] = str(save_every)
    if push_every > 0:
        env["PUSH_EVERY"] = str(push_every)
    if branch:
        env["BRANCH"] = branch
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)
    # Use _deploy_train.py (fast, direct vast.ai API call) instead of
    # vast_train.sh which hangs on Windows due to heredoc pipe issues.
    deploy_script = REPO_ROOT / "_deploy_train.py"
    python = _find_deploy_python()
    return subprocess.call([python, str(deploy_script)], cwd=str(REPO_ROOT), env=env)


def _deploy_dna(dna_path: str, steps: int, branch: Optional[str], extra_env: dict,
                ood_every: int = 0,
                log_every: int = 0, save_every: int = 0,
                push_every: int = 0) -> int:
    """Deploy a DNA-driven training run on vast.ai through the canonical
    workspace pipeline.

    The DNA is compiled to DSL → HypergraphIR LOCALLY (via
    ``prepare_run_workspace``) before any vast.ai network call. Two
    consequences:

      1. Bad DNA fails fast — the user never pays for vast.ai
         provisioning when their snapshot is broken.
      2. ``_deploy_train.py`` (and the on-box bash wrapper) consume the
         pre-compiled ``.neuro/arch/temp/`` workspace, not a raw .dna
         file. There is exactly ONE DNA→DSL unfold in the entire deploy
         path, in this function, just like ``cmd_train``.

    The vast.ai box sees ``ARCH=.neuro/arch/temp`` and runs the DSL
    training path. The DNA snapshot still ships in the git clone so
    on-box evolution callbacks (mutation persistence) can rewrite it.
    """
    # ── 1. compile DNA locally (fail-fast before any vast.ai call) ──
    try:
        from neuroslm.compiler.run_workspace import prepare_run_workspace
        workspace = prepare_run_workspace(dna=dna_path)
    except Exception as e:
        print(f"[deploy] workspace preparation failed: {e}", file=sys.stderr)
        print(f"[deploy] aborting — no vast.ai resources were provisioned.",
              file=sys.stderr)
        return 1
    print(f"[deploy] workspace ready: {workspace.arch_root}")
    print(f"        source: {workspace.source_kind}={workspace.source_path}")
    print(f"        hypergraph IR: {len(workspace.hypergraph_ir.nodes)} nodes, "
          f"{len(workspace.hypergraph_ir.hyperedges)} edges")

    # ── 2. hand off the prepared workspace path to the deploy script ──
    # We pass ARCH=<workspace> (NOT DNA=...) so _deploy_train.py reads
    # the pre-compiled tree via the existing DSL code path. The original
    # DNA path is exposed as BRIAN_SOURCE_DNA for telemetry / labels
    # only — no code on the vast.ai box should compile it again.
    env = os.environ.copy()
    env["ARCH"] = str(workspace.arch_root)
    env["BRIAN_SOURCE_DNA"] = dna_path
    env["STEPS"] = str(steps)
    if ood_every > 0:
        env["OOD_EVERY"] = str(ood_every)
    if log_every > 0:
        env["LOG_EVERY"] = str(log_every)
    if save_every > 0:
        env["SAVE_EVERY"] = str(save_every)
    if push_every > 0:
        env["PUSH_EVERY"] = str(push_every)
    if branch:
        env["BRANCH"] = branch
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)
    deploy_script = REPO_ROOT / "_deploy_train.py"
    python = _find_deploy_python()
    return subprocess.call([python, str(deploy_script)], cwd=str(REPO_ROOT), env=env)


def _find_deploy_python() -> str:
    """Return the Python interpreter to run ``_deploy_train.py`` with.

    Per CLAUDE.md §13 we have exactly one venv (``./.venv``). The
    canonical interpreter is whatever's currently running — it already
    has every dep the deploy script needs (torch is no longer required
    after the canonical-pipeline refactor; vastai is the only hard
    requirement, and it lives in ``./.venv``).
    """
    return sys.executable


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


def cmd_deploy(args: argparse.Namespace) -> int:
    """Launch a DSL or DNA training run on vast.ai.

    Source-of-truth precedence (highest wins) for ``steps``, ``branch``,
    and ``dna``::

        1. CLI flag (``--steps`` / ``--branch`` / ``--dna``)
        2. ``brian.toml`` (``[defaults].steps`` / ``[defaults].branch``
           / ``[current].dna``)
        3. Hardcoded fallback (``10_000`` steps, no BRANCH env →
           ``_deploy_train.py``'s own default kicks in)

    This matches the same precedence rule the rest of the CLI uses
    (CLI > workspace config > sensible default). It also makes the
    one-line invocation ``brian deploy`` produce a DEFINED training
    run: brian.toml is the single source of truth, the CLI is for
    one-off overrides.

    Examples::

        # Use brian.toml [defaults].steps + [defaults].branch + [current].dna
        brian deploy

        # Override just the step count, keep brian.toml's branch + dna
        brian deploy --steps 50000

        # Override branch, DSL mode (no DNA in brian.toml)
        brian deploy --branch feature/new-arch
    """
    # ── Pre-deploy hook (cross-platform, YAML-declared) ──
    # See ``hooks/pre-deploy.yaml`` + ``hooks/scripts/pre-deploy.{sh,ps1}``.
    # Default contract: recompile master arch → DNA → unfold into
    # ``architectures/current``. A non-zero return aborts the deploy
    # BEFORE any vast.ai call so the user never pays for a bad arch.
    #
    # Escape hatch: ``brian --no-verify deploy ...`` (or
    # ``brian deploy --no-verify ...``) short-circuits the hook for
    # quick-iteration cycles where the working tree is intentionally
    # dirty or the hook is known-redundant. We ALWAYS print a notice
    # so silent omission can't masquerade as a successful hook run.
    if getattr(args, "no_verify", False):
        print("[deploy] pre-deploy hook SKIPPED (--no-verify)")
    else:
        hook_rc = _run_hook("pre-deploy")
        if hook_rc != 0:
            print(f"[deploy] pre-deploy hook failed (exit {hook_rc}); aborting.",
                  file=sys.stderr)
            return hook_rc

    # ── Single config load: amortise the brian.toml read ──
    # All three CLI flags consult the same config; loading it once
    # avoids re-parsing the TOML file three times AND keeps the
    # precedence rule (CLI > brian.toml > hardcoded) consistent.
    from neuroslm.project_config import load_project_config
    cfg = load_project_config()

    # ── Steps: CLI > brian.toml [defaults].steps > 10_000 ──
    steps = args.steps
    if steps is None:
        steps = cfg.default_steps if cfg.default_steps > 0 else 10_000

    # ── Branch: CLI > brian.toml [defaults].branch > leave unset ──
    # When neither layer specifies a branch, we deliberately DON'T set
    # the BRANCH env var so the downstream ``_deploy_train.py`` falls
    # through to its own hardcoded fallback (a git HEAD lookup). This
    # avoids hardcoding "master" here — different workspaces / forks
    # use different default branches.
    branch = args.branch
    if branch is None and cfg.default_branch:
        branch = cfg.default_branch

    # ── DNA: CLI > brian.toml [current].dna (when file exists) > None ──
    # The ``is_dna_mode`` check tests file existence too, so a stale
    # ``[current].dna`` pointing at a deleted file won't silently switch
    # us into DNA mode with garbage.
    dna_path = getattr(args, "dna", None)
    if not dna_path and cfg.is_dna_mode:
        dna_path = cfg.dna
        print(f"[deploy] brian.toml DNA mode: {dna_path}")

    ood = args.ood if args.ood else 0
    extra = {}
    if args.scale:
        extra["SCALE"] = args.scale
    if getattr(args, "label", None):
        extra["LABEL_SUFFIX"] = args.label

    # ── Cadence: brian.toml [defaults] > trainer default ──
    # The trainer treats 0 as "use my own default"; we mirror that.
    # If a user wants a CLI override later, the place to add it is a
    # new ``--push-every`` arg on the deploy subcommand; for now the
    # source of truth is ``brian.toml [defaults]``.
    log_every = cfg.default_log_every
    save_every = cfg.default_save_every
    push_every = cfg.default_push_every

    if dna_path:
        return _deploy_dna(dna_path=dna_path, steps=steps,
                           branch=branch, extra_env=extra, ood_every=ood,
                           log_every=log_every, save_every=save_every,
                           push_every=push_every)
    else:
        return _deploy_dsl(steps=steps, branch=branch,
                           extra_env=extra, ood_every=ood,
                           log_every=log_every, save_every=save_every,
                           push_every=push_every)


def cmd_deploy_100k(args: argparse.Namespace) -> int:
    """Shortcut for a long-horizon (100k steps) DSL run."""
    return _deploy_dsl(steps=100_000, branch=args.branch, extra_env={})


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
    """List active vast.ai neuroslm instances + parsed last metric line.

    Output columns:
      ID | LABEL | GPU | $/hr | UPTIME | PHASE | STEP | PPL | OOD-PPL | TOK/S

    With --it (interactive/watch), redraws every `--interval` seconds
    until Ctrl-C. Uses double-buffering + cursor-home + clear-to-end-of-
    screen so the redraw is seamless (no flicker, no black flash).
    
    With --colab <url>, connects to a Colab log server and displays
    training status from the remote notebook.
    """
    # Colab mode — connect to remote log server
    if getattr(args, "colab", None):
        if args.it:
            return _ps_colab_watch(args)
        return _ps_colab_once(args)
    if args.it:
        return _ps_watch(args)
    return _render_ps_once(args)


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


def _render_ps_once(args: argparse.Namespace, out=None) -> int:
    """Single ps render — extracted from cmd_ps so --it can call it in a loop.

    `out`: optional file-like (e.g. io.StringIO from the watch loop). When
    set, all output goes there instead of stdout; status/error messages
    that would normally hit stderr also route to `out` so the watch
    redraw stays atomic.
    """
    import json
    sink = out if out is not None else sys.stdout
    def _say(msg: str = "") -> None:
        sink.write(msg + "\n")
    # Polished header (one-time; not re-emitted in watch mode because the
    # caller already prints the per-tick timestamp line).
    if out is None:
        _say("Brian Task Manager — vast.ai instance + training monitor")
        _say("  Interactive: brian ps --it --interval 1")
        _say("  All GPUs:    brian ps --all")
        _say("")
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

    # ── Recently destroyed instances ──
    destroyed = _scan_recent_destroyed(top_n=5)
    if destroyed:
        _say()
        _say("Recent destroyed:")
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


def cmd_test(args: argparse.Namespace) -> int:
    path = args.pattern if args.pattern else "tests/dsl/"
    cli = [sys.executable, "-m", "pytest", path, "-q"]
    if not args.slow:
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
        if line.startswith("GITHUB_PAT="):
            pat = line.split("=", 1)[1].strip()
            break
        if line.startswith("GITHUB="):
            pat = line.split("=", 1)[1].strip()
    if not pat:
        print("no GITHUB_PAT found in .env")
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


# ── update-readme ──────────────────────────────────────────────────────

def cmd_update_readme(args: argparse.Namespace) -> int:
    """Render README.template.md → README.md using docs/readme_metrics.toml.

    ``brian update-readme``         — write README.md in place.
    ``brian update-readme --check`` — compare only; exit 1 if stale (pre-commit use).
    """
    from neuroslm.readme_renderer_v2 import (
    ReadmeRenderError,
    MissingMetricError,
    MissingClaimError,
    LogNotFoundError,
    render_readme
)

    template = REPO_ROOT / "README.template.md"
    metrics  = REPO_ROOT / "docs" / "readme_metrics.toml"
    output   = REPO_ROOT / "README.md"

    try:
        rendered, is_clean = render_readme(
            template, metrics, output, check=args.check
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
                    "(e.g. 100m, 300m, 1b). Default: arch's scales.default")
    sd.add_argument("--dna", help="path to evolved DNA file for training "
                    "(e.g., dna/evol/arch.dna). If set, trains from DNA instead of DSL arch")
    sd.add_argument("--label", help="Label suffix for the vast.ai instance")
    sd.add_argument("--ood", type=int, nargs="?", const=3000,
                    help="Run mid-training OOD eval every N steps "
                         "(default 3000 if flag passed without value)")
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

    # ps (parsed-status table — like `docker ps` for neuroslm runs)
    sps = sub.add_parser(
        "ps",
        help="List neuroslm instances + parsed last metric line + phase")
    sps.add_argument("--all", action="store_true",
                     help="include non-neuroslm instances too")
    sps.add_argument("-it", "--it", action="store_true",
                     help="interactive watch mode — redraw every --interval "
                          "seconds until Ctrl-C")
    sps.add_argument("--interval", type=float, default=1.0,
                     help="seconds between refreshes when --it is on (default 1)")
    sps.add_argument("--colab", metavar="URL",
                     help="connect to Colab log server URL (from cell 5b). "
                          "Shows training status from Colab notebook")
    sps.set_defaults(func=cmd_ps)

    # destroy
    sde = sub.add_parser("destroy", help="Tear down vast instance(s)")
    sde.add_argument("instance_id", nargs="?")
    sde.add_argument("--all", action="store_true",
                     help="destroy every neuroslm-* labelled instance")
    sde.set_defaults(func=cmd_destroy)

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

    # update-readme
    sur = sub.add_parser(
        "update-readme",
        help="Render docs/README.template.md → README.md (pre-commit: --check)")
    sur.add_argument(
        "--check", action="store_true",
        help="compare rendered output to README.md without writing; exit 1 if stale")
    sur.set_defaults(func=cmd_update_readme)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
