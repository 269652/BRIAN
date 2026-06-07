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
import os
import re
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

      1. ``architectures/rcc_bowtie/`` — a regular folder with arch.neuro
         (the legacy / canonical path).
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
    """Compile arch to a Neural Flow Graph: dict-of-dicts .py + .png render.

    Accepts:
      * a normal architecture folder with ``arch.neuro``,
      * a ``.dna`` snapshot file (auto-routed to its source arch),
      * a folder containing exactly one ``.dna`` (ditto).
    """
    from neuroslm.dsl.nfg import (
        compile_nfg, render_nfg, emit_python,
        RCC_BOWTIE_SPEC, SEMANTIC_SPEC,
    )
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
    arch = _resolve_arch(args.arch)
    ir = compile_folder(Path(arch))
    g = CodeGenerator(ir)
    src = g.generate_module_source()
    if args.out:
        Path(args.out).write_text(src, encoding="utf-8")
        print(f"wrote {args.out}  ({len(src)} chars)")
    else:
        print(f"--- generated nn.Module source ({len(src)} chars) ---")
        print(src[: args.head] if args.head else src)
    return 0


# ── bundle ─────────────────────────────────────────────────────────────

def cmd_dna(args: argparse.Namespace) -> int:
    """Dispatch DNA compile/unfold commands."""
    from neuroslm.compiler.ribosome import RibosomeCompiler

    if args.dna_cmd == "compile":
        arch = _resolve_arch(args.arch)
        output = args.output or str(Path(arch) / "evolution.dna")
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
                arch_name = Path(arch).name or "evolution"
                output_path = output_path / f"{arch_name}.dna"
                output = str(output_path)

            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Compiling {arch} → {output}...")
            compiler = RibosomeCompiler()
            compiler.compile_file(arch, str(output_path))
            print(f"✓ DNA written to {output}")
            return 0
        except Exception as e:
            print(f"✗ Compilation failed: {e}", file=sys.stderr)
            return 1

    elif args.dna_cmd == "unfold":
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

            print(f"Unfolding {dna_path} → {output}...")
            compiler = RibosomeCompiler()
            compiler.unfold_file(dna_path, output)
            print(f"✓ DSL written to {output}")
            return 0
        except Exception as e:
            print(f"✗ Unfold failed: {e}", file=sys.stderr)
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
                 ood_every: int = 0) -> int:
    """Run _deploy_train.py with the appropriate env vars."""
    env = os.environ.copy()
    env["STEPS"] = str(steps)
    if ood_every > 0:
        env["OOD_EVERY"] = str(ood_every)
    if branch:
        env["BRANCH"] = branch
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)
    # Use _deploy_train.py (fast, direct vast.ai API call) instead of
    # vast_train.sh which hangs on Windows due to heredoc pipe issues.
    deploy_script = REPO_ROOT / "_deploy_train.py"
    python = _find_deploy_python()
    return subprocess.call([python, str(deploy_script)], cwd=str(REPO_ROOT), env=env)


def _find_deploy_python() -> str:
    """Find the python with vastai installed (.venv-2 preferred)."""
    venv2 = REPO_ROOT / ".venv-2" / "Scripts" / "python.exe"
    if venv2.is_file():
        return str(venv2)
    # Fallback: current interpreter
    return sys.executable


def cmd_deploy(args: argparse.Namespace) -> int:
    """Launch a DSL training run on vast.ai."""
    ood = args.ood if args.ood else 0
    extra = {}
    if args.scale:
        extra["SCALE"] = args.scale
    if getattr(args, "label", None):
        extra["LABEL_SUFFIX"] = args.label
    return _deploy_dsl(steps=args.steps, branch=args.branch,
                       extra_env=extra, ood_every=ood)


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

def cmd_logs(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return _run([_bash(), "scripts/vast.sh", "logs", str(args.instance_id)],
                env=env)


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


def cmd_train(args: argparse.Namespace) -> int:
    """Train from evol.dna or run minimal training.

    Usage:
        brian train --preset=tiny              # Run minimal CPU training
        brian train --arch=rcc_bowtie --steps=100
        brian train --dna=dna/evol/arch.dna    # Load from DNA with fitness config
    """
    # If tiny preset requested, run minimal training
    if args.preset == "tiny":
        import importlib.util
        original_cwd = os.getcwd()
        try:
            os.chdir(REPO_ROOT)
            # Import and run the minimal training
            spec = importlib.util.spec_from_file_location(
                "colab_train_minimal_cpu",
                REPO_ROOT / "colab_train_minimal_cpu.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Run main without sys.exit
            module.main()
            return 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1
        finally:
            os.chdir(original_cwd)

    # Build training command
    cmd = [sys.executable, "-m", "neuroslm.train_dsl"]

    if args.dna:
        # Train from DNA with embedded fitness config
        # This would require special handling in train_dsl
        print(f"[INFO] Training from DNA: {args.dna}")
        print("[TODO] DNA training not yet integrated into train_dsl")
        print("       For now, use colab_train_minimal_cpu.py or full Colab workflow")
        return 1

    # Standard training
    if args.arch:
        arch = _resolve_arch(args.arch)
        cmd.extend(["--arch", arch])
    else:
        cmd.extend(["--arch", "architectures/rcc_bowtie"])

    if args.preset:
        cmd.extend(["--preset", args.preset])
    else:
        cmd.extend(["--preset", "rcc_bowtie_30m_p4"])

    if args.steps:
        cmd.extend(["--steps", str(args.steps)])

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


# ── arg parser ────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brian",
        description="Unified CLI for the NeuroSLM / BRIAN project.")
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
                    help="(nfg only) use semantic layout inference (data-driven)")
    sc.add_argument("--head", type=int, default=2000,
                    help="when printing to stdout, truncate after N chars")
    sc.set_defaults(func=lambda a: (
        cmd_compile_nfg(argparse.Namespace(
            arch=a.arch, out=a.out, png=a.png, semantic=a.semantic))
        if a.arch_or_subcmd == "nfg"
        else cmd_compile(argparse.Namespace(
            arch=a.arch_or_subcmd, out=a.out, head=a.head))
    ))

    # dna (DNA encoding/decoding)
    sdna = sub.add_parser("dna", help="DNA encoding/decoding for evolutionary architecture")
    sdna_sub = sdna.add_subparsers(dest="dna_cmd", required=True)

    sdna_compile = sdna_sub.add_parser("compile", help="Compile arch.neuro to DNA binary")
    sdna_compile.add_argument("arch", help="architecture name (e.g., rcc_bowtie)")
    sdna_compile.add_argument("--output", "-o", help="DNA output file (default: architectures/<arch>/evolution.dna)")
    sdna_compile.set_defaults(func=cmd_dna)

    sdna_unfold = sdna_sub.add_parser("unfold", help="Unfold DNA binary back to .neuro DSL")
    sdna_unfold.add_argument("dna", help="path to DNA binary file")
    sdna_unfold.add_argument("--output", "-o", help="DSL output file (default: <dna>.neuro)")
    sdna_unfold.set_defaults(func=cmd_dna)

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
                        help="Launch a DSL training run on vast.ai")
    sd.add_argument("--steps", type=int, default=10_000)
    sd.add_argument("--branch", help="git branch to train (default: current)")
    sd.add_argument("--scale", help="Scale variant from arch.neuro scales block "
                    "(e.g. 100m, 300m, 1b). Default: arch's scales.default")
    sd.add_argument("--label", help="Label suffix for the vast.ai instance")
    sd.add_argument("--ood", type=int, nargs="?", const=3000,
                    help="Run mid-training OOD eval every N steps "
                         "(default 3000 if flag passed without value)")
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
    sl = sub.add_parser("logs", help="Tail container logs for a vast instance")
    sl.add_argument("instance_id")
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
    str_train.add_argument("--arch", help="architecture name (default: rcc_bowtie)")
    str_train.add_argument("--dna", help="path to evolved DNA file (e.g., dna/evol/arch.dna)")
    str_train.add_argument("--steps", type=int, help="number of training steps")
    str_train.add_argument("--batch", type=int, help="batch size")
    str_train.add_argument("--seq_len", type=int, help="sequence length")
    str_train.add_argument("--d_sem", type=int, help="semantic dimension")
    str_train.set_defaults(func=cmd_train)

    # test
    st = sub.add_parser("test", help="Run the DSL test suite (or a subset)")
    st.add_argument("pattern", nargs="?",
                    help="optional pytest path/file pattern")
    st.add_argument("-v", "--verbose", action="store_true")
    st.add_argument("--slow", action="store_true",
                    help="include slow tests (by default slow tests are skipped)")
    st.set_defaults(func=cmd_test, slow=False)

    # push
    sp = sub.add_parser("push",
                        help="Push current branch via PAT (no credential helper)")
    sp.set_defaults(func=cmd_push)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
