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

  Training
    deploy [--steps N]        Launch DSL training run on vast (default 10k)
    deploy-100k               Long DSL run (100k steps)
    deploy-brain [...]        Launch a Brain (non-DSL) training run
    logs <id>                 Tail container logs
    status                    List active vast instances
    destroy <id> | --all      Tear down instance(s)

  Evaluation
    ood <ckpt> [--branch B]   Spin a throwaway OOD-eval instance

  Project
    test [pattern]            pytest tests/dsl (optional path filter)
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

def cmd_compile_nfg(args: argparse.Namespace) -> int:
    """Compile arch to a Neural Flow Graph: dict-of-dicts .py + .png render."""
    from neuroslm.dsl.nfg import compile_nfg, render_nfg, emit_python
    arch = _resolve_arch(args.arch)
    g = compile_nfg(arch)
    out_py = args.out or os.path.join(arch, "nfg.py")
    out_png = args.png or os.path.join(arch, "nfg.png")
    emit_python(g, out_py)
    print(f"wrote NFG definition  -> {out_py}")
    try:
        render_nfg(g, out_png)
        print(f"wrote NFG render      -> {out_png}")
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
    """Run scripts/vast_train.sh with USE_DSL=1 + STEPS=N + (BRANCH)."""
    env = os.environ.copy()
    env["USE_DSL"] = "1"
    env["FRESH"] = "1"
    env["STEPS"] = str(steps)
    if ood_every > 0:
        env["OOD_EVERY"] = str(ood_every)
    if branch:
        env["BRANCH"] = branch
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)
    return _run([_bash(), "scripts/vast_train.sh"], env=env)


def cmd_deploy(args: argparse.Namespace) -> int:
    """Launch a DSL training run on vast.ai."""
    ood = args.ood if args.ood else 0
    return _deploy_dsl(steps=args.steps, branch=args.branch,
                       extra_env={}, ood_every=ood)


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
    """
    import json
    vastai = _vastai_exe()
    raw, rc = _run_capture([vastai, "show", "instances", "--raw"])
    if rc != 0 and "DEPRECATED" not in raw:
        print(f"vastai show failed: {raw[:300]}", file=sys.stderr)
        return rc
    # Strip DEPRECATED banner + any trailing post-JSON content. Walk
    # the bracket depth from the first `[` to its matching close.
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
            print(f"json parse error: {e}", file=sys.stderr)
            return 1
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
    if not rows:
        print("(no neuroslm instances — pass --all to list everything)")
        return 0
    # Render a table
    hdr = (f"{'ID':>10}  {'LABEL':<28}  {'GPU':<12}  {'$/hr':>5}  "
           f"{'UP(m)':>6}  {'PHASE':<16}  {'STEP':>6}  {'PPL':>8}  "
           f"{'OOD-PPL':>9}  {'TOK/S':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        step  = str(r["step"]) if r["step"] is not None else "-"
        ppl   = f"{r['ppl']:.1f}" if r["ppl"] is not None else "-"
        ood   = (f"{r['mid_ood_ppl']:.0f}@{r['mid_ood_step']}"
                 if r["mid_ood_ppl"] is not None else "-")
        tps   = f"{r['tps']/1000:.0f}k" if r["tps"] else "-"
        cost  = f"{r['cost']:.2f}" if r["cost"] else "-"
        print(f"{str(r['id']):>10}  {r['label'][:28]:<28}  "
              f"{r['gpu'][:12]:<12}  {cost:>5}  {r['uptime_mins']:>6}  "
              f"{r['phase']:<16}  {step:>6}  {ppl:>8}  {ood:>9}  {tps:>7}")
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
    """Return [(step, path), ...] sorted desc by step. Local first, then origin."""
    ckpt_dir = REPO_ROOT / "lfs_checkpoints"
    items: List[Tuple[int, str]] = []
    if ckpt_dir.is_dir():
        for p in ckpt_dir.glob("dsl_arch_step*.pt"):
            m = re.search(r"step(\d+)", p.name)
            if m:
                items.append((int(m.group(1)), str(p.relative_to(REPO_ROOT))))
    # Also list what's on origin (via git ls-tree on the lfs_checkpoints/ dir)
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), text=True).strip()
        out = subprocess.check_output(
            ["git", "ls-tree", f"origin/{branch}",
             "lfs_checkpoints/", "--name-only"],
            cwd=str(REPO_ROOT), text=True)
        for line in out.splitlines():
            if "dsl_arch_step" not in line:
                continue
            m = re.search(r"step(\d+)", line)
            if not m:
                continue
            entry = (int(m.group(1)), line)
            if entry not in items:
                items.append(entry)
    except Exception:
        pass
    items.sort(key=lambda r: -r[0])
    return items


def _eval_ood(args: argparse.Namespace) -> int:
    """Interactive checkpoint picker → deploy vast_ood_eval.sh."""
    ckpts = _find_dsl_checkpoints()
    if not ckpts:
        print("no DSL checkpoints found in lfs_checkpoints/ or on origin")
        return 1
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Default to the highest-step checkpoint (latest training run "best").
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
        # Restrict to verified A100 by default (override with VAST_GPU_QUERY)
        "VAST_GPU_QUERY": env.get(
            "VAST_GPU_QUERY",
            "gpu_name in [A100_SXM4,A100_PCIE,A100_SXM,A100X] num_gpus=1 "
            "rentable=true verified=true reliability>0.99"),
    })
    if args.windows:
        env["MAX_OOD_WINDOWS"] = str(args.windows)
    return _run([_bash(), "scripts/vast_ood_eval.sh"], env=env)


# ── test / push ────────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    path = args.pattern if args.pattern else "tests/dsl/"
    cli = [sys.executable, "-m", "pytest", path, "-q"]
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
    sc.add_argument("--head", type=int, default=2000,
                    help="when printing to stdout, truncate after N chars")
    sc.set_defaults(func=lambda a: (
        cmd_compile_nfg(argparse.Namespace(
            arch=a.arch, out=a.out, png=a.png))
        if a.arch_or_subcmd == "nfg"
        else cmd_compile(argparse.Namespace(
            arch=a.arch_or_subcmd, out=a.out, head=a.head))
    ))

    # wolfram
    sw = sub.add_parser("wolfram",
                        help="Emit Mathematica/Wolfram code for an arch")
    sw.add_argument("arch")
    sw.add_argument("--full", action="store_true",
                    help="IIT-grade: populations + synapses + modulations + NT dynamics")
    sw.add_argument("--out", help="write Wolfram code to this .m file")
    sw.set_defaults(func=cmd_wolfram)

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

    # eval (group: `brian eval ood` interactive picker)
    se = sub.add_parser("eval",
                        help="Evaluate a checkpoint (ood/...) — interactive picker")
    ese = se.add_subparsers(dest="eval_kind", required=True)
    ese_ood = ese.add_parser("ood",
                             help="OOD eval on a DSL checkpoint (interactive)")
    ese_ood.add_argument("--checkpoint",
                         help="explicit ckpt path (skips picker)")
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

    # test
    st = sub.add_parser("test", help="Run the DSL test suite (or a subset)")
    st.add_argument("pattern", nargs="?",
                    help="optional pytest path/file pattern")
    st.add_argument("-v", "--verbose", action="store_true")
    st.set_defaults(func=cmd_test)

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
