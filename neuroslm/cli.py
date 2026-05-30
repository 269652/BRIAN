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
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ── Locate the repo root so commands work from any cwd ────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent


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

def cmd_compile(args: argparse.Namespace) -> int:
    """Compile an architecture .neuro folder to a runnable nn.Module class."""
    from neuroslm.dsl.codegen import CodeGenerator
    from neuroslm.dsl.multifile import compile_folder
    ir = compile_folder(Path(args.arch))
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
    if args.full:
        code = architecture_to_wolfram_full(args.arch)
    else:
        code = architecture_to_wolfram(args.arch)
    if args.out:
        save_wolfram(args.arch, args.out, full=args.full)
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
    cli.append(args.arch)
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
    return A.main(cli)


# ── deploy / deploy-100k / deploy-brain ───────────────────────────────

def _deploy_dsl(steps: int, branch: Optional[str], extra_env: dict) -> int:
    """Run scripts/vast_train.sh with USE_DSL=1 + STEPS=N + (BRANCH)."""
    env = os.environ.copy()
    env["USE_DSL"] = "1"
    env["FRESH"] = "1"
    env["STEPS"] = str(steps)
    if branch:
        env["BRANCH"] = branch
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)
    return _run([_bash(), "scripts/vast_train.sh"], env=env)


def cmd_deploy(args: argparse.Namespace) -> int:
    """Launch a DSL training run on vast.ai."""
    return _deploy_dsl(steps=args.steps, branch=args.branch, extra_env={})


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

    # compile
    sc = sub.add_parser("compile",
                        help="Compile an arch.neuro folder to a runnable nn.Module")
    sc.add_argument("arch", help="architectures/<name>/ path")
    sc.add_argument("--out", help="write generated .py to this path")
    sc.add_argument("--head", type=int, default=2000,
                    help="when printing to stdout, truncate after N chars")
    sc.set_defaults(func=cmd_compile)

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
    sa.add_argument("--discover", choices=["phi", "modularity", "sparsity"],
                    help="propose architecture mods maximising the metric")
    sa.add_argument("--top-k", type=int, default=10,
                    help="top-K proposals for --discover")
    sa.add_argument("--all", action="store_true",
                    help="run every analysis above")
    sa.set_defaults(func=cmd_analyze)

    # deploy
    sd = sub.add_parser("deploy",
                        help="Launch a DSL training run on vast.ai")
    sd.add_argument("--steps", type=int, default=10_000)
    sd.add_argument("--branch", help="git branch to train (default: current)")
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
    ss = sub.add_parser("status", help="List active vast instances")
    ss.set_defaults(func=cmd_status)

    # destroy
    sde = sub.add_parser("destroy", help="Tear down vast instance(s)")
    sde.add_argument("instance_id", nargs="?")
    sde.add_argument("--all", action="store_true",
                     help="destroy every neuroslm-* labelled instance")
    sde.set_defaults(func=cmd_destroy)

    # ood
    so = sub.add_parser("ood", help="Run OOD eval on a checkpoint")
    so.add_argument("ckpt", help="ckpt path (e.g. lfs_checkpoints/dsl_arch_step10000.pt)")
    so.add_argument("--branch")
    so.add_argument("--tag", default="eval", help="role tag for the eval JSON")
    so.add_argument("--windows", type=int)
    so.set_defaults(func=cmd_ood)

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
