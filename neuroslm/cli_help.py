# -*- coding: utf-8 -*-
"""Help, tease, and cite commands for the Brian CLI.

Commands exposed via cli.py
---------------------------
  brian help [topic]           List all topics or show a specific doc/topic.
  brian help platforms         Describe every deployment platform.
  brian tease runs [--n N]     Show the N most-recent run ledger entries.
  brian tease log  [--tail N] [--best] [PATH]
                               Tail N lines of a log (best run by default).
  brian tease findings [--n N] Tease N most-recent findings entries.
  brian cite <run-id>          Full citation block for one run.
  brian cite --list            List all citable run IDs.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# -- Constants -----------------------------------------------------------------

RUNS_LEDGER_PATH = REPO_ROOT / "docs" / "runs.md"
FINDINGS_PATH = REPO_ROOT / "docs" / "findings.md"
BEST_RUN_LN = REPO_ROOT / ".brian" / "best_run.ln"

# Topic name → relative doc path
DOC_MAP: dict[str, str] = {
    "brian":        "docs/brian.md",
    "cli":          "docs/cli.md",
    "dsl":          "docs/dsl.md",
    "architecture": "docs/architecture.md",
    "arch":         "docs/architecture.md",
    "ste":          "docs/ste.md",
    "runs":         "docs/runs.md",
    "findings":     "docs/findings.md",
    "harness":      "docs/harness.md",
    "formal":       "docs/formal_framework.md",
    "technical":    "docs/technical_report.md",
    "thsd":         "docs/THSD_IMPLEMENTATION_SUMMARY.md",
    "ood":          "docs/OOD_MECHANISMS.md",
    "history":      "docs/history.md",
    "changelog":    "docs/changelog.md",
}

# Short blurbs shown by `brian help` (topic index)
DOC_BLURBS: dict[str, str] = {
    "platforms":    "Deployment platforms — vast.ai, lightning.ai, capabilities & config",
    "brian":        "Architecture overview, 11-stage bowtie, design rationale",
    "cli":          "Complete CLI reference — every command, flag, and example",
    "dsl":          "The .neuro DSL — syntax, grammar, directives, and compilation",
    "arch":         "Full architecture technical reference (alias: architecture)",
    "ste":          "Semantic Turbulence Engine — RG cascade, GPE layer, criticality monitor",
    "runs":         "Run ledger — every meaningful training run with metrics and significance",
    "findings":     "Hypothesis ledger — Layer A/B evidence, confirmed/falsified status",
    "harness":      "Training harness — loss composition, inhibition, fusion gate",
    "formal":       "THSD formal framework — mathematical foundations, axioms",
    "technical":    "Technical report (external-AI-facing) — full system description",
    "thsd":         "THSD implementation summary — Simplex, Sheaf, Coboundary, Phi",
    "ood":          "OOD mechanisms — evaluation protocol, gap-ratio scoring, caveat list",
    "history":      "Session history — decisions and notes",
    "changelog":    "Changelog — commit-derived version history",
}

# -- Platform registry ---------------------------------------------------------

PLATFORM_DOCS: dict[str, dict] = {
    "vast": {
        "name": "vast.ai",
        "config_key": '"vast"',
        "description": (
            "GPU spot-market with per-second billing. "
            "Instances are ephemeral: the onstart script clones the repo, "
            "runs training, and the instance self-destructs on completion "
            "(or via `brian destroy`). No persistent Studio -- every run "
            "starts from a fresh image. Logs are pushed to GitHub via "
            "log_pusher.sh and also live in logs/ after a run ends."
        ),
        "machines": "Any GPU offered on vast.ai (H100, A100, RTX 4090, ...).",
        "auth": "VAST_API_KEY in .env",
        "strengths": [
            "Cheapest $/GPU-hr for spot capacity",
            "Any GPU type, including consumer cards",
            "Zero idle cost — instances die when training ends",
            "Simple SSH-less workflow via vastai CLI",
        ],
        "weaknesses": [
            "Instances can be pre-empted (rare but possible)",
            "No persistent storage — must push checkpoints to HF Hub",
            "Cold-start per run (apt-get, git clone, pip install)",
        ],
        "cli_flag": "--platform vast",
        "toml_key": '[deploy]\nplatform = "vast"',
        "brian_ops": [
            "brian deploy [arch] [--steps N]",
            "brian ps                         # list running instances",
            "brian logs <instance-id>         # tail container stdout",
            "brian destroy <instance-id>      # kill the instance",
        ],
    },
    "lightning": {
        "name": "Lightning AI",
        "config_key": '"lightning"',
        "description": (
            "Studio-based GPU cloud. A Studio is a persistent VM that "
            "survives between runs -- the first launch clones + installs "
            "the repo; subsequent runs on the same Studio reuse the "
            "environment. Training is fire-and-forget via `Studio.run_and_detach`; "
            "logs live on-Studio at ~/brian/logs/run-<job_id>.log. "
            "Job state persists in .brian/jobs/<job_id>.json so "
            "`brian ps` can reconnect."
        ),
        "machines": (
            "T4, A10G, A100 (40 GB / 80 GB), L4, H100. "
            "Select via --machine or brian.toml [deploy].machine."
        ),
        "auth": "LIGHTNING_API_KEY + LIGHTNING_USER_ID in .env  (or `lightning login`)",
        "strengths": [
            "Persistent Studio: no cold-start after first launch",
            "Managed environment with pre-installed CUDA",
            "Teamspace support for multi-user orgs",
            "SSH access via `s_<id>@ssh.lightning.ai`",
        ],
        "weaknesses": [
            "Idle Studio costs money if not stopped",
            "Slower first-run cold-start (full pip install)",
            "Log streaming requires SSH; `brian logs` shows cached tail",
        ],
        "cli_flag": "--platform lightning",
        "toml_key": '[deploy]\nplatform = "lightning"',
        "brian_ops": [
            "brian deploy [arch] --platform lightning [--machine A10G]",
            "brian ps                         # list running jobs",
            "brian stop <job-id>              # halt a Lightning job",
        ],
    },
}

# -- RunEntry dataclass --------------------------------------------------------

@dataclass
class RunEntry:
    """One entry from docs/runs.md."""
    run_id: str
    title: str
    date: str
    log_path: str
    checkpoint: str
    significance: str
    gap_ratio: Optional[float] = None
    ood_ppl: Optional[float] = None
    train_ppl: Optional[float] = None
    steps: Optional[int] = None


# -- Parsing -------------------------------------------------------------------

_RUN_HEADER_RE = re.compile(r"^## Run: (\S+) · (.+)$", re.MULTILINE)
_METRIC_KV_RE = re.compile(r"(\w+)=([\d.]+)")


def _parse_metrics_line(line: str) -> dict:
    """Extract key=value pairs from a Metrics line."""
    result: dict = {}
    for m in _METRIC_KV_RE.finditer(line):
        key, val = m.group(1), m.group(2)
        try:
            result[key] = float(val) if "." in val else int(val)
        except ValueError:
            pass
    return result


def _extract_field(line: str, field_name: str) -> str:
    """Extract value after '**Field:** ...' stripping backticks."""
    prefix = f"**{field_name}:**"
    if not line.startswith(prefix):
        return ""
    raw = line[len(prefix):].strip()
    # Remove markdown link syntax [text](url) → take the text part (inner path)
    link_m = re.search(r"\[([^\]]+)\]\([^)]+\)", raw)
    if link_m:
        return link_m.group(1).strip().strip("`")
    return raw.strip("`").strip()


def parse_runs_ledger(text: str) -> list[RunEntry]:
    """Parse docs/runs.md text; return RunEntry list in document order."""
    parts = _RUN_HEADER_RE.split(text)
    # After split: [preamble, run_id, title, body, run_id, title, body, ...]
    if len(parts) < 4:
        return []

    entries: list[RunEntry] = []
    idx = 1
    while idx + 2 <= len(parts):
        run_id = parts[idx].strip()
        title = parts[idx + 1].strip()
        body = parts[idx + 2]
        idx += 3

        date = log_path = checkpoint = ""
        gap_ratio = ood_ppl = train_ppl = None
        steps: Optional[int] = None
        sig_lines: list[str] = []
        in_sig = False

        for line in body.splitlines():
            s = line.strip()
            if s.startswith("**Date:**"):
                date = _extract_field(s, "Date")
            elif s.startswith("**Log:**"):
                log_path = _extract_field(s, "Log")
            elif s.startswith("**Checkpoint:**"):
                checkpoint = _extract_field(s, "Checkpoint")
            elif s.startswith("**Metrics:**"):
                metrics_line = s[len("**Metrics:**"):].strip()
                parsed = _parse_metrics_line(metrics_line)
                train_ppl = parsed.get("train_ppl")
                ood_ppl = parsed.get("ood_ppl")
                gap_ratio = parsed.get("gap_ratio")
                raw_steps = parsed.get("steps")
                steps = int(raw_steps) if raw_steps is not None else None
            elif s.startswith("**") and ":**" in s:
                pass  # other field — skip
            elif s in ("---", ""):
                if in_sig:
                    sig_lines.append("")
            else:
                in_sig = True
                sig_lines.append(line)

        significance = "\n".join(sig_lines).strip()

        entries.append(RunEntry(
            run_id=run_id,
            title=title,
            date=date,
            log_path=log_path,
            checkpoint=checkpoint,
            significance=significance,
            gap_ratio=float(gap_ratio) if gap_ratio is not None else None,
            ood_ppl=float(ood_ppl) if ood_ppl is not None else None,
            train_ppl=float(train_ppl) if train_ppl is not None else None,
            steps=steps,
        ))

    return entries


# -- Tease utilities -----------------------------------------------------------

def tease_runs(entries: list[RunEntry], n: int = 5) -> list[RunEntry]:
    """Return last n entries (most recently added = closest to end of ledger)."""
    return entries[-n:] if entries else []


def get_best_log_path(repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """Read .brian/best_run.ln and return the resolved log path, or None."""
    ln = repo_root / ".brian" / "best_run.ln"
    if not ln.exists():
        return None
    for line in ln.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            resolved = repo_root / line
            return resolved if resolved.exists() else None
    return None


def tease_log_tail(log_path: Path, n: int = 10) -> str:
    """Return last n non-empty lines from log_path, or '' if not found."""
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    non_empty = [l for l in lines if l.strip()]
    return "\n".join(non_empty[-n:])


# -- Formatting ----------------------------------------------------------------

def _steps_str(steps: Optional[int]) -> str:
    if steps is None:
        return "—"
    if steps >= 1000:
        return f"{steps // 1000}k"
    return str(steps)


def _metric_str(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def format_runs_terminal(entries: list[RunEntry]) -> str:
    """Format run entries as a terminal-friendly table."""
    if not entries:
        return "(no runs in ledger)"

    header = f"{'Run ID':<19} {'Date':<12} {'Title':<32} {'gap':>5} {'ood_ppl':>8} {'steps':>6}"
    sep = "-" * len(header)
    rows = []
    for e in reversed(entries):  # newest first
        title_trunc = (e.title[:30] + "...") if len(e.title) > 31 else e.title
        rows.append(
            f"{e.run_id:<19} {e.date:<12} {title_trunc:<32} "
            f"{_metric_str(e.gap_ratio, 2):>5} "
            f"{_metric_str(e.ood_ppl, 1):>8} "
            f"{_steps_str(e.steps):>6}"
        )
    return "\n".join([sep, header, sep] + rows + [sep])


def format_runs_table_md(entries: list[RunEntry]) -> str:
    """Format run entries as a GitHub Markdown table (most recent first)."""
    if not entries:
        return "_No runs recorded yet._"

    lines = [
        "| Run ID | Date | Title | gap_ratio | ood_ppl | steps |",
        "|--------|------|-------|-----------|---------|-------|",
    ]
    for e in reversed(entries):  # newest first in table
        rid = f"`{e.run_id}`"
        gap = _metric_str(e.gap_ratio, 2)
        ood = _metric_str(e.ood_ppl, 1)
        st = _steps_str(e.steps)
        lines.append(f"| {rid} | {e.date} | {e.title} | {gap} | {ood} | {st} |")
    return "\n".join(lines)


def cite_entry(entry: RunEntry, fmt: str = "md") -> str:
    """Return a formatted citation block for a single RunEntry."""
    parts = [
        f"**Run:** `{entry.run_id}` · {entry.title}",
        f"**Date:** {entry.date}",
    ]
    if entry.log_path:
        parts.append(f"**Log:** `{entry.log_path}`")
    if entry.checkpoint:
        parts.append(f"**Checkpoint:** `{entry.checkpoint}`")
    metrics = []
    if entry.train_ppl is not None:
        metrics.append(f"train_ppl={entry.train_ppl:.1f}")
    if entry.ood_ppl is not None:
        metrics.append(f"ood_ppl={entry.ood_ppl:.1f}")
    if entry.gap_ratio is not None:
        metrics.append(f"gap_ratio={entry.gap_ratio:.2f}")
    if entry.steps is not None:
        metrics.append(f"steps={entry.steps:,}")
    if metrics:
        parts.append("**Metrics:** " + " · ".join(metrics))
    if entry.significance:
        parts.append("")
        parts.append(entry.significance)
    return "\n".join(parts)


# -- Findings teaser -----------------------------------------------------------

_FINDINGS_H_RE = re.compile(r"^### (H\d+[\.\d]* .+)$", re.MULTILINE)
_FINDINGS_STATUS_RE = re.compile(r"(✅|🟡|🟠|❌|⚠)\s*\*\*(\w+)\*\*")


def tease_findings(text: str, n: int = 3) -> str:
    """Extract the last n hypothesis entries from findings.md as a summary."""
    matches = list(_FINDINGS_H_RE.finditer(text))
    if not matches:
        return "(no findings entries found)"

    recent = matches[-n:]
    lines = []
    for m in recent:
        h_title = m.group(1).strip()
        # Grab a small snippet after the header to find the status
        snippet_start = m.end()
        snippet_end = min(snippet_start + 400, len(text))
        snippet = text[snippet_start:snippet_end]
        status_m = _FINDINGS_STATUS_RE.search(snippet)
        status = f"{status_m.group(1)} {status_m.group(2)}" if status_m else "—"
        lines.append(f"- **{h_title}** — {status}")

    return "\n".join(lines)


# -- Command handlers ----------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text, replacing characters unsupported by the current encoding."""
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc)
    print(safe)


def _pager_print(text: str) -> None:
    """Print text, using a pager when stdout is a TTY and text is long."""
    if sys.stdout.isatty() and text.count("\n") > 40:
        try:
            import pydoc
            pydoc.pager(text)
            return
        except Exception:
            pass
    _safe_print(text)


def cmd_help(args: argparse.Namespace) -> int:
    """brian help [topic]"""
    topic: Optional[str] = getattr(args, "topic", None)

    if topic is None:
        # Print index of all topics
        lines = [
            "brian — unified CLI for the NeuroSLM / BRIAN project",
            "",
            "Usage:  brian help <topic>",
            "",
            "Topics:",
        ]
        # Always-available special topics
        lines.append(f"  {'platforms':<16} {DOC_BLURBS['platforms']}")
        lines.append("")
        for name, blurb in DOC_BLURBS.items():
            if name in ("platforms", "arch"):
                continue  # arch is an alias, platforms is special
            lines.append(f"  {name:<16} {blurb}")
        lines.append("")
        lines.append("Run `brian <command> --help` for per-command argument help.")
        _safe_print("\n".join(lines))
        return 0

    if topic == "platforms":
        return _show_platforms()

    # Doc alias
    if topic in DOC_MAP:
        doc_rel = DOC_MAP[topic]
        doc_path = REPO_ROOT / doc_rel
        if not doc_path.exists():
            print(f"[brian help] doc not found: {doc_rel}", file=sys.stderr)
            print("(File may not yet exist — try `brian help` for available topics.)",
                  file=sys.stderr)
            return 1
        _pager_print(doc_path.read_text(encoding="utf-8"))
        return 0

    print(f"[brian help] unknown topic: {topic!r}", file=sys.stderr)
    print("Run `brian help` for a list of topics.", file=sys.stderr)
    return 1


def _show_platforms() -> int:
    """Detailed platform comparison for `brian help platforms`."""
    lines = [
        "Deployment Platforms",
        "====================",
        "",
        "Brian supports two cloud platforms for GPU training.",
        "Switch with --platform <name> or set [deploy].platform in brian.toml.",
        "",
    ]
    for key, info in PLATFORM_DOCS.items():
        lines += [
            f"{'-' * 60}",
            f"  {info['name']}   (config key: {info['config_key']})",
            f"{'-' * 60}",
            "",
            f"  {info['description']}",
            "",
            f"  Machines:  {info['machines']}",
            f"  Auth:      {info['auth']}",
            "",
            "  Strengths:",
        ]
        for s in info["strengths"]:
            lines.append(f"    + {s}")
        lines += ["", "  Trade-offs:"]
        for w in info["weaknesses"]:
            lines.append(f"    - {w}")
        lines += ["", "  Brian CLI ops:"]
        for op in info["brian_ops"]:
            lines.append(f"    {op}")
        toml_lines = info["toml_key"].replace("\n", "\n    ")
        lines += ["", "  brian.toml:", f"    {toml_lines}", ""]

    lines += [
        "-" * 60,
        "",
        "Quick-switch examples:",
        "  brian deploy --platform vast                  # spot GPU, cheap",
        "  brian deploy --platform lightning --machine A10G  # persistent Studio",
        "",
        "Docs: brian help cli  |  brian help brian",
    ]
    _pager_print("\n".join(lines))
    return 0


def cmd_tease(args: argparse.Namespace) -> int:
    """brian tease <what> [options]"""
    what: str = getattr(args, "what", "")

    if what == "runs":
        n = getattr(args, "n", 5)
        if not RUNS_LEDGER_PATH.exists():
            print("[brian tease runs] docs/runs.md not found — "
                  "no run ledger yet.", file=sys.stderr)
            return 1
        entries = parse_runs_ledger(RUNS_LEDGER_PATH.read_text(encoding="utf-8"))
        recent = tease_runs(entries, n=n)
        fmt = getattr(args, "format", "terminal")
        if fmt == "md":
            _safe_print(format_runs_table_md(recent))
        else:
            label = f"Run Ledger - {len(recent)} most recent"
            _safe_print(label)
            _safe_print(format_runs_terminal(recent))
        return 0

    if what == "log":
        tail_n = getattr(args, "tail", 10)
        best = getattr(args, "best", False)
        path_arg = getattr(args, "path", None)

        if path_arg:
            log_path = Path(path_arg)
            if not log_path.is_absolute():
                log_path = REPO_ROOT / log_path
        elif best or path_arg is None:
            log_path = get_best_log_path()
            if log_path is None:
                print("[brian tease log] no best-run pointer found. "
                      "Run `brian best update` first, or pass a log path.",
                      file=sys.stderr)
                return 1
        else:
            log_path = get_best_log_path()
            if log_path is None:
                print("[brian tease log] pass a log path or --best",
                      file=sys.stderr)
                return 1

        tail = tease_log_tail(log_path, n=tail_n)
        if not tail:
            print(f"[brian tease log] log empty or not found: {log_path}",
                  file=sys.stderr)
            return 1
        try:
            display = log_path.relative_to(REPO_ROOT)
        except ValueError:
            display = log_path
        label = f">> {display}  (last {tail_n} lines)"
        _safe_print(label)
        _safe_print("-" * min(len(str(label)), 80))
        _safe_print(tail)
        return 0

    if what == "findings":
        n = getattr(args, "n", 3)
        if not FINDINGS_PATH.exists():
            print("[brian tease findings] docs/findings.md not found.",
                  file=sys.stderr)
            return 1
        text = FINDINGS_PATH.read_text(encoding="utf-8")
        _safe_print(f"findings.md - {n} most recent hypotheses")
        _safe_print(tease_findings(text, n=n))
        return 0

    print(f"[brian tease] unknown sub-topic: {what!r}", file=sys.stderr)
    print("Available: runs, log, findings", file=sys.stderr)
    return 1


def cmd_cite(args: argparse.Namespace) -> int:
    """brian cite [--list | <run-id>] [--format md|text]"""
    if not RUNS_LEDGER_PATH.exists():
        print("[brian cite] docs/runs.md not found — no run ledger yet.",
              file=sys.stderr)
        return 1

    entries = parse_runs_ledger(RUNS_LEDGER_PATH.read_text(encoding="utf-8"))
    by_id = {e.run_id: e for e in entries}

    if getattr(args, "list", False):
        if not entries:
            _safe_print("(ledger is empty)")
            return 0
        _safe_print("Citable run IDs  (docs/runs.md)")
        _safe_print("-" * 40)
        for e in reversed(entries):
            metrics = []
            if e.gap_ratio is not None:
                metrics.append(f"gap={e.gap_ratio:.2f}")
            if e.ood_ppl is not None:
                metrics.append(f"ood={e.ood_ppl:.1f}")
            suffix = "  " + " ".join(metrics) if metrics else ""
            _safe_print(f"  {e.run_id}  {e.date}  {e.title[:40]}{suffix}")
        return 0

    run_id = getattr(args, "run_id", None)
    if not run_id:
        print("[brian cite] pass a run-id or --list", file=sys.stderr)
        return 1

    entry = by_id.get(run_id)
    if entry is None:
        print(f"[brian cite] run {run_id!r} not found in ledger.", file=sys.stderr)
        close = [k for k in by_id if run_id in k]
        if close:
            print(f"  Did you mean: {', '.join(close)}", file=sys.stderr)
        return 1

    fmt = getattr(args, "format", "md")
    _safe_print(cite_entry(entry, fmt=fmt))
    return 0
