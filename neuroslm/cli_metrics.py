# -*- coding: utf-8 -*-
"""Log analysis + metrics ledger for the BRIAN/NeuroSLM project.

Two artifacts are maintained automatically:

  * docs/metrics.md   — one-row-per-run comparison table
                        (loss / lm / ppl / Phi / OOD PPL / OOD ratio / ...)
  * docs/FINDINGS.md  — narrative observations extracted from each log

`brian analyze-log <logfile>` calls both: parses the file with regexes
(works on training logs and OOD result JSON), upserts a row in
metrics.md keyed on the run id, then — if the `claude` CLI is on PATH —
invokes Claude Code with the log + last metrics row to extract any
surprising/insightful observation and appends it to FINDINGS.md.

The metrics.md row is the cheap-and-deterministic part. The
Claude-derived insights are the qualitative layer on top.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METRICS_PATH = Path("docs/metrics.md")
FINDINGS_PATH = Path("docs/FINDINGS.md")
OOD_DIR = Path("logs/vast/benchmarks/ood")


@dataclass
class RunMetrics:
    """One row of the metrics ledger. Any field can be None ('?' in MD)."""
    run_id: str                                # vast instance id OR ood role tag
    date: str                                  # YYYY-MM-DD
    branch: str = ""
    arch: str = ""                             # e.g. "dsl rcc_bowtie_30m_p4"
    steps: Optional[int] = None
    final_loss: Optional[float] = None
    final_lm: Optional[float] = None
    final_ppl: Optional[float] = None
    phi: Optional[float] = None
    ood_ppl: Optional[float] = None
    ood_ratio: Optional[float] = None          # OOD ppl / train ppl (>1 = generalisation gap)
    tok_per_sec: Optional[float] = None
    notes: str = ""

    def md_row(self) -> str:
        def f(v, fmt="{:.2f}"):
            return fmt.format(v) if v is not None else "?"
        return (f"| {self.run_id} | {self.date} | {self.branch} | "
                f"{self.arch} | {self.steps or '?'} | "
                f"{f(self.final_loss)} | {f(self.final_lm)} | "
                f"{f(self.final_ppl)} | {f(self.phi, '{:.3f}')} | "
                f"{f(self.ood_ppl, '{:.1f}')} | {f(self.ood_ratio, '{:.2f}')} | "
                f"{f(self.tok_per_sec, '{:.0f}')} | {self.notes} |")

    @staticmethod
    def md_header() -> List[str]:
        return [
            "| Run | Date | Branch | Arch | Steps | Loss | LM | PPL | "
            "Phi | OOD-PPL | OOD-ratio | tok/s | Notes |",
            "|-----|------|--------|------|-------|------|----|-----|"
            "-----|---------|-----------|-------|-------|",
        ]


# ── Parsing ────────────────────────────────────────────────────────────


_STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+"
    r"loss\s+(?P<loss>[\d.]+)\s+\|\s+"
    r"lm\s+(?P<lm>[\d.]+)\s+\|\s+"
    r"ppl\s+(?P<ppl>[\d.]+)\s+\|\s+"
    r"gnorm\s+(?P<gnorm>[\d.]+)\s+\|\s+"
    r"lr\s+(?P<lr>[\d.eE+-]+)\s+\|\s+"
    r"(?P<tps>[\d]+)\s+tok/s"
)
_PHI_RE = re.compile(r"(?:Φ|Phi)\s+(?P<phi>[\d.+-]+)")
_PRESET_RE = re.compile(r"preset\s+(\S+)")
_DSL_LM_RE = re.compile(r"DSL-LM.*vocab=(\d+).*d_model=(\d+).*depth=(\d+)")
_CKPT_SAVE_RE = re.compile(r"saved checkpoint.*?(\S+\.pt)")


def parse_training_log(text: str) -> Dict:
    """Extract every step line + the latest scalar values."""
    steps: List[Dict] = []
    for m in _STEP_RE.finditer(text):
        steps.append({k: m.group(k) for k in
                      ("step", "loss", "lm", "ppl", "gnorm", "lr", "tps")})
    preset = None
    pm = _PRESET_RE.search(text)
    if pm:
        preset = pm.group(1).rstrip(":")
    is_dsl = bool(_DSL_LM_RE.search(text))
    final = steps[-1] if steps else {}
    # Mean throughput over last 20 steps as the "stable" tok/s estimate
    tail = steps[-20:] if len(steps) >= 20 else steps
    mean_tps = (sum(int(s["tps"]) for s in tail) / len(tail)
                if tail else None)

    # Latest Phi observed
    phi_match = list(_PHI_RE.finditer(text))
    final_phi = float(phi_match[-1].group("phi")) if phi_match else None

    checkpoints = [m.group(1) for m in _CKPT_SAVE_RE.finditer(text)]

    return {
        "preset": preset,
        "is_dsl": is_dsl,
        "n_steps": len(steps),
        "final": {k: float(v) if k != "step" else int(v)
                  for k, v in final.items()} if final else {},
        "mean_tok_per_sec": mean_tps,
        "phi_final": final_phi,
        "checkpoints_saved": checkpoints,
    }


def parse_ood_json(path: Path) -> Optional[Dict]:
    """Read an OOD result JSON written by brian_ood_test.py.

    Expected keys (approximate, parser tolerates absence):
      ood_ppl, train_ppl, gap_ratio, ckpt, branch, max_ood_windows, ...
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Metrics ledger upsert ──────────────────────────────────────────────


def _load_metrics_md() -> List[str]:
    if METRICS_PATH.is_file():
        return METRICS_PATH.read_text(encoding="utf-8").splitlines()
    return [
        "# BRIAN/NeuroSLM Metrics — Run-by-Run Comparison",
        "",
        "One row per training or OOD-eval run. Auto-updated by",
        "`brian analyze-log <logfile>`. Rows are upserted by run id —",
        "rerunning a log replaces the prior row.",
        "",
        *RunMetrics.md_header(),
    ]


def upsert_metric_row(metric: RunMetrics) -> None:
    """Add or replace metric row keyed by run_id in docs/metrics.md."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = _load_metrics_md()
    new_row = metric.md_row()
    key = f"| {metric.run_id} |"
    replaced = False
    out = []
    for line in lines:
        if line.startswith(key) and not replaced:
            out.append(new_row)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        # Ensure the table header exists (older file may be empty body)
        if not any(l.startswith("| Run ") for l in out):
            out.extend(RunMetrics.md_header())
        out.append(new_row)
    METRICS_PATH.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


# ── Findings narrative (via claude CLI if available) ───────────────────


def claude_available() -> bool:
    return shutil.which("claude") is not None


def claude_extract_insights(log_text: str, summary: Dict) -> Optional[str]:
    """Pipe `log_text` to Claude Code with a structured prompt; return the
    insight markdown (one or two paragraphs) or None on failure."""
    if not claude_available():
        return None
    prompt = (
        "You are reviewing a training or OOD-eval log from the BRIAN "
        "NeuroSLM project. The parsed summary is:\n\n"
        + json.dumps(summary, indent=2)
        + "\n\nWrite a short Markdown insight block (under 200 words) that:\n"
        + "1. Names the run (use the run_id from the summary).\n"
        + "2. Notes any *surprising or non-obvious* observation — loss "
          "spikes, recovery patterns, anomalies, deviations from expected "
          "trajectories, anything worth remembering.\n"
        + "3. Suggests one concrete follow-up if relevant.\n"
        + "If the run was uneventful, say so in one sentence — do not "
          "invent observations."
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            input=log_text[-30000:],   # tail; logs can be 100s of KB
            text=True, capture_output=True, timeout=120)
        if proc.returncode != 0:
            return f"_claude failed (exit {proc.returncode}): {proc.stderr[:200]}_"
        return proc.stdout.strip()
    except Exception as e:
        return f"_claude error: {e}_"


def append_finding(run_id: str, body: str) -> None:
    """Append a timestamped section to docs/FINDINGS.md."""
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (f"\n## Run {run_id} — "
              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
    with FINDINGS_PATH.open("a", encoding="utf-8") as f:
        f.write(header + body.rstrip() + "\n")


# ── Top-level driver used by `brian analyze-log` ────────────────────────


def analyze_log_file(logfile: Path, run_id: Optional[str] = None,
                     branch: Optional[str] = None,
                     use_claude: bool = True) -> RunMetrics:
    """Parse a single log file, upsert metrics row, append finding.

    Returns the populated RunMetrics for the caller to print.
    """
    if not logfile.is_file():
        raise FileNotFoundError(logfile)

    text = logfile.read_text(encoding="utf-8", errors="replace")
    name = logfile.stem
    rid = run_id or name.split("__")[0]
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_ood_log = "/benchmarks/ood/" in str(logfile).replace("\\", "/")

    summary = parse_training_log(text)
    final = summary.get("final") or {}

    metric = RunMetrics(
        run_id=rid,
        date=date,
        branch=branch or "",
        arch=("dsl " + (summary.get("preset") or "?")
              if summary.get("is_dsl")
              else "brain " + (summary.get("preset") or "?")),
        steps=int(final.get("step")) if "step" in final else None,
        final_loss=float(final.get("loss")) if "loss" in final else None,
        final_lm=float(final.get("lm")) if "lm" in final else None,
        final_ppl=float(final.get("ppl")) if "ppl" in final else None,
        phi=summary.get("phi_final"),
        tok_per_sec=summary.get("mean_tok_per_sec"),
        notes=("OOD" if is_ood_log else "train"),
    )

    # If a matching OOD JSON exists, pull ood_ppl / ratio in too
    ood_json = OOD_DIR / f"ood_results_{rid}.json"
    if ood_json.is_file():
        odata = parse_ood_json(ood_json) or {}
        if "ood_ppl" in odata:
            metric.ood_ppl = float(odata["ood_ppl"])
        if "train_ppl" in odata and metric.ood_ppl:
            metric.ood_ratio = metric.ood_ppl / float(odata["train_ppl"])

    upsert_metric_row(metric)

    if use_claude:
        insight = claude_extract_insights(text, {
            "run_id": rid,
            "logfile": str(logfile),
            "summary": summary,
            "metric": metric.__dict__,
        })
        if insight:
            append_finding(rid, insight)

    return metric


def scan_ood_dir() -> List[RunMetrics]:
    """Walk every JSON under logs/vast/benchmarks/ood/ and upsert metrics."""
    out = []
    if not OOD_DIR.is_dir():
        return out
    for p in OOD_DIR.glob("ood_results_*.json"):
        rid = p.stem.replace("ood_results_", "")
        data = parse_ood_json(p) or {}
        metric = RunMetrics(
            run_id=rid,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            arch="ood",
            ood_ppl=float(data.get("ood_ppl")) if "ood_ppl" in data else None,
            ood_ratio=(float(data["ood_ppl"]) / float(data["train_ppl"])
                       if data.get("ood_ppl") and data.get("train_ppl")
                       else None),
            notes=f"OOD eval, ckpt={data.get('checkpoint', '?')}",
        )
        upsert_metric_row(metric)
        out.append(metric)
    return out
