# -*- coding: utf-8 -*-
"""DSL-driven training entrypoint.

Parallel to `neuroslm.train` (which trains the hand-written `Brain`):
this module loads an architecture from `architectures/<name>/`,
compiles it via the DSL pipeline, wraps it in a `BRIANHarness`, and
runs a language-model training loop.

The harness reads `training { ... }` from the architecture's
`arch.neuro` for loss clipping, optimizer choice, label smoothing,
grad accumulation, and grad clipping. Per-step model behavior is
otherwise determined entirely by the .neuro files — no Python
architecture code path involved.

Usage:
    python -m neuroslm.train_dsl --arch architectures/rcc_bowtie \\
        --steps 10000 --batch 4 --seq_len 256 --d_sem 256
"""
from __future__ import annotations
import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# Per-run id stamped into checkpoint filenames so concurrent / successive
# runs to the same ckpt_dir never overwrite each other. Settable via the
# DSL_RUN_ID env var (vast_train_dsl_loop sets it when resuming); falls
# back to current UTC time at import.
_RUN_ID = os.environ.get(
    "DSL_RUN_ID", datetime.utcnow().strftime("%Y%m%d-%H%M%S"))

import torch

from neuroslm.dsl.codegen import CodeGenerator
from neuroslm.dsl.multifile import compile_folder
from neuroslm.dsl.training_config import load_training_config_from_arch
from neuroslm.dsl.param_scopes import load_param_scopes_from_arch
from neuroslm.harness import BRIANHarness


# ── Tokenizer ────────────────────────────────────────────────────────

def _load_tokenizer():
    """Reuse the project's tokenizer if available; fall back to a
    synthetic vocab for smoke tests when the tokenizer module is missing
    or its data files aren't present.
    """
    try:
        from neuroslm.tokenizer import Tokenizer
        return Tokenizer()
    except Exception as e:
        print(f"[train_dsl] tokenizer unavailable ({e!r}); using synthetic vocab")
        class _SynthTok:
            vocab_size = 1024
            def encode(self, _: str): return [0]
            def decode(self, _): return ""
        return _SynthTok()


# ── Data — synthetic random batches for Phase A smoke ───────────────

class SyntheticBatchSource:
    """Generates random (ids, targets) batches — fallback when the real
    data pipeline is unavailable (no network / no `datasets`)."""
    def __init__(self, vocab_size: int, batch: int, seq_len: int,
                 device: str = "cpu", seed: int = 0):
        self.vocab_size = vocab_size
        self.batch = batch
        self.seq_len = seq_len
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def next(self):
        ids = torch.randint(0, self.vocab_size, (self.batch, self.seq_len),
                            device=self.device, generator=self.gen)
        targets = torch.randint(0, self.vocab_size,
                                (self.batch, self.seq_len),
                                device=self.device, generator=self.gen)
        return ids, targets


class RealDataSource:
    """Streams real tokenized batches via neuroslm.data.batch_iterator.

    Yields next-token-prediction (ids, targets) from a `(B, ctx_len+1)`
    window: ids = window[:, :-1], targets = window[:, 1:]. Falls back to
    SyntheticBatchSource if the data pipeline can't initialise (e.g. no
    network in a local run).
    """
    def __init__(self, tokenizer, batch: int, seq_len: int,
                 device: str = "cpu", mode: str = "mix",
                 chat_ratio: float = 0.6, seed: int = 0,
                 ratio_ref=None, with_labels: bool = False):
        self.device = device
        self._fallback = None
        self._with_labels = with_labels
        self._last_labels = None  # set by next() when with_labels=True
        try:
            from neuroslm.data import batch_iterator
            self._it = batch_iterator(
                tokenizer, ctx_len=seq_len, batch_size=batch,
                seed=seed, mode=mode, chat_ratio=chat_ratio,
                ratio_ref=ratio_ref, with_labels=with_labels,
            )
            # Prime one batch to surface failures early
            self._primed = next(self._it)
        except Exception as e:
            print(f"[train_dsl] real data unavailable ({e!r}); using synthetic")
            self._fallback = SyntheticBatchSource(
                tokenizer.vocab_size, batch, seq_len, device, seed)
            self._primed = None

    def next(self):
        if self._fallback is not None:
            return self._fallback.next()
        if self._primed is not None:
            item, self._primed = self._primed, None
        else:
            item = next(self._it)
        if self._with_labels:
            window, labels = item
            self._last_labels = labels.to(self.device)
        else:
            window = item
        window = window.to(self.device)
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()

    def domain_id_fn(self, _ids):
        """Return per-sample domain labels (0=text, 1=chat) for last batch.

        DARReweighter calls this via harness.set_domain_id_fn(). The
        contract: it's called *during* compute_loss for the same batch
        that was just produced by next(), so we return self._last_labels.
        """
        return self._last_labels


# ── Build the harness ────────────────────────────────────────────────

def build_dsl_lm_harness(arch_root: Path, vocab_size: int, d_model: int,
                         depth: int, n_heads: int, max_ctx: int,
                         device: str = "cpu") -> BRIANHarness:
    """Build a harness wrapping the exact-match DSL transformer LM (N4/N5).

    This is the real language-model path: embedding → DSL TransformerBlocks
    → final norm → lm_head, each component bit-identical to the PyTorch
    reference. Loss clipping / optimizer / schedule come from arch.neuro's
    training block via the harness.
    """
    from neuroslm.dsl.nn_lang import build_dsl_language_cortex

    cfg = load_training_config_from_arch(arch_root)
    print(f"[train_dsl] DSL-LM (full N8 cortex): vocab={vocab_size} "
          f"d_model={d_model} depth={depth} heads={n_heads} ctx={max_ctx}")
    print(f"[train_dsl] training config: "
          f"loss_clip={cfg.loss_clipping.enabled}(f={cfg.loss_clipping.factor}), "
          f"opt={cfg.optimizer}, lr={cfg.learning_rate}, "
          f"grad_accum={cfg.grad_accum}, label_smooth={cfg.label_smoothing}, "
          f"wd={cfg.weight_decay}, dropout={cfg.dropout}, "
          f"pct_strength={cfg.pct_strength}")

    # Full DSL LanguageCortex: interleaved Standard/Diff/MoD blocks +
    # NeuralGeometryAdapter after each, bit-identical to Brain's
    # LanguageCortex(baseline=False) on the LM-logits path (N8 passes).
    # OOD-targeted: dropout on embed + per-block output controlled by
    # cfg.dropout (defaults to 0 to preserve bit-identical behavior).
    lm = build_dsl_language_cortex(
        vocab=vocab_size, d_model=d_model, depth=depth,
        n_heads=n_heads, max_ctx=max_ctx,
        dropout=cfg.dropout, pct_trunk=cfg.pct_trunk,
        tonnetz_period=cfg.tonnetz_period,
        stochastic_depth=cfg.stochastic_depth,
        grid_positions=cfg.grid_positions,
        episodic_memory=cfg.episodic_memory,
        surprise_head=cfg.surprise_head).to(device)
    harness = BRIANHarness.from_language_model(
        lm, vocab_size=vocab_size, d_sem=d_model, training_config=cfg,
    ).to(device)
    # PCT trunk-strength override: bump the PCH aux weight in the trunk
    # path. AuxWeights.pred_coding tuple = (weight, center, width).
    # Setting pct_strength > 0 multiplies the weight so PCH gradient
    # actually shapes the trunk (default 0.10 weight × 0.30 strength
    # boost = 0.13 effective trunk weight on PCH).
    if cfg.pct_strength > 0:
        from neuroslm.dsl.maturity import AuxWeights
        w, c, width = harness.total_loss_config.aux.pred_coding
        harness.total_loss_config.aux.pred_coding = (
            w * (1.0 + cfg.pct_strength), c, width)
        print(f"[train_dsl] PCT trunk-strength boost: "
              f"pred_coding weight {w} -> {w * (1.0 + cfg.pct_strength)}")

    # Attach the metric observer (Φ, λ₁, ignition, oscillations, NT,
    # trophic, mesoLG), seeded with the architecture's NT baselines.
    from neuroslm.dsl.metrics import MetricObserver
    nt_baselines = _nt_baselines_from_arch(arch_root)
    harness._observer = MetricObserver(n_layers=depth, nt_baselines=nt_baselines)
    harness.last_metrics = None

    n_params = sum(p.numel() for p in harness.parameters())
    print(f"[train_dsl] DSL-LM parameters: {n_params/1e6:.1f}M")
    return harness


def _nt_baselines_from_arch(arch_root: Path) -> Optional[dict]:
    """Read neurotransmitter base_concentrations from arch.neuro so the NT
    metric column starts at the architecture's declared baselines."""
    try:
        from neuroslm.dsl.multifile import compile_folder
        ir = compile_folder(arch_root)
        name_map = {"dopamine": "DA", "norepinephrine": "NE", "serotonin": "5HT",
                    "acetylcholine": "ACh", "endocannabinoid": "eCB",
                    "glutamate": "Glu", "gaba": "GABA"}
        out = {}
        for nt in ir.neurotransmitter_systems:
            key = name_map.get(nt.name)
            if key:
                out[key] = float(nt.base_concentration)
        return out or None
    except Exception:
        return None


def build_harness(arch_root: Path, vocab_size: int, d_sem: int,
                  device: str = "cpu",
                  sink_population: str = "motor") -> BRIANHarness:
    print(f"[train_dsl] compiling architecture from {arch_root}")
    ir = compile_folder(arch_root)
    print(f"[train_dsl]   populations:  {len(ir.populations)}")
    print(f"[train_dsl]   synapses:     {len(ir.synapses)}")
    print(f"[train_dsl]   modulations:  {len(ir.modulations)}")

    cfg = load_training_config_from_arch(arch_root)
    print(f"[train_dsl] training config: "
          f"loss_clip={cfg.loss_clipping.enabled}(f={cfg.loss_clipping.factor}), "
          f"opt={cfg.optimizer}, lr={cfg.learning_rate}, "
          f"grad_accum={cfg.grad_accum}, label_smooth={cfg.label_smoothing}")

    Cls = CodeGenerator(ir, module_name="DSLCircuit").compile_to_module()
    circuit = Cls(d_sem=d_sem).to(device)

    harness = BRIANHarness(
        circuit=circuit, vocab_size=vocab_size, d_sem=d_sem,
        training_config=cfg, sink_population=sink_population,
    ).to(device)

    # Apply declarative gradient isolation (p3 fix) from param_scope blocks.
    scopes = load_param_scopes_from_arch(arch_root)
    if scopes:
        harness.apply_param_scopes(scopes)
        detached = [s.name for s in scopes
                    if s.gradient == "detached_from_main_loss"]
        print(f"[train_dsl] param_scopes: {len(scopes)} declared, "
              f"detached-from-main-loss: {detached}")

    return harness


_RUN_ID_RE = re.compile(r"dsl_arch_(\d{8}-\d{6})_step(\d+)\.pt$")
_LEGACY_STEP_RE = re.compile(r"dsl_arch_step(\d+)\.pt$")


def _checkpoint_step(path: Path) -> int:
    """Extract step from `dsl_arch_{TS}_step{N}.pt` or legacy `dsl_arch_step{N}.pt`."""
    m = _RUN_ID_RE.search(path.name) or _LEGACY_STEP_RE.search(path.name)
    return int(m.group(2 if _RUN_ID_RE.search(path.name) else 1)) if m else 0


def _is_lfs_pointer(path: Path) -> bool:
    """True if `path` is an unpulled Git LFS pointer file (not the real blob).

    `git clone ... GIT_LFS_SKIP_SMUDGE=1` leaves .pt files on disk as plain
    text pointers (`version https://git-lfs.github.com/spec/v1\\n...`).
    `torch.load` on those raises UnpicklingError('invalid load key, v.')
    and looks like a checkpoint-corruption bug. Skip them here so resume
    only considers real checkpoints.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(48)
        return head.startswith(b"version https://git-lfs")
    except OSError:
        return False


def _maybe_resume(harness: BRIANHarness, ckpt_dir: Path) -> int:
    """Load the highest-step dsl_arch checkpoint, regardless of run-id prefix.

    Returns the resumed step (0 if no real checkpoint found). LFS pointer
    files are filtered out so a freshly-cloned container without LFS
    smudge doesn't crash on pickle-load.
    """
    if not ckpt_dir.is_dir():
        return 0
    all_ckpts = (list(ckpt_dir.glob("dsl_arch_*_step*.pt"))
                 + list(ckpt_dir.glob("dsl_arch_step*.pt")))
    real_ckpts = [p for p in all_ckpts if not _is_lfs_pointer(p)]
    skipped = len(all_ckpts) - len(real_ckpts)
    if skipped:
        print(f"[train_dsl] skipping {skipped} LFS pointer file(s) in {ckpt_dir}")
    if not real_ckpts:
        return 0
    latest = sorted(real_ckpts, key=_checkpoint_step)[-1]
    step = harness.load_checkpoint(str(latest))
    print(f"[train_dsl] resumed from {latest} @ step {step}")
    return step


# ── Train loop ───────────────────────────────────────────────────────

def _format_metrics_line(step: int, avg_loss: float, avg_lm: float,
                         gnorm: float, lr: float, tok_per_s: float,
                         metrics: Optional[Dict] = None) -> str:
    """Emit the native train.py per-step format so DSL and Brain logs are
    directly comparable. LM metrics are real; bowtie metrics (Φ, λ₁, ign,
    mesoLG, troph, NT, osc) come from `metrics` if the subsystems are
    present, else honest placeholders until N7-N8 ports them.
    """
    import math as _m
    ppl = _m.exp(min(avg_lm, 20.0))   # cap to avoid overflow on early steps
    m = metrics or {}
    phi = m.get("phi", 0.0)
    fid = m.get("fiedler", 0.0)
    ign = m.get("ignition", 0.0)
    lg = m.get("meso_lg", 0.0)
    t_act = m.get("troph_active", 0)
    t_tot = m.get("troph_total", 0)
    t_mu = m.get("troph_mean", 0.0)
    nt = m.get("nt", {})
    nt_str = " ".join(f"{k}={v:.2f}" for k, v in nt.items())
    osc = m.get("osc", {})
    osc_str = ""
    if osc:
        osc_str = " | osc[" + " ".join(f"{k}={v:.3f}" for k, v in osc.items()) + "]"
    # PR2: OOD-intervention telemetry (only printed if any is non-zero, so
    # the per-step line is unchanged for runs that don't enable them).
    reg_str = ""
    reg_keys = ("reg_dar", "reg_pcc", "reg_isotropy", "reg_cmd",
                "reg_total", "chat_ratio", "reg_warmup")
    reg_vals = {k: m.get(k, 0.0) for k in reg_keys}
    # Show row even during warmup-ramp (when total~0) so user sees
    # interventions are running — gated on warmup multiplier being
    # tracked (presence of reg_warmup key, even if 0) OR any non-zero
    # contribution.
    has_warmup_metric = "reg_warmup" in m
    if has_warmup_metric or any(abs(v) > 1e-9 for v in reg_vals.values()):
        reg_str = (" | reg["
                   f"dar={reg_vals['reg_dar']:.3f} "
                   f"pcc={reg_vals['reg_pcc']:.3f} "
                   f"iso={reg_vals['reg_isotropy']:.4f} "
                   f"cmd={reg_vals['reg_cmd']:.3f} "
                   f"Σ={reg_vals['reg_total']:.3f} "
                   f"w={reg_vals['reg_warmup']:.2f} "
                   f"chat={reg_vals['chat_ratio']:.2f}]")
    # Emergent C1–C6 telemetry tail (printed only when those keys are
    # present, so legacy runs without enable_emergent see no change).
    em_str = ""
    em_present = any(k in m for k in
                     ("ign_rate", "Q_total", "pac", "pc_residual",
                      "lattice_spec"))
    if em_present:
        em_parts = []
        if "ign_rate" in m:
            em_parts.append(
                f"C2:ρ={m.get('ign_rate', 0.0):.2f}"
                f"/τ={m.get('ign_threshold', 0.0):.2f}"
                f"/s={m.get('ign_strength', 0.0):.2f}")
        if "Q_total" in m:
            em_parts.append(
                f"C4:Q={m.get('Q_total', 0.0):.1f}"
                f" W={int(m.get('Q_walls', 0))}"
                f" pl={int(m.get('Q_plateau_len', 0))}")
        if "pc_residual" in m:
            em_parts.append(f"C3:pc={m.get('pc_residual', 0.0):.3f}")
        if "lattice_spec" in m:
            em_parts.append(f"C5:lat={m.get('lattice_spec', 0.0):.2f}")
        if "pac" in m:
            em_parts.append(f"C6:pac={m.get('pac', 0.0):.2f}")
        em_str = " | em[" + " ".join(em_parts) + "]"
    return (f"step {step:5d} | loss {avg_loss:.4f} | lm {avg_lm:.4f} "
            f"| ppl {ppl:.1f} | gnorm {gnorm:.3f} | lr {lr:.2e} "
            f"| {tok_per_s:.0f} tok/s "
            f"| Φ {phi:.3f} | λ₁ {fid:.3f} | ign {ign:.2f} "
            f"| mesoLG {lg:.2f} "
            f"| troph {t_act}/{t_tot} μ{t_mu:.2f} "
            f"| NT[{nt_str}]{osc_str}{em_str}{reg_str}")


def _eval_pass_marks(rules, step: int,
                      ppl_history, ood_history):
    """Evaluate all pass-mark rules; return (should_exit: bool, reason: str).

    `ppl_history` and `ood_history` are dicts {step: value}. The rule
    semantics mirror the PassMark dataclass.
    """
    for r in rules:
        # 1. Threshold-at-step
        if r.at_step > 0 and r.window == 0 and not r.trend:
            if step >= r.at_step:
                history = ppl_history if r.metric == "train_ppl" else ood_history
                # find closest measurement at-or-before at_step
                hits = [v for s, v in history.items() if s <= r.at_step]
                if not hits:
                    continue   # no measurement yet
                v = hits[-1]
                if r.max is not None and v > r.max:
                    return True, f"{r.name}: {r.metric}={v:.1f} > max {r.max}"
                if r.min is not None and v < r.min:
                    return True, f"{r.name}: {r.metric}={v:.1f} < min {r.min}"
        # 2. Stability over window
        elif r.window > 0 and r.trend == "stable":
            history = ppl_history if r.metric == "train_ppl" else ood_history
            recent = [v for s, v in sorted(history.items())
                      if s > step - r.window]
            if len(recent) < 3:
                continue
            spread = (max(recent) - min(recent)) / max(min(recent), 1e-6)
            if spread < r.tol:
                return True, (f"{r.name}: {r.metric} stable "
                              f"(spread {spread:.3f} < {r.tol}) over "
                              f"last {r.window} steps")
        # 3. Falling trend over window
        elif r.window > 0 and r.trend == "falling":
            history = ppl_history if r.metric == "train_ppl" else ood_history
            recent = sorted([(s, v) for s, v in history.items()
                             if s > step - r.window])
            # Need at least 4 datapoints in the window so a single noisy
            # OOD eval (50-window WikiText is ±3% noisy) can't trip the
            # rule. Split into two halves and compare half-MIN vs half-MIN
            # so we react to the *trend* not the noisy endpoints.
            min_evals = max(4, int(getattr(r, "min_evals", 0) or 0))
            if len(recent) < min_evals:
                continue
            mid = len(recent) // 2
            first_half = [v for _, v in recent[:mid]]
            second_half = [v for _, v in recent[mid:]]
            first_min = min(first_half)
            second_min = min(second_half)
            # "still falling" = second-half min strictly below first-half min
            # by more than `tol` relative. Otherwise → exit.
            if second_min >= first_min * (1.0 - r.tol):
                return True, (f"{r.name}: {r.metric} not falling "
                              f"(min[first_half]={first_min:.1f} vs "
                              f"min[second_half]={second_min:.1f}, tol={r.tol}, "
                              f"n={len(recent)}) over last {r.window} steps")
    return False, ""


def _mid_ood_eval(harness: BRIANHarness, step: int,
                   ckpt_dir: Optional[Path],
                   observer,
                   train_ppl_history: Optional[dict] = None) -> Optional[float]:
    """Quick WikiText-103 ppl snapshot, logged inline + written as JSON.

    Capped to 50 sliding windows so the eval costs <30s on an A100 and
    doesn't disrupt training cadence noticeably. Writes one JSON per
    checkpoint to <ckpt_dir>/../logs/vast/benchmarks/ood/.

    Args:
        train_ppl_history: optional {step: train_ppl} dict — if provided,
            computes `gap_ratio = ood_ppl / latest_train_ppl` and logs
            it inline. Lets you SEE generalization (ood/train) move
            during training without waiting for the final eval.

    Returns the OOD ppl as a float (or None on error). The caller
    feeds it into the pass-marks history for early-exit checks.
    """
    import json
    import math
    import torch
    from datasets import load_dataset

    print(f"[mid-ood] step {step}: WikiText-103 snapshot...", flush=True)
    was_training = harness.training
    harness.eval()
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test",
                          streaming=True, trust_remote_code=True)
        n_seq, total_loss, total_tok = 0, 0.0, 0
        # Use the same tokenizer the harness was built with — derived
        # from neuroslm.tokenizer.Tokenizer to keep BPE alignment exact.
        from neuroslm.tokenizer import Tokenizer
        tok = Tokenizer()
        ctx = getattr(harness.language_model, "max_ctx", 1024) or 1024
        for ex in ds:
            text = ex.get("text", "")
            if not text or len(text) < 50:
                continue
            ids = tok.encode(text)[: ctx + 1]
            if len(ids) < 16:
                continue
            ids_t = torch.tensor([ids[:-1]], device=next(harness.parameters()).device)
            tgt_t = torch.tensor([ids[1:]], device=ids_t.device)
            with torch.no_grad():
                logits = harness(ids_t)
                # cross-entropy per-token
                vocab = logits.shape[-1]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab), tgt_t.reshape(-1),
                    reduction="sum")
            total_loss += float(loss)
            total_tok += tgt_t.numel()
            n_seq += 1
            if n_seq >= 50:
                break
        avg_nll = total_loss / max(1, total_tok)
        ppl = math.exp(min(avg_nll, 20.0))
        # gap_ratio = ood_ppl / latest in-distribution train_ppl. >1 ⇒
        # generalization gap; <1.5 is excellent, >3 strong overfit.
        gap_ratio = None
        train_ppl = None
        if train_ppl_history:
            # use the most recent train_ppl at or before this step
            recent = [v for s, v in train_ppl_history.items() if s <= step]
            if recent:
                train_ppl = recent[-1]
                gap_ratio = ppl / train_ppl if train_ppl > 0 else None
        gap_str = (f" gap_ratio={gap_ratio:.2f} (train_ppl={train_ppl:.1f})"
                   if gap_ratio is not None else "")
        print(f"[mid-ood] step {step}: wikitext ppl={ppl:.1f}{gap_str} "
              f"({n_seq} seq, {total_tok} tok)", flush=True)
        # Persist to logs/vast/benchmarks/ood/ so analyze-log picks it up
        out_dir = Path("logs/vast/benchmarks/ood")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ood_mid_{_RUN_ID}_step{step}.json"
        out_path.write_text(json.dumps({
            "step": step,
            "run_id": _RUN_ID,
            "ood_dataset": "wikitext-103-v1",
            "ood_ppl": ppl,
            "train_ppl": train_ppl,
            "gap_ratio": gap_ratio,
            "n_sequences": n_seq,
            "n_tokens": total_tok,
            "kind": "mid-training",
        }, indent=2), encoding="utf-8")
        return ppl
    finally:
        if was_training:
            harness.train()


def train(harness: BRIANHarness, source: SyntheticBatchSource,
          steps: int, log_every: int = 20, save_every: int = 1000,
          ckpt_dir: Optional[Path] = None, start_step: int = 0,
          tokens_per_step: int = 0,
          ood_every: int = 0) -> None:
    """Run train_steps from `start_step+1` to `steps`. Emits the native
    train.py metric format; saves checkpoints.

    If `ood_every > 0`: every `ood_every` steps, runs a quick OOD ppl
    eval on WikiText-103 (capped to 50 windows for speed), prints the
    result inline, and writes a JSON to logs/vast/benchmarks/ood/
    ood_mid_<RUN_ID>_step{N}.json so it lands in the same per-step
    metrics ledger as the final OOD eval. Lets you SEE generalization
    improving (or not) while training is still running.
    """
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_log = t0
    log_buf = []

    observer = getattr(harness, "_observer", None)

    # ── Pass-mark histories — feed the early-exit checker ──
    pass_marks = getattr(harness.training_config, "pass_marks", None)
    pass_rules = pass_marks.rules if pass_marks else []
    train_ppl_history: dict = {}    # {step: train ppl}
    ood_ppl_history: dict = {}      # {step: ood ppl}

    for step in range(start_step + 1, steps + 1):
        ids, targets = source.next()
        if not tokens_per_step:
            tokens_per_step = ids.numel()
        loss = harness.train_step(ids, targets)
        log_buf.append(loss)

        # Update the metric observers from the live model's activations
        # (Φ, λ₁, ignition, oscillations, NT, trophic, mesoLG).
        # When the emergent layer is enabled (default), drive it with
        # the live training-state scalars (grad_norm, class label).
        if observer is not None:
            lm = getattr(harness, "language_model", None)
            acts = getattr(lm, "_layer_acts", None) if lm is not None else None
            if acts:
                _gnorm = float(getattr(harness, "_last_gnorm", 0.0)) or None
                _class = None
                # In `mix` mode the RealDataSource exposes the last
                # batch's per-window labels via ``_last_labels`` (a
                # 1-D tensor of length B with 0=text, 1=chat). Take
                # the modal label for the batch — that's what the C5
                # lattice probe wants as a single int.
                _labels = getattr(source, "_last_labels", None)
                if _labels is not None and getattr(_labels, "numel", lambda: 0)() > 0:
                    try:
                        _class = int(_labels.mode().values.item())
                    except Exception:
                        _class = int(_labels[0].item())
                else:
                    # Back-compat: synthetic sources may still publish
                    # the older string flag.
                    _kind = getattr(source, "last_batch_kind", None)
                    if _kind == "chat":
                        _class = 1
                    elif _kind in ("prose", "text"):
                        _class = 0
                # C3 motor/sensory analogs: last block's output is the
                # motor end of the cortical bow-tie; first block's
                # output is the sensory end. Both are (B, T, D).
                _h_motor = acts[-1] if len(acts) >= 1 else None
                _h_sensory = acts[0] if len(acts) >= 2 else None
                try:
                    harness.last_metrics = observer.observe(
                        acts, loss,
                        grad_norm=_gnorm,
                        h_motor=_h_motor,
                        h_sensory=_h_sensory,
                        class_label=_class,
                    )
                except TypeError:
                    # Back-compat: older observer signature (legacy
                    # MetricObserver.observe(acts, loss) without kwargs).
                    harness.last_metrics = observer.observe(acts, loss)

        if step % log_every == 0:
            now = time.time()
            avg = sum(log_buf) / len(log_buf)
            log_buf.clear()
            tok_per_s = tokens_per_step * log_every / max(now - last_log, 1e-6)
            gnorm = float(getattr(harness, "_last_gnorm", 0.0))
            lr = float(getattr(harness, "_last_lr", 0.0))
            metrics = getattr(harness, "last_metrics", None)
            # PR2: merge harness._metrics (carries reg_dar/pcc/iso/cmd/total
            # and chat_ratio published by the OOD-intervention controller)
            # into the per-step display dict.
            harness_metrics = getattr(harness, "_metrics", None)
            if harness_metrics:
                merged = dict(metrics) if metrics else {}
                for k, v in harness_metrics.items():
                    if k.startswith("reg_") or k == "chat_ratio":
                        merged[k] = v
                metrics = merged
            # avg_lm == avg total loss until auxiliary losses are added.
            print(_format_metrics_line(step, avg, avg, gnorm, lr,
                                        tok_per_s, metrics), flush=True)
            last_log = now
            # Record train PPL for pass-mark checks
            import math as _m
            train_ppl_history[step] = _m.exp(min(avg, 20.0))

        # ── Pass-mark early-exit check ──
        if pass_rules and step % log_every == 0:
            should_exit, reason = _eval_pass_marks(
                pass_rules, step, train_ppl_history, ood_ppl_history)
            if should_exit:
                print(f"[train_dsl] PASS-MARK EARLY EXIT @ step {step}: {reason}",
                      flush=True)
                # Save a final checkpoint so the run isn't lost
                if ckpt_dir is not None:
                    path = ckpt_dir / f"dsl_arch_{_RUN_ID}_step{step}.pt"
                    harness.save_checkpoint(str(path), step=step)
                    print(f"[train_dsl] saved early-exit checkpoint {path}",
                          flush=True)
                return

        if ckpt_dir is not None and step % save_every == 0:
            # Filename includes a per-run timestamp prefix so concurrent /
            # successive runs to the same dir never overwrite each other.
            # Pattern: dsl_arch_{YYYYMMDD-HHMMSS}_step{N}.pt
            # Resume globs all `dsl_arch_*_step*.pt` and picks highest step.
            path = ckpt_dir / f"dsl_arch_{_RUN_ID}_step{step}.pt"
            harness.save_checkpoint(str(path), step=step)
            print(f"[train_dsl] saved checkpoint {path}", flush=True)

        # Mid-training OOD eval — quick WikiText-103 ppl snapshot so we
        # can SEE generalization moving without waiting for end-of-run.
        # Writes one JSON per checkpoint to logs/vast/benchmarks/ood/
        # so analyze-log groups them with the final eval.
        if ood_every > 0 and step % ood_every == 0 and step > 0:
            try:
                ood_ppl = _mid_ood_eval(harness, step, ckpt_dir, observer,
                                          train_ppl_history=train_ppl_history)
                if ood_ppl is not None:
                    ood_ppl_history[step] = ood_ppl
                # Pass-mark check IMMEDIATELY after mid-OOD lands so
                # OOD-based rules (e.g. "exit when OOD < 700 at step 7k")
                # fire as soon as the data is available.
                if pass_rules:
                    should_exit, reason = _eval_pass_marks(
                        pass_rules, step, train_ppl_history, ood_ppl_history)
                    if should_exit:
                        print(f"[train_dsl] PASS-MARK EARLY EXIT @ step "
                              f"{step}: {reason}", flush=True)
                        if ckpt_dir is not None:
                            path = ckpt_dir / f"dsl_arch_{_RUN_ID}_step{step}.pt"
                            harness.save_checkpoint(str(path), step=step)
                            print(f"[train_dsl] saved early-exit "
                                  f"checkpoint {path}", flush=True)
                        return
            except Exception as e:
                print(f"[train_dsl] mid-OOD eval failed at step {step}: {e}",
                      flush=True)

    # ── End of training: ensure final checkpoint + run a final OOD pass ──
    # The loop exits when `step == steps` but the periodic save fires AT
    # `step % save_every == 0` BEFORE the step number rolls past. With
    # steps=10000, save_every=1000, the last saved checkpoint was step
    # 9000 (the 10000-loop iteration ends without re-checking save_every).
    # Always save a final checkpoint named with the target step + a
    # comprehensive OOD eval so the run has gap-ratio data.
    if ckpt_dir is not None:
        final_step = steps
        final_path = ckpt_dir / f"dsl_arch_{_RUN_ID}_step{final_step}.pt"
        harness.save_checkpoint(str(final_path), step=final_step)
        print(f"[train_dsl] saved final checkpoint {final_path}", flush=True)

    # Final OOD pass: a longer WikiText-103 eval (cap=200 windows instead
    # of the mid-OOD cap=50) plus a train-set ppl on the same loader, so
    # we can report gap_ratio = ood_ppl / train_ppl.
    try:
        _final_ood_eval(harness, steps, ckpt_dir, observer,
                         train_ppl_history=train_ppl_history,
                         ood_ppl_history=ood_ppl_history)
    except Exception as e:
        print(f"[train_dsl] final OOD eval failed: {e}", flush=True)


def _final_ood_eval(harness, step: int, ckpt_dir: Optional[Path],
                     observer, train_ppl_history: dict,
                     ood_ppl_history: dict) -> None:
    """End-of-training OOD eval: longer WikiText pass + gap_ratio.

    Writes `ood_final_{run_id}.json` to logs/vast/benchmarks/ood/ so the
    metrics ledger picks it up. Uses cap=200 sequences (4x the mid-OOD
    snapshot) for a less-noisy final estimate.
    """
    import json, math, torch
    from datasets import load_dataset
    from neuroslm.tokenizer import Tokenizer

    print(f"[train_dsl] final OOD eval @ step {step} (WikiText-103, cap=200)...",
          flush=True)
    was_training = harness.training
    harness.eval()
    tok = Tokenizer()
    ctx = getattr(harness.language_model, "max_ctx", 1024) or 1024
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test",
                      streaming=True, trust_remote_code=True)
    n_seq, total_loss, total_tok = 0, 0.0, 0
    for ex in ds:
        text = ex.get("text", "")
        if not text or len(text) < 50:
            continue
        ids = tok.encode(text)[: ctx + 1]
        if len(ids) < 16:
            continue
        ids_t = torch.tensor([ids[:-1]], device=next(harness.parameters()).device)
        tgt_t = torch.tensor([ids[1:]], device=ids_t.device)
        with torch.no_grad():
            logits = harness(ids_t)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt_t.reshape(-1),
                reduction="sum")
        total_loss += float(loss); total_tok += tgt_t.numel(); n_seq += 1
        if n_seq >= 200:
            break
    ood_ppl = math.exp(min(total_loss / max(1, total_tok), 20.0))
    # Train PPL = the latest in-distribution training perplexity (already
    # logged). gap_ratio = ood_ppl / train_ppl.
    last_train_step = max(train_ppl_history) if train_ppl_history else 0
    train_ppl = train_ppl_history.get(last_train_step, float("nan"))
    gap_ratio = ood_ppl / train_ppl if train_ppl and train_ppl > 0 else float("nan")
    print(f"[train_dsl] final OOD: wikitext ppl={ood_ppl:.1f}  "
          f"train_ppl={train_ppl:.1f}  gap_ratio={gap_ratio:.2f}  "
          f"({n_seq} seq, {total_tok} tok)", flush=True)
    out_dir = Path("logs/vast/benchmarks/ood")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ood_final_{_RUN_ID}.json"
    out_path.write_text(json.dumps({
        "kind": "final",
        "step": step,
        "ood_ppl": ood_ppl,
        "train_ppl": train_ppl,
        "gap_ratio": gap_ratio,
        "n_seq": n_seq,
        "n_tok": total_tok,
        "run_id": _RUN_ID,
    }, indent=2), encoding="utf-8")
    print(f"[train_dsl] wrote final OOD result → {out_path}", flush=True)
    if was_training:
        harness.train()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True,
                        help="path to architecture folder (containing arch.neuro)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_sem", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--ood_every", type=int, default=0,
                        help="If > 0, run a mid-training WikiText-103 OOD "
                             "ppl snapshot every N steps. Writes JSON to "
                             "logs/vast/benchmarks/ood/ for analyze-log "
                             "to pick up alongside the final eval.")
    parser.add_argument("--ckpt_dir", default="lfs_checkpoints")
    parser.add_argument("--sink", default="motor",
                        help="population whose output feeds the LM head")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="0 → take from tokenizer")
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16"],
                        help="mixed-precision dtype (cuda only)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest dsl_arch_step*.pt in ckpt_dir")
    parser.add_argument("--model", default="dsl_lm",
                        choices=["dsl_lm", "circuit"],
                        help="dsl_lm: exact-match transformer LM (N4/N5); "
                             "circuit: legacy per-token cognitive overlay")
    parser.add_argument("--preset", default=None,
                        help="size the DSL LM from a BrainConfig preset's "
                             "trunk dims (e.g. rcc_bowtie_30m_p4); overrides "
                             "--d_sem/--depth/--n_heads")
    parser.add_argument("--depth", type=int, default=6,
                        help="transformer depth (dsl_lm only)")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="attention heads (dsl_lm only)")
    parser.add_argument("--data", default="real", choices=["real", "synthetic"],
                        help="real: stream tokenized corpus; synthetic: random")
    parser.add_argument("--mode", default="mix",
                        help="data mode for real loader (text/chat/mix)")
    # 2026-06-03: default lowered 0.6 → 0.3 after the ground-truth
    # baseline run showed that at 0.6 the OOD gap_ratio holds steady
    # around 5× and prose ppl still falls in lockstep with train. At
    # 0.3 the model sees ~equal chat and prose tokens, which is the
    # cleanest mix for tracking WikiText OOD without architectural
    # interventions.
    parser.add_argument("--chat_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    arch_root = Path(args.arch).resolve()
    if not (arch_root / "arch.neuro").is_file():
        parser.error(f"missing {arch_root}/arch.neuro")

    tok = _load_tokenizer()
    vocab_size = args.vocab_size or tok.vocab_size

    # A --preset sizes the DSL LM to that preset's trunk (matches P4 etc.)
    d_model, depth, n_heads = args.d_sem, args.depth, args.n_heads
    preset_sched = None
    if args.preset:
        from neuroslm.dsl.preset_bridge import dsl_lm_config_from_preset
        pc = dsl_lm_config_from_preset(args.preset)
        d_model, depth, n_heads = pc["d_model"], pc["depth"], pc["n_heads"]
        if not args.vocab_size:
            vocab_size = pc["vocab"]
        preset_sched = pc   # carries lr / warmup_steps / min_lr_ratio / wd
        print(f"[train_dsl] preset {args.preset}: d_model={d_model} "
              f"depth={depth} heads={n_heads} vocab={vocab_size} "
              f"lr={pc['lr']} warmup={pc['warmup_steps']} "
              f"min_ratio={pc['min_lr_ratio']} wd={pc['weight_decay']}")

    # SCALE env var (or arch.neuro's `scales {}` block) overrides preset.
    # The deploy script sets SCALE + D_MODEL/DEPTH/... so the training
    # process picks up the right trunk + batch/ctx without rebuilding
    # the bash chain.
    _scale_env = os.environ.get("SCALE", "")
    if _scale_env or os.environ.get("D_MODEL"):
        from neuroslm.dsl.training_config import load_training_config_from_arch
        _tc_dsl = load_training_config_from_arch(arch_root)
        _v = _tc_dsl.scales.variants.get(_scale_env)
        if _v is not None:
            d_model = int(os.environ.get("D_MODEL", _v.d_model))
            depth   = int(os.environ.get("DEPTH",   _v.depth))
            n_heads = int(os.environ.get("N_HEADS", _v.n_heads))
            # argparse attribute names: --batch → args.batch, --seq_len →
            # args.seq_len, --d_sem → args.d_sem. The SCALE block in
            # arch.neuro overrides whatever the bash launcher passed.
            args.seq_len = int(os.environ.get("SEQ_LEN",  _v.seq_len))
            args.batch   = int(os.environ.get("BATCH_SIZE", _v.batch_size))
            args.d_sem   = d_model
            print(f"[train_dsl] scale {_scale_env} (~{_v.approx_params}): "
                  f"d_model={d_model} depth={depth} heads={n_heads} "
                  f"batch={args.batch} ctx={args.seq_len} "
                  f"grad_accum={_v.grad_accum} dist={(_v.hardware or _tc_dsl.hardware).dist_strategy}")

    if args.model == "dsl_lm":
        harness = build_dsl_lm_harness(
            arch_root=arch_root, vocab_size=vocab_size, d_model=d_model,
            depth=depth, n_heads=n_heads, max_ctx=args.seq_len,
            device=args.device,
        )
    else:
        harness = build_harness(
            arch_root=arch_root, vocab_size=vocab_size, d_sem=args.d_sem,
            device=args.device, sink_population=args.sink,
        )

    # LR schedule — use the preset's exact warmup/peak/min_ratio so the
    # DSL run's learning-rate curve matches Brain's cosine_lr step-for-step
    # (validated in tests/dsl/test_lr_parity.py). Fall back to 10% warmup.
    if preset_sched is not None:
        harness.training_config.learning_rate = preset_sched["lr"]
        harness.training_config.weight_decay = preset_sched["weight_decay"]
        warmup = preset_sched["warmup_steps"]
        min_ratio = preset_sched["min_lr_ratio"]
    else:
        warmup = max(1, args.steps // 10)
        min_ratio = 0.1
    harness.set_schedule(warmup=warmup, total=args.steps, min_lr_ratio=min_ratio)
    if args.device == "cuda":
        harness.enable_mixed_precision(dtype=args.amp)
        print(f"[train_dsl] mixed precision: {args.amp}")

    # Distributed wrapping — read DIST_STRATEGY from env (set by the deploy
    # script from arch.neuro's hardware {} block). torchrun is responsible
    # for launching N processes; this just initialises the process group
    # and wraps the LM in DDP/FSDP. Single-process runs are a no-op.
    _dist = os.environ.get("DIST_STRATEGY", "single")
    if _dist not in ("", "single") and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print(f"[train_dsl] distributed: {_dist} (world_size="
              f"{os.environ.get('WORLD_SIZE')} local_rank="
              f"{os.environ.get('LOCAL_RANK')})")
        harness.enable_distributed(strategy=_dist)

    print(harness.topology_summary())
    print(f"[train_dsl] device={args.device}, warmup={warmup}/{args.steps}")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    start_step = 0
    if args.resume and ckpt_dir is not None:
        start_step = _maybe_resume(harness, ckpt_dir)
        harness._global_step = start_step

    if args.data == "real":
        # PR2: wire AdaptiveMixtureController + DARReweighter to the data
        # source. Only activate when the corresponding intervention is
        # enabled in the .neuro regularization {} block — otherwise default
        # to plain batch_iterator (back-compat).
        reg_cfg = getattr(harness.training_config, "regularization", None)
        adaptive_on = (
            reg_cfg is not None
            and getattr(reg_cfg, "adaptive_mixture", None) is not None
            and reg_cfg.adaptive_mixture.enabled
            and args.mode == "mix"
        )
        dar_on = (
            reg_cfg is not None
            and getattr(reg_cfg, "dar", None) is not None
            and reg_cfg.dar.enabled
            and args.mode == "mix"
        )
        ratio_ref = harness.current_chat_ratio if adaptive_on else None
        # Always emit per-sample class labels: the emergent C5 lattice
        # probe consumes them via ``source._last_labels`` even when DAR
        # is off. The cost is one int per window — negligible.
        with_labels = True

        # Loud, unmissable PR2 banner — if the next training log doesn't
        # show this exact line, the new code path is NOT active.
        if reg_cfg is not None:
            flags = []
            for name in ("dar", "pcc", "isotropy", "cmd", "adaptive_mixture"):
                sub = getattr(reg_cfg, name, None)
                on = bool(sub and getattr(sub, "enabled", False))
                flags.append(f"{name}={'ON' if on else 'off'}")
            print("[train_dsl] ══════════════════════════════════════════════")
            print(f"[train_dsl] PR2 OOD interventions: {' '.join(flags)}")
            print("[train_dsl] ══════════════════════════════════════════════")
        if adaptive_on:
            print(f"[train_dsl] adaptive_mixture ON — ratio_ref → harness.current_chat_ratio()")
        if dar_on:
            print(f"[train_dsl] DAR domain labels ON — emitting (window, source_id) tuples")

        source = RealDataSource(
            tok, batch=args.batch, seq_len=args.seq_len, device=args.device,
            mode=args.mode, chat_ratio=args.chat_ratio, seed=args.seed,
            ratio_ref=ratio_ref, with_labels=with_labels,
        )
        if dar_on:
            harness.set_domain_id_fn(source.domain_id_fn)
    else:
        source = SyntheticBatchSource(
            vocab_size=vocab_size, batch=args.batch, seq_len=args.seq_len,
            device=args.device, seed=args.seed,
        )

    train(
        harness=harness, source=source,
        steps=args.steps, log_every=args.log_every,
        save_every=args.save_every, ckpt_dir=ckpt_dir,
        start_step=start_step, ood_every=args.ood_every,
    )

    print("[train_dsl] done.")


if __name__ == "__main__":
    main()
