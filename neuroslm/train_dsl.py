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
                 chat_ratio: float = 0.6, seed: int = 0):
        self.device = device
        self._fallback = None
        try:
            from neuroslm.data import batch_iterator
            self._it = batch_iterator(
                tokenizer, ctx_len=seq_len, batch_size=batch,
                seed=seed, mode=mode, chat_ratio=chat_ratio,
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
            window, self._primed = self._primed, None
        else:
            window = next(self._it)
        window = window.to(self.device)
        return window[:, :-1].contiguous(), window[:, 1:].contiguous()


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
        n_heads=n_heads, max_ctx=max_ctx, dropout=cfg.dropout).to(device)
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


def _maybe_resume(harness: BRIANHarness, ckpt_dir: Path) -> int:
    """Load the highest-step dsl_arch checkpoint, regardless of run-id prefix.

    Returns the resumed step (0 if no checkpoint found).
    """
    if not ckpt_dir.is_dir():
        return 0
    ckpts = sorted(
        list(ckpt_dir.glob("dsl_arch_*_step*.pt"))
        + list(ckpt_dir.glob("dsl_arch_step*.pt")),
        key=_checkpoint_step,
    )
    if not ckpts:
        return 0
    latest = ckpts[-1]
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
    return (f"step {step:5d} | loss {avg_loss:.4f} | lm {avg_lm:.4f} "
            f"| ppl {ppl:.1f} | gnorm {gnorm:.3f} | lr {lr:.2e} "
            f"| {tok_per_s:.0f} tok/s "
            f"| Φ {phi:.3f} | λ₁ {fid:.3f} | ign {ign:.2f} "
            f"| mesoLG {lg:.2f} "
            f"| troph {t_act}/{t_tot} μ{t_mu:.2f} "
            f"| NT[{nt_str}]{osc_str}")


def _mid_ood_eval(harness: BRIANHarness, step: int,
                   ckpt_dir: Optional[Path],
                   observer) -> None:
    """Quick WikiText-103 ppl snapshot, logged inline + written as JSON.

    Capped to 50 sliding windows so the eval costs <30s on an A100 and
    doesn't disrupt training cadence noticeably. Writes one JSON per
    checkpoint to <ckpt_dir>/../logs/vast/benchmarks/ood/.
    """
    import json
    import math
    import torch
    from datasets import load_dataset

    print(f"[mid-ood] step {step}: WikiText-103 snapshot...", flush=True)
    was_training = harness.training
    harness.eval()
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test",
                          streaming=True)
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
        print(f"[mid-ood] step {step}: wikitext ppl={ppl:.1f} "
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
            "n_sequences": n_seq,
            "n_tokens": total_tok,
            "kind": "mid-training",
        }, indent=2), encoding="utf-8")
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

    for step in range(start_step + 1, steps + 1):
        ids, targets = source.next()
        if not tokens_per_step:
            tokens_per_step = ids.numel()
        loss = harness.train_step(ids, targets)
        log_buf.append(loss)

        # Update the metric observers from the live model's activations
        # (Φ, λ₁, ignition, oscillations, NT, trophic, mesoLG).
        if observer is not None:
            lm = getattr(harness, "language_model", None)
            acts = getattr(lm, "_layer_acts", None) if lm is not None else None
            if acts:
                harness.last_metrics = observer.observe(acts, loss)

        if step % log_every == 0:
            now = time.time()
            avg = sum(log_buf) / len(log_buf)
            log_buf.clear()
            tok_per_s = tokens_per_step * log_every / max(now - last_log, 1e-6)
            gnorm = float(getattr(harness, "_last_gnorm", 0.0))
            lr = float(getattr(harness, "_last_lr", 0.0))
            metrics = getattr(harness, "last_metrics", None)
            # avg_lm == avg total loss until auxiliary losses are added.
            print(_format_metrics_line(step, avg, avg, gnorm, lr,
                                        tok_per_s, metrics), flush=True)
            last_log = now

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
                _mid_ood_eval(harness, step, ckpt_dir, observer)
            except Exception as e:
                print(f"[train_dsl] mid-OOD eval failed at step {step}: {e}",
                      flush=True)


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
    parser.add_argument("--chat_ratio", type=float, default=0.6)
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

    print(harness.topology_summary())
    print(f"[train_dsl] device={args.device}, warmup={warmup}/{args.steps}")

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    start_step = 0
    if args.resume and ckpt_dir is not None:
        start_step = _maybe_resume(harness, ckpt_dir)
        harness._global_step = start_step

    if args.data == "real":
        source = RealDataSource(
            tok, batch=args.batch, seq_len=args.seq_len, device=args.device,
            mode=args.mode, chat_ratio=args.chat_ratio, seed=args.seed,
        )
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
