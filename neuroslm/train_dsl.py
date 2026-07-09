# -*- coding: utf-8 -*-
"""DSL-driven training entrypoint (supports both DSL architectures and DNA).

Parallel to `neuroslm.train` (which trains the hand-written `Brain`):
this module loads an architecture from `architectures/<name>/` OR from
a DNA file, compiles it via the DSL pipeline, wraps it in a `BRIANHarness`,
and runs a language-model training loop.

The harness reads `training { ... }` from the architecture's
`arch.neuro` for loss clipping, optimizer choice, label smoothing,
grad accumulation, and grad clipping. Per-step model behavior is
otherwise determined entirely by the .neuro files — no Python
architecture code path involved.

Usage (DSL):
    python -m neuroslm.train_dsl --arch architectures/current \\
        --steps 10000 --batch 4 --seq_len 256 --d_sem 256

Usage (DNA):
    python -m neuroslm.train_dsl --dna dna/evol/arch.dna \\
        --steps 10000 --batch 4 --seq_len 256 --d_sem 256
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
from typing import Optional, Dict


# Per-run id stamped into checkpoint filenames so concurrent / successive
# runs to the same ckpt_dir never overwrite each other. Settable via the
# DSL_RUN_ID env var (vast_train_dsl_loop sets it when resuming); falls
# back to current UTC time at import.
_RUN_ID = os.environ.get(
    "DSL_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))

# Per-run checkpoint subdir label. Mirrors log_pusher.sh `LABEL` so the
# checkpoint directory and the log filename can be cross-referenced
# mechanically (same RUN_ID + same label). Settable via DSL_ARCH_LABEL
# env var (_deploy_train.py exports it from the deploy --label). Falls
# back to "run" for unlabelled local-dev runs.
_ARCH_LABEL = os.environ.get("DSL_ARCH_LABEL", "run")

import torch

from neuroslm.checkpoint_push import push_checkpoint
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


# ── Boot stamp (forensic-friction reduction) ────────────────────────
#
# Every training run prints a 3-line boot stamp BEFORE any other
# train_dsl output. The stamp answers the three questions a forensic
# investigator always asks first:
#
#   1. WHEN was this run?              → UTC ISO-8601 timestamp
#   2. WHICH git commit produced it?   → 40-hex sha + branch name
#   3. WHICH DSL did it compile?       → SHA-256 of the unfolded
#                                         arch.neuro files
#
# (1) and (2) together let you ``git checkout <sha>`` and reproduce
# the run. (3) catches the "I edited arch.neuro between launches and
# forgot" failure mode that produced two identical-looking deploys
# with different behaviour.
#
# Regression-pinned by ``tests/training/test_train_dsl_boot_stamp.py``.


def _git_commit_info() -> tuple[str, str]:
    """Return ``(sha40, branch)`` for the current HEAD. On failure
    (not a git repo, missing git binary), returns ``("-", "-")`` —
    the boot stamp must never crash the trainer."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if len(sha) == 40 and re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha, branch or "-"
    except Exception:
        pass
    return "-", "-"


def _arch_dsl_sha256(arch_root: Optional[Path]) -> str:
    """SHA-256 of the canonical unfolded DSL bytes for ``arch_root``.

    Canonicalisation: concatenate every ``*.neuro`` file in the folder
    (recursive), sorted by relative path, with a separator line that
    embeds the relative path so a file rename also changes the hash.
    This matches the unfolded view ``brian unfold`` would produce —
    two runs with the same arch SHA loaded byte-identical DSL.

    Returns ``"-"`` if ``arch_root`` is None or contains no ``*.neuro``
    files — the boot stamp must never crash.
    """
    if arch_root is None:
        return "-"
    root = Path(arch_root)
    if not root.is_dir():
        return "-"
    neuro_files = sorted(
        root.rglob("*.neuro"),
        key=lambda p: str(p.relative_to(root)).replace("\\", "/"),
    )
    if not neuro_files:
        return "-"
    h = hashlib.sha256()
    for p in neuro_files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        # Separator embeds the path so renames change the hash
        h.update(b"\n--- " + rel.encode("utf-8") + b" ---\n")
        try:
            h.update(p.read_bytes())
        except OSError:
            # Unreadable file is treated as empty — don't crash the boot
            continue
    return h.hexdigest()


def _collect_nfo_metrics(harness) -> Dict[str, float]:
    """Pull NFO telemetry from any ``NeuralFieldOscillator`` block in
    *harness*.

    Walks the module tree once per call (cheap — typically <50 modules)
    and returns a flat ``{nfo_R_mean, nfo_R_max, nfo_A_mean, nfo_A_std,
    nfo_phi_circular_var, nfo_kappa, nfo_dt, nfo_alpha, nfo_phi_kappa}``
    dict mirroring :class:`neuroslm.emergent.nfo_coherence.NFOCoherenceProbe`.

    Returns an empty dict when no NFO block is present (legacy archs
    without the H015..H018 block) or when ``last_state`` is empty
    (block hasn't been forward-passed yet, e.g. step 0 in a freshly
    constructed harness).

    Multiple NFO blocks (unusual but legal) are aggregated as a simple
    mean — they currently share `last_state` keys so the aggregation
    is well-defined.
    """
    try:
        from neuroslm.modules.neural_field_oscillator import (
            NeuralFieldOscillator,
        )
    except ImportError:
        # NFO module missing → arch can't have NFO either. Cheap exit.
        return {}

    SCALAR_KEYS = (
        ("R_mean", "nfo_R_mean"),
        ("R_max", "nfo_R_max"),
        ("A_mean", "nfo_A_mean"),
        ("A_std", "nfo_A_std"),
        ("phi_circular_var", "nfo_phi_circular_var"),
        ("kappa", "nfo_kappa"),
        ("dt", "nfo_dt"),
        ("alpha", "nfo_alpha"),
        ("phi_kappa", "nfo_phi_kappa"),
    )

    blocks: list = []
    if hasattr(harness, "modules"):
        for m in harness.modules():
            if isinstance(m, NeuralFieldOscillator):
                state = getattr(m, "last_state", None)
                if state:
                    blocks.append(state)

    if not blocks:
        return {}

    out: Dict[str, float] = {}
    for src_key, dst_key in SCALAR_KEYS:
        vals = []
        for state in blocks:
            v = state.get(src_key)
            if v is None:
                continue
            if hasattr(v, "detach"):
                # Avoid sync-on-print: copy + flatten to CPU scalar once.
                try:
                    vals.append(float(v.detach().mean().cpu().item()))
                except Exception:
                    continue
            else:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
        if vals:
            out[dst_key] = sum(vals) / len(vals)
    return out


def _print_boot_stamp(arch_root: Optional[Path] = None) -> None:
    """Emit the 3-line boot stamp. See the module-level comment above.

    Always prints exactly 3 lines, each prefixed with ``[train_dsl] ``
    so the existing log-pipeline grep filters keep matching. Robust to
    missing ``arch_root``, missing git, missing files — degraded fields
    show as ``-`` rather than crashing the trainer.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha, branch = _git_commit_info()
    dsl_sha = _arch_dsl_sha256(arch_root)
    arch_label = str(arch_root) if arch_root is not None else "-"
    print(f"[train_dsl] boot @ {now}", flush=True)
    print(f"[train_dsl] git_commit {sha} ({branch})", flush=True)
    print(
        f"[train_dsl] arch_dsl_sha256 {dsl_sha} ({arch_label})",
        flush=True,
    )


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
    window: ids = window[:, :-1], targets = window[:, 1:].

    Fallback chain (in order):
      1. HuggingFace streaming (FineWeb-Edu / SmolLM / TinyStories /
         wikitext) via :func:`neuroslm.data.batch_iterator`.
      2. Bundled local corpus (no network needed) via
         :func:`neuroslm.data.local_corpus_batch_iterator`.
      3. ``SyntheticBatchSource`` as a last resort with a LOUD warning,
         because random batches produce the ``log(V)`` loss plateau.

    HF priming uses ``max_open_attempts=1`` so an unreachable network
    fails *fast* (within seconds) instead of hanging in the iterator's
    exponential-backoff reconnect loop. The fallback chain then runs
    immediately so the training run starts on real text either way.
    """
    def __init__(self, tokenizer, batch: int, seq_len: int,
                 device: str = "cpu", mode: str = "mix",
                 chat_ratio: float = 0.6, seed: int = 0,
                 ratio_ref=None, with_labels: bool = False):
        self.device = device
        self._fallback = None
        self._local_it = None        # ``local_corpus_batch_iterator`` handle
        self._with_labels = with_labels
        self._last_labels = None  # set by next() when with_labels=True
        self.source_label = "real-hf"

        # ── Step 1: try real HF stream (fail fast on offline) ───────
        try:
            from neuroslm.data import batch_iterator
            self._it = batch_iterator(
                tokenizer, ctx_len=seq_len, batch_size=batch,
                seed=seed, mode=mode, chat_ratio=chat_ratio,
                ratio_ref=ratio_ref, with_labels=with_labels,
                max_open_attempts=1,
            )
            # Prime one batch to surface failures early.
            self._primed = next(self._it)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[train_dsl] HF stream unavailable "
                  f"({type(e).__name__}: {e}); trying local corpus...")
            self._it = None
            self._primed = None

        # ── Step 2: bundled local corpus ───────────────────────────
        try:
            from neuroslm.data import local_corpus_batch_iterator
            self._local_it = local_corpus_batch_iterator(
                tokenizer, ctx_len=seq_len, batch_size=batch,
            )
            # Prime once so failures surface here.
            self._primed = next(self._local_it)
            self.source_label = "local-corpus"
            print("[train_dsl] using bundled local corpus "
                  "(neuroslm/assets/local_corpus.txt) — no network needed")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[train_dsl] local corpus unavailable "
                  f"({type(e).__name__}: {e}); falling back to synthetic")

        # ── Step 3: synthetic last resort (with WARNING) ───────────
        print("=" * 70)
        print("[train_dsl] WARNING: all real-data sources failed — using "
              "synthetic torch.randint batches.")
        print("[train_dsl] WARNING: lm_loss will plateau at log(vocab_size) "
              "and the model will NOT learn linguistic structure.")
        print("[train_dsl] Fix: provide network access OR restore "
              "neuroslm/assets/local_corpus.txt.")
        print("=" * 70)
        self._fallback = SyntheticBatchSource(
            tokenizer.vocab_size, batch, seq_len, device, seed)
        self.source_label = "synthetic-random"

    def next(self):
        if self._fallback is not None:
            return self._fallback.next()
        if self._local_it is not None:
            if self._primed is not None:
                window, self._primed = self._primed, None
            else:
                window = next(self._local_it)
            window = window.to(self.device)
            # Local corpus iterator never emits per-sample domain
            # labels — synthesize all-zeros so DAR (when on) still has
            # a tensor to consume.
            if self._with_labels:
                import torch as _torch
                self._last_labels = _torch.zeros(
                    window.shape[0], dtype=_torch.long, device=self.device)
            return (window[:, :-1].contiguous(),
                    window[:, 1:].contiguous())
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

def _run_trunk_probe(harness, ids, targets, *, step: int, arch_name, preset_name,
                     root=None, pop: int = 24, gens: int = 10, length: int = 8,
                     sites: int = 2):
    """Read-only discovery probe on the real trunk (best-effort; never crashes).

    Every ``explore_every`` steps: when the trunk exposes ``forward_from_layer``
    (DSLLanguageCortex — the class real training builds), run the multi-site
    probe: measure per-layer headroom under the TRUE loss (each perturbation
    re-run through the real remaining blocks + PCT + norm + head), then search
    NGL modulations only at layers that still have slack. H46 proved the
    terminal hidden is a null site — the budget goes where the model is
    demonstrably under-optimized. Falls back to the legacy terminal-hidden
    re-projection for models without the layer stash. Never touches training
    weights or the forward — pure measurement — so it cannot perturb the run.
    Winners persist to ``modulations/`` with their measured Δ + site (run_id
    encodes the layer: ``trunk-<preset>-L<k>``).
    """
    import torch
    from neuroslm.dsl import nn_ops
    from neuroslm.genetic.ledger import SearchLedger
    from neuroslm.genetic.modulation_store import ModulationStore
    from neuroslm.genetic.training_explorer import probe_hidden_modulation, ExploreConfig

    lm = getattr(harness, "language_model", None)
    if lm is None or getattr(lm, "lm_head", None) is None:
        return None

    root = Path(root) if root is not None else _REPO_ROOT
    store = ModulationStore(root / "modulations")
    # normalize=False: skip per-candidate canonicalization (only needed for
    # cross-run dedup, which a fresh per-probe ledger doesn't do) so the deep
    # search spends its budget scoring real candidates, not simplifying them.
    cfg = ExploreConfig(pop_size=pop, generations=gens, length=length,
                        normalize=False)
    run_id = f"trunk-{preset_name or arch_name or 'run'}"

    if hasattr(lm, "forward_from_layer"):
        from neuroslm.genetic.layer_probe import probe_optimizable_regions
        return probe_optimizable_regions(
            lm, ids, targets, store=store, config=cfg, step=step,
            run_id=run_id, top_k=sites, seed=step)

    # ── legacy fallback: terminal-hidden re-projection through the head ──
    with torch.no_grad():
        harness(ids)                       # populate _last_hidden without grad
    h = getattr(lm, "_last_hidden", None)
    if h is None or h.dim() != 3:
        return None
    h = h.detach().float()
    W = lm.lm_head
    cosine = bool(getattr(lm, "_cosine_head", False))
    _t = getattr(lm, "head_temperature", 1.0)
    temp = float(_t.detach()) if torch.is_tensor(_t) else float(_t)

    def head_fn(x):
        if cosine:
            return nn_ops.cosine_lm_head(x, W, temp)
        return nn_ops.linear(x, W)

    # Fresh in-memory ledger per probe: the trunk changes every explore_every
    # steps, so a modulation that didn't help at one checkpoint may help at the
    # next — cross-checkpoint dud-skipping (from the shared/toy ledger) would
    # over-prune the search. Each checkpoint gets a full search; winners still
    # persist to modulations/.
    led = SearchLedger(":memory:")
    return probe_hidden_modulation(
        h, head_fn, targets.to(h.device), ledger=led, store=store,
        config=cfg, step=step, run_id=run_id)


def _scale_override_note(scale_name: str, cli_seq: int, cli_batch: int,
                         eff_seq: int, eff_batch: int) -> Optional[str]:
    """Warn when the arch SCALE block silently discards CLI --seq_len/--batch.

    The scale is the source of truth for trunk dims, so ``args.seq_len`` /
    ``args.batch`` from the launcher are overwritten by the scale's values unless
    ``SEQ_LEN`` / ``BATCH_SIZE`` env vars are set. Passing ``--seq_len 2048
    --batch 16`` and silently training at 512/1 is a real footgun — surface it.
    """
    changed = []
    if cli_seq != eff_seq:
        changed.append(f"--seq_len {cli_seq}→{eff_seq}")
    if cli_batch != eff_batch:
        changed.append(f"--batch {cli_batch}→{eff_batch}")
    if not changed:
        return None
    return (f"[train_dsl] NOTE: scale '{scale_name}' overrode CLI args ("
            + ", ".join(changed)
            + "). The arch SCALE block wins on dims — set SEQ_LEN / BATCH_SIZE "
            "env vars (or edit the scale) to keep your CLI values.")


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

    # Resolve n_kv_heads from config (0 → full MHA = n_heads).
    _n_kv_heads = int(getattr(cfg, "n_kv_heads", 0) or 0)
    if _n_kv_heads <= 0:
        # Try to read from the active scale variant.
        try:
            _sv = cfg.scales.variants.get(cfg.scales.default)
            if _sv is not None and _sv.n_kv_heads > 0:
                _n_kv_heads = _sv.n_kv_heads
        except Exception:
            pass
    _n_kv_heads = _n_kv_heads or n_heads
    _rope_base = float(getattr(cfg, "rope_base", 10000.0))
    print(f"[train_dsl] GQA: n_kv_heads={_n_kv_heads}/{n_heads}  "
          f"RoPE base={_rope_base:.0f}")

    # Full DSL LanguageCortex: interleaved Standard/Diff/MoD blocks +
    # NeuralGeometryAdapter after each, bit-identical to Brain's
    # LanguageCortex(baseline=False) on the LM-logits path (N8 passes).
    # OOD-targeted: dropout on embed + per-block output controlled by
    # cfg.dropout (defaults to 0 to preserve bit-identical behavior).
    lm = build_dsl_language_cortex(
        vocab=vocab_size, d_model=d_model, depth=depth,
        n_heads=n_heads, max_ctx=max_ctx,
        n_kv_heads=_n_kv_heads,
        dropout=cfg.dropout, pct_trunk=cfg.pct_trunk,
        tonnetz_period=cfg.tonnetz_period,
        stochastic_depth=cfg.stochastic_depth,
        grid_positions=cfg.grid_positions,
        episodic_memory=cfg.episodic_memory,
        surprise_head=cfg.surprise_head,
        nfo=cfg.nfo,
        cosine_head=cfg.cosine_head,
        rope_base=_rope_base).to(device)
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
    harness._observer = MetricObserver(
        n_layers=depth,
        nt_baselines=nt_baselines,
        # Item 6: when the arch declares `nt_w_trainable: true`, the
        # DrivenNTSystem inside the observer exposes its 7×5 coupling
        # matrix as an nn.Parameter so the optimiser can refine the
        # NT-driver coupling end-to-end via the differentiable
        # `predict_nt_tensor` readout. Off by default.
        nt_w_trainable=bool(getattr(cfg, "nt_w_trainable", False)),
    )
    harness.last_metrics = None

    # Item 6: if W is trainable, register its single Parameter with
    # the harness's optimiser so SGD actually updates it. The observer
    # is not a submodule of the LM, so `harness.parameters()` would
    # miss it without an explicit add_param_group on first optim
    # construction. The harness handles this lazily in `train_step`
    # via the `_extra_params` hook (added below).
    if bool(getattr(cfg, "nt_w_trainable", False)):
        nt_module = harness._observer._emergent["nt"]
        if getattr(nt_module, "W_param", None) is not None:
            existing = getattr(harness, "_extra_trainable_params", [])
            harness._extra_trainable_params = list(existing) + [
                nt_module.W_param
            ]
            print(
                f"[train_dsl] Item 6: NT coupling matrix W "
                f"({nt_module.W_param.numel()} scalars) registered as "
                f"trainable Parameter."
            )

    n_total     = sum(p.numel() for p in harness.parameters())
    n_trainable = sum(p.numel() for p in harness.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable
    # Frozen-fraction is what tells you "this number looks huge because
    # of the HF experts loaded into multi_cortex.experts.*" — keeps the
    # 1.1 B total honest while making the ~30 M trainable trunk visible.
    print(
        f"[train_dsl] DSL-LM parameters: total {n_total/1e6:.1f}M  "
        f"(trainable {n_trainable/1e6:.1f}M · frozen {n_frozen/1e6:.1f}M); "
        f"checkpoint will only save the trainable subset "
        f"(frozen HF experts excluded via _CKPT_EXTERNAL_PREFIXES)"
    )

    # ── TRUNK-OPT monitor: auto-attach (Phase 1 measurement) ────────
    # Wires all six probes into the harness so trunk[budget bpp erank pac]
    # appear in every log line and in harness._metrics — zero overhead
    # on runs that don't care, full measurement on every SmolLM run.
    from neuroslm.emergent.trunk_opt import TrunkOptMonitor as _TOM
    _monitor = _TOM(n_train=max(1, n_trainable), pac_delta=0.05,
                    prior_sigma=0.02)
    harness.attach_trunk_opt_monitor(_monitor)
    print(f"[train_dsl] TrunkOptMonitor attached "
          f"(n_trainable={n_trainable/1e6:.1f}M, "
          f"pac_sigma=0.02, pac_delta=0.05)")

    # ── GIF OOD probe: arm the held-out evaluator ──
    # The probe needs a tokenizer + device to download and cache WikiText
    # sequences. Without this call, the adaptive GIF controller never
    # sees OOD signal and stays at the static floor.
    from neuroslm.tokenizer import Tokenizer as _TokCls
    _probe_tok = _TokCls()
    _probe_device = torch.device(device)
    if harness.load_gif_probe(_probe_tok, _probe_device):
        print("[train_dsl] GIF OOD probe armed")

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
                  sink_population: str = "motor",
                  use_hypergraph_executor: bool = True,
                  heatmap_every: int = 500,
                  heatmap_path: Optional[str] = None,
                  heatmap_push: bool = False) -> BRIANHarness:
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

    hir = None
    if use_hypergraph_executor:
        from neuroslm.compiler.hypergraph_ir import lift_arch_to_hypergraph
        from neuroslm.compiler.hypergraph_executor import HypergraphExecutor
        hir = lift_arch_to_hypergraph(arch_root)
        circuit = HypergraphExecutor(hir, d_model=d_sem).to(device)
        print(f"[train_dsl] circuit: HypergraphExecutor "
              f"({len(hir.nodes_of_kind('population'))} populations, "
              f"{len([e for e in hir.hyperedges if e.kind == 'synapse'])} synapses)")
    else:
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

    # ── GIF OOD probe: arm the held-out evaluator ──
    from neuroslm.tokenizer import Tokenizer as _TokCls2
    _probe_tok2 = _TokCls2()
    _probe_device2 = torch.device(device)
    if harness.load_gif_probe(_probe_tok2, _probe_device2):
        print("[train_dsl] GIF OOD probe armed")

    # ── Heatmap hook: incremental gradient heat over the hypergraph ──
    # Only active when the executor is in use and heatmap_every > 0.
    harness._heatmap_hook = None
    if use_hypergraph_executor and hir is not None and heatmap_every > 0:
        from neuroslm.compiler.hypergraph_executor import executor_activation_norms
        from neuroslm.evolution.harness_hook import HeatmapHook
        from neuroslm.evolution.publisher import HeatmapPublisher

        _hm_path = heatmap_path or "results/heatmaps/hypergraph.heatmap.json"
        _png_path = str(Path(_hm_path).with_suffix(".png"))

        def _png_renderer(hm, path, _ir=hir):
            from neuroslm.compiler.nfg_graphviz import render_hypergraph
            render_hypergraph(_ir, path, engine="neato", heat=hm, format="png")

        publisher = HeatmapPublisher(
            heatmap_path=_hm_path,
            commit_every=heatmap_every,
            push=heatmap_push,
            png_renderer=_png_renderer,
            png_path=_png_path,
        )

        _circuit = circuit  # capture for lambda
        harness._heatmap_hook = HeatmapHook(
            model=harness,
            ir=hir,
            every_n=heatmap_every,
            publisher=publisher,
            grad_norm_fn=lambda: executor_activation_norms(_circuit),
            verbose=True,
        )
        print(f"[train_dsl] heatmap hook armed: every {heatmap_every} steps → {_png_path}")

    return harness


_RUN_ID_RE = re.compile(r"dsl_arch_(\d{8}-\d{6})_step(\d+)\.pt$")
_LEGACY_STEP_RE = re.compile(r"dsl_arch_step(\d+)\.pt$")
# New per-run subdir layout (H24+):
#   lfs_checkpoints/<RUN_ID>_<GIT_SHORT>_<ARCH_LABEL>/step<N>.pt
# - <RUN_ID>   = ``%Y%m%d-%H%M%S`` UTC, mirroring _RUN_ID env var
# - <GIT_SHORT>= 8-char git short SHA at deploy time
# - <ARCH_LABEL>= filesystem-safe slug of the deploy --label suffix
#   (or "run" when unlabelled). Slashes / spaces are normalised so
#   the directory name is portable across Linux + Windows checkouts.
_NEW_LAYOUT_RE = re.compile(r"step(\d+)\.pt$")


def _git_short_sha() -> str:
    """Return the 8-char short SHA of HEAD, or "unknown" if git fails.

    Used to embed the trunk version into checkpoint directory names so
    you can tell at a glance which commit produced which artefact.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _sanitise_label(label: str) -> str:
    """Make ``label`` safe to use as a directory component.

    Strips path separators, collapses whitespace runs to single ``-``,
    drops control characters, and lower-cases. Empty input becomes
    ``"run"`` so the directory name always has a third component.
    """
    if not label:
        return "run"
    # Replace path separators + whitespace with single dashes
    cleaned = re.sub(r"[\s/\\]+", "-", label.strip())
    # Drop anything that's not alnum, dash, dot, or underscore
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "", cleaned)
    cleaned = cleaned.strip("-_.")
    return cleaned or "run"


def build_run_dir_name(run_id: str, git_short: str, arch_label: str) -> str:
    """Build the per-run checkpoint subdirectory name.

    Layout: ``<RUN_ID>_<GIT_SHORT>_<ARCH_LABEL>`` — same separator
    convention as ``scripts/log_pusher.sh`` uses for log filenames so
    cross-referencing logs ↔ checkpoints is mechanical (same prefix).

    Mirrors the contract:

      * RUN_ID first → ``ls lfs_checkpoints/`` sorts chronologically.
      * GIT_SHORT next → per-commit traceability.
      * ARCH_LABEL last → per-arch differentiation; arbitrary length.
    """
    return f"{run_id}_{git_short}_{_sanitise_label(arch_label)}"


def checkpoint_path_for_step(
        ckpt_root: Path, run_dir_name: str, step: int) -> Path:
    """Return ``<ckpt_root>/<run_dir_name>/step<N>.pt`` (no mkdir)."""
    return ckpt_root / run_dir_name / f"step{step}.pt"


_GIT_SHORT_SHA: Optional[str] = None


def _cached_git_short_sha() -> str:
    """Return ``_git_short_sha()`` cached on first call.

    The on-box log-pusher cron commits + pushes ``training.log`` every
    ~5 minutes, which advances ``HEAD`` mid-run. Calling
    ``_git_short_sha()`` live at each save therefore produces a
    DIFFERENT subdir per save (observed in the H24-cfd-10k log:
    ``…_41072f7b_…/step1000.pt``, ``…_31677927_…/step2000.pt``,
    ``…_7fdc3ccd_…/step3000.pt``). That scatters checkpoints across N
    directories and breaks the resume globber's expectation of one
    canonical run-dir. Caching at first use (which happens during the
    training-loop save path, AFTER any startup-time commits the box
    might have made) pins the suffix for the rest of the run.
    """
    global _GIT_SHORT_SHA
    if _GIT_SHORT_SHA is None:
        _GIT_SHORT_SHA = _git_short_sha()
    return _GIT_SHORT_SHA


def _run_dir_name() -> str:
    """Convenience wrapper: build the per-run dir name from module-level
    ``_RUN_ID`` + ``_ARCH_LABEL`` + cached ``_cached_git_short_sha()``.

    Computed lazily (not at module import) so a test can monkey-patch
    ``_git_short_sha`` to make the dir deterministic.
    """
    return build_run_dir_name(_RUN_ID, _cached_git_short_sha(), _ARCH_LABEL)


def _checkpoint_step(path: Path) -> int:
    """Extract step from any of the three supported layouts:

      * ``dsl_arch_{TS}_step{N}.pt``  (H21–H23 flat)
      * ``dsl_arch_step{N}.pt``        (very-old flat)
      * ``<RUN_DIR>/step{N}.pt``       (H24+ per-run subdir)
    """
    m = (_RUN_ID_RE.search(path.name)
         or _LEGACY_STEP_RE.search(path.name)
         or _NEW_LAYOUT_RE.match(path.name))
    if not m:
        return 0
    # _RUN_ID_RE has two groups (TS, step); the others have one (step).
    return int(m.group(2 if m.re is _RUN_ID_RE else 1))


def _find_resume_candidates(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return ``[(step, path), ...]`` for every resume-eligible checkpoint
    under ``ckpt_dir``, traversing BOTH the legacy flat layout AND the
    new per-run subdirectory layout. LFS pointer files are filtered out.

    Used by ``_maybe_resume`` to pick the highest-step entry.
    """
    if not ckpt_dir.is_dir():
        return []
    # Legacy flat layout: direct children of ckpt_dir.
    flat = (list(ckpt_dir.glob("dsl_arch_*_step*.pt"))
            + list(ckpt_dir.glob("dsl_arch_step*.pt")))
    # New per-run subdir layout: <ckpt_dir>/*/step<N>.pt
    subdir = list(ckpt_dir.glob("*/step*.pt"))
    all_ckpts = flat + subdir
    real = [p for p in all_ckpts if not _is_lfs_pointer(p)]
    skipped = len(all_ckpts) - len(real)
    if skipped:
        print(f"[train_dsl] skipping {skipped} LFS pointer file(s) in {ckpt_dir}")
    return [(_checkpoint_step(p), p) for p in real if _checkpoint_step(p) > 0]


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
    """Load the highest-step dsl_arch checkpoint, regardless of layout.

    Traverses both the legacy flat layout (``dsl_arch_*_step*.pt``)
    AND the new per-run subdir layout (``<RUN_DIR>/step<N>.pt``).
    Returns the resumed step (0 if no real checkpoint found).
    """
    candidates = _find_resume_candidates(ckpt_dir)
    if not candidates:
        return 0
    top_step, top_path = max(candidates, key=lambda x: x[0])
    step = harness.load_checkpoint(str(top_path))
    print(f"[train_dsl] resumed from {top_path} @ step {step}")
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
    m = metrics or {}
    # When cortex fusion is active, lm_loss_ema tracks the trunk's pre-fusion
    # CE (SmolLM excluded). Use it so the displayed ppl reflects only what the
    # trunk itself has learned. Falls back to avg_lm when cortex is absent.
    _trunk_ema = m.get("lm_loss_ema", None)
    _ppl_nats = _trunk_ema if (_trunk_ema is not None and _trunk_ema > 0) else avg_lm
    ppl = _m.exp(min(_ppl_nats, 20.0))
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
    # Cortex fusion telemetry — appears only when fusion is active
    # (any of alpha_effective / cortex_inhibition / distill_* present).
    # Reports α_eff (the actual mixing weight after NT-gating), the
    # inhibition level (0=cortex active, 1=cortex gated off), and the
    # KL-distillation strength λ (0 ⇒ distillation idle).
    cortex_str = ""
    cortex_keys = ("alpha_effective", "cortex_inhibition",
                   "distill_lambda", "distill_kl",
                   "lm_loss_ema", "cortex_loss_ema")
    if any(k in m for k in cortex_keys):
        cortex_str = (" | cortex["
                      f"α_eff={m.get('alpha_effective', 0.0):.3f} "
                      f"inh={m.get('cortex_inhibition', 0.0):.3f} "
                      f"λ={m.get('distill_lambda', 0.0):.3f} "
                      f"kl={m.get('distill_kl', 0.0):.3f} "
                      f"lm_ema={m.get('lm_loss_ema', 0.0):.2f} "
                      f"cx_ema={m.get('cortex_loss_ema', 0.0):.2f}]")
    # Allostasis (synthetic HPA axis) telemetry — surfaces ONLY when
    # the controller is active so legacy runs without the block see no
    # change to the log format. `load` is the fast stress integrator
    # (acute); `cort` is the slow integrator (chronic); the three
    # multipliers show how strongly each effector is damping.
    allostasis_str = ""
    if "allostasis_cort" in m:
        allostasis_str = (" | hpa["
            f"load={m.get('allostasis_load', 0.0):.2f} "
            f"cort={m.get('allostasis_cort', 0.0):.2f} "
            f"NE×{m.get('allostasis_ne_mult', 1.0):.2f} "
            f"T×{m.get('allostasis_trophic_mult', 1.0):.2f} "
            f"LR×{m.get('allostasis_lr_mult', 1.0):.2f}]")
    # ── GIF-7 (Homeostatic Gradient Equilibrium) telemetry ──
    # Surfaces the three mechanism read-outs so we can confirm
    # divisive norm, loss-variance damping, and KL floor are active.
    gif7_str = ""
    gif7_keys = ("gif7_dgn_scale", "gif7_lr_mult", "gif7_kl_floor")
    if any(k in m for k in gif7_keys):
        g7_parts = []
        if "gif7_dgn_scale" in m:
            g7_parts.append(f"dgn={m['gif7_dgn_scale']:.3f}")
        if "gif7_lr_mult" in m:
            g7_parts.append(f"lr×{m['gif7_lr_mult']:.3f}")
        if "gif7_kl_floor" in m:
            g7_parts.append(f"klf={m['gif7_kl_floor']:.1f}")
        gif7_str = " | gif7[" + " ".join(g7_parts) + "]"
    # GIF (Geometric Information Funnel) telemetry — shows only when
    # GIF is active. Surfaces the three mechanism read-outs so we can
    # confirm the funnel is actually operating: vbb_α (IB tightness),
    # ood_ema (true-generalisation probe EMA), iso_w (isotropy weight).
    # Adaptive mode adds: p (progress 0→1), gap (PPL gap ratio).
    gif_str = ""
    gif_keys = ("gif_ood_probe_ema", "gif_ood_probe_ce",
                "gif_isotropy_weight", "gif_vbb_alpha",
                "gif_progress", "gif_gap_ratio")
    if any(k in m for k in gif_keys):
        parts = [f"α={m.get('gif_vbb_alpha', 0.0):.4f}"]
        if "gif_progress" in m:
            parts.append(f"p={m['gif_progress']:.3f}")
        if "gif_gap_ratio" in m:
            parts.append(f"gap={m['gif_gap_ratio']:.2f}")
        parts.append(f"ood_ema={m.get('gif_ood_probe_ema', 0.0):.2f}")
        parts.append(f"ood_ce={m.get('gif_ood_probe_ce', 0.0):.2f}")
        parts.append(f"iso_w={m.get('gif_isotropy_weight', 0.0):.4f}")
        if "gif_label_smooth" in m:
            parts.append(f"ls={m['gif_label_smooth']:.3f}")
        if "gif_head_div_loss" in m:
            parts.append(f"div={m['gif_head_div_loss']:.3f}")
        gif_str = " | gif[" + " ".join(parts) + "]"
    # ── NFO (Neural Field Oscillator) telemetry ──
    # Surfaces only when the arch has an NFO block (cf. SmolLM
    # arch.neuro `nfo: { enabled: true, ... }`). Keys flow from the
    # block's ``last_state`` dict through ``_collect_nfo_metrics``.
    #
    #  R   — Kuramoto order parameter (0=incoherent, 1=phase-locked)
    #  R★  — peak across oscillators (max coherence in any cluster)
    #  A   — Swift–Hohenberg amplitude mean (sigmoid-bounded by μ)
    #  σA  — amplitude std (high σA + low cVar = traveling waves)
    #  cV  — phase circular variance (1=uniform, 0=delta peak)
    #  κ   — coupling strength (Kuramoto K parameter)
    #  α   — ReZero gate (0=identity, 1=full NFO contribution)
    #  Φκ  — bipartition coherence lower bound (provable Φ surrogate)
    nfo_str = ""
    nfo_keys = ("nfo_R_mean", "nfo_R_max", "nfo_A_mean", "nfo_A_std",
                "nfo_phi_circular_var", "nfo_kappa", "nfo_alpha",
                "nfo_phi_kappa")
    if any(k in m for k in nfo_keys):
        nparts = []
        if "nfo_R_mean" in m:
            nparts.append(f"R={m['nfo_R_mean']:.2f}")
        if "nfo_R_max" in m:
            nparts.append(f"R★={m['nfo_R_max']:.2f}")
        if "nfo_A_mean" in m:
            nparts.append(f"A={m['nfo_A_mean']:.2f}")
        if "nfo_A_std" in m:
            nparts.append(f"σA={m['nfo_A_std']:.2f}")
        if "nfo_phi_circular_var" in m:
            nparts.append(f"cV={m['nfo_phi_circular_var']:.2f}")
        if "nfo_kappa" in m:
            nparts.append(f"κ={m['nfo_kappa']:.2f}")
        if "nfo_alpha" in m:
            nparts.append(f"α={m['nfo_alpha']:.3f}")
        if "nfo_phi_kappa" in m:
            nparts.append(f"Φκ={m['nfo_phi_kappa']:.2f}")
        nfo_str = " | nfo[" + " ".join(nparts) + "]"
    # Emergent C1–C6 telemetry tail (printed only when those keys are
    # present, so legacy runs without enable_emergent see no change).
    em_str = ""
    em_present = any(k in m for k in
                     ("ign_rate", "Q_total", "pac", "pc_residual",
                      "lattice_spec", "vbb_beta", "vbb_sigma_mean"))
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
        if "vbb_beta" in m:
            em_parts.append(
                f"VBB:β={m.get('vbb_beta', 0.0):.2f}"
                f" σ={m.get('vbb_sigma_mean', 0.0):.4f}"
                f" kl={m.get('vbb_kl', 0.0):.3f}")
        if "lattice_spec" in m:
            em_parts.append(f"C5:lat={m.get('lattice_spec', 0.0):.2f}")
        if "pac" in m:
            em_parts.append(f"C6:pac={m.get('pac', 0.0):.2f}")
        em_str = " | em[" + " ".join(em_parts) + "]"

    # ── TRUNK-OPT metrics (Phase 1) ──
    trunk_str = ""
    to_keys = ("trunk_opt_grad_budget", "trunk_opt_bits_per_param",
               "trunk_opt_effective_rank", "trunk_opt_pac_bayes_bound",
               "trunk_opt_layer_uniformity",
               "trunk_opt_power_alpha", "trunk_opt_power_r2",
               "trunk_opt_dpr")
    to_parts = []
    if m.get("trunk_opt_grad_budget") is not None:
        to_parts.append(f"budget={m['trunk_opt_grad_budget']:.2f}")
    if m.get("trunk_opt_bits_per_param") is not None:
        to_parts.append(f"bpp={m['trunk_opt_bits_per_param']:.2e}")
    if m.get("trunk_opt_effective_rank") is not None:
        to_parts.append(f"erank={m['trunk_opt_effective_rank']:.1f}")
    if m.get("trunk_opt_pac_bayes_bound") is not None:
        to_parts.append(f"pac≤{m['trunk_opt_pac_bayes_bound']:.3f}")
    if m.get("trunk_opt_layer_uniformity") is not None:
        to_parts.append(f"uni={m['trunk_opt_layer_uniformity']:.2f}")
    # Spectral power-law geometry (novel invariant — biological 1/f signature).
    # α ≈ 1.0 + R² > 0.9 means the trunk has crystallised onto the
    # cortical scale-free regime; α >> 2 signals bottleneck collapse.
    if m.get("trunk_opt_power_alpha") is not None:
        to_parts.append(f"α={m['trunk_opt_power_alpha']:.2f}")
    if m.get("trunk_opt_power_r2") is not None:
        to_parts.append(f"R²={m['trunk_opt_power_r2']:.2f}")
    if m.get("trunk_opt_dpr") is not None:
        to_parts.append(f"PR={m['trunk_opt_dpr']:.1f}")
    if to_parts:
        trunk_str = " | trunk[" + " ".join(to_parts) + "]"

    # ── STE (Semantic Turbulence Engine) telemetry ──
    ste_str = ""
    if "ste_rho" in m or "ste_sigma" in m:
        sp = []
        if "ste_sigma" in m:
            sp.append(f"σ={m['ste_sigma']:.3f}")
        if "ste_rho" in m:
            sp.append(f"ρ={m['ste_rho']:.3f}")
        if "ste_gaba" in m:
            sp.append(f"G={m['ste_gaba']:.2f}")
        if "ste_ne" in m:
            sp.append(f"N={m['ste_ne']:.2f}")
        if "ste_da" in m:
            sp.append(f"D={m['ste_da']:.3f}")
        if "ste_crit_loss" in m:
            sp.append(f"crit={m['ste_crit_loss']:.4f}")
        ste_str = " | ste[" + " ".join(sp) + "]"

    return (f"step {step:5d} | loss {avg_loss:.4f} | lm {avg_lm:.4f} "
            f"| ppl {ppl:.1f} | gnorm {gnorm:.3f} | lr {lr:.2e} "
            f"| {tok_per_s:.0f} tok/s "
            f"| Φ {phi:.3f} | λ₁ {fid:.3f} | ign {ign:.2f} "
            f"| mesoLG {lg:.2f} "
            f"| troph {t_act}/{t_tot} μ{t_mu:.2f} "
            f"| NT[{nt_str}]{osc_str}{em_str}{reg_str}"
            f"{cortex_str}{gif_str}{gif7_str}{nfo_str}"
            f"{allostasis_str}{trunk_str}{ste_str}")


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


def _ood_eval_logits(harness, ids):
    """Logits for the mid-training OOD probe — the STANDALONE TRUNK, with
    the cortex dropped.

    The deploy target is a trunk that stands on its own (distill-then-drop-
    the-cortex), so the OOD ppl must reflect what the trunk has internalised,
    NOT the fused cortex+trunk output. This makes the OOD metric consistent
    with the trunk-only train ppl (both ``exp(lm_loss_ema)``); otherwise
    gap_ratio compares a fused OOD against a trunk-only train ppl and is
    meaningless. Falls back to the full fused forward when the harness has
    no separable ``language_model`` (legacy per-token circuit path)."""
    lm = getattr(harness, "language_model", None)
    if lm is not None:
        return lm(ids)
    return harness(ids)


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
                logits = _ood_eval_logits(harness, ids_t)  # trunk-only
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
          ood_every: int = 0,
          push_every: int = 0,
          push_backend: str = "hf",
          push_optimizer: bool = False,
          heatmap_every: int = 0,
          explore_every: int = 0,
          explore_pop: int = 24,
          explore_gens: int = 10,
          explore_len: int = 8,
          explore_sites: int = 2,
          arch_name: Optional[str] = None,
          preset_name: Optional[str] = None,
          collect_heatmap: bool = True,
          run_heatmap_every: int = 500) -> None:
    """Run train_steps from `start_step+1` to `steps`. Emits the native
    train.py metric format; saves checkpoints.

    If `ood_every > 0`: every `ood_every` steps, runs a quick OOD ppl
    eval on WikiText-103 (capped to 50 windows for speed), prints the
    result inline, and writes a JSON to logs/vast/benchmarks/ood/
    ood_mid_<RUN_ID>_step{N}.json so it lands in the same per-step
    metrics ledger as the final OOD eval. Lets you SEE generalization
    improving (or not) while training is still running.

    If `push_every > 0`: every save whose step is divisible by
    `push_every` is followed by a push of the checkpoint via the
    dispatcher (:func:`neuroslm.checkpoint_push.push_checkpoint`).
    The active backend is `push_backend` — one of:

      * ``"hf"`` — HuggingFace Hub ``upload_file`` (default after
        2026-06-15; ran 41063959 hung at step 500 because the old
        ``git push`` of a 569 MB LFS object raced the background
        log-pusher and never returned)
      * ``"lfs"`` — legacy ``git add``/``commit``/``push``
      * ``"none"`` — no remote push (local-dev runs)

    Closes the H24 (run 41031063, 2026-06-15) loss-hole where the
    vast box self-destroyed before the end-of-training push and
    stranded every checkpoint. Defaults to 0 (off) so local-dev runs
    aren't tied to the git remote.
    """
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    last_log = t0
    log_buf = []
    # Separate LM-only buffer so the `lm` column reports the bare
    # cross-entropy, not the aux-inflated total. Without this the user
    # sees ``loss 13.7 | lm 13.7`` even when the true LM CE is 10.8
    # nats (the spread comes from VBB + PC-reentry + MSPCC + distill).
    # That cosmetic equality masks a 3-nat-wide aux contribution.
    lm_log_buf = []

    observer = getattr(harness, "_observer", None)

    # ── Pass-mark histories — feed the early-exit checker ──
    pass_marks = getattr(harness.training_config, "pass_marks", None)
    pass_rules = pass_marks.rules if pass_marks else []
    train_ppl_history: dict = {}    # {step: train ppl}
    ood_ppl_history: dict = {}      # {step: ood ppl}

    # Per-arch/preset run heatmap: a grad hook captures per-parameter grad norms
    # during backward (train_step zeroes grads before we could read them), so the
    # heatmap reflects real training. Best-effort — never breaks training.
    _grad_collector = None
    _hstore = None
    if arch_name is not None and collect_heatmap and run_heatmap_every > 0:
        try:
            from neuroslm.genetic.heatmap_store import HeatmapStore, GradHeatCollector
            _grad_collector = GradHeatCollector(harness)
            _hstore = HeatmapStore(Path(__file__).resolve().parent.parent / "heatmaps")
            print(f"[train_dsl] run-heatmap armed: every {run_heatmap_every} steps "
                  f"→ heatmaps/{Path(arch_name).name}/{preset_name or 'default'}.json", flush=True)
        except Exception as _e:
            print(f"[train_dsl] run-heatmap disabled: {_e!r}", flush=True)

    for step in range(start_step + 1, steps + 1):
        ids, targets = source.next()
        if not tokens_per_step:
            tokens_per_step = ids.numel()
        # ── NT distribution: source the live homeostat dict from the
        # PREVIOUS step's observer state and pass it into train_step.
        # The observer is updated AFTER train_step below, so this dict
        # is one step stale — negligible over a 2k-step run, and the
        # natural ~1-step latency mirrors transcription delay in real
        # NT signalling. At step 0 the observer hasn't been touched
        # yet; `getattr` returns None and the harness leaves all NT
        # consumers at their identity defaults (back-compat).
        _live_nt = None
        if observer is not None and observer.enable_emergent:
            try:
                _live_nt = observer._emergent["nt"].levels()
            except Exception:
                _live_nt = None
        loss = harness.train_step(ids, targets, nt_levels=_live_nt)
        log_buf.append(loss)

        # Heatmap: fire the hook every heatmap_every steps (after backward).
        _hm_hook = getattr(harness, "_heatmap_hook", None)
        if _hm_hook is not None and heatmap_every > 0:
            _hm_hook.step(step)

        # Read-only trunk discovery probe (best-effort — never crashes training).
        if explore_every > 0 and step % explore_every == 0 and step > 0:
            try:
                _pr = _run_trunk_probe(harness, ids, targets, step=step,
                                       arch_name=arch_name, preset_name=preset_name,
                                       pop=explore_pop, gens=explore_gens,
                                       length=explore_len, sites=explore_sites)
                if _pr is not None:
                    _tag = f"saved {_pr['saved']}" if _pr.get("saved") else "no keep"
                    print(f"[train_dsl] explore step {step}: baseline_ce={_pr['baseline_ce']:.4f} "
                          f"best_ce={_pr['best_ce']:.4f} Δ={_pr['delta_ce']:.4f} "
                          f"evaluated={_pr['evaluated']} ({_tag})", flush=True)
                    from neuroslm.genetic.modulation_pusher import push_artifacts
                    push_artifacts(_REPO_ROOT, ["modulations"],
                                   message=f"explore: trunk probe step {step}")
            except Exception as _e:
                print(f"[train_dsl] explore probe skipped: {_e!r}", flush=True)

        # Per-arch/preset run heatmap: record + push every run_heatmap_every steps.
        if _grad_collector is not None and step % run_heatmap_every == 0:
            try:
                from neuroslm.genetic.heatmap_store import record_training_run
                from neuroslm.genetic.modulation_pusher import push_artifacts
                _rh = record_training_run(_hstore, arch_name, preset_name or "default",
                                          harness, step=step,
                                          grad_norms=_grad_collector.latest())
                _hot = _rh.summary.get("hot", [])
                print(f"[train_dsl] heatmap step {step}: heatmaps/{_rh.arch}/{_rh.preset}.json "
                      f"(max={_rh.summary.get('max', 0):.2f}, hottest="
                      f"{_hot[0][0] if _hot else '-'})", flush=True)
                _pr = push_artifacts(Path(__file__).resolve().parent.parent, ["heatmaps"],
                                     message=f"heatmap: {_rh.arch}/{_rh.preset} step {step}")
                if _pr.get("pushed"):
                    print(f"[train_dsl] heatmap pushed → {_pr.get('branch')}", flush=True)
            except Exception as _e:
                print(f"[train_dsl] heatmap step skipped: {_e!r}", flush=True)

        # `harness._last_lm_loss_value` is the LM-only CE (set inside
        # compute_loss before aux terms are added). When aux losses are
        # active (VBB, PC reentry, MSPCC, distillation) this lags below
        # the total by 2-4 nats — surfaced as the `lm` column below.
        lm_only = float(getattr(harness, "_last_lm_loss_value", loss))
        lm_log_buf.append(lm_only)

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
            avg_lm = sum(lm_log_buf) / len(lm_log_buf) if lm_log_buf else avg
            log_buf.clear()
            lm_log_buf.clear()
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
                    # Forward every key the harness publishes (PR2 reg_*,
                    # chat_ratio, cortex fusion telemetry, allostasis HPA
                    # state). `_format_metrics_line` decides which of
                    # these to render.
                    merged[k] = v
                metrics = merged
            # NFO telemetry — pulled from any NeuralFieldOscillator
            # block in the harness's module tree. Returns an empty
            # dict when the arch has no NFO (so legacy archs see no
            # change to the log format). Surfaces R, κ, α, Φκ so we
            # can verify the H015..H018 block is actually firing.
            nfo_metrics = _collect_nfo_metrics(harness)
            if nfo_metrics:
                if metrics is None:
                    metrics = {}
                metrics.update(nfo_metrics)
            # `avg` = total loss (LM + aux); `avg_lm` = LM-only CE.
            # When aux losses are off they're equal; when on, the
            # `loss` column shows the optimization target and the `lm`
            # column shows the bare language-modeling cross-entropy.
            print(_format_metrics_line(step, avg, avg_lm, gnorm, lr,
                                        tok_per_s, metrics), flush=True)
            last_log = now
            # Record train PPL for pass-mark checks + mid-OOD gap_ratio.
            # When cortex fusion is active, use lm_loss_ema (trunk-only CE)
            # so gap_ratio measures trunk vs OOD, not SmolLM vs OOD.
            # Without cortex, fall back to avg_lm (LM-only CE, not total).
            # Pinned by tests/test_mid_ood_uses_lm_only_loss.py.
            import math as _m
            _trunk_ema_hist = (metrics or {}).get("lm_loss_ema", None)
            _nats_for_hist = (
                _trunk_ema_hist
                if (_trunk_ema_hist is not None and _trunk_ema_hist > 0)
                else avg_lm
            )
            train_ppl_history[step] = _m.exp(min(_nats_for_hist, 20.0))

        # ── Pass-mark early-exit check ──
        if pass_rules and step % log_every == 0:
            should_exit, reason = _eval_pass_marks(
                pass_rules, step, train_ppl_history, ood_ppl_history)
            if should_exit:
                print(f"[train_dsl] PASS-MARK EARLY EXIT @ step {step}: {reason}",
                      flush=True)
                # Save a final checkpoint so the run isn't lost
                if ckpt_dir is not None:
                    path = checkpoint_path_for_step(
                        ckpt_dir, _run_dir_name(), step)
                    harness.save_checkpoint(str(path), step=step)
                    print(f"[train_dsl] saved early-exit checkpoint {path}",
                          flush=True)
                    # Always push the early-exit ckpt regardless of
                    # cadence — it's the LAST artefact of the run.
                    # Force push_optimizer=True so the HF copy is a
                    # perfect resume target (no ~500-step LR-warmup
                    # blip on a fresh-box reload — losing that on a
                    # final artefact wastes whatever compute the
                    # early-exit just saved).
                    if push_every > 0:
                        push_checkpoint(
                            str(path), backend=push_backend,
                            push_optimizer=True,
                        )
                return

        if ckpt_dir is not None and step % save_every == 0:
            # H24+ per-run subdir layout:
            #   lfs_checkpoints/<RUN_ID>_<GIT_SHORT>_<ARCH_LABEL>/step<N>.pt
            # Mirrors scripts/log_pusher.sh naming so logs ↔ checkpoints
            # share a unique prefix per (commit, arch, day). Resume globs
            # both this layout AND the legacy flat ones.
            path = checkpoint_path_for_step(
                ckpt_dir, _run_dir_name(), step)
            harness.save_checkpoint(str(path), step=step)
            print(f"[train_dsl] saved checkpoint {path}", flush=True)
            # Optionally push to the configured backend so an
            # instance crash never strands artefacts. ``push_every``
            # is normally either 0 (off) or equal to ``save_every``
            # (push every save). A value > save_every pushes less
            # often than it saves. ``push_backend`` selects HF Hub
            # (default) or legacy Git LFS — see :func:`push_checkpoint`.
            # ``push_optimizer`` honours the CLI flag: when False
            # (default) the HF backend strips Adam state for ~3×
            # smaller uploads; the local .pt always has the full
            # payload for same-box crash resume.
            if push_every > 0 and step % push_every == 0:
                push_checkpoint(
                    str(path), backend=push_backend,
                    push_optimizer=push_optimizer,
                )

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
                            path = checkpoint_path_for_step(
                                ckpt_dir, _run_dir_name(), step)
                            harness.save_checkpoint(str(path), step=step)
                            print(f"[train_dsl] saved early-exit "
                                  f"checkpoint {path}", flush=True)
                            # Pass-mark early-exit is a final
                            # artefact too → push full optimiser
                            # so the HF copy is a perfect resume
                            # target (see the matching reasoning
                            # at the top of the loop).
                            if push_every > 0:
                                push_checkpoint(
                                    str(path), backend=push_backend,
                                    push_optimizer=True,
                                )
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
        final_path = checkpoint_path_for_step(
            ckpt_dir, _run_dir_name(), final_step)
        harness.save_checkpoint(str(final_path), step=final_step)
        print(f"[train_dsl] saved final checkpoint {final_path}", flush=True)
        # ALWAYS push the final checkpoint if push_every is on — the
        # _deploy_train.py end-of-training push exists as belt-and-
        # braces but the box may self-destroy first (H24). Force
        # push_optimizer=True so this canonical resume target is
        # always complete; the bandwidth saving from the strip is
        # for the noisy mid-run cadence, not the one ckpt at the
        # end you actually want to ship.
        if push_every > 0:
            push_checkpoint(
                str(final_path), backend=push_backend,
                push_optimizer=True,
            )

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
                      streaming=True)
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
    parser.add_argument("--arch",
                        help="path to architecture folder (containing arch.neuro); "
                             "required if --dna not provided")
    parser.add_argument("--dna",
                        help="path to evolved DNA file (e.g., dna/evol/arch.dna); "
                             "alternative to --arch for training from DNA")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_sem", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--push_every", type=int, default=0,
                        help="If > 0, push each periodically-saved "
                             "checkpoint via the configured backend "
                             "(see --push_backend) so the vast.ai box "
                             "self-destruction can no longer eat them "
                             "(H24 loss-hole). Should normally equal "
                             "--save_every or be a small multiple thereof.")
    parser.add_argument("--push_backend",
                        choices=["hf", "lfs", "none"], default="hf",
                        help="Checkpoint push backend:\n"
                             "  hf   = HuggingFace Hub upload_file "
                             "(default; post-2026-06-15)\n"
                             "  lfs  = legacy git add/commit/push to "
                             "Git LFS (run 41063959 hung at step 500 "
                             "because of git-push/log-pusher race)\n"
                             "  none = no remote push (local-dev runs)\n"
                             "Overridden by CHECKPOINT_PUSH_BACKEND env "
                             "on the box.")
    parser.add_argument("--push_optimizer", action="store_true",
                        default=False,
                        help="HF backend only. When set, the FULL "
                             "ckpt payload (weights + Adam m + Adam "
                             "v) is uploaded. Default: strip "
                             "optimizer state, ~2/3 smaller upload "
                             "(weights + m + v ≈ 1.3 GB for the "
                             "107M trunk → ~430 MB stripped). The "
                             "on-disk .pt always has the full state "
                             "so same-box crash resume is unaffected. "
                             "Trade-off: a fresh-box resume from the "
                             "stripped HF copy shows a ~500-step "
                             "LR-warmup-shape loss blip while Adam's "
                             "2nd-moment EMA rebuilds from zero.")
    parser.add_argument("--ood_every", type=int, default=0,
                        help="If > 0, run a mid-training WikiText-103 OOD "
                             "ppl snapshot every N steps. Writes JSON to "
                             "logs/vast/benchmarks/ood/ for analyze-log "
                             "to pick up alongside the final eval.")
    parser.add_argument("--explore_every", type=int, default=0,
                        help="If > 0, every N steps run a read-only discovery probe "
                             "on the trunk's final hidden state (searches a residual "
                             "modulation that lowers next-token CE, persists winners "
                             "to modulations/). Never touches training — pure "
                             "measurement. 0 = off.")
    parser.add_argument("--explore_pop", type=int, default=24,
                        help="probe search population per checkpoint (deeper = more candidates)")
    parser.add_argument("--explore_gens", type=int, default=10,
                        help="probe search generations per checkpoint")
    parser.add_argument("--explore_len", type=int, default=8,
                        help="probe candidate program length (more expressive modulations)")
    parser.add_argument("--explore_sites", type=int, default=2,
                        help="how many layers (ranked by measured headroom under "
                             "the true loss) the multi-site probe searches per "
                             "checkpoint; insensitive/converged layers are skipped")
    parser.add_argument("--ckpt_dir", default="lfs_checkpoints")
    parser.add_argument("--sink", default="motor",
                        help="population whose output feeds the LM head")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="0 → take from tokenizer")
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16"],
                        help="mixed-precision dtype (cuda only)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest dsl_arch_step*.pt in ckpt_dir")
    parser.add_argument(
        "--heatmap", action=argparse.BooleanOptionalAction, default=True,
        help="record a per-arch/preset run heatmap (heatmaps/<arch>/<preset>.json). "
             "Default ON; pass --no-heatmap to disable.")
    parser.add_argument(
        "--heatmap-every", type=int, default=500,
        help="record + push the run heatmap every N steps (default 500).")
    parser.add_argument(
        "--resume_from", default=None, metavar="PATH_OR_URI",
        help="Resume from a SPECIFIC checkpoint. Accepts a local path "
             "or an hf://owner/repo/path URI; HF URIs are downloaded "
             "into --ckpt_dir before the load. Wins over --resume's "
             "globber. Set by ``brian deploy --resume`` /"
             " ``brian deploy --latest`` via the ``RESUME_FROM`` env "
             "var.")
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

    # Validate that either --arch or --dna is provided
    if not args.arch and not args.dna:
        parser.error("either --arch or --dna must be provided")
    if args.arch and args.dna:
        parser.error("--arch and --dna are mutually exclusive")

    torch.manual_seed(args.seed)

    # ── DNA mode: resolve architecture from DNA file ──
    if args.dna:
        from neuroslm.compiler.ribosome import RibosomeCompiler
        dna_path = Path(args.dna).resolve()
        if not dna_path.is_file():
            parser.error(f"missing DNA file: {dna_path}")
        try:
            compiler = RibosomeCompiler()
            dsl_code = compiler.dna_translator.translate_from_file(str(dna_path))
            arch_match = re.search(r"\barchitecture\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
                                  dsl_code)
            if not arch_match:
                parser.error(
                    f"cannot extract `architecture <name>` block from {dna_path}")
            arch_name = arch_match.group(1)
            arch_root = Path("architectures") / arch_name
            if not (arch_root / "arch.neuro").is_file():
                parser.error(
                    f"DNA references architecture {arch_name!r} but "
                    f"{arch_root}/arch.neuro not found")
            print(f"[train_dsl] DNA mode: {dna_path} → {arch_name}", flush=True)
        except Exception as e:
            parser.error(f"failed to load DNA {dna_path}: {e}")
    else:
        # ── DSL mode: use provided architecture folder ──
        arch_root = Path(args.arch).resolve()
        if not (arch_root / "arch.neuro").is_file():
            parser.error(f"missing {arch_root}/arch.neuro")

    # Boot stamp: forensic record of WHEN, WHICH git commit, and WHICH
    # DSL produced this run. Printed before any other train_dsl output
    # so it survives even if the rest of the boot path crashes. See
    # _print_boot_stamp() docstring and CLAUDE.md §10.x for the
    # rationale (deploy 40923107 retrospective).
    _print_boot_stamp(arch_root=arch_root)

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
            _cli_seq, _cli_batch = args.seq_len, args.batch
            args.seq_len = int(os.environ.get("SEQ_LEN",  _v.seq_len))
            args.batch   = int(os.environ.get("BATCH_SIZE", _v.batch_size))
            args.d_sem   = d_model
            _ovr = _scale_override_note(_scale_env, _cli_seq, _cli_batch,
                                        args.seq_len, args.batch)
            if _ovr:
                print(_ovr)
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
    #
    # PRECEDENCE FIX (Jun 2026): when arch.neuro has an explicit `training {}`
    # block, that block is the source of truth and the preset's lr/wd/warmup
    # are NOT applied (preset still contributes trunk dims). Without this
    # guard, lr=5e-4 / warmup=2400 in arch.neuro were silently clobbered
    # back to the preset's lr=3e-4 / warmup=300 on every run.
    _arch_src_for_training = (arch_root / "arch.neuro").read_text(encoding="utf-8")
    _arch_has_training_block = (
        "training {" in _arch_src_for_training
        or "training{" in _arch_src_for_training
    )
    if preset_sched is not None and not _arch_has_training_block:
        harness.training_config.learning_rate = preset_sched["lr"]
        harness.training_config.weight_decay = preset_sched["weight_decay"]
        warmup = preset_sched["warmup_steps"]
        min_ratio = preset_sched["min_lr_ratio"]
        print(f"[train_dsl] LR schedule from preset {args.preset}: "
              f"lr={preset_sched['lr']} warmup={warmup} min_ratio={min_ratio}")
    elif preset_sched is not None and _arch_has_training_block:
        # arch.neuro wins; harness.training_config already carries its
        # lr/wd/warmup_steps/min_lr_ratio from the training{} block.
        warmup = harness.training_config.warmup_steps
        min_ratio = harness.training_config.min_lr_ratio
        print(f"[train_dsl] LR schedule from arch.neuro (preset "
              f"{args.preset} suppressed): "
              f"lr={harness.training_config.learning_rate} "
              f"wd={harness.training_config.weight_decay} "
              f"warmup={warmup} min_ratio={min_ratio}")
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
    # ── --resume_from PATH_OR_URI : take precedence over --resume ──
    # The trainer accepts a SPECIFIC checkpoint URI (local or hf://)
    # passed via --resume_from or the RESUME_FROM env var. HF URIs are
    # downloaded into ``ckpt_dir`` before the load so the resumed run
    # naturally falls back to the same per-run subdir layout the
    # training-loop save path uses.
    resume_from = args.resume_from or os.environ.get("RESUME_FROM", "").strip()
    if resume_from:
        path_to_load: Optional[Path] = None
        if resume_from.startswith("hf://"):
            from neuroslm.hf_checkpoints import (
                parse_hf_uri, download_checkpoint,
            )
            try:
                repo_id, path_in_repo = parse_hf_uri(resume_from)
            except ValueError as e:
                print(f"[train_dsl] --resume_from: {e}", file=sys.stderr)
                return  # fail-fast: don't continue with a broken URI
            print(f"[train_dsl] --resume_from: pulling {resume_from}")
            local = download_checkpoint(
                path_in_repo, repo_id=repo_id,
                dest_dir=str(ckpt_dir) if ckpt_dir else None,
            )
            path_to_load = local
        else:
            path_to_load = Path(resume_from)
            if not path_to_load.is_file():
                print(f"[train_dsl] --resume_from: not a file: "
                      f"{path_to_load}", file=sys.stderr)
                path_to_load = None
        if path_to_load is not None:
            start_step = harness.load_checkpoint(str(path_to_load))
            harness._global_step = start_step
            print(f"[train_dsl] resumed from {path_to_load} "
                  f"@ step {start_step}")
    elif args.resume and ckpt_dir is not None:
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
        push_every=args.push_every,
        push_backend=args.push_backend,
        push_optimizer=args.push_optimizer,
        arch_name=args.arch,
        preset_name=args.preset,
        collect_heatmap=args.heatmap,
        run_heatmap_every=args.heatmap_every,
        explore_every=args.explore_every,
        explore_pop=args.explore_pop,
        explore_gens=args.explore_gens,
        explore_len=args.explore_len,
        explore_sites=args.explore_sites,
    )

    print("[train_dsl] done.")


if __name__ == "__main__":
    main()
