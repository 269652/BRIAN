# -*- coding: utf-8 -*-
"""Expert-cortex probe — discovery on the frozen pretrained LM cortices.

Why the experts are the strongest discovery target (H53 follow-up):

- **Real slack, well-defined.** The frozen experts (SmolLM2-360M, CodeGPT,
  Qwen2.5 — PPL≈50 territory) were optimized for THEIR pretraining
  distribution, not BRIAN's data mixture. H46's null-site argument ("the
  terminal hidden is fully end-to-end optimized") does not apply to a frozen
  model on shifted data — the headroom scan measures whether it holds.
- **Durable winners.** Frozen weights never move, so a banked winner stays
  valid for the whole project lifetime (no per-checkpoint staleness — unlike
  the trunk probe, which re-searches every ``explore_every`` steps).
- **Feeds the whole arch.** The experts teach the trunk via KL distillation;
  a modulation that lowers an expert's CE on BRIAN's data sharpens the
  teacher signal every cortex consumes.

Scoring is the expert's OWN next-token CE in its OWN token space — the vocab
bridge is a distillation concern, not a discovery one. v1 probes the final
hidden through the expert's own head; multi-site inside HF architectures is
the follow-up (needs per-architecture tail plumbing).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F


# ── expert loading (light: no VocabBridge — probing needs no trunk) ────────
@dataclass
class ProbedExpert:
    model_id: str
    lm: object                 # full HF model (kept for cleanup/device moves)
    backbone: object           # (input_ids=…) -> .last_hidden_state
    lm_head: object            # hidden -> expert-vocab logits
    tokenizer: object          # expert's own tokenizer
    max_ctx: int

    @classmethod
    def load(cls, model_id: str, device: str = "cpu") -> "ProbedExpert":
        from neuroslm.experts import (
            _load_lm_cached, _load_tokenizer_cached, _split_lm,
            resolve_expert_alias,
        )
        mid = resolve_expert_alias(str(model_id))
        lm = _load_lm_cached(mid)
        lm.eval()
        for p in lm.parameters():
            p.requires_grad = False
        backbone, head = _split_lm(lm)
        if backbone is None or head is None:
            raise RuntimeError(
                f"cannot split backbone/head for {mid!r} — expert probe "
                f"needs both (unknown architecture)")
        tok = _load_tokenizer_cached(mid)
        if tok is None:
            raise RuntimeError(f"no tokenizer for {mid!r}")
        cfg = getattr(lm, "config", None)
        n = (getattr(cfg, "n_positions", None)
             or getattr(cfg, "max_position_embeddings", None) or 2048)
        dev = torch.device(device)
        lm.to(dev)
        return cls(model_id=mid, lm=lm, backbone=backbone, lm_head=head,
                   tokenizer=tok, max_ctx=int(n))


# ── data ────────────────────────────────────────────────────────────────────
def texts_from_stream(n: int, *, min_chars: int = 200, mode: str = "text"
                      ) -> List[str]:
    """``n`` raw text snippets from the training data chain; offline fallback
    is the bundled local corpus (paragraph-split, cycled)."""
    import neuroslm.data as data_mod
    try:
        ds, fmt, _label = data_mod.open_stream(mode)
        texts: List[str] = []
        for ex in ds:
            t = fmt(ex)
            if t and len(t) >= min_chars:
                texts.append(t)
                if len(texts) >= n:
                    return texts
        if texts:
            return (texts * ((n // len(texts)) + 1))[:n]
    except Exception:
        pass
    raw = data_mod._bundled_corpus_path().read_text(encoding="utf-8",
                                                    errors="replace")
    paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) >= 80]
    if not paras:
        paras = [raw]
    return (paras * ((n // len(paras)) + 1))[:n]


def make_texts_provider(*, min_chars: int = 200, mode: str = "text"
                        ) -> Callable[[int], List[str]]:
    """Stateful text provider: each call draws the NEXT ``n`` texts.

    Recurrence evidence requires fresh text per round — re-opening the stream
    each round returns the same first-N texts, making every "independent"
    probe measure the identical batch. This provider holds one iterator across
    rounds (HF stream when reachable; bundled-corpus paragraphs cycled as the
    offline fallback) so consecutive rounds see different data.
    """
    import itertools
    import neuroslm.data as data_mod

    def _gen():
        try:
            ds, fmt, _label = data_mod.open_stream(mode)
            for ex in ds:
                t = fmt(ex)
                if t and len(t) >= min_chars:
                    yield t
        except Exception:
            pass
        raw = data_mod._bundled_corpus_path().read_text(encoding="utf-8",
                                                        errors="replace")
        paras = [p.strip() for p in raw.split("\n\n")
                 if len(p.strip()) >= min(min_chars, 80)]
        if not paras:
            paras = [raw]
        # cycle with a round marker so repeated passes still differ
        for i in itertools.count():
            yield f"{paras[i % len(paras)]}\n[pass {i // len(paras)}]"

    it = _gen()

    def provider(n: int) -> List[str]:
        return [next(it) for _ in range(n)]

    return provider


def expert_batch(expert: ProbedExpert, texts: Sequence[str], *,
                 batch: int, seq_len: int, device: str = "cpu"):
    """(ids, targets) next-token windows in the EXPERT's own token space.

    Joins the texts, tokenizes once with the expert's tokenizer, chops into
    ``batch`` windows of ``seq_len+1`` (tiling when the corpus is short), and
    caps ``seq_len`` at the expert's hard context limit.
    """
    seq_len = min(int(seq_len), expert.max_ctx)
    joined = "\n\n".join(texts)
    ids_list = expert.tokenizer(joined)["input_ids"]
    need = batch * (seq_len + 1)
    while len(ids_list) < need:            # tile a short corpus
        ids_list = ids_list + ids_list
    flat = torch.tensor(ids_list[:need], dtype=torch.long)
    win = flat.reshape(batch, seq_len + 1).to(device)
    return win[:, :-1], win[:, 1:]


def _fp32_head(head):
    """A fp32 view of the expert's LM head, without mutating the (process-wide
    cached, bf16) module. Experts load in bf16 — a CE computed through a bf16
    head is quantized to ~1/32-nat steps, coarser than the Δs the probe hunts.
    For the common ``nn.Linear`` head, run the matmul in fp32 on a detached
    weight copy (freed when the probe's closure dies); otherwise fall back to
    the module in its own dtype with fp32-cast logits."""
    if isinstance(head, torch.nn.Linear):
        W = head.weight.detach().float()
        b = head.bias.detach().float() if head.bias is not None else None
        return lambda h: F.linear(h, W, b)
    try:
        dt = next(head.parameters()).dtype
    except StopIteration:
        dt = torch.float32
    return lambda h: head(h.to(dt)).float()


# ── the probe ───────────────────────────────────────────────────────────────
def probe_expert(expert: ProbedExpert, ids: torch.Tensor,
                 targets: torch.Tensor, *, store=None, config=None,
                 round_idx: int = 0, progress=None) -> dict:
    """Search an NGL modulation of the expert's final hidden, scored by the
    expert's own next-token CE. Read-only (no_grad throughout; weights are
    already frozen). Winners persist site-tagged ``expert_<alias>_step<r>``."""
    from neuroslm.genetic.layer_probe import headroom_scan
    from neuroslm.genetic.ledger import SearchLedger
    from neuroslm.genetic.training_explorer import probe_hidden_modulation

    def _say(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    alias = expert.model_id.split("/")[-1].replace("-", "_").replace(".", "_").lower()

    with torch.no_grad():
        hidden = expert.backbone(input_ids=ids).last_hidden_state.detach().float()

    head_fn = _fp32_head(expert.lm_head)

    with torch.no_grad():
        baseline_ce = float(F.cross_entropy(
            head_fn(hidden).reshape(-1, head_fn(hidden).shape[-1]),
            targets.reshape(-1)))

    # Headroom at the (single, v1) site — measures whether domain shift left
    # trivial slack in this frozen model on THIS data.
    reports = headroom_scan([hidden], lambda _k, h: head_fn(h), targets,
                            seed=round_idx, baseline_ce=baseline_ce)
    hr = reports[0]
    _say(f"[expert:{alias}] round {round_idx}: baseline_ce={baseline_ce:.4f} "
         f"({hr.line().replace('L0', 'final')})")

    res = probe_hidden_modulation(
        hidden, head_fn, targets,
        ledger=SearchLedger(":memory:"), store=store, config=config,
        step=round_idx, run_id=f"expert-{alias}")
    res["model_id"] = expert.model_id
    res["headroom"] = {"sensitivity": hr.sensitivity,
                       "improvement": hr.improvement,
                       "best_perturbation": hr.best_perturbation}
    return res


def run_expert_discovery(*, experts: Optional[Sequence[ProbedExpert]] = None,
                         models: Optional[Sequence[str]] = None,
                         rounds: int = 10, batch: int = 2, seq_len: int = 256,
                         pop: int = 24, gens: int = 10, length: int = 8,
                         device: str = "cpu",
                         texts_fn: Optional[Callable[[int], List[str]]] = None,
                         store_root=None, push: bool = False,
                         progress=None) -> List[dict]:
    """Multi-round expert discovery: fresh texts per round, every roster
    expert probed per round. Winners bank to ``<store_root>/modulations``
    (repo root by default) — recurrence across rounds is real evidence, since
    frozen weights make every round an independent measurement of the SAME
    model on different data."""
    from neuroslm.genetic.modulation_store import ModulationStore

    def _say(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    if experts is None:
        experts = []
        for mid in (models or []):
            _say(f"[expert-discovery] loading {mid} …")
            experts.append(ProbedExpert.load(mid, device=device))
    if not experts:
        raise ValueError("no experts to probe (pass experts= or models=)")

    root = Path(store_root) if store_root is not None else \
        Path(__file__).resolve().parent.parent.parent
    store = ModulationStore(root / "modulations")
    texts_fn = texts_fn or make_texts_provider()

    from neuroslm.genetic.training_explorer import ExploreConfig
    cfg = ExploreConfig(pop_size=pop, generations=gens, length=length,
                        normalize=False)

    results: List[dict] = []
    kept = 0
    for r in range(1, rounds + 1):
        texts = texts_fn(max(4, batch * 2))
        for ex in experts:
            ids, targets = expert_batch(ex, texts, batch=batch,
                                        seq_len=seq_len, device=device)
            out = probe_expert(ex, ids, targets, store=store, config=cfg,
                               round_idx=r, progress=progress)
            results.append(out)
            tag = f"saved {out['saved']}" if out.get("saved") else "no keep"
            _say(f"[expert-discovery] round {r}/{rounds} "
                 f"{out['model_id']}: best_ce={out['best_ce']:.4f} "
                 f"Δ={out['delta_ce']:.4f} evaluated={out['evaluated']} ({tag})")
            if out.get("saved"):
                kept += 1
        if push and any(o.get("saved") for o in results[-len(experts):]):
            try:
                from neuroslm.genetic.modulation_pusher import push_artifacts
                push_artifacts(root, ["modulations"],
                               message=f"explore: expert probe round {r}")
            except Exception as e:
                _say(f"[expert-discovery] push skipped: {e!r}")
    _say(f"[expert-discovery] done: {rounds} rounds × {len(experts)} experts, "
         f"{kept} winners banked → modulations/")
    return results
