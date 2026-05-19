"""Benchmark evaluation: HellaSwag, ARC-Easy, ARC-Challenge, MMLU.

All use likelihood-based scoring (no generation needed), compatible with
models at any stage of training. Run standalone:

    python -m neuroslm.benchmarks --ckpt checkpoints/latest.pt --preset xl

Or from Python:

    from neuroslm.benchmarks import eval_all
    from .xla_utils import get_device
    results = eval_all(brain, tok, device=str(get_device()), max_samples=500)
"""
from __future__ import annotations
import argparse
import json
import os
import time
import torch
import torch.nn.functional as F


def _tokenize_and_score(brain, tok, prefix: str, continuation: str,
                        device: str = "cuda") -> float:
    """Average log-likelihood of continuation tokens given prefix."""
    prefix_ids = tok.encode(prefix)
    cont_ids = tok.encode(continuation)
    if not cont_ids:
        return -100.0
    full_ids = prefix_ids + cont_ids
    max_len = brain.cfg.lang_ctx
    if len(full_ids) > max_len:
        # Truncate prefix from the left, keep full continuation
        full_ids = full_ids[-max_len:]
        prefix_len = max(0, len(full_ids) - len(cont_ids))
    else:
        prefix_len = len(prefix_ids)

    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = brain.forward_lm(ids_t[:, :-1], ids_t[:, 1:])
        # out is a dict with "logits" or the first return is logits-based
        if isinstance(out, dict):
            logits = out.get("logits")
            if logits is None:
                # Fallback: call the language cortex directly.  The return
                # signature evolved (3 → 4 → optionally 5 values once
                # return_tap was added for SRC-TEH), so unpack defensively.
                lang_out = brain.language(ids_t[:, :-1])
                logits = lang_out[0] if isinstance(lang_out, tuple) else lang_out
        else:
            logits = out[0] if isinstance(out, tuple) else out

    # Score only the continuation tokens
    # logits: (1, seq_len-1, vocab_size) after forward on ids[:,:-1]
    # targets for continuation start at position prefix_len
    if prefix_len > 0:
        cont_logits = logits[:, prefix_len - 1:, :]
        cont_targets = ids_t[:, prefix_len:]
    else:
        cont_logits = logits
        cont_targets = ids_t[:, 1:]

    n_tokens = min(cont_logits.size(1), cont_targets.size(1))
    if n_tokens == 0:
        return -100.0
    cont_logits = cont_logits[:, :n_tokens, :]
    cont_targets = cont_targets[:, :n_tokens]

    log_probs = F.log_softmax(cont_logits, dim=-1)
    token_scores = log_probs.gather(2, cont_targets.unsqueeze(-1)).squeeze(-1)
    return float(token_scores.mean().item())


# ── HellaSwag ────────────────────────────────────────────────────────

def eval_hellaswag(brain, tok, device="cpu", max_samples=1000) -> dict:
    """HellaSwag: commonsense sentence completion (4-way)."""
    from datasets import load_dataset
    print("\n[bench] HellaSwag ...", flush=True)
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True,
                      trust_remote_code=True)
    correct = total = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        ctx = row.get("ctx", "") or row.get("ctx_a", "")
        endings = row["endings"]
        label = int(row["label"])
        scores = [_tokenize_and_score(brain, tok, ctx, e, device)
                  for e in endings]
        if max(range(len(scores)), key=lambda j: scores[j]) == label:
            correct += 1
        total += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{max_samples}] acc={correct/total:.4f}", flush=True)
    acc = correct / total if total else 0
    dt = time.time() - t0
    print(f"[bench] HellaSwag: {correct}/{total} = {acc:.4f} ({dt:.0f}s)")
    return {"name": "hellaswag", "correct": correct, "total": total,
            "accuracy": acc, "time_s": dt}


# ── ARC ──────────────────────────────────────────────────────────────

def eval_arc(brain, tok, device="cpu", challenge=False,
             max_samples=500) -> dict:
    """ARC-Easy or ARC-Challenge: science reasoning (3-5 way)."""
    from datasets import load_dataset
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    label = "ARC-Challenge" if challenge else "ARC-Easy"
    print(f"\n[bench] {label} ...", flush=True)
    ds = load_dataset("allenai/ai2_arc", config, split="test",
                      streaming=True, trust_remote_code=True)
    correct = total = 0
    t0 = time.time()
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer_key = row["answerKey"]
        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            continue
        prefix = f"Question: {question}\nAnswer:"
        scores = [_tokenize_and_score(brain, tok, prefix, f" {c}", device)
                  for c in choices]
        if max(range(len(scores)), key=lambda j: scores[j]) == correct_idx:
            correct += 1
        total += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{max_samples}] acc={correct/total:.4f}", flush=True)
    acc = correct / total if total else 0
    dt = time.time() - t0
    print(f"[bench] {label}: {correct}/{total} = {acc:.4f} ({dt:.0f}s)")
    return {"name": label.lower().replace("-", "_"), "correct": correct,
            "total": total, "accuracy": acc, "time_s": dt}


# ── MMLU ─────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_physics",
    "computer_security", "conceptual_physics", "high_school_biology",
    "high_school_chemistry", "logical_fallacies", "machine_learning",
    "high_school_mathematics", "world_religions", "moral_scenarios",
    "professional_law", "clinical_knowledge",
]

def eval_mmlu(brain, tok, device="cpu", max_per_subject=30) -> dict:
    """MMLU: multi-subject multiple-choice knowledge (4-way)."""
    from datasets import load_dataset
    print(f"\n[bench] MMLU ({len(MMLU_SUBJECTS)} subjects) ...", flush=True)
    letters = ["A", "B", "C", "D"]
    all_correct = all_total = 0
    subject_acc = {}
    t0 = time.time()
    for subj in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test",
                              streaming=True, trust_remote_code=True)
        except Exception:
            continue
        sc = st = 0
        for i, row in enumerate(ds):
            if i >= max_per_subject:
                break
            q = row["question"]
            choices = row["choices"]
            answer = int(row["answer"])
            prefix = f"Question: {q}\n"
            for j, c in enumerate(choices):
                prefix += f"{letters[j]}. {c}\n"
            prefix += "Answer:"
            scores = [_tokenize_and_score(brain, tok, prefix,
                                          f" {letters[j]}", device)
                      for j in range(len(choices))]
            if max(range(len(scores)), key=lambda j: scores[j]) == answer:
                sc += 1
            st += 1
        if st:
            subject_acc[subj] = sc / st
            all_correct += sc
            all_total += st
    acc = all_correct / all_total if all_total else 0
    dt = time.time() - t0
    print(f"[bench] MMLU: {all_correct}/{all_total} = {acc:.4f} ({dt:.0f}s)")
    for s, a in sorted(subject_acc.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s:30s} {a:.3f}")
    return {"name": "mmlu", "correct": all_correct, "total": all_total,
            "accuracy": acc, "subjects": subject_acc, "time_s": dt}


# ── Combined ─────────────────────────────────────────────────────────

def eval_all(brain, tok, device="cpu", max_samples=500) -> dict:
    """Run all benchmarks and print comparison table."""
    brain.eval()
    results = {}
    results["hellaswag"] = eval_hellaswag(brain, tok, device, max_samples)
    results["arc_easy"] = eval_arc(brain, tok, device, challenge=False,
                                   max_samples=max_samples)
    results["arc_challenge"] = eval_arc(brain, tok, device, challenge=True,
                                        max_samples=max_samples)
    results["mmlu"] = eval_mmlu(brain, tok, device, max_per_subject=30)

    avg = sum(r["accuracy"] for r in results.values()) / len(results)
    results["average"] = avg

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"  {name:20s}: {r['accuracy']:.4f} "
                  f"({r['correct']}/{r['total']})")
    print(f"  {'AVERAGE':20s}: {avg:.4f}")
    print("=" * 70)
    print("\n  Reference scores (for comparison):")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Model              HellaSwag  ARC-E  ARC-C  MMLU   Params")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Random baseline       0.250  0.250  0.250  0.250    -")
    print("  SmolLM2-135M          0.420  0.480  0.280  0.260  135M")
    print("  TinyLlama-1.1B        0.590  0.550  0.310  0.250  1.1B")
    print("  Phi-1.5-1.3B          0.630  0.730  0.440  0.380  1.3B")
    print("  Phi-3-mini-3.8B       0.780  0.850  0.640  0.690  3.8B")
    print("  Qwen2.5-3B            0.730  0.780  0.530  0.650  3.0B")
    print("  Qwen2.5-7B            0.800  0.830  0.580  0.720  7.0B")
    print("  Llama-3.1-8B          0.820  0.850  0.600  0.680  8.0B")
    print("  ─────────────────────────────────────────────────────────────────")
    return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run NeuroSLM benchmarks")
    ap.add_argument("--ckpt", required=True,
                    help="Path to model checkpoint (.pt)")
    ap.add_argument("--preset", default=None,
                    choices=["tiny", "small", "medium", "large", "xl", "xxl"],
                    help="Preset override.  By default the cfg stored INSIDE "
                         "the checkpoint is used (correct for all post-2026-05 "
                         "checkpoints).  Only override when loading a legacy "
                         "checkpoint whose stored cfg is missing or stale.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--output", default=None,
                    help="Path to save JSON results (default: <ckpt>_bench.json)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any state-dict shape mismatch.  Default off "
                         "so SRC-TEH/legacy cross-loads warn instead of crash.")
    args = ap.parse_args()

    from .config import PRESETS, BrainConfig
    from .tokenizer import Tokenizer
    from .brain import Brain

    tok = Tokenizer()

    # ── Load checkpoint FIRST so we can read its stored cfg ──
    # Old code constructed Brain from `args.preset` defaults, then tried to
    # load_state_dict — which crashed whenever the checkpoint was trained at
    # a different (preset, d_hidden, lang_layers, ...).  Now we reconstruct
    # Brain from the cfg that was saved alongside the weights.
    print(f"[bench] loading checkpoint {args.ckpt} ...", flush=True)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    # Prefer the saved cfg ALWAYS when present.  `--preset` is a fallback
    # only used if the checkpoint doesn't carry a `cfg` dict (which only
    # happens for very old saves predating the cfg-round-trip fix).
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = BrainConfig()
        valid_fields = set(cfg.__dict__.keys())
        for k, v in ckpt["cfg"].items():
            if k in valid_fields:
                setattr(cfg, k, v)
        saved_preset = ckpt.get("preset", "<saved-cfg>")
        if args.preset is not None and args.preset != saved_preset:
            print(f"[bench] note: --preset={args.preset} ignored "
                  f"(checkpoint carries cfg from preset={saved_preset}; "
                  f"the saved cfg is authoritative)", flush=True)
        print(f"[bench] using cfg saved in checkpoint "
              f"(preset={saved_preset}, d_sem={cfg.d_sem}, "
              f"d_hidden={cfg.d_hidden}, lang_layers={cfg.lang_layers}, "
              f"src_teh={getattr(cfg, 'enable_src_teh', False)})", flush=True)
    else:
        preset = args.preset or "xl"
        cfg = PRESETS[preset]()
        print(f"[bench] no saved cfg in checkpoint — using --preset={preset} "
              f"defaults (this is the legacy path; the next save will "
              f"include a cfg so future loads round-trip)", flush=True)

    cfg.vocab_size = tok.vocab_size
    brain = Brain(cfg).to(args.device)

    missing, unexpected = brain.load_state_dict(ckpt["model"], strict=args.strict)
    if missing or unexpected:
        # Report instead of crash — common when bench-loading a checkpoint
        # whose Brain assembled a different optional-module set than today's.
        print(f"[bench] non-strict load: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys", flush=True)
        if len(missing) <= 5:
            for k in missing:
                print(f"        missing: {k}", flush=True)
        if len(unexpected) <= 5:
            for k in unexpected:
                print(f"        unexpected: {k}", flush=True)
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"[bench] loaded {args.ckpt} (step {step}, {n_params/1e6:.1f}M params)",
          flush=True)

    results = eval_all(brain, tok, args.device, args.max_samples)

    out_path = args.output or args.ckpt.replace(".pt", "_bench.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
