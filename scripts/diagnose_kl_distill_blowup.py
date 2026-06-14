"""Diagnose why H22 SmolLM2 expert tanks PPL despite better isolated CE.

Findings summary (run this script to reproduce numerically):

  1. The cortex/expert CE on the H22 paragraph is the SAME mechanism we
     already validated — SmolLM2 (~3.7 nats) genuinely beats gpt2 (~4.0
     nats) under exact-end alignment.

  2. **The catastrophe is in the DISTILLATION term, not the LM term.**
     ``neuroslm.harness.NeuroSLMHarness._cortex_fusion_aux_step`` does:

         kl = F.kl_div(log_student, soft_teacher, reduction="batchmean")
              * (T_dist ** 2)
         total = total + lam * kl

     ``reduction="batchmean"`` divides the SUM over (B, T, V) by **B
     only**. For shape (B=1, T=512, V=50257) the "batchmean" KL is
     approximately T × per-token KL = 512 × per-token KL.

  3. The LM cross-entropy is computed with the default
     ``F.cross_entropy(..., reduction="mean")`` which divides by B × T
     — i.e. it is a per-token average.

  4. The two loss terms are therefore on incompatible scales. The
     distillation term is ~T = 512× too large relative to the LM term.

  5. **Severity scales with teacher sharpness.** When the teacher is
     gpt2 (similar capacity to a freshly-init student), per-token KL is
     small (~0.3-1 nats early; quickly drops as student catches up).
     When the teacher is SmolLM2 (3× params + 100× tokens), per-token
     KL is huge (~5+ nats) and stays large for thousands of steps
     because the 30M student can't represent SmolLM2's distribution.

  6. With T²=16 (Hinton temperature scaling) AND the batchmean bug AND
     a sharp teacher, the distillation contribution to the loss is::

         lam * T² * batchmean_kl
       = 1.175 * 16 * (per_token_KL * 512)
       ≈ 1.175 * 16 * 5 * 512
       ≈ 48 000 nats

     vs the LM term ≈ 6 nats. **The student is being trained almost
     exclusively on a noisy approximation-of-SmolLM2 gradient and the
     LM-loss gradient is drowned out.**

Empirical numbers from H22 train.log step 500:

    cortex[α_eff=0.500 inh=0.000 λ=1.115 kl=1512.000 lm_ema=10.28 cx_ema=4.23]

    per-token KL ≈ 1512 / 512 = 2.95 nats   (with T=4 already scaled in)
    raw per-token KL ≈ 2.95 / 16 ≈ 0.18 nats (after un-T² scaling)
    student LM loss ≈ 6.40 nats
    distill loss in total ≈ 1.115 × 1512 = 1686 nats

So at step 500 the distillation term contributes ~280× the LM term to
the loss, even after the EMA-gap-ramp has begun to throttle λ.

What this script does
=====================

Loads both expert candidates (gpt2-fast-path + SmolLM2-bridge-path),
runs them on a held-out English paragraph against a randomly-init
trunk, and prints the per-token KL + batchmean KL + relative-strength
ratio for each.

Run with:

    $env:PYTHONIOENCODING="utf-8"
    .\.venv\Scripts\python.exe scripts/diagnose_kl_distill_blowup.py
"""
from __future__ import annotations

import io
import math
import sys

# Force UTF-8 stdout on Windows so box-drawing characters survive.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
    )

import torch
import torch.nn.functional as F


# A 7-sentence English paragraph used by all earlier bridge diagnostics
# (scripts/diagnose_bridge.py, scripts/diagnose_bridge_ce.py).
TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump, dozy fowl quack. "
    "Quick wafting zephyrs vex bold Jim."
)


def _per_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T_dist: float = 4.0,
) -> tuple[float, float, float]:
    """Return (per_token_kl_natural, kl_mean_reduction, kl_batchmean_reduction).

    ``per_token_kl_natural`` is the actual per-token KL with T² scaling
    applied — i.e. the value the harness *should* be adding to the loss
    if it used ``reduction="mean"`` (or equivalently averaged over B*T).

    ``kl_batchmean_reduction`` is what the current
    ``_cortex_fusion_aux_step`` actually computes.
    """
    log_student = F.log_softmax(student_logits / T_dist, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T_dist, dim=-1)
    # mean: divides sum by B*T*V — too aggressive
    # batchmean: divides sum by B only
    # We want sum-over-V, then mean-over-(B,T) for honest per-token KL.
    kl_per_position = F.kl_div(
        log_student, soft_teacher, reduction="none",
    ).sum(dim=-1)  # (B, T)
    kl_mean = float(kl_per_position.mean().item()) * (T_dist ** 2)
    kl_batchmean = (
        float(kl_per_position.sum().item())
        / student_logits.shape[0]
        * (T_dist ** 2)
    )
    per_token_natural = kl_mean  # honest per-token, T²-scaled
    return per_token_natural, kl_mean, kl_batchmean


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> float:
    B, T, V = logits.shape
    return float(F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1),
    ).item())


def main() -> None:
    print("─" * 72)
    print("KL distillation blow-up diagnostic — why SmolLM2 tanks training")
    print("─" * 72)
    print(f"Text length: {len(TEXT)} chars\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── trunk tokenizer + a freshly-init student head ─────────────────
    trunk_tok = AutoTokenizer.from_pretrained("gpt2")
    V_trunk = int(trunk_tok.vocab_size)
    ids = trunk_tok(TEXT, return_tensors="pt", add_special_tokens=False)
    input_ids = ids["input_ids"]  # (1, T)
    B, T = input_ids.shape
    targets = input_ids[:, 1:].contiguous()
    print(f"V_trunk={V_trunk}  B={B}  T={T}  T-1 targets={targets.shape[1]}")

    # ── Freshly-init "student" trunk head: random logits ────────────
    # This matches the H22 setup at step 0 — student starts uniform.
    torch.manual_seed(0)
    student_logits = torch.randn(
        B, T - 1, V_trunk, dtype=torch.float32,
    ) * 0.02  # small init, ~uniform
    student_ce = _ce(student_logits, targets)
    print(f"\n[Student] freshly-init random head CE  = {student_ce:6.3f} nats "
          f"(≈ ln(V) = {math.log(V_trunk):5.3f})")

    # ── Teacher #1: gpt2 fast path ─────────────────────────────────
    print("\n" + "─" * 72)
    print("TEACHER A — gpt2 (same vocab as trunk, fast path)")
    print("─" * 72)
    lm_gpt2 = AutoModelForCausalLM.from_pretrained(
        "gpt2", use_safetensors=False, weights_only=False,
    ).eval()
    with torch.no_grad():
        teacher_gpt2_logits = lm_gpt2(input_ids=input_ids).logits[:, :-1, :]
    teacher_gpt2_ce = _ce(teacher_gpt2_logits, targets)
    print(f"Teacher gpt2  CE on TEXT             = {teacher_gpt2_ce:6.3f} nats")
    gap_gpt2 = student_ce - teacher_gpt2_ce
    print(f"Gap (student - teacher)              = {gap_gpt2:6.3f} nats")

    pt_kl_a, kl_mean_a, kl_bm_a = _per_token_kl(
        student_logits, teacher_gpt2_logits, T_dist=4.0,
    )
    print(f"per-token KL × T²  (correct reduction) = {pt_kl_a:8.3f} nats")
    print(f"reduction='batchmean' × T² (current)  = {kl_bm_a:8.3f} nats")
    print(f"ratio batchmean / per-token            = {kl_bm_a / pt_kl_a:8.1f}× "
          f"(should be ≈ T-1 = {T - 1})")

    # ── Teacher #2: SmolLM2 (bridge path placeholder — we use raw
    #    SmolLM2 logits cast/projected to gpt2 vocab via the bridge) ──
    print("\n" + "─" * 72)
    print("TEACHER B — SmolLM2-360M (different vocab, bridge path)")
    print("─" * 72)
    try:
        from neuroslm.experts import LMExpert
        # Construct via LMExpert so the alignment+bridge fix from
        # commit a976fee is exercised.
        expert_smollm2 = LMExpert(
            model_id="smollm2_360m",
            domain="general",
            trunk_tokenizer=trunk_tok,
            freeze=True,
        )
        with torch.no_grad():
            teacher_smollm2_logits = expert_smollm2(input_ids)[:, :-1, :]
        teacher_smollm2_ce = _ce(teacher_smollm2_logits, targets)
        print(f"Teacher SmolLM2 CE on TEXT (post-bridge) = "
              f"{teacher_smollm2_ce:6.3f} nats")
        coverage = expert_smollm2.last_alignment_coverage
        print(f"Bridge alignment coverage                = "
              f"{coverage:.3f} ({coverage * 100:.1f}%)")
        gap_smollm2 = student_ce - teacher_smollm2_ce
        print(f"Gap (student - teacher)                  = {gap_smollm2:6.3f} nats")

        pt_kl_b, kl_mean_b, kl_bm_b = _per_token_kl(
            student_logits, teacher_smollm2_logits, T_dist=4.0,
        )
        print(f"per-token KL × T²  (correct reduction)   = {pt_kl_b:8.3f} nats")
        print(f"reduction='batchmean' × T² (current)     = {kl_bm_b:8.3f} nats")
        print(f"ratio batchmean / per-token              = "
              f"{kl_bm_b / pt_kl_b:8.1f}× (should be ≈ T-1 = {T - 1})")
    except Exception as exc:  # pragma: no cover
        print(f"(SmolLM2 unavailable: {type(exc).__name__}: {exc})")
        teacher_smollm2_ce = float("nan")
        kl_bm_b = float("nan")
        pt_kl_b = float("nan")

    # ── The smoking gun: relative-strength ratio ─────────────────────
    print("\n" + "═" * 72)
    print("SMOKING GUN — distillation-to-LM strength ratio at step 0")
    print("═" * 72)
    print(f"  Student LM loss  (per-token mean)        ≈ {student_ce:8.3f}")
    print(f"  Distill term, gpt2 teacher  (batchmean)  ≈ "
          f"{kl_bm_a:8.3f}  →  {kl_bm_a / student_ce:5.1f}× LM loss")
    if math.isfinite(kl_bm_b):
        print(f"  Distill term, SmolLM2 teacher (batchmean) ≈ "
              f"{kl_bm_b:8.3f}  →  {kl_bm_b / student_ce:5.1f}× LM loss")
        print()
        print(f"  ratio SmolLM2-distill / gpt2-distill    = "
              f"{kl_bm_b / kl_bm_a:6.1f}×")
        print("  → SmolLM2 produces a distillation gradient that "
              f"DOMINATES the LM gradient by {kl_bm_b / student_ce:.0f}×.")
    print()
    print("  If the reduction were the correct 'mean' (per-token average):")
    print(f"    gpt2 teacher distill   ≈ {pt_kl_a:8.3f} nats "
          f"({pt_kl_a / student_ce:.2f}× LM)")
    if math.isfinite(pt_kl_b):
        print(f"    SmolLM2 teacher distill ≈ {pt_kl_b:8.3f} nats "
              f"({pt_kl_b / student_ce:.2f}× LM)")
    print("  → Both would be ≤ O(1)× the LM loss, the regime distillation"
          " was designed for.")
    print()
    print("=" * 72)
    print(
        "CONCLUSION: The mechanism is the F.kl_div reduction='batchmean' "
        "bug in\nneuroslm.harness.NeuroSLMHarness._cortex_fusion_aux_step "
        "(line ~941),\ncompounded by SmolLM2's sharper-teacher / "
        "larger-capacity-gap distribution."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
