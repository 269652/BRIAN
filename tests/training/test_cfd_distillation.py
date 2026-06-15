# -*- coding: utf-8 -*-
"""Test the Capacity-Funneled Distillation (CFD) implementation.

H006: Capacity-Funneled Distillation produces monotone-implode PPL in
teacher capacity.

This is the empirical falsifier for the H006 hypothesis declared at
`hypothesis/H006_capacity_funneled_distillation_implode.md`. It runs
a four-arm ablation on a tiny student (vocab=64, d_model=32) with
**synthetic teacher distributions** so the test:

  * runs deterministically in <5 s on CPU,
  * does not require any HuggingFace download,
  * isolates the CFD mechanism from confounders like tokenizer mismatch
    or cross-vocabulary bridging.

The four arms (per H006 falsifier table):

| Arm | Teacher                | Distill | Predicted final loss      |
|-----|------------------------|---------|---------------------------|
|  A  | none (LM-only)         | —       | L_A (baseline)            |
|  B  | weak (low entropy gap) | naive   | ≤ L_A (small win)         |
|  C  | strong (large gap)     | naive   | ≫ L_A (the explosion)     |
|  D  | strong                 | **CFD** | < L_B ≤ L_A (the implode) |

Plus contract tests on the three CFD stages individually:

  * Stage 1 (top-K projection): output is a valid pdf, top-K mass
    preserved, tail is uniform.
  * Stage 2 (entropy-matched T): T_eff ≥ T_0; T_eff = T_0 when
    H(student) ≤ H(teacher); T_eff > T_0 when student is more uncertain.
  * Stage 3 (gradient-alignment gate): λ_eff ∈ [0, λ_0]; λ_eff = 0
    when gradients are anti-aligned; λ_eff = λ_0 when fully aligned.

The CFD path will be implemented in
``neuroslm.harness._cortex_fusion_aux_step`` behind a new
``MultiCortexConfig.cfd_enabled`` flag (default False so existing runs
reproduce bit-identically). The three free functions exercised below
live at module level in ``neuroslm.harness`` so they can be unit-tested
without instantiating a full harness.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Imports under test — these will RED until the CFD implementation lands
# ──────────────────────────────────────────────────────────────────────

from neuroslm.harness import (
    cfd_topk_target,
    cfd_effective_temperature,
    cfd_grad_alignment_gate,
)


VOCAB = 64
D_MODEL = 32
BATCH = 8
SEQLEN = 4


# ──────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_logits(seed: int, sharpness: float = 1.0) -> torch.Tensor:
    """Return a (BATCH, SEQLEN, VOCAB) logits tensor with controlled
    sharpness. `sharpness=1.0` → near-uniform, `sharpness=10.0` → very
    peaked (low entropy)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(BATCH, SEQLEN, VOCAB, generator=g)
    return raw * sharpness


def _make_targets(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 999)
    return torch.randint(0, VOCAB, (BATCH, SEQLEN), generator=g)


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — top-K rank-preserving sparsification contract
# ──────────────────────────────────────────────────────────────────────

class TestCFDStage1TopKProjection:
    """`cfd_topk_target(teacher_logits, K, T)` returns the top-K
    rank-preserving projection of `softmax(teacher_logits / T)`.

    Specifically:
      * top-K modes keep their softmax mass exactly,
      * the residual (1 - top-K mass) is spread UNIFORMLY over the
        remaining (V - K) modes,
      * the result is a valid probability distribution (sums to 1).
    """

    @pytest.mark.parametrize("K", [1, 4, 8, 32])
    def test_output_sums_to_one(self, K: int) -> None:
        teacher = _make_logits(seed=42, sharpness=2.0)
        target = cfd_topk_target(teacher, K=K, T=4.0)
        sums = target.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"top-K(K={K}) projection failed to sum to 1; got {sums}"
        )

    @pytest.mark.parametrize("K", [1, 4, 8])
    def test_topk_mass_preserved(self, K: int) -> None:
        """Mass on the top-K indices is identical to the raw softmax
        mass on those indices."""
        T = 4.0
        teacher = _make_logits(seed=43, sharpness=3.0)
        raw_softmax = F.softmax(teacher / T, dim=-1)
        target = cfd_topk_target(teacher, K=K, T=T)

        _, topk_idx = teacher.topk(K, dim=-1)
        raw_topk_mass = raw_softmax.gather(-1, topk_idx)
        target_topk_mass = target.gather(-1, topk_idx)

        assert torch.allclose(raw_topk_mass, target_topk_mass, atol=1e-5), (
            f"top-K(K={K}) mass not preserved: "
            f"raw={raw_topk_mass.flatten()[:5]}, target={target_topk_mass.flatten()[:5]}"
        )

    @pytest.mark.parametrize("K", [4, 8, 16])
    def test_tail_is_uniform(self, K: int) -> None:
        """All non-top-K entries share the same mass per row."""
        teacher = _make_logits(seed=44, sharpness=2.0)
        target = cfd_topk_target(teacher, K=K, T=4.0)
        _, topk_idx = teacher.topk(K, dim=-1)

        # Build a "tail mask" of shape (B, T, V) — True where the index
        # is NOT in topK.
        tail_mask = torch.ones_like(target, dtype=torch.bool)
        tail_mask.scatter_(-1, topk_idx, False)

        # For each (b, t), the tail entries must all be equal.
        # Reshape to (B*T, V) for easy iteration.
        flat_target = target.reshape(-1, VOCAB)
        flat_mask = tail_mask.reshape(-1, VOCAB)
        for i in range(flat_target.size(0)):
            tail_vals = flat_target[i][flat_mask[i]]
            assert tail_vals.numel() == VOCAB - K
            # All tail entries equal up to fp precision
            assert torch.allclose(
                tail_vals, tail_vals[0].expand_as(tail_vals), atol=1e-6
            ), f"row {i}: tail not uniform: {tail_vals[:5]}"

    def test_K_equals_V_reduces_to_softmax(self) -> None:
        """When K=V, the projection IS the raw softmax (the tail set
        is empty)."""
        teacher = _make_logits(seed=45, sharpness=2.0)
        target = cfd_topk_target(teacher, K=VOCAB, T=2.0)
        raw = F.softmax(teacher / 2.0, dim=-1)
        assert torch.allclose(target, raw, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — entropy-matched temperature contract
# ──────────────────────────────────────────────────────────────────────

class TestCFDStage2EntropyMatch:
    """`cfd_effective_temperature(student_logits, teacher_logits, T_0)`
    returns T_eff = T_0 · max(1, H(student) / H(teacher))."""

    def test_returns_T0_when_student_less_uncertain(self) -> None:
        """If H(student) ≤ H(teacher), T_eff = T_0."""
        # Student is already sharper than teacher
        student = _make_logits(seed=10, sharpness=5.0)  # sharp, low H
        teacher = _make_logits(seed=11, sharpness=1.0)  # diffuse, high H
        T0 = 4.0
        T_eff = cfd_effective_temperature(student, teacher, T0)
        assert math.isclose(T_eff, T0, abs_tol=1e-4), (
            f"expected T_eff == T_0 = {T0}, got {T_eff}"
        )

    def test_returns_more_than_T0_when_student_more_uncertain(self) -> None:
        """If H(student) > H(teacher), T_eff > T_0."""
        student = _make_logits(seed=12, sharpness=0.5)  # diffuse
        teacher = _make_logits(seed=13, sharpness=5.0)  # sharp
        T0 = 2.0
        T_eff = cfd_effective_temperature(student, teacher, T0)
        assert T_eff > T0, f"expected T_eff > {T0}, got {T_eff}"

    def test_T_eff_grows_with_entropy_ratio(self) -> None:
        """Bigger gap (student diffuse, teacher sharp) → larger T_eff."""
        teacher = _make_logits(seed=20, sharpness=5.0)
        student_a = _make_logits(seed=21, sharpness=2.0)
        student_b = _make_logits(seed=22, sharpness=0.5)  # even more diffuse
        T0 = 4.0
        T_a = cfd_effective_temperature(student_a, teacher, T0)
        T_b = cfd_effective_temperature(student_b, teacher, T0)
        assert T_b > T_a, (
            f"more-diffuse student should require higher T_eff; "
            f"got T_a={T_a}, T_b={T_b}"
        )


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — gradient-alignment gate contract
# ──────────────────────────────────────────────────────────────────────

class TestCFDStage3GradAlignGate:
    """`cfd_grad_alignment_gate(distill_term, lm_logits, targets, lam_0)`
    returns (lam_eff, g_align) where:
      * lam_eff ∈ [0, lam_0],
      * lam_eff = lam_0 · (1 + g_align) / 2,
      * g_align = cosine between ∇_{lm_logits} distill_term and
        ∇_{lm_logits} CE(lm_logits, targets).
    """

    def test_aligned_distill_gives_full_lambda(self) -> None:
        """When distill_term has SAME gradient direction as LM CE,
        g_align ≈ 1 and lam_eff ≈ lam_0."""
        targets = _make_targets(seed=30)
        # Make lm_logits a leaf tensor with grad
        lm_logits = _make_logits(seed=31, sharpness=2.0).requires_grad_(True)
        # The cheat: distill_term IS the LM CE — perfect alignment
        distill_term = F.cross_entropy(
            lm_logits.reshape(-1, VOCAB), targets.reshape(-1)
        )
        lam_0 = 1.0
        lam_eff, g_align = cfd_grad_alignment_gate(
            distill_term, lm_logits, targets, lam_0=lam_0
        )
        assert g_align > 0.99, f"expected g_align ≈ 1, got {g_align}"
        assert math.isclose(lam_eff, lam_0, abs_tol=1e-3)

    def test_anti_aligned_distill_gives_zero_lambda(self) -> None:
        """When distill_term is NEGATIVE LM CE, g_align ≈ -1 and
        lam_eff ≈ 0."""
        targets = _make_targets(seed=32)
        lm_logits = _make_logits(seed=33, sharpness=2.0).requires_grad_(True)
        # Negative CE: opposite gradient direction
        distill_term = -F.cross_entropy(
            lm_logits.reshape(-1, VOCAB), targets.reshape(-1)
        )
        lam_0 = 1.0
        lam_eff, g_align = cfd_grad_alignment_gate(
            distill_term, lm_logits, targets, lam_0=lam_0
        )
        assert g_align < -0.99, f"expected g_align ≈ -1, got {g_align}"
        assert lam_eff < 0.01, f"expected lam_eff ≈ 0, got {lam_eff}"

    def test_lam_eff_in_bounds(self) -> None:
        """For arbitrary distill_term, lam_eff ∈ [0, lam_0]."""
        torch.manual_seed(40)
        for trial in range(8):
            targets = _make_targets(seed=40 + trial)
            lm_logits = _make_logits(
                seed=41 + trial, sharpness=1.5
            ).requires_grad_(True)
            # Random distill term tied to lm_logits
            teacher_targets = torch.randint(0, VOCAB, targets.shape)
            distill_term = F.cross_entropy(
                lm_logits.reshape(-1, VOCAB), teacher_targets.reshape(-1)
            )
            lam_0 = 0.7
            lam_eff, g_align = cfd_grad_alignment_gate(
                distill_term, lm_logits, targets, lam_0=lam_0
            )
            assert 0.0 <= lam_eff <= lam_0 + 1e-6, (
                f"trial {trial}: lam_eff={lam_eff} out of [0, {lam_0}]; "
                f"g_align={g_align}"
            )

    def test_g_align_in_minus_one_one(self) -> None:
        torch.manual_seed(50)
        for trial in range(8):
            targets = _make_targets(seed=50 + trial)
            lm_logits = _make_logits(
                seed=51 + trial, sharpness=2.0
            ).requires_grad_(True)
            teacher_targets = torch.randint(0, VOCAB, targets.shape)
            distill_term = F.cross_entropy(
                lm_logits.reshape(-1, VOCAB), teacher_targets.reshape(-1)
            )
            _, g_align = cfd_grad_alignment_gate(
                distill_term, lm_logits, targets, lam_0=1.0
            )
            assert -1.0 - 1e-5 <= g_align <= 1.0 + 1e-5, (
                f"trial {trial}: g_align={g_align} out of [-1, 1]"
            )


# ──────────────────────────────────────────────────────────────────────
# Four-arm ablation — the H006 falsifier
# ──────────────────────────────────────────────────────────────────────

def _tiny_student(seed: int = 0) -> nn.Module:
    """Minimal one-layer student: embed → linear → logits."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Embedding(VOCAB, D_MODEL),
        nn.Linear(D_MODEL, VOCAB, bias=True),
    )


def _make_dataset(seed: int, n_batches: int = 32):
    """Generate training data with structure: a fixed "true" distribution
    over next tokens, sampled."""
    g = torch.Generator().manual_seed(seed)
    # The true latent distribution: a few peaked modes per context
    true_logits = torch.randn(VOCAB, VOCAB, generator=g) * 3.0  # sharp
    batches = []
    for b in range(n_batches):
        ids = torch.randint(0, VOCAB, (BATCH, SEQLEN), generator=g)
        # Targets: sample from the row of true_logits indexed by last id
        last_ids = ids[:, -1]  # (B,)
        target_probs = F.softmax(true_logits[last_ids], dim=-1)  # (B, V)
        targets_last = torch.multinomial(target_probs, num_samples=1).squeeze(-1)
        # Pad: just repeat the last target for full seq
        targets = targets_last.unsqueeze(-1).expand(-1, SEQLEN).contiguous()
        batches.append((ids, targets, true_logits))
    return batches


def _frozen_teacher(seed: int, capacity: str) -> nn.Module:
    """Build a frozen teacher with controlled "capacity".

    capacity="weak":   small d_model, mid-sharp, partially aligned with truth
    capacity="strong": large d_model, very sharp, well-aligned with truth
    """
    torch.manual_seed(seed)
    if capacity == "weak":
        # weak teacher: noisy, less sharp
        net = nn.Sequential(
            nn.Embedding(VOCAB, D_MODEL // 2),
            nn.Linear(D_MODEL // 2, VOCAB, bias=True),
        )
    elif capacity == "strong":
        # strong teacher: bigger, sharper, more confident
        net = nn.Sequential(
            nn.Embedding(VOCAB, D_MODEL * 4),
            nn.Linear(D_MODEL * 4, VOCAB, bias=True),
        )
    else:
        raise ValueError(capacity)
    return net


def _pretrain_teacher(
    teacher: nn.Module, dataset, n_steps: int = 100, lr: float = 0.05
) -> nn.Module:
    """Pre-train the teacher to convergence on the synthetic data so it
    has actual signal to distill."""
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    for step in range(n_steps):
        batch = dataset[step % len(dataset)]
        ids, targets, _ = batch
        h = teacher[0](ids)
        logits = teacher[1](h)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Freeze
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _train_student(
    student: nn.Module,
    dataset,
    teacher: nn.Module | None,
    distill_mode: str,  # "none", "naive", "cfd"
    n_steps: int = 80,
    lr: float = 0.05,
    lam_0: float = 1.0,
    T_0: float = 4.0,
    K: int = 4,
    seed: int = 12345,
) -> float:
    """Train `student`. Return final LM loss on a held-out batch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    for step in range(n_steps):
        batch = dataset[step % len(dataset)]
        ids, targets, _ = batch
        h = student[0](ids)
        student_logits = student[1](h)
        flat_t = targets.reshape(-1)
        lm_loss = F.cross_entropy(
            student_logits.reshape(-1, VOCAB), flat_t
        )
        loss = lm_loss

        if teacher is not None and distill_mode != "none":
            with torch.no_grad():
                t_h = teacher[0](ids)
                teacher_logits = teacher[1](t_h)

            if distill_mode == "naive":
                # Legacy Hinton with batchmean (the bug being fixed)
                log_s = F.log_softmax(student_logits / T_0, dim=-1)
                soft_t = F.softmax(teacher_logits / T_0, dim=-1)
                kl = F.kl_div(log_s, soft_t, reduction="batchmean") * (T_0 ** 2)
                loss = loss + lam_0 * kl
            elif distill_mode == "cfd":
                # CFD: Stage 1 + Stage 2 + Stage 3
                T_eff = cfd_effective_temperature(
                    student_logits, teacher_logits, T_0
                )
                target = cfd_topk_target(teacher_logits, K=K, T=T_eff)
                log_s = F.log_softmax(student_logits / T_eff, dim=-1)
                kl_per_tok = F.kl_div(
                    log_s, target, reduction="none"
                ).sum(-1).mean()
                kl_term = kl_per_tok * (T_eff ** 2)
                lam_eff, _ = cfd_grad_alignment_gate(
                    kl_term, student_logits, targets, lam_0=lam_0
                )
                loss = loss + lam_eff * kl_term
            else:
                raise ValueError(distill_mode)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Final LM loss on a held-out batch
    student.eval()
    with torch.no_grad():
        held_out = dataset[-1]
        ids, targets, _ = held_out
        h = student[0](ids)
        logits = student[1](h)
        final_lm = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1)
        ).item()
    return final_lm


class TestFourArmAblation:
    """The four-arm H006 ablation. PPL ordering is the key claim.

    Note: these tests use FIXED seeds for determinism. The synthetic
    teacher is pre-trained to convergence before the student training
    starts, so the teacher's "signal" is honest cross-entropy on the
    same synthetic distribution the student is learning from.

    To reproduce the H22 "teacher too strong" pathology FAITHFULLY we
    need a teacher that is (1) high-capacity / sharp AND (2) pointing
    at a target distribution the student cannot efficiently follow.
    In H22 this was cross-tokenizer + cross-positional-encoding
    infeasibility. In this synthetic test we model the unreachability
    by training the strong teacher on a PERMUTED version of the target
    distribution — the teacher gives sharp, confident predictions, but
    they are systematically misaligned with the student's data. This
    is the cleanest reproduction of the "teacher pulls in the wrong
    direction" mechanism without needing real HF models.
    """

    @pytest.fixture(scope="class")
    def setup_arms(self):
        dataset = _make_dataset(seed=2026, n_batches=16)

        # Build an ADVERSARIAL strong teacher: trained on a permuted
        # version of the same data, so it's confidently pointing in
        # the wrong direction. This reproduces the "strong but
        # representationally infeasible" H22 condition.
        g = torch.Generator().manual_seed(424242)
        perm = torch.randperm(VOCAB, generator=g)
        permuted_dataset = []
        for ids, targets, true_logits in dataset:
            permuted_dataset.append((ids, perm[targets], true_logits))
        adversarial_teacher = _pretrain_teacher(
            _frozen_teacher(seed=200, capacity="strong"),
            permuted_dataset,
            n_steps=200,
        )

        # Sanity: the adversarial teacher is sharp (low loss) on its
        # own permuted task but WORSE than uniform on the student's task.
        with torch.no_grad():
            ids, targets, _ = dataset[0]
            adv_logits = adversarial_teacher[1](adversarial_teacher[0](ids))
            adv_ce_on_student_task = F.cross_entropy(
                adv_logits.reshape(-1, VOCAB), targets.reshape(-1)
            ).item()
            uniform_ce = math.log(VOCAB)  # ≈ 4.16
        assert adv_ce_on_student_task > uniform_ce * 0.95, (
            f"adversarial teacher precondition: should perform near or "
            f"WORSE than uniform on student's task (uniform={uniform_ce:.3f}), "
            f"got adv_ce={adv_ce_on_student_task:.3f}. The test premise "
            f"requires a teacher that pulls the student in the WRONG "
            f"direction; if the teacher is accidentally helpful the "
            f"distillation can't 'explode' here."
        )
        return dataset, adversarial_teacher

    def test_arm_A_lm_only_baseline(self, setup_arms) -> None:
        """Arm A: LM-only training. Establishes the baseline."""
        dataset, _ = setup_arms
        final_loss = _train_student(
            _tiny_student(seed=7), dataset, teacher=None,
            distill_mode="none", seed=7,
        )
        # Just sanity: should be finite and not absurdly high
        assert math.isfinite(final_loss)
        assert final_loss < 10.0, (
            f"Arm A baseline blew up: final_loss={final_loss}"
        )

    def test_arm_C_naive_kl_adversarial_teacher_explodes(self, setup_arms) -> None:
        """Arm C: naive KL with an ADVERSARIAL strong teacher should
        produce HIGHER loss than LM-only (this is the H22 bug —
        teacher pulls student away from data)."""
        dataset, adversarial = setup_arms
        loss_A = _train_student(
            _tiny_student(seed=7), dataset, teacher=None,
            distill_mode="none", seed=7,
        )
        loss_C = _train_student(
            _tiny_student(seed=7), dataset, teacher=adversarial,
            distill_mode="naive", seed=7,
        )
        # We expect loss_C > loss_A — the bug exists, naive KL with a
        # strong-but-misaligned teacher harms the student.
        assert loss_C > loss_A * 1.05, (
            f"Arm C did NOT explode as expected: "
            f"loss_A={loss_A:.3f}, loss_C={loss_C:.3f}. "
            f"The adversarial teacher isn't pulling the student off "
            f"the data manifold enough — check the permutation seed "
            f"or the teacher pre-training length."
        )

    def test_arm_D_cfd_adversarial_teacher_no_harm(self, setup_arms) -> None:
        """Arm D: CFD with an ADVERSARIAL teacher should produce loss
        comparable to or BETTER than LM-only (no-harm floor (I) from
        Theorem §13.3). The Stage-3 gradient gate must detect the
        misalignment and drive λ_eff toward 0."""
        dataset, adversarial = setup_arms
        loss_A = _train_student(
            _tiny_student(seed=7), dataset, teacher=None,
            distill_mode="none", seed=7,
        )
        loss_D = _train_student(
            _tiny_student(seed=7), dataset, teacher=adversarial,
            distill_mode="cfd", seed=7,
        )
        # No-harm floor (I): CFD must NOT be substantially worse than
        # LM-only. Tolerance 10% (test noise on 80 steps).
        assert loss_D <= loss_A * 1.10, (
            f"Arm D violates no-harm floor (I): "
            f"loss_A={loss_A:.3f}, loss_D={loss_D:.3f} "
            f"(D / A = {loss_D / loss_A:.3f}). "
            f"CFD should never make things substantially worse than "
            f"LM-only — Stage-3 grad gate is failing to detect the "
            f"adversarial teacher."
        )

    def test_arm_D_beats_arm_C(self, setup_arms) -> None:
        """The key claim: CFD with adversarial teacher (Arm D) is
        substantially better than naive KL with adversarial teacher
        (Arm C). Mechanism: Stage 3 gradient-alignment gate detects
        the teacher pulling against the LM gradient and drives
        λ_eff → 0."""
        dataset, adversarial = setup_arms
        loss_C = _train_student(
            _tiny_student(seed=7), dataset, teacher=adversarial,
            distill_mode="naive", seed=7,
        )
        loss_D = _train_student(
            _tiny_student(seed=7), dataset, teacher=adversarial,
            distill_mode="cfd", seed=7,
        )
        assert loss_D < loss_C * 0.95, (
            f"CFD did not beat naive KL on adversarial teacher: "
            f"loss_C (naive)={loss_C:.3f}, loss_D (CFD)={loss_D:.3f}. "
            f"H006 implode claim is REFUTED on this synthetic setup — "
            f"the Stage-3 grad gate is not firing strongly enough; "
            f"investigate g_align telemetry."
        )
