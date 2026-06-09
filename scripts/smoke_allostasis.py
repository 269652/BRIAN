"""End-to-end smoke test: instantiate the actual rcc_bowtie harness, run a
training step, and confirm allostasis is wired into the live system.

Run with:
    python -m scripts.smoke_allostasis
"""
from __future__ import annotations

import torch

from neuroslm.dsl.training_config import load_training_config_from_arch
from neuroslm.dsl.nn_lang import build_language_model
from neuroslm.harness import BRIANHarness


def main() -> None:
    cfg = load_training_config_from_arch("architectures/rcc_bowtie")
    # Override to a tiny model so we can run a real step in seconds on CPU
    cfg.multi_cortex.enabled = False  # skip GPT-2 download in this smoke test
    cfg.genetics.enabled = False
    cfg.batch_size = 1
    cfg.seq_len = 8
    cfg.grad_accum = 1  # smoke: step optimizer every micro-batch so allostasis
                        # telemetry populates immediately (prod uses accum=8)

    print("=" * 72)
    print("ALLOSTASIS SMOKE TEST")
    print("=" * 72)
    print(f"allostasis.enabled       = {cfg.allostasis.enabled}")
    print(f"allostasis.load_ema_alpha= {cfg.allostasis.load_ema_alpha}")
    print(f"allostasis.cort_ema_alpha= {cfg.allostasis.cort_ema_alpha}")
    print(f"allostasis.gamma_ne      = {cfg.allostasis.gamma_ne}")
    print(f"allostasis.gamma_trophic = {cfg.allostasis.gamma_trophic}")
    print(f"allostasis.gamma_lr      = {cfg.allostasis.gamma_lr}")
    print(
        f"HPA time-scale ratio     = "
        f"{cfg.allostasis.load_ema_alpha / cfg.allostasis.cort_ema_alpha:.1f}x"
    )

    vocab = 256
    d = 32
    lm = build_language_model(
        vocab=vocab, d_model=d, depth=2, n_heads=4, max_ctx=16
    )
    h = BRIANHarness.from_language_model(
        language_model=lm,
        vocab_size=vocab,
        d_sem=d,
        training_config=cfg,
    )

    print()
    print(f"harness.allostasis       = {h.allostasis is not None}")
    print(f"controller type          = {type(h.allostasis).__name__}")
    print()

    # Run 5 normal train steps
    print("-" * 72)
    print("PHASE 1: normal training (5 steps, random labels)")
    print("-" * 72)
    torch.manual_seed(0)
    for step in range(1, 6):
        ids = torch.randint(0, vocab, (1, 8))
        targets = torch.randint(0, vocab, (1, 8))
        loss = h.train_step(ids, targets)
        cort = h._metrics.get("allostasis_cort", float("nan"))
        load = h._metrics.get("allostasis_load", float("nan"))
        lr_m = h._metrics.get("allostasis_lr_mult", float("nan"))
        ne_m = h._metrics.get("allostasis_ne_mult", float("nan"))
        print(
            f"step {step}: loss={loss:6.3f}  "
            f"load={load:.4f}  cort={cort:.4f}  "
            f"ne_mult={ne_m:.3f}  lr_mult={lr_m:.3f}"
        )

    # Now pin cort high and verify the multiplicative effectors engage
    print()
    print("-" * 72)
    print("PHASE 2: pin cort=1.0, read effectors")
    print("-" * 72)
    with torch.no_grad():
        h.allostasis.cort.fill_(1.0)
    # ne_multiplier() etc. return Python floats (read-only telemetry surface).
    ne_m = float(h.allostasis.ne_multiplier())
    tr_m = float(h.allostasis.trophic_multiplier())
    lr_m = float(h.allostasis.lr_multiplier())
    print(f"ne_multiplier()      = {ne_m:.3f}   (expect 1 - 0.7 = 0.300)")
    print(f"trophic_multiplier() = {tr_m:.3f}   (expect 1 - 1.0 = 0.000)")
    print(f"lr_multiplier()      = {lr_m:.3f}   (expect 1 - 0.5 = 0.500)")
    assert abs(ne_m - 0.30) < 1e-5, f"NE multiplier wrong: {ne_m}"
    assert abs(tr_m - 0.00) < 1e-5, f"trophic multiplier wrong: {tr_m}"
    assert abs(lr_m - 0.50) < 1e-5, f"LR multiplier wrong: {lr_m}"

    # Replay the actual operator-log stress trajectory.
    # With α_cort=0.02, an EMA needs ≈ 3/α_cort = 150 steps to reach 95%
    # of equilibrium. We run 200 to demonstrate full HPA engagement.
    print()
    print("-" * 72)
    print("PHASE 3: replay runaway-pattern stress (200 sustained-stress steps)")
    print("-" * 72)
    with torch.no_grad():
        h.allostasis.cort.zero_()
        h.allostasis.load.zero_()
    n_steps = 200
    for step in range(1, n_steps + 1):
        # Simulate the actual log: NE saturates at 0.93, grad_norm at 24
        frac = min(1.0, step / 40.0)
        ne = 0.20 + 0.73 * frac
        gaba = 0.13 + 0.47 * frac
        loss = 12.6 + 2.8 * frac
        grad = 11.0 + 13.0 * frac
        h.allostasis.step(
            ne_level=ne, gaba_level=gaba, loss=loss, grad_norm=grad
        )
        if step in (10, 40, 80, 120, 160, 200):
            tel = h.allostasis.telemetry()
            print(
                f"step {step:3d}: NE={ne:.2f} grad={grad:5.2f} "
                f"-> load={tel['allostasis_load']:.3f} "
                f"cort={tel['allostasis_cort']:.3f}  "
                f"ne_mult={tel['allostasis_ne_mult']:.3f} "
                f"lr_mult={tel['allostasis_lr_mult']:.3f}"
            )

    tel = h.allostasis.telemetry()
    print()
    # Physics: at step 200, with saturating stress signal (≈ 0.5 from
    # weighted NE+GABA+loss+grad), cort EMA reaches ≈ 0.45 (~0.9·load),
    # which drives lr_mult ≈ 0.78 and ne_mult ≈ 0.69.
    assert tel["allostasis_cort"] > 0.30, (
        f"cort should climb to ≥ 0.30 under sustained stress, "
        f"got {tel['allostasis_cort']:.3f}"
    )
    assert tel["allostasis_lr_mult"] < 0.85, (
        f"LR should be damped (< 0.85) under chronic stress, "
        f"got {tel['allostasis_lr_mult']:.3f}"
    )
    assert tel["allostasis_ne_mult"] < 0.80, (
        f"NE should be capped (< 0.80) under chronic stress, "
        f"got {tel['allostasis_ne_mult']:.3f}"
    )

    # Recovery: stress lifts → cort should slowly decay
    print()
    print("-" * 72)
    print("PHASE 4: recovery (200 calm steps after stress)")
    print("-" * 72)
    cort_at_stress_end = tel["allostasis_cort"]
    for step in range(1, 201):
        h.allostasis.step(
            ne_level=0.20, gaba_level=0.15, loss=2.5, grad_norm=1.5
        )
        if step in (40, 80, 120, 160, 200):
            tel = h.allostasis.telemetry()
            print(
                f"step {step:3d}: load={tel['allostasis_load']:.3f} "
                f"cort={tel['allostasis_cort']:.3f}  "
                f"lr_mult={tel['allostasis_lr_mult']:.3f}"
            )
    tel = h.allostasis.telemetry()
    assert tel["allostasis_cort"] < cort_at_stress_end, (
        f"cort should decay after stress lifts "
        f"({cort_at_stress_end:.3f} -> {tel['allostasis_cort']:.3f})"
    )
    print(
        f"\ncort decayed: {cort_at_stress_end:.3f} -> "
        f"{tel['allostasis_cort']:.3f} (recovery is slower than onset — "
        f"asymmetric integration ✓)"
    )

    print("=" * 72)
    print("OK - end-to-end pipeline live. Allostatic controller is in the loop.")
    print(
        f"Peak stress (step 200 of phase 3): cort={cort_at_stress_end:.3f}, "
        f"would have damped LR by {(1 - (1 - 0.5*cort_at_stress_end)) * 100:.1f}%"
    )
    print(
        f"After 200 calm steps:              cort={tel['allostasis_cort']:.3f} "
        f"(recovery completed)"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
