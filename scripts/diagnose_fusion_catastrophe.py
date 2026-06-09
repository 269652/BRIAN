"""Numerical confirmation of the catastrophic-loss root cause.

Hypothesis: at fusion_init=0.5, the cortex path injects ~50% of magnitude
into final_logits in a direction uncorrelated with the LM trunk's path.
Since lm.embed is a fresh-init random matrix (training-from-scratch),
both lm_logits and cortex_logits are random Gaussians of similar std,
but with no alignment to each other. The user observed CE=13.84 ≫ ln(V)=10.82,
which is ~3 nats *worse than uniform*. This script isolates which
configuration reproduces that.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    V = 50257     # GPT-2 vocab
    B, T = 4, 64  # tiny batch
    D = 512       # d_sem

    # ── ACTUAL DSL LM init (neuroslm/dsl/nn_lang.py) ──────────────────
    # self.embed   = nn.Parameter(_alloc("normal", (vocab, d_model)))
    #   → torch.randn × 0.02   → std = 0.02
    # self.lm_head = nn.Parameter(_alloc("xavier", (vocab, d_model)))
    #   → xavier_uniform_      → std ≈ sqrt(2/(V+D)) ≈ 0.00627
    embed = torch.randn(V, D) * 0.02
    lm_head = torch.empty(V, D)
    torch.nn.init.xavier_uniform_(lm_head)

    # LM trunk hidden — post RMSNorm + per-block adapters. Init: ones
    # gamma → unit RMS. So h std ≈ 1 at init.
    lm_h = torch.randn(B, T, D)

    # Cortex hidden — GPT-2 ln_f-normed (std ≈ 1) → random Linear
    # projection (768→512, kaiming-uniform, gain=sqrt(5)).
    gpt2_h = torch.randn(B, T, 768)
    proj = torch.empty(D, 768)
    torch.nn.init.kaiming_uniform_(proj, a=5 ** 0.5)
    cortex_h = gpt2_h @ proj.T

    # ── Logits both ways ─────────────────────────────────────────────
    # LM trunk: uses its OWN xavier-init lm_head (small std).
    lm_logits = lm_h @ lm_head.T
    # Cortex: uses the TIED cortex_lm_head.weight = embed (larger std).
    cortex_logits = cortex_h @ embed.T

    print(f"embed         std={embed.std().item():.4f}   "
          f"lm_head std={lm_head.std().item():.4f}")
    print(f"lm_h          std={lm_h.std().item():.3f}    "
          f"cortex_h std={cortex_h.std().item():.3f}")
    print()
    print(f"lm_logits     std={lm_logits.std().item():.3f}  "
          f"max={lm_logits.max().item():+.3f}  "
          f"|max|={lm_logits.abs().max().item():.3f}")
    print(f"cortex_logits std={cortex_logits.std().item():.3f}  "
          f"max={cortex_logits.max().item():+.3f}  "
          f"|max|={cortex_logits.abs().max().item():.3f}")
    ratio = cortex_logits.std().item() / lm_logits.std().item()
    print(f"\n>>> cortex_logits std is {ratio:.2f}× lm_logits std")
    print(">>> Cortex DOMINATES the mixture at any α > 0.06 or so.")

    targets = torch.randint(0, V, (B, T))
    ln_V = math.log(V)
    print(f"\nln(vocab)={ln_V:.3f}  (CE of perfectly-uniform prediction)\n")

    print(f"{'alpha':<8}{'std':<10}{'|max|':<10}{'CE':<10}{'ΔvsUniform':<14}")
    print("-" * 52)
    for alpha in [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        final = (1.0 - alpha) * lm_logits + alpha * cortex_logits
        ce = F.cross_entropy(final.reshape(-1, V), targets.reshape(-1)).item()
        print(f"{alpha:<8.2f}{final.std().item():<10.3f}"
              f"{final.abs().max().item():<10.3f}"
              f"{ce:<10.4f}{ce - ln_V:+12.3f}")

    # ── What if cortex_lm_head were UNTIED + ZERO-INIT? (proposed fix) ──
    print("\n── PROPOSED FIX: untied cortex_lm_head with zero init ──")
    cortex_head_zero = torch.zeros(V, D)
    cortex_logits_zero = cortex_h @ cortex_head_zero.T
    print(f"  cortex_logits std={cortex_logits_zero.std().item():.4f}  "
          f"(zero by construction)")
    for alpha in [0.05, 0.5]:
        final = (1.0 - alpha) * lm_logits + alpha * cortex_logits_zero
        ce = F.cross_entropy(final.reshape(-1, V), targets.reshape(-1)).item()
        print(f"  α={alpha:.2f}  std={final.std().item():.3f}  "
              f"CE={ce:.4f}  Δ={ce - ln_V:+.3f}  "
              "(matches LM-only baseline)")


if __name__ == "__main__":
    main()
