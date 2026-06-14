"""Forensic test: pinpoint what kills cross-tok bridge for SmolLM2 vs gpt2 trunk.

Three suspect mechanisms:
  M1. Vocab coverage (single-token-only mapping)
  M2. Alignment SHIFT (expert end-offset > trunk end-offset => prediction
      horizon mismatch + leakage)
  M3. CE on natural English with current vs proposed bridge
"""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoTokenizer

print("== loading tokenizers ==")
t0 = time.time()
trunk = AutoTokenizer.from_pretrained("gpt2")
expert = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
print(f"loaded in {time.time()-t0:.1f}s")

V_t = trunk.vocab_size
V_e = expert.vocab_size
print(f"V_trunk (gpt2)    = {V_t}")
print(f"V_expert (smollm2)= {V_e}")

# ── M1: VOCAB COVERAGE ────────────────────────────────────────────
print("\n== M1: vocab coverage (this takes ~1 min for 50k vocab) ==")
n_strict, n_relaxed = 0, 0
hist = {}
t0 = time.time()
for tid in range(V_t):
    if tid % 10000 == 0:
        print(f"  tid={tid}/{V_t} ({tid/V_t*100:.0f}%)", flush=True)
    try:
        s = trunk.decode([tid])
    except Exception:
        continue
    if not s:
        continue
    try:
        eids = expert.encode(s, add_special_tokens=False)
    except Exception:
        continue
    n = len(eids)
    if n == 0:
        continue
    if n == 1 and 0 <= eids[0] < V_e:
        n_strict += 1
    if n >= 1 and 0 <= eids[0] < V_e:
        n_relaxed += 1
    bucket = n if n <= 5 else 6  # 6 means ">=6"
    hist[bucket] = hist.get(bucket, 0) + 1

print(f"\n  STRICT  (n==1): {n_strict/V_t:.3%} ({n_strict})")
print(f"  RELAXED (n>=1): {n_relaxed/V_t:.3%} ({n_relaxed})")
print(f"  Histogram of decomposition length: {hist}")
print(f"  COVERAGE GAIN: +{(n_relaxed - n_strict)/V_t:.1%}")
print(f"  scan took {time.time()-t0:.1f}s")

# ── M2: ALIGNMENT SHIFT on a real English paragraph ───────────────
print("\n== M2: alignment shift ==")
text = (
    "Mathematics is the language in which the universe is written. "
    "Theorems are proved by logical deduction from a small set of axioms. "
    "Numbers, equations, and geometric forms describe physical reality "
    "with extraordinary precision. From the orbits of planets to the "
    "behaviour of subatomic particles, the patterns of nature reveal a "
    "deep mathematical structure."
)
trunk_enc = trunk(text, add_special_tokens=False, return_offsets_mapping=True)
expert_enc = expert(text, add_special_tokens=False, return_offsets_mapping=True)
t_off = trunk_enc["offset_mapping"]
e_off = expert_enc["offset_mapping"]
T = len(t_off)
E = len(e_off)
print(f"  trunk tokens: {T}, expert tokens: {E}, ratio E/T = {E/T:.2f}")

# For every trunk position t, find:
#   (a) the smallest e with expert_offsets[e][1] >= trunk_offsets[t][1]  (current alignment)
#   (b) whether there's an EXACT match expert_offsets[e][1] == trunk_offsets[t][1]
n_shift = 0
shift_chars_total = 0
n_exact = 0
e_idx_cur = 0
e_idx_strict = 0
for t in range(T):
    t_end = t_off[t][1]
    # Current alignment: smallest e with e_end >= t_end
    while e_idx_cur < E - 1 and e_off[e_idx_cur][1] < t_end:
        e_idx_cur += 1
    cur_e_end = e_off[e_idx_cur][1]
    if cur_e_end > t_end:
        n_shift += 1
        shift_chars_total += (cur_e_end - t_end)
    # Exact alignment: any e with e_end == t_end
    while e_idx_strict < E and e_off[e_idx_strict][1] < t_end:
        e_idx_strict += 1
    if e_idx_strict < E and e_off[e_idx_strict][1] == t_end:
        n_exact += 1

print(f"  positions with SHIFT (current alignment past trunk end): "
      f"{n_shift}/{T} = {n_shift/T:.1%}")
print(f"  mean shift (chars): {shift_chars_total / max(1,n_shift):.2f}")
print(f"  positions with EXACT alignment:                        "
      f"{n_exact}/{T} = {n_exact/T:.1%}")
print(f"  positions LOST under exact-only:                       "
      f"{(T - n_exact)/T:.1%}")

print("\n== summary ==")
print(f"  M1: relaxing single-token rule lifts coverage from "
      f"{n_strict/V_t:.0%} to {n_relaxed/V_t:.0%}")
print(f"  M2: current alignment shifts past trunk boundary at "
      f"{n_shift/T:.0%} of positions (leakage + wrong-horizon prediction)")
print(f"  M2: exact alignment would keep signal at {n_exact/T:.0%} of "
      f"positions, abstain at {(T-n_exact)/T:.0%}")
