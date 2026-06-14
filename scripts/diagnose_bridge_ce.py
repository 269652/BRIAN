"""Measure ACTUAL CE impact of M1 fix (multi-token bridge) on real English.

Compares:
  current bridge (strict 1-token)   : ~73% of vocab gets abstain mass
  proposed bridge (>=1 token, first) : ~100% of vocab uses expert's first-subtoken logit

We compute next-token CE on a held-out paragraph using:
  - the trunk's own model (gpt2)  -> baseline
  - SmolLM2 via CURRENT bridge
  - SmolLM2 via PROPOSED bridge
  - SmolLM2 via PROPOSED bridge + EXACT alignment

This isolates exactly how much CE we recover with each fix.
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# Test paragraph — natural English, no fancy chars
TEXT = (
    "Mathematics is the language in which the universe is written. "
    "Theorems are proved by logical deduction from a small set of axioms. "
    "Numbers, equations, and geometric forms describe physical reality "
    "with extraordinary precision. From the orbits of planets to the "
    "behaviour of subatomic particles, the patterns of nature reveal a "
    "deep mathematical structure. The discovery of these patterns is "
    "one of the great triumphs of human intellect."
)


print("== loading models ==")
t0 = time.time()
trunk_tok = AutoTokenizer.from_pretrained("gpt2")
expert_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
trunk_lm = AutoModelForCausalLM.from_pretrained("gpt2", use_safetensors=False, weights_only=False).eval()
expert_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M", use_safetensors=True).eval()
print(f"loaded in {time.time()-t0:.1f}s")

V_t = trunk_tok.vocab_size
V_e = expert_tok.vocab_size

# ── Build trunk-side ids + targets ────────────────────────────────
trunk_enc = trunk_tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
trunk_ids = torch.tensor([trunk_enc["input_ids"]], dtype=torch.long)
trunk_offsets = trunk_enc["offset_mapping"]
T = trunk_ids.shape[1]

# Targets for next-token prediction: shift by 1
targets = trunk_ids[0, 1:]   # (T-1,)
print(f"\ntext: {T} trunk tokens, {len(expert_tok.encode(TEXT, add_special_tokens=False))} expert tokens")

# ── (A) Baseline: gpt2 trunk's own next-token CE ──────────────────
with torch.no_grad():
    trunk_logits = trunk_lm(input_ids=trunk_ids).logits.squeeze(0)  # (T, V_t)
# CE on positions 0..T-2 predicting tokens 1..T-1
ce_trunk = F.cross_entropy(trunk_logits[:-1].float(), targets).item()
print(f"\n== (A) BASELINE: gpt2 trunk own next-token CE = {ce_trunk:.3f} nats ==")

# ── (B) Build bridges (both modes) ────────────────────────────────
def build_bridge(strict_single_token: bool) -> torch.Tensor:
    """Return trunk_to_expert LongTensor[V_t]."""
    bridge = torch.full((V_t,), -1, dtype=torch.long)
    for tid in range(V_t):
        try:
            s = trunk_tok.decode([tid])
        except Exception:
            continue
        if not s:
            continue
        try:
            eids = expert_tok.encode(s, add_special_tokens=False)
        except Exception:
            continue
        if strict_single_token:
            if len(eids) == 1 and 0 <= eids[0] < V_e:
                bridge[tid] = eids[0]
        else:
            if len(eids) >= 1 and 0 <= eids[0] < V_e:
                bridge[tid] = eids[0]
    return bridge


print("\n== building bridges ==")
t0 = time.time()
bridge_strict = build_bridge(True)
bridge_relaxed = build_bridge(False)
print(f"  strict  coverage: {(bridge_strict >= 0).float().mean():.3%}")
print(f"  relaxed coverage: {(bridge_relaxed >= 0).float().mean():.3%}")
print(f"  build time: {time.time()-t0:.1f}s")


# ── (C) Run expert on its own tokenization, then bridge ──────────
expert_enc = expert_tok(TEXT, add_special_tokens=False, return_offsets_mapping=True)
expert_ids = torch.tensor([expert_enc["input_ids"]], dtype=torch.long)
expert_offsets = expert_enc["offset_mapping"]
E = expert_ids.shape[1]
with torch.no_grad():
    expert_logits_full = expert_lm(input_ids=expert_ids).logits.squeeze(0)  # (E, V_e)


def align_smallest_ge(t_off, e_off):
    """Current alignment: smallest e with e_end >= t_end."""
    out = []
    e_idx = 0
    n_e = len(e_off)
    for _, t_end in t_off:
        while e_idx < n_e - 1 and e_off[e_idx][1] < t_end:
            e_idx += 1
        out.append(e_idx)
    return out


def align_exact(t_off, e_off):
    """Proposed exact alignment: e_idx s.t. e_end == t_end, else -1."""
    out = []
    e_idx = 0
    n_e = len(e_off)
    for _, t_end in t_off:
        while e_idx < n_e and e_off[e_idx][1] < t_end:
            e_idx += 1
        if e_idx < n_e and e_off[e_idx][1] == t_end:
            out.append(e_idx)
        else:
            out.append(-1)
    return out


def apply_bridge(expert_logits_e, bridge_table):
    """Project (T, V_e) -> (T, V_t) using per-position relative abstain."""
    idx = bridge_table.to(expert_logits_e.device)
    idx_safe = idx.clamp(min=0)
    gathered = expert_logits_e.index_select(-1, idx_safe)  # (T, V_t)
    mask = (idx == -1)
    if not mask.any():
        return gathered
    neg_inf = torch.full_like(gathered, float("-inf"))
    mapped_only = torch.where(mask, neg_inf, gathered)
    max_mapped = mapped_only.amax(dim=-1, keepdim=True)
    degenerate = ~torch.isfinite(max_mapped)
    max_mapped = torch.where(degenerate, torch.zeros_like(max_mapped), max_mapped)
    ln_v = math.log(V_t)
    abstain = max_mapped - ln_v
    return torch.where(mask, abstain.expand_as(gathered), gathered)


def bridge_to_trunk(alignment_fn, bridge_table):
    """For each trunk position, pick aligned expert position, bridge to trunk vocab.
    Misaligned positions (idx == -1) get UNIFORM logits (zero everywhere)."""
    align = alignment_fn(trunk_offsets, expert_offsets)
    out = torch.zeros((T, V_t), dtype=torch.float32)
    for t, e in enumerate(align):
        if e < 0:
            continue  # uniform
        bridged_row = apply_bridge(expert_logits_full[e:e+1], bridge_table)
        out[t] = bridged_row.squeeze(0)
    n_aligned = sum(1 for e in align if e >= 0)
    return out, n_aligned / T


print("\n== bridge experiments ==")

# (C1) Current bridge: strict 1-token + smallest-ge alignment
logits_c1, cov_c1 = bridge_to_trunk(align_smallest_ge, bridge_strict)
ce_c1 = F.cross_entropy(logits_c1[:-1], targets).item()
print(f"(C1) CURRENT  (strict + smallest_ge):   CE={ce_c1:.3f} nats   align_coverage={cov_c1:.0%}")

# (C2) Just M1 fix: relaxed bridge + smallest-ge alignment
logits_c2, cov_c2 = bridge_to_trunk(align_smallest_ge, bridge_relaxed)
ce_c2 = F.cross_entropy(logits_c2[:-1], targets).item()
print(f"(C2) +M1  (relaxed + smallest_ge):      CE={ce_c2:.3f} nats   align_coverage={cov_c2:.0%}")

# (C3) Just M2 fix: strict bridge + exact alignment
logits_c3, cov_c3 = bridge_to_trunk(align_exact, bridge_strict)
ce_c3 = F.cross_entropy(logits_c3[:-1], targets).item()
print(f"(C3) +M2  (strict + exact):             CE={ce_c3:.3f} nats   align_coverage={cov_c3:.0%}")

# (C4) Both fixes: relaxed bridge + exact alignment
logits_c4, cov_c4 = bridge_to_trunk(align_exact, bridge_relaxed)
ce_c4 = F.cross_entropy(logits_c4[:-1], targets).item()
print(f"(C4) +M1+M2 (relaxed + exact):          CE={ce_c4:.3f} nats   align_coverage={cov_c4:.0%}")

uniform_ce = math.log(V_t)
print(f"\nUniform baseline: {uniform_ce:.3f} nats")
print(f"Trunk (gpt2) own:  {ce_trunk:.3f} nats")

print(f"\n== DELTAS ==")
print(f"M1 alone:   {ce_c2 - ce_c1:+.3f} nats vs current")
print(f"M2 alone:   {ce_c3 - ce_c1:+.3f} nats vs current")
print(f"M1+M2:      {ce_c4 - ce_c1:+.3f} nats vs current")
print(f"gpt2 gap:   M1+M2 vs gpt2 = {ce_c4 - ce_trunk:+.3f} nats "
      f"(positive means smollm2 is still worse)")
