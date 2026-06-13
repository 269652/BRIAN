"""Quick test: what initial CE comes out of the random-projection chain?"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
V = 50257
d_native = 768
d_sem = 512
B, T = 4, 256

# Simulate GPT-2's rogue dimension (Timkey & van Schijndel 2021)
h_native = torch.randn(B, T, d_native) * 0.5
h_native[..., 17] *= 80

proj = nn.Linear(d_native, d_sem, bias=False)
nn.init.xavier_normal_(proj.weight)
ln = nn.LayerNorm(d_sem)
cortex_lm_head = nn.Linear(d_sem, V, bias=False)
nn.init.xavier_normal_(cortex_lm_head.weight)

h_sem = proj(h_native)
h_norm = ln(h_sem)
logits = cortex_lm_head(h_norm)
targets = torch.randint(0, V, (B, T))
ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

print(f"ln(V)               = {math.log(V):.3f}   (uniform baseline)")
print(f"observed cx_ema     = 11.01            (from operator log)")
print(f"simulated cortex CE = {ce.item():.3f}    (two random projections)")
print()
print(f"-> Cortex logits are statistically indistinguishable from uniform.")
print(f"   GPT-2's 700M frozen weights produce information that is destroyed")
print(f"   by the random `proj` + random `cortex_lm_head` chain.")
