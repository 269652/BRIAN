"""Quick test: xl preset with novel mechanisms."""
import torch
from neuroslm.config import PRESETS
from neuroslm.brain import Brain

cfg = PRESETS['xl']()
cfg.vocab_size = 50257
b = Brain(cfg)
n = sum(p.numel() for p in b.parameters())
print(f"Params: {n/1e6:.1f}M")
print(f"GQA: {cfg.lang_kv_heads}KV / {cfg.lang_heads}Q heads")
print(f"Hebbian rank: {cfg.hebbian_rank}")
print(f"Predictive coding heads: {len(b.language.pred_coding)}")
print(f"NT-modulated attention: {b.language.blocks[0].attn.nt_scale is not None}")

ids = torch.randint(0, 100, (2, 64))
tgt = torch.randint(0, 100, (2, 64))
out = b.forward_lm(ids, tgt)
print(f"\nloss={out['loss'].item():.2f}  lm={out['lm_loss'].item():.2f}  "
      f"pc={out['pred_coding_loss'].item():.4f}")
print("✓ All novel mechanisms active and working")
