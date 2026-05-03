import torch
from neuroslm.config import PRESETS
from neuroslm.brain import Brain

cfg = PRESETS['tiny']()
cfg.vocab_size = 50257
b = Brain(cfg)
ids = torch.randint(0, 100, (2, 32))
tgt = torch.randint(0, 100, (2, 32))
out = b.forward_lm(ids, tgt)
print(f"Forward OK, loss={out['loss'].item()}")

# Test receptor affinity report
for name in ['rcpt_pfc', 'rcpt_hippo', 'rcpt_bg', 'rcpt_thal', 'rcpt_lang', 'rcpt_dmn']:
    bank = getattr(b, name)
    report = bank.affinity_report()
    print(f"\n{name} affinities:")
    for rtype, affinities in report.items():
        top = sorted(affinities.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
        print(f"  {rtype:8s}: {top_str}")

# Test NT shape similarity matrix
shapes = b.nt_shapes
aff = shapes.affinity_matrix().detach()
from neuroslm.neurochem.transmitters import NT_NAMES
print("\nNT Shape Affinity Matrix:")
print(f"{'':>6s}", "  ".join(f"{n:>5s}" for n in NT_NAMES))
for i, n in enumerate(NT_NAMES):
    row = "  ".join(f"{aff[i,j]:.2f}" for j in range(len(NT_NAMES)))
    print(f"{n:>6s} {row}")
