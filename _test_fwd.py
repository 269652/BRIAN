# -*- coding: utf-8 -*-
try:
    import torch
except Exception:
    print('SKIP: torch is not installed. Install requirements (pip install -r requirements.txt) to run forward tests.')
    raise SystemExit(0)

from neuroslm.config import PRESETS
from neuroslm.brain import Brain

cfg = PRESETS['tiny']()
cfg.vocab_size = 50257
b = Brain(cfg)
ids = torch.randint(0, 100, (2, 32))
tgt = torch.randint(0, 100, (2, 32))
out = b.forward_lm(ids, tgt)
print("Forward OK, loss={:.6f}".format(out['loss'].item()))

# Test receptor affinity report
for name in ['rcpt_pfc', 'rcpt_hippo', 'rcpt_bg', 'rcpt_thal', 'rcpt_lang', 'rcpt_dmn']:
    bank = getattr(b, name)
    report = bank.affinity_report()
    print('\n{} affinities:'.format(name))
    for rtype, affinities in report.items():
        top = sorted(affinities.items(), key=lambda x: -x[1])[:3]
        top_str = ', '.join('{}={:.3f}'.format(k, v) for k, v in top)
        print('  {0:8s}: {1}'.format(rtype, top_str))

# Test NT shape similarity matrix
shapes = b.nt_shapes
aff = shapes.affinity_matrix().detach()
from neuroslm.neurochem.transmitters import NT_NAMES
print('\nNT Shape Affinity Matrix:')
print('{:>6s} {}'.format('', '  '.join('{:>5s}'.format(n) for n in NT_NAMES)))
for i, n in enumerate(NT_NAMES):
    row = '  '.join('{:.2f}'.format(aff[i, j]) for j in range(len(NT_NAMES)))
    print('{:>6s} {}'.format(n, row))
