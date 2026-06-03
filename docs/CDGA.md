# CDGA — Cross-Distribution Gradient Alignment

> *Gradient surgery for distribution shift in single-task language modelling.*

**Status**: Implemented behind `regularization.cdga { enabled: ... }` in
`arch.neuro`. Dormant (telemetry-only) until an anchor function is
registered via `harness.set_cdga_anchor_fn(fn)`.

**Math-first spec**: `architectures/rcc_bowtie/lib/cdga.neuro`

**Implementation**: `neuroslm/regularizers.py :: CDGAController`

**Tests**: `tests/test_cdga.py` (30 cases)

---

## 1. Motivation

The 2026-06-03 training trace exposed a failure mode that no LOSS-SPACE
regularizer could fix:

| step | train_ppl | wiki_ppl | gap_ratio |
|------|-----------|----------|-----------|
| 500  | 1926      | 7698     | 4.00      |
| 1000 | 872       | 6437     | 7.38      |

Train loss dropped 55% while OOD loss barely moved. The five PR2
interventions (DAR, PCC, Isotropy, CMD, Adaptive Mixture) all add scalar
penalties to the loss — they're **isotropic in parameter space**. When
the data distribution is dominated by one mode (chat 80%) the vanilla LM
gradient still points into the chat-attractor basin every step;
defensive regularizers add friction but no direction.

What is missing is an **anisotropic, OOD-aware gradient signal**. CDGA
supplies it directly in gradient space, not in loss space.

## 2. Algorithm

At every Nth optimizer step:

1. After the standard `loss.backward()`, snapshot the training gradient

$$g_{\text{train}} = \nabla_\theta\, \mathcal{L}_{\text{LM}}(x_{\text{train}};\, \theta)$$

2. Zero the gradients. Run a forward+backward on a **held-out OOD
   anchor batch** (e.g. pure WikiText prose, never overlapping with
   the OOD evaluation slice):

$$g_{\text{anchor}} = \nabla_\theta\, \mathcal{L}_{\text{LM}}(x_{\text{anchor}};\, \theta)$$

3. Compute the **conflict coefficient** — the negative projection
   coefficient, rectified:

$$c = \max\!\left(0,\; -\,\frac{\langle g_{\text{train}},\, g_{\text{anchor}}\rangle}{\langle g_{\text{anchor}},\, g_{\text{anchor}}\rangle}\right)$$

4. Subtract the conflicting component (scaled by the global strength α):

$$g_{\text{aligned}} = g_{\text{train}} - \alpha \cdot c \cdot g_{\text{anchor}}$$

5. Write `g_aligned` back into `p.grad`. Optimizer step proceeds normally.

The anchor parameters are **never updated** — the anchor pass is purely
a directional reference.

## 3. Properties

### P1 — Non-worsening anchor (first order)

When $\alpha = 1$ and $c > 0$, the first-order expected change in the
anchor loss is non-positive:

$$\langle \nabla_\theta\, \mathcal{L}_{\text{anchor}},\; -\eta \cdot g_{\text{aligned}}\rangle \;=\; -\eta\, \langle g_{\text{anchor}}, g_{\text{train}}\rangle + \eta \cdot c \cdot \|g_{\text{anchor}}\|^2$$

When the gradients oppose ($\langle g_{\text{anchor}}, g_{\text{train}}\rangle < 0$):

$$c = -\frac{\langle g_{\text{train}}, g_{\text{anchor}}\rangle}{\|g_{\text{anchor}}\|^2}$$

so the second term cancels the first exactly. The first-order change in
$\mathcal{L}_{\text{anchor}}$ from the step is zero. The training step
*cannot increase anchor loss to first order*.

### P2 — Bounded perturbation

$$\|g_{\text{aligned}} - g_{\text{train}}\| \;=\; \alpha\, c\, \|g_{\text{anchor}}\| \;\le\; \alpha \, \|g_{\text{train}}\| \, |\cos\theta| \;\le\; \alpha \, \|g_{\text{train}}\|$$

CDGA never changes the gradient by more than $\alpha \cdot \|g_{\text{train}}\|$
(at $\alpha = 1$, never more than the training gradient itself). Tested
in `test_perturbation_bounded`.

### P3 — Idempotent on aligned tasks

When $\langle g_{\text{train}}, g_{\text{anchor}}\rangle \ge 0$ we have $c = 0$ and
$g_{\text{aligned}} = g_{\text{train}}$. CDGA is a no-op when training and OOD
gradients already point the same way — exactly the regime where no
intervention is needed.

### P4 — Warmup safety

The same `warmup_steps` ramp used by the rest of the regularization
controller scales α. Before warmup completes, both gradients are random
noise and surgery would be a coin-flip. The CDGA telemetry (cos θ)
still gets computed during warmup so it can be plotted as a leading
indicator of when surgery becomes meaningful.

## 4. Related work

CDGA is the same projection operator as **PCGrad** (Yu et al. 2020,
"Gradient Surgery for Multi-Task Learning", NeurIPS), applied to a
*single* LM task evaluated on *two distributions*, rather than two
distinct task objectives.

Adjacent ideas that don't solve the same problem:

| Method | Operates in | Direction-aware? | OOD-target-aware? |
|---|---|---|---|
| GroupDRO (Sagawa 2020) | loss (sample reweighting) | no | yes |
| IRM (Arjovsky 2019) | loss (invariance penalty) | no | partial |
| DAR (Ganin 2015) | loss (adversarial) | no | yes |
| PCC / CPC (Oord 2018) | loss (contrastive) | no | no |
| Isotropy whitening | loss (Gram regularizer) | no | no |
| **CDGA (this doc)** | **gradient (projection)** | **yes** | **yes** |

The closest method in spirit is **gradient projection memory** (Lin
2019) for continual learning — but that projects *out* the subspace of
*old* gradients to prevent catastrophic forgetting. CDGA does the
opposite: it projects out the component that *conflicts* with the *OOD
reference* to prevent distribution-shift overfitting.

## 5. Cost analysis

CDGA adds one extra forward+backward every `refresh_every` steps. With
the default anchor batch size of ¼ × training batch size and
`refresh_every = 4`:

$$\text{overhead} \;=\; \frac{1}{4}\, \times\, \frac{1}{4}\, \times\, 100\% \;=\; 6.25\%\, \text{ per training step amortised}$$

For a 122M param model at batch 4 × ctx 2048 on bf16 this is ~30 ms /
step extra, well within the existing 200-300 ms / step budget.

No new trainable parameters. Memory overhead is one flat snapshot of
`g_train` (~500 MB for 122M params at fp32, freed after surgery).

## 6. Test contamination

The anchor corpus partition **must be disjoint** from the OOD
evaluation slice. The DSL spec captures this as a `formal_spec` block
in `lib/cdga.neuro`:

```neuro
formal_spec cdga_test_contamination {
    rule:   "anchor_corpus_disjoint_from_eval",
    metric: "intersect(anchor_doc_ids, ood_eval_doc_ids) = empty"
}
```

The current contract is that `set_cdga_anchor_fn(fn)` is the only
attach point for anchor data; the function is responsible for using a
partition that does not overlap with the OOD evaluator (see
`deploy/ood_eval.py` for the slice currently in use).

## 7. Worked example: WikiText-103 anchor

The recommended setup for the current architecture:

```python
# In neuroslm/train_dsl.py, after harness construction:
from neuroslm.data.text_corpus import WikiTextCorpus  # provided
anchor_corpus = WikiTextCorpus(
    split="train",                       # NOT validation/test
    skip_first_n_docs=1000,              # leave room for eval slice
    seq_len=512,                          # ¼ of training ctx
    batch_size=1,                         # ¼ of training batch
    tokenizer=tokenizer,
    device=device,
)

def cdga_anchor_fn(harness):
    """Sample a held-out prose batch and return its LM loss."""
    ids, targets = anchor_corpus.next_batch()
    logits = harness.language_model(ids)
    flat_l = logits.reshape(-1, logits.shape[-1])
    flat_t = targets.reshape(-1)
    return torch.nn.functional.cross_entropy(flat_l, flat_t)

harness.set_cdga_anchor_fn(cdga_anchor_fn)
```

Then flip `cdga: { enabled: true, ... }` in `arch.neuro`.

## 8. Expected telemetry

A healthy CDGA run shows:

- **`cdga_alpha`**: ramps linearly from 0 to `alpha_max` over `warmup_steps`.
- **`cdga_cosine`**: should be near-zero or slightly positive at the
  start of warmup, then trend toward **positive** values as the model
  learns features that help both distributions. A monotonically
  decreasing cosine is the diagnostic signal CDGA is designed to catch
  — it means the chat-attractor is pulling the gradient away from the
  prose manifold.
- **`cdga_conflict`**: spike at the start of warmup (random gradients
  often oppose), then trend toward 0 as the LM learns shared features.
  Persistent conflict > 0.5 means CDGA is doing meaningful work every
  step (the failure mode it's preventing is recurring) — investigate
  the data distribution.

If `cdga_cosine` stays negative for >5000 steps, the anchor and
training corpora are misaligned and CDGA is fighting itself. Audit the
anchor partition.

## 9. Future work

- **Per-layer surgery**: compute conflict per layer block rather than
  globally. Avoids over-projecting when only the top layers carry the
  shortcut.
- **Multi-anchor**: extend to a *set* of anchor distributions
  (`{wiki, code, conversational}`) with a tiered projection rule.
- **Adaptive α**: tune α with a controller that targets a specific
  `cdga_cosine` floor (close the loop on the metric, not the step count).
- **Surgical optimizer**: integrate the projection step into the
  optimizer (e.g. as a hook in `torch.optim.Adam.step`) so the snapshot
  + flat tensor manipulation can be elided.

---

*Original design and implementation: 2026-06-03, in response to the
gap_ratio regression observed in run dsl_arch_20260603-144410.*
