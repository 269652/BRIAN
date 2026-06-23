# Run Ledger

> **Purpose.** A curated record of every meaningful training run — significant
> not just because it finished, but because it taught something: a new
> baseline, a confirmed/falsified hypothesis, a best-in-class metric, or a
> first for a new mechanism.
>
> **Rule.** Every entry links to its raw log and checkpoint. No entry is added
> without at least one number (train_ppl, ood_ppl, or gap_ratio). Entries are
> in chronological order, oldest first. Most-recent = end of file.
>
> **How to cite.** `brian cite <run-id>` formats a citation block.
> `brian tease runs` shows the N most recent entries.
> In the README template: `${TEASE:runs:5}` renders a markdown table.
>
> **How to add an entry.** Append a new `## Run: <id> · <title>` section.
> The id is `YYYYMMDD-HHMMSS` (UTC boot stamp from the log filename).
> Run `brian cite --list` to browse existing entries.

---

## Run: 20260615-175931 · H22 SmolLM2 Upgrade — First Complete 10k

**Date:** 2026-06-15
**Log:** [`logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log`](../logs/20260615/neuroslm-full-dna-arch/175931_0_10000.log)
**Checkpoint:** `hf://moritzroessler/BRIAN/checkpoints/20260615-175931_c19bf629_neuroslm-full-dna-arch/step10000.pt`
**Metrics:** train_ppl=23.6 · ood_ppl=155.0 · gap_ratio=6.55 · steps=10000

First complete 10,000-step run with the new cortex fusion stack after the
SmolLM2 architecture upgrade (H22). Total params: 1.12B (146.9M trainable).
WikiText-103 OOD PPL 155.0, train PPL 23.6. Establishes the SmolLM2
baseline — all subsequent runs are measured against this gap_ratio of 6.55.

Significance: establishes that the full cortex+trunk stack trains stably at
this scale and that OOD generalisation is measurable (gap_ratio < 10).

## Run: 20260616-140627 · H22 Best Combined Score

**Date:** 2026-06-16
**Log:** [`logs/20260616/gpt2/140627_500_10000.log`](../logs/20260616/gpt2/140627_500_10000.log)
**Checkpoint:** `hf://moritzroessler/BRIAN/checkpoints/20260616-140629_96cbeff8_neuroslm-full/step7500.pt`
**Metrics:** train_ppl=22.1 · ood_ppl=148.3 · gap_ratio=6.70 · steps=7500

Follow-up to 20260615-175931. Lower absolute train PPL (22.1 vs 23.6) and
lower OOD PPL (148.3 vs 155.0), but slightly higher gap_ratio (6.70 vs 6.55)
because train PPL improved faster than OOD PPL. This run became `.brian/best_run.ln`
by combined score (train_ppl + 4 * gap_ratio).

Significance: confirms reproducibility of SmolLM2 results and sets the
combined-score best for the gpt2-tokenizer configuration.
