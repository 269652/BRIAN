# Project history — claims and evidence

This file is the canonical narrative of the project's architectural
hypotheses, with citations. Every claim is tagged PROVEN, DISPROVEN, or
INCONCLUSIVE and linked to the run that decided it.

Maintained by `brian ai document` (see [docs/CLAUDE.md](CLAUDE.md) for
the maintenance rules). Don't edit by hand — instead update the source
docs and re-run `brian ai document`.

---

<!-- Entries populated by `brian ai document`. Each entry shape:

### YYYY-MM-DD · <Hypothesis name>

**Claim:** <verbatim from source>
**Source:** <commit / doc>
**Test:** <run name + log path>
**Result:** <PROVEN | DISPROVEN | INCONCLUSIVE> — <summary>
**Evidence:**
  - logfile: `logs/vast/<file>.log:<line>` — "<quoted>"
  - ood JSON: `logs/vast/benchmarks/ood/<file>.json` — `gap_ratio: X`
  - metric: `docs/metrics.md` row `<run_id>`

-->
