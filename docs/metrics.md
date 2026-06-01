# BRIAN/NeuroSLM Metrics — Run-by-Run Comparison

One row per training or OOD-eval run. Auto-updated by
`brian analyze-log <logfile>`. Rows are upserted by run id —
rerunning a log replaces the prior row.

| Run | Date | Branch | Arch | Steps | Loss | LM | PPL | Phi | OOD-PPL | OOD-ratio | tok/s | Notes |
|-----|------|--------|------|-------|------|----|-----|-----|---------|-----------|-------|-------|
| dsl-step10000-v2 | 2026-06-01 |  | ood | ? | ? | ? | ? | ? | 837.6 | 7.04 | ? | OOD eval, ckpt=? |
| 38569395 | 2026-05-30 | arch/rcc-p4-loss-clip | dsl rcc_bowtie_30m_p4 | 10000 | 4.62 | 4.62 | 101.30 | 0.428 | ? | ? | 102885 | train |
| 38469631 | 2026-06-01 |  | dsl rcc_bowtie_30m_p4 | 10000 | 5.49 | 5.49 | 242.10 | 1.060 | ? | ? | 30913 | train |
| 249a6c1e08e9 | 2026-05-30 |  | dsl rcc_bowtie_30m_p4 | 10000 | 4.62 | 4.62 | 101.30 | 0.428 | ? | ? | 102885 | train |
