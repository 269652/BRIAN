# NeuroSLM Documentation Index

This directory contains all technical and reference documentation for the BRIAN project. Use this index to find what you need.

---

## Quick Navigation by Purpose

### 🚀 **Getting Started** (Start Here)

| Document | Read if... | Time |
|----------|-----------|------|
| [`../README.md`](../README.md) | You're new to BRIAN and want the big picture | 5 min |
| [`BRIAN.md`](BRIAN.md) | You want to understand the 11-stage architecture and why each design choice exists | 15 min |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | You're going to build or modify the code | 20 min |

---

### 📚 **Complete Specifications**

| Document | For... | Contains |
|----------|--------|----------|
| **[`architecture.md`](architecture.md)** | Researchers, reproducers | Full spec: tensor shapes, mathematical formulas, all modules (§0–12), 1300+ lines. Every claim is citable to a line or test. |
| **[`technical_report.md`](technical_report.md)** | External AIs, new contributors | Executive summary: project charter, current state, proven claims, evidence artifacts, open questions. Designed for NotebookLM/ChatGPT. |
| **[`formal_framework.md`](formal_framework.md)** | Theorists, evolutionary loop | **Normative** mathematical contract for THSD (simpliziale ontology, $H^1$ guard, symbolic-simplex discovery operator, Φ guard, Tonnetz filter, Fisher-Rao retrieval, RAID-5 DNA). Source of Truth for evolutionary mutations. |
| **[`BRAIN.md`](BRAIN.md)** | Architects, designers | Why BRIAN exists: the 11-stage forward pass explained, each design decision with rationale and evidence. |
| **[`harness.md`](harness.md)** | Training engineers | How BRIANHarness works: loss clipping, gradient accumulation, maturity phasing, OOD evaluation, metrics. |

---

### 🔬 **Evidence & Results**

| Document | Purpose | When to read |
|----------|---------|-------------|
| **[`findings.md`](findings.md)** | Hypothesis ledger with Layer A (mechanisms) + Layer B (OOD eval) results. Every claim is linked to a test or result JSON. | When verifying claims, checking reproducibility, or understanding what we've actually proven vs overclaimed. |
| **[`metrics.md`](metrics.md)** | Auto-updated during training. Training loss curves, convergence rates, NT dynamics, etc. | When debugging training behavior or comparing runs. |
| **[`history.md`](history.md)** | Session notes, decisions, insights from past runs. Auto-maintained. | When understanding the context behind a design choice. |
| **[`changelog.md`](changelog.md)** | Git-derived commit history (auto-maintained). | When tracing when a feature was added. |

---

### 🛠️ **Reference & How-To**

| Document | For... | Contains |
|----------|--------|----------|
| **[`CLI.md`](CLI.md)** | Users of command-line tools | Complete reference: `train`, `train_dsl`, `generate`, OOD eval scripts, checkpoint management. |
| **[`dsl.md`](dsl.md)** | DSL users & implementers | `.neuro` language syntax, compilation pipeline, equation IR, examples. |

---

### 📂 **Archived / Historical**

Stale documentation is moved to `archive/` with a date prefix (e.g., `2026-06-01_experiment_name.md`). Check here when:
- You want to understand why something was abandoned
- You're looking for session notes from a past experiment

---

## The Documentation Workflow

### For Users
1. Start at [`../README.md`](../README.md) (5 min big picture)
2. Pick your path:
   - **Want to understand the design?** → [`BRAIN.md`](BRAIN.md) (15 min)
   - **Want to code?** → [`../CONTRIBUTING.md`](../CONTRIBUTING.md) (20 min)
   - **Want to train?** → [`CLI.md`](CLI.md) (reference)
   - **Want to verify claims?** → [`findings.md`](findings.md) (evidence)

### For External AIs (NotebookLM, Perplexity, ChatGPT)
1. Load [`technical_report.md`](technical_report.md) — self-contained, all evidence links included
2. Reference [`architecture.md`](architecture.md) for mathematical details if needed
3. Check [`findings.md`](findings.md) for what's actually proven vs what's pending

### For Contributors (AI or Human)
1. Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — TDD workflow, DSL conventions, testing patterns
2. Reference [`architecture.md`](architecture.md) (§0–12) for detailed specs
3. Check [`findings.md`](findings.md) before starting work (understand what's already proven)
4. After making changes, run: `python scripts/maintain_technical_report.py --verbose` to audit docs sync

---

## At a Glance: What Each Document Covers

| Doc | Depth | Audience | Length | Status |
|-----|-------|----------|--------|--------|
| `README.md` (root) | Overview | Everyone | 2 pages | Maintained |
| `CONTRIBUTING.md` | Implementation | Builders | 8 sections | Stable |
| `BRIAN.md` | Architecture | Designers | 10 sections | Stable |
| `architecture.md` | Full spec | Researchers | 1300+ lines | Primary source |
| `formal_framework.md` | THSD theory | Theorists / evo loop | 9 sections + glossary | Normative (v0.1) |
| `technical_report.md` | Executive | External AIs | 16 sections | Auto-maintained |
| `findings.md` | Evidence | Verifiers | 15 hypotheses | Live |
| `harness.md` | Training | Engineers | 10 sections | Stable |
| `CLI.md` | Reference | Users | Commands | Reference |
| `dsl.md` | DSL syntax | Implementers | Language spec | Reference |
| `history.md` | Sessions | Context | Auto | Auto-maintained |
| `changelog.md` | Commits | Tracing | Git-derived | Auto |
| `metrics.md` | Training | Debug | Auto | Auto-updated |

---

## Synchronization Rules

From `CLAUDE.md` § 9: **Every architectural change syncs docs automatically.**

1. **Code changes** → Update `arch.neuro` first (canonical source)
2. **Spec changes** → Update `architecture.md` + `technical_report.md`
3. **New evidence** → Update `findings.md` + `technical_report.md`
4. **Run audit script:** `python scripts/maintain_technical_report.py --verbose`
5. **Commit together:** One logical change = one commit with all docs

This keeps everything in sync. No more "the code changed but the docs are stale."

---

## Key Principles

### Clarity Over Comprehensiveness
- Each doc has a single clear purpose
- Cross-links, don't duplicate
- If you're writing the same section twice, move it to a shared file

### Discoverability
- This index tells you where to find things
- Use the table at the top to navigate
- Each doc has a README comment explaining what's inside

### Maintainability
- Specs (architecture.md) are primary; others derive from them
- Auto-maintain derived docs (history.md, changelog.md, metrics.md)
- Archive old docs (don't delete); keep audit trail

---

## Links by Purpose

**I want to...**

- **Understand BRIAN's design** → [`BRIAN.md`](BRIAN.md)
- **Implement a new mechanism** → [`../CONTRIBUTING.md`](../CONTRIBUTING.md) → [`architecture.md`](architecture.md) § relevant section
- **Train a model** → [`CLI.md`](CLI.md)
- **Verify that claims are real** → [`findings.md`](findings.md)
- **Understand why a decision was made** → [`history.md`](history.md) (search for the decision)
- **Work with the DSL** → [`dsl.md`](dsl.md)
- **Debug training behavior** → [`harness.md`](harness.md)
- **Share the project with external AI** → [`technical_report.md`](technical_report.md)
- **Check what changed in a commit** → [`changelog.md`](changelog.md)
- **See training metrics over time** → [`metrics.md`](metrics.md)

---

**Last updated:** 2026-06-01  
**Maintainer:** Automatically curated per CLAUDE.md § 9

