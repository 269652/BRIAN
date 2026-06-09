---
code_refs: [neuroslm/regularizers.py]
created_at: "2026-06-09T18:00:16Z"
id: H002
proof_path: hypothesis/proofs/H002_ood_gap_decrease_under_cdga.lean
proof_status: stub
references: [formal_framework.md §10.2, docs/CDGA.md]
status: stated
tags: [ood, cdga, monotonicity, regularisation]
test_refs: [tests/test_cdga_smoke.py]
theorem_name: Brian.OodGapDecrease
title: OOD gap decrease under CDGA
updated_at: "2026-06-09T18:00:16Z"
---

**Statement.** Let $L_{\text{base}}$ be the base training loss and $\mathrm{CDGA}$ the Cross-Distribution Gradient Alignment term (``docs/CDGA.md``, $\lambda \ge 0$). Then

$$\Delta_{\mathrm{OOD}}(\theta + \lambda\cdot\mathrm{CDGA}) \;\le\; \Delta_{\mathrm{OOD}}(\theta)$$

where $\Delta_{\mathrm{OOD}} = L_{\text{OOD}} - L_{\text{ID}}$ is the OOD generalisation gap.

**Why it matters.** Adds a free generalisation guarantee: turning on the CDGA term never widens the OOD gap, even when it doesn't help. The empirical version is exercised in ``test_cdga_smoke.py``.