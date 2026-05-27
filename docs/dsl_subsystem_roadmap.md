# DSL Full-Coverage Roadmap — porting every BRIAN subsystem

Goal: express the **entire** `Brain` architecture — every subsystem, plus
its gradient flow — formally in the `.neuro` DSL, with the compiler and
BRIAN harness supporting whatever language features that requires. This
is the path to `train_dsl.py` producing a model semantically equal to the
hand-written `Brain`.

This is a large, multi-week effort. The doc breaks it into the **new
language constructs** required and the **subsystems** each unblocks, so
work can proceed in dependency order with strict TDD (every ported
subsystem gets a `torch.allclose` equivalence test against the Brain
reference).

---

## Status: what already exists

| Capability | Status |
|---|---|
| Algebraic population equations (`y = f(x)`) | ✅ Phase 7 S1 |
| ODE population dynamics (`dV/dt = …`) + Euler + stability | ✅ Phase 7 S2 |
| Synapse + modulation equations | ✅ multifile S5 |
| Multi-file architectures, imports/exports, lib | ✅ multifile S1–7 |
| `training { … }` config (loss clipping, optimizer, etc.) | ✅ Phase A |
| BRIAN harness: embed + LM head + loss + clip + sched + AMP + resume | ✅ Phase A / A.2 |

## The missing language constructs (in dependency order)

### L1 — `state` blocks with explicit update rules  ⟵ foundational
Most subsystems are *stateful*: a per-population (or per-synapse) value
that persists across steps and updates by a declared rule.

```neuro
population hippo {
    state mem { shape: "(M, d_sem)", init: "zeros" }
    equation: "y = read(mem, x)"
    update mem: "mem + write_gate * outer(x, key)"
}
```

Unblocks: vesicle pools, trophic state, Hebbian fast weights, hippocampus
memory bank, maturity scalar, NT concentration kinetics, DNC temporal-link
matrix.

### L2 — `auxiliary_loss` declarations
Subsystems contribute extra loss terms beyond the LM cross-entropy.

```neuro
auxiliary_loss phi {
    equation: "-mutual_information(modules)",
    weight: 0.1,
    gated_by: "maturity > 0.3"
}
```

Unblocks: Φ-integration objective, actual-causation head, world-model
prediction loss, VQ commitment loss, NEMORI predictive-forgetting,
ReasoningCortex causality loss, survival-causal-head ΔS loss.

### L3 — `param_scope` + gradient routing  ⟵ **Phase B, next**
Declarative gradient isolation (the p3 fix) and separate optimizer groups.

```neuro
param_scope trunk { populations: [sensory, thalamus, gws, pfc, motor, ...] }
param_scope bio  { populations: [amygdala, hippo, vta, ...],
                   gradient: "detached_from_main_loss" }
```

Unblocks: p3 parameter-closure isolation, trunk/bio optimizer
partitioning, frozen-during-infancy params.

### L4 — event triggers (`on …`)
Discrete events fire actions: spikes, ignition, sleep onset.

```neuro
population amygdala {
    ode: "tau * dV/dt = -(V - V_rest) + R * x"
    on V > V_thresh { emit: spike, reset: "V = V_rest" }
}
on schedule(every: 2000) { run: sleep_cycle }
```

Unblocks: integrate-and-fire spiking (full LIF), GWS ignition gate,
sleep-cycle CLS scheduling, NEMORI gating events.

### L5 — conditional execution (`when …`)
Maturation-gated paths and adaptive compute.

```neuro
when maturity > 0.3 { enable: [auxiliary_loss.phi, auxiliary_loss.causal] }
when calm_halt(token) { early_exit }
```

Unblocks: topological maturation (infancy → awakening), adaptive compute
(MoD + CALM early exit).

### L6 — structured ops (attention, codebook, gather/scatter)
First-class ML primitives beyond elementwise + matmul.

```neuro
population thought_transformer {
    equation: "y = attention(q: Wq @ x, k: Wk @ x, v: Wv @ x)"
}
population bg {
    equation: "y = vq_lookup(codebook, x)",
    state codebook { shape: "(K, d_sem)", init: "kaiming" }
}
```

Unblocks: ThoughtTransformer (real self-attention), BG VQH (vector-
quantized codebook), DNC content-addressable read/write, geometry adapter.

### L7 — sheaf / cohomology operators
Domain-specific operators for the formal-spec subsystems.

```neuro
sheaf narrative { sections: episodes, gluing: overlap_consistency }
formal_spec phi { operator: "integrated_information", over: modules }
```

Unblocks: H¹ contradiction detection + SUPERSEDES, IIT 4.0 Φ measurement,
actual-causation (κ_cause).

---

## Subsystem coverage matrix

| Brain subsystem (arch.md ref) | Needs | Phase |
|---|---|---|
| Parameter-closure isolation (p3) | L3 | **B** |
| NT-modulation leak cut (p2) | (done — wiring) | B-audit |
| Trunk transformer body | L6 (attention) | C |
| Topological maturation | L1 + L5 | C |
| Adaptive compute (MoD/CALM) | L5 | C |
| Vesicle pools (κ_cause, κ_neg) | L1 + L2 | D.1 |
| Trophic system (BDNF) | L1 + L2 | D.2 |
| Hebbian fast weights | L1 | D.3 |
| Sleep-cycle CLS | L1 + L4 | D.4 |
| WorldModel + SelfModel | L2 + L6 | D.5 |
| Hippocampus DNC + sheaf | L1 + L6 + L7 | D.6 |
| GWS ignition gate | L4 | D.7 |
| BG VQH + NAcc RPE | L1 + L2 + L6 | D.8 |
| DMN + ThoughtTransformer + Claustrum | L6 | D.9 |
| Amygdala (full LIF) + LHb + Insula | L4 | D.10 |
| Qualia + homeostatic warp | L1 + L2 | D.11 |
| BRIAN narrative + sheaf stack | L1 + L7 | D.12 |
| NEMORI predictive forgetting | L2 + L4 | D.13 |
| Personality vector + trust | L1 | D.14 |
| Cognitive closure (gridworld loop) | L1 + L2 + L4 | D.15 |
| ActualCausationHead (IIT 4.0) | L2 + L7 | D.16 |

## Execution order (strict-sequential, TDD-gated)

1. **Phase B** — L3 (`param_scope`) → p3 fix declarative. *(starting now)*
2. **Phase C** — L1 + L5 + L6-attention → trunk parity + maturation.
3. **Phase D.1–D.16** — one subsystem per stage, each with an equivalence
   test vs the Brain reference impl, in the dependency order above.
4. **Phase E** — real data loader, BEMA, multi-stream checkpoints,
   episode recording.
5. **Phase F** — bit-identical forward + benchmark parity (HellaSwag/
   ARC/MMLU) between `train.py` (Brain) and `train_dsl.py` (DSL).

Realistic calendar: **6–12 weeks** of focused work for full coverage.
Each phase ships independently; the vast deploy can re-launch on the DSL
path at any phase boundary with strictly more Brain-equivalent behavior.

## TDD discipline

Every subsystem port lands a test of the form:

```python
def test_<subsystem>_matches_brain_reference():
    brain_out = reference_brain_subsystem(x, seed=0)
    dsl_out   = compiled_dsl_subsystem(x, seed=0)
    assert torch.allclose(brain_out, dsl_out, atol=1e-5)
```

No phase merges to master unless its equivalence test is green.
