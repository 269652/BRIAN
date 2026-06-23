# Semantic Turbulence Engine (STE)

> **Source:** `neuroslm/emergent/semantic_turbulence.py`
> **Tests:** `tests/training/test_ste_harness_norm.py`
> **DSL key:** `feature { id: "semantic_turbulence" ... }`

The Semantic Turbulence Engine is a suite of three physics-inspired mechanisms
that enrich the trunk's hidden states with multi-scale structure, measure
semantic coherence via a superfluid order parameter, and drive the network
toward the self-organised critical point where information transmission is
maximised.

All three modules are composable — they can be wired in individually or as a
stack. In the current architecture they slot between transformer blocks as
additional residual enrichments.

---

## Module 1 — RenormalizationGroupCascade

**Class:** `RenormalizationGroupCascade`  
**Reference:** Kolmogorov (1941), "Local structure of turbulence in
incompressible viscous fluid for very large Reynolds numbers."

### What it does

Partitions the sequence into G groups operating at token scales 2^g
(group 0 = block 2, group 1 = block 4, group 2 = block 8, …). For each
group:

1. **Coarse-grain** via block-mean pooling:
   `H_coarse_g = mean(H[:, :T_eff, :].view(B, n_blocks, block_size, d), dim=2)`

2. **Extract fluctuations** (the "turbulent" residual at scale g):
   `dH_g = H[:, :T_eff, :] - upsample(H_coarse_g)`

3. **Enrich** with a scale-specific linear projection + coupling scalar:
   `out += proj_g(upsample(H_coarse_g)) + lambda_g * dH_g`

### Kolmogorov coupling

The coupling scalars `lambda_g` are initialised according to Kolmogorov's
5/3 law (energy cascade in turbulent fluids):

```
lambda_g = 2^(-5g/6)
```

Group 0 (finest scale): lambda ≈ 1.0  
Group 1: lambda ≈ 0.56  
Group 2: lambda ≈ 0.31  

This means coarser (longer-range) structure contributes progressively less to
the residual, matching the energy dissipation rate in turbulent cascades.

### Zero-init discipline

Output projections `scale_proj[g]` are zero-initialised (+ 1e-3 * I),
so the cascade starts as an exact no-op. The model learns to activate
cross-scale coupling only when beneficial — the same ReZero discipline used
in the trunk's gate scalars.

### Constructor

```python
RenormalizationGroupCascade(
    d_model=512,       # hidden dim
    n_groups=3,        # number of scale groups G
    kolmogorov_init=True,  # init lambda_g by 5/3 law
)
```

### Forward signature

```python
out: Tensor = rg_cascade(H)  # H: (B, T, d) -> (B, T, d)
```

---

## Module 2 — GrossPitaevskiiLayer

**Class:** `GrossPitaevskiiLayer`  
**References:** Gross (1961); Pitaevskii (1961), Bose-Einstein condensate
mean-field equation.

### What it does

Encodes the real hidden state as a **complex superfluid field** `psi in
C^{d/2}`, runs N imaginary-time Euler steps of the Gross-Pitaevskii
equation, and decodes back to real.

The **order parameter** rho in [0, 1] measures semantic coherence:

```
rho = |<psi / |psi|>|^2
```

- `rho -> 1` (condensate): tokens are semantically unambiguous — the
  model has a sharp, coherent interpretation of the context.
- `rho -> 0` (disordered): context is highly polysemous or semantically
  conflicted.

### Imaginary-time GPE step

```
psi_t <- psi_t - dt * g * |psi_t|^2 * psi_t
```

followed by norm-preserving rescaling (particle-number conservation).
The full kinetic Laplacian term (-nabla^2 psi / 2) is omitted — sequence
attention already handles long-range phase coupling. What remains is the
**contact interaction** (mean-field Bose-Hubbard) which drives condensation
when the coupling g is positive.

The coupling `g = exp(log_g)` is a learnable parameter initialised to a
small positive value (ReZero: ~0.01), so the layer starts near identity.

### Forward

```python
# Simple forward (drops rho)
out: Tensor = gpe_layer(x)            # x: (B, T, d) -> (B, T, d)

# Forward with order parameter
out, rho = gpe_layer.forward_with_rho(x)
# rho: scalar in [0, 1] — useful for telemetry and loss regularisation
```

### Constructor

```python
GrossPitaevskiiLayer(
    d_model=512,           # real hidden dim (must be even)
    gpe_steps=4,           # N imaginary-time Euler steps
    gpe_coupling_init=0.01, # initial interaction strength g
    gpe_dt=0.01,           # imaginary-time step size dt
)
```

---

## Module 3 — BranchingRatioMonitor

**Class:** `BranchingRatioMonitor`  
**Reference:** Beggs & Plenz (2003), "Neuronal avalanches in neocortical
circuits."

### What it does

Tracks the **branching ratio** sigma — the layer-to-layer amplification
factor. At the critical point sigma = 1.0 (Beggs & Plenz), a network
maximises its dynamic range, susceptibility, and information transmission.

```
sigma ≈ ||h_next||_F / (||h_prev||_F + eps)
```

This is the leading-order approximation to the mean Jacobian Frobenius norm
(exact for linear layers; valid as an EMA-smoothed signal for nonlinear ones).

### Neuromodulator signals

The monitor maps sigma to three neuromodulator signals:

| Signal | Trigger | Effect |
|--------|---------|--------|
| **GABA** (inhibitory) | sigma > sigma* (supercritical) | dampen activity |
| **NE** (excitatory) | sigma < sigma* (subcritical) | boost activity |
| **DA** (reward) | sigma ≈ sigma* (critical) | reinforce current state |

```python
sigs = monitor.nt_signals(sigma)
# {"gaba": 0.12, "ne": 0.0, "da": 0.08}
```

### Criticality loss

Adds `weight * (sigma - sigma*)^2` to the training loss, pulling sigma
toward the critical point as a soft regulariser.

```python
loss_crit = monitor.criticality_loss(sigma)  # scalar tensor
```

### Usage pattern

```python
monitor = BranchingRatioMonitor(target=1.0, weight=0.01)

# Inside training loop, between layer l and l+1:
sigma = monitor.measure_sigma(h_l, h_lp1)
monitor.update_ema(sigma.item())
loss = loss + monitor.criticality_loss(sigma)

# Optional NT telemetry:
nt = monitor.nt_signals(monitor.sigma_ema)
log({"sigma": monitor.sigma_ema, **nt})
```

### Constructor

```python
BranchingRatioMonitor(
    target=1.0,      # sigma* — critical point
    ema_alpha=0.05,  # EMA smoothing factor
    da_reward=0.1,   # DA amplitude at criticality
    weight=0.01,     # criticality loss weight
)
```

---

## Composing the STE stack

The three modules can be combined as a residual stack:

```python
# Wiring sketch (simplified from harness.py)
H = trunk_block(H)                        # standard transformer block

H = rg_cascade(H)                         # Module 1: scale enrichment

out_gpe, rho = gpe_layer.forward_with_rho(H)
H = H + out_gpe                           # Module 2: superfluid coherence
telemetry["rho"] = rho.item()

sigma = monitor.measure_sigma(H_prev, H)  # Module 3: criticality
monitor.update_ema(sigma.item())
loss = loss + monitor.criticality_loss(sigma)
```

---

## DSL surface

The STE is exposed via the `feature` block in `arch.neuro`:

```
feature {
    id: "semantic_turbulence"
    active: true
    params {
        rg_groups: 3
        rg_kolmogorov_init: true
        gpe_steps: 4
        gpe_coupling_init: 0.01
        gpe_dt: 0.01
        brm_target: 1.0
        brm_weight: 0.01
    }
}
```

Setting `active: false` disables the stack for clean A/B ablations without
removing the wiring.

---

## Tests

All STE wiring contracts are covered by
`tests/training/test_ste_harness_norm.py`.

Key contracts:
- `RenormalizationGroupCascade` starts as identity (zero-init projections).
- `GrossPitaevskiiLayer.forward_with_rho` returns `rho in [0, 1]`.
- `BranchingRatioMonitor.criticality_loss` is zero when sigma == sigma*.
- NT signals are non-negative and sum to a finite value.
- Full STE stack preserves output shape `(B, T, d_model)`.

---

## Theory notes

### Why turbulence?

In fluid dynamics, Kolmogorov's theory describes how kinetic energy
injected at large scales cascades down to smaller and smaller eddies,
dissipating at the molecular scale. The RG cascade borrows this framing:
semantic information injected at the document level (long contexts) should
influence word-level representations via a structured cascade — not just
via attention scores. The 5/3 coupling law gives the cascade a principled
initialisation that avoids either over- or under-weighting fine-scale
fluctuations.

### Why a superfluid?

Bose-Einstein condensation provides a natural order parameter for
"coherence across many modes." In the linguistic analogy: a polysemous
word in an ambiguous sentence has many possible interpretations active
simultaneously (disordered, rho near 0); a word with a clear meaning in
context has most probability mass collapsed onto one interpretation
(condensed, rho near 1). The GPE evolution lets the network discover this
coherence dynamically rather than encoding it structurally.

### Why criticality?

Beggs & Plenz (2003) showed that cortical networks operate near sigma = 1,
the boundary between subcritical (exponential decay of activity) and
supercritical (explosive amplification) regimes. This critical point is a
fixed point of the system's information dynamics: small perturbations
propagate maximally far without diverging. The BranchingRatioMonitor
implements a soft constraint that pushes sigma toward this operating point
during training.

---

## Cross-references

- Architecture spec: `docs/architecture.md §5.x` (STE mechanisms section)
- Formal framework: `docs/formal_framework.md` (THSD analysis of criticality)
- Findings: `docs/findings.md` (STE-related hypotheses: search for "STE")
- Run ledger: `docs/runs.md` (runs with STE enabled)
