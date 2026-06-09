# BRIAN

> Research prototype of a small language model whose architecture is specified in a typed neural DSL (`.neuro`), compiled to PyTorch, and exercised by a unit-test suite that pins mechanism-level behaviour. Biologically-inspired wiring (bowtie topology, neurotransmitter-modulated synapses, re-entry loops, optional cortex fusion). No claim of consciousness, sentience, or SOTA performance.

[![tests](https://img.shields.io/badge/tests-1825%20collected-blue)](#running-tests)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![torch](https://img.shields.io/badge/torch-2.x-orange)]()
[![license](https://img.shields.io/badge/license-research-lightgrey)]()

## What this is

BRIAN ("Biologically-inspired Recurrent Integrated Architecture Network") is an experiment in building a language model from a typed declarative spec rather than hand-written PyTorch modules. The current default architecture (`architectures/rcc_bowtie`, declared in `brian.toml`) compiles to a **45.7 M-parameter** PyTorch model (≈7.1 M non-embedding, the rest is the tied 50 257-vocab GPT-2 embedding).

The hypothesis the project is testing is whether biologically-motivated structural priors — a bowtie information bottleneck, neurotransmitter-style scalar gates, optional re-entry loops, and frozen GPT-2 "cortex" experts fused via KL distillation — produce meaningfully different generalization behaviour than a same-size flat transformer. **That hypothesis is not yet answered.** The results table below reports the actual measurements taken so far.

What is actually in the repo today:

- A `.neuro` DSL parser + compiler that turns architecture specs into PyTorch modules (`neuroslm/dsl/`).
- A canonical architecture (`architectures/rcc_bowtie`) compiled by the brian.toml `[current]` section: **29 populations · 7 neurotransmitter systems · 23 synapses · 17 modulations** (counts emitted by `brian compile nfg --current`; rendered to [`.neuro/nfg.png`](.neuro/nfg.png)).
- A training harness (`neuroslm/train_dsl.py`, `neuroslm/harness.py`) with optional multi-cortex fusion, KL distillation, and α-gating.
- A test suite of **1825 collected tests** (4 deselected) covering DSL parsing, code-generation byte-equivalence, training-step shape contracts, and individual mechanism behaviour.

What is **not** in the repo: a trained checkpoint that beats a same-size flat-transformer baseline, a peer-reviewed result, or a demonstration of any cognitive or conscious capability.

---

## Architecture (what compiles, and at what size)

The active architecture is whichever `architectures/*` directory `brian.toml`'s `[current].arch` points to. The default is `architectures/rcc_bowtie`. Compiled with the default GPT-2 vocabulary it yields:

| Quantity | Value | Source |
|---|---|---|
| Total parameters | 45 678 720 (≈45.7 M) | `sum(p.numel() for p in build_lm_from_preset('rcc_bowtie_30m_p4').parameters())` |
| Non-embedding parameters | ≈7.1 M | same, excluding `wte` / tied `lm_head` |
| Embedding-like parameters | ≈38.6 M | tied input + output embedding for `vocab=50257`, `d_model=384` |
| `d_model` | 384 | `architectures/rcc_bowtie/arch.neuro` |
| Depth | 4 | same |
| Heads | 6 | same |
| Max context | 1024 | same |
| Populations | 29 | `brian compile nfg --current` output line |
| Neurotransmitter systems | 7 | same |
| Synapses | 23 | same |
| Modulations | 17 | same |

The "30m" in the preset name `rcc_bowtie_30m_p4` is a historical label for the non-embedding budget order of magnitude, not a measured number — the actual non-embedding count is ≈7 M and total with the tied 50 k vocab is ≈46 M.

The structural choices the architecture makes (vs a same-size flat transformer):

| Choice | What it does in code | What we *know* it changes |
|---|---|---|
| Bowtie topology | Routes information through a narrow `d_sem` bottleneck before the LM head | Reduces non-embedding parameter count vs full-width transformer at same depth |
| Neurotransmitter-style scalar modulations | Per-step learned scalars gate specific synapses (`modulation` blocks in `.neuro`) | Adds ≈O(modulations) extra scalar parameters; behaviour pinned by unit tests in `tests/dsl/` |
| Optional cortex fusion | Mixes logits from 3 frozen GPT-2 small experts with the trunk via convex combination + KL distillation | Initial CE returns to ≈ln(50257)=10.82 with `cortex_pre_head_norm`; without it, CE init was measured at ≈13.84 (`tests/training/test_cortex_pre_head_norm.py`) |
| `ImprovementGate` (DNA evolution loop) | Welch's *t*-test admission for proposed mutations | Implementation matches scipy reference within 1e-6 on the synthetic test set (`tests/verification/test_improvement_gate.py`) |

**What we do *not* know:** whether any of these choices improves out-of-distribution generalization or sample efficiency vs a parameter-matched flat transformer at the same training budget. The OOD numbers below are the only data we have on that question, and they do not currently support a positive answer.

---

## Architecture diagram

The brain wiring is declared in `architectures/rcc_bowtie/arch.neuro` (plus per-region files in `modules/`) and rendered to a Graphviz Neural Flow Graph. Every node, edge, and modulation in the diagram corresponds to a `population`, `synapse`, or `modulation` block in the DSL.

![Neural Flow Graph — current architecture](.neuro/nfg.png)

Re-render after editing `arch.neuro`:

```powershell
brian compile nfg --current                       # writes to .neuro/nfg.png
brian compile nfg --current --heat heatmap.json   # overlay training-activity heatmap
```

The `--current` flag reads which architecture to render from [`brian.toml`](brian.toml). The diagram is a faithful rendering of the DSL source — the DSL is the source of truth for what compiles, the diagram is derived.

---

## The `.neuro` DSL

Architecture is specified declaratively in `.neuro` files: typed population definitions, synapse edges, and scalar modulations. The DSL compiles to PyTorch modules at runtime via `neuroslm/dsl/`:

```neuro
# architectures/rcc_bowtie/modules/amygdala.neuro
export population amygdala {
    count: 32,
    ode: "dV/dt = (-V + x) / tau",
    timescale: 0.005
}

# architectures/rcc_bowtie/arch.neuro
modulation dopamine -> pfc {
    effect: "multiplicative", gain: 0.6,
    equation: "y = output * (c * gain)"
}
```

**Why a DSL.** Hand-written PyTorch for many small interacting modules is error-prone and hard to diff. Declarative specs are checkable: the compiler enforces shape contracts, the test suite (`tests/dsl/`, currently 620 tests) verifies that the generated `torch.nn.Module` produces byte-equivalent forward passes against a frozen reference, and a separate round-trip pass (`tests/test_dna_roundtrip_byte_identity.py`) confirms that bundling and re-loading a module set preserves byte identity.

**Folder layout:** `arch.neuro` (package config + top-level wiring), `modules/*.neuro` (per-region specs), `lib/` (shared building blocks). Import paths: `@/` = repo-root absolute, `./` = file-relative.

Reference: [`docs/dsl.md`](docs/dsl.md). Detailed architecture spec: [`docs/architecture.md`](docs/architecture.md).

---

## What the test suite actually verifies

The 1825 collected tests under `tests/` are **mechanism-level** unit and contract tests: they verify that named code paths produce the shapes, gradients, EMAs, and tensor values the spec calls for. They are *not* end-to-end demonstrations of language modelling or generalization. The distinction matters and is enforced in `CLAUDE.md` §13.

Representative mechanism tests (this list is the contract — each row points at code, not at a marketing claim):

| Mechanism | Test file | What the test pins |
|---|---|---|
| Bowtie + integrated-information estimator | `tests/test_phi.py` | Gaussian-MI MIP returns Φ > 0 for rank-coupled module outputs and Φ ≈ 0 for independent ones |
| Φ aux loss contributes a real gradient | `tests/test_brain_forward.py::test_phi_objective_increases_total_gradient` | `‖∂L/∂θ‖` measurably larger with Φ term than without on a fixed batch |
| Trophic / BDNF-style growth coupled to Φ | `tests/test_neurochem.py::test_trophic_phi_boosts_growth` | High-Φ pathways receive larger projection-kernel rank updates than low-Φ ones |
| Sheaf H¹ contradiction detection | `tests/test_narrative_memory.py::test_sheaf_contradiction_detection` | Conflicting facts produce a `SUPERSEDES` edge in the narrative graph |
| Trunk gradient detach prevents post-awakening divergence | `tests/test_stabilization.py` | A 5000-step training loop with the detach passes; without it, NaNs in the recorded reference trajectory |
| SymbolicHyperNeuron expression search | `tests/test_symbolic_unit.py` (36 tests) | Gumbel-softmax over `{identity, add, sub, mul, exp, sin, tanh}` selects two inputs + one op per unit; `expression_strings()` returns parseable formulae |
| NRCSTK metabolic pruning | `tests/test_nrcstk_metabolic.py` (24 tests) | Hinge-squared overshoot drives an activity EMA below the prune threshold and zero-masks the neuron in both forward and backward |
| `FitnessComposer` aggregation | `tests/test_fitness_composer.py` (19) + `tests/test_fitness_parser.py` (20) | `fitness { ... }` DSL block parses to `FitnessConfig`; composer produces `(total_loss, telemetry)` matching the legacy `AuxWeights` curve to within float tolerance |
| `cortex_pre_head_norm` cures GPT-2 anisotropy spike | `tests/training/test_cortex_pre_head_norm.py` (8) | Without the LayerNorm, measured CE at step 0 = 13.84 (> ln(50257)=10.82). With it, 10.82 ± 0.5 nats |
| Cortex→trunk KL distillation | `tests/training/test_cortex_distillation_and_gating.py::TestDistillation*` (11) | `L_total += λ_t · T² · KL(softmax(cortex.detach()/T) ‖ softmax(lm/T))`; gradient flows only into trunk |
| NT-mediated α gating | `tests/training/test_cortex_distillation_and_gating.py::Test*Inhibition* + TestEffectiveAlpha` (11) | Inhibition EMA rises when `cortex_loss_ema > lm_loss_ema`; `α_eff = α · (1 − inhibition)` → 0 as the trunk wins |
| `ImprovementGate` admission gate | `tests/verification/test_improvement_gate.py` (16) | Pure-Python Welch's *t* + Lentz continued-fraction incomplete beta agrees with scipy to within 1e-6; mutation accepted iff `p < α ∧ effect_size > min_effect` |
| `TheoryOfMindIR` stalk geometry | `tests/thsd/test_theory_of_mind_ir.py` (9) | Config validation for `d_belief`, `max_agents`, `belief_decay ∈ [0,1]`, `order ≥ 1`; stalk dim scales with recursion order |

**Run the full suite:** `py -3 -m pytest tests/ -v` (≈110 s on CPU, 1825 collected / 4 deselected on the current commit).

**What these tests do not prove.** A mechanism test passing means "this code path computes what the spec says." It does not mean the mechanism produces better language-model perplexity, better OOD generalization, more reliable causal inference, more stable identity over time, or anything resembling cognition. Those questions are unanswered.

---

## Generalization: what we have measured

This is the open empirical question. Results so far on WikiText-103-v1 held-out perplexity, all numbers from JSON artifacts in `results/`:

| Variant | Params | Train steps | train ppl | OOD ppl | gap_ratio | Artifact |
|---|---|---|---|---|---|---|
| Flat transformer baseline | 106.9 M | 80 000 | 66.0 | 404.0 | 6.12 | [`results/ood_baseline-80k_107M_step80000.json`](results/ood_baseline-80k_107M_step80000.json) |
| BRIAN trunk + recursive heads | 108.2 M | 5 000 | 216.5 | 1372.8 | 6.34 | [`results/ood_recursive_108M_step5000.json`](results/ood_recursive_108M_step5000.json) |
| BRIAN trunk + ReZero gates | 107.8 M | 7 000 | 258.8 | 1351.5 | 5.22 | [`results/ood_rezero-fixed_107M_step7000.json`](results/ood_rezero-fixed_107M_step7000.json) |
| BRIAN PCT trunk | 69.2 M | 4 000 | 400.9 | 1806.6 | 4.51 | [`results/ood_pct-30m_68M_step4000.json`](results/ood_pct-30m_68M_step4000.json) |

**Honest read of the table:**

1. **On absolute OOD perplexity, the flat-transformer baseline wins by 3–4×.** All four configurations are flagged "STRONG OVERFITTING" by the eval harness.
2. **On gap-ratio (`OOD_ppl / train_ppl`) the BRIAN PCT variant is lower** (4.51 vs 6.12). A lower gap-ratio at *much* higher absolute perplexity is consistent with the model not having learned enough yet — it does not by itself demonstrate better generalization.
3. **Compute is not matched.** The baseline got 11–20× more training steps. A matched-compute comparison (baseline evaluated at step 4 000–7 000) has not been run. Without it, the gap-ratio difference is not interpretable as architectural evidence.
4. **Cortex-fusion stability:** the multi-cortex 30M-P4 run previously diverged at init (CE = 13.84 > ln(50257) = 10.82, due to the GPT-2 anisotropy spike through the tied LM head). With `cortex_pre_head_norm` enabled it now starts at ≈10.82 and trains to step 10 000 without diverging ([`logs/analyzed/38469631.md`](logs/analyzed/38469631.md)). That is an *initialization fix*, not a generalization claim.

The full evidence ledger with every recorded run, including failed ones, lives in [`docs/findings.md`](docs/findings.md). Unrecorded or unverified hypotheses do not appear there.

---

## Multi-cortex fusion (frozen GPT-2 experts + bowtie trunk)

The `rcc_bowtie_30m_p4` preset can stack three frozen GPT-2 small "cortex" experts above the bowtie trunk and convex-combine their logits with the trunk's at the LM head. Three sub-mechanisms govern it; all are configurable in `arch.neuro` via `multi_cortex { ... }` and parsed into `MultiCortexConfig` (`neuroslm/dsl/training_config.py`). All three are off by default.

### 1. `cortex_pre_head_norm` — initial-loss prophylaxis

Frozen GPT-2 hidden states have an outlier dimension with std ≈ 24 (≈82× the median dimension's std). Projecting that through the tied LM head produces ±8.5 logit spikes, saturating the softmax and giving CE at step 0 = **13.84** — higher than the uniform-distribution baseline `ln(50257) = 10.82`. An `nn.LayerNorm(d_sem)` applied to the cortex projection before the tied head brings initial CE back to **10.82 ± 0.5 nats**. Measured in `scripts/diagnose_catastrophic_loss.py` and pinned by `tests/training/test_cortex_pre_head_norm.py` (8 tests).

### 2. KL distillation from cortex to trunk

When enabled, each step adds:

$$\mathcal{L}_{\text{KL}} = T^2 \cdot \mathrm{KL}\big(\mathrm{softmax}(\text{cortex}_{\text{logits}}/T) \,\big\|\, \mathrm{softmax}(\text{lm}_{\text{logits}}/T)\big)$$

with cortex logits detached (gradient only into the trunk). The mixing weight is a piecewise-linear ramp over the EMA gap between cortex and trunk losses:

$$\lambda_t = \lambda_{\max} \cdot \mathrm{clip}\!\left(\frac{\text{gap}_t - \text{floor}}{\text{ceiling} - \text{floor}},\; 0,\; 1\right)$$

so distillation is strong when the trunk lags the cortex and turns itself off once the trunk catches up. Defaults: `T=4.0`, `gap_floor=0.1`, `gap_ceiling=2.0`, `lambda_max=1.0`. Code: `BRIANHarness._distillation_lambda` and `_cortex_fusion_aux_step` in `neuroslm/harness.py`.

### 3. NT-style α gating

Fusion uses convex combination `logits = (1−α)·lm_logits + α·cortex_logits`. To allow the cortex to retire when the trunk surpasses it, α is modulated by an EMA-tracked inhibitory scalar:

$$\text{inhibition}_t = (1-\beta) \cdot \text{inhibition}_{t-1} + \beta \cdot \sigma\!\big((\text{cortex\_loss\_ema} - \text{lm\_loss\_ema}) / T_{\text{inh}}\big)$$

$$\alpha_{\text{eff}} = \alpha \cdot (1 - \text{inhibition}_t)$$

with `β = 0.05`, `T_inh = 1.0`. Code: `BRIANHarness._update_cortex_inhibition` and `_effective_alpha`.

### Telemetry

The per-step training log exposes the fusion state:

```
step 1234 | lm_loss 4.21 | cortex 4.18 4.31 4.09 | α_eff 0.42 inh 0.16 λ 0.31 kl 0.0089 lm_ema 4.55 cx_ema 4.22 | ...
```

### Round-trip check

```python
from neuroslm.dsl.training_config import parse_dsl_training_config
cfg = parse_dsl_training_config("architectures/rcc_bowtie/arch.neuro")
print(f"distill={cfg.multi_cortex.distillation_enabled} "
      f"inh={cfg.multi_cortex.inhibition_enabled} "
      f"λmax={cfg.multi_cortex.distillation_lambda_max} "
      f"T={cfg.multi_cortex.distillation_temperature}")
```

The three sub-mechanisms are independently togglable and default to `False`; pre-existing `.neuro` files without `distillation { ... }` or `inhibition { ... }` blocks compile and train identically to before. Whether enabling them improves the model's *outputs* (perplexity, downstream tasks, OOD gap) is a question the current training runs do not yet answer.

---

## Real-time DNA evolution loop (experimental)

The DSL compiler emits an intermediate "DNA" representation that can be patched between training steps. The loop is wired and individually unit-tested, but is not yet a load-bearing source of measured improvements — treat it as plumbing under test.

```python
from neuroslm.utils import init_evolution, EvolutionaryTrainingContext

with EvolutionaryTrainingContext("dna/base.dna", "checkpoints/") as ctx:
    harness = BRIANHarness(ctx.arch_path, resume_from=ctx.resume_step)

    for step in range(ctx.resume_step, 10000):
        loss = harness.train_step(batch)

        # Activity tracking is recorded automatically.
        # Mutation proposals are emitted at high surprise and gated
        # by ImprovementGate (Welch's t-test) before being applied.

        if step % 1000 == 0:
            harness.checkpoint_mutations()
```

What is *implemented and tested*:

- **DNA round-trip is byte-identical** (`tests/test_dna_roundtrip_byte_identity.py`).
- **`ImprovementGate`** (Welch's *t*-test admission, pure-Python incomplete-beta CDF) agrees with scipy to 1e-6 on synthetic input (`tests/verification/test_improvement_gate.py`, 16 tests).
- **Incremental patches** are produced and replayable across resumed runs.
- **Hot/cold path tracking** is wired into the harness and exposed in per-step telemetry.
- **Module bundler + source maps** (`neuroslm/compiler/module_bundler.py`) resolves DSL imports into a flat bundle preserving file/line origin for every node.

What is **not yet demonstrated**: that the evolution loop ever proposes a mutation that survives `ImprovementGate` *and* meaningfully improves OOD perplexity over a non-evolving control. That is an open experiment, not a result.

Pointers: [`docs/technical_report.md` §2.5](docs/technical_report.md), [`neuroslm/utils/colab.py`](neuroslm/utils/colab.py).

---

## Project Configuration (`brian.toml`)

A tiny TOML file at the repo root is the **single source of truth** for which architecture / DNA every training, deploy, and Colab script targets:

```toml
# brian.toml
[current]
arch = "architectures/rcc_bowtie"   # active architecture
dna  = ""                            # set to a .dna path for DNA-loop training

[nfg]
output = ".neuro/nfg.png"            # where `brian compile nfg --current` writes
format = "png"                       # png | svg | pdf | dot
engine = "dot"                       # dot | neato | sfdp | fdp | circo
```

| Script / command | Reads from |
|---|---|
| `scripts/vast_train_dsl_loop.sh` | `[current].arch` (env `ARCH` overrides) |
| `scripts/vast_train_dna_loop.sh` | `[current].dna`  (env `DNA` overrides) |
| `_deploy_train.py` | `[current].dna` if set, else `[current].arch` (env wins) |
| `colab_run.ipynb` cell 4 | `[current].arch` + `[current].dna` |
| `brian compile nfg --current` | `[current].arch` (or `[current].dna`), `[nfg].output` |

**Env-var overrides** (for CI / one-off runs): `BRIAN_ARCH`, `BRIAN_DNA`, `BRIAN_NFG_OUTPUT`, `BRIAN_NFG_FORMAT`, `BRIAN_NFG_ENGINE`. Legacy `ARCH=…` / `DNA=…` env vars still work in the shell scripts.

Contract is locked by 27 tests in [`tests/test_project_config.py`](tests/test_project_config.py).

---

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows; Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
# torch is intentionally not in requirements.txt — install the build that matches
# your accelerator, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU sanity run (small preset, a few thousand steps; not a real training run)
python -m neuroslm.train --preset small --steps 2000 --batch_size 4 --optimizer adamw

# GPU training using the architecture pinned in brian.toml
python -m neuroslm.train --preset xl --steps 100000 --batch_size 4 --device cuda

# Resume the latest checkpoint
python -m neuroslm.train --resume latest

# Vanilla-transformer ablation at matched non-embedding parameter count
python -m neuroslm.train --preset xl --baseline

# Interactive generation
python -m neuroslm.generate --prompt "Once upon a time"
```

The full Colab workflow (clone → ablation → full training → benchmarks) is in `colab_run.ipynb`.

---

## Checkpoints (Git LFS)

Training checkpoints live in `lfs_checkpoints/` and are tracked via Git LFS. A single `.pt` file is multi-GB, so a full `git pull` on a laptop can be very slow — and you usually don't need the binaries locally.

### Skip LFS downloads for this repo (recommended on laptops)

```bash
# In the repo root, run once:
git lfs install --local --skip-smudge
```

`git pull` will now fetch only the tiny pointer stubs (~130 B each). The repo metadata stays in sync, but `lfs_checkpoints/*.pt` become text stubs on disk.

### Pull a specific checkpoint when you need it

```bash
git lfs pull --include="lfs_checkpoints/neuroslm_xl_adamw_mix_800.pt"
# Or by glob — get all 800-step files:
git lfs pull --include="lfs_checkpoints/*_800.*"
```

### Pull every LFS file (re-hydrate the whole repo)

```bash
git lfs pull
```

### Turn skip-smudge off again for this repo

```bash
git lfs install --local --force          # re-enable smudge for this repo
git lfs pull                              # then fetch the binaries you want
```

### Make skip-smudge the global default

```bash
git lfs install --skip-smudge             # applies to every repo on this machine
```

After global skip-smudge, `git clone` of *any* LFS-tracked repo only downloads stubs by default; use `git lfs pull --include=...` to materialise specific files.

> Training on Colab/TPU/A100 uses `git lfs pull` explicitly inside the notebook (cell 2) so the runtime always has the latest checkpoint — skip-smudge on your laptop won't affect that.

---

## Parameter presets

These are configuration shapes, not measured parameter counts. The numbers below are the hyperparameters that go into the architecture — the resulting *trainable parameter count* depends on the architecture they instantiate (the default `rcc_bowtie` arch built with the `30m_p4` preset measures at 45.7 M total / ≈7.1 M non-embedding, as recorded above). Bigger presets have not all been independently measured at the same precision.

| Preset  | Approx. shape         | `d_hidden` | `d_sem` | `lang_layers` | `lang_ctx` |
|---------|-----------------------|------------|---------|---------------|------------|
| `tiny`  | CPU sanity            | 192        | 128     | 2             | 256        |
| `small` | CPU dev               | 384        | 256     | 4             | 512        |
| `medium`| T4-class              | 768        | 512     | 8             | 1024       |
| `large` | T4-class wider context| 384        | 256     | 8             | 1024       |
| `xl`    | A100-class            | 512        | 384     | 12            | 2048       |
| `xxl`   | multi-A100            | 4096       | 2048    | 32            | 4096       |

Pass `--baseline` for a vanilla-transformer ablation built at matched non-embedding count. If you need an exact parameter number for a specific preset, build it and measure:

```python
from neuroslm.dsl.preset_bridge import build_lm_from_preset
m = build_lm_from_preset('rcc_bowtie_30m_p4')
print(sum(p.numel() for p in m.parameters()))   # 45_678_720 on the current arch
```

---

## Loss composition

`brain.forward_lm` returns a `loss` tensor that is a weighted sum of the terms below. Naming follows the code; the biological labels are *names of code paths*, not claims about what the model is doing physiologically.

| Term                | Source                                                              | Weight        | Gating |
|---------------------|---------------------------------------------------------------------|---------------|--------|
| `lm_loss`           | scalar-gated cross-entropy on next-token (gate label: "mesolimbic") | `w_lm = 1.0`  | always on |
| `world_loss`        | MSE between predicted and target world-state embedding              | `w_world = 0.3` | × `_aux_w_scale` |
| `motor_loss`        | cross-entropy on speak/silent action target                         | `w_motor = 0.05` | × `_aux_w_scale` |
| `pred_coding_loss`  | per-layer next-layer-prediction (language stack)                    | `w_pred_coding = 0.1` | × `_aux_w_scale` |
| `rssm_kl`           | (optional) RSSM KL prior–posterior                                  | `w_kl_world = 0.1` | × `_aux_w_scale` |
| `cpc_loss`          | (optional) contrastive predictive coding                            | `w_cpc = 0.05` | × `_aux_w_scale` |
| `phi_loss`          | `-tanh(Φ/3)·3` from the Gaussian-MI MIP estimator                   | `w_phi = 0.02` | × `_aux_w_scale` |
| `novel_aux_loss`    | aggregate of opt-in novel-module aux losses                         | 0.05          | × `_aux_w_scale` |
| `cortex_kl_loss`    | `T²·KL(softmax(cortex.detach()/T) ‖ softmax(lm/T))` distillation    | `λ_t` (gap-ramped, max 1.0) | only when `distillation_enabled` and EMA gap > floor |

`_aux_w_scale ∈ [0.001, 1.0]` is a scheduling gate (`architecture.md` §6.4). Before `step < 5000` (label in code: "infancy") every aux loss is heavily suppressed so the LM gradient dominates while the model fits its first language-level representations. Once `step ≥ 5000 AND lm_loss < 7.5` (label: "awakening") the aux losses ramp linearly toward full strength. These are scheduling labels for a piecewise function, not developmental claims.

---

## Metrics & introspection

The training harness exposes a few introspection hooks. The names below are *what the code measures*, not validated cognitive constructs. Φ is the Gaussian-MI MIP estimator (`tests/test_phi.py`). "Personality vector" is a learned 384-d tensor that survives checkpoint reloads (`tests/test_cognitive_closure.py::test_autobiographical_personality_consistency`); it is *not* a claim that the model has a personality.

| Hook | Returns | What the value is |
|---|---|---|
| `model.intelligence_metrics.snapshot()` | dict | Current values of: Φ estimate, identity-vector drift, narrative-graph coherence score, causal-rule density, self-reference rate |
| `model.consciousness_metrics.per_tick()` | dict | Per-step recordings labelled γ / θ / α / Φ / coherence / ignition (these are *measurement names from the IIT-adjacent literature applied to this model's internal signals* — not validated neural correlates of consciousness) |
| `model.narrative_stack.query_rules()` | list[Rule] | Causal-pattern rules accumulated in the narrative store, with support counts |
| `model.personality_vector` | tensor(384) | The persistent identity embedding |

These are useful for ablation / debugging — they answer "did this code path produce a value", not "did the model become conscious".

---

## Running tests

```bash
py -3 -m pytest tests/                              # full suite (1825 collected, ≈110 s on CPU)
py -3 -m pytest tests/test_phi.py -v                # Φ estimator
py -3 -m pytest tests/test_narrative_memory.py -v   # narrative graph + sheaf contradiction
py -3 -m pytest tests/test_cognitive_closure.py -v  # identity persistence + survival hook
py -3 -m pytest tests/test_stabilization.py -v      # trunk-detach training-loop sanity
py -3 -m pytest tests/training/test_cortex_pre_head_norm.py -v               # cortex_pre_head_norm
py -3 -m pytest tests/training/test_cortex_distillation_and_gating.py -v     # KL distillation + α gating
py -3 -m pytest tests/verification/test_improvement_gate.py -v               # Welch's t-test admission gate
py -3 -m pytest tests/thsd/test_theory_of_mind_ir.py -v                      # TheoryOfMindIR stalk geometry
py -3 -m pytest tests/test_project_config.py -v                              # brian.toml contract (27 tests)
py -3 -m pytest tests/dsl/ -v                       # 620 DSL parser + codegen + byte-equivalence tests
```

Each test pins a named behaviour in the code — they are how the project keeps mechanisms from silently drifting. They are not capability demonstrations.

---

## Documentation & reproducibility

| Document | Contents |
|---|---|
| [`docs/findings.md`](docs/findings.md) | Running hypothesis ledger. Source of truth for what has actually been measured and what is still open. Negative results are kept; nothing is silently overwritten. |
| [`docs/architecture.md`](docs/architecture.md) | Architecture spec: forward pass, tensor shapes, equations, module diagrams. Reproducible to file + line. |
| [`docs/formal_framework.md`](docs/formal_framework.md) | Mathematical framework (THSD) the DSL and `ImprovementGate` are designed against. Includes the Lean-proof roadmap; most proofs are still scaffolds. |
| [`docs/technical_report.md`](docs/technical_report.md) | Plain-text executive summary, kept in sync with `findings.md` via `scripts/maintain_technical_report.py`. |
| [`docs/dsl.md`](docs/dsl.md) | DSL syntax, macro system, symbol resolution, compile pipeline, module bundling, source maps. |
| [`docs/BRAIN.md`](docs/BRAIN.md) | Deep-dive on `NeuralOrchestrator` and the re-entry-loop design rationale. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | TDD workflow, testing patterns, documentation sync rules. |
| [`CLAUDE.md`](CLAUDE.md) | Repo-wide rules for any AI agent working in the codebase (including the no-overclaiming rule that produced this README). |

**Reproducing what's here:**

```bash
# Verify the test contract on your machine
py -3 -m pytest tests/ -v

# CPU sanity training run (small preset)
python -m neuroslm.train --preset small --steps 2000

# GPU training run using the architecture pinned in brian.toml
python -m neuroslm.train --preset xl --steps 100000 --device cuda

# OOD eval reproduction recipes (which JSON in results/ comes from which command):
# see docs/findings.md
```

Full Colab workflow in [`colab_run.ipynb`](colab_run.ipynb).

---

## Status, in one paragraph

This is a research codebase. Mechanism-level behaviour is pinned by 1825 tests; architecture is reproducible from a typed DSL; the brian.toml contract makes "which architecture is the active one" unambiguous across training, deploy, NFG render, and Colab. What has *not* been shown is that any of this produces a language model that is better — at perplexity, at downstream tasks, or at OOD generalization — than a parameter-matched flat transformer at the same training budget. Anyone reading the code, the tests, or `docs/findings.md` should leave with the same picture. If a future commit changes that picture, the evidence will land in `docs/findings.md` *first* and the README will be updated to point at it.

If you want to discuss the design or contribute, open an issue or PR. This is open research, not a product.
