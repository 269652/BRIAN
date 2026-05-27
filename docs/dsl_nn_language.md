# NeuroTensor DSL — general neural-network description language (redesign)

## Why redesign

The original `.neuro` DSL modeled **per-token population dynamics**:
each population mapped `(batch, d_sem) → (batch, d_sem)`. That's fine for
a cognitive-overlay simulation, but it **cannot express sequence models**
— attention needs the whole `(batch, seq, dim)` tensor so each position
attends to prior ones. To train anything that matches `Brain`
(transformer trunk + cognitive modules), the language itself must be able
to describe arbitrary tensor computation: any model from a tiny SLM to a
full LLM.

This document specifies the redesigned, tensor-first DSL. Goal: you can
write `TransformerBlock`, `LanguageCortex`, the bowtie modules — anything
— and the compiler lowers it to PyTorch that is **bit-identical** to the
hand-written reference.

## Guiding design principle — everything is a formal mathematical object

The long-term goal (see Phase N10–N11 below) is to compile a complete SLM
— every ML op, every BRIAN subsystem, every gradient path, the memory and
neurotransmitter systems — into a **single multidimensional mathematical
representation** (a "hypershape") that can be inspected geometrically and
analyzed with graph-theory algorithms, to optimize the model's
*intelligence density*, reasoning capability, integrated information (Φ /
IIT) and effective information (EI).

That goal constrains the language design **now**, not later. Concretely
the IR is built so it is, by construction:

1. **A typed directed graph.** Every statement is a node; every tensor
   dependency is an edge typed by its shape. Forward computation is the
   graph; the **gradient flow is its transpose (adjoint) graph** —
   reverse-mode AD is a well-defined dual, so gradient paths are
   first-class and inspectable, not hidden in autograd.
2. **Formally-semantic ops.** Each built-in op has a closed-form
   mathematical definition (the lowering table is the spec), so the whole
   model is a composition of known functions — differentiable, and
   amenable to symbolic analysis (Jacobians, fixed points, spectral
   properties) the way the Phase-7 equation layer already does for scalar
   dynamics.
3. **No opaque control flow in the hot path.** Loops over layers unroll to
   explicit subgraphs; conditionals (maturation gating, MoD routing) are
   represented as typed gates with both branches present in the graph, so
   the structure is statically analyzable.
4. **Subsystems are subgraphs with declared boundaries.** Each BRIAN
   subsystem (memory, NT system, vesicle pool, trophic) is a named region
   of the graph with explicit input/output ports — so the hypershape can
   be decomposed, and metrics like Φ (which require a bipartition over
   well-defined parts) are computable directly from the IR.

Every later design decision defers to this: if a feature can't be
expressed as a typed node in a differentiable graph with formal op
semantics, it doesn't go in the language.

## Core model

A program is a **computation graph** over **shape-typed tensors**, not a
set of scalar equations. Three declaration kinds:

```
tensor x: (B, T, D)              # runtime input, symbolic dims
param  W: (D, H) init=xavier     # learnable nn.Parameter
const  eps = 1e-6                # compile-time scalar
```

Statements are SSA-style typed tensor ops:

```
h = rmsnorm(x, gamma)            # (B,T,D)
q = linear(h, Wq)                # (B,T,H)
a = causal_attention(q, k, v, heads=8)
y = x + a                        # residual
```

## Layers and models

Reusable parameterized sub-graphs:

```
layer TransformerBlock(D: int, n_heads: int, max_ctx: int) {
    param gamma1: (D,) init=ones
    param gamma2: (D,) init=ones
    param Wq: (D, D) init=xavier
    param Wkv: (D, 2*D) init=xavier
    param Wo: (D, D) init=xavier
    sublayer mlp: SwiGLU(D)

    forward(x: (B, T, D)) -> (B, T, D) {
        a = causal_self_attention(rmsnorm(x, gamma1),
                                  Wq, Wkv, Wo, heads=n_heads,
                                  rope=true, qk_norm=true)
        x = x + a
        m = mlp(rmsnorm(x, gamma2))
        return x + m
    }
}

model LanguageCortex(vocab: int, D: int, depth: int) {
    param embed: (vocab, D) init=normal(0, 0.02)
    layers blocks: TransformerBlock(D, n_heads=8, max_ctx=2048) * depth
    param gamma_f: (D,) init=ones
    param lm_head: (D, vocab) init=xavier

    forward(ids: (B, T)) -> (B, T, vocab) {
        h = embed[ids]                 # gather → (B,T,D)
        for blk in blocks: h = blk(h)
        h = rmsnorm(h, gamma_f)
        return linear(h, lm_head)
    }
}
```

## Built-in op library (each lowers to exact torch, each exact-match tested)

| Op | Lowering | Reference |
|---|---|---|
| `linear(x, W)` | `x @ W` (or `F.linear` w/ W.T) | nn.Linear(bias=False) |
| `embedding(ids, table)` | `table[ids]` | nn.Embedding |
| `rmsnorm(x, g, eps)` | `x * (x.pow(2).mean(-1,keepdim).add(eps).rsqrt()) * g` | common.RMSNorm |
| `swiglu(x, w1,w2,w3)` | `w3(silu(x@w1) * (x@w2))` | common.SwiGLU |
| `silu(x)` / `gelu` / `relu` | `F.silu` / `F.gelu` / `F.relu` | torch |
| `softmax(x, dim)` | `F.softmax` | torch |
| `rope(x, cos, sin)` | interleaved rotate-half | common.apply_rope |
| `causal_self_attention(...)` | RoPE + QK-norm + GQA + SDPA(causal) | common.CausalSelfAttention |
| `layernorm(x, g, b, eps)` | `F.layer_norm` | nn.LayerNorm |
| matmul / transpose / reshape / scale / add / mul | torch native | torch |

## Shape system

- Symbolic dims (`B`, `T`, `D`, `H`) are bound at runtime from inputs.
- The compiler checks shape compatibility statically where it can
  (matmul inner dims, residual adds), and emits a clear error otherwise.
- `param` shapes are concrete (resolved from layer args), so nn.Parameters
  are allocated with exact sizes.

## Execution / codegen

`model … { forward(ids) {…} }` compiles to an `nn.Module` subclass:
- `param` → `nn.Parameter` (init per `init=` spec)
- `sublayer` / `layers` → child modules
- the forward block → straight-line PyTorch in `forward()`
- generated source is `ast.parse`-validated then `exec`'d (same pipeline
  as the current codegen)

The whole-sequence tensor flows end-to-end — no per-token flattening.

## Exact-match testing discipline

Every built-in op and every composed layer ships a test of the form:

```python
def test_<op>_matches_reference():
    ref = neuroslm.modules.common.<RefModule>(...)
    dsl = compile_dsl_layer("<op or layer in DSL>")
    sync_weights(dsl, ref)                  # copy params by name
    x = torch.randn(B, T, D)
    assert torch.allclose(dsl(x), ref(x), atol=1e-6)
```

This is the gate: **no op or layer merges unless it is bit-identical to
the PyTorch reference it claims to implement.** The end state is a DSL
`LanguageCortex` whose forward + backward match `Brain`'s exactly, then
the bowtie cognitive modules layered on, all the way to full
`Brain(rcc_bowtie_30m_p4)` equivalence verified on a real-data training
run before any vast spend.

## Phased build (each phase = green exact-match tests)

| Phase | Delivers | Gate |
|---|---|---|
| **N1** | op library atoms: linear, embedding, rmsnorm, swiglu, silu | each `allclose` vs reference |
| **N2** | RoPE + causal_self_attention (GQA, QK-norm) | `allclose` vs common.CausalSelfAttention |
| **N3** | tensor-graph parser + `layer`/`model`/`forward` syntax | parse → codegen → run |
| **N4** | TransformerBlock + LanguageCortex in DSL | `allclose` vs modules.language.LanguageCortex |
| **N5** | real-data loader + LM training loop on DSL LanguageCortex | loss curve matches Brain trunk |
| **N6** | bowtie cognitive modules (the existing .neuro) layered onto trunk | module-by-module `allclose` |
| **N7** | NT-modulation, Hebbian, vesicle, trophic subsystems | each `allclose` vs Brain subsystem |
| **N8** | full Brain(rcc_bowtie_30m_p4) equivalence | end-to-end training-curve parity |
| **N9** | vast 10k run on DSL — matches p4 | benchmark parity |
| **N10** | **hypershape compiler** — lower the typed IR (forward graph + adjoint gradient graph + subsystem regions) to a multidimensional geometric representation | graph round-trips; node/edge/shape metadata preserved |
| **N11** | **geometric / graph-theoretic analysis toolkit** — inspect the hypershape: Φ/IIT bipartition search, EI estimation, intelligence-density metrics, spectral + centrality analysis of the computation/gradient graphs, visual rendering | metrics computable from IR; visualization renders; optimization hooks surface candidate edits |

### Future feature (N10–N11) — the mathematical "hypershape"

The end state: compile the complete RCC-bowtie-P4 SLM into one
multidimensional mathematical object and *reason about it*:

- **What it is.** A typed multigraph: nodes = ops/parameters/state,
  edges = shaped tensor flows, with a parallel adjoint graph for
  gradients and labeled subgraph regions per BRIAN subsystem (memory,
  NT, vesicle, trophic, bowtie stages).
- **What we do with it.** Run graph-theory + spectral algorithms to
  measure and optimize intelligence density, reasoning capacity, Φ (IIT
  integrated information), and EI (effective information) — by exploring
  the mathematical model of every mechanism the SLM implements, finding
  bottlenecks/cut-vertices, redundant paths, low-Φ partitions, and
  proposing structural edits that raise integration.
- **Why the design supports it already.** Per the guiding principle
  above, the IR is born as a differentiable typed graph with formal op
  semantics and explicit subsystem boundaries — exactly the structure
  these analyses require. N10 is a *lowering* of that IR to a geometric
  form, not a re-derivation.

Starting with **N1** now — the op atoms with exact-match tests against
`neuroslm.modules.common`.
