# cmd — Cross-Module Disagreement

**Category:** regularizer  
**Implementation:** `neuroslm.regularizers.CMDLoss`  
**DSL spec:** [`mechanics/cmd.neuro`](../../mechanics/cmd.neuro)

## Overview

CMD penalises the Jensen-Shannon divergence between two prediction heads, driving them to produce complementary distributions. By maximising disagreement, each head is forced to specialise on different aspects of the task (e.g. LM head on fluency, narrative head on topic consistency).

## Equation

```
p_a = softmax(W_a · h_a),  p_b = softmax(W_b · h_b)
m   = 0.5·(p_a + p_b)
JSD = 0.5·KL(p_a‖m) + 0.5·KL(p_b‖m)    ∈ [0, ln 2]
L_cmd = −weight · JSD    # negative: MAXIMISE divergence
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | `0.05` | Scale on −JSD |
| `divergence` | `"jsd"` | Divergence: `"jsd"`, `"kl_sym"`, or `"l1"` |
| `heads` | `["lm", "narrative"]` | Names of the two prediction heads |

## When to Use

When the model has multiple prediction heads and you want them to specialise. Use JSD (bounded) rather than KL (unbounded) for numerical stability.

## When NOT to Use

- Single-head architecture
- Large batch + long sequence: two full V-size softmaxes → OOM at batch=16, seq=2048 (disabled for this reason in master arch)
- Heads over different vocabularies (JSD is meaningless across token spaces)

## Properties

- **JSD bounded**: JSD ∈ [0, ln 2]; never diverges
- **Complementarity**: maximising JSD forces head specialisation
- **OOM risk**: full softmax over V at each token per head

## Empirical Evidence

Disabled in master arch (cmd: {enabled: false}) due to OOM at batch=16/seq=2048. Not yet ablated.

## References

- Internal: `neuroslm/regularizers.py::CMDLoss`
