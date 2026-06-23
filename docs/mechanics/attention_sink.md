# attention_sink — Summary

**Category:** attention  
**DSL spec:** [`mechanics/attention_sink.neuro`](../../mechanics/attention_sink.neuro)

## Overview

Always retains the first `n_sink` tokens in the KV cache alongside a rolling window, stabilising softmax and enabling infinite-length streaming. Sink tokens absorb surplus probability mass that softmax must place somewhere even when no key is relevant.

## Equation

`keep(i) = {0, 1, ..., n_sink−1} ∪ {i−w+1, ..., i}`  
Mask is 0 for keys in `keep(i)`, −∞ otherwise. Standard scaled-dot-product attention proceeds on the unmasked set.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_sink_tokens` | 4 | Initial tokens always kept as global anchors; 4 captures most sink mass |
| `window_size` | 1024 | Size of the rolling recent-token window kept alongside the sinks |

## When to Use / When NOT to Use

**Use when:** streaming / never-ending generation with unbounded context; sliding-window attention causes perplexity spike on early-token eviction; constant memory and stable quality without retraining.

**Avoid when:** fixed-length fully-cached contexts (no eviction); genuine recall of evicted middle tokens is needed; encoder/bidirectional models.

## References

- Xiao, Tian, Chen, Han, Lewis (2023) Efficient Streaming Language Models with Attention Sinks. arXiv 2309.17453
- Han, Xiao, Tsvetkov, Han et al. (2023) LM-Infinite: Simple On-the-Fly Length Generalization. arXiv 2308.16137
