# NeuroSLM DSL Standard Library

This directory contains the canonical `.neuro` library for specifying transformer-based language models.

## Structure

```
lib/
├── primitives/           # Mathematical building blocks
│   ├── attention.neuro   # 8 attention mechanisms (MHA, GQA, MQA, RoPE, ALiBi, Flash, etc.)
│   ├── feedforward.neuro # 7 FFN variants (GELU, SwiGLU, GeGLU, MoE, etc.)
│   ├── normalization.neuro # Normalization strategies (LayerNorm, RMSNorm, pre/post-norm)
│   └── optimizers.neuro  # Optimizer equations (SGD, Adam, AdamW, Lion, schedules)
│
└── architectures/        # Complete model specifications
    ├── transformer.neuro # Canonical Vaswani 2017 architecture
    ├── gpt.neuro         # GPT-2, GPT-3, GPT-NeoX, GPT-4 (speculative)
    ├── llama.neuro       # LLaMA, LLaMA 2, LLaMA 3
    └── claude.neuro      # Claude (speculative, long-context architecture)
```

## Design Philosophy

### 1. Separation of Concerns

**Primitives** define **WHAT** (mathematical equations):
- `attention.neuro`: Scaled dot-product, causal masking, multi-head, grouped-query
- `feedforward.neuro`: Activation functions, gating mechanisms, MoE routing
- `normalization.neuro`: LayerNorm, RMSNorm, placement strategies

**Architectures** define **COMPOSITION** (how primitives combine):
- `gpt.neuro`: "Use causal MHA + GELU FFN + LayerNorm pre-norm"
- `llama.neuro`: "Use GQA + SwiGLU + RMSNorm pre-norm + RoPE"

**Compiler** handles **HOW** (implementation details):
- Flash Attention for memory efficiency
- Kernel fusion for speed
- Tensor parallelism for multi-GPU
- Mixed precision (FP16/BF16)

### 2. Math-First Specification

Each primitive is defined with:
- **Signature:** Input/output shapes
- **Equation:** Mathematical formula (LaTeX-style)
- **Where:** Variable definitions and constraints
- **Properties:** Complexity, memory, invariants

Example:
```neuro
export equation scaled_dot_product_attention {
    signature: "(Q, K, V) → output  where shapes = (batch, seq, d_model)"
    
    equation: """
        scores = (Q @ K^T) / √d_k
        weights = softmax(scores)
        output = weights @ V
    """
    
    where: {
        d_k: "key dimension (typically d_model / n_heads)"
    }
    
    properties: {
        complexity: "O(n² d_k) — quadratic in sequence length"
        memory: "O(n²) — attention matrix scales quadratically"
    }
}
```

### 3. Import-Based Composition

User architectures import from this library:

```neuro
# User's custom model
import { grouped_query_attention, swiglu_ffn, rms_norm } from "~/lib/primitives"
import { transformer_decoder } from "~/lib/architectures"

export architecture my_model {
    base: "transformer_decoder"
    
    primitives: {
        attention: "grouped_query_attention(n_kv_heads=8)"
        feedforward: "swiglu_ffn"
        normalization: "rms_norm"
    }
    
    config: {
        d_model: 4096
        n_layers: 32
        n_heads: 32
        vocab_size: 50257
    }
}
```

The compiler:
1. Resolves imports from `neuroslm/dsl/lib/`
2. Inlines equations and constraints
3. Generates optimized PyTorch code
4. Chooses implementation (Flash Attention, kernel fusion, etc.)

## Usage

### For Users

**Specify a model in <30 lines:**

```neuro
import { llama } from "~/lib/architectures"

export architecture my_llama {
    base: "llama"
    variant: "7B"  # Uses predefined 7B config from library
}
```

**Or customize primitives:**

```neuro
import { transformer_decoder } from "~/lib/architectures"
import { flash_attention, moe_ffn, rms_norm } from "~/lib/primitives"

export architecture my_efficient_model {
    base: "transformer_decoder"
    
    primitives: {
        attention: "flash_attention"  # Memory-efficient
        feedforward: "moe_ffn(n_experts=8, top_k=2)"  # Sparse capacity
        normalization: "rms_norm"  # Faster than LayerNorm
    }
    
    config: {
        d_model: 2048
        n_layers: 24
        context_length: 8192
    }
}
```

### For Library Authors

**Add a new primitive:**

1. Create equation in appropriate file (`primitives/attention.neuro`, etc.)
2. Export with `export equation <name>`
3. Include canonical equation, constraints, properties
4. Document complexity, memory, design notes

**Add a new architecture:**

1. Create file in `architectures/<name>.neuro`
2. Import required primitives
3. Specify composition and variants
4. Include comparison tables and design notes

## Examples in the Wild

### GPT-4 (Speculative)

From `architectures/gpt.neuro`:

```neuro
export architecture gpt4_speculative {
    primitives: {
        attention: "grouped_query_attention(n_kv_heads=8)"
        feedforward: "moe_ffn(n_experts=16, top_k=2)"
        normalization: "rms_norm"
    }
    
    estimated_config: {
        d_model: 16384
        n_layers: 120
        active_params: "~220B (only 2/16 experts active)"
    }
}
```

### LLaMA 3 405B

From `architectures/llama.neuro`:

```neuro
export architecture llama3 {
    primitives: {
        attention: "grouped_query_attention(n_kv_heads=8)"
        feedforward: "swiglu_ffn"
        normalization: "rms_norm"
        positional: "rotary_embeddings"
    }
    
    variants: {
        "405B": {
            d_model: 16384
            n_heads: 128
            n_kv_heads: 8
            n_layers: 126
            context_length: 128000  # RoPE scaling
        }
    }
}
```

## Compiler Integration

The DSL compiler (`neuroslm/dsl/multifile.py`) will:

1. **Parse** `.neuro` files with import statements
2. **Resolve** imports from `~/lib/` paths (mapped to `neuroslm/dsl/lib/`)
3. **Inline** equations and constraints into user architecture
4. **Validate** constraints (e.g., `n_heads` must divide `d_model`)
5. **Generate** optimized PyTorch `nn.Module` classes
6. **Choose** implementations (Flash Attention, kernel fusion) based on runtime conditions

## Roadmap

- [ ] Implement import resolution in `multifile.py`
- [ ] Add constraint validation pass
- [ ] Add equation inlining and optimization
- [ ] Extend primitives: linear attention, sparse patterns, adapter layers
- [ ] Add training protocol library (curriculum learning, few-shot, RLHF)
- [ ] Add data pipeline library (tokenization, packing, mixture sampling)

## Vision

**Goal:** Make specifying GPT-4 or Claude as simple as:

```neuro
import { gpt4_speculative } from "~/lib/architectures"

export architecture my_model {
    base: "gpt4_speculative"
    
    # Override just the config
    config: {
        d_model: 8192  # Smaller than GPT-4
        n_layers: 60
    }
}
```

The DSL handles:
- ✅ Mathematical correctness (equations are canonical references)
- ✅ Implementation efficiency (compiler chooses Flash Attention, etc.)
- ✅ Multi-GPU scaling (tensor parallelism, pipeline parallelism)
- ✅ Mixed precision (FP16/BF16/FP8)

The researcher focuses on:
- 🧪 Novel architecture ideas (new attention mechanisms, routing strategies)
- 📊 Training protocols (schedules, regularization, data mixtures)
- 🔬 Scientific hypotheses (testing architectural choices)

**Not** on:
- ❌ Implementing attention kernels
- ❌ Writing distributed training loops
- ❌ Debugging CUDA out-of-memory errors
