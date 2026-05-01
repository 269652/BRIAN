"""Thought Transformer: meta-learnable reasoning amplifier.

Sits between the DMN floating thought and the language cortex output.
Transforms the neural flow of floating thought through a series of
self-attention + cross-attention layers to amplify:
  - Causal reasoning chains
  - Analogical transfer
  - Abstract pattern extraction
  - Logical consistency checking

The key innovation: this layer is META-LEARNED — its parameters are
updated by the learned optimizer based on comprehension/reasoning signals,
not just LM loss. This allows it to develop reasoning strategies that
emerge from the meta-learning objective.

Architecture:
  1. Thought Encoder: projects floating_thought + GWS slots into reasoning space
  2. Reasoning Blocks (N layers): self-attention among thought tokens +
     cross-attention to language hidden states (grounding)
  3. Consistency Head: detects logical contradictions
  4. Abstraction Head: extracts reusable patterns
  5. Output Projection: back to d_sem for downstream use

The layer operates at ~10Hz (once per cognitive tick) and its output
replaces/modulates the floating thought before it enters the GWS.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThoughtBlock(nn.Module):
    """Single reasoning block: self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, thought_tokens: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        """thought_tokens: (B, N_thought, D), context: (B, T_ctx, D)"""
        # Self-attention among thought tokens (internal reasoning)
        x = thought_tokens
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)

        # Cross-attention to language context (grounding in evidence)
        if context is not None:
            x2 = self.norm2(x)
            x2, _ = self.cross_attn(x2, context, context)
            x = x + self.dropout(x2)

        # FFN (nonlinear transformation / reasoning step)
        x2 = self.norm3(x)
        x = x + self.dropout(self.ffn(x2))
        return x


class ThoughtTransformer(nn.Module):
    """Meta-learnable thought transformation layer.

    Takes floating_thought + GWS slots + qualia and produces:
      - transformed_thought: enhanced reasoning representation
      - consistency_score: how logically coherent the current thought is
      - abstraction: reusable pattern extracted from current reasoning
      - reasoning_depth: how many effective "hops" of reasoning occurred
    """

    def __init__(self, d_sem: int, n_layers: int = 4, n_heads: int = 8,
                 n_thought_tokens: int = 8):
        super().__init__()
        self.d_sem = d_sem
        self.n_thought_tokens = n_thought_tokens

        # Learnable thought token embeddings (like a reasoning scratchpad)
        self.thought_tokens = nn.Parameter(
            torch.randn(1, n_thought_tokens, d_sem) * 0.02)

        # Project inputs into thought space
        self.input_proj = nn.Linear(d_sem, d_sem)
        self.gws_proj = nn.Linear(d_sem, d_sem)
        self.qualia_proj = nn.Linear(d_sem, d_sem)

        # Reasoning blocks
        self.blocks = nn.ModuleList([
            ThoughtBlock(d_sem, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Output heads
        self.output_proj = nn.Linear(d_sem, d_sem)
        self.output_gate = nn.Sequential(
            nn.Linear(d_sem * 2, d_sem),
            nn.Sigmoid(),
        )

        # Consistency head: detects contradictions in reasoning
        self.consistency_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2), nn.GELU(),
            nn.Linear(d_sem // 2, 1), nn.Sigmoid(),
        )

        # Abstraction head: extracts reusable pattern
        self.abstraction_head = nn.Sequential(
            nn.Linear(d_sem, d_sem // 2), nn.GELU(),
            nn.Linear(d_sem // 2, d_sem),
        )

        # Reasoning depth estimator (how many layers actually contributed)
        self.depth_head = nn.Sequential(
            nn.Linear(d_sem, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

        # Layer-wise contribution gates (meta-learned: which layers matter)
        self.layer_gates = nn.Parameter(torch.ones(n_layers) * 0.5)

    def forward(self, floating_thought: torch.Tensor,
                gws_slots: torch.Tensor | None = None,
                qualia: torch.Tensor | None = None,
                lang_hidden: torch.Tensor | None = None) -> dict:
        """
        Args:
            floating_thought: (B, D) current floating thought
            gws_slots: (B, N_slots, D) GWS broadcast content
            qualia: (B, D) qualia embedding
            lang_hidden: (B, T, D) language cortex hidden states for grounding

        Returns dict:
            transformed_thought: (B, D) — enhanced thought for downstream
            consistency: (B,) — logical coherence score [0,1]
            abstraction: (B, D) — extracted pattern
            reasoning_depth: (B,) — effective depth [0,1]
            thought_tokens: (B, N, D) — full reasoning scratchpad state
        """
        B = floating_thought.size(0)
        device = floating_thought.device

        # Initialize thought tokens: learnable base + projected inputs
        tokens = self.thought_tokens.expand(B, -1, -1).clone()

        # Inject floating thought into first token
        tokens[:, 0] = tokens[:, 0] + self.input_proj(floating_thought)

        # Inject qualia into second token (subjective experience context)
        if qualia is not None:
            tokens[:, 1] = tokens[:, 1] + self.qualia_proj(qualia)

        # Inject GWS slots into remaining tokens
        if gws_slots is not None:
            n_inject = min(gws_slots.size(1), self.n_thought_tokens - 2)
            for i in range(n_inject):
                tokens[:, 2 + i] = tokens[:, 2 + i] + self.gws_proj(gws_slots[:, i])

        # Run reasoning blocks with layer-wise gating
        layer_gates = torch.sigmoid(self.layer_gates).to(device)
        for i, block in enumerate(self.blocks):
            residual = tokens
            tokens = block(tokens, context=lang_hidden)
            # Gated residual: meta-learned how much each layer contributes
            tokens = residual + layer_gates[i] * (tokens - residual)

        # Extract outputs from thought tokens
        # Use mean-pool of all tokens for global reasoning state
        reasoning_state = tokens.mean(dim=1)  # (B, D)

        # Gated output: blend original thought with transformed
        gate_input = torch.cat([floating_thought, reasoning_state], dim=-1)
        gate = self.output_gate(gate_input)
        transformed = gate * self.output_proj(reasoning_state) + (1 - gate) * floating_thought

        # Auxiliary heads
        consistency = self.consistency_head(reasoning_state).squeeze(-1)
        abstraction = self.abstraction_head(reasoning_state)
        reasoning_depth = self.depth_head(reasoning_state).squeeze(-1)

        return {
            "transformed_thought": transformed,
            "consistency": consistency,
            "abstraction": abstraction,
            "reasoning_depth": reasoning_depth,
            "thought_tokens": tokens,
        }
