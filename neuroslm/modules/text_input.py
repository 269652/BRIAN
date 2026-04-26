"""Text Input Cortex.

A *separate* sensory boundary for typed/written external text — distinct
from acoustic input (speech) and visual input (vision/reading-pixels).

Biologically: roughly the visual word-form area (VWFA) plus the gateway
into Wernicke's area; it converts an exteroceptive text stream into
token IDs that the language cortex can comprehend, while also injecting
a salience signal into LC (NE) so the brain's attention is captured by
fresh external input.

This is intentionally a *runtime* helper rather than an nn.Module that
needs training. The language cortex already handles all the heavy
encoding work; the role of this cortex is only:
  - own a token buffer of the most recent external input,
  - decay salience over ticks so old input fades from attention,
  - expose an NE surge when new input arrives,
  - hand its tokens to the cognitive loop as the next context window.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import torch

from ..tokenizer import Tokenizer


@dataclass
class TextInputCortex:
    tokenizer: Tokenizer
    max_buffer: int = 256
    decay: float = 0.85          # per-tick salience decay
    surge_amount: float = 0.8    # NE surge injected when fresh input arrives
    _buf: deque = field(default_factory=lambda: deque(maxlen=4096))
    salience: float = 0.0
    pending_surge: float = 0.0
    _new_since_last_tick: bool = False

    def __post_init__(self):
        self._buf = deque(maxlen=4096)

    # ----------------------------------------------------------------
    def receive(self, text: str) -> int:
        """Push raw text from the outside world into the buffer.
        Returns number of tokens added.
        """
        if not text:
            return 0
        ids = self.tokenizer.encode(text)
        # Add a separator so the model knows a fresh utterance arrived.
        # Newline acts as a soft delimiter in our streaming text data.
        sep = self.tokenizer.encode("\n")
        for t in ids + sep:
            self._buf.append(t)
        self.salience = 1.0
        self.pending_surge = self.surge_amount
        self._new_since_last_tick = True
        return len(ids)

    def emit(self, token_id: int) -> None:
        """When the language cortex emits a token (model speaking), echo it
        into our own buffer so the next tick sees it as context. This is
        the analog of efference copy / hearing yourself speak."""
        self._buf.append(int(token_id))

    # ----------------------------------------------------------------
    def context_ids(self, ctx_len: int, device: torch.device) -> torch.Tensor:
        """Return the current context window as a (1, T) tensor."""
        if len(self._buf) == 0:
            # Empty world — feed a single newline so the model has something.
            seed = self.tokenizer.encode("\n")
            return torch.tensor([seed], dtype=torch.long, device=device)
        ids = list(self._buf)[-ctx_len:]
        return torch.tensor([ids], dtype=torch.long, device=device)

    # ----------------------------------------------------------------
    def step(self) -> dict:
        """Advance one tick: decay salience, return any pending NE surge."""
        info = {
            "salience": self.salience,
            "ne_surge": self.pending_surge,
            "new_input": self._new_since_last_tick,
            "buffer_len": len(self._buf),
        }
        self.salience *= self.decay
        self.pending_surge = 0.0
        self._new_since_last_tick = False
        return info
