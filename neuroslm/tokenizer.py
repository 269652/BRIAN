"""Lightweight tokenizer wrapper using tiktoken's GPT-2 BPE.

Avoids pulling huggingface tokenizers as a hard dep.
"""
from __future__ import annotations
import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab
        self.eos_id = self.enc.eot_token  # 50256 for gpt2

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)
