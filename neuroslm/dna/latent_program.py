"""Latent Program Encoding — differentiable Lisp algorithm representations.

Converts Lisp algorithm files (.lisp) into dense latent embeddings where each
dimension encodes an opcode, operand, or control flow primitive. These latent
programs can then be:

  1. Interpreted/executed by a neural decoder at each module
  2. Optimized via gradient descent (the embedding IS the program)
  3. Mutated/crossed-over by the DNA evolution system
  4. Compared in embedding space (similar programs = nearby vectors)

Scientific basis:
  - Neural program induction (Reed & de Freitas, 2016)
  - Differentiable Forth / neural Turing machines
  - Program synthesis as continuous optimization (Gaunt et al., 2017)

Architecture:
  LispSource → Tokenizer → Embedding → LatentProgram (dense vector)
  LatentProgram → Decoder → OpcodeSequence → Interpreter → ModuleConfig

The latent program acts like DNA: a compressed, evolvable representation
of the algorithm that each brain region executes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# Opcode vocabulary: primitive operations that programs can perform
OPCODES = [
    'NOP',          # no-op
    'LOAD',         # load value from register
    'STORE',        # store value to register
    'ADD', 'SUB', 'MUL', 'DIV',  # arithmetic
    'RELU', 'SIGMOID', 'TANH',   # activations
    'CMP_GT', 'CMP_LT', 'CMP_EQ',  # comparisons
    'BRANCH',       # conditional jump
    'LOOP_START', 'LOOP_END',  # loop control
    'CALL',         # call subroutine
    'RETURN',       # return value
    'EMIT_NT',      # emit neurotransmitter signal
    'READ_NT',      # read neurotransmitter level
    'ATTEND',       # attention operation
    'PROJECT',      # linear projection
    'NORMALIZE',    # layer norm
    'GATE',         # multiplicative gate
    'RECALL',       # memory recall
    'WRITE_MEM',    # memory write
    'MODULATE',     # neuromodulatory gain
    'OSCILLATE',    # rhythmic signal generation
    'BIND',         # cross-modal binding
    'PREDICT',      # forward prediction
    'ERROR',        # prediction error signal
]
N_OPCODES = len(OPCODES)
OPCODE_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


# Lisp token vocabulary (simplified for encoding)
LISP_TOKENS = [
    '<PAD>', '<UNK>', '<BOS>', '<EOS>',
    '(', ')', 'define', 'defn', 'let', 'if', 'cond', 'lambda',
    'set!', 'begin', 'do', 'loop', 'for', 'while',
    'projection', 'nt_prod', 'region', 'layers', 'connections',
    'learning_rule', 'backprop', 'hebbian', 'reinforce',
    'score-candidate', 'score-action', 'query-memory',
    '+', '-', '*', '/', '>', '<', '=', 'and', 'or', 'not',
    'abs', 'max', 'min', 'sqrt', 'pow',
    'true', 'false', 'nil',
    'DA', 'NE', '5HT', 'ACh', 'GABA', 'Glu', 'eCB',
    'PFC', 'DMN', 'Hippo', 'BG', 'Thalamus', 'Language',
    'skip', 'dense', 'sparse', 'recurrent',
    '<NUM>', '<SYM>',
]
LISP_TOKEN_TO_IDX = {t: i for i, t in enumerate(LISP_TOKENS)}


def tokenize_lisp(source: str, max_len: int = 256) -> list[int]:
    """Tokenize Lisp source into integer indices."""
    # Remove comments
    lines = source.splitlines()
    cleaned = ' '.join(line.partition(';')[0] for line in lines)
    cleaned = cleaned.replace('(', ' ( ').replace(')', ' ) ')
    cleaned = cleaned.replace('[', ' ( ').replace(']', ' ) ')
    tokens_raw = cleaned.split()

    indices = [LISP_TOKEN_TO_IDX.get('<BOS>', 2)]
    for t in tokens_raw[:max_len - 2]:
        if t in LISP_TOKEN_TO_IDX:
            indices.append(LISP_TOKEN_TO_IDX[t])
        elif t.lstrip('-').replace('.', '').isdigit():
            indices.append(LISP_TOKEN_TO_IDX['<NUM>'])
        else:
            indices.append(LISP_TOKEN_TO_IDX.get('<SYM>', 1))
    indices.append(LISP_TOKEN_TO_IDX.get('<EOS>', 3))

    # Pad
    while len(indices) < max_len:
        indices.append(0)
    return indices[:max_len]


class LatentProgramEncoder(nn.Module):
    """Encodes Lisp source code into a dense latent program vector.

    The latent vector represents the algorithm in a continuous space
    where similar programs are nearby and the representation is
    differentiable.
    """

    def __init__(self, d_latent: int = 256, max_tokens: int = 256,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d_latent = d_latent
        self.max_tokens = max_tokens
        n_vocab = len(LISP_TOKENS)

        self.token_emb = nn.Embedding(n_vocab, d_latent, padding_idx=0)
        self.pos_emb = nn.Embedding(max_tokens, d_latent)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=n_heads, dim_feedforward=d_latent * 2,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool to single vector
        self.pool = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
            nn.Tanh(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) → latent: (B, D)"""
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        h = self.token_emb(token_ids) + self.pos_emb(positions)
        h = self.encoder(h)
        # Mean pool over non-padding positions
        mask = (token_ids != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.pool(pooled)

    def encode_source(self, source: str) -> torch.Tensor:
        """Encode a Lisp source string into a latent vector (no grad)."""
        ids = tokenize_lisp(source, self.max_tokens)
        ids_t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            return self.forward(ids_t).squeeze(0)


class LatentProgramDecoder(nn.Module):
    """Decodes a latent program vector into an opcode sequence.

    The decoded opcodes configure how a brain module processes its input.
    This is interpreted by the NeuralProgramInterpreter.
    """

    def __init__(self, d_latent: int = 256, max_steps: int = 32):
        super().__init__()
        self.d_latent = d_latent
        self.max_steps = max_steps

        # Decode latent → opcode logits + operand values per step
        self.step_proj = nn.Linear(d_latent, max_steps * (N_OPCODES + 4))
        # 4 extra dims per step: 2 operand indices + 1 scalar value + 1 gate

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """latent: (B, D) → opcodes: (B, S, N_OPS), operands: (B, S, 4)"""
        B = latent.size(0)
        raw = self.step_proj(latent)  # (B, S * (N_OPS + 4))
        raw = raw.view(B, self.max_steps, N_OPCODES + 4)
        opcode_logits = raw[:, :, :N_OPCODES]
        operands = raw[:, :, N_OPCODES:]
        return opcode_logits, operands


class NeuralProgramInterpreter(nn.Module):
    """Interprets a decoded opcode sequence to modulate a neural signal.

    Instead of executing opcodes literally (which isn't differentiable),
    we use the opcode distribution as a soft mixture of differentiable
    operations. This makes the program execution fully differentiable.

    The interpreter takes:
      - opcode_logits: (B, S, N_OPS) — soft program
      - operands: (B, S, 4) — parameters
      - signal: (B, D) — input neural signal

    And produces:
      - output: (B, D) — transformed signal
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learnable operation kernels (one per opcode)
        self.op_kernels = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.01)
            for _ in range(N_OPCODES)
        ])

    def forward(self, opcode_logits: torch.Tensor,
                operands: torch.Tensor,
                signal: torch.Tensor) -> torch.Tensor:
        """Execute the soft program on the signal."""
        B, S, _ = opcode_logits.shape

        # Soft opcode selection per step
        weights = F.softmax(opcode_logits, dim=-1)  # (B, S, N_OPS)

        h = signal  # (B, D)
        for step in range(min(S, 16)):  # limit execution steps
            # Weighted sum of operation kernels
            w = weights[:, step]  # (B, N_OPS)
            gate = torch.sigmoid(operands[:, step, 3:4])  # (B, 1)

            # Soft-mix all operation kernels
            kernel = torch.zeros(B, self.d_model, self.d_model,
                                 device=signal.device)
            for i, k in enumerate(self.op_kernels):
                kernel += w[:, i:i+1, None] * k.unsqueeze(0)

            # Apply: h = gate * (h @ kernel) + (1-gate) * h
            transformed = torch.bmm(h.unsqueeze(1), kernel).squeeze(1)
            h = gate * transformed + (1 - gate) * h

        return h


class LatentProgramDecompiler:
    """Decompiles decoded opcode sequences back to human-readable Lisp.

    After training, the latent programs have drifted from their initial
    Lisp source. This decompiler reconstructs readable Lisp from the
    learned opcode distributions, so every module's algorithm is
    *inspectable* — no black boxes.

    The decompiler works by:
      1. Taking opcode_logits (B, S, N_OPS) → hard argmax per step
      2. Reading operand values (register indices, scalars, gates)
      3. Emitting Lisp s-expressions for each instruction

    The output is *approximate* — a soft program with 60% GATE and 40%
    PROJECT at a step becomes (gate 0.6 (project ...)), but it captures
    the dominant learned behavior.
    """

    # Map opcodes to Lisp forms
    _LISP_FORMS: dict[str, str] = {
        'NOP':        '(nop)',
        'LOAD':       '(load r{src})',
        'STORE':      '(store r{dst} {val:.3f})',
        'ADD':        '(+ r{src} r{dst})',
        'SUB':        '(- r{src} r{dst})',
        'MUL':        '(* r{src} r{dst})',
        'DIV':        '(/ r{src} (max r{dst} 1e-6))',
        'RELU':       '(relu r{src})',
        'SIGMOID':    '(sigmoid r{src})',
        'TANH':       '(tanh r{src})',
        'CMP_GT':     '(> r{src} r{dst})',
        'CMP_LT':     '(< r{src} r{dst})',
        'CMP_EQ':     '(= r{src} r{dst})',
        'BRANCH':     '(if r{src} (goto {dst}))',
        'LOOP_START': '(loop-begin {val:.0f})',
        'LOOP_END':   '(loop-end)',
        'CALL':       '(call sub-{src})',
        'RETURN':     '(return r{src})',
        'EMIT_NT':    '(emit-nt {nt} {val:.3f})',
        'READ_NT':    '(read-nt {nt})',
        'ATTEND':     '(attend r{src} r{dst})',
        'PROJECT':    '(project r{src} :dim {dst})',
        'NORMALIZE':  '(layer-norm r{src})',
        'GATE':       '(gate r{src} :alpha {val:.3f})',
        'RECALL':     '(recall-memory r{src})',
        'WRITE_MEM':  '(write-memory r{src} r{dst})',
        'MODULATE':   '(modulate r{src} :gain {val:.3f})',
        'OSCILLATE':  '(oscillate :freq {val:.3f})',
        'BIND':       '(bind r{src} r{dst})',
        'PREDICT':    '(predict r{src} :horizon {dst})',
        'ERROR':      '(prediction-error r{src} r{dst})',
    }

    _NT_NAMES = ['DA', 'NE', '5HT', 'ACh', 'GABA', 'Glu', 'eCB']

    @classmethod
    def decompile_step(cls, opcode_idx: int, operands: list[float],
                       gate: float) -> str:
        """Decompile a single program step to a Lisp expression."""
        op_name = OPCODES[opcode_idx] if opcode_idx < len(OPCODES) else 'NOP'
        src = int(abs(operands[0]) * 8) % 8
        dst = int(abs(operands[1]) * 8) % 8
        val = operands[2]
        nt = cls._NT_NAMES[int(abs(val) * len(cls._NT_NAMES)) % len(cls._NT_NAMES)]

        template = cls._LISP_FORMS.get(op_name, f'({op_name.lower()})')
        try:
            expr = template.format(src=src, dst=dst, val=val, nt=nt)
        except (KeyError, IndexError):
            expr = f'({op_name.lower()} r{src} r{dst} {val:.3f})'

        # Wrap with gate if gate < 0.95 (i.e. not fully active)
        if gate < 0.95:
            expr = f'(blend {gate:.2f} {expr} identity)'
        return expr

    @classmethod
    def decompile(cls, opcode_logits: torch.Tensor,
                  operands: torch.Tensor,
                  region_name: str = 'unknown',
                  top_k: int = 2,
                  min_gate: float = 0.05) -> str:
        """Decompile a full program to Lisp source.

        Args:
            opcode_logits: (S, N_OPS) logits per step
            operands: (S, 4) per step
            region_name: name for the region header
            top_k: show top-k opcodes per step (for soft programs)
            min_gate: skip steps where gate < this (effectively NOPs)

        Returns:
            Human-readable Lisp source string
        """
        S = opcode_logits.size(0)
        probs = F.softmax(opcode_logits, dim=-1)  # (S, N_OPS)

        lines = [f';; Decompiled latent program for region: {region_name}',
                 f';; {S} steps, showing dominant ops (top-{top_k})',
                 f'(region {region_name}',
                 f'  (program']

        active_steps = 0
        for step in range(S):
            gate = torch.sigmoid(operands[step, 3]).item()
            if gate < min_gate:
                continue  # effectively a NOP, skip

            # Top-k opcodes for this step
            topk_probs, topk_idx = probs[step].topk(top_k)
            ops_raw = operands[step, :3].tolist()

            # Primary opcode
            primary_idx = topk_idx[0].item()
            primary_prob = topk_probs[0].item()
            primary_expr = cls.decompile_step(primary_idx, ops_raw, gate)

            if top_k > 1 and topk_probs[1].item() > 0.15:
                # Show secondary opcode as comment
                sec_idx = topk_idx[1].item()
                sec_prob = topk_probs[1].item()
                sec_name = OPCODES[sec_idx] if sec_idx < len(OPCODES) else '?'
                lines.append(
                    f'    {primary_expr}  '
                    f'; [{primary_prob:.0%} {OPCODES[primary_idx]}, '
                    f'{sec_prob:.0%} {sec_name}]')
            else:
                lines.append(f'    {primary_expr}  ; [{primary_prob:.0%}]')
            active_steps += 1

        lines.append(f'  )  ; {active_steps} active steps')
        lines.append(')')
        return '\n'.join(lines)


class LatentProgramSystem(nn.Module):
    """Complete system: encode Lisp → latent → decode → interpret.

    Each brain region gets a latent program that controls its behavior.
    Programs are stored as dense vectors and can be:
      - Initialized from Lisp source files
      - Optimized via backprop
      - Mutated by the DNA evolution system
      - Compared for similarity
    """

    def __init__(self, d_sem: int, d_latent: int = 128,
                 max_tokens: int = 256, max_steps: int = 16):
        super().__init__()
        self.d_sem = d_sem
        self.d_latent = d_latent

        self.encoder = LatentProgramEncoder(d_latent, max_tokens)
        self.decoder = LatentProgramDecoder(d_latent, max_steps)
        self.interpreter = NeuralProgramInterpreter(d_sem)

        # Program bank: named latent programs (one per brain region)
        self._programs: dict[str, nn.Parameter] = {}

    def register_program(self, name: str, source: str | None = None):
        """Register a named program, optionally initialized from Lisp source."""
        if source is not None:
            latent = self.encoder.encode_source(source)
        else:
            latent = torch.randn(self.d_latent) * 0.01
        param = nn.Parameter(latent)
        self._programs[name] = param
        # Register as a proper parameter
        setattr(self, f'prog_{name}', param)

    def execute(self, name: str, signal: torch.Tensor) -> torch.Tensor:
        """Execute a named program on a neural signal.

        Args:
            name: program/region name
            signal: (B, D) input

        Returns:
            (B, D) transformed output
        """
        if name not in self._programs:
            return signal  # passthrough if no program registered

        latent = self._programs[name].unsqueeze(0).expand(signal.size(0), -1)
        opcode_logits, operands = self.decoder(latent)
        return self.interpreter(opcode_logits, operands, signal)

    def program_similarity(self, name_a: str, name_b: str) -> float:
        """Cosine similarity between two programs."""
        if name_a not in self._programs or name_b not in self._programs:
            return 0.0
        a = self._programs[name_a]
        b = self._programs[name_b]
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def mutate(self, name: str, mutation_rate: float = 0.1):
        """Mutate a program's latent vector (for evolution)."""
        if name in self._programs:
            with torch.no_grad():
                noise = torch.randn_like(self._programs[name]) * mutation_rate
                self._programs[name].add_(noise)

    def crossover(self, name_a: str, name_b: str, child_name: str,
                  alpha: float = 0.5):
        """Create a child program by interpolating two parents."""
        if name_a in self._programs and name_b in self._programs:
            with torch.no_grad():
                a = self._programs[name_a]
                b = self._programs[name_b]
                child = alpha * a + (1 - alpha) * b
            self.register_program(child_name)
            with torch.no_grad():
                self._programs[child_name].copy_(child)

    # ---- Decompilation (latent → readable Lisp) ----

    def decompile(self, name: str, top_k: int = 2) -> str:
        """Decompile a named program back to human-readable Lisp.

        This is the key interpretability feature: after training, the
        latent vector has been optimized by gradient descent, and this
        method reveals *what algorithm it learned* in Lisp form.
        """
        if name not in self._programs:
            return f';; No program registered for "{name}"'

        latent = self._programs[name].unsqueeze(0)
        with torch.no_grad():
            opcode_logits, operands = self.decoder(latent)
        return LatentProgramDecompiler.decompile(
            opcode_logits.squeeze(0), operands.squeeze(0),
            region_name=name, top_k=top_k)

    def decompile_all(self, top_k: int = 2) -> dict[str, str]:
        """Decompile all registered programs to Lisp source.

        Returns a dict of {region_name: lisp_source_string}.
        """
        return {name: self.decompile(name, top_k=top_k)
                for name in self._programs}

    def save_decompiled(self, directory: str, top_k: int = 2):
        """Decompile all programs and write .lisp files to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        for name, source in self.decompile_all(top_k=top_k).items():
            path = os.path.join(directory, f'{name}_learned.lisp')
            with open(path, 'w') as f:
                f.write(source)
        return directory
