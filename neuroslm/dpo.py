"""Simple DPO-style preference optimization skeleton.

This module provides a minimal DPO training loop scaffold. It's not a full
production RLHF system but gives a starting point for training with pairwise
preferences. It assumes you have pairs (a, b, label) where label=1 if a preferred.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import List, Tuple


class PreferencePairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int], int]]):
        # pairs: (ids_a, ids_b, label) label=1 if a preferred else 0
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    # Pad sequences and return lengths for each side
    as_, bs, labs = zip(*batch)
    la = max(len(x) for x in as_)
    lb = max(len(x) for x in bs)
    import torch
    def pad_and_lens(xs, L):
        out = [x + [0] * (L - len(x)) for x in xs]
        lens = [len(x) for x in xs]
        return torch.tensor(out, dtype=torch.long), torch.tensor(lens, dtype=torch.long)
    a_p, a_lens = pad_and_lens(as_, la)
    b_p, b_lens = pad_and_lens(bs, lb)
    return a_p, a_lens, b_p, b_lens, torch.tensor(labs, dtype=torch.float)


def dpo_loss(logp_a, logp_b, epsilon=1.0):
    # DPO objective (simplified): minimize -log sigmoid((logp_a - logp_b) / epsilon)
    # where higher logp for preferred sample is encouraged.
    return -torch.log(torch.sigmoid((logp_a - logp_b) / epsilon) + 1e-12).mean()


def train_dpo(model, tokenizer, pairs, device='cuda', epochs=1, batch_size=8):
    ds = PreferencePairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    # Only train adapter (LoRA) params if present; otherwise train all params
    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=1e-5)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for a_p, a_lens, b_p, b_lens, labs in dl:
            a_p = a_p.to(device)
            b_p = b_p.to(device)
            a_lens = a_lens.to(device)
            b_lens = b_lens.to(device)
            labs = labs.to(device)

            # Compute log-prob of sequences under the model
            # Expect model to implement `lm_logprob(ids, lengths)` returning (B,) log-probs
            logp_a = model.lm_logprob(a_p, lengths=a_lens)
            logp_b = model.lm_logprob(b_p, lengths=b_lens)

            loss = dpo_loss(logp_a, logp_b)
            optim.zero_grad()
            loss.backward()
            optim.step()

    return model
