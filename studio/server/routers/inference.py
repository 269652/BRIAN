# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["inference"])

_REPO = Path(__file__).parent.parent.parent.parent


class InferRequest(BaseModel):
    prompt: str
    arch: str = "gpt2"
    max_new_tokens: int = 128
    temperature: float = 0.8
    checkpoint: str = ""  # HF URI or local path; empty = pretrained weights


@router.post("")
def run_inference(body: InferRequest) -> dict:
    """Run text generation via the model's pretrained weights or a checkpoint."""
    script = _build_infer_script(body)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=_REPO, timeout=120
    )
    return {
        "success": result.returncode == 0,
        "output": result.stdout,
        "error": result.stderr[-1000:] if result.stderr else "",
        "prompt": body.prompt,
        "arch": body.arch,
    }


def _build_infer_script(body: InferRequest) -> str:
    arch_dir = _REPO / "architectures" / body.arch
    neuro_file = arch_dir / "arch.neuro"
    # Determine if we can use HF transformers directly (pretrained GPT-2, etc.)
    if neuro_file.exists():
        import re
        src = neuro_file.read_text()
        kind_m = re.search(r"kind\s*:\s*(\w+)", src)
        weights_m = re.search(r'weights\s*:\s*"([^"]+)"', src)
        kind = kind_m.group(1) if kind_m else "custom"
        weights = weights_m.group(1) if weights_m else ""
    else:
        kind, weights = "custom", ""

    # For HF-backed models use transformers directly
    if weights.startswith("hf:") and not body.checkpoint:
        hf_id = weights[3:]
        return f"""
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = {hf_id!r}
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

prompt = {body.prompt!r}
inputs = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens={body.max_new_tokens},
        temperature={body.temperature},
        do_sample={str(body.temperature > 0)},
        pad_token_id=tok.eos_token_id,
    )
gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen, end="", flush=True)
"""

    # Checkpoint-based inference via brian chat
    ckpt_arg = body.checkpoint or "--latest"
    return f"""
import subprocess, sys
from pathlib import Path
result = subprocess.run(
    [sys.executable, "-m", "neuroslm.cli", "chat", {ckpt_arg!r}],
    input={body.prompt!r} + "\\n",
    capture_output=True, text=True,
    cwd={str(_REPO)!r},
    timeout=90,
)
print(result.stdout or result.stderr, end="", flush=True)
"""
