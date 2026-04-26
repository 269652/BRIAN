# NeuroSLM — A Neuroanatomically Inspired Small Language Model

A research prototype of a brain-inspired SLM with explicit modules for sensory cortices,
global workspace, DMN, hippocampus, PFC, basal ganglia, forward model, evaluator,
neuromodulators, and motor cortex.

## Status: Prototype / Stage 1

- ✅ All 15 modules implemented as PyTorch modules
- ✅ Cognitive (DMN) loop wired end-to-end
- ✅ Streaming training on Cosmopedia (open Phi-style dataset)
- ⚠️ Sized to be CPU-trainable (~15M params default). Scale via `config.py`.
- ⚠️ Not yet competitive with production SLMs — this is research scaffolding.

## Layout

```
neuroslm/
  config.py            # All hyperparameters & dimensions
  tokenizer.py         # BPE wrapper (uses tiktoken / GPT-2 vocab by default)
  modules/
    sensory.py         # Sensory cortices (modality encoders)
    association.py     # Multi-modal fusion
    language.py        # Language cortex (Wernicke + Broca)
    world_model.py     # Recurrent world-state SSM
    self_model.py      # Agent self-state SSM
    workspace.py       # Global Workspace (broadcast bus)
    dmn.py             # Default Mode Network orchestrator
    hippocampus.py     # DG/CA3/CA1 episodic memory + novelty
    pfc.py             # Prefrontal cortex (selection + gating)
    basal_ganglia.py   # Action gating (Go/NoGo, dopamine-modulated)
    forward_model.py   # Cerebellum-like next-state predictor
    evaluator.py       # ACC/OFC value head
    motor.py           # Motor cortex (token decoder)
    neuromods.py       # DA / NE / 5HT / ACh dynamics
  brain.py             # Wires all modules; runs DMN cognitive loop
  train.py             # Streaming pretraining on Cosmopedia
  generate.py          # Interactive REPL (feed text, observe output)
data/                  # Cached HF datasets
checkpoints/           # Model snapshots
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m neuroslm.train --steps 2000 --batch_size 4
python -m neuroslm.generate --prompt "Once upon a time"
```

See `docs/architecture.md` for the design rationale.
