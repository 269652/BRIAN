Running preference tuning (DPO) on Colab (14GB)

Overview
- Use quantization (bitsandbytes) + LoRA adapters to fit larger effective models on 14GB.
- Run supervised fine-tuning (SFT) first, then collect preference pairs and run DPO.

Steps
1. Install dependencies in Colab (include bitsandbytes + accelerate if using offload):

   pip install -r requirements.txt bitsandbytes accelerate

2. Use `neuroslm.tools.model_selector.pick_for_current_gpu()` to pick strategy.

3. For DPO:
   - Prepare pairwise preference data: list of (prompt, a_resp, b_resp, label) where label=1 if a preferred.
   - Convert responses to token ids with `Tokenizer.encode` and build pairs.
   - Call `neuroslm.dpo.train_dpo` (note: you must integrate model log-prob computation into train_dpo). 

Notes
- DPO scaffold is intentionally minimal; you will need to implement log-prob calculation for your LM's forward pass (sum of token log probabilities) and appropriate batching/padding.
- For large models, prefer 4-bit quantization + LoRA. This allows training up to ~2B effective params on 14GB by freezing base weights and training adapters.
