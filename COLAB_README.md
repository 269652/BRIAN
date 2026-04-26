NeuroSLM Colab / GPU Quickstart

1. Install dependencies

   pip install -r requirements.txt

2. (Optional) Request a GPU-scaled build profile via DNA

   Create `neuroslm/dna/templates/build.lisp` with the following content to request a GPU profile:

   (build
     (profile gpu_medium)
   )

   The Ribosome will read this template if present and choose the corresponding builtin build profile
   (cpu, gpu_small, gpu_medium, gpu_large). If no template is present, the code auto-detects CUDA and
   picks an appropriate profile.

3. Start training (example)

   python train_colab.py --preset medium --steps 2000 --batch_size 8

   The helper script will detect CUDA and pass --device cuda when available. You can also run the
   training entrypoint directly and pass --device cuda.
