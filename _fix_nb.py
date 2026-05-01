"""Fix colab_run.ipynb cell 2: total_mem -> total_memory"""
import json

with open("colab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][1]["source"] = [
    "# 1) Check GPU\n",
    "import torch, sys\n",
    'print(f"Python {sys.version}")\n',
    'print(f"PyTorch {torch.__version__}")\n',
    'print(f"CUDA available: {torch.cuda.is_available()}")\n',
    "if torch.cuda.is_available():\n",
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
    '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\n',
    "else:\n",
    '    print("WARNING: No GPU detected. Go to Runtime -> Change runtime type -> GPU")\n',
]

with open("colab_run.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed total_mem -> total_memory")
