import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.config import ModelConfig
from model.model import GPTModel

# Updated 2026-09-26 for the three-way comparison deploy (see CLAUDE.md's
# "Three-Model Comparison Design", REVISED 2026-09-13). The original version
# of this script (2026-09-10) only quantized raw Model A, which is no longer
# part of the comparison — it's been replaced by Model A-Instruct. Now
# quantizes all three currently-compared models (A-Instruct, B, C) so the
# Gradio demo (deployment/app.py) can show all three side-by-side, mirroring
# evaluation/compare_models.py's qualitative comparison. Architecture numbers
# (vocab_size/context_length/n_layers/etc.) are identical across all three
# configs — only the checkpoint weights differ.
MODELS = {
    "a-instruct": {
        "config": "configs/model_117M_instruct.yaml",
        "checkpoint": "checkpoints/model-a-instruct/best.pt",
    },
    "b": {
        "config": "configs/model_117M.yaml",
        "checkpoint": "checkpoints/model-b/best.pt",
    },
    "c": {
        "config": "configs/model_117M_finetune_c.yaml",
        "checkpoint": "checkpoints/model-c/best.pt",
    },
}
OUTPUT_DIR = "deployment/model_int8"


def quantize_one(letter, config_path, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"  Skipping model-{letter}: no checkpoint found at {checkpoint_path}")
        return

    config = ModelConfig.from_yaml(config_path)
    model = GPTModel(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    original_size = sum(p.numel() * p.element_size() for p in model.parameters())

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    out_dir = os.path.join(OUTPUT_DIR, letter)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "model_int8.pt")
    torch.save(quantized_model.state_dict(), output_path)

    quantized_size = os.path.getsize(output_path)
    print(
        f"  model-{letter}: original {original_size / (1024 ** 2):.2f} MB (in-memory fp32) "
        f"-> quantized {quantized_size / (1024 ** 2):.2f} MB on disk -> {output_path}"
    )


def main():
    print("Quantizing all available compared models (A-Instruct, B, C) to int8 ...")
    for letter, paths in MODELS.items():
        quantize_one(letter, paths["config"], paths["checkpoint"])
    print("Done. Missing models above were skipped — deployment/app.py handles partial availability gracefully.")


if __name__ == "__main__":
    main()
