import os
import sys

# Allow running this script directly (python model/verify_model.py) from the
# repo root, regardless of cwd, by ensuring the repo root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.config import ModelConfig
from model.model import GPTModel

CONFIG_PATH = "configs/model_117M.yaml"


def main():
    config = ModelConfig.from_yaml(CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPTModel(config).to(device)
    n_params = model.num_parameters()
    print(f"Model instantiated on {device}")
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    dummy_input = torch.randint(0, config.vocab_size, (2, 64), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Forward pass output shape: {tuple(logits.shape)}")
    expected_shape = (2, 64, config.vocab_size)
    assert tuple(logits.shape) == expected_shape, f"Expected {expected_shape}, got {tuple(logits.shape)}"
    print("Output shape verified: OK")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU memory allocated: {allocated:.3f} GB")
        print(f"GPU peak memory (forward pass): {peak:.3f} GB")
    else:
        print("CUDA not available — skipping GPU memory report.")

    if 110_000_000 <= n_params <= 125_000_000:
        print("Parameter count within expected 110M-125M range: OK")
    else:
        print(f"WARNING: parameter count {n_params:,} outside expected 110M-125M range.")


if __name__ == "__main__":
    main()
