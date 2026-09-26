import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.config import ModelConfig
from model.model import GPTModel

TOKENIZER_MODEL = "tokenizer/tokenizer.model"
N_EVAL_BATCHES = 50
SAMPLE_PROMPT = "The capital of France is"
SAMPLE_MAX_TOKENS = 50


def load_split(tokenized_dir, split):
    path = os.path.join(tokenized_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, context_length, device):
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)
    x = np.stack([data[s: s + context_length].astype(np.int64) for s in starts])
    y = np.stack([data[s + 1: s + 1 + context_length].astype(np.int64) for s in starts])
    return (
        torch.from_numpy(x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


@torch.no_grad()
def generate_sample(model, sp, config, device, prompt, max_tokens):
    ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_tokens):
        idx_cond = input_ids[:, -config.context_length:]
        logits = model(idx_cond)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == sp.eos_id():
            break
    return sp.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["a", "b", "c"], default="a")
    parser.add_argument("--config", default="configs/model_117M.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig.from_yaml(args.config)

    model = GPTModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenized_dir = f"data/tokenized/model-{args.model}"
    val_data = load_split(tokenized_dir, "val")

    losses = []
    for _ in range(N_EVAL_BATCHES):
        x, y = get_batch(val_data, config.batch_size, config.context_length, device)
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    val_loss = sum(losses) / len(losses)
    perplexity = math.exp(val_loss)

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    sample = generate_sample(model, sp, config, device, SAMPLE_PROMPT, SAMPLE_MAX_TOKENS)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Sample generation ({SAMPLE_PROMPT!r}): {sample!r}")


if __name__ == "__main__":
    main()
