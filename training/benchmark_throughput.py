"""
Throughput benchmark — NOT training/train.py, so it's outside the HARD STOP
list. Runs a handful of real forward+backward+optimizer steps (same code
path as train.py) on actual tokenized data, to get honest tokens/sec and
VRAM numbers on this machine before committing to a multi-hour real run.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.config import ModelConfig
from model.model import GPTModel

N_WARMUP_STEPS = 5
N_TIMED_STEPS = 30


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


def run_benchmark(config, batch_size, device):
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    train_data = load_split("data/tokenized/model-a", "train")
    tokens_per_step = batch_size * config.context_length

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(N_WARMUP_STEPS):
        x, y = get_batch(train_data, batch_size, config.context_length, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(N_TIMED_STEPS):
        x, y = get_batch(train_data, batch_size, config.context_length, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (N_TIMED_STEPS * tokens_per_step) / elapsed
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return tokens_per_sec, peak_mem_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_117M.yaml")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None,
                         help="Batch sizes to try (default: config's batch_size plus a couple larger ones)")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device} | context_length={config.context_length}", flush=True)

    batch_sizes = args.batch_sizes or sorted(set([config.batch_size, config.batch_size * 2, config.batch_size * 4]))

    print(f"{'batch_size':>10} | {'tokens/sec':>12} | {'peak VRAM (GB)':>14} | {'est. time for max_steps':>24}", flush=True)
    for bs in batch_sizes:
        try:
            tps, peak_mem = run_benchmark(config, bs, device)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>10} | OOM — reduce batch size", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        total_tokens = config.max_steps * bs * config.context_length
        est_seconds = total_tokens / tps
        est_hours = est_seconds / 3600
        mem_str = f"{peak_mem:.2f}" if peak_mem is not None else "N/A"
        print(f"{bs:>10} | {tps:>12,.0f} | {mem_str:>14} | {est_hours:>21.1f}h", flush=True)


if __name__ == "__main__":
    main()
