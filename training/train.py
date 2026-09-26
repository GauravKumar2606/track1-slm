import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.config import ModelConfig
from model.model import GPTModel

TOKENIZED_DIR_TEMPLATE = "data/tokenized/model-{model}"
CHECKPOINT_DIR_TEMPLATE = "checkpoints/model-{model}"
LOG_DIR_TEMPLATE = "logs/model-{model}"

VAL_EVAL_BATCHES = 20
LR_MIN_RATIO = 0.1  # cosine decay floor as a fraction of peak LR


def load_split(tokenized_dir, split):
    path = os.path.join(tokenized_dir, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, context_length, device):
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)
    x = np.stack([data[s: s + context_length].astype(np.int64) for s in starts])
    y = np.stack([data[s + 1: s + 1 + context_length].astype(np.int64) for s in starts])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def lr_at_step(step, warmup_steps, max_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (LR_MIN_RATIO + (1 - LR_MIN_RATIO) * cosine)


@torch.no_grad()
def estimate_val_loss(model, val_data, config, device, n_batches=VAL_EVAL_BATCHES):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, config.batch_size, config.context_length, device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f[len("ckpt_"):-len(".pt")]))
    return os.path.join(checkpoint_dir, candidates[-1])


def rotate_checkpoints(checkpoint_dir, keep_last_n):
    """Deletes older periodic ckpt_*.pt files, keeping only the keep_last_n
    most recent (by step number). best.pt is a separate file (doesn't start
    with "ckpt_" the way these do — it's named literally "best.pt") and is
    never touched here; it's always kept regardless of rotation.

    Without this, a full run saves one ~1.32GB checkpoint (110M fp32 weights
    + AdamW's 2x optimizer state) every checkpoint_every steps and never
    deletes any of them — for configs/model_117M.yaml's 133,500 steps /
    1,000-step interval, that's ~178GB for a single model's checkpoints
    alone, enough to exhaust a much smaller disk mid-run. Only the most
    recent few are actually useful (resuming after a crash close to where
    it left off) — everything older is pure waste once superseded.
    """
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if len(candidates) <= keep_last_n:
        return
    candidates.sort(key=lambda f: int(f[len("ckpt_"):-len(".pt")]))
    for old_file in candidates[:-keep_last_n]:
        old_path = os.path.join(checkpoint_dir, old_file)
        os.remove(old_path)
        print(f"Removed old checkpoint (rotation, keeping last {keep_last_n}): {old_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", "c"], required=True)
    parser.add_argument("--config", default="configs/model_117M.yaml")
    parser.add_argument("--init-from", default=None,
                         help="Checkpoint path to initialize model WEIGHTS from (optimizer state is not "
                              "restored, training starts at step 0). Only used when this model's own "
                              "checkpoint directory is empty — e.g. Model C fine-tunes from Model A: "
                              "--model c --init-from checkpoints/model-a/best.pt")
    parser.add_argument("--keep-last-checkpoints", type=int, default=3,
                         help="Number of most recent periodic ckpt_*.pt files to keep on disk (best.pt "
                              "and the final checkpoint are always kept regardless). Default 3 keeps "
                              "steady-state disk use to ~(3+1)*1.32GB per model instead of growing "
                              "unbounded for the whole run.")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training model-{args.model} on {device}")

    tokenized_dir = TOKENIZED_DIR_TEMPLATE.format(model=args.model)
    checkpoint_dir = CHECKPOINT_DIR_TEMPLATE.format(model=args.model)
    log_dir = LOG_DIR_TEMPLATE.format(model=args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_data = load_split(tokenized_dir, "train")
    val_data = load_split(tokenized_dir, "val")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    writer = SummaryWriter(log_dir=log_dir)

    start_step = 0
    best_val_loss = float("inf")

    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    elif args.init_from:
        # Weight-only init (e.g. Model C fine-tuning from Model A's trained
        # checkpoint): optimizer state is intentionally NOT restored, and
        # step/best_val_loss reset to 0/inf, since this is a new training
        # run over different data (or a different config's LR schedule),
        # not a resume of the same run.
        print(f"No checkpoint in {checkpoint_dir} — initializing weights from: {args.init_from}")
        init_checkpoint = torch.load(args.init_from, map_location=device)
        model.load_state_dict(init_checkpoint["model_state_dict"])
    else:
        if args.model == "c":
            print("WARNING: --model c with no --init-from and no existing checkpoint — "
                  "this will train Model C from scratch, not fine-tune it from Model A. "
                  "Pass --init-from checkpoints/model-a/best.pt if that's not intended.")
        print("No existing checkpoint found — starting from scratch.")

    model.train()
    tokens_per_step = config.batch_size * config.context_length
    t_last = time.time()

    for step in range(start_step, config.max_steps):
        lr = lr_at_step(step, config.warmup_steps, config.max_steps, config.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.batch_size, config.context_length, device)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # clip_grad_norm_ returns the pre-clip total L2 norm across all parameters —
        # a standard training-health signal (spikes/instability show up here before
        # they show up in loss). Free to capture: we already call this for clipping,
        # just weren't logging its return value.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            now = time.time()
            elapsed = now - t_last
            tps = (100 * tokens_per_step / elapsed) if step > start_step else 0.0
            t_last = now
            print(f"step {step:>7} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tps:,.0f} | grad_norm {grad_norm:.3f}")
            writer.add_scalar("train_loss", loss.item(), step)
            writer.add_scalar("learning_rate", lr, step)
            writer.add_scalar("tokens_per_second", tps, step)
            writer.add_scalar("grad_norm", grad_norm.item(), step)

        if step % 500 == 0 and step > start_step:
            val_loss = estimate_val_loss(model, val_data, config, device)
            print(f"step {step:>7} | val_loss {val_loss:.4f}")
            writer.add_scalar("val_loss", val_loss, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
                print(f"New best val_loss {best_val_loss:.4f} — saved {best_path}")

        if step % config.checkpoint_every == 0 and step > start_step:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{step}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            rotate_checkpoints(checkpoint_dir, args.keep_last_checkpoints)

    final_path = os.path.join(checkpoint_dir, f"ckpt_{config.max_steps}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": config.max_steps,
            "best_val_loss": best_val_loss,
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")
    rotate_checkpoints(checkpoint_dir, args.keep_last_checkpoints)
    writer.close()


if __name__ == "__main__":
    main()
