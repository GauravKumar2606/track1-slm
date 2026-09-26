import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.model import GPTModel

# Standalone, post-hoc analysis tool — NOT part of capture_progress.py's live
# loop. Unlike the 5 scalar graphs (which just read numbers already written
# to the TensorBoard log, zero GPU cost), this needs to load the model and
# run many forward passes to actually evaluate the loss surface — that
# competes for GPU memory/compute with anything else using the GPU. Run this
# against a saved checkpoint when the GPU isn't busy training (e.g. after a
# run finishes, or between runs), not continuously alongside live training.
#
# Technique: Li et al., "Visualizing the Loss Landscape of Neural Nets"
# (2018) — the standard way to plot a loss surface for a model with millions
# of parameters (you can't grid-search the real parameter space directly).
# Pick 2 random directions shaped like the full parameter set, each
# individually rescaled ("filter normalization") to match the norm of the
# tensor it corresponds to (so no single huge/tiny tensor dominates the
# picture), then evaluate loss at a grid of points
# theta_0 + alpha*direction1 + beta*direction2 around the checkpoint's
# actual trained weights (alpha=beta=0 is the real checkpoint, the point
# training actually reached).

LABEL_IGNORE_INDEX = -100  # must match data/prepare_instruct_data.py / training/train_instruct.py


def load_flat_eval_batches(model_letter, config, device, n_batches, batch_size, seed):
    """For model-a/b/c: reuses train.py's flat-continuous-stream format."""
    path = f"data/tokenized/model-{model_letter}/val.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    batches = []
    max_start = len(data) - config.context_length - 1
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([data[s: s + config.context_length].astype(np.int64) for s in starts])
        y = np.stack([data[s + 1: s + 1 + config.context_length].astype(np.int64) for s in starts])
        batches.append((
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            False,  # masked?
        ))
    return batches


def load_instruct_eval_batches(model_letter, config, device, n_batches, batch_size, seed):
    """For model-a-instruct: reuses train_instruct.py's fixed per-example, masked format."""
    import json
    tokenized_dir = f"data/tokenized/model-{model_letter}"
    with open(os.path.join(tokenized_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = meta["num_examples"]["val"]
    x_data = np.memmap(os.path.join(tokenized_dir, "val_input_ids.bin"), dtype=np.uint16, mode="r",
                        shape=(n, config.context_length))
    y_data = np.memmap(os.path.join(tokenized_dir, "val_labels.bin"), dtype=np.int16, mode="r",
                        shape=(n, config.context_length))
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        idx = rng.integers(0, n, size=batch_size)
        x = torch.from_numpy(x_data[idx].astype(np.int64)).to(device)
        y = torch.from_numpy(y_data[idx].astype(np.int64)).to(device)
        batches.append((x, y, True))  # masked
    return batches


@torch.no_grad()
def evaluate_loss(model, batches):
    losses = []
    for x, y, masked in batches:
        logits = model(x)
        if masked:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=LABEL_IGNORE_INDEX)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def get_random_direction(params):
    """One random direction per parameter tensor, filter-normalized: each
    tensor's random direction is rescaled to match that SAME tensor's own
    norm, so e.g. a huge embedding matrix and a tiny layernorm bias
    contribute proportionally rather than the direction being dominated by
    whichever tensor happens to be largest."""
    direction = []
    for p in params:
        d = torch.randn_like(p)
        d = d * (p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["a", "a-instruct", "b", "c"], required=True,
                         help="Which model's val data / architecture config to evaluate against.")
    parser.add_argument("--config", default=None,
                         help="Defaults to configs/model_117M.yaml, or configs/model_117M_instruct.yaml for a-instruct.")
    parser.add_argument("--resolution", type=int, default=15, help="Grid is resolution x resolution points.")
    parser.add_argument("--span", type=float, default=1.0, help="alpha/beta range: [-span, +span].")
    parser.add_argument("--n-batches", type=int, default=5, help="Batches averaged per grid point (reduces noise).")
    parser.add_argument("--batch-size", type=int, default=8, help="Kept small on purpose — resolution^2 forward passes add up fast.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output PNG path (default: alongside the checkpoint).")
    args = parser.parse_args()

    config_path = args.config or (
        "configs/model_117M_instruct.yaml" if args.model == "a-instruct" else "configs/model_117M.yaml"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig.from_yaml(config_path)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = GPTModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading {args.n_batches} fixed eval batch(es) (batch_size={args.batch_size}) from model-{args.model}'s val split ...", flush=True)
    if args.model == "a-instruct":
        batches = load_instruct_eval_batches(args.model, config, device, args.n_batches, args.batch_size, args.seed)
    else:
        batches = load_flat_eval_batches(args.model, config, device, args.n_batches, args.batch_size, args.seed)

    original_params = [p.detach().clone() for p in model.parameters()]
    print("Generating 2 random filter-normalized directions ...", flush=True)
    torch.manual_seed(args.seed)
    direction1 = get_random_direction(original_params)
    direction2 = get_random_direction(original_params)

    center_loss = evaluate_loss(model, batches)
    print(f"Loss at the actual checkpoint (alpha=beta=0): {center_loss:.4f}", flush=True)

    alphas = np.linspace(-args.span, args.span, args.resolution)
    betas = np.linspace(-args.span, args.span, args.resolution)
    loss_grid = np.zeros((args.resolution, args.resolution))

    total_points = args.resolution * args.resolution
    print(f"Evaluating {total_points} grid points ({args.n_batches} batches each = "
          f"{total_points * args.n_batches} forward passes total) ...", flush=True)

    done = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                for p, orig, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                    p.copy_(orig + alpha * d1 + beta * d2)
            loss_grid[j, i] = evaluate_loss(model, batches)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total_points} grid points done", flush=True)

    # restore original weights (this checkpoint file is untouched either way,
    # but avoids leaving the in-memory model in a perturbed state)
    with torch.no_grad():
        for p, orig in zip(model.parameters(), original_params):
            p.copy_(orig)

    out_path = args.out or (os.path.splitext(args.checkpoint)[0] + "_loss_landscape.png")
    fig, ax = plt.subplots(figsize=(9, 8))
    # cap extreme values so a few blown-up points (perturbation pushed the
    # model into a bad region) don't wash out the contour detail near the
    # actual trained minimum, which is the part worth seeing clearly.
    capped = np.clip(loss_grid, None, np.percentile(loss_grid, 90))
    contour = ax.contourf(alphas, betas, capped, levels=25, cmap="viridis")
    ax.contour(alphas, betas, capped, levels=25, colors="black", linewidths=0.3, alpha=0.4)
    ax.plot(0, 0, "r*", markersize=18, label=f"trained checkpoint (loss={center_loss:.3f})")
    ax.set_xlabel("direction 1 (alpha)")
    ax.set_ylabel("direction 2 (beta)")
    ax.set_title(f"Model {args.model.upper()} loss landscape — {os.path.basename(args.checkpoint)}\n"
                 f"(values above the 90th percentile capped for contrast)")
    ax.legend(loc="upper right")
    fig.colorbar(contour, ax=ax, label="loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
