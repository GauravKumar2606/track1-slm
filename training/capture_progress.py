import argparse
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless — this runs unattended for the whole training duration, no display available
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Must match the scalar tags train.py's/train_instruct.py's SummaryWriter
# actually logs. grad_norm added 2026-09-14 (clip_grad_norm_'s pre-clip
# total L2 norm — was already being computed for gradient clipping, just
# wasn't being logged before).
TAGS = ["train_loss", "learning_rate", "tokens_per_second", "val_loss", "grad_norm"]


def load_scalars(log_dir):
    """Reads train.py's TensorBoard event files directly (no server/browser
    involved) and returns {tag: (steps, values)} for whichever tags exist."""
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})  # 0 = keep every point, no downsampling
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    data = {}
    for tag in TAGS:
        if tag in available:
            events = ea.Scalars(tag)
            data[tag] = ([e.step for e in events], [e.value for e in events])
    return data


def plot_and_save(data, out_dir, model_label):
    if not data:
        print("  No scalar data found yet (training may not have started logging).", flush=True)
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # 6 cells for 5 metrics — one left blank
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(TAGS):
            ax.axis("off")  # unused 6th cell (5 metrics in a 2x3 grid)
            continue
        tag = TAGS[i]
        if tag in data:
            steps, values = data[tag]
            ax.plot(steps, values, linewidth=1)
            ax.set_title(tag)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")
    fig.suptitle(f"Model {model_label.upper()} training progress — "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fig.tight_layout()

    latest_step = max((max(steps) for steps, _ in data.values()), default=0)
    out_path = os.path.join(
        out_dir,
        f"progress_step_{latest_step:08d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "a-instruct", "b", "c"], required=True)
    parser.add_argument("--interval-seconds", type=int, default=300,
                         help="How often to save a new screenshot (default: 300s = 5 min).")
    parser.add_argument("--out-dir", default=None,
                         help="Where to save screenshots (default: logs/model-<model>/screenshots).")
    args = parser.parse_args()

    log_dir = f"logs/model-{args.model}"
    out_dir = args.out_dir or os.path.join(log_dir, "screenshots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Capturing training progress for model-{args.model} every {args.interval_seconds}s", flush=True)
    print(f"Reading TensorBoard logs from: {log_dir}", flush=True)
    print(f"Saving screenshots to: {out_dir}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            data = load_scalars(log_dir)
            out_path = plot_and_save(data, out_dir, args.model)
            if out_path:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved {out_path}", flush=True)
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
