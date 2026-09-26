import os
import random

# Model C fine-tunes on C# with a MINORITY blend of instruction data mixed
# in, to reduce catastrophic forgetting of the instruction-following
# behavior Model A-Instruct already has (decided 2026-09-13 — see
# "Three-Model Comparison Design" in CLAUDE.md). Target: ~15% instruct /
# 85% C# by character count. Using the FULL cleaned instruct file
# (1.47GB) alongside C# (3.65GB) would give ~29% instruct — too high — so
# this samples a subset down to the target ratio instead.
#
# Sampling is RANDOM (fixed seed, reproducible), not "first N bytes" —
# OpenHermes-2.5's example ordering may reflect its source composition
# (multiple sub-datasets concatenated), so a prefix-truncation would risk
# only capturing part of that diversity.
#
# Output is registered as its own pseudo-dataset name ("instruct_sample_c")
# so data/prepare_splits.py's existing per-name file lookup
# (`{name}_cleaned.txt`) picks it up with zero changes to that script —
# MODEL_SOURCES["c"] just lists "instruct_sample_c" instead of "instruct".

CSHARP_CLEANED = "data/cleaned/csharp_cleaned_stream.txt"
INSTRUCT_CLEANED = "data/cleaned/instruct_cleaned.txt"
OUTPUT_PATH = "data/cleaned/instruct_sample_c_cleaned.txt"

TARGET_RATIO = 0.15  # instruct's target share of (csharp + instruct_sample)
SEED = 42


def main():
    csharp_size = os.path.getsize(CSHARP_CLEANED)
    target_bytes = int(csharp_size * (TARGET_RATIO / (1 - TARGET_RATIO)))
    print(f"C# cleaned size: {csharp_size / 1e6:.1f} MB", flush=True)
    print(f"Target instruct sample size for ~{TARGET_RATIO:.0%} share: {target_bytes / 1e6:.1f} MB", flush=True)

    print(f"Reading {INSTRUCT_CLEANED} ...", flush=True)
    with open(INSTRUCT_CLEANED, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    total_lines = len(lines)
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)
    print(f"  {total_lines:,} examples, {total_bytes / 1e6:.1f} MB total", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(lines)

    kept = []
    kept_bytes = 0
    for line in lines:
        if kept_bytes >= target_bytes:
            break
        kept.append(line)
        kept_bytes += len(line.encode("utf-8"))

    print(f"Sampled {len(kept):,} / {total_lines:,} examples ({kept_bytes / 1e6:.1f} MB)", flush=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(kept)

    print(f"Saved to {OUTPUT_PATH}", flush=True)
    actual_ratio = kept_bytes / (kept_bytes + csharp_size)
    print(f"Actual resulting instruct share vs C#: {actual_ratio:.1%}", flush=True)


if __name__ == "__main__":
    main()
