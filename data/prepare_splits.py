import argparse
import os
import glob

CLEANED_DIR = "data/cleaned"

# Which of clean_text.py's registered datasets belong to each model's
# corpus. Model A is English-only; Model B adds C# + instruction data on
# top of the same English sources (trained from scratch on the mix).
# Model C fine-tunes from Model A-Instruct's weights (see training/
# train.py's --init-from) on C# only, full corpus — REVERTED 2026-09-13
# from an earlier ~15% instruct-blend attempt (see
# data/sample_instruct_for_model_c.py, kept for reference but no longer
# used here): since Model C's base checkpoint (Model A-Instruct) already
# has instruction-following behavior baked into its weights, diluting the
# C# fine-tune corpus to "preserve" that was redundant — the concern the
# blend was meant to address doesn't apply the same way once the starting
# checkpoint itself is already instruction-tuned.
# Extend this map (and clean_text.py's DATASETS) when adding a new source.
MODEL_SOURCES = {
    "a": ["wikitext", "pg19", "openwebtext", "c4"],
    "b": ["wikitext", "pg19", "openwebtext", "c4", "csharp", "instruct"],
    "c": ["csharp"],
}


def collect_cleaned_files(names):
    """Finds the cleaned corpus file(s) produced by data/clean_text.py for
    the given dataset names (in-memory or streamed variant, whichever exists)."""
    files = []
    for name in names:
        patterns = [
            os.path.join(CLEANED_DIR, f"{name}_cleaned_stream.txt"),
            os.path.join(CLEANED_DIR, f"{name}_cleaned.txt"),
        ]
        matches = [p for p in patterns if os.path.exists(p)]
        if not matches:
            print(f"  Warning: no cleaned file found for '{name}' (looked for {patterns}).")
            continue
        files.extend(matches)
    return sorted(set(files))


def estimate_total_chars(cleaned_files):
    """Sums each file's byte size as a stand-in for character count.

    Exact enough for computing 90/5/5 split boundaries across billions of
    characters (UTF-8 byte count vs. actual character count only diverges
    on multi-byte characters, a negligible fraction of these corpora) —
    avoids a full read-through pass just to count.
    """
    total = 0
    for path in cleaned_files:
        size = os.path.getsize(path)
        print(f"  - {path} ({size / (1024 * 1024):.1f} MB)")
        total += size
    return total


def split_from_sources(cleaned_files, base_path, total_chars, train_frac=0.90, val_frac=0.05, read_chunk_chars=8 * 1024 * 1024):
    """Streams cleaned source files directly into train/val/test (90/5/5),
    without ever materializing a combined corpus.txt.

    A prior version concatenated all cleaned files into a standalone
    corpus.txt first, then split that into train/val/test — needing the
    full combined size TWICE on disk at once (corpus.txt + train/val/test),
    which ran Model B's 63GB combined corpus out of disk space mid-split.
    Nothing downstream ever reads corpus.txt (tokenize_corpus.py only reads
    train.txt/val.txt/test.txt), so it added disk cost with no benefit.
    This reads each cleaned source file in sequence, in bounded chunks (also
    avoiding the earlier MemoryError from a prior version's full-file
    f.read()), and routes each chunk's characters to train/val/test based on
    a running position against the precomputed split boundaries — the same
    routing logic as before, just fed directly from the source files.
    """
    train_end = int(total_chars * train_frac)
    val_end = train_end + int(total_chars * val_frac)

    print(f"Target split (90/5/5): Train ({train_end:,} chars), "
          f"Val ({val_end - train_end:,} chars), Test ({total_chars - val_end:,} chars).")

    with open(f"{base_path}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"{base_path}/val.txt", "w", encoding="utf-8") as f_val, \
         open(f"{base_path}/test.txt", "w", encoding="utf-8") as f_test:
        pos = 0
        for path in cleaned_files:
            print(f"  - Splitting from {path}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
                while True:
                    chunk = f_in.read(read_chunk_chars)
                    if not chunk:
                        break

                    local_train_end = max(0, min(len(chunk), train_end - pos))
                    local_val_end = max(0, min(len(chunk), val_end - pos))

                    if local_train_end > 0:
                        f_train.write(chunk[:local_train_end])
                    if local_val_end > local_train_end:
                        f_val.write(chunk[local_train_end:local_val_end])
                    if len(chunk) > local_val_end:
                        f_test.write(chunk[local_val_end:])

                    pos += len(chunk)

    return total_chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SOURCES), default="a",
                         help="Which corpus to build: 'a' (English-only) or 'b' (English + C#).")
    args = parser.parse_args()

    model_dir = f"data/model-{args.model}"
    source_names = MODEL_SOURCES[args.model]
    os.makedirs(model_dir, exist_ok=True)

    print(f"Building Model {args.model.upper()} corpus from sources: {source_names}")
    cleaned_files = collect_cleaned_files(source_names)
    if not cleaned_files:
        print(f"No cleaned corpus files found in {CLEANED_DIR} for {source_names}. Run data/clean_text.py first.")
        return

    print(f"Found {len(cleaned_files)} cleaned corpus file(s):")
    total_chars = estimate_total_chars(cleaned_files)

    split_from_sources(cleaned_files, model_dir, total_chars)
    print(f"Split directly from source files ({total_chars:,} characters total) — "
          f"no combined corpus.txt written (unused downstream, and doubles disk usage at this scale).")

    # Rough token estimate: ~4 chars/token is a standard heuristic for English text.
    token_estimate = total_chars // 4
    print(f"Estimated token count: ~{token_estimate:,} tokens")

    print("\n--- PHASE 1 COMPLETION REPORT ---")
    print(f"Task 3: Data preparation for model-{args.model} completed successfully.")
    print(f"Datasets combined: {[os.path.basename(p) for p in cleaned_files]}")
    print(f"Final files: {model_dir}/train.txt, val.txt, test.txt")
    print(f"Estimated tokens: ~{token_estimate:,}")
    print("=====================================")


if __name__ == "__main__":
    main()
