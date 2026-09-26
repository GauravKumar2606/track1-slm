import argparse
import re
import hashlib
import os
import time
from tqdm import tqdm

# --- Configuration ---
RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

# Threshold for switching to streaming mode (10 GB) — per-dataset decision
STREAMING_THRESHOLD_BYTES = 10 * 1024 * 1024 * 1024

CHUNK_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB read chunks

# Datasets forced onto the streaming path regardless of byte size. The 10GB
# threshold above was calibrated for prose (few, long lines per byte); source
# code has many more, much shorter lines per byte (csharp_train.txt: ~38
# bytes/line across 227M lines for an 8.8GB file). process_in_memory_text()'s
# full-file split('\n') would materialize hundreds of millions of tiny Python
# string objects (~50 bytes of object overhead each, on top of the original
# file string still held at the same time) — a real MemoryError risk that
# the byte-size threshold alone doesn't capture for this kind of source.
FORCE_STREAMING = {"csharp"}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REGISTRY — must stay in sync with data/download_datasets.py.
# Downloaded sets: wikitext, pg19, openwebtext, c4 (bookcorpus/ccnews retired).
# csharp is Model B-only — see MODEL_SOURCES in prepare_splits.py, which
# keeps it out of Model A's corpus despite living in this same DATASETS dict
# and data/cleaned/ directory.
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "wikitext":    "data/raw/wikitext/wikitext_train.txt",
    "pg19":        "data/raw/pg19/pg19_train.txt",
    "openwebtext": "data/raw/openwebtext/openwebtext_train.txt",
    "c4":          "data/raw/c4/c4_train.txt",
    "csharp":      "data/raw/csharp/csharp_train.txt",
    "instruct":    "data/raw/instruct/instruct_train.txt",
}

# Skip re-cleaning a dataset whose cleaned output already exists and is
# above this size — mirrors download_datasets.py's already_downloaded().
MIN_CLEANED_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def already_cleaned(name, is_large):
    suffix = "_cleaned_stream.txt" if is_large else "_cleaned.txt"
    path = os.path.join(CLEANED_DIR, f"{name}{suffix}")
    if os.path.exists(path) and os.path.getsize(path) >= MIN_CLEANED_SIZE_BYTES:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [SKIP] {path} already exists ({size_mb:.1f} MB) — skipping cleaning.", flush=True)
        return path
    return None

# --- Precompiled patterns (avoids re-parsing the pattern on every call) ---
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http\S+')
ASCII_LETTER_RE = re.compile(r'[a-zA-Z]')

# C0 control characters and DEL, excluding tab/newline/carriage-return.
# NUL and friends are valid ASCII code points, so a plain ascii-only filter
# lets them through — and a stray NUL anywhere in a huge training file
# silently truncates SentencePiece's C++ trainer at that byte. Strip them
# here so they never reach data/cleaned/ in the first place.
_CONTROL_CODEPOINTS = [c for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)] + [0x7F]
CONTROL_CHAR_TABLE = {cp: None for cp in _CONTROL_CODEPOINTS}

# --- Utility Functions ---

def is_english(text):
    """Simple heuristic to check for English content (detecting non-English characters)."""
    return len(text) > 10 and bool(ASCII_LETTER_RE.search(text))


def clean_text_segment(text):
    """Applies all cleaning rules to a given text segment."""
    text = HTML_TAG_RE.sub('', text)
    text = URL_RE.sub('', text)
    # Vectorized ASCII filter (C-level), replaces the old char-by-char Python loop.
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.translate(CONTROL_CHAR_TABLE).strip()
    return text


def filter_and_clean_paragraphs(chunk_text):
    """Splits raw text into examples and applies cleaning rules (no dedup).

    download_datasets.py writes one example per line (text + "\n"), not
    blank-line-separated paragraphs, so we split on single newlines here.
    """
    cleaned_paragraphs = []
    for paragraph in chunk_text.split('\n'):
        paragraph = paragraph.strip()
        if len(paragraph) < 20:
            continue
        cleaned = clean_text_segment(paragraph)
        if not cleaned:
            continue
        if not is_english(cleaned):
            continue
        cleaned_paragraphs.append(cleaned)
    return cleaned_paragraphs


def stream_clean_text(source_path, target_name, output_dir=CLEANED_DIR):
    """Streaming clean + in-memory hash-set dedup for large files.

    The raw file is still read in bounded chunks (memory-safe for a 40GB+
    source), but deduplication uses an in-memory set of 8-byte hashes instead
    of SQLite: a prior SQLite-based version saturated disk I/O (WAL
    checkpointing) for tens of millions of small inserts, while a hash set of
    this size (well under 1GB for ~8M entries) fits easily in RAM and gives
    O(1) membership checks with no disk round-trips.

    output_dir defaults to data/cleaned/, shared by every registered
    dataset including csharp (Model B-only) — prepare_splits.py's
    MODEL_SOURCES map decides which cleaned files go into which model's
    corpus, so this function doesn't need to know about that split.
    """
    print(f"--- Running STREAMING Cleaning for {target_name} (Deduplication) ---", flush=True)

    total_size = os.path.getsize(source_path)
    seen_hashes = set()

    lines_before = 0
    lines_after = 0
    bytes_read = 0
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{target_name}_cleaned_stream.txt")

    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            print("Reading file in chunks for processing...", flush=True)
            chunk_index = 0
            leftover = ""
            while True:
                chunk = f_in.read(CHUNK_SIZE_BYTES)
                is_last_chunk = not chunk
                if is_last_chunk and not leftover:
                    break
                chunk_index += 1
                bytes_read += len(chunk.encode('utf-8', errors='ignore'))

                # A line can be split across two reads; hold the trailing
                # incomplete line back and prepend it to the next chunk.
                text = leftover + chunk
                if is_last_chunk:
                    leftover = ""
                else:
                    split_at = text.rfind('\n')
                    if split_at == -1:
                        leftover = text
                        continue
                    leftover = text[split_at + 1:]
                    text = text[:split_at]

                lines_before += text.count('\n') + 1

                candidates = filter_and_clean_paragraphs(text)

                new_count = 0
                for paragraph in candidates:
                    h = hashlib.blake2b(paragraph.encode('utf-8'), digest_size=8).digest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    f_out.write(paragraph)
                    f_out.write("\n\n")
                    new_count += 1
                lines_after += new_count

                elapsed = time.time() - start_time
                pct = (bytes_read / total_size) * 100 if total_size else 100
                print(
                    f"  [{target_name}] chunk {chunk_index}: "
                    f"{bytes_read / (1024**2):.0f} MB / {total_size / (1024**2):.0f} MB "
                    f"({pct:.1f}%) | +{new_count} unique paragraphs this chunk | "
                    f"elapsed {elapsed:.0f}s",
                    flush=True,
                )
    except Exception as e:
        print(f"An unexpected error occurred during streaming for {target_name}: {e}. Skipping.", flush=True)
        return None

    print(f"Streaming clean corpus saved to: {output_path}", flush=True)
    print(f"  Lines before cleaning (approx): {lines_before:,}", flush=True)
    print(f"  Lines after cleaning/dedup:     {lines_after:,}", flush=True)
    return output_path


def process_in_memory_text(source_path, target_name, output_dir=CLEANED_DIR):
    """Handles small files using standard in-memory processing.

    output_dir defaults to data/cleaned/; see stream_clean_text's docstring.
    """
    print(f"--- Running IN-MEMORY Cleaning for {target_name} (Deduplication) ---", flush=True)

    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_content = f.read()
    except FileNotFoundError:
        print(f"Failed to process in-memory clean text for {target_name}: Source file not found at {source_path}. Skipping.", flush=True)
        return None

    all_paragraphs = [p.strip() for p in full_content.split('\n') if p.strip()]
    lines_before = len(all_paragraphs)

    cleaned_paragraphs = []
    seen_hashes = set()

    for paragraph in tqdm(all_paragraphs, desc=f"Cleaning {target_name}", mininterval=1.0):
        if len(paragraph) < 20:
            continue
        cleaned = clean_text_segment(paragraph)
        if not cleaned:
            continue
        if not is_english(cleaned):
            continue
        h = hashlib.blake2b(cleaned.encode('utf-8'), digest_size=8).digest()
        if h in seen_hashes:
            continue
        cleaned_paragraphs.append(cleaned)
        seen_hashes.add(h)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{target_name}_cleaned.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned_paragraphs))

    print(f"In-memory clean corpus saved to: {output_path}", flush=True)
    print(f"  Lines before cleaning: {lines_before:,}", flush=True)
    print(f"  Lines after cleaning/dedup: {len(cleaned_paragraphs):,}", flush=True)
    return output_path


def main():
    """Main function to orchestrate all data cleaning. Per-dataset size decision."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=None,
        help="Clean only this dataset (e.g. csharp). Omit to run all registered datasets "
             "(already_cleaned() still skips any whose output already exists either way).",
    )
    args = parser.parse_args()

    datasets_to_run = DATASETS if args.dataset is None else {args.dataset: DATASETS[args.dataset]}

    os.makedirs(CLEANED_DIR, exist_ok=True)

    print("--- Running Dataset Size Pre-Check ---", flush=True)
    if args.dataset:
        print(f"Cleaning only: {args.dataset}", flush=True)
    size_check_results = {}
    for name, path in datasets_to_run.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_check_results[name] = size
            is_large = size > STREAMING_THRESHOLD_BYTES or name in FORCE_STREAMING
            reason = "LARGE >10GB" if size > STREAMING_THRESHOLD_BYTES else "forced" if is_large else None
            print(f"  - {name}: {size / (1024 * 1024):.2f} MB ({size} bytes) "
                  f"{f'[{reason} -> streaming]' if is_large else '[small -> in-memory]'}", flush=True)
        else:
            size_check_results[name] = 0
            print(f"  - {name}: 0.00 MB (File not found, skipping)", flush=True)

    print("\n--- Starting Text Cleaning (per-dataset method selection) ---", flush=True)

    cleaned_corpus_paths = []

    for name, source_path in datasets_to_run.items():
        if not os.path.exists(source_path):
            print(f"Skipping {name}: Source file not found at {source_path}.", flush=True)
            continue

        size = size_check_results[name]
        is_large = size > STREAMING_THRESHOLD_BYTES or name in FORCE_STREAMING

        output_path = already_cleaned(name, is_large)
        if not output_path:
            if is_large:
                output_path = stream_clean_text(source_path, name)
            else:
                output_path = process_in_memory_text(source_path, name)

        if output_path:
            cleaned_corpus_paths.append(output_path)

    if not cleaned_corpus_paths:
        print("\nCLEANING FAILED: Could not generate any cleaned corpus paths. Aborting.", flush=True)
        return

    print("\n==========================================================", flush=True)
    print("Phase 1, Task 2 Complete: All available datasets cleaned and deduplicated.", flush=True)
    print(f"Total datasets processed: {len(cleaned_corpus_paths)}.", flush=True)
    for p in cleaned_corpus_paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  - {p} ({size_mb:.1f} MB)", flush=True)
    print("Next: python data/prepare_splits.py", flush=True)
    print("===============================================================", flush=True)


if __name__ == "__main__":
    main()
