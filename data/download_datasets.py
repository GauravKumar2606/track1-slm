import argparse
import os
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# DATASET REGISTRY
# Fixes applied:
#   wikitext    → Salesforce/wikitext + wikitext-103-raw-v1  ✅ working
#   pg19        → sedthh/gutenberg_english (pg19 uses deprecated script format)
#   openwebtext → Skylion007/openwebtext                     ✅ working
#   c4          → allenai/c4 english                         ✅ working
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "name":        "wikitext",
        "description": "Salesforce/wikitext (wikitext-103-raw-v1)",
        "raw_dir":     "data/raw/wikitext",
        "file_name":   "wikitext_train.txt",
    },
    {
        "name":        "pg19",
        "description": "sedthh/gutenberg_english (pg19 replacement — Gutenberg books)",
        "raw_dir":     "data/raw/pg19",
        "file_name":   "pg19_train.txt",
    },
    {
        "name":        "openwebtext",
        "description": "Skylion007/openwebtext",
        "raw_dir":     "data/raw/openwebtext",
        "file_name":   "openwebtext_train.txt",
    },
    {
        "name":        "c4",
        "description": "allenai/c4 english (CC-News replacement — no gating)",
        "raw_dir":     "data/raw/c4",
        "file_name":   "c4_train.txt",
    },
    {
        "name":        "csharp",
        "description": "bigcode/the-stack-dedup (c-sharp config, deduped, capped at 2M examples) — Model B/C only",
        "raw_dir":     "data/raw/csharp",
        "file_name":   "csharp_train.txt",
    },
    {
        "name":        "instruct",
        "description": "teknium/OpenHermes-2.5 (instruction/response pairs, template-formatted) — Model B + Model A-Instruct source",
        "raw_dir":     "data/raw/instruct",
        "file_name":   "instruct_train.txt",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SKIP LOGIC
# If output .txt already exists AND is larger than 10MB → skip.
# wikitext, openwebtext, c4 already succeeded → will be skipped.
# Only pg19 (failed) will execute.
# ─────────────────────────────────────────────────────────────────────────────

MIN_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def already_downloaded(raw_dir, file_name):
    path = os.path.join(raw_dir, file_name)
    if os.path.exists(path) and os.path.getsize(path) >= MIN_SIZE_BYTES:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [SKIP] {file_name} already exists ({size_mb:.1f} MB) — skipping.")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def download_wikitext(raw_dir, file_name):
    """
    Salesforce/wikitext wikitext-103-raw-v1.
    Clean Wikipedia-derived English text. ~513MB.
    Already succeeded — SKIP logic handles this.
    """
    print("  Loading Salesforce/wikitext wikitext-103-raw-v1 ...")
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        trust_remote_code=False,
    )
    save_path = os.path.join(raw_dir, file_name)
    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
    return count


def download_pg19(raw_dir, file_name):
    """
    sedthh/gutenberg_english — Project Gutenberg books in parquet format.
    Direct replacement for deepmind/pg19 which uses deprecated script format.
    No trust_remote_code needed — pure parquet, no custom loader script.
    Provides long-form literary English prose for diverse language modelling.
    """
    print("  Loading sedthh/gutenberg_english (Gutenberg books, parquet format) ...")
    ds = load_dataset(
        "sedthh/gutenberg_english",
        split="train",
        trust_remote_code=False,
    )
    save_path = os.path.join(raw_dir, file_name)
    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            # Field name in this dataset is TEXT (uppercase)
            text = item.get("TEXT", "").strip()
            if not text:
                # fallback to lowercase in case schema varies
                text = item.get("text", "").strip()
            if text:
                f.write(text + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"    Books written: {count}", end="\r")
    print()
    return count


def download_openwebtext(raw_dir, file_name):
    """
    Skylion007/openwebtext.
    8M web documents, ~38GB. Already succeeded in previous run.
    SKIP logic handles this — function only runs if file is missing.
    """
    print("  Loading Skylion007/openwebtext ...")
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        trust_remote_code=False,
    )
    save_path = os.path.join(raw_dir, file_name)
    count = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
                count += 1
                if count % 500_000 == 0:
                    print(f"    Examples written: {count:,}", end="\r")
    print()
    return count


def download_c4(raw_dir, file_name):
    """
    allenai/c4 english split.
    Clean web text, no gating. Capped at 2M examples (~4GB).
    Already succeeded in previous run.
    SKIP logic handles this — function only runs if file is missing.
    """
    print("  Loading allenai/c4 english via streaming (capped at 2M examples) ...")
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    save_path = os.path.join(raw_dir, file_name)
    count = 0
    MAX_EXAMPLES = 2_000_000
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "").strip()
            if text:
                f.write(text + "\n")
                count += 1
                if count % 100_000 == 0:
                    print(f"    Examples written: {count:,} / {MAX_EXAMPLES:,}", end="\r")
                if count >= MAX_EXAMPLES:
                    break
    print()
    return count


def download_csharp(raw_dir, file_name):
    """
    bigcode/the-stack-dedup, "c-sharp" config (already near-deduplicated),
    capped at 2M examples.

    NOT bigcode/the-stack-v2: that dataset's parquet shards were confirmed
    (via a schema inspection, not a guess) to contain only metadata —
    blob_id/repo_name/path/etc — with no content or text field at all.
    Its actual file bytes live in Software Heritage's separate
    content-addressed storage and require a different access process
    entirely, so every "successful" download from it would have silently
    written zero real examples. the-stack-dedup's schema was verified to
    have a real `content` column before switching to it (confirmed:
    107 c-sharp shards, ~101k rows each, ~10.8M rows total — comfortably
    over the 2M cap, so at most ~20 shards are ever needed).

    Downloads shards one at a time via hf_hub_download(resume_download=True)
    instead of datasets.load_dataset(streaming=True): the CDN host serving
    this dataset's files (us.aws.cdn.hf.co) intermittently times out
    mid-transfer, and huggingface_hub's downloader resumes from the last
    received byte on retry, while the streaming row-iterator used for the
    other sources restarts the whole HTTP GET from scratch — that's what
    repeatedly stalled/failed here. Each shard's local parquet copy is
    deleted after its content is extracted, so disk usage doesn't grow
    with shards already processed.

    Requires the gated bigcode/the-stack-dedup terms to be accepted on the
    Hub for the logged-in account (separate acceptance from the-stack-v2).

    FIXED 2026-09-14 (was writing raw multi-line file content with only a
    single trailing "\n" as a separator — see docs/MODEL_C_TRAINING_NOTES.md's
    "Sample generations" section for the full bug writeup). That gave
    clean_text.py's per-line cleaning no notion of file boundaries at all:
    it saw one flat stream of every individual source line from every
    file, so its <20-char length filter (tuned for prose junk lines)
    silently deleted every standalone brace/using-directive/etc., and its
    exact-line hash dedup (tuned for redundant prose paragraphs) kept only
    the FIRST occurrence of universally common lines like "using System;"
    project-wide, destroying normal code structure. Now each file's
    content is flattened to a single line (internal whitespace runs,
    including real newlines, collapsed to single spaces via `text.split()`
    + " ".join — the same normalization download_openhermes() already
    applies to multi-turn conversations) BEFORE writing, so each line
    clean_text.py sees is one whole file: the length filter now almost
    never triggers on real code, and dedup now correctly operates at
    whole-file granularity (matching how the-stack-dedup is already
    near-deduplicated), instead of per-internal-line. No changes needed to
    clean_text.py, prepare_splits.py, or tokenize_corpus.py — this fix is
    fully contained to how the raw file gets written.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    REPO_ID = "bigcode/the-stack-dedup"
    MAX_EXAMPLES = 2_000_000
    shard_cache_dir = os.path.join(raw_dir, "_shard_cache")
    os.makedirs(shard_cache_dir, exist_ok=True)

    print("  Listing bigcode/the-stack-dedup c-sharp shards ...")
    api = HfApi()
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    shard_paths = sorted(f for f in all_files if f.startswith("data/c-sharp/") and f.endswith(".parquet"))
    print(f"  Found {len(shard_paths)} shard(s); will stop early once {MAX_EXAMPLES:,} examples are collected.")

    save_path = os.path.join(raw_dir, file_name)
    count = 0
    with open(save_path, "w", encoding="utf-8") as f_out:
        for shard_path in shard_paths:
            if count >= MAX_EXAMPLES:
                break

            print(f"  Downloading shard: {shard_path} ...")
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=shard_path,
                local_dir=shard_cache_dir,
                # resume_download is deprecated as of huggingface_hub — downloads
                # to a local_dir always resume from a partial file automatically.
            )

            parquet_file = pq.ParquetFile(local_path)
            for batch in parquet_file.iter_batches(columns=["content"], batch_size=10_000):
                for value in batch.column("content"):
                    text = (value.as_py() or "").strip()
                    if not text:
                        continue
                    # Flatten to one line per file (see fixed docstring above) —
                    # collapses all internal whitespace runs, including real
                    # newlines/indentation, to single spaces so this whole file
                    # becomes ONE example for clean_text.py's per-line pipeline.
                    flattened = " ".join(text.split())
                    if not flattened:
                        continue
                    f_out.write(flattened + "\n")
                    count += 1
                    if count % 50_000 == 0:
                        print(f"    Examples written: {count:,} / {MAX_EXAMPLES:,}", flush=True)
                    if count >= MAX_EXAMPLES:
                        break
                if count >= MAX_EXAMPLES:
                    break

            try:
                os.remove(local_path)
            except OSError:
                pass

    try:
        os.rmdir(shard_cache_dir)
    except OSError:
        pass  # non-empty (a .cache subdir from hf_hub_download) — harmless to leave behind

    return count


def download_openhermes(raw_dir, file_name):
    """
    teknium/OpenHermes-2.5: ~1M conversation examples (mostly a single
    human -> gpt exchange), stored as ONE 1.9GB JSON array file (not
    JSONL, not sharded parquet — confirmed by listing the repo's files).
    Downloaded whole via hf_hub_download (resumable), then parsed with
    ijson's streaming array iterator rather than json.load() — loading a
    1.9GB JSON array fully into Python objects would multiply well past
    that in memory, the same class of problem hit with the C# raw file's
    227M-line count earlier in this pipeline.

    Each conversation is reformatted onto a SINGLE line (matching the
    "one example per line" convention every other source in this pipeline
    already uses — internal newlines within the turns are flattened to
    spaces):
        ### Instruction: {human turn} ### Response: {gpt turn}
    This is deliberately NOT the multi-line "### Instruction:\n...\n\n###
    Response:\n..." template Model A-Instruct's dedicated fine-tune will
    use — a multi-line version breaks clean_text.py's per-line dedup: the
    literal "### Instruction:" marker line would be byte-identical across
    ~1M examples, so dedup's exact-hash matching would keep only the
    first occurrence and silently drop that marker (and thus the example
    structure) from every other example. Single-line-per-example avoids
    this: dedup now hashes each example's actual full content, and the
    combined line is long enough that the 20-char minimum-length filter
    never triggers either.
    Model B mixes this in as plain text alongside the English/C# sources
    (no loss masking — that only happens in Model A-Instruct's own data
    prep, which will re-derive the structured instruction/response
    boundary from source data directly, not from this flattened file).
    Examples with anything other than a clean human->gpt pair (multi-turn,
    missing a turn, other role names) are skipped to keep the format
    well-defined.
    """
    import ijson
    from huggingface_hub import hf_hub_download

    print("  Downloading teknium/OpenHermes-2.5 (1.9GB, single JSON file) ...")
    local_json_path = hf_hub_download(
        repo_id="teknium/OpenHermes-2.5",
        repo_type="dataset",
        filename="openhermes2_5.json",
    )

    save_path = os.path.join(raw_dir, file_name)
    count = 0
    skipped = 0
    with open(local_json_path, "rb") as f_in, open(save_path, "w", encoding="utf-8") as f_out:
        for item in ijson.items(f_in, "item"):
            turns = item.get("conversations", [])
            human_turn = next((t.get("value") for t in turns if t.get("from") == "human"), None)
            gpt_turn = next((t.get("value") for t in turns if t.get("from") == "gpt"), None)
            if not human_turn or not gpt_turn:
                skipped += 1
                continue
            # Flatten internal newlines to spaces — this must stay a single
            # physical line (see docstring: multi-line breaks per-line dedup).
            human_flat = " ".join(human_turn.strip().split())
            gpt_flat = " ".join(gpt_turn.strip().split())
            formatted = f"### Instruction: {human_flat} ### Response: {gpt_flat}"
            f_out.write(formatted + "\n")
            count += 1
            if count % 50_000 == 0:
                print(f"    Examples written: {count:,} (skipped {skipped:,}) ...", flush=True)

    print(f"  Done: {count:,} examples written, {skipped:,} skipped (missing/malformed turns).", flush=True)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH TABLE
# ─────────────────────────────────────────────────────────────────────────────

DOWNLOAD_FN = {
    "wikitext":    download_wikitext,
    "pg19":        download_pg19,
    "openwebtext": download_openwebtext,
    "c4":          download_c4,
    "csharp":      download_csharp,
    "instruct":    download_openhermes,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[ds["name"] for ds in DATASETS],
        default=None,
        help="Download only this dataset (e.g. csharp). Omit to download all registered datasets.",
    )
    args = parser.parse_args()

    datasets_to_run = DATASETS if args.dataset is None else [ds for ds in DATASETS if ds["name"] == args.dataset]

    print("=" * 60)
    print("Phase 1.1 — Dataset Download")
    if args.dataset:
        print(f"Downloading only: {args.dataset}")
    else:
        print("Downloading all registered datasets.")
    print("Completed datasets will be skipped automatically.")
    print("=" * 60)

    results = {}

    for ds in datasets_to_run:
        name      = ds["name"]
        desc      = ds["description"]
        raw_dir   = ds["raw_dir"]
        file_name = ds["file_name"]

        print(f"\n--- {desc} ---")
        os.makedirs(raw_dir, exist_ok=True)

        # Skip if already downloaded successfully
        if already_downloaded(raw_dir, file_name):
            save_path = os.path.join(raw_dir, file_name)
            size_mb   = os.path.getsize(save_path) / (1024 * 1024)
            results[name] = {"status": "SKIPPED", "size_mb": round(size_mb, 1)}
            continue

        # Download
        try:
            fn    = DOWNLOAD_FN[name]
            count = fn(raw_dir, file_name)
            save_path = os.path.join(raw_dir, file_name)
            size_mb   = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  Saved {count:,} examples -> {save_path}")
            print(f"  File size: {size_mb:.1f} MB")
            results[name] = {
                "status":  "OK",
                "size_mb": round(size_mb, 1),
                "count":   count,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"status": "FAILED", "error": str(e)}

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1.1 — Download Summary")
    print("=" * 60)
    total_mb = 0
    for name, r in results.items():
        status = r["status"]
        if status in ("OK", "SKIPPED"):
            mb = r["size_mb"]
            total_mb += mb
            label = "NEW  " if status == "OK" else "SKIP "
            print(f"  {label}  {name:<15} {mb:>10.1f} MB")
        else:
            print(f"  FAIL   {name:<15} {r.get('error', '')[:55]}")
    print(f"  {'':5}  {'TOTAL':<15} {total_mb:>10.1f} MB")
    print("=" * 60)

    failed = [n for n, r in results.items() if r["status"] == "FAILED"]
    if failed:
        print(f"\nFailed: {failed}")
        print("Fix the error above then re-run.")
        print("Completed datasets will be skipped automatically.")
    else:
        print("\nAll datasets ready.")
        print("Next step: python data/clean_text.py")


if __name__ == "__main__":
    main()
