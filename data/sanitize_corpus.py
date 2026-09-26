"""
One-off remediation script.

clean_text.py's ASCII filter (encode('ascii','ignore')) strips bytes >127 but
lets control characters (0x00-0x1F, 0x7F) through untouched, since they are
valid ASCII code points. Raw scraped text (c4/openwebtext/pg19) had embedded
NUL bytes that survived cleaning. SentencePiece's C++ trainer silently
truncates its input at the first NUL byte in a file, which cut tokenizer
training down to the first ~43MB of a 54GB train.txt without erroring.

This strips control characters (keeping \t \n \r) from the already-cleaned
and already-split corpus files in place, since data/raw/ has been deleted
and a full re-clean isn't possible without re-downloading everything.
"""
import os
import time

FILES = [
    "data/model-a/corpus.txt",
    "data/model-a/train.txt",
    "data/model-a/val.txt",
    "data/model-a/test.txt",
]

CHUNK_CHARS = 50 * 1024 * 1024  # 50M chars per read

# Remove all C0 control chars and DEL except tab/newline/carriage-return.
REMOVE_CODEPOINTS = [c for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)] + [0x7F]
TRANSLATE_TABLE = {cp: None for cp in REMOVE_CODEPOINTS}


def sanitize_file(path):
    tmp_path = path + ".sanitizing"
    total_size = os.path.getsize(path)
    bytes_read = 0
    removed_count = 0
    start_time = time.time()

    with open(path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(tmp_path, "w", encoding="utf-8") as f_out:
        chunk_index = 0
        while True:
            chunk = f_in.read(CHUNK_CHARS)
            if not chunk:
                break
            chunk_index += 1
            bytes_read += len(chunk.encode("utf-8", errors="ignore"))

            cleaned = chunk.translate(TRANSLATE_TABLE)
            removed_count += len(chunk) - len(cleaned)
            f_out.write(cleaned)

            elapsed = time.time() - start_time
            pct = (bytes_read / total_size) * 100 if total_size else 100
            print(
                f"  [{os.path.basename(path)}] chunk {chunk_index}: "
                f"{bytes_read / (1024**2):.0f} MB / {total_size / (1024**2):.0f} MB "
                f"({pct:.1f}%) | control chars removed so far: {removed_count} | "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    os.replace(tmp_path, path)
    print(f"Sanitized {path}: removed {removed_count} control characters.", flush=True)
    return removed_count


def main():
    total_removed = 0
    for path in FILES:
        if not os.path.exists(path):
            print(f"Skipping {path}: not found.", flush=True)
            continue
        print(f"--- Sanitizing {path} ---", flush=True)
        total_removed += sanitize_file(path)

    print("\n==========================================================", flush=True)
    print(f"Sanitization complete. Total control characters removed: {total_removed}", flush=True)
    print("===============================================================", flush=True)


if __name__ == "__main__":
    main()
