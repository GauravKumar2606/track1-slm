import argparse
import os
import time
import numpy as np
import sentencepiece as spm

TOKENIZER_MODEL = "tokenizer/tokenizer.model"
CHUNK_CHARS = 50 * 1024 * 1024  # 50M chars per read, keeps memory bounded


def tokenize_split(sp, name, path, out_dir):
    out_path = os.path.join(out_dir, f"{name}.bin")
    total_size = os.path.getsize(path)
    bytes_read = 0
    total_tokens = 0
    start_time = time.time()

    with open(path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(out_path, "wb") as f_out:
        leftover = ""
        chunk_index = 0
        while True:
            chunk = f_in.read(CHUNK_CHARS)
            is_last = not chunk
            if is_last and not leftover:
                break
            chunk_index += 1
            bytes_read += len(chunk.encode("utf-8", errors="ignore"))

            text = leftover + chunk
            if is_last:
                leftover = ""
            else:
                split_at = text.rfind("\n")
                if split_at == -1:
                    leftover = text
                    continue
                leftover = text[split_at + 1:]
                text = text[:split_at]

            chunk_tokens = 0
            for line in text.split("\n"):
                if not line:
                    continue
                ids = sp.encode(line, out_type=int)
                ids.append(sp.eos_id())
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(f_out)
                chunk_tokens += len(ids)
            total_tokens += chunk_tokens

            elapsed = time.time() - start_time
            pct = (bytes_read / total_size) * 100 if total_size else 100
            print(
                f"  [{name}] chunk {chunk_index}: {bytes_read / (1024**2):.0f} MB / "
                f"{total_size / (1024**2):.0f} MB ({pct:.1f}%) | "
                f"+{chunk_tokens:,} tokens this chunk | elapsed {elapsed:.0f}s",
                flush=True,
            )

    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", "c"], default="a")
    args = parser.parse_args()

    splits = {
        "train": f"data/model-{args.model}/train.txt",
        "val": f"data/model-{args.model}/val.txt",
        "test": f"data/model-{args.model}/test.txt",
    }
    out_dir = f"data/tokenized/model-{args.model}"

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    os.makedirs(out_dir, exist_ok=True)

    report = {}
    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found.", flush=True)
            continue
        print(f"--- Tokenizing {name} ({path}) ---", flush=True)
        n_tokens = tokenize_split(sp, name, path, out_dir)
        report[name] = n_tokens
        out_path = os.path.join(out_dir, f"{name}.bin")
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"Saved {out_path} ({size_mb:.1f} MB, {n_tokens:,} tokens)", flush=True)

    total = sum(report.values())
    print("\n==========================================================", flush=True)
    print("Tokenization complete.", flush=True)
    for name, n_tokens in report.items():
        print(f"  - {name}: {n_tokens:,} tokens", flush=True)
    print(f"Total tokens across all splits: {total:,}", flush=True)
    print("===============================================================", flush=True)


if __name__ == "__main__":
    main()
