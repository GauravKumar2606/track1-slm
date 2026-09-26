import argparse
import json
import os
import random

import numpy as np
import sentencepiece as spm

# Builds masked, fixed-length (input_ids, labels) example pairs for an
# instruction fine-tune — structurally different from tokenizer/
# tokenize_corpus.py's flat continuous token stream (used by Model A/B/C's
# from-scratch/continued-pretraining-style training), so it's a separate
# script rather than a mode of that one. See CLAUDE.md's "Three-Model
# Comparison Design" for why this exists (Model A-Instruct).
#
# Source data: data/cleaned/instruct_cleaned.txt, already one example per
# line as "### Instruction: {q} ### Response: {a}" (see
# data/download_datasets.py's download_openhermes() for why it's single-
# line — that was to survive clean_text.py's per-line dedup). Re-split
# each line on " ### Response: " to recover the instruction/response
# boundary needed for masking.
#
# --model flag: only "a-instruct" is wired up today, but kept flag-based
# (not hardcoded) so a future instruction-tuned variant can reuse this
# script — matches every other data-prep script in this pipeline
# (download_datasets.py --dataset, clean_text.py --dataset,
# prepare_splits.py --model, tokenizer/tokenize_corpus.py --model).
MODEL_SOURCES = {
    "a-instruct": "data/cleaned/instruct_cleaned.txt",
}

RESPONSE_MARKER = " ### Response: "
TOKENIZER_MODEL = "tokenizer/tokenizer.model"

LABEL_IGNORE_INDEX = -100
TRAIN_FRAC, VAL_FRAC = 0.90, 0.05  # remainder (0.05) is test
SEED = 42


def load_examples(source_path):
    """Reads the flattened one-line-per-example file and splits each line
    back into (instruction, response) on the response marker. Lines that
    don't contain the marker (shouldn't happen, but data is data) are
    skipped."""
    examples = []
    skipped = 0
    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if RESPONSE_MARKER not in line:
                skipped += 1
                continue
            prompt_part, response = line.split(RESPONSE_MARKER, 1)
            # prompt_part looks like "### Instruction: {instruction}"
            instruction = prompt_part.replace("### Instruction: ", "", 1)
            if not instruction.strip() or not response.strip():
                skipped += 1
                continue
            examples.append((instruction.strip(), response.strip()))
    print(f"Loaded {len(examples):,} examples ({skipped:,} skipped — missing marker or empty field).", flush=True)
    return examples


def build_example(sp, instruction, response, context_length):
    """Tokenizes one (instruction, response) pair into fixed-length
    input_ids/labels arrays (length context_length+1, so callers can take
    x = ids[:-1], y = ids[1:] the same way training/train.py's get_batch
    does for the from-scratch models).

    labels masks (LABEL_IGNORE_INDEX) every position whose TARGET token is
    part of the prompt or padding — loss only counts on response tokens
    (+ the trailing EOS), so the model is graded on generating a good
    answer, not on reproducing the question.
    """
    prompt_text = f"### Instruction: {instruction}{RESPONSE_MARKER}"
    prompt_ids = [sp.bos_id()] + sp.encode(prompt_text, out_type=int)
    response_ids = sp.encode(response, out_type=int) + [sp.eos_id()]

    full_ids = prompt_ids + response_ids
    max_len = context_length + 1

    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        if overflow < len(prompt_ids):
            # Truncate from the FRONT of the prompt (drop earliest/least
            # relevant context first), keep the full response intact.
            prompt_ids = prompt_ids[overflow:]
        else:
            # Prompt alone doesn't fit even after dropping all of it (rare) —
            # keep just the BOS + response, truncating the response itself.
            prompt_ids = prompt_ids[:1]
            response_ids = response_ids[: max_len - len(prompt_ids)]
        full_ids = prompt_ids + response_ids

    prompt_len = len(prompt_ids)
    real_len = len(full_ids)

    pad_needed = max_len - real_len
    padded_ids = full_ids + [sp.pad_id()] * pad_needed

    ids_arr = np.asarray(padded_ids, dtype=np.uint16)
    x = ids_arr[:context_length]
    y = ids_arr[1 : context_length + 1].astype(np.int16)

    # Mask every y position whose target (the token it's predicting) falls
    # in the prompt portion or in padding.
    for j in range(context_length):
        target_pos = j + 1
        if target_pos < prompt_len or target_pos >= real_len:
            y[j] = LABEL_IGNORE_INDEX

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SOURCES), default="a-instruct")
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    source_path = MODEL_SOURCES[args.model]
    out_dir = f"data/tokenized/model-{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Preparing instruction data for model-{args.model} from {source_path}", flush=True)
    examples = load_examples(source_path)

    rng = random.Random(SEED)
    rng.shuffle(examples)

    n = len(examples)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)
    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)

    for split_name, split_examples in splits.items():
        print(f"--- Tokenizing {split_name}: {len(split_examples):,} examples ---", flush=True)
        x_out = np.zeros((len(split_examples), args.context_length), dtype=np.uint16)
        y_out = np.zeros((len(split_examples), args.context_length), dtype=np.int16)

        for i, (instruction, response) in enumerate(split_examples):
            x, y = build_example(sp, instruction, response, args.context_length)
            x_out[i] = x
            y_out[i] = y
            if (i + 1) % 50_000 == 0:
                print(f"    {i + 1:,} / {len(split_examples):,}", flush=True)

        x_path = os.path.join(out_dir, f"{split_name}_input_ids.bin")
        y_path = os.path.join(out_dir, f"{split_name}_labels.bin")
        x_out.tofile(x_path)
        y_out.tofile(y_path)
        print(f"Saved {x_path} and {y_path} ({len(split_examples):,} examples x {args.context_length} tokens)", flush=True)

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "context_length": args.context_length,
                "label_ignore_index": LABEL_IGNORE_INDEX,
                "pad_id": sp.pad_id(),
                "num_examples": {name: len(exs) for name, exs in splits.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved {meta_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
