# Session Notes — Data Pipeline, Tokenizer, Model, and Training-Prep Fixes

This documents the work done in this session, in order: diagnosing and fixing
the Phase 1 data pipeline, running Phase 2 (tokenizer) and Phase 3 (model
architecture), writing the Phase 4/5/6/7 scripts, and the bugs/decisions
found while preparing for the actual Model A training run.

## 1. Phase 1 — Data cleaning pipeline fixes

**Problem reported:** `clean_text.py` kept referencing a `bookcorpus`
dataset that was never downloaded, and the cleaning script hung with no
visible progress on `openwebtext` (38GB).

**Root causes found and fixed:**
- `clean_text.py`'s dataset registry (`wikitext`, `bookcorpus`, `openwebtext`,
  `ccnews`) was stale — `download_datasets.py` actually downloads
  `wikitext`, `pg19`, `openwebtext`, `c4`. Registry corrected to match, and
  `pg19` (18.6GB) was previously invisible to the size-check entirely.
- `prepare_splits.py` was hardcoded to a single dataset and an 80/10/10
  split, contradicting its own docstring and the 90/5/5 requirement.
  Rewritten to dynamically combine whatever `clean_text.py` produced and
  split 90/5/5.
- The streaming cleaner used SQLite for deduplication, doing one
  `SELECT` + `INSERT` per paragraph with no batching — this saturated disk
  I/O (WAL checkpointing) and produced zero visible progress. Replaced with
  an in-memory hash set (blake2b, 8-byte digest) — no disk round-trips,
  ~40x+ throughput improvement, and added per-chunk progress logging with
  `flush=True` (stdout is block-buffered when not a TTY, which is why the
  original progress prints never appeared).
- **Critical bug:** the paragraph splitter used `'\n\n'` (blank-line
  separated paragraphs), but `download_datasets.py` writes one example per
  line (single `\n`). Confirmed via direct byte inspection — zero
  double-newlines in the raw files. This meant whole files were being
  treated as a single "paragraph". Fixed to split on single `\n`, with
  careful handling of lines split across streaming chunk boundaries.
- **Critical bug (found via SentencePiece training):** `clean_text_segment`'s
  ASCII filter (`encode('ascii', 'ignore')`) let NUL and other control
  characters (0x00–0x1F, 0x7F) through, since they're valid ASCII code
  points. A stray NUL byte at ~42MB into a 54GB `train.txt` caused
  SentencePiece's C++ trainer to silently truncate its input there —
  training "succeeded" but only saw 0.08% of the corpus. Fixed
  `clean_text.py` to also strip control characters, and wrote
  `data/sanitize_corpus.py` as a one-off remediation pass over the
  already-split `data/model-a/{corpus,train,val,test}.txt` files (since
  `data/raw/` had already been deleted to free disk space, so a full
  re-clean from raw wasn't an option). Removed 1,137,472 control characters
  across those files; confirmed zero NUL bytes remain afterward.
- Also hit and resolved a `MemoryError` (a naive `f.read()` on a 60GB
  combined corpus) and a full disk (`data/raw/` + `data/cleaned/` +
  `data/model-a/` together exceeded available space) — `prepare_splits.py`'s
  split step was rewritten to stream rather than load-all, and `data/raw/`
  was deleted (with explicit user confirmation) once cleaning was done.

**Final Phase 1 numbers:**
- `data/model-a/corpus.txt`: 59,359,554,808 characters (~60GB)
- `train.txt` / `val.txt` / `test.txt`: 90/5/5 split
- Estimated tokens (char/4 heuristic): ~14.84B

## 2. Phase 2 — Tokenizer

- `tokenizer/train_tokenizer.py`: SentencePiece BPE, vocab_size 32000,
  trained on a 10M-line shuffled sample of `train.txt` (not the full 54GB —
  standard practice; vocab quality plateaus well before that much data).
  Outputs `tokenizer.model` (authoritative, used by all other scripts),
  `tokenizer.vocab`, `tokenizer.json` (full vocab dump for
  inspection/interop), `vocab.txt`.
- `tokenizer/verify_tokenizer.py`: roundtrip-tested 3 required sentences —
  all passed. (Hit and fixed a Windows console encoding issue: `▁` and other
  SentencePiece unicode markers can't print under the default cp1252
  console codepage — added `sys.stdout.reconfigure(encoding="utf-8")`.)
- `tokenizer/tokenize_corpus.py`: tokenizes train/val/test into `uint16`
  binary chunks in `data/tokenized/model-a/`. Generalized to accept
  `--model a|b` (defaults to `a`) for reuse in Phase 4's Model B step.
  **Final token counts:** train 12,016,279,511 / val 727,573,115 /
  test 712,448,359 — **total 13,456,300,985 tokens.**

## 3. Phase 3 — Model architecture

`model/config.py`, `attention.py`, `transformer.py`, `model.py`,
`verify_model.py` — standard GPT-style decoder-only transformer (token +
positional embedding, N pre-norm decoder blocks, final LayerNorm, weight-tied
output projection). **Verified: 110,025,216 parameters (110.0M)**, forward
pass shape `(2, 64, 32000)` confirmed, ~0.5GB peak GPU memory.

Later optimized `attention.py` to use `torch.nn.functional.
scaled_dot_product_attention` (fused/flash-attention kernel) instead of the
original hand-written QK^T/softmax/mask — same math, faster, less memory,
no quality change. Re-verified after the change (still 110.0M params,
correct output shape).

## 4. Phase 4 & 5 — Training and Evaluation scripts

Per your instruction, Phase 4 (Training) and Phase 5 (Evaluation) are yours
to run manually — but the code itself still needed writing (only the actual
multi-hour GPU runs are deferred; the AUTO-EXECUTE list already covered
writing `compare_models.py`/`generate_report.py`, and `train.py`/
`validate.py` had to exist for you to have anything to run at all). Wrote:
- `training/train.py` — AdamW, linear warmup + cosine decay LR, grad
  clipping (max norm 1.0), fp16 mixed precision, TensorBoard logging
  (train_loss/lr/tokens_per_sec every 100 steps, val_loss every 500),
  checkpointing to `checkpoints/model-{a,b}/`, auto-resume from latest
  checkpoint. HARD STOP per CLAUDE.md — never auto-executed.
- `training/validate.py` — val loss, perplexity, one sample generation.
- `training/benchmark_throughput.py` — **not** `train.py`, so outside the
  HARD STOP rule; runs a handful of real train-step iterations to measure
  actual tokens/sec and VRAM on this machine before committing to the real
  multi-hour run. See "Batch size" section below — this caught two bugs
  that would have crashed the real training run.
- `data/prepare_model_b.py` — downloads the C# subset of
  `bigcode/the-stack-v2` (note: gated on HuggingFace, needs
  `huggingface-cli login` + accepting the dataset's terms) and combines it
  with Model A's corpus for Model B.
- `evaluation/compare_models.py`, `evaluation/perplexity.py` (HARD STOP),
  `evaluation/generate_report.py` — per the original spec.

**Two real bugs found and fixed via the benchmark script (would have
crashed the actual `train.py` run hours in):**
1. `learning_rate: 3e-4` in the YAML was read by PyYAML as the *string*
   `"3e-4"` (a known PyYAML quirk — exponent notation without a decimal
   point isn't auto-parsed as float). Fixed in `ModelConfig.from_yaml` by
   explicitly coercing `dropout` and `learning_rate` to `float`.
2. Random batch sampling (`np.random.randint(0, max_start, size=batch_size)`)
   defaults to int32 bounds (~2.1 billion), but the training set has ~12
   billion tokens — this would raise `ValueError: high is out of bounds for
   int32` on the very first batch. Fixed by adding `dtype=np.int64`
   everywhere this pattern appears (`train.py`, `validate.py`,
   `benchmark_throughput.py`).

## 5. Batch size — the actual discussion and decision

The original config (`batch_size: 32`, `max_steps: 100000`) targets a
~1.64 billion token training budget (batch_size × context_length ×
max_steps), which is a sensible target for a 110M-parameter model (roughly
in line with the "Chinchilla" ~20 tokens/parameter compute-optimal
heuristic — 110M × 20 ≈ 2.2B).

Benchmarked real throughput on this GPU (RTX 5070 Ti Laptop, ~12.8GB usable
VRAM) at a fixed step count (100,000) first:

| batch_size | tokens/sec | peak VRAM | time for 100,000 steps |
|---|---|---|---|
| 8  | 38,814 | 4.65 GB  | 2.9h |
| 16 | 41,939 | 7.71 GB  | 5.4h |
| 24 | 42,451 | 10.74 GB | 8.0h |
| 32 (config default) | 10,234 | 13.77 GB | 44.5h |

batch_size=32 exceeds real VRAM capacity and silently spills into Windows'
slow shared-GPU-memory fallback (system RAM standing in for VRAM), which is
why its throughput collapses to a quarter of the smaller batch sizes'.

**Important correction made during discussion:** comparing at a *fixed step
count* is misleading, because `total tokens trained = steps × batch_size ×
context_length` — a smaller batch size at the same step count trains on
proportionally fewer tokens (batch=8 at 100,000 steps sees only 410M tokens,
vs. 1.64B at batch=32 — 4x less actual learning, not a fair comparison).
The correct comparison holds the **token budget** fixed (~1.64B, the
original design target) and adjusts `max_steps` per batch size accordingly:

| batch_size | max_steps for ~1.64B tokens | tokens/sec | time |
|---|---|---|---|
| 8  | ~400,391 | 38,814 | 11.7h |
| 16 | ~200,195 | 41,939 | 10.9h |
| **24** | **~133,500** | **42,451** | **~10.7h** |
| 32 (overflowing) | 100,000 | 10,234 | 44.5h |

At equal training amount, batch_size=24 is both the fastest *and* trains on
the originally-intended token budget — batch_size=8's apparent "2.9h"
advantage was an artifact of doing 4x less work, not genuine efficiency.

**Decision: `configs/model_117M.yaml` updated to `batch_size: 24`,
`max_steps: 133500`** (all other fields unchanged). This keeps the same
~1.64B-token training target as the original config, at ~10.7h instead of
~44.5h, purely by avoiding the VRAM-overflow inefficiency at batch=32.
`batch_size: 16` / `max_steps: 200000` (~10.9h, ~60% VRAM instead of ~84%)
was offered as a safer-margin alternative but not chosen.

## 6. Concepts discussed (for future reference)

- **Step-based vs. epoch-based training:** training here is defined by a
  token/step budget, not epochs over the corpus, because the corpus
  (~12B tokens) vastly exceeds what a 110M-parameter model can usefully
  absorb (an epoch would be ~5.5x the compute-optimal token budget) — the
  same reason every modern LLM (GPT-2/3, LLaMA, etc.) is described in terms
  of a token budget rather than corpus passes. Random per-step sampling
  (`get_batch`) is also far simpler than epoch bookkeeping over a
  multi-billion-token file.
- **What `batch_size` does:** the number of independent sequences processed
  in parallel per gradient update — a unit of parallelism/GPU utilization,
  not "amount of training per step" in isolation. Larger batches use the
  GPU more efficiently (up to the point they exceed VRAM), but each step
  also covers proportionally more tokens, so batch_size and `max_steps`
  must be considered together, not independently.
- **Algorithm used:** AdamW optimizer, linear warmup + cosine LR decay,
  gradient clipping (max norm 1.0), fp16 mixed precision with a
  `GradScaler`, cross-entropy next-token-prediction loss. Standard GPT-2/
  nanoGPT-style recipe.
- **"GPT-style decoder-only"**: the original Transformer has an encoder
  (reads full input) and decoder (generates output, with causal masking).
  GPT-style models drop the encoder and use only the causally-masked
  decoder stack, so each position can only attend to itself and earlier
  positions — this is what allows autoregressive one-token-at-a-time
  generation.
- **Live progress / not a black box:** `train.py` logs to TensorBoard every
  100 steps (`train_loss`, `learning_rate`, `tokens_per_second`) and every
  500 steps (`val_loss`). Run `tensorboard --logdir logs/model-a` in a
  second terminal and open `http://localhost:6006` to watch live-updating
  curves while training runs; the training terminal itself also prints a
  line every 100 steps.
- **Manual intervention during training:** none required — LR schedule,
  gradient clipping, and best-checkpoint saving are all automatic. If
  interrupted, rerunning the same command auto-resumes from the latest
  checkpoint in `checkpoints/model-a/`.

## 7. What's left to run (in order) — see CLAUDE.md for the exact commands

1. `python training/train.py --model a --config configs/model_117M.yaml`
   (HARD STOP — manual, ~10.7h at the new config)
2. (optional) `python training/validate.py --checkpoint checkpoints/model-a/best.pt --model a`
3. `python data/prepare_model_b.py` (needs HuggingFace login for the gated
   `bigcode/the-stack-v2` dataset)
4. `python tokenizer/tokenize_corpus.py --model b`
5. `python training/train.py --model b --config configs/model_117M.yaml`
   (HARD STOP — manual)
6. `python evaluation/compare_models.py`
7. `python evaluation/perplexity.py` (HARD STOP — manual)
8. `python evaluation/generate_report.py`

Once Model A alone is trained (step 1), Phase 6 (`inference/generate.py`,
`api/main.py`) and Phase 7 (`deployment/quantize.py`, `deployment/app.py`)
code is already written and ready to actually run/verify against it.
