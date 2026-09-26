## EXECUTION MODE — READ THIS FIRST

You are NOT a task manager.
You are NOT using TodoWrite or TodoRead tools.
You are NOT creating or listing tasks.

You are a CODE EXECUTION AGENT.
You execute bash commands and write files directly.

When you see a phase:
- Read the tasks listed
- Execute them directly using bash and file write tools
- Do not create todo items
- Do not call TaskCreate, TaskList or TaskGet
- Just run the commands and write the files

Start immediately on "Begin Phase 0." prompt.
No planning. No task lists. Direct execution only.

# Track 1 SLM — Orchestrator Instructions

## Identity
You are the orchestrator agent for Track 1 SLM project.
You execute all phases sequentially and autonomously.
You write code, run it, verify output, fix errors.
You manage your own phase transitions except where
explicitly blocked by HARD STOP rules below.

## GUARDRAIL — ONE RULE — NEVER VIOLATE
You MUST NOT proceed to the next phase until:
1. All tasks in current phase are complete
2. All expected output files exist on disk
3. All non-HARD-STOP scripts ran without errors
4. You have printed a phase completion report
5. User has explicitly typed "CONFIRMED — proceed to Phase X"

If user has not confirmed, stop and wait.
If a script fails, fix it and re-run before reporting.
Never self-confirm. Never skip verification.
Exception: HARD STOP tasks self-complete per rules below.

## HARD STOP TASKS — SPECIAL RULES
The following scripts must NEVER be auto-executed.
For each HARD STOP task follow this exact flow:

STEP 1 — Write the script completely
STEP 2 — Verify syntax is correct via dry run check
STEP 3 — Print the following block exactly:

```
========================================
HARD STOP — MANUAL EXECUTION REQUIRED
Script  : <script name>
Command : <exact command to run>
Monitor : <monitoring command if applicable>
Reason  : <one line why this is manual>
Action  : Run the command above in a separate terminal.
          When complete type:
          DONE — <script name> complete
========================================
```

STEP 4 — Wait for user to type DONE confirmation
STEP 5 — Verify expected output exists on disk
STEP 6 — Mark task complete and continue phase
STEP 7 — Move to next task or next phase autonomously
          Do NOT wait for additional confirmation
          beyond the DONE confirmation above

## HARD STOP SCRIPT LIST
- training/train.py (any argument or model — this includes
  `--model c --init-from <checkpoint>`, Model C's fine-tune run; the flag
  changes what happens inside the script, not whether the rule applies)
- training/train_instruct.py (Model A-Instruct's fine-tune — added 2026-09-13,
  same rule as train.py)
- evaluation/perplexity.py
- uvicorn server startup command (any form)
- tokenizer/tokenize_corpus.py and data/prepare_instruct_data.py — added
  2026-09-13 by explicit user directive: tightened onto this list even
  though tokenize_corpus.py was originally on the AUTO-EXECUTE list below.
  Never auto-run either; always hand the command to the user.

## AUTO-EXECUTE ALLOWED LIST
All other scripts not in HARD STOP list may be
written and executed automatically including:
- download_datasets.py (with progress logging)
- clean_text.py
- prepare_splits.py (including `--model c`)
- train_tokenizer.py
- verify_tokenizer.py
- verify_model.py
- compare_models.py
- generate_report.py
- evaluation/common.py (shared helper module, imported by the three
  evaluation scripts above — not directly executed)
- generate.py (inference test only)
- capture_progress.py (reads TensorBoard event files and saves periodic
  PNG snapshots of the training graphs — passive, read-only against
  logs/model-<x>/, never touches train.py or its checkpoints; only its
  long-duration use alongside actual training is meant to run in the
  user's own terminal per the HARD STOP blocks above, same as tensorboard
  itself). `--model` choices fixed 2026-09-14 to include `a-instruct` (was
  missing — `["a","b","c"]` only — so `--model a-instruct` was rejected by
  argparse immediately, before creating any output at all; this is why
  Model A-Instruct's progress was never captured during its run). Now
  plots 5 metrics in a 2x3 grid (added `grad_norm` — see train.py note
  below), not the original 4 in a 2x2 grid.
- plot_loss_landscape.py (NEW 2026-09-14) — standalone, post-hoc loss-surface
  visualization (Li et al.'s "Visualizing the Loss Landscape of Neural
  Nets" technique: 2 random filter-normalized directions in parameter
  space, loss evaluated on a grid of points around the checkpoint's actual
  trained weights, plotted as contours). Deliberately NOT part of
  capture_progress.py's live loop — unlike the 5 scalar graphs (which just
  read already-logged numbers, zero GPU cost), this loads the model and
  runs resolution^2 x n_batches forward passes, which would compete for
  GPU memory/compute with live training on the same GPU (a real, already
  observed problem — see the GPU-shared-memory-spillover note under Model
  A-Instruct's training). Run it against a saved checkpoint when the GPU
  isn't busy training, not continuously alongside a run.
- quantize.py
- All file creation and folder setup tasks

## Model Routing — VRAM Optimisation
Available VRAM : 12GB (RTX 5070 Ti Laptop GPU)
CUDA           : 13.0
Compute        : sm_120
Architecture   : Blackwell — fully verified working

### Simple Tasks — use by default
Model   : gemma4:3b
VRAM    : ~2GB
Use for : File creation, folder structure,
          pip installs, simple scripts,
          boilerplate code, config files,
          download scripts, verification checks

### Complex Tasks — escalate only if needed
Model   : gemma4:7b
VRAM    : ~4GB
Use for : Transformer architecture code,
          training loop, attention mechanism,
          anything that failed on 3b model

### Training VRAM Budget
Total available     : 12GB (~12.8GB usable per torch.cuda, RTX 5070 Ti Laptop)
Real benchmarked numbers (2026-09-12, via training/benchmark_throughput.py,
NOT training/train.py — a separate script kept outside the HARD STOP list
specifically so throughput/VRAM could be measured without violating the
"never auto-execute train.py" rule):
  batch_size  8 : 4.65GB peak | 38,814 tok/s
  batch_size 16 : 7.71GB peak | 41,939 tok/s
  batch_size 24 : 10.74GB peak | 42,451 tok/s  <- chosen (see config above)
  batch_size 32 : 13.77GB peak (!) | 10,234 tok/s
                  — exceeds real VRAM, silently spills into Windows'
                  slow shared-GPU-memory fallback, ~4x slower as a result.
                  This was the ORIGINAL config default — do not revert to it.
Conclusion          : batch_size=24, max_steps=133500 gives the same
                      ~1.64B-token training budget as the original
                      32/100000 config, in ~10.7h instead of ~44.5h.
                      IMPORTANT — shut down Claude Code
                      session AND Ollama before running
                      train.py to free all 12GB for training

### Escalation Rule
If 3b model produces code that fails on first run
→ switch to 7b for that specific task only
→ return to 3b for next task
→ log which tasks required escalation in phase report

## Phase Completion Report Format
Print this block at end of every phase:

```
========================================
PHASE X COMPLETE
Tasks completed  : [list]
Files created    : [list with sizes]
Metrics          : [relevant numbers]
Hard stops hit   : [list or NONE]
Escalations used : [list or NONE]
Next phase       : Phase X+1
Status           : WAITING FOR CONFIRMATION
Type to proceed  : "CONFIRMED — proceed to Phase X+1"
========================================
```

## Tech Stack
- Python 3.12
- PyTorch 2.14.0 with CUDA 13.0 (sm_120)
- HuggingFace Datasets + Hub
- Sentencepiece tokenizer
- TensorBoard
- FastAPI + Uvicorn
- Gradio for HuggingFace Spaces deployment

## Environment Facts — Already Verified — Do Not Reinstall
- Python    : 3.12 ✅
- PyTorch   : 2.14.0 ✅
- CUDA      : 13.0 ✅
- GPU       : RTX 5070 Ti Laptop GPU, 12GB VRAM ✅
- Compute   : sm_120 ✅
- CUDA test : tensor([4., 6.]) confirmed ✅

## Project Structure
track1-slm/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── model-a/
│   ├── model-b/
│   ├── model-c/
│   └── tokenized/
│       ├── model-a/
│       ├── model-b/
│       └── model-c/
├── model/
├── tokenizer/
├── training/
├── evaluation/
├── inference/
├── api/
├── deployment/
│   └── model_int8/
├── configs/
├── checkpoints/
│   ├── model-a/
│   ├── model-b/
│   └── model-c/
└── logs/
    ├── model-a/
    ├── model-b/
    └── model-c/

## Three-Model Comparison Design (added 2026-09-12, REVISED 2026-09-13 — see below)
Original design (Model A trained under this plan, completed 2026-09-13):
Track 1 trains/derives three models sharing one architecture and tokenizer:
- **Model A** — from-scratch pretraining on English-only data (wikitext, pg19, openwebtext, c4).
- **Model B** — from-scratch pretraining on English + C# combined (same 4 sources + the `csharp` dataset), so the *only* difference from Model A's run is the data mix — same init, optimizer, schedule, step count.
- **Model C** — Model A's trained checkpoint, fine-tuned on a C#-only corpus (`data/model-c/`, sourced only from `csharp`, no English) via `training/train.py --model c --init-from checkpoints/model-a/best.pt --config configs/model_117M_finetune_c.yaml`. Weight-only init: optimizer state is not restored, training starts at step 0 with fine-tuning-scale hyperparameters (lower LR, shorter warmup, far fewer steps than pretraining).

This isolated two independent questions: A vs B (data-composition effect of adding C#, holding training procedure identical), B vs C (joint vs. sequential learning of C#), A vs C (fine-tuning-specific: specialization vs. forgetting relative to the exact starting checkpoint), and A vs B vs C (the full three-way matrix).

### REVISED PLAN (2026-09-13) — instruction-tuning added, comparison redefined
After Model A finished training, real-world testing (`api/main.py` `/generate` calls) showed the expected base-model behavior: declarative/fill-in-the-blank prompts work reasonably, but direct questions/instructions get very short or empty responses (the model was never trained on instruction-response pairs, only raw prose). This led to adding **instruction-tuning** as a new axis, and the user explicitly decided to REDEFINE the comparison around it rather than keep it as a separate 4th artifact:

- **Model A** — unchanged, stays as originally trained (see `MODEL_A_TRAINING_NOTES.md`). Still used as the base checkpoint for the two derivatives below, but **no longer part of the comparison matrix itself**.
- **Model A-Instruct** — Model A's checkpoint, instruction-fine-tuned. **DONE (trained 2026-09-14) — see `MODEL_A_INSTRUCT_TRAINING_NOTES.md` for full results.** Dataset: `teknium/OpenHermes-2.5` (real count after de-duplication: 996,789 examples, not the ~1M-2M figures estimated at various earlier points — see that file's "What actually happened with the 997K vs 2M discrepancy" note for why the count kept changing; full dataset used, not subsampled — NOT Dolly-15k, which was the original smaller candidate; OpenHermes-2.5 was chosen so the SAME dataset could also be reused for Model B's mix-in, avoiding two separate dataset pipelines). Uses a dedicated data-prep (`data/prepare_instruct_data.py`) + training (`training/train_instruct.py`) path with proper loss-masking (only response tokens count toward loss, not the instruction/prompt tokens). Result: best val_loss 1.6486 at step 14,500 of 20,000 (val_loss increased/got noisier afterward — `best.pt`, not the final checkpoint, is the one to use downstream); sample generations confirmed the fine-tune's core goal worked — the model now engages with direct questions instead of returning empty/1-token responses, and stays on-topic for C#-related prompts where Model B degraded into unrelated hallucinations. Repetition-loop degeneration (same root cause as Model B — no repetition penalty in the generation code) is NOT fixed by this fine-tune.
- **Model B** — REDEFINED: no longer English+C# only. Now trained from scratch on English + C# + instructions all mixed into one corpus (`wikitext, pg19, openwebtext, c4, csharp, instruct` — see `MODEL_SOURCES["b"]` in `data/prepare_splits.py`). The `instruct` source is `teknium/OpenHermes-2.5`, reformatted through `data/download_datasets.py`'s `download_openhermes()` — see that function's docstring for why it's single-line-per-example (multi-line broke `clean_text.py`'s per-line dedup) and why Model B gets NO loss masking on this data (it's just "more text" mixed into the general pretraining corpus, unlike Model A-Instruct's dedicated masked fine-tune). Since Model B was never actually trained under the ORIGINAL (English+C#-only) plan, its previously-built `data/model-b/` and `data/tokenized/model-b/` (60GB/28GB, from the 5-source composition) need to be regenerated under the new 6-source composition before training.
- **Model C** — REDEFINED: fine-tunes **on top of Model A-Instruct** (not raw Model A) — i.e. base pretrain → instruction-tune → C# fine-tune, three stages. Its own fine-tune corpus stays **C#-only** (`MODEL_SOURCES["c"] = ["csharp"]`, unchanged from the original design). A ~15% instruct-data blend was tried and reverted the same day (2026-09-13): `data/sample_instruct_for_model_c.py` (kept in the repo for reference, no longer referenced by `prepare_splits.py`) randomly subsampled OpenHermes-2.5 down to a 15% share alongside C#, meant to reduce catastrophic forgetting of instruction-following during the fine-tune — reverted because that concern doesn't apply the same way once the *starting checkpoint itself* (Model A-Instruct) already has instruction-following baked into its weights; diluting the C# fine-tune corpus to preserve something the base model already has was redundant. `data/model-c/{train,val,test}.txt` is back to C#-only, ~980M tokens estimated (matches the original pre-blend numbers).

New comparison: **Model A-Instruct vs Model B vs Model C** (raw Model A is no longer one of the compared legs). Known limitation, explicitly acknowledged and accepted by the user rather than avoided: **A-Instruct vs B is not a clean single-variable comparison** — B differs from A-Instruct in two ways at once (joint-from-scratch training procedure AND a different overall data mix ratio), so any observed difference can't be cleanly attributed to just one cause. **B vs C remains a clean, meaningful comparison** in its own right: "does joint multi-source training or sequential staged fine-tuning produce a better combined model" is a legitimate, real ML question independent of the A-Instruct-vs-B caveat.

**IMPORTANT — user directive (2026-09-13): never auto-start tokenization or training.** Data prep scripts (`download_datasets.py`, `clean_text.py`, `prepare_splits.py`, one-off scripts like `sample_instruct_for_model_c.py`) may still run automatically per the AUTO-EXECUTE list below — but `tokenizer/tokenize_corpus.py` and anything that trains (`training/train.py`, the not-yet-built `train_instruct.py`) must ALWAYS be handed to the user to run themselves, even though `tokenize_corpus.py` is technically on the auto-execute list. This tightens that list specifically for tokenization; treat it as HARD-STOP-equivalent going forward despite not being originally listed there.

**`data/prepare_instruct_data.py` — built 2026-09-13.** Reads `data/cleaned/instruct_cleaned.txt` (already one example per line, `"### Instruction: {q} ### Response: {a}"`), splits each line back into (instruction, response) on the `" ### Response: "` marker, tokenizes into fixed-length (`context_length+1`) arrays per example, and masks `labels` (`-100`, matching `F.cross_entropy`'s `ignore_index`) everywhere the *target* token falls in the prompt or padding — only response tokens (+ trailing EOS) count toward loss. Truncation (rare, for oversized examples): drops from the front of the prompt first, preserving the full response; only truncates the response itself if the response alone doesn't fit. Output: `data/tokenized/model-a-instruct/{split}_input_ids.bin` (uint16) + `{split}_labels.bin` (int16, holds -100) + `meta.json` (example counts, context_length, pad_id) — NOT the same flat-stream `.bin` format `tokenizer/tokenize_corpus.py` produces, since each example here is independent and fixed-length rather than one continuous stream. `--model` flag (only `a-instruct` wired up today, kept flag-based for consistency with every other data-prep script and in case a future instruction-tuned variant reuses it). Verified via an isolated synthetic-example test (not run against the real ~2M-example file yet, per the no-auto-tokenize rule below): masking correctly isolates response-only tokens (decoded output matched the input response exactly), padding correctly masked, long-input truncation path doesn't crash and preserves the full response.

**`training/train_instruct.py` + `configs/model_117M_instruct.yaml` — built 2026-09-13.** Mirrors `training/train.py`'s cosine LR schedule, mixed precision, and checkpoint rotation exactly, but `get_batch()` samples random EXAMPLE indices (not random offsets into a flat stream) from `data/prepare_instruct_data.py`'s fixed-length arrays, and the loss line adds `ignore_index=-100` (`LABEL_IGNORE_INDEX`) so prompt/padding positions never count. Verified end-to-end with a tiny synthetic dataset + tiny model (not the real ~1.8M-example data, per the no-auto-tokenize/train rule): `load_examples()` → `get_batch()` → masked `cross_entropy` all produced correct shapes and a valid loss with no errors.

`configs/model_117M_instruct.yaml`: `max_steps=20000` (~1.6h at this GPU's benchmarked ~42,451 tok/s) — a deliberate choice, NOT a full epoch. The real cleaned instruct data turned out to be ~2.0M examples (~1.8M after the 90/5/5 split), not the ~45K-example subsample originally assumed when this was first scoped — 1 full epoch would be ~75,000 steps (~6h), closer to a scaled-down pretraining run than a typical instruction fine-tune, and risks overfitting to exact repeated Q&A pairs. User chose the smaller, Model-C-scale budget (~20,000 steps, ~480K examples, ~27% of one epoch) when presented with the three options and their time costs.

Training command (HARD STOP — same rule as `train.py`, not auto-executed):
```
python training/train_instruct.py --model a-instruct --init-from checkpoints/model-a/best.pt --config configs/model_117M_instruct.yaml
```

Evaluation (Phase 5) is written to handle this generically already: `evaluation/compare_models.py`, `evaluation/perplexity.py`, and `evaluation/generate_report.py` auto-detect however many of the three checkpoints exist (via `evaluation/common.py`'s `available_models()`). Note: these currently look for `checkpoints/model-{a,b,c}/` — will need a `model-a-instruct` entry added to `MODEL_LETTERS`/`MODEL_LABEL` once that model exists, since it's replacing `a` in the comparison, not just adding a 4th slot.

---

# --- PHASE 0 — Environment Setup ---

### Environment Verification
Run this exact script to verify — do not reinstall anything:
python -c "
import torch
print('Python OK')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
x = torch.tensor([1.0, 2.0]).cuda()
y = torch.tensor([3.0, 4.0]).cuda()
print('CUDA compute test:', x + y)
"

Expected output:
Python OK
PyTorch: 2.14.0
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
VRAM: 12.0 GB
CUDA compute test: tensor([4., 6.], device='cuda:0')

### Tasks
1. Run environment verification script above
2. Verify pip packages — install only if missing:
   pip show datasets huggingface_hub tokenizers
             sentencepiece transformers tensorboard
             fastapi uvicorn gradio tqdm numpy pandas
   Install any that are missing with:
   pip install <missing_package_name>
3. Create any missing folders in project structure
4. Create configs/model_117M.yaml:
   vocab_size: 32000
   context_length: 512
   n_layers: 12
   n_heads: 12
   d_model: 768
   d_ff: 3072
   dropout: 0.1
   batch_size: 24
   learning_rate: 3e-4
   warmup_steps: 2000
   max_steps: 133500
   checkpoint_every: 1000
   (batch_size/max_steps updated 2026-09-12 from the original 32/100000 —
   see "Training VRAM Budget" below and SESSION_NOTES_PHASE1-4_PREP.md
   for why: 32 silently overflowed real VRAM into slow shared memory,
   making it ~5.5x slower than needed for the same ~1.64B-token budget.)
5. Create evaluation/eval_questions.json:
   [
     "What is Python?",
     "Write a C# for loop",
     "What is Machine Learning?",
     "What is AGI?",
     "Explain gravity",
     "What is the capital of France?",
     "Tell me a short story",
     "What is a neural network?"
   ]

### Expected Output
- Environment verification script passes all checks
- configs/model_117M.yaml exists
- evaluation/eval_questions.json exists
- All folders in project structure exist
- All packages verified or installed

### Completion Check
All verifications pass, all files exist on disk.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 1"

---

## PHASE 1 — Data Download and Cleaning

### CORE DATA PIPELINE GUARDRAIL (Updated 2026-09-12 — supersedes the SQLite design below)
**CRITICAL: Before running any tasks in this phase, you MUST check the size of every dataset located in `data/raw/`.**
1. **Size Check:** For every subdirectory (`wikitext/`, `pg19/`, `openwebtext/`, `c4/`) under `data/raw/`, determine its total text content size.
2. **Conditional Flow:**
    * **If ALL datasets are $\le 10$ GB:** Follow the original, in-memory standard approach (Tasks 1 and 2 below).
    * **If ANY dataset is $> 10$ GB:** The process MUST switch to the **Streaming + In-Memory Hash-Set Dedup Approach** for that specific dataset (NOT SQLite — see note below).

**Why not SQLite:** the first implementation used SQLite as an external
disk-based key-value store to track hashes of unique paragraphs. At tens of
millions of small inserts, WAL checkpointing saturated disk I/O and stretched
cleaning runs to multiple days. It was replaced with: read the raw file in
bounded chunks (memory-safe for 40GB+ sources) and dedup against an
**in-memory `set()` of 8-byte BLAKE2b hashes** — a hash set of this size
(well under 1GB for ~8M entries) fits easily in RAM, gives O(1) membership
checks, and needs no disk round-trips per insert. This took the same
cleaning job from days down to roughly 15 minutes. See
`data/clean_text.py`'s `stream_clean_text()` for the implementation. Any new
large (>10GB) dataset added to this pipeline — including Model B's C#
source — MUST use this same in-memory hash-set approach, never SQLite.

### Tasks
1. Write data/download_datasets.py
   Downloads with progress logging every 100MB:
   - Salesforce/wikitext (wikitext-103-raw-v1)
     → data/raw/wikitext/
   - sedthh/gutenberg_english (pg19 replacement — bookcorpusopen/pg19
     script format is deprecated, this is a drop-in Gutenberg-books
     substitute)
     → data/raw/pg19/
   - Skylion007/openwebtext
     → data/raw/openwebtext/
   - allenai/c4 english (cc_news replacement — no gating, capped at
     2M examples)
     → data/raw/c4/
   Each dataset saved as text file.
   Print file size after each download.
   **Execute data/download_datasets.py (Task 1.1):** Run this script automatically.
   
2. Write data/clean_text.py
   This script must encapsulate the size check logic described above.
   - **If the relevant dataset is large (>10GB):** Use the streaming approach: read the raw file in fixed-size chunks (`stream_clean_text()`), and dedup using an **in-memory hash set of 8-byte BLAKE2b hashes** (not SQLite — see guardrail note above), writing the final output stream to `data/cleaned/<dataset_name>_cleaned_stream.txt`.
   - **If the relevant dataset is small ($\le 10$GB):** Use the original in-memory approach (`process_in_memory_text()`): Read the file entirely, apply cleaning rules (Remove HTML tags, URLs, lines < 20 chars, non-English, duplicate paragraphs via the same hash-set dedup), and save the cleaned output to `data/cleaned/<dataset_name>_cleaned.txt`.
   - **Combine & Deduplicate:** After individual cleaning, combine all cleaned streams/files into a final, clean corpus.
   - Both cleaning functions accept an `output_dir` parameter (default `data/cleaned/`) for reuse by other callers; every registered dataset, including Model B's `csharp` source, is cleaned into this same `data/cleaned/` directory — `data/prepare_splits.py`'s `MODEL_SOURCES` map (not a separate directory) is what keeps C# out of Model A's corpus.
   - **`instruct` source (added 2026-09-13):** `teknium/OpenHermes-2.5`, downloaded via `download_datasets.py`'s `download_openhermes()` (streams the 1.9GB single-JSON-array file with `ijson`, not `json.load()` — see that function's docstring). Reformatted to ONE example per line (`### Instruction: {q} ### Response: {a}`, internal newlines flattened to spaces) specifically so it fits this pipeline's existing per-line dedup/length-filter assumptions without any exception needed — a multi-line template was tried first and rejected: the repeated `"### Instruction:"` marker line would be byte-identical across ~1M examples, and dedup's exact-hash matching would silently keep only the first occurrence, destroying the format almost everywhere else. Goes through the normal in-memory cleaning path (well under 10GB, no `FORCE_STREAMING` needed).
   Print line counts before and after cleaning, and mention which deduplication method was used.
   **Execute data/clean_text.py (Task 2.1):** Run this script automatically.

3. Write data/prepare_splits.py
   - Combine all cleaned text from the definitive corpus file (the in-memory-hash-set stream output or the direct cleaned output).
   - Split 90/5/5 directly from the cleaned source files (see 2026-09-12
     note below — no intermediate combined corpus.txt is written).
   - Save data/model-a/train.txt
         data/model-a/val.txt
         data/model-a/test.txt
   - Print total token estimate
   Run prepare_splits.py automatically.

   **NOTE (2026-09-12) — corpus.txt removed:** the original design wrote a
   combined `corpus.txt` first, then split that into train/val/test —
   requiring the full combined size on disk TWICE at once (corpus.txt +
   train/val/test together). This actually ran the disk out of space
   building Model B (63GB combined corpus + up to 63GB of train/val/test
   simultaneously, on a 392GB drive already holding Model A's ~113GB and
   ~60GB of cleaned sources). `corpus.txt` was never read by anything
   downstream anyway (`tokenizer/tokenize_corpus.py` only reads
   `train.txt`/`val.txt`/`test.txt`), so `prepare_splits.py` was rewritten
   to stream train/val/test directly from the cleaned source files,
   `estimate_total_chars()` + `split_from_sources()` — halving peak disk
   usage. Model A's original `data/model-a/corpus.txt` (60GB) was left in
   place from before this change (not required, but not deleted either —
   delete it manually if you need to reclaim the space, since it isn't
   used by any later phase).

### Expected Output
- data/raw/ folders populated with downloaded files
- data/cleaned/ folders populated with cleaned files, and a report on the deduplication method used.
- data/model-a/train.txt exists
- data/model-a/val.txt exists
- data/model-a/test.txt exists
- Token count estimate printed

### Completion Check
All files exist, sizes reported, token count printed, and the size check/method used was logged.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 2"

---

## PHASE 2 — Tokenizer Training

### Tasks
1. Write tokenizer/train_tokenizer.py
   - BPE tokenizer using sentencepiece
   - Vocab size: 32000
   - Trained on data/model-a/train.txt
   - Saved to tokenizer/tokenizer.json
   - Saved vocab to tokenizer/vocab.txt
   Run train_tokenizer.py automatically.

2. Write tokenizer/verify_tokenizer.py
   Tests these exact sentences:
   - "The cat is sleeping on the mat"
   - "public class LoanService"
   - "Machine learning is a subset of AI"
   Prints tokens and IDs for each.
   Confirms encode then decode roundtrip matches original.
   Run verify_tokenizer.py automatically.

3. Write tokenizer/tokenize_corpus.py
   - Tokenizes data/model-a/train.txt
   - Tokenizes data/model-a/val.txt
   - Tokenizes data/model-a/test.txt
   - Saves binary chunks to data/tokenized/model-a/
   - Reports total token count
   **Hand off to user (2026-09-13 — see HARD STOP SCRIPT LIST): do not run
   this automatically. Print the exact command and wait for the user to
   run it and confirm.**

### Expected Output
- tokenizer/tokenizer.json exists
- tokenizer/vocab.txt exists
- data/tokenized/model-a/ binary files exist
- Roundtrip test passes for all 3 sentences
- Total token count printed

### Completion Check
All files exist, roundtrip verified, token count reported.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 3"

---

## PHASE 3 — Model Architecture
Use gemma4:7b for ALL tasks in this phase.

### Tasks
1. Write model/config.py
   Loads hyperparameters from configs/model_117M.yaml
   Exposes as Python dataclass or namespace.

2. Write model/attention.py
   Causal multi-head self attention:
   - QKV projection (single linear layer split)
   - Scaled dot product attention
   - Causal mask (lower triangular, no future leakage)
   - Output projection
   - Attention dropout
   Compatible with PyTorch 2.14.0 and sm_120.

3. Write model/transformer.py
   Single transformer decoder block:
   - Pre-norm architecture (LayerNorm before attention)
   - Multi-head causal attention
   - Feed forward network (Linear, GELU, Linear)
   - Residual connections on both sublayers
   - Residual dropout

4. Write model/model.py
   Complete GPT-style decoder-only model:
   - Token embedding (vocab_size × d_model)
   - Learned positional embedding (context_length × d_model)
   - N stacked transformer blocks
   - Final LayerNorm
   - Output linear projection (d_model → vocab_size)
   - Weight tying between input embedding and output projection

5. Write model/verify_model.py
   - Instantiate model from config
   - Print exact total parameter count
   - Run forward pass with dummy input (batch=2, seq=64)
   - Verify output shape is (2, 64, 32000)
   - Print GPU memory allocated after forward pass
   Run verify_model.py automatically.

### Expected Output
- model/config.py exists
- model/attention.py exists
- model/transformer.py exists
- model/model.py exists
- model/verify_model.py exists
- Parameter count confirmed between 110M and 125M
- Forward pass output shape (2, 64, 32000) confirmed
- No CUDA errors

### Completion Check
Parameter count verified, forward pass clean, no errors.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 4"

---

## PHASE 4 — Training
**NOTE (2026-09-10): The user is running this phase manually.** The
orchestrator should not write/run training/train.py or execute the
HARD STOP training runs — skip straight to Phase 6 once Phase 3 is
verified, and revisit this phase only if the user asks for it directly.
Use gemma4:7b for writing train.py and validate.py.

**NOTE (2026-09-12):** training/train.py, training/validate.py, and
training/benchmark_throughput.py (the last one not part of the original
spec — see below) are already written. Two real bugs were found and fixed
via the benchmark script before any real training was attempted: (1)
`learning_rate: 3e-4` in the YAML was read by PyYAML as a string, not a
float (fixed in model/config.py); (2) `np.random.randint`'s default int32
bounds can't address this ~12-billion-token training set (fixed to
`dtype=np.int64` everywhere batches are sampled). configs/model_117M.yaml's
`batch_size`/`max_steps` were also changed from 32/100000 to 24/133500 —
see "Training VRAM Budget" above for the real benchmarked numbers behind
that. Full write-up: SESSION_NOTES_PHASE1-4_PREP.md (repo root).

### Tasks
1. Write training/train.py
   - DataLoader over tokenized binary chunks
   - AdamW optimizer (weight decay 0.1)
   - Cosine LR schedule with linear warmup
   - Gradient clipping max norm 1.0
   - Mixed precision fp16 via torch.cuda.amp
   - TensorBoard logging every 100 steps:
     train_loss, learning_rate, tokens_per_second, grad_norm (added
     2026-09-14 — clip_grad_norm_'s return value, the pre-clip total L2
     norm across all parameters, was already being computed for gradient
     clipping every step; just wasn't being logged before. A standard
     training-health signal — instability/spikes show up here, often
     before they show up in loss. Same change made in
     training/train_instruct.py.)
   - Checkpoint save every 1000 steps to checkpoints/model-a/
     or checkpoints/model-b/ based on --model arg
   - Validation loss logged every 500 steps
   - Resume from latest checkpoint if exists
   - CLI: python train.py --model [a|b|c]
                          --config configs/model_117M.yaml
                          --init-from <checkpoint>  (optional, Model C only)
                          --keep-last-checkpoints <N>  (optional, default 3)

   **NOTE (2026-09-13) — checkpoint rotation added:** with no cleanup,
   saving one ~1.32GB checkpoint (110M fp32 weights + AdamW's 2x optimizer
   state) every 1000 steps for the full 133,500-step run never deletes any
   of them — ~178GB for a single model, ~410GB across A+B+C combined,
   exceeding the project's entire 392GB drive (which was already down to
   79GB free after the C# data pipeline work). `rotate_checkpoints()` now
   runs after every periodic save and after the final save, deleting all
   but the most recent `--keep-last-checkpoints` (default 3) periodic
   `ckpt_*.pt` files. `best.pt` (separate filename, always kept) and the
   final `ckpt_{max_steps}.pt` are never touched by rotation. Steady-state
   disk use per model becomes `(keep_last_checkpoints + 1) * 1.32GB` —
   ~5.3GB at the default of 3, ~15.8GB across all three models kept
   simultaneously (needed for the Phase 5 three-way comparison), instead
   of the unbounded ~410GB total.

2. Write training/validate.py
   - Load model from checkpoint
   - Calculate validation loss on val split
   - Calculate perplexity from validation loss
   - Generate one sample text continuation
   - Print all metrics

3. HARD STOP — Train Model A:
```
Script  : training/train.py
Command : python training/train.py --model a --config configs/model_117M.yaml
Monitor : tensorboard --logdir logs/model-a
          (optional, separate terminal) python training/capture_progress.py --model a
Reason  : 24-48 hour runtime — must run outside Claude Code session
Action  : Close Claude Code. Close Ollama.
          Run command in standalone terminal.
          Monitor via TensorBoard in browser.
          Optionally also run capture_progress.py in another terminal to
          save periodic PNG snapshots of the same graphs (see note below)
          for later review without needing the TensorBoard server running.
          When training complete type:
          DONE — train.py model a complete
```
   After DONE: verify checkpoints/model-a/ has .pt files.
   Then continue automatically to Task 4.

4. Model B data prep (rewritten 2026-09-12 — folded into the Model A
   scripts instead of a standalone prepare_model_b.py, which no longer
   exists):
   - **Download:** `data/download_datasets.py` has a 5th registry entry,
     `csharp` → `bigcode/the-stack-dedup` (`"c-sharp"` config, capped at 2M
     examples). Two dead ends before landing here, both confirmed by
     actually inspecting the data, not guessed:
     1. `bigcode/the-stack-v2`, config `"C-Sharp"` — tried first, failed
        with `BuilderConfig 'C#' not found` (the dataset names configs
        after GitHub Linguist language names, not `"C#"` itself).
     2. `bigcode/the-stack-v2`, config `"C-Sharp"` (fixed name) — loaded
        fine, but a direct parquet-schema inspection (via `HfFileSystem`
        range-reads, no full download needed) showed this config has
        **no `content`/`text` field at all** — only metadata
        (`blob_id`, `repo_name`, `path`, etc.). The actual file bytes for
        `the-stack-v2` live in Software Heritage's separate
        content-addressed storage and need a different access process
        entirely. Every "successful" download from it would have silently
        written zero real examples.
     `bigcode/the-stack-dedup`'s `c-sharp` config was schema-verified to
     have a real `content` column (107 shards, ~101k rows each, ~10.8M
     rows total — well over the 2M cap) before switching to it. It's
     gated separately from `the-stack-v2` — accept its terms on the Hub
     before running the download.
     `download_csharp()` downloads shards one at a time via
     `huggingface_hub.hf_hub_download` (resumable — picks up from the
     last received byte on a dropped connection) rather than
     `datasets.load_dataset(streaming=True)`, because this dataset's CDN
     host (`us.aws.cdn.hf.co`) was observed to intermittently time out or
     drop mid-transfer, and the streaming row-iterator restarts the whole
     HTTP GET from scratch on every such error instead of resuming. Each
     shard's local parquet copy is read with `pyarrow` and deleted after
     its `content` column is extracted, so disk usage doesn't grow with
     shards already processed; downloading stops as soon as 2M examples
     are collected (typically ~20 of the 107 shards, not all of them) →
     `data/raw/csharp/csharp_train.txt`. The script's existing
     skip-if-exists loop means re-running it after Model A's raw files
     already exist downloads only the C# source.
     **Known slow-network caveat:** each shard is ~168MB; on a
     connection-constrained network this can be very slow (one measured
     sandbox environment saw ~120KB/s to this specific CDN host — ~23
     min/shard, ~7-8h total — while a normal terminal on the same machine
     had already downloaded the 38GB `openwebtext` source without issue).
     If a download of this dataset seems stalled, check real transfer
     speed (e.g. watching `data/raw/csharp/_shard_cache/`'s size grow)
     before assuming the code is broken — it may just be the network path
     in use.
     **Resolved (2026-09-12):** `codeparrot/github-code` was also tried as
     an alternative and hit the exact same `us.aws.cdn.hf.co` SSL failure,
     confirming the repeated failures were a host-level network issue (the
     user's own internet connection), not specific to any dataset or to
     this code. Once that connectivity issue was fixed on the user's end,
     `download_csharp()` against `bigcode/the-stack-dedup` completed
     successfully unmodified: 8,407.7 MB, capped at 2,000,000 examples.
     `--dataset csharp` (see below) was used throughout troubleshooting to
     avoid re-touching the other 4 already-downloaded sources.
   - **`--dataset` flag:** `data/download_datasets.py` takes an optional
     `--dataset {wikitext,pg19,openwebtext,c4,csharp}` argument — omit it
     to download every registered dataset (skip-if-exists still applies
     per source), or pass one name to download only that source, e.g.
     `python data/download_datasets.py --dataset csharp`.
   - **Clean:** `data/clean_text.py`'s `DATASETS` dict also registers
     `csharp`, so it goes through the exact same size-checked cleaning as
     every English source — `process_in_memory_text()` if ≤10GB,
     `stream_clean_text()` (in-memory hash-set dedup, NOT SQLite) if >10GB —
     writing to `data/cleaned/csharp_cleaned[_stream].txt`. The script also
     skips re-cleaning any dataset whose cleaned output already exists
     (`already_cleaned()`), so re-running it only cleans csharp. On top of
     that automatic skip, it also takes the same kind of explicit
     `--dataset {wikitext,pg19,openwebtext,c4,csharp}` flag as
     `download_datasets.py` (added 2026-09-12) — e.g.
     `python data/clean_text.py --dataset csharp` restricts the run to
     just that one source, so the other 4 are never even size-checked,
     not just skipped after the fact.
     **`FORCE_STREAMING` set (added 2026-09-12):** the 10GB threshold was
     calibrated for prose (few, long lines per byte). `csharp_train.txt` is
     8.8GB but has 227M lines (~38 bytes/line — source code has far more,
     much shorter lines per byte than prose), so `process_in_memory_text()`
     would still qualify it for the in-memory path by size alone while
     actually risking a MemoryError: splitting the full file into one
     Python string object per line, at that line count, costs roughly
     11GB in per-object overhead alone (~50 bytes/string), on top of the
     original 8.8GB file string still held at the same time. `csharp` (and
     any future source-code dataset) is listed in `FORCE_STREAMING` to
     always take the streaming path regardless of its byte size.
   - **Combine & split:** `data/prepare_splits.py` takes a `--model {a,b,c}`
     flag backed by a `MODEL_SOURCES` map (`a` → the 4 English sources,
     `b` → the same 4 + `csharp`, `c` → `csharp` only — see "Three-Model
     Comparison Design" above). `python data/prepare_splits.py --model b`
     splits the already-cleaned English files plus the cleaned C# file
     directly into `data/model-b/{train,val,test}.txt`, 90/5/5 — no
     combined `corpus.txt` is written (see the 2026-09-12 note under Phase
     1 Task 3: it doubled peak disk usage and was never read downstream,
     which is what ran Model B's build out of disk space the first time);
     `--model c` does the same but
     C#-only into `data/model-c/`. Model A is unaffected by either —
     `--model a` (the default) still only ever combines the 4 English
     sources, so C# never leaks into Model A's corpus. Always pass
     `--model` explicitly for B/C — the bare command defaults to `a` and
     would just redundantly rebuild Model A's corpus from files that
     haven't changed.
   - Print token count estimate (done by prepare_splits.py already).

5. Tokenize Model B corpus:
   Run tokenizer/tokenize_corpus.py on model-b splits.
   Save binary output to data/tokenized/model-b/
   Run automatically.

6. HARD STOP — Train Model B:
```
Script  : training/train.py
Command : python training/train.py --model b --config configs/model_117M.yaml
Monitor : tensorboard --logdir logs/model-b
          (optional, separate terminal) python training/capture_progress.py --model b
Reason  : 48-72 hour runtime — must run outside Claude Code session
Action  : Close Claude Code. Close Ollama.
          Run command in standalone terminal.
          Optionally also run capture_progress.py in another terminal (see
          Model A's block above for what it does).
          When training complete type:
          DONE — train.py model b complete
```
   After DONE: verify checkpoints/model-b/ has .pt files.
   Then continue automatically to Task 7.

7. Model C data prep + fine-tune (added 2026-09-12 — see "Three-Model
   Comparison Design" above for why this exists):
   - **Data:** `python data/prepare_splits.py --model c` — `MODEL_SOURCES["c"]`
     is `["csharp"]` only (no English), so this reuses the already-cleaned
     `data/cleaned/csharp_cleaned[_stream].txt` from Task 4 and writes
     `data/model-c/{train,val,test}.txt` (no `corpus.txt` — see 2026-09-12
     note under Phase 1 Task 3). Run automatically.
   - **Tokenize:** `python tokenizer/tokenize_corpus.py --model c` — same
     shared tokenizer/vocab as A and B — → `data/tokenized/model-c/`.
     Run automatically.
   - **HARD STOP — Fine-tune Model C:**
```
Script  : training/train.py
Command : python training/train.py --model c --init-from checkpoints/model-a/best.pt --config configs/model_117M_finetune_c.yaml
Monitor : tensorboard --logdir logs/model-c
          (optional, separate terminal) python training/capture_progress.py --model c
Reason  : GPU-intensive training run — must run outside Claude Code session,
          same as Model A/B (see HARD STOP SCRIPT LIST — the --init-from
          flag does not exempt this from the rule)
Action  : Close Claude Code. Close Ollama.
          Run command in standalone terminal.
          Monitor via TensorBoard in browser.
          Optionally also run capture_progress.py in another terminal (see
          Model A's block above for what it does).
          When training complete type:
          DONE — train.py model c complete
```
     `--init-from` loads Model A's weights only (optimizer state and step
     counter are NOT restored — see train.py's checkpoint-loading logic);
     `configs/model_117M_finetune_c.yaml` uses fine-tuning-scale
     hyperparameters (lower LR, shorter warmup, far fewer steps) rather than
     `model_117M.yaml`'s pretraining schedule.
   After DONE: verify checkpoints/model-c/ has .pt files.
   Then continue automatically to Phase Completion Report.

### Expected Output
- training/train.py exists
- training/validate.py exists
- checkpoints/model-a/ contains .pt checkpoint files
- checkpoints/model-b/ contains .pt checkpoint files
- checkpoints/model-c/ contains .pt checkpoint files
- data/model-b/train.txt, val.txt, test.txt exist
- data/model-c/train.txt, val.txt, test.txt exist
- data/tokenized/model-b/ binary files exist
- data/tokenized/model-c/ binary files exist

### Completion Check
All three model checkpoints verified on disk after DONE confirmations.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 5"

---

## PHASE 5 — Evaluation
**NOTE (2026-09-10): The user is running this phase manually** (it also
depends on Model A/B checkpoints from Phase 4, which the user is
producing outside this orchestrated flow). Skip straight to Phase 6.

**NOTE (2026-09-12):** All three scripts below are already written —
`evaluation/compare_models.py`, `evaluation/perplexity.py`,
`evaluation/generate_report.py`, plus a shared helper module
`evaluation/common.py` (model loading, tokenizer loading, generation,
sequential-window perplexity — imported by all three, not run directly).
They were extended for the three-model design (see "Three-Model Comparison
Design" near the top of this file) and auto-detect however many of
Model A/B/C's checkpoints exist via `available_models()` — running them with
only 1 or 2 models trained degrades gracefully rather than erroring; a dry
run with zero checkpoints present prints a message and exits cleanly (no
stub JSON/MD files written until there's real data to fill them with).

### Tasks
1. evaluation/compare_models.py
   - Loads eval_questions.json
   - Loads the best checkpoint for every model in {a, b, c} that has one
     (`evaluation/common.find_best_checkpoint`, prefers `best.pt`, falls
     back to the highest-numbered `ckpt_*.pt`)
   - Runs every question through every available model — records response
   - Saves side by side results to evaluation/model_comparison.json
   Format: `{ "question": "", "model_a": "", "model_b": "", "model_c": "" }`
   (keys present only for models that were actually loaded)
   Run compare_models.py automatically.

2. evaluation/perplexity.py
   - Loads every available model's best checkpoint
   - Loads every available model's tokenized test set
     (`data/tokenized/model-{a,b,c}/test.bin`)
   - Computes the full **cross-perplexity matrix**: every available model
     scored against every available test set (not just each model on its
     own test set) — e.g. Model A (English-only) scored on Model C's
     C#-only test set shows how out-of-distribution C# is for it, while
     Model C on that same test set shows what fine-tuning bought it.
   - Saves to evaluation/perplexity_results.json:
     `{"models": [...], "test_sets": [...], "own_test_set_perplexity": {...}, "matrix": {model: {test_set: {loss, perplexity}}}}`

   HARD STOP — Perplexity Evaluation:
```
Script  : evaluation/perplexity.py
Command : python evaluation/perplexity.py
Reason  : GPU intensive, 30-60 min runtime (scales with number of models x test sets trained)
Action  : Run in separate terminal.
          When complete type:
          DONE — perplexity.py complete
```
   After DONE: verify perplexity_results.json exists
   and contains numeric scores for every trained model.
   Then continue automatically to Task 3.

3. evaluation/generate_report.py
   - Reads perplexity_results.json
   - Reads model_comparison.json
   - Generates evaluation/final_report.md containing:
     * Perplexity cross-matrix table (rows = model, columns = test set, own-set cells marked)
     * Side by side response table for each question, one column per available model
     * Key observations section (auto-derived deltas between every pair of available models)
     * Conclusion section explaining what each available pairing isolates:
       A vs B (data composition), A vs C (fine-tuning vs. its own base
       checkpoint), B vs C (joint vs. sequential learning of C#), and the
       full A vs B vs C synthesis once all three exist
   Run generate_report.py automatically.
   Print contents of final_report.md to console.

### Expected Output
- evaluation/model_comparison.json exists
- evaluation/perplexity_results.json exists
- evaluation/final_report.md exists
- final_report.md content printed to console

### Completion Check
All evaluation files exist, report content printed.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 6"

---

## PHASE 6 — Inference Pipeline

### Tasks
1. Write inference/generate.py
   - Load model from checkpoint path argument
   - Load tokenizer from tokenizer/tokenizer.json
   - Implement greedy decoding
   - Implement top-k sampling
   - Temperature scaling
   - Max new tokens limit
   - CLI: python generate.py --checkpoint <path>
                              --prompt "The capital of France is"
                              --max_tokens 200
                              --temperature 0.8
                              --top_k 50
   Run one test generation automatically using
   best Model A checkpoint.
   Print generated output to verify model responds.

2. Write api/main.py FastAPI application:
   - Startup: load model and tokenizer once into memory
   - POST /generate
     Request  : { "prompt": str, "max_tokens": int,
                  "temperature": float, "top_k": int }
     Response : { "response": str, "tokens_generated": int,
                  "model": str }
   - GET /health
     Response : { "status": "ok", "model_loaded": bool }
   - Error handling for empty prompt or model not loaded

3. HARD STOP — API Server:
```
Script  : uvicorn api.main:app
Command : uvicorn api.main:app --host 0.0.0.0 --port 8000
Reason  : Blocking server process
Action  : Run in a separate terminal.
          Then in another terminal run this test:
          curl -X POST http://localhost:8000/generate
            -H "Content-Type: application/json"
            -d "{\"prompt\":\"The capital of France is\",
                 \"max_tokens\":50,\"temperature\":0.8,
                 \"top_k\":50}"
          Copy the full curl response output here.
          Then type:
          DONE — uvicorn server running
```
   After DONE: continue automatically to Phase 7.
   Do NOT wait for any additional confirmation.

### Expected Output
- inference/generate.py exists
- generate.py test output printed (model response visible)
- api/main.py exists
- DONE confirmation received with curl response pasted

### Completion Check
generate.py output printed, api/main.py written,
DONE received, curl response confirmed.
Print Phase Completion Report then wait for
"CONFIRMED — proceed to Phase 7"

---

## PHASE 7 — Deployment

### Tasks
1. Write deployment/quantize.py
   - Load best Model A checkpoint
   - Apply dynamic int8 quantization via torch.quantization
   - Save quantized model to deployment/model_int8/
   - Print original size vs quantized size comparison
   Run quantize.py automatically.

2. Write deployment/app.py Gradio UI:
   - Load quantized model from deployment/model_int8/
   - Load tokenizer from tokenizer/tokenizer.json
   - UI components:
     * Text input box for prompt
     * Slider: max tokens 50-500 (default 200)
     * Slider: temperature 0.1-1.0 (default 0.8)
     * Generate button
     * Output text box
     * Info panel showing:
       - Model: Track 1 SLM — 117M parameters
       - Training data: English corpus (Model A)
       - GPU: RTX 5070 Ti Laptop

3. Write deployment/requirements.txt
   Exact package versions needed for HuggingFace Spaces:
   torch, gradio, sentencepiece, tokenizers, numpy

4. Print exact HuggingFace deployment commands:
   - huggingface-cli login
   - huggingface-cli repo create track1-slm --type space --sdk gradio
   - Commands to upload model weights
   - Commands to upload app.py and requirements.txt
   - Expected public URL:
     https://huggingface.co/spaces/<username>/track1-slm

5. HARD STOP — HuggingFace Upload:
```
Script  : huggingface-cli + git commands
Command : Print exact step by step commands above
Reason  : Requires HuggingFace login credentials
Action  : Run printed commands in terminal.
          Verify Space builds successfully on HF website.
          Paste the live public URL here.
          Then type:
          DONE — HuggingFace deployment complete
```
   After DONE: print public URL and mark Track 1 complete.

### Expected Output
- deployment/model_int8/ exists with quantized model
- deployment/app.py exists
- deployment/requirements.txt exists
- Live public HuggingFace Spaces URL confirmed

### Completion Check
All deployment files exist, DONE received, URL printed.

```
========================================
TRACK 1 COMPLETE
Print full summary:
- Total phases completed
- Model A perplexity score
- Model B perplexity score
- Key difference: Model A vs Model B responses
- Public deployment URL
- Estimated time taken per phase
========================================
```