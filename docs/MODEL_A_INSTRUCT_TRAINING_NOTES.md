# Model A-Instruct — Training Notes

Started: 2026-09-14. Completed: 2026-09-14. Instruction fine-tune of Model
A's checkpoint (see CLAUDE.md's "Three-Model Comparison Design", REVISED
2026-09-13).

Note while this ran: GPU showed 75% utilization instead of the usual 100%,
with Task Manager confirming actual spillover into shared GPU memory
(1.5GB of 12.4GB total demand, dedicated VRAM maxed at 10.9/12.0GB) —
caused by Ollama sitting resident on the GPU alongside training, not by
this run's own settings (`batch_size=24` is the same already-benchmarked
value used for Model A/B). Training still completed correctly, just
somewhat slower than the ~1.6-1.7h estimate as a result.

## What's different about this run vs. Model A/B

Model A/B are **from-scratch pretraining** — random initialization, plain
next-token-prediction loss over a giant flat token stream. Model A-Instruct
is a **fine-tune**: it starts from Model A's already-trained weights, uses
a fixed per-example (not flat-stream) data format, and masks the loss so
only response tokens count — see `training/train_instruct.py` (a separate
script from `training/train.py`, not a modification of it, since the
batching/masking semantics differ enough to make sharing one script more
confusing than two focused ones).

## Architecture: same 110,025,216-parameter GPT-style decoder

Identical to Model A/B — `n_layers=12, n_heads=12, d_model=768,
d_ff=3072, context_length=512, vocab_size=32000`. Same shared tokenizer
too (trained once on Model A's corpus, never retrained). See
`MODEL_A_TRAINING_NOTES.md` / `MODEL_B_TRAINING_NOTES.md` for the full
parameter-count derivation and tokenizer explanation — unchanged here.

## What actually happens differently: weight init

| Step | What happens |
|---|---|
| Model construction | Fresh `GPTModel` built from `configs/model_117M_instruct.yaml` (same architecture numbers as `model_117M.yaml`) |
| Weight loading | `--init-from checkpoints/model-a/best.pt` — loads Model A's `model_state_dict` only. Optimizer state is NOT restored (fresh AdamW), step counter starts at 0 — this is a new training run over different data, not a resume of Model A's own run |

## Data: instruction fine-tune corpus (OpenHermes-2.5, template-formatted)

| Split | Examples | Role |
|---|---|---|
| train | 897,110 | What the model learns from |
| val | 49,839 | Checked every 500 steps to select `best.pt` |
| test | 49,840 | Untouched during training — reserved for final evaluation |

**Note on this count (worth recording — a real investigation happened
here)**: an earlier estimate assumed ~2.0M examples, based on a raw line
count of `data/cleaned/instruct_cleaned.txt`. That count was wrong:
`clean_text.py`'s in-memory cleaning writes its output as
`"\n\n".join(cleaned_paragraphs)` — double newlines between examples — so
a naive line-count sees each blank separator line as its own "line,"
roughly doubling the apparent total. The REAL example count (897,110 +
49,839 + 49,840 = 996,789) is correct and complete; no data was lost —
`data/prepare_instruct_data.py`'s `load_examples()` correctly skipped the
blank artifact lines, just with a misleading "missing marker" label in
its print statement (cosmetic only).

**Format**: each example is tokenized into a fixed `(context_length=512,)`
`input_ids` array plus a `labels` array of the same shape, where every
position whose *target* token falls in the prompt (`### Instruction:
{q}\n\n### Response: `) or in padding is set to `-100` — `F.cross_entropy`'s
`ignore_index` skips those positions, so only response tokens (+ trailing
EOS) count toward the loss. Verified correct beforehand via an isolated
synthetic-example test: decoded non-masked-target tokens matched the
input response text exactly.

Truncation (rare — for oversized examples): drops from the front of the
prompt first, keeping the full response intact; only truncates the
response itself if it doesn't fit even with the whole prompt removed.

Stored as: `data/tokenized/model-a-instruct/{split}_input_ids.bin`
(`uint16`) + `{split}_labels.bin` (`int16`, holds `-100`) + `meta.json`
(example counts, `context_length`, `pad_id`) — NOT the same flat-stream
format `tokenizer/tokenize_corpus.py` produces for A/B/C.

## Training hyperparameters (`configs/model_117M_instruct.yaml`)

| Param | Value | Meaning | Why this value |
|---|---|---|---|
| `batch_size` | 24 | Examples per optimizer step | Same as A/B — benchmarked safe at this VRAM |
| `learning_rate` (peak) | 3e-5 | 10x lower than A/B's 3e-4 | Standard fine-tuning practice — large peak LR risks damaging already-trained weights |
| `warmup_steps` | 200 | Linear ramp 0 → peak | Proportionally short, matching the much shorter total run |
| `max_steps` | 20,000 | Total optimizer steps | Deliberately short relative to pretraining — instruction fine-tunes conventionally use a small budget; also ~53.5% of one epoch over the real 897,110-example train set (not a full multi-epoch pass, to avoid overfitting on repeated exact Q&A pairs) |
| `checkpoint_every` | 500 | Periodic checkpoint save | Finer-grained than A/B given the shorter run (~40 periodic saves total before rotation) |

**Compute budget**: `20,000 x 24 x 512 = 245,760,000` token-positions
processed — but only the response-token subset of those actually
contributes to the loss/gradient (see masking above); the rest (prompt +
padding) are forward-pass compute that gets excluded at the loss step.
Estimated time: ~1.6-1.7 hours at this GPU's ~40,000-42,000 tok/s.

## Prerequisites confirmed before this run started

- Model A's checkpoint (`checkpoints/model-a/best.pt`) — the weights this run initializes from
- `data/prepare_instruct_data.py` — built, and verified via an isolated
  synthetic-example test before running on the real data (masking,
  truncation, and shapes all confirmed correct)
- `training/train_instruct.py` — built, verified end-to-end with a tiny
  synthetic dataset + tiny model (masked loss computed correctly, no
  shape errors) before running on the real data
- Checkpoint rotation code reused unchanged from `training/train.py`
  (`--keep-last-checkpoints 3` default)
- Disk headroom — 58GB free at start of this run

## Results

| Metric | Value |
|---|---|
| Best val_loss (masked, response-tokens-only) | **1.6486** at step 14,500 (saved as `checkpoints/model-a-instruct/best.pt`) |
| "Perplexity" from that loss | exp(1.6486) ≈ **5.20** — NOT directly comparable to Model A/B's perplexity (34.98 / TBD): this loss is masked (response tokens only, not every token) and measured on a different, narrower domain (instruction Q&A, not general prose), so a much lower number is expected on its own terms, not a sign of a "better" model in an absolute sense |
| Final val_loss (step 19,500) | ~1.82 — noticeably higher than the step-14,500 best |
| Total training steps | 20,000 (final checkpoint `ckpt_20000.pt`) |

**val_loss behavior — NOT a clean monotonic decrease, worth recording honestly**: dropped steadily from 2.32 (step 500) down to the 1.65-1.79 range by steps 10,000-18,000, hit its actual minimum (1.6486) at step 14,500, then **increased and got noisier** for the remaining ~5,500 steps (bouncing between 1.65 and 1.91, ending around 1.82 at step 19,500) — consistent with the model starting to overfit the training set once past its optimum, exactly why `best.pt` (not the final checkpoint) is the one to use downstream. Confirmed directly from the TensorBoard event log (39 val_loss points total).

### Sample generations — real before/after comparison against Model A/B

Same prompts tested earlier against Model A and Model B, now against Model A-Instruct (`--temperature 0.7 --top_k 50`):

| Prompt | Model A-Instruct's response | vs. Model A/B |
|---|---|---|
| "What is capital of India?" | *"The capital is India. It is the capital of India..."* (repetitive, factually wrong) | **Real improvement**: Model A/B gave an empty/1-token response to this exact prompt. A-Instruct now actually attempts an answer instead of stopping immediately — the core behavior this fine-tune targeted |
| "What is C#?" | *"...C# is a C# project with a few similarities and differences. The `C#` package is a library for C# that you can use to create and run C# applications..."* | On-topic and engages properly with the question (vs. Model B degrading into an unrelated "Chess Board in the Garden of Eden" repetition loop for the same prompt) |
| "Is C# a programming language?" | *"...It is a programming language that is used for various purposes... including C#, Python..."* | Coherent, on-topic discussion of programming languages before degrading into a `"C#: C#: C#:"` repetition loop |
| "give me 2 differences between C# and Python" | *"...in C#, `params.parameters` is not a string, and in Python, `parameters` is a string..."* | Attempts a real, topically-relevant technical distinction (typing differences) before looping on `"parameters is a string"` |

**Honest takeaway**: the fine-tune clearly worked for its intended purpose — the model now engages with direct questions instead of cutting off, and stays more on-topic for C#-related prompts than Model B did. It has **not** fixed the repetition-loop degeneration (same root cause as before: no repetition penalty implemented in `inference/generate.py`/`api/main.py`, and this is still a 110M-parameter model) — every sample above eventually collapses into a repeated phrase given enough tokens. That fix remains a separate, not-yet-implemented change (offered previously, on hold per explicit instruction not to implement anything without approval).

### Checkpoint files (after rotation)

`checkpoints/model-a-instruct/`: `best.pt` (step 14,500), `ckpt_19000.pt`, `ckpt_19500.pt`, `ckpt_20000.pt` (final) — matches the `--keep-last-checkpoints 3` default (3 most recent periodic + best, final never pruned).
