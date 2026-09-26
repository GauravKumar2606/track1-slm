# Model B — Training Notes

Started: 2026-09-14. Trained from scratch on English + C# + instructions
combined (see CLAUDE.md's "Three-Model Comparison Design", REVISED
2026-09-13). Status at time of writing: **in progress** — this file
covers the architecture/data/hyperparameter setup; a Results section will
be added once training completes (see `MODEL_A_TRAINING_NOTES.md` for the
format that section will follow).

## Architecture: 110,025,216-parameter GPT-style decoder

Same architecture as Model A — this is what makes cross-model comparison
meaningful at all.

| Term | Meaning | Role in training | Where the number came from |
|---|---|---|---|
| GPT-style decoder | Causal/masked-attention transformer — predicts the next token using only tokens seen so far, never future ones | The entire trainable model; training = adjusting its weights so next-token predictions get closer to the real text | Standard architecture family for autoregressive text generation (GPT-2/3/4, LLaMA, etc.) |
| 110,025,216 parameters | Total learnable weights across the whole model | The model's capacity — sets GPU memory use, training speed, and the ceiling on how much it can encode | Falls out of the 6 architecture numbers below (verified two ways: `sum(p.numel() for p in model.parameters())`, and by hand from the config — see `MODEL_A_TRAINING_NOTES.md`) |
| n_layers=12 | Number of stacked transformer blocks | Depth — more layers = more capacity for complex, multi-step patterns, at more compute cost per token | Copied from GPT-2-small's published recipe |
| n_heads=12 | Parallel attention sub-mechanisms per layer | Lets the model track several kinds of token relationships at once. `head_dim = d_model/n_heads = 768/12 = 64` | Same source |
| d_model=768 | Width of each token's vector representation throughout the network | The model's internal information bandwidth | Same source |
| d_ff=3072 | Hidden width of each block's feed-forward sublayer (`4 x d_model`) | Extra per-token processing capacity, separate from attention | Standard 4x expansion convention since the original Transformer paper |
| context_length=512 | Max tokens the model can attend over in one pass | Limits how much preceding text informs a prediction; attention compute scales ~quadratically with this | Chosen smaller than GPT-2's 1024, likely for this project's single-GPU compute budget |

**Origin of 12/12/768/3072**: GPT-2-small's exact published configuration
("the 117M model"), deliberately reused rather than inventing new
hyperparameters — written into `CLAUDE.md`'s Phase 0 spec before any code
existed.

## Tokenizer: SentencePiece BPE, vocab_size=32000

- **SentencePiece**: treats raw text as a stream of Unicode characters
  with no assumption of whitespace word boundaries — matters here since
  the corpus includes C# code and templated instruction text, not just
  English prose.
- **BPE**: iteratively merges the most frequent adjacent character/token
  pairs into single tokens, producing a vocabulary of common subwords.
- **Role**: translates text into integer IDs (0-31,999) the model can
  process; `vocab_size` sizes `token_emb` and `head` (32000x768 each,
  tied — same weights).
- **Origin of 32,000**: a predetermined config choice (Phase 0), matching
  a common convention (also used by RoBERTa, LLaMA's tokenizer). The
  actual tokenizer (which merge rules produce those 32,000 tokens) was
  trained once, in Phase 2, on **Model A's** English `train.txt`, and
  reused unchanged for every model since (B, C, A-Instruct) — essential,
  since a per-model tokenizer would make token IDs mean different things
  in each model, breaking comparability.

## Data (Model B = English + C# + instructions, from scratch)

| Split | Size | Tokens | Role |
|---|---|---|---|
| train | 27.66 GB | 13,828,318,092 | What the model learns from — every gradient update samples a batch from here |
| val | 1.61 GB | 803,766,171 | Never trained on; checked every 500 steps (forward pass only) to track generalization and select `best.pt` |
| test | 1.09 GB | 544,677,421 | Untouched during training — reserved for one final evaluation afterward |
| **Total** | **30.35 GB** | **~15.18B** | |

Sources: `wikitext`, `pg19`, `openwebtext`, `c4` (same 4 as Model A) +
`csharp` (`bigcode/the-stack-dedup`, capped at 2M examples) + `instruct`
(`teknium/OpenHermes-2.5`, ~2M examples, reformatted to one line per
example). Combined + split 90/5/5 via `data/prepare_splits.py --model b`
(no intermediate `corpus.txt` — dropped mid-project after an earlier
version of this design caused a real disk-full failure). Tokenized via
`tokenizer/tokenize_corpus.py --model b`; each token stored as a `uint16`
(2 bytes — covers vocab_size 32000, half the size of a 4-byte `int32`),
so `token count = file size in bytes / 2`.

## Training hyperparameters (`configs/model_117M.yaml` — unchanged from Model A)

| Param | Value | Meaning | Origin |
|---|---|---|---|
| `batch_size` | 24 | Sequences processed per optimizer step | Empirically benchmarked (`training/benchmark_throughput.py`) as the largest batch fitting in 12GB VRAM without spilling into slow shared memory — the original spec's 32 ran ~4x slower due to that spillover |
| `learning_rate` (peak) | 3e-4 | AdamW's step size at the schedule's peak | Standard/conventional peak LR for this model size |
| `warmup_steps` | 2,000 | Steps ramping LR linearly from 0 to peak | Prevents unstable early updates near random initialization |
| `max_steps` | 133,500 | Total optimizer steps | Targets a ~1.64B-token compute budget (`133,500 x 24 x 512`) — a deliberate budget, independent of how much larger Model B's actual data pool (15.18B tokens) is |
| `checkpoint_every` | 1,000 steps | Periodic checkpoint save frequency | Balances resume-safety against disk use; paired with `rotate_checkpoints()` (`--keep-last-checkpoints 3` default) to keep steady-state disk use bounded — without it, a full run would write ~178GB of checkpoints with no cleanup (this is what actually exhausted disk space once already during data prep) |
| GPU | RTX 5070 Ti Laptop, 12GB VRAM | | Benchmarked ~40,000-42,000 tok/s at `batch_size=24` |

## Prerequisites confirmed before this run started

- Shared architecture code (`model/*.py`) — Phase 3, verified 110,025,216 params
- Shared tokenizer — trained once on Model A's corpus, unchanged since
- Disk headroom — ~90GB freed (stale HF cache + Model A's redundant `corpus.txt`) before this run, given the checkpoint-disk-exhaustion incident earlier in the project
- Checkpoint rotation code in place (`training/train.py`'s `rotate_checkpoints()`)
- Claude Code / Ollama closed to free full 12GB VRAM for training, per CLAUDE.md's HARD STOP rule

## Results

**Best val_loss: 3.5300 at step 128,000** (`checkpoints/model-b/best.pt`).

**Actual training duration: 11.22 hours** (step 0 → 133,400), recovered
directly from TensorBoard's own logged `wall_time` timestamps on each scalar
event — not estimated, not derived from checkpoint file mtimes (this run's
earlier periodic checkpoints were rotated out, and no live screenshot capturer
was running during it — see `CLAUDE.md`'s AUTO-EXECUTE list note on
`capture_progress.py`). This is almost identical to Model A's own 11.04-hour
duration, which makes sense in hindsight: both were trained on the exact same
~1.64B-token budget (133,500 steps x 24 x 512), same batch size, same
benchmarked throughput — only the data mix differs. The original 48-72h
HARD STOP estimate for this run predates that fixed-budget decision and was
never revisited against it; it was sized more conservatively than the actual
compute budget ended up requiring.

**Own-test-set perplexity: 47.03** (via `evaluation/perplexity.py`'s
cross-model matrix — see `evaluation/final_report.md` for the full 4x4
comparison against Model A, Model A-Instruct, and Model C, and
`community_article/ARTICLE.md` for the full qualitative writeup, sample
generations, and loss landscape plot).
