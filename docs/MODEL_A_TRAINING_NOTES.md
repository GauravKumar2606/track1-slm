# Model A — Training Summary

Trained: completed 2026-09-13. English-only corpus (wikitext + pg19/Gutenberg
+ openwebtext + c4). First of the three-model comparison (see CLAUDE.md's
"Three-Model Comparison Design").

## Architecture / Config (`configs/model_117M.yaml`)

| Parameter | Value | Meaning |
|---|---|---|
| Architecture | GPT-style decoder-only Transformer | Same family as GPT-2 |
| `vocab_size` | 32,000 | SentencePiece BPE vocab, shared across A/B/C |
| `context_length` | 512 | Max tokens of context the model attends over per example |
| `n_layers` | 12 | Transformer decoder blocks stacked |
| `n_heads` | 12 | Attention heads per block (head_dim = 768/12 = 64) |
| `d_model` | 768 | Hidden/embedding dimension |
| `d_ff` | 3,072 | Feed-forward inner dimension (4x `d_model`) |
| `dropout` | 0.1 | Applied to attention + residual paths |
| **Total trainable parameters** | **110,025,216 (~110M)** | Unique params — `token_emb`/`head` weights are tied, so summing all 149 state_dict tensors naively gives 134,601,216 (double-counts the tied embedding matrix, 32,000x768 = 24,576,000 extra) |

## Training hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `batch_size` | 24 | Sequences processed per optimizer step |
| `max_steps` | 133,500 | Total optimizer steps for the full run — one "step" = one forward+backward+weight-update on one batch of 24x512 tokens (12,288 tokens/step) |
| Total tokens trained on | ~1.64B | `133,500 x 24 x 512` — a budget, tokens can repeat (12.02B train tokens available, so each token seen ~0.14x on average — well under 1 epoch) |
| `learning_rate` (peak) | 3e-4 | AdamW peak LR |
| `warmup_steps` | 2,000 | Linear warmup from 0 to peak LR |
| LR schedule | Cosine decay | Decays from peak down to 0.1x peak (floor) after warmup, over `max_steps` |
| `weight_decay` | 0.1 | AdamW |
| Gradient clipping | max norm 1.0 | |
| Mixed precision | fp16 autocast + GradScaler | |
| `checkpoint_every` | 1,000 steps | Periodic checkpoint save (kept: last 3 + best + final, via rotation — see `training/train.py`'s `rotate_checkpoints()`) |
| GPU | RTX 5070 Ti Laptop, 12GB VRAM | Benchmarked ~42,451 tok/s at this batch size (`training/benchmark_throughput.py`) |
| Actual training time | ~10.7-11 hours | Matches the pre-run benchmark estimate |

## Data (Model A = English-only)

| Split | Text size | Tokens |
|---|---|---|
| train | 54.09 GB | 12,016,279,511 (~12.02B) |
| val | 3.07 GB | 727,573,115 (~727.6M) |
| test | 3.06 GB | 712,448,359 (~712.4M) |
| **Total** | **60.2 GB** | **~13.46B tokens** |

Sources: `Salesforce/wikitext` (wikitext-103-raw-v1), `sedthh/gutenberg_english`
(pg19 replacement), `Skylion007/openwebtext`, `allenai/c4` (English, capped at
2M examples). Cleaned via `data/clean_text.py` (HTML/URL stripping, non-English
filter, hash-set dedup), split 90/5/5 via `data/prepare_splits.py --model a`.

## Results

| Metric | Value |
|---|---|
| Best val_loss | **3.5159** at step 128,500 (saved as `checkpoints/model-a/best.pt`) |
| Final val_loss | 3.5373 at step 133,000 (essentially the same — plateaued, not diverging) |
| Val_loss behavior, last ~20k steps | Flat/noisy between 3.51-3.57 — converged, no further improvement, confirmed directly from TensorBoard event logs in `logs/model-a/` |
| Perplexity (via `training/validate.py` on `best.pt`) | **34.98** |
| Sample generation (`"The capital of France is"`) | `"The capital of France is called the city of Salzburg."` — coherent English grammar, factually wrong (expected at 110M scale / this data budget) |

## Checkpoint file structure (`checkpoints/model-a/best.pt`, confirmed by direct inspection)

A `.pt` file is a serialized Python dict (via `torch.save`), not human-readable
as text — open it with `torch.load(path, map_location="cpu")` in Python. It
contains exactly 4 top-level keys:

- **`model_state_dict`** — 149 tensors: `token_emb.weight` (32000x768),
  `pos_emb.weight` (512x768), per-block `blocks.N.{ln1,ln2}.{weight,bias}`,
  `blocks.N.attn.{qkv_proj,out_proj}.{weight,bias}`,
  `blocks.N.ffn...` (x12 blocks), final `ln_f.{weight,bias}`, `head.weight`
  (tied to `token_emb.weight`)
- **`optimizer_state_dict`** — AdamW's momentum buffers (148 state entries,
  one per unique parameter tensor; 1 param group)
- **`step`** — 128,500 (which training step this checkpoint is from)
- **`best_val_loss`** — 3.515874457359314 (full precision)

Files kept on disk after rotation: `best.pt`, `ckpt_133500.pt` (final),
`ckpt_133000.pt`, `ckpt_132000.pt` (last 3 periodic, per
`--keep-last-checkpoints 3` default).

## How to reproduce / verify these numbers

```
python training/validate.py --checkpoint checkpoints/model-a/best.pt --model a --config configs/model_117M.yaml
python inference/generate.py --checkpoint checkpoints/model-a/best.pt --prompt "The capital of France is" --max_tokens 100 --temperature 0.7 --top_k 50
```

## What's still open (see CLAUDE.md for the full plan)

- Model B (English + C#, from scratch) — not yet trained
- Model C (fine-tuned from this checkpoint on C#-only) — not yet trained
- Full three-way evaluation (`evaluation/compare_models.py`,
  `evaluation/perplexity.py`, `evaluation/generate_report.py`) — needs B and
  C trained first for the complete comparison
- Quantization + deployment (Phase 7) — not yet started
