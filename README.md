# Track 1 SLM — Small Language Model Training & Comparison

A from-scratch GPT-style decoder-only transformer (110M parameters), trained
and fine-tuned into four variants on a single consumer laptop GPU (RTX 5070 Ti
Laptop, 12GB VRAM), to study how training data composition and staging (joint
vs. sequential, with vs. without instruction fine-tuning) affect a small
language model's behavior.

**Start here for the full narrative** (purpose, architecture, decisions made
and why, screenshots, results, and an honest account of what broke along the
way — including a real catastrophic-forgetting incident and a data-pipeline
bug that corrupted an entire training corpus): **[`community_article/ARTICLE.md`](community_article/ARTICLE.md)**.

For the raw evaluation numbers: **[`evaluation/final_report.md`](evaluation/final_report.md)**.

## The four models

| Model | What it is | Trained via |
|---|---|---|
| **A** | From-scratch pretrain, English-only | `training/train.py --model a` |
| **A-Instruct** | Model A, instruction fine-tuned | `training/train_instruct.py --model a-instruct --init-from checkpoints/model-a/best.pt` |
| **B** | From-scratch pretrain, English + C# + instructions, jointly | `training/train.py --model b` |
| **C** | Model A-Instruct, fine-tuned further on C# only | `training/train.py --model c --init-from checkpoints/model-a-instruct/best.pt` |

All four share the exact same architecture and tokenizer (see below) — that's
what makes comparing them meaningful.

## Solution structure

```
track1-slm/
├── README.md                    — this file
├── CLAUDE.md                    — full technical spec / orchestrator instructions (authoritative project history)
├── .gitignore
│
├── docs/                        — project history and design docs
│   ├── MODEL_A_TRAINING_NOTES.md
│   ├── MODEL_A_INSTRUCT_TRAINING_NOTES.md
│   ├── MODEL_B_TRAINING_NOTES.md
│   ├── MODEL_C_TRAINING_NOTES.md       — includes the corrupted-data-bug writeup and rebuild
│   ├── SESSION_NOTES_PHASE1-4_PREP.md  — early data-pipeline design notes
│   └── TRACK1_SLM_COURSE.md            — the project reframed as a structured course
│
├── community_article/           — the public-facing writeup, self-contained
│   ├── ARTICLE.md                — full narrative: purpose, architecture, decisions, results, screenshots
│   └── screenshots/              — curated training-progress + loss-landscape + demo screenshots
│
├── configs/                      — architecture + hyperparameter YAML per training run
│   ├── model_117M.yaml                 — Model A / B (from-scratch pretrain)
│   ├── model_117M_instruct.yaml        — Model A-Instruct fine-tune
│   └── model_117M_finetune_c.yaml      — Model C fine-tune
│
├── model/                        — the GPT architecture itself
│   ├── config.py                       — loads a YAML config into a dataclass
│   ├── attention.py                    — causal multi-head self-attention
│   ├── transformer.py                  — one pre-norm decoder block
│   ├── model.py                        — full GPTModel (embeddings + N blocks + weight-tied output head)
│   └── verify_model.py                 — parameter count + forward-pass shape check
│
├── tokenizer/                    — one shared SentencePiece BPE tokenizer (vocab_size=32000), used by every model
│   ├── train_tokenizer.py
│   ├── tokenize_corpus.py               — flat-stream tokenization (models a/b/c)
│   ├── verify_tokenizer.py
│   ├── tokenizer.model / .json / .vocab / vocab.txt   — the trained tokenizer artifacts (tracked, small)
│
├── data/                          — data acquisition + cleaning + splitting (outputs gitignored, see below)
│   ├── download_datasets.py            — wikitext, pg19, openwebtext, c4, csharp, instruct
│   ├── clean_text.py                   — HTML/URL/length/dedup cleaning (in-memory or streaming per source size)
│   ├── sanitize_corpus.py              — one-off remediation script (control-character stripping; see its docstring)
│   ├── prepare_splits.py               — combines cleaned sources into each model's train/val/test.txt (--model a|b|c)
│   ├── prepare_instruct_data.py        — masked, fixed-length example format for Model A-Instruct's fine-tune
│   ├── sample_instruct_for_model_c.py  — reference only; an earlier ~15% instruct-blend attempt, reverted (see docs/MODEL_C_TRAINING_NOTES.md)
│   └── raw/ cleaned/ tokenized/ model-a/ model-b/ model-c/    — GITIGNORED generated data (~256GB total)
│
├── training/                     — training loops and monitoring tools
│   ├── train.py                        — flat-stream training (models a, b, c — c via --init-from)
│   ├── train_instruct.py               — masked, per-example training (model a-instruct)
│   ├── validate.py                     — checkpoint val_loss + perplexity + sample generation
│   ├── benchmark_throughput.py         — VRAM/tokens-per-second benchmark used to pick batch_size=24
│   ├── capture_progress.py             — periodic TensorBoard -> PNG screenshot capturer
│   └── plot_loss_landscape.py          — post-hoc loss-surface visualization (Li et al.), any checkpoint
│
├── evaluation/                   — cross-model comparison
│   ├── common.py                       — shared model/data loading, generation, perplexity computation
│   ├── compare_models.py               — qualitative side-by-side responses -> model_comparison.json
│   ├── perplexity.py                   — full cross-perplexity matrix -> perplexity_results.json
│   ├── generate_report.py              — renders both into final_report.md
│   ├── eval_questions.json
│   ├── final_report.md                 — tracked: the actual results
│   └── model_comparison.json / perplexity_results.json   — tracked: the raw results data
│
├── inference/
│   └── generate.py                     — CLI text generation (temperature/top_k/top_p/repetition_penalty/no_repeat_ngram_size)
│
├── api/
│   └── main.py                         — FastAPI server (/generate, /health)
│
├── deployment/                   — quantized inference + public demo
│   ├── quantize.py                     — int8 dynamic quantization for all 3 deployed models
│   ├── app.py                          — Gradio 3-way comparison UI (A-Instruct / B / C side-by-side)
│   ├── requirements.txt
│   └── model_int8/                     — GITIGNORED quantized weights (~600MB, regenerate via quantize.py)
│
├── checkpoints/                  — GITIGNORED trained model weights (~20GB, regenerate via training/)
└── logs/                         — GITIGNORED TensorBoard logs + full screenshot history (~25MB; curated subset tracked in community_article/screenshots/)
```

## Reproducing this project

The full phase-by-phase spec (with exact commands, hyperparameter rationale,
and every HARD STOP / auto-execute distinction) lives in `CLAUDE.md`. Short
version:

1. **Environment**: Python 3.12, PyTorch 2.14+ with CUDA, `pip install
   datasets huggingface_hub tokenizers sentencepiece transformers tensorboard
   fastapi uvicorn gradio tqdm numpy pandas`.
2. **Data**: `python data/download_datasets.py` → `python data/clean_text.py`
   → `python data/prepare_splits.py --model {a,b,c}`.
3. **Tokenizer**: `python tokenizer/train_tokenizer.py` (once, on Model A's
   English data — reused by every model since) → `python
   tokenizer/tokenize_corpus.py --model {a,b,c}`.
4. **Training**: see the table above for each model's exact command. Every
   run benefits from `python training/capture_progress.py --model <x>`
   running alongside it in a separate terminal.
5. **Evaluation**: `python evaluation/compare_models.py` → `python
   evaluation/perplexity.py` → `python evaluation/generate_report.py`.
6. **Deploy**: `python deployment/quantize.py` → `python deployment/app.py`.

Checkpoints, quantized weights, and raw/intermediate data are not included in
this repository (see `.gitignore` — collectively too large for a normal git
repo, ~276GB) — they're fully regenerable from the steps above, deterministic
given the same source datasets and the tokenizer artifacts that *are* tracked
here (`tokenizer/tokenizer.model` etc.).

## Data sources and licensing

Six datasets were used (wikitext, pg19, openwebtext, c4, a C# source from
`bigcode/the-stack-dedup`, and instruction data from `teknium/OpenHermes-2.5`),
each under a different license — including one, OpenHermes-2.5, with **no
license stated at all** on its dataset card (verified directly, not assumed).
Full license table and a basic model card for the four trained checkpoints:
see `community_article/ARTICLE.md`'s Appendix.

**This repository's own code has no LICENSE file yet.** If you intend to let
others reuse the training/data/eval code, add one (MIT and Apache 2.0 are
common defaults for this kind of research code) before treating this as
publicly reusable.
