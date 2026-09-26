# Training and Comparing Four Small Language Models on a Single Laptop GPU

*A from-scratch GPT build, an instruction fine-tune, a joint multi-source pretrain,
and a sequential domain fine-tune — what worked, what broke, and what it taught us
about catastrophic forgetting, data pipelines, and small-model limits.*

---

## 1. Purpose — why do this at all

The goal was to answer a concrete, practical question at small scale, on hardware
anyone can actually own: **does *how* you combine training data and stages matter
as much as *what* data you use?**

Four variants of the same 110M-parameter architecture were built to isolate
different training decisions:

- **Model A** — a from-scratch English-only pretrain. The common ancestor
  everything else in this project builds on.
- **Model A-Instruct** — Model A, instruction-tuned on top.
- **Model B** — English + C# + instructions, all learned *jointly*, from scratch,
  in one run (not derived from Model A at all — its own independent pretrain).
- **Model C** — the *same* English+instruction foundation as Model A-Instruct
  (literally the same checkpoint), then C#-specialized via a **separate,
  sequential** fine-tuning stage.

The comparison isolates several independent questions:
- **A vs. A-Instruct** — what does instruction fine-tuning change (and cost),
  relative to the untouched base model? A clean, single-variable comparison.
- **A vs. B** — the comparison this project actually started with: same
  from-scratch procedure and token budget, only the data mix differs. Also
  clean and single-variable.
- **A-Instruct vs. C** — what does further fine-tuning on new data cost you,
  relative to the exact checkpoint you started from? (Specialization vs.
  forgetting.)
- **B vs. C** — does learning everything jointly beat learning it in stages?
  (Joint vs. sequential multi-task/multi-domain learning.)

**A note on how this article's own scope changed while writing it**: Model A
was deliberately dropped from the *primary* three-way comparison partway
through this project (see §4) once its own results made clear it doesn't
follow instructions — Model A-Instruct seemed the more informative model to
compare against B and C. That was a reasonable call at the time, but it meant
Model A's own numbers — a real, fully-trained checkpoint — were left out of
the final writeup entirely. This version corrects that: Model A is back in,
with the same rigor (screenshots, loss landscape, perplexity matrix) as the
other three, both because it's a genuine data point and because "the base
model doesn't follow instructions" is much more convincing shown than asserted.

This was never about building a production model. It was about building something
small enough to fully understand, fully instrument, and fully break — and then
learning from exactly how it broke.

## 2. Architecture

All four models share **one identical architecture** and **one identical
tokenizer** — this is what makes any cross-model comparison meaningful in the
first place.

| Parameter | Value | Role |
|---|---|---|
| `vocab_size` | 32,000 | SentencePiece BPE vocabulary size |
| `context_length` | 512 | Max attended tokens per forward pass |
| `n_layers` | 12 | Stacked pre-norm transformer decoder blocks |
| `n_heads` | 12 | Attention heads per block (`head_dim` = 768/12 = 64) |
| `d_model` | 768 | Token embedding / residual stream width |
| `d_ff` | 3,072 | Feed-forward hidden width (4x `d_model`, standard convention) |
| `dropout` | 0.1 | Applied in attention and residual paths |

**Total trainable parameters: 110,025,216** — verified two independent ways:
`sum(p.numel() for p in model.parameters())`, and by hand from the six numbers
above. The architecture is a standard GPT-2-small-style decoder: causal
multi-head self-attention, pre-norm blocks, GELU feed-forward, and **weight tying**
between the input token embedding and the output projection (`self.head.weight =
self.token_emb.weight` — literally the same tensor, not a copy, which is why a
naive `state_dict()` sum overcounts parameters by ~24M).

The tokenizer (SentencePiece BPE, 32,000 merges) was trained **once**, on Model
A's English corpus, and reused unmodified across every model since — a shared
tokenizer is a prerequisite for any of the perplexity or generation comparisons
below to mean anything.

## 3. Data sources — what was used, how much, and why

| Source | Dataset | Used by | Cleaned size | Role |
|---|---|---|---|---|
| `wikitext` | Salesforce/wikitext (103-raw-v1) | A, B | 527 MB | General English prose |
| `pg19` | sedthh/gutenberg_english | A, B | 16.9 GB | Long-form literary English |
| `openwebtext` | Skylion007/openwebtext | A, B | 38.46 GB | Web-scraped English |
| `c4` | allenai/c4 (English, capped 2M examples) | A, B | 4.34 GB | Clean web text |
| `csharp` | bigcode/the-stack-dedup, `c-sharp` config, capped 2M examples | B, C | ~5.46 GB (after a major bug fix, see §11) | C# source code |
| `instruct` | teknium/OpenHermes-2.5 (~2M raw conversations) | A-Instruct, B | 1.47 GB | Instruction/response pairs |

**Actual tokens trained on vs. tokens available** — a distinction worth making
explicit, since they're very different numbers:

- **Model A**: pool of 4 English sources (~60 GB combined), but trained on a
  **fixed budget of ~1.64B tokens** (133,500 steps x 24 batch x 512 context) —
  a deliberate choice, not a full pass over the pool. Random-offset sampling with
  replacement, not sequential coverage.
- **Model B**: pool of all 6 sources, ~15.18B tokens total — trained on the
  **same ~1.64B-token budget** as Model A, so the comparison holds the compute
  budget constant while changing only the data mix.
- **Model A-Instruct**: 996,789 cleaned OpenHermes-2.5 examples (train 897,110 /
  val 49,839 / test 49,840) — trained on ~480K of those examples (20,000 steps x
  24 batch), about 53% coverage by example count.
- **Model C**: 1,780,895,852 real train tokens (after the data-pipeline fix) —
  trained on ~245.8M sampled tokens (20,000 steps x 24 x 512), about **13.8%
  coverage**.

Why C# and why OpenHermes specifically: C# needed a source with real license
clarity and pre-existing deduplication at scale — `bigcode/the-stack-dedup` was
chosen after two dead ends (see §4). OpenHermes-2.5 was chosen over a smaller
candidate (Dolly-15k) specifically so **one dataset and one cleaning pipeline**
could serve both Model A-Instruct's dedicated fine-tune and Model B's data-mix,
avoiding building two separate instruction-data pipelines for the same
comparison.

## 4. Decisions made along the way, and why

This project changed shape multiple times as real results came in. The most
consequential decisions, in order:

- **The comparison was redefined mid-project.** The original plan was a clean
  A vs. B vs. C. After Model A finished training, real testing showed the
  expected base-model limitation: it doesn't follow instructions, just completes
  text. Rather than force an unfair comparison, the design was changed to
  **A-Instruct vs. B vs. C**, adding a whole new fine-tuning stage
  (`train_instruct.py`, masked loss) instead of pretending the gap didn't exist.
- **SQLite deduplication was replaced with in-memory hashing.** The first cleaning
  pipeline used SQLite as a disk-based hash store for paragraph deduplication.
  At tens of millions of inserts, WAL checkpointing saturated disk I/O and
  stretched cleaning jobs to multiple days. Switched to an in-memory `set()` of
  8-byte BLAKE2b hashes (well under 1GB for ~8M entries) — the same job dropped
  to about 15 minutes.
- **A combined `corpus.txt` step was removed.** The original pipeline wrote a
  full combined corpus before splitting it into train/val/test — needing the
  full combined size on disk **twice** simultaneously. This is what actually ran
  the disk out of space building Model B once. Fixed by streaming directly from
  cleaned source files into the three split files.
- **Checkpoint rotation was added.** Saving a full ~1.32GB checkpoint (weights +
  AdamW's 2x optimizer state) every 1,000 steps with no cleanup would have used
  ~178GB for one model's run alone. `rotate_checkpoints()` now keeps only the
  most recent N periodic checkpoints plus `best.pt` and the final checkpoint.
- **The C# source went through two dead ends before landing on the right one.**
  `bigcode/the-stack-v2` was tried first and rejected — its parquet shards
  turned out to contain only metadata, no actual file content (verified via a
  direct schema inspection, not assumed). `bigcode/the-stack-dedup` was
  confirmed to have real content before switching.
- **Model C's data composition changed, then changed back.** A ~15% blend of
  instruction data into the C# fine-tune corpus was tried first, specifically to
  guard against forgetting instruction-following. It was reverted the same day,
  on the reasoning that the starting checkpoint (Model A-Instruct) already had
  that behavior — diluting the C# corpus to protect it seemed redundant. **This
  reasoning turned out to be a real mistake** — see §11.
- **Every tokenization and training command is hand-executed, never automated.**
  A standing project rule: any script that trains or tokenizes is treated as a
  manual, deliberate action — never auto-triggered, always confirmed explicitly
  before running. This is a safety/oversight choice, not a technical constraint.

## 5. The machine — configuration, constraints, and how they were handled

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti Laptop, 12GB VRAM |
| CUDA | 13.0, compute capability sm_120 (Blackwell) |
| Framework | PyTorch 2.14.0, Python 3.12 |
| OS | Windows 11 |
| Storage | Single local drive, 392GB total |

This was all done on **one consumer laptop**, not a cluster or a cloud instance —
every constraint below came from that.

**VRAM was the binding constraint**, and it was profiled *before* committing to a
training config, not discovered mid-run:

| Batch size | Peak VRAM | Throughput |
|---|---|---|
| 8 | 4.65 GB | 38,814 tok/s |
| 16 | 7.71 GB | 41,939 tok/s |
| **24 (chosen)** | **10.74 GB** | **42,451 tok/s** |
| 32 (original default) | 13.77 GB (!) | 10,234 tok/s — ~4x slower |

Batch size 32 silently exceeded real VRAM and spilled into Windows' slow
shared-GPU-memory fallback — throughput collapsed to a quarter of batch size 24's,
despite "fitting" in the sense that it didn't crash. This was caught with a
dedicated `benchmark_throughput.py` script *before* any real training run, which
is why every model in this project used batch size 24.

**GPU contention from a background process was also a real, measured problem**:
Ollama running concurrently with training pushed combined VRAM demand to
~11-12.4GB against the 12GB dedicated ceiling, causing ~1.5GB to spill into slow
shared memory and dropping GPU utilization from 100% to 74-75%. Diagnosed via
`nvidia-smi` and Windows Task Manager, not guessed — the fix was simply: close
Ollama (and this coding session) before starting a real training run.

**Disk space ran out twice** during data preparation (the `corpus.txt` double-write
issue and unbounded checkpoint accumulation, both described in §4) — both fixed
at the pipeline level rather than by buying more disk.

## 6. Key takeaways

- **A 110M-parameter decoder-only transformer trains end-to-end on a single
  12GB laptop GPU in a reasonable time budget** (~11.5 hours for ~1.64B tokens),
  *if* you benchmark VRAM/throughput before picking a batch size instead of
  after.
- **Text-cleaning heuristics tuned for English prose actively destroy source
  code if reused unmodified.** A "drop lines under 20 characters" rule silently
  deleted every standalone `{`/`}` from an entire C# corpus; a "deduplicate
  identical paragraphs" rule kept only the *first* occurrence of universally
  common code like `using System;` across ~2 million files. Neither rule is
  wrong for prose. Both are catastrophic for code. Domain-specific data needs
  domain-aware cleaning, not just a bigger byte-size threshold.
- **Sequential fine-tuning without replaying earlier-stage data has a real,
  measurable forgetting cost** — confirmed both qualitatively (a model that used
  to answer instruction-formatted prompts started returning nothing) and
  quantitatively (see §11).
- **Decoding-time fixes (repetition penalty, n-gram blocking, nucleus sampling)
  mitigate but do not cure small-model repetition loops.** They shift *which*
  degenerate pattern appears; they don't remove the underlying capacity
  limitation.
- **Verifying actual artifacts (not log summaries) repeatedly caught real bugs**
  that looked fine from a distance: a miscounted example count that turned out
  to be a blank-line-join artifact, an API response field that was hardcoded and
  silently lied about which model answered, a length filter that looked
  reasonable until checked against 500,000 real lines of code.
- **A comparison is only as good as the variables it actually isolates.** A-Instruct
  vs. B changes two things at once (training procedure *and* data mix) and had to
  be explicitly flagged as not-clean, rather than quietly treated as equivalent
  to the other two (clean) comparisons.
- **Instruction fine-tuning has a small, measurable cost on the base model's
  original task, not just benefits.** Model A-Instruct scores *worse* than raw
  Model A at predicting Model A's own held-out English text (63.44 vs. 48.75
  perplexity, §8) — becoming better at following instructions moved the model
  measurably further from pure prose continuation. Neither number is "wrong";
  they're different objectives, and the tradeoff is real and quantifiable, not
  just a qualitative impression.

## 7. Model training — screenshots

Training always logs scalars (loss, learning rate, tokens/sec, validation loss,
and — from a certain point onward — gradient norm) to TensorBoard regardless of
whether anything is watching. Two different ways of visualizing that data show
up below, for two different reasons.

**Model A** (from-scratch English-only pretrain, ~11.5h, step 0 → 133,500) had a
screenshot capturer running live throughout — the most granular record of any
model in this project (139 real screenshots). Nine evenly-spaced samples, no
`grad_norm` panel (this run finished before that logging existed):

![Model A, step 0](screenshots/a_01_step000000.png)
![Model A, step 16,000](screenshots/a_02_step016000.png)
![Model A, step 33,100](screenshots/a_03_step033100.png)
![Model A, step 50,300](screenshots/a_04_step050300.png)
![Model A, step 67,500 (midpoint)](screenshots/a_05_step067500.png)
![Model A, step 84,600](screenshots/a_06_step084600.png)
![Model A, step 101,800](screenshots/a_07_step101800.png)
![Model A, step 119,000](screenshots/a_08_step119000.png)
![Model A, step 133,400 (final)](screenshots/a_09_step133400_final.png)

**Model C — rebuilt run** (C# fine-tune of Model A-Instruct, ~1.9h, step 0 →
20,000, after the data-pipeline fix in §11) had a screenshot capturer running
*live* every 5 minutes during training, and grad_norm logging was already in
place by this point — so all five metrics (train_loss, learning_rate,
tokens_per_second, val_loss, grad_norm) are visible throughout. Nine
evenly-spaced samples across the full run, out of 50 real captured screenshots:

![Model C, step 0](screenshots/c_01_step00000.png)
![Model C, step 2,700](screenshots/c_02_step02700.png)
![Model C, step 5,300](screenshots/c_03_step05300.png)
![Model C, step 7,900](screenshots/c_04_step07900.png)
![Model C, step 10,600 (midpoint)](screenshots/c_05_step10600.png)
![Model C, step 13,200](screenshots/c_06_step13200.png)
![Model C, step 15,900](screenshots/c_07_step15900.png)
![Model C, step 18,500](screenshots/c_08_step18500.png)
![Model C, step 19,900 (final)](screenshots/c_09_step19900_final.png)

Watching train_loss and grad_norm settle across these nine frames is a genuinely
useful way to see the fine-tune converge in real time — sharp drop through the
first ~10,000 steps, then the plateau-with-noise pattern discussed in
`MODEL_C_TRAINING_NOTES.md`.

**Loss landscape** (Li et al.'s "Visualizing the Loss Landscape of Neural Nets"
technique — 2 random filter-normalized directions, loss evaluated on a grid
around the actual trained checkpoint):

![Model C loss landscape](screenshots/c_10_loss_landscape.png)

A smooth, wide, symmetric basin with the checkpoint sitting cleanly at the
minimum — no sharp cliffs or secondary minima, a healthy sign for how this
checkpoint generalizes.

**Model A-Instruct and Model B did not have a live capturer running during their
training** — for A-Instruct, the capture script had an argument-validation bug
that rejected its model name outright before it could log anything (fixed
afterward, too late for that run); for Model B, the capture script was simply
never started alongside that run. Live periodic snapshots of *those* two runs
don't exist and can't be reconstructed after the fact.

What *does* survive, because training itself always writes it regardless of any
capture script: the complete TensorBoard event log for both runs. The full
training curves below were generated after training finished, straight from
those logs — real data, just plotted once at the end instead of every 5 minutes
during the run. **One real, unfixable gap**: neither chart has a `grad_norm`
panel, because gradient-norm logging was only added to the training scripts
*after* both of these runs had already finished — the data was simply never
recorded, so it can't be reconstructed the way the rest of the curve could be.

**Model A-Instruct** (English base + instruction fine-tune, 20,000 steps — 4
panels, no grad_norm):

![Model A-Instruct, full training curve](screenshots/a-instruct_01_full_curve.png)

**Model B** (English + C# + instructions, joint from-scratch, 133,500 steps — 4
panels, no grad_norm):

![Model B, full training curve](screenshots/b_01_full_curve.png)

**Loss landscapes for all four** — unlike the training-progress screenshots
above, this doesn't depend on any historical per-step logging at all; it only
needs the final trained checkpoint and the validation data, so it's fully
recoverable after the fact even for runs with no live capture:

![Model A loss landscape](screenshots/a_10_loss_landscape.png)

![Model A-Instruct loss landscape](screenshots/a-instruct_02_loss_landscape.png)

![Model B loss landscape](screenshots/b_02_loss_landscape.png)

All four show the same healthy pattern: a smooth, wide, symmetric basin with the
trained checkpoint sitting cleanly at the minimum (loss 3.571 for Model A, 1.944
for A-Instruct, 3.562 for B, 1.545 for Model C — the higher values for A and B
reflect harder, broader training distributions, not a worse fit) — no sharp
cliffs or secondary minima on any of the four models.

## 8. Model comparison results

Full cross-perplexity matrix (every model scored against every model's held-out
test set — the diagonal is each model's performance on its own data):

| Model \ Test set | A | A-Instruct | B | C |
|---|---|---|---|---|
| **A** | 48.75 (own) | 14.71 | 46.05 | 16.38 |
| **A-Instruct** | 63.44 | 6.31 (own) | 61.37 | 15.34 |
| **B** | 50.08 | 8.33 | 47.03 (own) | 7.23 |
| **C** | 86.02 | 10.30 | 80.45 | 5.51 (own) |

Reading this:
- **Model C is the best C# specialist by a wide margin** (5.51 vs. everything
  else in that column) — the fine-tune worked, as far as raw next-token
  prediction on C# goes.
- **Model C is also the worst on general/mixed text** (86.02 on A's test set,
  80.45 on B's) — the flip side of successful specialization.
- **Model B sits in the middle on most axes**, consistent with a generalist
  trained jointly on everything at once.
- **Model A-Instruct is *worse* than raw Model A at predicting Model A's own
  held-out English text** (63.44 vs. 48.75, A's own score). Instruction
  fine-tuning measurably shifted the model away from pure prose continuation —
  a real, quantified cost of that fine-tune, not just a qualitative impression.
- **Model C scores worse than Model B on A-Instruct's own test set** (10.30 vs.
  8.33) — despite literally starting from A-Instruct's exact weights, C is now
  *worse* at predicting A-Instruct's held-out instruction-response data than a
  model (B) that never had those weights at all. This is the clearest single
  number in this whole project for what catastrophic forgetting costs (see §11).

Qualitative testing (`compare_models.py`, same 8 questions across all four
models) makes the base-model gap immediately obvious. Asked "What is Python?",
"What is Machine Learning?", and "What is AGI?", **raw Model A simply echoed
the question back** with no continuation at all — it has no learned notion that
a question expects an answer, only that text continues somehow (and on these
particular prompts, it apparently "chose" to continue with nothing). Model
A-Instruct, by contrast, engages with every one of the same prompts. Model C,
asked "What is Machine Learning?", responded with a C# unit test
(`[TestCase("Standard","Domain")] public void
MachineLoadedStandard_ThrowsArgumentNullException()...`) — it now defaults to
C# code for *any* prompt, English or not.

**Gradio comparison demo** — a live, side-by-side interface built for this
project, showing Model A-Instruct, Model B, and Model C (the three models this
project's design ultimately centered the *live demo* around — see §1's note on
scope; Model A's results above come from the offline evaluation scripts, not
this interactive demo):

![Gradio UI, empty](screenshots/gradio_01_empty_ui.png)

![Gradio UI, "Write a C# for loop"](screenshots/gradio_02_filled_example.png)

Notice the qualitative difference even here: Model A-Instruct writes a
reasonable English explanation without real code; Model B produces confused,
off-topic text; Model C produces actual brace-structured C# — on-topic, though
not literally a correct for-loop.

**Try it live**: `https://b5f88a3be4a7c8d47a.gradio.live` (verified working as of
2026-09-26).

A note on that link's nature, since it's a real constraint worth being upfront
about: HuggingFace Spaces now requires a **PRO subscription** ($9/month) to host
any Gradio or Docker Space on their free CPU tier — only static HTML Spaces
(which can't run this app's actual PyTorch inference) remain free. Rather than
pay for permanent hosting, this demo is shared via Gradio's own built-in
`share=True` tunnel (`deployment/app.py`'s `demo.launch(share=True)`), which is
free and needs no HuggingFace account at all, but comes with real tradeoffs:

- The link is only reachable while the original machine keeps the script
  running — close the terminal, stop the process, or put the machine to sleep,
  and the link goes dead immediately.
- It also expires on its own after roughly 72 hours, even if left running.
- **A new random URL is generated every time the script restarts** — there is no
  way to "resume" the same link. If the link above is dead when you're reading
  this, that's why — the demo isn't necessarily gone, just needs restarting on
  the author's end (`python deployment/app.py`), which will produce a different
  URL than the one printed here.

## 9. Inference adjustments made during this project

The base decoding loop (temperature + top-k sampling) degenerated into repetition
loops on every model in this project, Model A included — a known small-model
failure mode, not specific to any one of them. Three additions were made,
applied consistently across
`inference/generate.py`, the FastAPI server, and the evaluation scripts:

- **Repetition penalty** (CTRL-style, Keskar et al. 2019): every token already
  generated gets its logit divided (if positive) or multiplied (if negative) by
  a penalty factor, making it less likely to repeat. Mitigates, doesn't cure —
  in testing, it reliably broke one repeated-sentence loop, but the model then
  settled into a *different*, narrower loop.
- **`no_repeat_ngram_size`**: hard-blocks completing any n-gram already seen
  earlier in the sequence. Unlike the penalty (which makes a repeat *less
  likely*), this makes an exact n-gram repeat *impossible* — a different failure
  mode than the penalty catches, so the two are used together.
- **Top-p (nucleus) sampling**: keeps the smallest set of highest-probability
  tokens whose cumulative probability exceeds a threshold, instead of a fixed
  top-k count. Adapts the candidate pool to how confident the model actually is
  at each step.

A real, separate bug was also caught and fixed here: the API's `/generate`
response had a **hardcoded** `"model": "track1-slm-model-a"` field, regardless of
which checkpoint was actually loaded — meaning it was possible to test the wrong
model entirely and have the API confidently tell you otherwise. Fixed to derive
the label from the actual loaded checkpoint path.

## 10. Temperature and other decoding parameters

Worth separating clearly: **training hyperparameters** (learning rate, batch
size, warmup steps) shape what the model *learns*; **decoding parameters** shape
how a *trained* model's output is sampled at generation time. They're
independent — the same checkpoint produces very different outputs under
different decoding settings.

Settings landed on after testing (used as the fixed defaults in the Gradio demo):

| Parameter | Value | Effect |
|---|---|---|
| `temperature` | 0.7 (adjustable in the demo) | Lower = more deterministic/confident; 1.0 = unmodified distribution |
| `top_k` | 40 | Only the 40 highest-probability tokens are candidates at each step |
| `top_p` | 0.9 | Nucleus sampling threshold, stacks with top_k |
| `repetition_penalty` | 1.3 | CTRL-style penalty strength |
| `no_repeat_ngram_size` | 3 | Blocks repeating any 3-gram already generated |

Decoding order matters and is fixed across every entry point in this project:
repetition penalty → n-gram ban → temperature scaling → top-k filter → top-p
filter → sample.

## 11. Did we face catastrophic forgetting, and how was it handled

**Yes — directly, and it took two separate investigations to fully understand.**

**First signal (qualitative):** after fine-tuning Model C on C# only, testing the
exact instruction-template prompt Model A-Instruct was trained on
(`"### Instruction: Write a C# for loop ### Response:"`) produced **empty output
in 2 of 3 runs** — the model was predicting end-of-sequence immediately instead
of engaging with the prompt, where Model A-Instruct (its own starting point)
produced a real, on-topic attempt.

**But the first fix attempt uncovered a bigger, separate bug first.** Investigating
*why* the model looked broken led to discovering that Model C's entire C#
training corpus was corrupted — a text-cleaning bug (see §4/§6) had stripped
every brace and nearly all repeated boilerplate from the data. **Model C was
retrained from scratch on properly-cleaned data** (re-downloaded, re-cleaned,
re-tokenized, retrained — documented in full in `MODEL_C_TRAINING_NOTES.md`).

**After the data fix, the forgetting was still present, but measurably less
severe:**
- Best validation loss improved from 1.4911 to 1.3651.
- The instruction-template prompt no longer went silent — all 3 test runs
  engaged and produced real code-shaped output (though not always a *correct*
  for loop).
- But the quantitative comparison still shows real forgetting: Model C scores
  **10.30 perplexity** on Model A-Instruct's own test set, **worse** than Model
  B's 8.33 — despite Model C starting from A-Instruct's exact weights and Model
  B never having them. Being "closer" in lineage did not protect that behavior
  from being overwritten by 20,000 steps of C#-only training with zero
  instruction-formatted examples in it.

**What this project did *not* do**: implement and test a replay-blend fix (mixing
a fraction of instruction data back into the C# fine-tune to actively counteract
forgetting). That was scoped, understood as the standard continual-learning
mitigation for exactly this problem, and **deliberately left as a documented next
step**, not executed — an honest limitation of this project's current state
rather than a claimed solution. The real lesson captured here is *how much*
forgetting occurred and *why* (no replay data during the fine-tune stage,
compounded initially by a corrupted-data bug that made things look even worse
than the underlying forgetting alone), not that it was solved.

## 12. Time taken — estimated vs. actual

| Model | Estimated | Actual | Notes |
|---|---|---|---|
| Model A (from scratch, ~1.64B tokens) | ~10.7h (down from ~44.5h at the original, VRAM-overflowing batch size of 32) | **11.04h** (step 0 → 133,400, from TensorBoard's own logged wall-clock timestamps) | Close to estimate |
| Model B (from scratch, joint mix, same ~1.64B-token budget) | 48-72h (conservative HARD STOP estimate) | **11.22h** (step 0 → 133,400, same method) | **Far under the estimate** — see the note below, this is the most surprising number in this table |
| Model A-Instruct (fine-tune, 20,000 steps) | ~1.6h | **2.26h** (step 0 → 19,900) | Slightly longer than estimated |
| Model C rebuilt (fine-tune, 20,000 steps) | ~1.6-1.7h | **1.89h** (step 0 → 19,900) | Close to estimate |

These are now precise, not derived or estimated: every TensorBoard scalar
carries its own `wall_time` (a real Unix timestamp) alongside the step number,
so the exact duration of every run is recoverable directly from the log
itself, independent of screenshots or checkpoint file timestamps — including
for Model B, which has neither a periodic screenshot record nor an easily
inferred start time from surviving checkpoints (its earlier checkpoints were
rotated out), but *does* still have this.

**Why Model B's real duration is the most interesting number here**: it took
almost exactly as long as Model A (11.22h vs. 11.04h) — not the 48-72h
originally estimated. In hindsight this makes complete sense and shouldn't
have been surprising: Model B was deliberately trained on the **same
~1.64B-token budget** as Model A (133,500 steps, same batch size, same
benchmarked throughput), specifically so the comparison would hold compute
constant while only changing the data mix (§3/§4). The 48-72h estimate was set
early in the project, before that fixed-budget decision was locked in, and
was never revisited against the number that actually mattered — token budget,
not the size of the available data pool (15.18B tokens, ~9x more than what was
actually sampled). It's a small, harmless planning error in retrospect, but a
good example of an estimate that quietly outlived the decision that made it
obsolete.

The estimates came from a single benchmarked throughput number
(~42,451 tok/s at batch size 24) applied to each run's token budget — actuals ran
a bit longer across the board, consistent with real overhead (checkpoint saves,
validation passes, periodic screenshot I/O) not captured in the raw
tokens/second benchmark.

## 13. Why instruction fine-tuning, and how it actually works

**Why**: the raw pretrained model (Model A) completes text, but doesn't answer
questions. Given `"What is the capital of France?"`, a base model has no learned
distinction between "this is a question expecting an answer" and "this is just
more text to continue" — it might continue with more questions, or trail off, or
hallucinate confidently. Real usability requires teaching the model that a
specific format (`"### Instruction: ... ### Response: ..."`) means "the second
part should actually answer the first part."

**How it works, mechanically:**
1. Every training example is `(instruction, response)`, formatted as one string:
   `"### Instruction: {question} ### Response: {answer}"`.
2. The whole string is tokenized, but the **labels are masked** for every
   position in the prompt: `label = -100` (PyTorch's `ignore_index` convention)
   for every prompt/padding token, and the real target token ID only for
   response positions (+ the trailing end-of-sequence token).
3. During training, cross-entropy loss is computed with `ignore_index=-100` — so
   the gradient only ever pushes the model to predict *response* tokens well.
   The model still *sees* the instruction as context (full self-attention over
   the whole sequence), it's just never penalized or rewarded for how well it
   predicts the instruction text itself, only the answer.
4. This was verified correct before running on real data, using a synthetic
   example: decoding just the non-masked (response) positions reproduced the
   original response text exactly.

**A real before/after example** (same prompt, same decoding settings, tested
directly against Model A before this fine-tune and Model A-Instruct after):

> Prompt: *"Is C# a programming language?"*
> **Before (Model A/B)**: coherent, on-topic discussion of programming languages,
> before degrading into a `"C#: C#: C#:"` repetition loop.
> **After (Model A-Instruct)**: *"...It is a programming language that is used
> for various purposes... including C#, Python..."* — stays coherent longer, more
> directly responsive to the actual question asked.

The fine-tune's core goal — engaging with direct questions instead of just
continuing text — visibly worked. The repetition-loop tendency (see §9) is a
separate, not-fully-solved issue that persisted across every model in this
project regardless of fine-tuning.

## Appendix — data licensing, and a model card for the public demo

Since this project and its live demo are being shared publicly, here's the
information a reader would need to assess reuse and provenance — verified
directly against each dataset's official page, not assumed from memory.

### Source dataset licenses

| Source | Dataset | License | Notes |
|---|---|---|---|
| `wikitext` | Salesforce/wikitext | CC BY-SA 4.0 (also tagged CC BY-SA 3.0 / GFDL) | Derived from Wikipedia |
| `pg19` | sedthh/gutenberg_english | MIT | |
| `openwebtext` | Skylion007/openwebtext | CC0 (public domain) for the packaging | The creators note they don't own the underlying scraped web text itself, only the curation |
| `c4` | allenai/c4 | ODC-BY 1.0 | Also subject to Common Crawl's own terms of use |
| `csharp` | bigcode/the-stack-dedup | Custom "other" license (BigCode OpenRAIL-style terms) | Contains only permissively-licensed source repos (no copyleft/GPL); redistributing the dataset itself requires passing through its Terms of Use and respecting each file's original license/attribution; has a maintainer-run opt-out/removal process |
| `instruct` | teknium/OpenHermes-2.5 | **No license stated** on the dataset card | Verified directly (including the raw README) — genuinely unlicensed, not an oversight in this write-up. It's itself a compilation from multiple upstream sources with their own separate terms. |

**An honest, unresolved question**: whether *model weights* trained on a mix of
differently-licensed (and one unlicensed) corpora inherit any of those license
obligations is a genuinely unsettled question in current ML practice, not
something this project resolves. What's provided here instead is full
transparency about exactly what was used and where it came from, so a reader
can make their own informed judgment rather than relying on an unqualified
claim from this project.

### This project's own code

No `LICENSE` file exists in the project repository at the time of writing — if
you're the author publishing this and want others to be able to reuse the
training/data/eval code freely, adding one (MIT and Apache 2.0 are common
defaults for this kind of research code) is worth doing before wide release.

### Model card — Track 1 SLM (Model A / A-Instruct / B / C)

- **Model type**: 110,025,216-parameter GPT-style decoder-only transformer
  (12 layers, 12 heads, 768-dim, 512 context length), four variants sharing one
  architecture and tokenizer.
- **Intended use**: educational/research demonstration of how training data
  composition and staging (joint vs. sequential, with vs. without instruction
  fine-tuning) affects a small language model's behavior. Not intended for
  production use, factual question-answering, or any application where
  correctness matters.
- **Known limitations**:
  - All four models are small enough (110M parameters) that they degrade into
    repetition loops without decoding-time mitigation (§9), and even with it,
    can still produce incorrect, incoherent, or nonsensical output.
  - **Model A has no instruction-following ability at all** — it's a raw
    next-token-prediction base model. Asked a direct question, it may simply
    echo the question back with no answer (confirmed directly in §8's
    qualitative testing), continue it as if it were prose, or produce
    unrelated tangential text. This is expected, correct behavior for an
    un-instruction-tuned base model, not a defect — but it means Model A
    specifically should never be used for anything resembling a chat or
    Q&A interface.
  - Model C defaults to producing C# code for almost any prompt, including
    unrelated English questions (§8) — a direct consequence of its fine-tuning,
    not a bug, but a real behavioral limitation for general use.
  - None of the four models should be treated as factually reliable — see the
    "capital of France" and similar examples throughout this article for
    concrete evidence of confident, wrong output.
  - Training data includes web-scraped text (`openwebtext`, `c4`) which can
    carry the biases, errors, and occasionally low-quality content inherent to
    unfiltered web sources — no additional content filtering beyond the
    cleaning pipeline described in §4/§11 was applied.
- **Out-of-scope uses**: anything safety-critical, anything requiring factual
  accuracy, any chat/instruction-following use of Model A specifically (use
  A-Instruct instead), and any use case involving generating C# code intended
  for production without human review (Model C's code is frequently
  syntactically plausible but not necessarily correct or secure).

---

*All numbers in this article are pulled directly from this project's own
training logs, checkpoint metadata, and evaluation outputs — not estimated after
the fact. Where an exact figure wasn't recoverable from surviving artifacts
(§7's screenshot gap, §12's Model B duration), that's stated explicitly rather
than filled in.*
