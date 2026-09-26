# Model C — Training Notes

Started: 2026-09-14. C#-only fine-tune of Model A-Instruct's checkpoint
(see CLAUDE.md's "Three-Model Comparison Design", REVISED 2026-09-13).
Status: **complete, REBUILT 2026-09-15** — the first run (2026-09-14) used a
corrupted C# corpus (see "Data bug found and fixed" below); it was deleted
entirely and rebuilt from a re-downloaded, correctly-processed corpus. This
file documents the rebuilt run as current; the original run's results are
kept further down for the record, since the comparison between them is
itself an informative result.

## What kind of fine-tune this is — different from Model A-Instruct's

Both Model C and Model A-Instruct are fine-tunes (start from an existing
checkpoint, not random init), but they're mechanically different in the
same way Model B differs from Model A-Instruct's data handling:

| | Model A-Instruct | Model C |
|---|---|---|
| Starting weights | Model A (`checkpoints/model-a/best.pt`) | **Model A-Instruct** (`checkpoints/model-a-instruct/best.pt`) |
| Data format | Fixed per-example arrays, masked | **Flat continuous token stream, unmasked** — same mechanism as Model A/B, via `training/train.py` (not `train_instruct.py`) |
| Batching | Random example indices | **Random byte offsets** into the flat stream |
| Loss | Response-tokens-only (`ignore_index=-100`) | **Every token counts** — this is continued code-completion pretraining, not instruction-following, so there's no "prompt vs. response" distinction to mask |
| Script | `training/train_instruct.py` | `training/train.py` (same script Model A/B used, just with `--init-from`) |

So Model C is conceptually simpler than Model A-Instruct: it's Model
A-Instruct's weights, continuing standard next-token-prediction training,
just on a new (C#-only) corpus instead of the original English one.

## Architecture: same 110,025,216-parameter GPT-style decoder

Identical to every other model in this project. See
`MODEL_A_TRAINING_NOTES.md` for the full parameter-count derivation —
unchanged here.

## Weight initialization

| Step | What happens |
|---|---|
| Model construction | Fresh `GPTModel` built from `configs/model_117M_finetune_c.yaml` (same architecture numbers as every other config) |
| Weight loading | `--init-from checkpoints/model-a-instruct/best.pt` — loads Model A-Instruct's `model_state_dict` only. Optimizer state NOT restored (fresh AdamW), step counter starts at 0 |

This is the key design point from the revised three-model comparison:
Model C is base pretrain → instruction-tune → C# fine-tune, three stages
deep — it inherits whatever instruction-following behavior Model
A-Instruct already has, then gets C#-specialized on top of that.

## Data: C#-only (reverted from an earlier ~15% instruct blend — see CLAUDE.md)

| Split | Tokens (rebuilt 2026-09-15) | Tokens (original, corrupted, 2026-09-14) |
|---|---|---|
| train | 1,780,895,852 | 1,272,301,009 |
| val | 98,741,009 | 70,848,697 |
| test | 97,737,384 | 22,948,017 |
| **Total** | **~1.98B** | **~1.37B** |

Source: `bigcode/the-stack-dedup`, `c-sharp` config, capped at 2M examples.
Cleaned, split 90/5/5, and tokenized via the same pipeline as Model A/B
(`data/clean_text.py`, `data/prepare_splits.py --model c`,
`tokenizer/tokenize_corpus.py --model c`). The rebuilt corpus has ~40% more
tokens than the original despite the same 2M raw examples, because far
fewer characters are destructively deleted now (see next section) — more of
each real file's content survives cleaning.

### Data bug found and fixed (2026-09-14/15)

The original run's C# corpus was corrupted by a cleaning-pipeline bug, found
while investigating why Model C's outputs looked syntactically broken (no
braces, oddly truncated methods). Root cause, in two parts:

1. `data/download_datasets.py`'s `download_csharp()` wrote each raw file's
   full multi-line content to `csharp_train.txt` with only a trailing `"\n"`
   as separator — **no actual file-boundary marker**. Downstream,
   `clean_text.py` has no per-dataset code path; it splits everything on
   `"\n"` and treats every individual source line, from every file, as one
   independent "paragraph" (correct for prose, where `download_*` functions
   already write one paragraph per line — wrong for C#, where the file
   writer was instead dumping each file's *real internal* line breaks
   straight through).
2. Two of `clean_text.py`'s prose-tuned rules then ran on that flattened
   stream of individual code lines: the `<20-char` length filter (deletes
   "junk" short lines) silently deleted every standalone `{`/`}`/`};` line
   (1-2 chars) — confirmed via direct inspection: **zero** standalone brace
   lines in 500K sampled lines of the cleaned output. And the exact-line
   hash dedup (removes duplicate paragraphs — reasonable for prose, where a
   repeated paragraph usually is redundant) removed all but the **first**
   occurrence of universally common code lines: `"using
   System.Collections.Generic;"` appeared exactly **once** in the entire
   ~2M-example corpus, despite being near-universal boilerplate.

Net effect: Model C's original fine-tune never saw real, structurally
intact C# — it trained on a bag of surviving long, globally-unique lines
with no braces and almost no repeated boilerplate, joined by blank-line
separators.

**Fix** (contained entirely to `download_csharp()`, in
`data/download_datasets.py`): flatten each raw file's content to a single
line at download time (`" ".join(text.split())` — collapses all internal
whitespace, including real newlines, to single spaces), exactly like
`download_openhermes()` already does for multi-turn conversations. This
makes each line `clean_text.py` sees represent one *whole file* instead of
one internal source line — the length filter now almost never triggers on
real code, and the dedup now operates at whole-file granularity (matching
how `the-stack-dedup` is already near-deduplicated), instead of per
internal line. **No changes were needed to `clean_text.py`,
`prepare_splits.py`, or `tokenize_corpus.py`** — confirmed by re-running the
unmodified `--model c` tokenize step against the newly-fixed data with no
code changes there.

Rebuild procedure (2026-09-15): deleted `checkpoints/model-c/`,
`logs/model-c/`, `data/model-c/`, `data/tokenized/model-c/`,
`data/cleaned/csharp_cleaned_stream.txt`, and `data/raw/csharp/` (~15GB);
re-ran `download_datasets.py --dataset csharp` (2,000,000 examples,
6.58GB raw), `clean_text.py --dataset csharp` (1,998,857 of 2,000,000
survived — correct: whole-file dedup barely triggers, unlike the old
per-line dedup that gutted the corpus), `prepare_splits.py --model c`, and
`tokenize_corpus.py --model c`, then retrained from
`checkpoints/model-a-instruct/best.pt` with the same
`configs/model_117M_finetune_c.yaml` as before. Model A, Model
A-Instruct, and Model B were untouched by any of this — Model A-Instruct's
checkpoint never touched C# data, and Model B's C# share (~6% of its total
data pool, see the compute-cost discussion in this session's history) was
judged too small a fraction of its overall training signal to justify
redoing its full 48-72h pretraining run.

Spot-check confirming the fix, on the new cleaned output: standalone `{`
now appears in ~49% of a 200K-line sample (0% before), and `using System`
in ~38% (vs. a single occurrence across the entire previous corpus).

**Note on the ~15% instruct-blend reversal**: an earlier version of this
plan mixed a random ~15% sample of instruction data in alongside C#, to
guard against Model C forgetting instruction-following behavior during
the fine-tune. Reverted the same day: since Model C's *starting
checkpoint* (Model A-Instruct) already has that behavior baked into its
weights, diluting the C# corpus to "preserve" something already present
in the base weights was redundant. Data is back to pure C#, matching the
original design.

## Training hyperparameters (`configs/model_117M_finetune_c.yaml`)

| Param | Value | Meaning | Why this value |
|---|---|---|---|
| `batch_size` | 24 | Sequences per optimizer step | Same as every other model — benchmarked safe at this VRAM |
| `learning_rate` (peak) | 3e-5 | 10x lower than pretraining's 3e-4 | Standard fine-tuning practice — protects the already-trained weights from large disruptive updates |
| `warmup_steps` | 200 | Linear ramp 0 → peak | Proportionally short, matching the shorter total run |
| `max_steps` | **20,000** | Total optimizer steps | Sized to avoid re-training as long as the from-scratch runs — see coverage math below |
| `checkpoint_every` | 500 | Periodic checkpoint save | Finer-grained given the shorter run |

**Total steps needed: 20,000** (same budget as Model A-Instruct, by design
— both fine-tunes were sized to comparable compute, not to each other's
data volume).

## What 20,000 steps actually covers

Unlike Model A-Instruct (example-count coverage), Model C samples random
byte offsets from a flat 1.27B-token stream, so coverage is by tokens:

```
20,000 steps x 24 batch x 512 context = 245,760,000 tokens sampled
245,760,000 / 1,780,895,852 available train tokens (rebuilt corpus) = ~13.8% coverage
(one-pass-equivalent; actual sampling is random with replacement, not a
strict single sequential pass)
```

Estimated time: ~1.6-1.7 hours at this GPU's ~40,000-42,000 tok/s
(consistent with Model A-Instruct's actual run, same step/token budget).

## Step-by-step mechanics (same shape as Model A/B, not Model A-Instruct)

| # | What happens |
|---|---|
| 1 | Load Model A-Instruct's weights into a fresh `GPTModel`; fresh optimizer, step 0 |
| 2 | Load tokenized C# stream (memmap): 1,272,301,009 train tokens, 70,848,697 val tokens |
| 3 | **Per step:** LR = warmup (steps 0-199, linear 0→3e-5) then cosine decay to floor 3e-6 by step 20,000 |
| 4 | Sample batch: 24 random starting offsets into the flat stream, each giving a 512-token `x` and next-512-shifted `y` |
| 5 | Forward pass, plain `cross_entropy` (no masking — every token counts) |
| 6 | Backward + AdamW update, gradient clipped to max norm 1.0 — **`grad_norm` now logged** (added 2026-09-14, same change made to `train_instruct.py`) |
| 7 | Every 100 steps: log `train_loss`, `learning_rate`, `tokens_per_second`, `grad_norm` |
| 8 | Every 500 steps: evaluate `val_loss` on 20 random batches from the 70.8M-token val stream; save `best.pt` if improved |
| 9 | Every 500 steps: save `ckpt_{step}.pt`, rotate to keep only the 3 most recent (+ `best.pt` + final, never pruned) |
| 10 | Step 20,000: final checkpoint + last rotation pass |

## Monitoring set up for this run

- `python training/capture_progress.py --model c` — now includes `grad_norm` as a 5th plotted metric (2x3 grid), and the `--model` bug that silently broke this for Model A-Instruct's run is fixed
- `python training/plot_loss_landscape.py --checkpoint checkpoints/model-c/best.pt --model c` — to run once training finishes (not concurrently — competes for GPU with live training)

## Results (rebuilt run, 2026-09-15 — current)

**Best val_loss: 1.3651 at step 13,500** (`checkpoints/model-c/best.pt`).
Notably lower than the original corrupted-data run's best (1.4911) — this
model had much more learnable structure (real braces, real repeated
patterns) to fit. Late-run values stayed close to best (19,000: 1.3764;
19,500: 1.3704), still oscillating but not trending upward — a healthy
curve, similar in shape to the original run but at a meaningfully lower
loss floor throughout.

### Sample generations — real C# structure, and the instruction template no longer breaks

Tested `checkpoints/model-c/best.pt` on the same two prompt styles used for
the original run, same decoding settings
(`--temperature 0.7 --top_k 40 --top_p 0.9 --repetition_penalty 1.3
--no_repeat_ngram_size 3`):

**1. Raw code-completion** (`"public class LoanService"`) — 3 runs, all
producing full, brace-structured, plausible C#:
```
"public class LoanService : IHttpClient { private readonly ILogger _log; protected override
 async Task Run(ILog logger) { if (!_service.isReadOnly()) throw new ArgumentNullException
 ("library"); var settings = new List(); ... }"

"public class LoanService : IHttpRequestHandler { private readonly ILogger _log; public
 Loan Service(ILogger logger) { this._log = logger ?? throw new ArgumentNullException("log");
 } /// Initializes a new instance of the HttpMessage service ..."

"public class LoanService : IHttpClient { private readonly ILogger logger; public
 Loan(ILoggerLoggingContext context) => Log.Warning("Loaded"); } }"
```
A stark contrast with the original run's output on the same prompt (short,
brace-less fragments like `": IFilesServices"`). Real class bodies, real
constructors, real control flow, even an XML doc comment — the corrupted
corpus genuinely was the bottleneck.

**2. `### Instruction: Write a C# for loop ### Response:`** — 3 runs, none
hit immediate EOS this time (vs. 2 of 3 on the original run):
```
"...### Response: public static void Write(CodeDom.Compiler.GeneratedCodeData source)
 { if (source == null) throw new ArgumentNullException("source"); ... }"

"...### Response: using System; class Program { static void Main(string[] args) {
 Console.WriteLine("Loading program."); ... switch (input) { case "C": ... } }"

"...### Response: /// The current thread's state. private static readonly int
 CurrentThreadState = 0; }"
```

**Interpretation, with an honest caveat**: this is a real, qualitative
change from the original run — the model now engages with the instruction
template instead of terminating, and produces recognizable code shapes
(one response even builds a `using System; class Program { static void
Main(...) }` entry-point skeleton). None of the three responses actually
write a correct *for loop*, so this is not "instruction-following fully
preserved" — but "engages vs. terminates" is the specific failure mode
documented for the original run, and that failure mode is gone here.

This raises a real possibility (not proven, just plausible given the
evidence): the original run's severe forgetting may have been **compounded
by the corrupted data itself**, not purely a function of "zero instruction
examples during the C# fine-tune." Fitting real, well-formed code is
plausibly a gentler distribution shift from Model A-Instruct's existing
weights than fitting a garbled, brace-stripped, artificially-deduplicated
corpus was — the latter may have required more extreme parameter movement
that collaterally damaged the instruction template more than genuine code
fine-tuning does. This is a hypothesis worth keeping in mind, not a
settled conclusion — 3 runs per prompt isn't a rigorous evaluation, and a
proper replay-blend experiment (mixing instruction data back into the C#
corpus) would still be the more direct way to test and improve retention
further, if that's wanted later.

### Loss landscape and decoding-quality notes

Not yet re-run against the rebuilt checkpoint — `plot_loss_landscape.py
--checkpoint checkpoints/model-c/best.pt --model c` and the
`no_repeat_ngram_size`/`top_p` decoding additions (2026-09-14) both still
apply unchanged; see the original run's entries below for what those
outputs looked like on the corrupted data, for comparison once re-run.

---

## Results — ORIGINAL RUN (2026-09-14, corrupted C# data — superseded, kept for the record)

### val_loss trend (logged every 500 steps, from TensorBoard)

| Step | val_loss | | Step | val_loss |
|---|---|---|---|---|
| 500 | 2.0090 | | 10500 | 1.5964 |
| 1000 | 1.8297 | | 11000 | 1.5746 |
| 1500 | 1.8178 | | 11500 | 1.6120 |
| 2000 | 1.8161 | | 12000 | 1.5812 |
| 2500 | 1.7537 | | 12500 | 1.5974 |
| 3000 | 1.7314 | | 13000 | 1.5744 |
| 3500 | 1.7636 | | 13500 | 1.5247 |
| 4000 | 1.7101 | | 14000 | 1.5971 |
| 4500 | 1.7108 | | 14500 | 1.5645 |
| 5000 | 1.6867 | | 15000 | 1.6019 |
| 5500 | 1.6392 | | 15500 | 1.6042 |
| 6000 | 1.6604 | | 16000 | 1.5548 |
| 6500 | 1.6427 | | **16500** | **1.4911 (best)** |
| 7000 | 1.6439 | | 17000 | 1.5956 |
| 7500 | 1.6358 | | 17500 | 1.6242 |
| 8000 | 1.6179 | | 18000 | 1.6070 |
| 8500 | 1.6651 | | 18500 | 1.5933 |
| 9000 | 1.6268 | | 19000 | 1.5862 |
| 9500 | 1.5856 | | 19500 | 1.5888 |
| 10000 | 1.6004 | | | |

**Best val_loss: 1.4911 at step 16,500** (`checkpoints/model-c/best.pt`).
**Final val_loss: 1.5888 at step 19,500** (the last step-20,000-labeled
checkpoint's own step never gets a real forward/backward pass — `train.py`'s
loop runs `range(start_step, max_steps)`, so step 20,000 itself is only ever
used as the *label* on the final saved checkpoint, not an evaluated step;
19,500 is the last real logged value).

**Shape of the curve:** fast drop from 2.01 → ~1.60 through step ~9,500, then
a **plateau** — noisy oscillation roughly between 1.52 and 1.66 for the
remaining ~10,500 steps, no further systematic improvement. This is a milder
version of what happened to Model A-Instruct (which plateaued then trended
clearly *upward*, 1.6486 → ~1.82) — Model C's late-run noise stays close to
its best value (delta ≈0.10 vs. A-Instruct's ≈0.17) rather than climbing
away from it. Conclusion: **use `best.pt` (step 16,500), not the final
checkpoint** — same rule as Model A-Instruct — and the plateau starting
around step 9,500 is evidence the 20,000-step budget was already sufficient;
there's no sign more steps would have helped further.

### Loss landscape (`plot_loss_landscape.py --checkpoint checkpoints/model-c/best.pt --model c`)

Center loss at the checkpoint: **1.496** (close to the TensorBoard-logged
1.4911 — same ballpark, small difference expected since the landscape script
recomputes loss on 5 fresh random val batches rather than reusing the
training run's own 20-batch estimate). The plot shows a smooth, symmetric,
fairly **wide** basin around the checkpoint — no sharp spike, no ridge, no
secondary minima in the ±1.0 span of either random direction. Per Li et al.'s
framing, a wide/flat minimum is generally associated with better
generalization than a narrow sharp one; combined with the mild (not runaway)
late-training noise above, this checkpoint looks like a well-converged,
reasonably stable fine-tune.

### Decoding-quality follow-up

Same repetition-loop degeneration risk as every other model in this project
(root cause: no repetition control in the base decoding loop). Mitigations
added 2026-09-14 across `inference/generate.py`, `api/main.py`, and
`evaluation/common.py`: `no_repeat_ngram_size` (hard-blocks exact n-gram
repeats) and `top_p` nucleus sampling, on top of the existing
`repetition_penalty`. Recommended starting point for testing Model C on C#
prompts: `--temperature 0.7 --top_k 40 --top_p 0.9 --repetition_penalty 1.3
--no_repeat_ngram_size 3`.

### Sample generations — code completion works, instruction-following largely forgotten

Tested `checkpoints/model-c/best.pt` on two prompt styles (2026-09-14):

**1. Raw code-completion (no template)** — the thing this fine-tune stage
actually trained on. Works as expected, multiple runs:
```
"public class LoanService" -> ": IFilesServices"
"public class LoanService" -> ": BaseLoansService"
"public class LoanService" -> ": IEnumerable"
"public class LoanService" -> ": ILoaderLoadService"
```
Short completions (model hits EOS quickly), but on-topic and syntactically
plausible C# member/inheritance declarations.

**2. `### Instruction: Write a C# for loop ### Response:`** (the template
Model A-Instruct was trained on) — compared directly against Model
A-Instruct's own checkpoint on the identical prompt:

| Model | Output |
|---|---|
| Model A-Instruct (`checkpoints/model-a-instruct/best.pt`) | `"...### Response: Here is a C# for loop that accomplishes the same task: \`\`\`csharp using System; class Program { static void main(String[] args) { int n = 1; while (n <= null \|\| n<= n - 1) { char *= 2..."` — engages properly, on-topic (if syntactically imperfect) attempt |
| Model C (`checkpoints/model-c/best.pt`) | 2 of 3 runs: predicts EOS immediately, zero generated tokens. 1 of 3 runs: `"################";"`  — neither a response nor anything template-shaped |

**Conclusion: catastrophic forgetting of the instruction-following template.**
Model C's fine-tune stage trained for 20,000 steps (~19.3% coverage) on
C#-only source code — zero `### Instruction:`-formatted examples — so
nothing in that stage reinforced the behavior Model A-Instruct had learned.
That was enough to substantially overwrite it: the model now mostly treats
`### Response:` as an end-of-sequence signal rather than a cue to continue.
Meanwhile the code-completion capability (the thing this stage *was*
trained for) transferred and works fine.

This is a genuine, meaningful result for the three-model comparison design —
the Model A-Instruct vs. Model C leg was specifically meant to isolate
"fine-tuning specialization vs. forgetting relative to the starting
checkpoint," and this is a direct, measured instance of that forgetting,
not a bug or decoding artifact (verified by re-running the same prompt
multiple times with and without the new `top_p`/`no_repeat_ngram_size`
decoding options — the empty-output behavior is consistent across both,
confirming it's the model's learned behavior, not a sampling fluke).
