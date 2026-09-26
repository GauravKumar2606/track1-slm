import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.config import ModelConfig
from model.model import GPTModel

CONFIG_PATH = "configs/model_117M.yaml"  # architecture is identical across all three models' configs
TOKENIZER_MODEL = "tokenizer/tokenizer.model"

# Comparison REDEFINED 2026-09-13 (see CLAUDE.md's "Three-Model Comparison
# Design"): raw Model A was deliberately dropped from the PRIMARY compared
# legs at that point — replaced by Model A-Instruct (Model A fine-tuned on
# instruction data) as the more informative comparison, since a base model's
# lack of instruction-following made a direct A/B/C comparison less useful.
# Model B is trained from scratch on English+C#+instructions; Model C
# fine-tunes from Model A-Instruct's checkpoint on C# only.
# RE-ADDED 2026-09-26: raw Model A is back in the evaluation as a fourth
# reference point — it's the common ancestor A-Instruct (and therefore C)
# both derive from, and omitting its own standalone numbers meant losing a
# real, already-trained data point. All four models share the same
# architecture and tokenizer, which is what makes cross-model /
# cross-test-set perplexity comparisons meaningful.
# NOTE (fixed 2026-09-26): Model A-Instruct's own tokenized data
# (data/tokenized/model-a-instruct/) uses a different, masked per-example
# format (fixed-length input_ids/labels arrays, see data/prepare_instruct_data.py)
# than the flat token stream load_tokenized_split() below expects — so
# "a-instruct" couldn't be scored as a TEST SET through that path (it was
# still fine as a MODEL — its checkpoint loads and gets scored against
# other test sets). load_instruct_test_set()/compute_perplexity_masked()
# below add that missing path: every model can now be evaluated against
# A-Instruct's held-out test examples too.
MODEL_LETTERS = ["a", "a-instruct", "b", "c"]

EVAL_QUESTIONS_PATH = "evaluation/eval_questions.json"
MODEL_COMPARISON_PATH = "evaluation/model_comparison.json"
PERPLEXITY_RESULTS_PATH = "evaluation/perplexity_results.json"
FINAL_REPORT_PATH = "evaluation/final_report.md"

N_PERPLEXITY_BATCHES = 50  # sequential, non-overlapping windows sampled per (model, test-set) pair
SAMPLE_PROMPT = "The capital of France is"
SAMPLE_MAX_TOKENS = 50
LABEL_IGNORE_INDEX = -100  # must match data/prepare_instruct_data.py / training/train_instruct.py


def checkpoint_dir_for(letter):
    return f"checkpoints/model-{letter}"


def find_best_checkpoint(letter):
    """Prefers best.pt (lowest val loss seen during training); falls back to
    the highest-numbered ckpt_*.pt; returns None if the model hasn't been
    trained yet."""
    ckpt_dir = checkpoint_dir_for(letter)
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        return best_path

    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f[len("ckpt_"):-len(".pt")]))
    return os.path.join(ckpt_dir, candidates[-1])


def available_models():
    """Returns {letter: checkpoint_path} for every model that has a checkpoint on disk."""
    found = {}
    for letter in MODEL_LETTERS:
        path = find_best_checkpoint(letter)
        if path:
            found[letter] = path
    return found


def load_model(checkpoint_path, device):
    config = ModelConfig.from_yaml(CONFIG_PATH)
    model = GPTModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def load_tokenizer():
    return spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)


def _ban_repeated_ngrams(logits, input_ids, ngram_size):
    """See inference/generate.py's identical function for the full rationale —
    hard-blocks completing any n-gram already seen earlier in the sequence.
    0 = off."""
    if ngram_size <= 0:
        return logits
    seq = input_ids[0].tolist()
    if len(seq) < ngram_size - 1:
        return logits
    prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned = set()
    for i in range(len(seq) - ngram_size + 1):
        if tuple(seq[i:i + ngram_size - 1]) == prefix:
            banned.add(seq[i + ngram_size - 1])
    for token_id in banned:
        logits[0, token_id] = float("-inf")
    return logits


def _apply_top_p(logits, top_p):
    """See inference/generate.py's identical function — nucleus sampling.
    1.0 = off."""
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    sorted_mask = cum_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
    logits[mask] = float("-inf")
    return logits


@torch.no_grad()
def generate(model, sp, prompt, config, device, max_tokens=SAMPLE_MAX_TOKENS, temperature=0.8, top_k=50,
             repetition_penalty=1.2, no_repeat_ngram_size=0, top_p=1.0):
    """Greedy decoding when temperature <= 0, otherwise top-k/top-p + temperature sampling.

    repetition_penalty, no_repeat_ngram_size, top_p: same formulas as
    inference/generate.py and api/main.py — kept consistent across all three so
    compare_models.py's qualitative A-Instruct/B/C comparisons use identical
    decoding behavior. 1.0/0/1.0 respectively = off (original behavior).
    """
    ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        idx_cond = input_ids[:, -config.context_length:]
        logits = model(idx_cond)[:, -1, :]

        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        logits = _ban_repeated_ngrams(logits, input_ids, no_repeat_ngram_size)

        logits = logits / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        logits = _apply_top_p(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        if temperature <= 0:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == sp.eos_id():
            break

    return sp.decode(input_ids[0].tolist())


def load_tokenized_split(letter, split):
    path = f"data/tokenized/model-{letter}/{split}.bin"
    if not os.path.exists(path):
        return None
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_instruct_test_set(letter, split="test"):
    """Loads the fixed-length, masked (input_ids, labels) example arrays built
    by data/prepare_instruct_data.py — same format train_instruct.py and
    plot_loss_landscape.py's load_instruct_eval_batches() read. Returns
    (x_data, y_data, context_length) or None if this model has no data in
    that format (i.e. every model except a-instruct)."""
    tokenized_dir = f"data/tokenized/model-{letter}"
    meta_path = os.path.join(tokenized_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = meta["num_examples"][split]
    context_length = meta["context_length"]
    x_path = os.path.join(tokenized_dir, f"{split}_input_ids.bin")
    y_path = os.path.join(tokenized_dir, f"{split}_labels.bin")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None
    x_data = np.memmap(x_path, dtype=np.uint16, mode="r", shape=(n, context_length))
    y_data = np.memmap(y_path, dtype=np.int16, mode="r", shape=(n, context_length))
    return x_data, y_data, context_length


@torch.no_grad()
def compute_perplexity(model, config, device, test_data, n_batches=N_PERPLEXITY_BATCHES):
    """Sequential (non-overlapping, deterministic) windows over the test split,
    rather than validate.py's random sampling — makes cross-model perplexity
    numbers directly comparable run to run."""
    context_length = config.context_length
    max_windows = max(1, (len(test_data) - 1) // context_length)
    n_batches = min(n_batches, max_windows)

    losses = []
    for i in range(n_batches):
        start = i * context_length
        x = torch.from_numpy(test_data[start: start + context_length].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(test_data[start + 1: start + 1 + context_length].astype(np.int64)).unsqueeze(0).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def compute_perplexity_masked(model, device, x_data, y_data, batch_size=64):
    """Perplexity against A-Instruct's masked test set — usable for ANY model
    (not just a-instruct itself): masking is a property of this test set (only
    response tokens are meaningful prediction targets, prompt tokens are input
    context), not of how the model being scored was trained. Applying the same
    ignore_index=-100 mask regardless of which model is being scored keeps the
    comparison apples-to-apples — every model is judged on how well it predicts
    the actual instruction responses, nothing else.

    Evaluates the FULL test set (not a sampled subset like compute_perplexity's
    50-window sample) — at ~50K examples this is cheap, and unlike the huge
    flat streams (billions of tokens, sampling required for speed), exhaustive
    evaluation here is both feasible and more exact. Loss is accumulated
    token-weighted (total loss x valid-token-count, summed, then divided by
    total valid tokens) rather than averaged per-batch, so a smaller final
    batch doesn't skew the result.
    """
    n = x_data.shape[0]
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(x_data[start:end].astype(np.int64)).to(device)
        y = torch.from_numpy(y_data[start:end].astype(np.int64)).to(device)

        valid_tokens = (y != LABEL_IGNORE_INDEX).sum().item()
        if valid_tokens == 0:
            continue

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            ignore_index=LABEL_IGNORE_INDEX, reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)
