import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.config import ModelConfig
from model.model import GPTModel

CONFIG_PATH = "configs/model_117M.yaml"
TOKENIZER_MODEL = "tokenizer/tokenizer.model"


def load_model(checkpoint_path, device):
    config = ModelConfig.from_yaml(CONFIG_PATH)
    model = GPTModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def _ban_repeated_ngrams(logits, input_ids, ngram_size):
    """Hard-blocks any token that would complete an n-gram already seen earlier in
    the sequence (prompt + generated) — standard HF-style no_repeat_ngram_size.
    Distinct from repetition_penalty: the penalty makes a repeated token less
    likely but still possible; this makes an exact n-gram repeat impossible.
    They fix different failure modes (a token that keeps winning despite the
    penalty vs. a whole phrase looping) and are meant to be used together.
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
    """Nucleus sampling: keeps the smallest set of highest-probability tokens whose
    cumulative probability exceeds top_p, masks the rest. Unlike top_k's fixed
    candidate count, this adapts to how peaked the distribution is — a very
    confident next-token distribution keeps very few candidates, a flat/uncertain
    one keeps more. 1.0 = off. Applied after top_k so the two can stack (top_k as
    a hard ceiling, top_p trimming further within it) or top_p can be used alone
    by passing top_k=0."""
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    sorted_mask = cum_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False  # always keep at least the top token

    mask = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
    logits[mask] = float("-inf")
    return logits


@torch.no_grad()
def generate(model, sp, prompt, max_tokens, temperature, top_k, device, context_length, repetition_penalty=1.2,
             no_repeat_ngram_size=0, top_p=1.0):
    """Greedy decoding when temperature <= 0, otherwise top-k/top-p + temperature sampling.

    repetition_penalty (CTRL-style, Keskar et al. 2019 — same formula HuggingFace's
    `generate()` uses): for every token already present anywhere in the sequence so
    far (prompt + generated), its logit is divided by `repetition_penalty` if positive
    or multiplied if negative — either way pushing it toward zero, making that token
    less likely to be picked again. 1.0 = no effect (fully backward compatible);
    >1.0 penalizes repeats.

    Decoding order: repetition_penalty -> no_repeat_ngram_size ban -> temperature ->
    top_k -> top_p -> softmax -> sample. Penalize/ban first (on raw logits), then
    reshape the distribution for sampling.
    """
    ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        idx_cond = input_ids[:, -context_length:]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                         help="CTRL-style repetition penalty. 1.0 = off (original behavior). "
                              ">1.0 penalizes tokens already generated; 1.2 is a common default.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                         help="Hard-blocks repeating any n-gram of this size (e.g. 3). 0 = off.")
    parser.add_argument("--top_p", type=float, default=1.0,
                         help="Nucleus sampling threshold. 1.0 = off. Stacks with --top_k.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    model, config = load_model(args.checkpoint, device)

    output = generate(
        model, sp, args.prompt, args.max_tokens, args.temperature, args.top_k,
        device, config.context_length, args.repetition_penalty,
        args.no_repeat_ngram_size, args.top_p,
    )

    print(f"Prompt: {args.prompt!r}")
    print(f"Generated: {output!r}")


if __name__ == "__main__":
    main()
