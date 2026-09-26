import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import sentencepiece as spm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.config import ModelConfig
from model.model import GPTModel

CONFIG_PATH = "configs/model_117M.yaml"
TOKENIZER_MODEL = "tokenizer/tokenizer.model"
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", "checkpoints/model-a/best.pt")


def _model_label_from_checkpoint(path):
    """Derives a label like 'model-c' from the checkpoint path actually loaded,
    instead of a hardcoded string — the response's `model` field previously
    always said 'track1-slm-model-a' no matter which checkpoint MODEL_CHECKPOINT
    pointed to, which silently made it impossible to tell from the API response
    alone which model actually served a given request."""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("model-"):
            return f"track1-slm-{part}"
    return f"track1-slm-{path}"


MODEL_LABEL = _model_label_from_checkpoint(CHECKPOINT_PATH)

state = {"model": None, "tokenizer": None, "config": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig.from_yaml(CONFIG_PATH)
    state["device"] = device
    state["config"] = config
    state["tokenizer"] = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)

    if os.path.exists(CHECKPOINT_PATH):
        model = GPTModel(config).to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        state["model"] = model
        print(f"Loaded model from {CHECKPOINT_PATH} on {device}")
    else:
        state["model"] = None
        print(
            f"WARNING: checkpoint not found at {CHECKPOINT_PATH} — "
            "/generate will return 503 until a model is trained."
        )

    yield
    state.clear()


app = FastAPI(title="Track 1 SLM API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    model: str


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
def run_generation(prompt: str, max_tokens: int, temperature: float, top_k: int, repetition_penalty: float = 1.2,
                    no_repeat_ngram_size: int = 0, top_p: float = 1.0):
    """repetition_penalty: CTRL-style (Keskar et al. 2019, same formula HuggingFace's
    generate() uses) — divides (or multiplies, if negative) the logit of every token
    already present in the sequence so far by this value, pushing it toward zero so
    it's less likely to be picked again. 1.0 = off (original behavior).

    no_repeat_ngram_size (0=off) and top_p (1.0=off) — see inference/generate.py's
    identical helper functions for the full rationale. Decoding order:
    repetition_penalty -> no_repeat_ngram_size ban -> temperature -> top_k -> top_p
    -> softmax -> sample."""
    model = state["model"]
    sp = state["tokenizer"]
    config = state["config"]
    device = state["device"]

    ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    start_len = input_ids.shape[1]

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

        if top_k and top_k > 0:
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

    new_ids = input_ids[0, start_len:].tolist()
    return sp.decode(new_ids), len(new_ids)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded — no checkpoint available yet")

    text, n_tokens = run_generation(req.prompt, req.max_tokens, req.temperature, req.top_k, req.repetition_penalty,
                                     req.no_repeat_ngram_size, req.top_p)
    return GenerateResponse(response=text, tokens_generated=n_tokens, model=MODEL_LABEL)
