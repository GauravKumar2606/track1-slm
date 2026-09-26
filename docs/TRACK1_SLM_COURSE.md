# Track 1 SLM — Course Guide
### (Structured like Andrew Ng's Machine Learning Specialization — 3 Courses, each with weekly modules)

This document explains the Track 1 SLM project as a self-contained "course,"
using the same course → week → learning-objectives → key-concepts → lab
structure as the DeepLearning.AI / Andrew Ng ML Specialization. It does not
change any existing code — it's a teaching map onto what already exists in
this repo (see `CLAUDE.md` for the authoritative technical spec).

Where the ML Specialization teaches classical/foundational ML (regression,
plain neural nets, clustering, recommenders, RL), this course covers the next
layer up: building a decoder-only Transformer language model from scratch,
training it under compute constraints, and comparing model variants — the
same way the Specialization builds intuition course-by-course, but for LLMs
instead of tabular ML.

---

## Course 1 — Data & Representation
*(Equivalent role to Specialization Course 1 "Supervised Machine Learning":
get the input representation right before touching any model.)*

### Week 1 — Why Language Modeling Is a Different Problem
**Learning objectives**
- Explain why next-token prediction over free-form text can't be framed as a
  fixed-feature-vector supervised problem the way house-price regression can.
- Explain what "context" and "sequence order" mean, and why a plain Dense
  network can't represent them.

**Key concepts**
- Autoregressive language modeling: predict token *t+1* from tokens *1..t*
- Variable-length input vs. fixed-size feature vectors

**Code — `get_batch()` from `training/train.py` (actual repo code, in full)**
```python
def get_batch(data, batch_size, context_length, device):
    max_start = len(data) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)
    x = np.stack([data[s: s + context_length].astype(np.int64) for s in starts])
    y = np.stack([data[s + 1: s + 1 + context_length].astype(np.int64) for s in starts])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y
```

**Line-by-line explanation**
- `data` here is a memory-mapped `uint16` array of ~12.02 billion token IDs
  (`data/tokenized/model-a/train.bin`) — one giant flat stream of text, not
  a table of separate labeled rows the way the Specialization's house-price
  CSV would be (`m` rows, each with its own `x` and `y`).
- `max_start = len(data) - context_length - 1` — the last valid position a
  512-token window (`context_length`) can start from, leaving room for both
  the window itself and its "one token ahead" shifted counterpart.
- `starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)`
  — picks `batch_size=24` random starting positions anywhere in the ~12.02B
  tokens. `dtype=np.int64` matters: `np.random.randint`'s default (int32) tops
  out at ~2.1 billion, which is smaller than this training set — using the
  default silently crashes or wraps around on a corpus this size (a real bug
  hit and fixed during this project, see `SESSION_NOTES_PHASE1-4_PREP.md`).
- `x = np.stack([data[s : s+context_length] ...])` — for each of the 24
  random starts, slices out a 512-token window. This produces the `x` half
  of the batch: shape `(24, 512)`.
- `y = np.stack([data[s+1 : s+1+context_length] ...])` — slices the *same*
  512-token window, just shifted one position to the right. This is the
  entire "predict token *t+1* from tokens *1..t*" idea from this week's
  learning objective, expressed as code: **`y` is `x` shifted by one
  token.** There's no separate label column anywhere — the label at every
  position is just "whatever token comes next in the same text."
- Contrast with the Specialization's typical batch: one `x` vector (e.g.
  house features: sqft, bedrooms, age) paired with **one single** `y` value
  (price) per example. Here, one `x` of 512 tokens is paired with **512
  separate targets** (`y`), one per position — every position in the window
  is simultaneously an input (for predicting what comes after it) and part
  of the context (for predicting positions before it).

**Lab (maps to repo)**
- Read `CLAUDE.md`'s "Three-Model Comparison Design" section for the actual
  problem framing used in this project.

### Week 2 — Data Collection & Cleaning
**Learning objectives**
- Understand large-corpus collection (wikitext, pg19, openwebtext, c4, C#,
  instruction data) and why raw text needs cleaning before use.
- Understand deduplication at scale (why an in-memory hash-set beats a
  disk-backed SQLite store for tens of millions of small lookups).

**Key concepts**
- HTML/URL stripping, short-line filtering, non-English filtering
- Streaming vs. in-memory processing (the >10GB threshold rule)
- BLAKE2b hash-set dedup — O(1) membership check, no disk round-trips

**Code — `filter_and_clean_paragraphs()` + the dedup loop from `data/clean_text.py` (actual repo code)**
```python
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http\S+')
ASCII_LETTER_RE = re.compile(r'[a-zA-Z]')

def is_english(text):
    return len(text) > 10 and bool(ASCII_LETTER_RE.search(text))

def clean_text_segment(text):
    text = HTML_TAG_RE.sub('', text)
    text = URL_RE.sub('', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.translate(CONTROL_CHAR_TABLE).strip()
    return text

def filter_and_clean_paragraphs(chunk_text):
    cleaned_paragraphs = []
    for paragraph in chunk_text.split('\n'):
        paragraph = paragraph.strip()
        if len(paragraph) < 20:
            continue
        cleaned = clean_text_segment(paragraph)
        if not cleaned:
            continue
        if not is_english(cleaned):
            continue
        cleaned_paragraphs.append(cleaned)
    return cleaned_paragraphs

# --- the in-memory-hash-set dedup loop (from process_in_memory_text) ---
seen_hashes = set()
for paragraph in all_paragraphs:
    ...
    h = hashlib.blake2b(cleaned.encode('utf-8'), digest_size=8).digest()
    if h in seen_hashes:
        continue
    cleaned_paragraphs.append(cleaned)
    seen_hashes.add(h)
```

**Line-by-line explanation**
- `HTML_TAG_RE = re.compile(r'<[^>]+>')` — matches anything between `<` and
  `>` (e.g. `<div>`, `<br/>`) so it can be stripped; compiled once at module
  load instead of re-parsing the same regex pattern on every one of the
  hundreds of millions of lines processed.
- `URL_RE = re.compile(r'http\S+')` — matches a URL and everything
  non-whitespace after it (`\S+`), so links get removed as noise tokens the
  language model shouldn't waste vocabulary capacity learning to predict.
- `is_english(text)`: `len(text) > 10 and ASCII_LETTER_RE.search(text)` — a
  cheap heuristic, not a real language-detection model: a line must be
  longer than 10 characters *and* contain at least one plain a-z/A-Z letter
  to be treated as English prose. Fast enough to run on hundreds of
  millions of lines, at the cost of being approximate (it would also accept
  a line that's mostly non-English but has one stray English word).
- `clean_text_segment`: runs three passes in order — strip HTML tags, strip
  URLs, then `text.encode('ascii', 'ignore').decode('ascii')` drops every
  character that isn't plain ASCII (this is what actually removes
  non-English scripts, emoji, and other Unicode noise — cheaper than a
  proper language-detection model). `.translate(CONTROL_CHAR_TABLE)` then
  strips invisible control characters (a stray NUL byte was found to
  silently truncate SentencePiece's tokenizer trainer at that exact byte in
  an earlier run — see `SESSION_NOTES_PHASE1-4_PREP.md`).
- `if len(paragraph) < 20: continue` — drops very short lines (menu items,
  bare punctuation, table fragments) that add noise without adding
  learnable language structure.
- **The dedup loop** — `hashlib.blake2b(cleaned.encode('utf-8'), digest_size=8).digest()`
  computes an 8-byte (64-bit) hash of the cleaned paragraph's exact text.
  `seen_hashes` is a plain Python `set()` living entirely in RAM: checking
  `if h in seen_hashes` is an O(1) hash-table lookup, and `seen_hashes.add(h)`
  records it going forward — so if the exact same paragraph appears twice
  anywhere in the corpus (a common occurrence with web-scraped text), only
  the first occurrence survives into the cleaned output. This is the exact
  mechanism referenced in Week 2's key concepts as the in-memory hash-set
  approach that replaced an earlier, much slower SQLite-based version (SQLite's
  disk-backed WAL checkpointing couldn't keep up with tens of millions of
  small inserts — stretching the same cleaning job from ~15 minutes to
  multiple days).

**Lab (maps to repo)**
- `data/download_datasets.py`, `data/clean_text.py`
- CLAUDE.md → "CORE DATA PIPELINE GUARDRAIL" section

### Week 3 — Tokenization
**Learning objectives**
- Understand subword tokenization (BPE) and why LLMs don't use word-level or
  character-level vocabularies.
- Understand the train/val/test split for a text corpus (90/5/5) and why it's
  done directly from cleaned sources rather than via one combined file.

**Key concepts**
- Byte-Pair Encoding, vocab size (32,000), encode/decode roundtrip testing
- Fixed-length example packing (for the instruction-tuning fixed-array format)
  vs. flat-stream packing (for pretraining)

**Worked example — Model A's actual numbers**
The Specialization always grounds a concept in a real dataset count (e.g.
"m = 47 training examples"). Here's the equivalent for Model A's corpus,
taken from `MODEL_A_TRAINING_NOTES.md`:

| Split | Size on disk | Token count |
|---|---|---|
| train | 54.09 GB | 12,016,279,511 (~12.02B) |
| val | 3.07 GB | 727,573,115 (~727.6M) |
| test | 3.06 GB | 712,448,359 (~712.4M) |
| **Total** | **60.2 GB** | **~13.46B tokens** |

That's the "m" (dataset size) of this project — except instead of 47 rows of
house-price data, it's ~13.46 billion tokens. The 90/5/5 split ratio is the
same idea the Specialization uses for train/cv/test, just applied to raw
text volume instead of row count: 12.02B / 13.46B ≈ 89.3% train,
727.6M / 13.46B ≈ 5.4% val, 712.4M / 13.46B ≈ 5.3% test.

**Lab (maps to repo)**
- `tokenizer/train_tokenizer.py`, `tokenizer/verify_tokenizer.py`,
  `tokenizer/tokenize_corpus.py`, `data/prepare_splits.py`

---

## Course 2 — Transformer Architecture
*(Equivalent role to Specialization Course 2 "Advanced Learning Algorithms":
this is where the actual model gets built, layer by layer.)*

### Week 1 — Self-Attention
**Learning objectives**
- Explain how self-attention lets every token look at every earlier token,
  weighted by learned relevance — the mechanism a Dense network has no
  equivalent for.
- Explain Q/K/V projections, scaled dot-product attention, and the causal
  mask (why token *t* must never see token *t+1..T*).

**Key concepts**
- Single fused QKV linear layer, multi-head split, `scaled_dot_product_attention`
- `is_causal=True` masking (lower-triangular, no future leakage)
- Attention dropout

**Worked example — tiny numbers, by hand**
The Specialization always walks a small numeric example before scaling up
(e.g. one house with 850 sqft). Here's the attention mechanism the same way,
with a toy 4-token sentence and a tiny 2-dimensional embedding
(`d_model=2`, `n_heads=1` — the real model uses `d_model=768`, `n_heads=12`,
so each real head only sees `768/12 = 64` dimensions):

- Input sentence (already tokenized): `["The", "cat", "sat", "down"]`, T = 4
- Each token's embedding is a 2-number vector, e.g. `The → [0.1, 0.2]`
- Q, K, V are each computed as `embedding @ weight_matrix` (a 2x2 matrix here,
  768x768 in the real model) — three separate weighted views of the same input
- Attention score between token *i* and token *j* is `Q_i · K_j / sqrt(head_dim)`
  — for our toy example `head_dim = 2`, so we'd divide by `sqrt(2) ≈ 1.41`;
  in the real model `head_dim = 64`, so it divides by `sqrt(64) = 8`
- **Causal mask in action**: token 3 ("sat") is allowed to compute scores
  against tokens 1-3 ("The", "cat", "sat") but token 4's ("down") score is
  forced to `-inf` before the softmax — so "sat" can never be influenced by
  a word that comes after it. This is the literal mechanism that makes the
  model autoregressive (can generate one token at a time, left to right).
- After softmax, the scores become attention *weights* that sum to 1 across
  the allowed positions — e.g. "sat" might end up attending 70% to itself,
  20% to "cat", 10% to "The", 0% to "down" (masked out)
- The output for "sat" is then a weighted average of the V vectors for
  "The", "cat", "sat" using those weights — this is the number that actually
  flows forward into the rest of the network

Scaling this up: the real model runs this same computation **12 times in
parallel per layer** (`n_heads=12`, each with `head_dim=64`), and repeats the
whole block **12 times** (`n_layers=12`), for every one of the up to 512
tokens (`context_length=512`) in a training example.

**Code — `model/attention.py` (actual repo code, in full)**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
```

**Line-by-line explanation**
- `assert d_model % n_heads == 0` — with `d_model=768, n_heads=12`, this
  guarantees `768/12 = 64` divides evenly; each head gets exactly 64
  dimensions to work with (`self.head_dim`).
- `self.qkv_proj = nn.Linear(d_model, 3 * d_model)` — **one** linear layer
  produces Q, K, and V all at once (`768 → 2304`), instead of three separate
  layers. Cheaper to run as a single matrix multiply on the GPU; split apart
  immediately after.
- `qkv = self.qkv_proj(x)` then `q, k, v = qkv.split(C, dim=-1)` — this is
  the split: the 2304-wide output is chopped back into three 768-wide
  tensors along the last dimension.
- `q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)` — reshapes
  each of Q/K/V from "one 768-wide vector per token" into "12 separate
  64-wide vectors per token," then moves the head dimension before the
  sequence dimension so each head can be processed independently — shape
  becomes `(batch, n_heads, seq_len, head_dim)`.
- `F.scaled_dot_product_attention(q, k, v, ..., is_causal=True)` — this
  single call does the entire attention computation from the Week 1 worked
  example (Q·K score, divide by `sqrt(head_dim)`, mask future positions,
  softmax, weighted sum with V) using PyTorch's fused/flash-attention
  kernel, instead of writing those steps out by hand. `is_causal=True` is
  what applies the lower-triangular mask automatically.
- `out.transpose(1, 2).contiguous().view(B, T, C)` — undoes the earlier
  reshape: merges the 12 heads' 64-dim outputs back into one 768-dim vector
  per token.
- `self.out_proj(out)` — one final linear layer (768→768) mixes information
  across the 12 heads' outputs before this result is added back into the
  residual stream in `transformer.py`.

**Lab (maps to repo)**
- `model/attention.py`

### Week 2 — The Transformer Block
**Learning objectives**
- Explain pre-norm architecture (LayerNorm *before* attention/feed-forward,
  not after) and why it stabilizes training at depth.
- Explain the feed-forward sublayer (Linear → GELU → Linear) and residual
  connections around both sublayers.

**Key concepts**
- Pre-norm vs. post-norm
- Residual/skip connections
- GELU activation (vs. ReLU, which the Specialization uses)

**Code — `model/transformer.py` (actual repo code, in full)**
```python
import torch.nn as nn

from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, context_length: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

**Line-by-line explanation**
- `self.ln1 = nn.LayerNorm(d_model)` / `self.ln2 = nn.LayerNorm(d_model)` —
  two separate normalization layers, one before attention, one before the
  feed-forward network. Normalizing means rescaling each token's 768-number
  vector to have mean 0, variance 1 before it's used — keeps the numbers in
  a stable range as they pass through 12 stacked blocks.
- `self.mlp = nn.Sequential(Linear(768→3072), GELU, Linear(3072→768), Dropout)`
  — the feed-forward sublayer. It expands each token's vector from 768 to
  3072 dimensions (`d_ff=3072`, 4x wider — a very common transformer ratio),
  applies the GELU non-linearity, then projects back down to 768 so it can
  be added back to the residual stream.
- `x = x + self.attn(self.ln1(x))` — **this is pre-norm plus residual**, the
  two ideas from this week combined in one line: normalize `x` first
  (`self.ln1(x)`), run it through attention, then **add the result back to
  the original, un-normalized `x`** (the `x + ...`). The addition is the
  residual/skip connection — it lets gradients flow directly through the
  `+` during backpropagation without having to pass through the attention
  layer's math, which is what makes 12 stacked blocks trainable instead of
  suffering vanishing gradients.
- `x = x + self.mlp(self.ln2(x))` — the same pre-norm + residual pattern,
  applied to the feed-forward sublayer instead of attention. By the end of
  this line, `x` has been updated by both sublayers and is ready to enter
  the next `TransformerBlock` (or, if this was the last of the 12, the
  final `ln_f` in `model.py`).

**Lab (maps to repo)**
- `model/transformer.py`

### Week 3 — Assembling the Full Model
**Learning objectives**
- Explain how token embeddings + learned positional embeddings + N stacked
  blocks + final LayerNorm + output projection form a complete GPT-style model.
- Explain weight tying between the input embedding and output projection, and
  why it saves parameters without hurting quality.

**Key concepts**
- Token embedding table, learned positional embedding table
- Weight tying (`head.weight = token_emb.weight`)
- Parameter counting, forward-pass shape verification

**Worked example — counting all 110,025,216 parameters by hand**
The Specialization always shows the arithmetic behind a number (e.g. "this
network has 25×15 + 15 weights in this layer"). Here's the full breakdown for
Model A's actual config (`vocab_size=32000, context_length=512, d_model=768,
n_heads=12, d_ff=3072, n_layers=12`), confirmed against
`MODEL_A_TRAINING_NOTES.md`:

| Component | Formula | Count |
|---|---|---|
| Token embedding | `vocab_size × d_model` = 32,000 × 768 | 24,576,000 |
| Positional embedding | `context_length × d_model` = 512 × 768 | 393,216 |
| Per-block: QKV projection | `d_model × 3d_model + 3d_model` (weight + bias) | 1,771,776 |
| Per-block: output projection | `d_model × d_model + d_model` | 590,592 |
| Per-block: 2 LayerNorms | `2 × 2 × d_model` | 3,072 |
| Per-block: feed-forward (Linear→GELU→Linear) | `(d_model×d_ff+d_ff) + (d_ff×d_model+d_model)` | 4,722,432 |
| **Total per block** | sum of the four rows above | **7,087,872** |
| **All 12 blocks** | 7,087,872 × 12 | **85,054,464** |
| Final LayerNorm | `2 × d_model` | 1,536 |
| Output head | tied to token embedding — **0 extra params** | 0 |
| **Grand total** | 24,576,000 + 393,216 + 85,054,464 + 1,536 | **110,025,216 (~110M)** |

This matches the real, printed value from `verify_model.py` exactly — same
role as the Specialization asking you to verify a layer's weight-matrix shape
by hand before trusting the code. Note the "0 extra params" for the output
head: without weight tying it would cost another 24,576,000 params
(`d_model × vocab_size`), pushing the model to ~134.6M — this is the concrete
number behind the "weight tying saves parameters" claim.

**Code — `model/model.py` (actual repo code, in full)**
```python
import torch
import torch.nn as nn

from .transformer import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.context_length, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between input embedding and output projection.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.context_length
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

**Line-by-line explanation**
- `self.token_emb = nn.Embedding(config.vocab_size, config.d_model)` — a
  32,000 x 768 lookup table. Given a token ID (an integer 0-31,999), this
  returns that token's learned 768-number vector — the model's internal
  representation of that subword.
- `self.pos_emb = nn.Embedding(config.context_length, config.d_model)` — a
  second, separate 512 x 768 lookup table, one learned vector per *position*
  (0 through 511) in the sequence, regardless of which token sits there.
  This is what tells the model "this token is 3rd in the sequence" as
  distinct from "this token is 300th."
- `self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(config.n_layers)])`
  — literally builds a list of 12 `TransformerBlock` instances (Week 2),
  each with its own independent weights, stacked to be run one after another.
- `self.head.weight = self.token_emb.weight` — **the weight-tying line**
  from the parameter-count worked example above. After this line, the
  output projection and the input embedding table are the exact same tensor
  in memory — updating one during training updates the other.
- `x = self.token_emb(idx) + self.pos_emb(pos)` — this is where the two
  embedding tables from above are combined: each token's meaning-vector is
  added element-wise to its position-vector, so the same word gets a
  slightly different representation depending on where it appears.
- `for block in self.blocks: x = block(x)` — runs the input through all 12
  transformer blocks in sequence, each one updating `x` via its own
  attention + feed-forward + residual computation (Week 2).
- `x = self.ln_f(x)` then `logits = self.head(x)` — one final normalization,
  then the (weight-tied) linear projection from 768 dimensions back up to
  32,000 — one raw score per vocabulary token, for every position in the
  sequence. Shape: `(batch, seq_len, 32000)` — this is the `(2, 64, 32000)`
  shape `verify_model.py` checks for its `batch=2, seq=64` dummy input.

**Lab (maps to repo)**
- `model/model.py`, `model/config.py`, `model/verify_model.py`

### Week 4 — Diagnosing the Model (the Specialization's "bias/variance" analog)
**Learning objectives**
- Translate the Specialization's train/val/test bias-variance framework into
  language-model terms: train loss vs. validation loss vs. perplexity.
- Understand what overfitting looks like for a language model (memorizing
  training text vs. generalizing).

**Key concepts**
- Validation loss logged every 500 steps
- Perplexity = exp(cross-entropy loss) — the LM equivalent of an accuracy metric

**Worked example — Model A's real convergence numbers**
The Specialization shows a learning curve (train error vs. cv error vs.
iterations) to teach "is this overfitting, underfitting, or converged?" Here
is the actual data from Model A's run (`MODEL_A_TRAINING_NOTES.md`):

| Checkpoint | Step | Validation loss |
|---|---|---|
| `best.pt` | 128,500 | 3.5159 |
| final | 133,000 | 3.5373 |

Val loss over the last ~20,000 steps stayed flat/noisy between 3.51 and 3.57
— it stopped improving well before the run ended, the same pattern the
Specialization calls "converged, diminishing returns from more training" on
a learning-curve plot, just read off two checkpoint numbers instead of a
graph.

Converting to perplexity (`perplexity = e^(val_loss)`):
- `e^3.5159 ≈ 33.65` — at the best checkpoint, the model was, on average,
  as "confused" about the next token as if it had to guess uniformly among
  ~33.65 equally-likely options (out of the full 32,000-token vocabulary).
- `e^3.5373 ≈ 34.39` at the final checkpoint — very slightly worse, another
  numeric sign the run had plateaued, consistent with why `best.pt`
  (step 128,500) rather than the final step is what gets used downstream.

**Lab (maps to repo)**
- `training/validate.py`, `evaluation/perplexity.py`

---

## Course 3 — Training, Comparison & Deployment
*(Equivalent role to Specialization Course 3 "Unsupervised Learning,
Recommenders, Reinforcement Learning": applying the model to real,
larger-scale scenarios and evaluating trade-offs.)*

### Week 1 — Optimization at Scale
**Learning objectives**
- Understand AdamW, weight decay, gradient clipping, and cosine learning-rate
  schedules with linear warmup — and why these matter more at this scale than
  the plain fixed-LR Adam taught in the Specialization.
- Understand mixed-precision training (fp16/fp32) and its VRAM trade-offs.

**Key concepts**
- AdamW (weight decay 0.1), grad clip max-norm 1.0
- Cosine decay + warmup schedule
- `torch.cuda.amp` mixed precision
- VRAM budgeting (batch size vs. throughput vs. GPU memory ceiling)

**Worked example — why batch_size=24 was chosen, with the real benchmark**
The Specialization teaches you to plot a metric against a hyperparameter
before picking a value. Here's the actual benchmarked table from
`training/benchmark_throughput.py`, run on this project's RTX 5070 Ti
Laptop GPU (12GB VRAM):

| batch_size | peak VRAM | tokens/sec | time for 100,000 steps |
|---|---|---|---|
| 8 | 4.65 GB | 38,814 | 2.9h |
| 16 | 7.71 GB | 41,939 | 5.4h |
| **24 (chosen)** | **10.74 GB** | **42,451** | **8.0h** |
| 32 (original default) | 13.77 GB (!) | 10,234 | 44.5h |

Batch 32 exceeds the real ~12.8GB usable VRAM, so it silently spills into
Windows' slow shared-GPU-memory fallback — throughput *collapses* from
~42,000 tok/s to ~10,234 tok/s, a ~4x slowdown, even though it "worked"
without crashing. This is the concrete number behind "don't just pick the
biggest batch size that doesn't crash" — the same caution the Specialization
gives about learning rate (too high can silently make things worse, not just
error out).

Holding the **token budget** fixed at ~1.64B tokens (`batch_size × context_length
× max_steps` = `24 × 512 × 133,500`), instead of holding step count fixed,
is what makes the batch_size=24/max_steps=133,500 config equivalent in total
training signal to the original batch_size=32/max_steps=100,000 config —
just ~10.7h instead of ~44.5h, because it avoids the memory-overflow penalty.
Model A's ~12.02B available training tokens vs. the ~1.64B token training
budget also means each token is seen on average only ~0.14x — well under
one full pass (one "epoch") over the corpus, since a 110M-parameter model
can't usefully absorb 12B tokens' worth of unique gradient signal in one run
(this project follows the "Chinchilla" heuristic of roughly 20 tokens of
training per model parameter: 110M × 20 ≈ 2.2B, in the same order of
magnitude as the ~1.64B actually used).

**Code — the core training step from `training/train.py` (actual repo code, key excerpt)**
```python
def lr_at_step(step, warmup_steps, max_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (LR_MIN_RATIO + (1 - LR_MIN_RATIO) * cosine)

# ... inside the main training loop, once per step:
lr = lr_at_step(step, config.warmup_steps, config.max_steps, config.learning_rate)
for group in optimizer.param_groups:
    group["lr"] = lr

x, y = get_batch(train_data, config.batch_size, config.context_length, device)

with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**Line-by-line explanation**
- `lr_at_step(...)` — implements the two-phase schedule in one function.
  For `step < warmup_steps` (the first 2,000 steps, per
  `configs/model_117M.yaml`), the LR ramps up **linearly** from ~0 to the
  peak `3e-4` — this is the warmup. After that, `cosine = 0.5*(1+cos(...))`
  produces a curve that starts at 1 and smoothly decays to 0 as `progress`
  goes from 0 to 1 across the remaining steps; the final line rescales that
  curve so it decays from the peak LR down to `LR_MIN_RATIO * peak_lr`
  (10% of peak) instead of all the way to zero — this is the cosine decay.
- `for group in optimizer.param_groups: group["lr"] = lr` — manually pushes
  the freshly computed LR into the optimizer before every single step, since
  PyTorch's plain `AdamW` doesn't have a built-in schedule of its own — this
  loop *is* the schedule being applied.
- `get_batch(...)` — pulls `batch_size=24` random windows of
  `context_length=512` tokens each from the ~12.02B-token memory-mapped
  training file (`x`), plus the same windows shifted one token to the right
  (`y`, the "next token" targets) — this is the concrete mechanism behind
  "predict token *t+1* from tokens *1..t*" from Week 1 of Course 1.
- `torch.autocast(..., dtype=torch.float16, ...)` — runs the forward pass
  (`model(x)`) and loss computation in 16-bit floating point instead of the
  default 32-bit, roughly halving memory use and increasing throughput —
  this is the "mixed precision" from the VRAM benchmark table above; it's
  *mixed* because some internal operations still use fp32 for numerical
  stability.
- `cross_entropy(logits.view(-1, 32000), y.view(-1))` — the loss function:
  flattens the `(batch, seq_len, 32000)` logits and `(batch, seq_len)`
  targets down to one long list of predictions and one long list of correct
  next-token IDs, then computes the standard classification cross-entropy
  over all of them at once. This is the same loss `e^loss = perplexity`
  worked example above is computed from.
- `scaler.scale(loss).backward()` / `scaler.unscale_` / `scaler.step` /
  `scaler.update()` — the `GradScaler` dance required specifically because
  of fp16 mixed precision: it scales the loss up before backprop (so small
  gradients don't underflow to zero in fp16), unscales the gradients back
  down before clipping, then steps the optimizer and adjusts the scale
  factor for next time.
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — measures the
  combined size ("norm") of every parameter's gradient across the whole
  110M-parameter model, and rescales them all proportionally if that
  combined size exceeds `1.0` — this is the safeguard against a single bad
  batch producing a gradient spike large enough to destabilize training.

**Lab (maps to repo)**
- `training/train.py`, `training/benchmark_throughput.py`
- CLAUDE.md → "Training VRAM Budget" section

### Week 2 — Comparative Experimentation
**Learning objectives**
- Understand how to isolate one variable at a time across model variants
  (data composition vs. training procedure vs. fine-tuning), similar in
  spirit to how the Specialization compares algorithm choices on the same
  dataset.
- Understand instruction fine-tuning and loss masking (only response tokens
  count toward loss, not prompt tokens).

**Key concepts**
- Model A-Instruct / Model B / Model C — three ways of incorporating new data
  (fine-tune, joint pretrain, sequential fine-tune)
- Loss masking with `ignore_index=-100`
- Why A-Instruct vs. B is *not* a clean single-variable comparison, while
  B vs. C is

**Lab (maps to repo)**
- `training/train_instruct.py`, `data/prepare_instruct_data.py`
- CLAUDE.md → "Three-Model Comparison Design" section

### Week 3 — Evaluation
**Learning objectives**
- Understand cross-perplexity matrices (scoring every model against every
  test set, not just its own) and what an out-of-distribution score reveals.
- Understand qualitative side-by-side comparison (same prompts, different
  models) as a complement to the quantitative perplexity numbers.

**Key concepts**
- Cross-perplexity matrix
- Side-by-side generation comparison

**Lab (maps to repo)**
- `evaluation/compare_models.py`, `evaluation/perplexity.py`,
  `evaluation/generate_report.py`

### Week 4 — Deployment
**Learning objectives**
- Understand post-training quantization (dynamic int8) and its size/quality
  trade-off.
- Understand serving a model behind an API and a simple UI.

**Key concepts**
- Dynamic int8 quantization
- FastAPI request/response contract
- Gradio UI for interactive demos

**Lab (maps to repo)**
- `deployment/quantize.py`, `deployment/app.py`, `api/main.py`

---

## Quick-Reference: Specialization Concept → Track 1 SLM Equivalent

| Specialization concept | Track 1 SLM equivalent |
|---|---|
| Feature vector `x` | Sequence of token IDs |
| `Dense` layer | Transformer block (attention + feed-forward) |
| ReLU | GELU |
| Fixed learning rate | Cosine schedule with linear warmup |
| Adam | AdamW (adds decoupled weight decay) |
| `model.fit(epochs=...)` | Manual step loop in `training/train.py` (step-based, not epoch-based) |
| Bias/variance diagnosis via train/val curves | Train loss vs. validation loss vs. perplexity |
| Accuracy | Perplexity |
| Train/val/test split | Same concept, same 90/5/5 ratio, applied to raw text corpora |
| Comparing algorithms on one dataset | Comparing Model A-Instruct / B / C on the same eval questions + test sets |

---

*This file is documentation only — it does not modify any existing model,
training, or data-pipeline code. See `CLAUDE.md` for the authoritative
technical specification and phase-by-phase execution plan.*
