import os
import json
import sentencepiece as spm

TRAIN_CORPUS = "data/model-a/train.txt"
TOKENIZER_DIR = "tokenizer"
MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "tokenizer")
VOCAB_SIZE = 32000

# Training SentencePiece over the full ~54GB corpus is neither necessary
# nor practical — BPE vocab quality plateaus well before that much data.
# Sample a large, shuffled subset instead (sentencepiece does this via
# reservoir sampling while streaming, so it never loads the whole file).
INPUT_SENTENCE_SIZE = 10_000_000


def main():
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print(f"Training SentencePiece BPE tokenizer (vocab_size={VOCAB_SIZE}) on {TRAIN_CORPUS} ...", flush=True)
    print(f"Sampling up to {INPUT_SENTENCE_SIZE:,} shuffled lines for training.", flush=True)

    spm.SentencePieceTrainer.train(
        input=TRAIN_CORPUS,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )
    print(f"Saved model: {MODEL_PREFIX}.model", flush=True)
    print(f"Saved native vocab: {MODEL_PREFIX}.vocab", flush=True)

    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    vocab_size = sp.get_piece_size()

    # tokenizer/vocab.txt — plain one-token-per-line, ordered by id.
    vocab_txt_path = os.path.join(TOKENIZER_DIR, "vocab.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            f.write(sp.id_to_piece(i) + "\n")
    print(f"Saved vocab.txt: {vocab_txt_path} ({vocab_size:,} tokens)", flush=True)

    # tokenizer/tokenizer.json — full vocab + metadata dump for interop and
    # inspection. Actual encode/decode always goes through the authoritative
    # tokenizer.model file via sentencepiece, not this JSON.
    tokenizer_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    payload = {
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "model_file": "tokenizer.model",
        "special_tokens": {
            "pad": {"id": sp.pad_id(), "piece": "<pad>"},
            "unk": {"id": sp.unk_id(), "piece": "<unk>"},
            "bos": {"id": sp.bos_id(), "piece": "<s>"},
            "eos": {"id": sp.eos_id(), "piece": "</s>"},
        },
        "vocab": [
            {"id": i, "piece": sp.id_to_piece(i), "score": sp.get_score(i)}
            for i in range(vocab_size)
        ],
    }
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved tokenizer.json: {tokenizer_json_path}", flush=True)


if __name__ == "__main__":
    main()
