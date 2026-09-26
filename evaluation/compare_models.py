import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from evaluation.common import (
    EVAL_QUESTIONS_PATH,
    MODEL_COMPARISON_PATH,
    available_models,
    generate,
    load_model,
    load_tokenizer,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_found = available_models()
    if not models_found:
        print("No trained checkpoints found under checkpoints/model-{a,a-instruct,b,c}/. "
              "Train at least one model first.", flush=True)
        return

    print(f"Found checkpoints for: {sorted(models_found)}", flush=True)

    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    sp = load_tokenizer()

    loaded = {}
    for letter, ckpt_path in models_found.items():
        print(f"Loading model-{letter} from {ckpt_path} ...", flush=True)
        model, config = load_model(ckpt_path, device)
        loaded[letter] = (model, config)

    results = []
    for question in questions:
        print(f"--- Q: {question!r} ---", flush=True)
        row = {"question": question}
        for letter, (model, config) in loaded.items():
            response = generate(model, sp, question, config, device)
            row[f"model_{letter}"] = response
            print(f"  model_{letter}: {response!r}", flush=True)
        results.append(row)

    os.makedirs(os.path.dirname(MODEL_COMPARISON_PATH), exist_ok=True)
    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} question(s) x {len(loaded)} model(s) to {MODEL_COMPARISON_PATH}", flush=True)


if __name__ == "__main__":
    main()
