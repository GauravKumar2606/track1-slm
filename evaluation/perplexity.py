import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from evaluation.common import (
    MODEL_LETTERS,
    PERPLEXITY_RESULTS_PATH,
    available_models,
    compute_perplexity,
    compute_perplexity_masked,
    load_instruct_test_set,
    load_model,
    load_tokenized_split,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_found = available_models()
    if not models_found:
        print("No trained checkpoints found under checkpoints/model-{a,a-instruct,b,c}/. "
              "Train at least one model first.", flush=True)
        return

    # Cross-perplexity matrix: every available model's checkpoint evaluated
    # against every available model's own test set. The diagonal (model X
    # on test-set X) is the "did this model learn its own data" number;
    # off-diagonal cells show generalization/specialization — e.g. Model A
    # (English-only) on Model C's test-set (C#-only) should be much worse
    # than Model C on its own test-set if fine-tuning actually specialized it.
    #
    # Two test-set formats coexist here (fixed 2026-09-26): B's and C's test
    # sets are flat, unmasked token streams (test.bin); A-Instruct's test set
    # is fixed-length, masked per-example arrays (only response tokens count
    # toward loss — see compute_perplexity_masked's docstring for why that
    # masking convention is applied uniformly regardless of which model is
    # being scored, not just when scoring a-instruct itself).
    flat_test_sets = {letter: split for letter in MODEL_LETTERS
                       if (split := load_tokenized_split(letter, "test")) is not None}
    masked_test_sets = {}
    instruct_test = load_instruct_test_set("a-instruct")
    if instruct_test is not None:
        masked_test_sets["a-instruct"] = instruct_test  # (x_data, y_data, context_length)

    test_letters = sorted(set(flat_test_sets) | set(masked_test_sets))
    if not test_letters:
        print("No tokenized test sets found under data/tokenized/model-{a,a-instruct,b,c}/.", flush=True)
        return

    print(f"Models available: {sorted(models_found)}", flush=True)
    print(f"Test sets available: {test_letters}", flush=True)

    matrix = {}
    own_scores = {}
    for model_letter, ckpt_path in models_found.items():
        print(f"\n--- Loading model-{model_letter} from {ckpt_path} ---", flush=True)
        model, config = load_model(ckpt_path, device)

        matrix[model_letter] = {}
        for test_letter in test_letters:
            if test_letter in masked_test_sets:
                x_data, y_data, _ctx = masked_test_sets[test_letter]
                loss, ppl = compute_perplexity_masked(model, device, x_data, y_data)
            else:
                loss, ppl = compute_perplexity(model, config, device, flat_test_sets[test_letter])
            matrix[model_letter][test_letter] = {"loss": loss, "perplexity": ppl}
            print(f"  model-{model_letter} on test-{test_letter}: "
                  f"loss={loss:.4f} perplexity={ppl:.2f}", flush=True)
            if model_letter == test_letter:
                own_scores[model_letter] = {"loss": loss, "perplexity": ppl}

    output = {
        "models": sorted(models_found),
        "test_sets": test_letters,
        # own_test_set_perplexity: each model scored on its own test split —
        # the simple/original two-model comparison this project started with.
        "own_test_set_perplexity": own_scores,
        # matrix: full cross table, matrix[model][test_set] = {loss, perplexity}.
        "matrix": matrix,
    }

    os.makedirs(os.path.dirname(PERPLEXITY_RESULTS_PATH), exist_ok=True)
    with open(PERPLEXITY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved perplexity matrix to {PERPLEXITY_RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
