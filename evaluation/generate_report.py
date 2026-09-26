import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evaluation.common import FINAL_REPORT_PATH, MODEL_COMPARISON_PATH, PERPLEXITY_RESULTS_PATH

MODEL_LABEL = {
    "a": "Model A (English-only, from scratch, base pretrain)",
    "a-instruct": "Model A-Instruct (English base + instruction fine-tune)",
    "b": "Model B (English+C#+instructions, from scratch)",
    "c": "Model C (fine-tuned from Model A-Instruct on C#-only)",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_perplexity_matrix(perplexity_data):
    if not perplexity_data:
        return "_No perplexity results found — run evaluation/perplexity.py first._\n"

    models = perplexity_data["models"]
    test_sets = perplexity_data["test_sets"]
    matrix = perplexity_data["matrix"]

    lines = []
    header = "| Model \\ Test set | " + " | ".join(f"{t.upper()}" for t in test_sets) + " |"
    sep = "|---|" + "---|" * len(test_sets)
    lines.append(header)
    lines.append(sep)
    for m in models:
        row = [f"**{m.upper()}**"]
        for t in test_sets:
            cell = matrix.get(m, {}).get(t)
            if cell is None:
                row.append("—")
            else:
                marker = " (own)" if m == t else ""
                row.append(f"{cell['perplexity']:.2f}{marker}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_response_table(comparison_data, models_present):
    if not comparison_data:
        return "_No comparison results found — run evaluation/compare_models.py first._\n"

    lines = []
    header = "| Question | " + " | ".join(MODEL_LABEL[m] for m in models_present) + " |"
    sep = "|---|" + "---|" * len(models_present)
    lines.append(header)
    lines.append(sep)
    for row in comparison_data:
        cells = [row["question"].replace("|", "\\|")]
        for m in models_present:
            resp = row.get(f"model_{m}", "—")
            cells.append(str(resp).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_observations(perplexity_data, models_present):
    if not perplexity_data:
        return "_Perplexity results not available yet._\n"

    matrix = perplexity_data["matrix"]
    own = perplexity_data["own_test_set_perplexity"]
    lines = []

    for m in models_present:
        if m in own:
            lines.append(f"- {MODEL_LABEL[m]} scores {own[m]['perplexity']:.2f} perplexity on its own test set.")

    for a, b in itertools.combinations(models_present, 2):
        a_on_b = matrix.get(a, {}).get(b)
        b_on_a = matrix.get(b, {}).get(a)
        if a_on_b and b_on_a and a in own and b in own:
            if a_on_b["perplexity"] > own[b]["perplexity"] * 1.05:
                lines.append(
                    f"- {a.upper()} performs worse on {b.upper()}'s test set ({a_on_b['perplexity']:.2f}) "
                    f"than {b.upper()} does on its own ({own[b]['perplexity']:.2f}) — "
                    f"{b.upper()}'s data is out-of-distribution for {a.upper()}."
                )
            if b_on_a["perplexity"] > own[a]["perplexity"] * 1.05:
                lines.append(
                    f"- {b.upper()} performs worse on {a.upper()}'s test set ({b_on_a['perplexity']:.2f}) "
                    f"than {a.upper()} does on its own ({own[a]['perplexity']:.2f}) — "
                    f"{a.upper()}'s data is out-of-distribution for {b.upper()}."
                )

    if not lines:
        return "_No cross-model deltas available yet._\n"
    return "\n".join(lines) + "\n"


def render_conclusion(models_present):
    if len(models_present) < 2:
        return "_Need at least two trained models to draw a comparison._\n"

    lines = []
    pair_notes = {
        frozenset({"a", "a-instruct"}): (
            "**A vs A-Instruct** — the foundational comparison: identical base pretrain, only the "
            "instruction fine-tuning stage differs. A clean, single-variable comparison — isolates exactly "
            "what instruction-tuning changed (and cost, if anything) relative to the untouched base model."
        ),
        frozenset({"a", "b"}): (
            "**A vs B** — the comparison this project originally started with: same from-scratch training "
            "procedure and token budget, only the data mix differs (English-only vs. English+C#+instructions). "
            "A clean, single-variable comparison — isolates the effect of adding C#+instructions to the "
            "pretraining mix, holding everything else constant."
        ),
        frozenset({"a", "c"}): (
            "**A vs C** — since C's lineage runs through A-Instruct back to A, this comparison spans two "
            "sequential fine-tuning stages (instruction-tuning, then C#) relative to the original base "
            "pretrain. Not single-variable (two stages changed at once), but shows the cumulative drift "
            "from the common ancestor both A-Instruct and C share."
        ),
        frozenset({"a-instruct", "b"}): (
            "**A-Instruct vs B** — NOT a clean single-variable comparison (known, accepted limitation): "
            "B differs from A-Instruct in two ways at once — joint-from-scratch training procedure AND a "
            "different overall data mix — so any difference can't be cleanly attributed to just one cause."
        ),
        frozenset({"a-instruct", "c"}): (
            "**A-Instruct vs C** — since C is initialized from A-Instruct's own weights, this shows what "
            "fine-tuning on C# changed relative to the exact checkpoint it started from (catastrophic "
            "forgetting of instruction-following vs. C# specialization)."
        ),
        frozenset({"b", "c"}): (
            "**B vs C** — a clean, meaningful comparison: both models end up exposed to C#+instructions, "
            "but B learned everything jointly from scratch while C learned English+instructions first, then "
            "C# sequentially via fine-tuning. Differences here isolate *how* it was learned (joint vs. "
            "sequential), independent of the A-Instruct-vs-B caveat above."
        ),
    }
    for a, b in itertools.combinations(sorted(models_present), 2):
        note = pair_notes.get(frozenset({a, b}))
        if note:
            lines.append(f"- {note}")

    if len(models_present) >= 3:
        all_names = " vs ".join(m.upper() for m in models_present)
        clean_pairs = [
            f"{a.upper()} vs {b.upper()}" for a, b in itertools.combinations(sorted(models_present), 2)
            if frozenset({a, b}) in ({frozenset({"a", "a-instruct"}), frozenset({"a", "b"}), frozenset({"b", "c"})})
            and frozenset({a, b}) in pair_notes
        ]
        confounded_pairs = [
            f"{a.upper()} vs {b.upper()}" for a, b in itertools.combinations(sorted(models_present), 2)
            if frozenset({a, b}) in ({frozenset({"a-instruct", "b"}), frozenset({"a", "c"})})
            and frozenset({a, b}) in pair_notes
        ]
        lines.append(
            f"- **{all_names}** — the full picture across every trained model. "
            + (f"Single-variable, directly interpretable: {', '.join(clean_pairs)}. " if clean_pairs else "")
            + (f"Two-variable, interpret with caution: {', '.join(confounded_pairs)}. " if confounded_pairs else "")
        )

    return "\n".join(lines) + "\n"


def main():
    perplexity_data = load_json(PERPLEXITY_RESULTS_PATH)
    comparison_data = load_json(MODEL_COMPARISON_PATH)

    models_present = set()
    if perplexity_data:
        models_present |= set(perplexity_data["models"])
    if comparison_data:
        for row in comparison_data:
            models_present |= {k.split("_", 1)[1] for k in row if k.startswith("model_")}
    models_present = sorted(models_present)

    report = []
    report.append("# Track 1 SLM — Model Comparison Report\n")
    report.append(f"Models evaluated: {', '.join(MODEL_LABEL.get(m, m.upper()) for m in models_present) or '_none_'}\n")

    report.append("## Perplexity Matrix\n")
    report.append("Rows = model that generated the checkpoint being scored; columns = which model's test set it "
                   "was scored against. The diagonal is each model's perplexity on its own held-out data.\n")
    report.append(render_perplexity_matrix(perplexity_data))

    report.append("\n## Side-by-Side Responses\n")
    report.append(render_response_table(comparison_data, models_present))

    report.append("\n## Key Observations\n")
    report.append(render_observations(perplexity_data, models_present))

    report.append("\n## Conclusion\n")
    report.append(render_conclusion(models_present))

    content = "\n".join(report)

    os.makedirs(os.path.dirname(FINAL_REPORT_PATH), exist_ok=True)
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)
    print(f"\n(Saved to {FINAL_REPORT_PATH})")


if __name__ == "__main__":
    main()
