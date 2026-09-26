import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
import sentencepiece as spm

from model.config import ModelConfig
from model.model import GPTModel
from inference.generate import generate as generate_tokens

# Updated 2026-09-26 for the three-way comparison deploy — see
# deployment/quantize.py's note. Loads whichever of the three quantized
# models are actually available (run quantize.py first) and generates from
# all of them side-by-side for one prompt, mirroring
# evaluation/compare_models.py's qualitative comparison. Reuses
# inference/generate.py's generate() (repetition_penalty + no_repeat_ngram_size
# + top_p decoding, same as every other entry point in this project) rather
# than reimplementing decoding a fourth time.

CONFIG_PATH = "configs/model_117M.yaml"  # architecture is identical across all three models' configs
TOKENIZER_MODEL = "tokenizer/tokenizer.model"
MODEL_INT8_DIR = "deployment/model_int8"

MODEL_LABELS = {
    "a-instruct": "Model A-Instruct (English + instruction fine-tune)",
    "b": "Model B (English + C# + instructions, joint from-scratch)",
    "c": "Model C (C# fine-tune of Model A-Instruct)",
}

# Decoding settings fixed to the values found to work well during this
# project's own testing (see docs/MODEL_C_TRAINING_NOTES.md's "Decoding-quality
# follow-up") — not exposed as UI sliders, to keep the demo simple.
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3

config = ModelConfig.from_yaml(CONFIG_PATH)
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)

models = {}
for letter in MODEL_LABELS:
    path = os.path.join(MODEL_INT8_DIR, letter, "model_int8.pt")
    if not os.path.exists(path):
        print(f"Skipping {letter}: no quantized model at {path} (run deployment/quantize.py first)")
        continue
    m = GPTModel(config)
    m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    state_dict = torch.load(path, map_location="cpu")
    m.load_state_dict(state_dict)
    m.eval()
    models[letter] = m

print(f"Loaded {len(models)}/3 quantized model(s): {sorted(models) or 'none'}")


def generate_all(prompt, max_tokens, temperature):
    if not prompt or not prompt.strip():
        return ["(enter a prompt)"] * 3

    outputs = []
    for letter in ("a-instruct", "b", "c"):
        if letter not in models:
            outputs.append(f"(model-{letter} not available — run deployment/quantize.py first)")
            continue
        text = generate_tokens(
            models[letter], sp, prompt,
            max_tokens=int(max_tokens), temperature=temperature, top_k=TOP_K,
            device="cpu", context_length=config.context_length,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, top_p=TOP_P,
        )
        outputs.append(text)
    return outputs


with gr.Blocks(title="Track 1 SLM — Model Comparison") as demo:
    gr.Markdown(
        "## Track 1 SLM — Model Comparison\n"
        "Three 117M-parameter GPT-style models, identical architecture and tokenizer, "
        "trained/fine-tuned differently. Enter one prompt to see all three respond side-by-side."
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=3)
        with gr.Column(scale=1):
            max_tokens = gr.Slider(50, 500, value=150, step=1, label="Max tokens")
            temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature")
            generate_btn = gr.Button("Generate all three", variant="primary")

    with gr.Row():
        out_a = gr.Textbox(label=MODEL_LABELS["a-instruct"], lines=10)
        out_b = gr.Textbox(label=MODEL_LABELS["b"], lines=10)
        out_c = gr.Textbox(label=MODEL_LABELS["c"], lines=10)

    gr.Markdown(
        "**Trained on:** RTX 5070 Ti Laptop, 12GB VRAM  \n"
        "**Deployed here:** dynamic int8 quantization, CPU inference  \n"
        f"**Decoding:** top_k={TOP_K}, top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}, "
        f"no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE} (fixed, temperature/max_tokens adjustable above)"
    )

    generate_btn.click(generate_all, inputs=[prompt, max_tokens, temperature], outputs=[out_a, out_b, out_c])


if __name__ == "__main__":
    # share=True tunnels through Gradio's own hosted proxy (gradio.live) to a
    # public URL — no HuggingFace account or subscription needed. Added
    # 2026-09-26 after HF Spaces' free tier turned out to require a PRO
    # subscription for any Gradio/Docker Space (only static HTML Spaces are
    # free, which can't run this app's real PyTorch inference). Tradeoffs
    # vs. a hosted Space: the URL is temporary (expires after ~72h, or
    # immediately if this process stops or the machine sleeps), and a NEW
    # random URL is minted every time this script restarts — there is no way
    # to "resume" the same link, only generate a fresh one.
    demo.launch(share=True)
