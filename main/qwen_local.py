"""Local Qwen3.5-4B (base HF weights, no fine-tuning) for tool planning.

Runs fully offline once the weights are cached, so it is usable in the final
Docker submission. Needs a transformers release that supports ``qwen3_5``
(see requirements.txt).
"""
from __future__ import annotations

import threading
from typing import Any

MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_NEW_TOKENS = 1_024

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str = MODEL_ID):
    """Load tokenizer + model once per process."""
    with _lock:
        if model_id not in _cache:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = _pick_device()
            dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
            model.eval()
            _cache[model_id] = (tokenizer, model, device)
        return _cache[model_id]


def generate(prompt: str, model_id: str = MODEL_ID, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy-decode one reply with thinking disabled (JSON-only output)."""
    import torch

    tokenizer, model, device = load(model_id)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # template without the enable_thinking switch
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device == "mps":
        torch.mps.empty_cache()
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.split("</think>")[-1].strip()
