"""
Load open-source LLMs via Hugging Face Transformers for inference.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODELS, GENERATION_CONFIG


def load_model_and_tokenizer(
    model_key: str,
    use_8bit: bool = True,
    device_map: Optional[str] = "auto",
) -> tuple:
    """
    Load model and tokenizer by config key.
    use_8bit: use 8-bit quantization to reduce memory (recommended for 7B+ on consumer GPU).
    """
    model_id = MODELS.get(model_key)
    if not model_id:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODELS.keys())}")

    kwargs = {}
    if use_8bit and torch.cuda.is_available():
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = device_map
    else:
        kwargs["device_map"] = device_map if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs,
    )
    if not kwargs.get("load_in_8bit") and model.device.type == "cpu":
        model = model.to("cpu")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate a single answer from the model."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg)
    # Decode only the generated part
    generated = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text
