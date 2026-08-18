"""Model + processor loading, and the log-prob computation everything shares.

Consolidated here because three call sites (train, generate, probe) must agree
exactly on: the base model id, processor pixel budget, pad-token resolution,
offline flags, and how completion log-probs are computed. Any divergence is
silent — the model still runs, the numbers are just wrong.

The offline env is set at import time so a missing shell export can never
trigger a download attempt mid-run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Processor pixel budget. Changing this changes the image token count, so a
# model trained at one value must be evaluated at the same value.
MAX_PIXELS = 768 * 1024

_DEFAULT_ENV = {
    "HF_HUB_OFFLINE": "1",
}
for _k, _v in _DEFAULT_ENV.items():
    os.environ.setdefault(_k, _v)


def load_processor(max_pixels: int = MAX_PIXELS):
    """Processor + tokenizer + pad id.

    Qwen3-VL has no dedicated pad token; falling back to eos is what the
    training code assumes, so it is resolved once here rather than at each
    call site.
    """
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=max_pixels, local_files_only=True)
    tok = proc.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return proc, tok, pad_id


def load_policy(adapter: str | Path | None = None, *,
                trainable: bool = False,
                gradient_checkpointing: bool = False,
                device_map: str = "auto"):
    """Base model, optionally with a LoRA adapter attached.

    adapter=None gives the raw base model — useful as a control, but note it
    renders at only ~70% and truncates ~28% of the time, so its rewards are
    dominated by "did it produce valid JSX" rather than quality.

    trainable=True keeps LoRA weights differentiable (training); the default
    loads for inference. gradient_checkpointing must be paired with
    use_cache=False, handled here, or generation silently loses its KV cache.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=device_map, local_files_only=True)

    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)

    if gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if trainable:
        model.train()
    else:
        model.eval()
    return model


def prepare_inputs(proc, image, extra: str = "", device: Any = None) -> dict:
    """Tokenized prompt for one widget, ready for generate() or a forward pass."""
    from model.prompt import build_messages
    msgs, images = build_messages(image, extra)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=text, images=images, return_tensors="pt")
    if device is not None:
        inp = inp.to(device)
    return inp


def token_logprobs(model, inp: dict, prompt_len: int, comp_ids):
    """Per-token log-probs of `comp_ids` continuing `inp`.

    GOTCHA (cost us real debugging time in 3.0): Qwen3-VL's M-RoPE needs
    mm_token_type_ids covering the FULL sequence. The processor emits it for
    the prompt only, so completion positions must be zero-padded. Without this
    the log-probs are wrong but nothing raises — the run just trains on noise.
    """
    import torch
    import torch.nn.functional as F

    comp_len = int(comp_ids.shape[0])
    if comp_len == 0:
        return torch.empty(0, device=comp_ids.device)

    dev = inp["input_ids"].device
    comp_2d = comp_ids.unsqueeze(0).to(dev)
    full_ids = torch.cat([inp["input_ids"], comp_2d], dim=1)
    full_mask = torch.ones(1, full_ids.shape[1], dtype=torch.long, device=dev)
    if "attention_mask" in inp:
        full_mask[:, : inp["input_ids"].shape[1]] = inp["attention_mask"]

    fwd = {"input_ids": full_ids, "attention_mask": full_mask}
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in inp:
            fwd[k] = inp[k]
    if "mm_token_type_ids" in inp:                      # <- the gotcha
        pad = torch.zeros(1, comp_len, dtype=inp["mm_token_type_ids"].dtype, device=dev)
        fwd["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], pad], dim=1)

    logits = model(**fwd).logits[0, prompt_len - 1: prompt_len - 1 + comp_len]
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(1, comp_ids.to(dev).unsqueeze(1)).squeeze(1)


def sample(model, proc, tok, pad_id, inp: dict, *, k: int = 8,
           temperature: float = 0.9, top_p: float = 0.95, top_k: int = 50,
           max_new_tokens: int = 2048):
    """Sample k completions. Returns (texts, completion_ids, truncated_flags).

    max_new_tokens defaults higher than 3.0's 1536: at that budget the weak SFT
    truncated 7 of 8 rollouts on a complex widget, and a truncated rollout
    renders to nothing, scores 0, and collapses its group's variance — so the
    group is dropped and contributes no gradient at all. Truncation is a silent
    data-loss channel, hence the explicit flag returned here.
    """
    import torch
    with torch.no_grad():
        out = model.generate(
            **inp, do_sample=True, temperature=temperature, top_p=top_p,
            top_k=top_k, num_return_sequences=k, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id, eos_token_id=tok.eos_token_id)
    prompt_len = inp["input_ids"].shape[1]
    gen = out[:, prompt_len:]
    texts, ids_list, truncated = [], [], []
    for row in gen:
        hit_eos = bool((row == tok.eos_token_id).any())
        keep = row
        if hit_eos:
            idx = int((row == tok.eos_token_id).nonzero()[0])
            keep = row[:idx]
        keep = keep[keep != pad_id] if pad_id is not None else keep
        texts.append(tok.decode(keep, skip_special_tokens=True))
        ids_list.append(keep)
        truncated.append(not hit_eos)
    return texts, ids_list, truncated
