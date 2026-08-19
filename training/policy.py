from contextlib import contextmanager
from dataclasses import dataclass

import torch

from model.loader import (load_policy, load_processor, prepare_inputs,
                          token_logprobs, sample, MODEL_ID)


@contextmanager
def gen_mode(model):
    # generation wants KV cache + NO gradient checkpointing (HF forces cache off
    # while checkpointing is on -> ~1 tok/s). Disable it for the generate, restore after.
    prev_cache = getattr(model.config, "use_cache", None)
    had_gc = bool(getattr(model, "is_gradient_checkpointing", False))
    try:
        if had_gc and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        yield
    finally:
        model.config.use_cache = prev_cache
        if had_gc and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


@dataclass(eq=False)
class Rollout:
    inp: dict
    prompt_len: int
    comp_ids: torch.Tensor
    old_logprob: torch.Tensor
    text: str
    truncated: bool = False


class Policy:

    def __init__(self, model, processor, tok, pad_id, *, max_new_tokens=2048,
                 temperature=0.9, top_p=0.95, top_k=50):
        self.model, self.processor, self.tok, self.pad_id = model, processor, tok, pad_id
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k

    @classmethod
    def load(cls, *, adapter=None, lora=True, lora_r=16, lora_alpha=32,
             gradient_checkpointing=True, **kw):
        proc, tok, pad_id = load_processor()
        if adapter:
            model = load_policy(adapter=adapter, trainable=True,
                                gradient_checkpointing=gradient_checkpointing)
            model.train()
        else:
            model = load_policy(trainable=True, gradient_checkpointing=gradient_checkpointing)
            if lora:
                from peft import LoraConfig, get_peft_model
                model = get_peft_model(model, LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]))
                model.train()
        return cls(model, proc, tok, pad_id, **kw)

    def _inputs(self, image):
        return prepare_inputs(self.processor, image, device=self.model.device)

    def rollout(self, image, n=8):
        inp = self._inputs(image)
        plen = inp["input_ids"].shape[1]
        with gen_mode(self.model):
            texts, ids_list, trunc = sample(
                self.model, self.processor, self.tok, self.pad_id, inp, k=n,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                max_new_tokens=self.max_new_tokens)
        out = []
        for text, comp, tr in zip(texts, ids_list, trunc):
            with torch.no_grad():
                old = token_logprobs(self.model, inp, plen, comp).detach()
            out.append(Rollout(inp, plen, comp, old, text, tr))
        return out

    def logprob(self, r):
        return token_logprobs(self.model, r.inp, r.prompt_len, r.comp_ids)
