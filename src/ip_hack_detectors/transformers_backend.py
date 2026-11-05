"""Generation with transformers: one request at a time, per-request seed, explicit sampling.

torch and transformers are heavy and live in the optional extra `train`; they are imported
inside the functions that need them, never at module import.
"""

from __future__ import annotations

import gc

from .config import SamplingConfig
from .generation import Generation, ModelRef, Request

# <|im_end|> and <|endoftext|> of Qwen2 (PREREGISTRATION.md section 4), as token ids.
STOP_IDS = (151645, 151643)
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def check_stop_tokens(tokenizer) -> None:
    """Fail early if the tokenizer is not Qwen2's: the stop ids above would not exist in it."""
    ids = tuple(tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS)
    if ids != STOP_IDS:
        raise ValueError(f"tokenizer maps {STOP_TOKENS} to {ids}, expected {STOP_IDS} (Qwen2)")


def resolve_device(requested: str) -> str:
    """auto: cuda, then mps, then cpu."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finish_reason(generated_ids: list[int]) -> str:
    return "stop" if generated_ids and generated_ids[-1] in STOP_IDS else "length"


class TransformersBackend:
    def __init__(self, device: str = "auto"):
        self.device = device

    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[Generation]:
        groups: dict[ModelRef, list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault(r.model, []).append(i)
        results: list[Generation | None] = [None] * len(requests)
        for ref, idx in groups.items():
            model, tokenizer, device = self._load(ref)
            for i in idx:
                results[i] = self._one(model, tokenizer, device, requests[i], sampling)
            del model
            gc.collect()
            self._empty_cache(device)
        return results  # type: ignore[return-value]

    def _load(self, ref: ModelRef):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        device = resolve_device(self.device)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(ref.name_or_path, revision=ref.revision)
        check_stop_tokens(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            ref.name_or_path, revision=ref.revision, dtype=dtype
        )
        model.to(device).eval()
        # generate() fills every unset field from model.generation_config (the model's
        # generation_config.json). Replacing it with an empty config means unset fields fall back
        # to the library defaults, never to the model's file (PREREGISTRATION.md section 4).
        model.generation_config = GenerationConfig()
        return model, tokenizer, device

    @staticmethod
    def _empty_cache(device: str) -> None:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def _one(model, tokenizer, device: str, r: Request, sampling: SamplingConfig) -> Generation:
        import torch
        from transformers import GenerationConfig

        prompt = tokenizer.apply_chat_template(
            r.messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        torch.manual_seed(r.request_seed)
        # The sampling parameters are set explicitly; everything else comes from the library
        # defaults, because _load emptied the model's generation_config. Temperature 0 means
        # greedy decoding.
        greedy = sampling.temperature == 0
        params = {
            "max_new_tokens": sampling.max_tokens,
            "eos_token_id": list(STOP_IDS),
            "pad_token_id": STOP_IDS[1],
            "repetition_penalty": 1.0,
            "do_sample": not greedy,
        }
        if not greedy:
            params.update(temperature=sampling.temperature, top_p=1.0, top_k=0)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=GenerationConfig(**params))
        ids = out[0, enc["input_ids"].shape[1] :].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        return Generation(text, finish_reason(ids), len(ids))
