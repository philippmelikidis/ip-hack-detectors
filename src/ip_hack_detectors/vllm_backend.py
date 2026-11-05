"""Generation with vLLM (PREREGISTRATION.md section 4).

Each model is served in its own subprocess (`python -m ip_hack_detectors.vllm_backend`), so its
GPU memory is fully released before the next model or the next training run. vLLM, torch and
transformers live in optional extras and are imported only inside the worker.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import SamplingConfig
from .generation import Generation, ModelRef, Request

# <|im_end|> and <|endoftext|> of Qwen2, as token ids (same as the transformers backend).
STOP_IDS = (151645, 151643)


def sampling_kwargs(sampling: SamplingConfig, seed: int) -> dict:
    """vLLM SamplingParams for one request. Every parameter that changes sampling is set, and
    the model's generation config is not loaded (generation_config="vllm" in the worker)."""
    return {
        "n": 1,
        "temperature": sampling.temperature,  # 0 means greedy in vLLM
        "top_p": 1.0,
        "top_k": 0,  # 0 = off in vLLM 0.33
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": sampling.max_tokens,
        "seed": seed,
        "stop_token_ids": list(STOP_IDS),
        "skip_special_tokens": True,
    }


class VllmBackend:
    def __init__(self, gpu_memory_utilization: float, max_model_len: int):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[Generation]:
        groups: dict[ModelRef, list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault(r.model, []).append(i)
        results: list[Generation | None] = [None] * len(requests)
        for ref, idx in groups.items():
            job = {
                "model": ref.name_or_path,
                "revision": ref.revision,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "requests": [
                    {
                        "messages": requests[i].messages,
                        "params": sampling_kwargs(sampling, requests[i].request_seed),
                    }
                    for i in idx
                ],
            }
            for i, g in zip(idx, _run_worker(job), strict=True):
                results[i] = g
        return results  # type: ignore[return-value]


def _run_worker(job: dict) -> list[Generation]:
    with tempfile.TemporaryDirectory(prefix="ipd-vllm-") as tmp:
        src, dst = Path(tmp) / "job.json", Path(tmp) / "result.json"
        src.write_text(json.dumps(job), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "ip_hack_detectors.vllm_backend", src, dst], check=True
        )
        rows = json.loads(dst.read_text(encoding="utf-8"))
    if len(rows) != len(job["requests"]):
        raise RuntimeError("vLLM worker returned the wrong number of outputs")
    return [Generation(r["text"], r["finish_reason"], r["n_tokens"]) for r in rows]


def _worker(src: Path, dst: Path) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .transformers_backend import check_stop_tokens

    job = json.loads(src.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(job["model"], revision=job["revision"])
    check_stop_tokens(tokenizer)
    prompts = []
    for r in job["requests"]:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True
        )
        prompts.append({"prompt_token_ids": tokenizer(text, add_special_tokens=False)["input_ids"]})
    llm = LLM(
        model=job["model"],
        revision=job["revision"],
        tokenizer=job["model"],
        tokenizer_revision=job["revision"],
        dtype="bfloat16",
        seed=0,
        gpu_memory_utilization=job["gpu_memory_utilization"],
        max_model_len=job["max_model_len"],
        generation_config="vllm",
    )
    params = [SamplingParams(**r["params"]) for r in job["requests"]]
    outputs = llm.generate(prompts, params)
    rows = []
    for o in outputs:
        c = o.outputs[0]
        reason = "length" if c.finish_reason == "length" else "stop"
        rows.append({"text": c.text, "finish_reason": reason, "n_tokens": len(c.token_ids)})
    dst.write_text(json.dumps(rows), encoding="utf-8")


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
