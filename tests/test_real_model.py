"""Slow tests with the real Qwen2-0.5B (IPD_SLOW=1; needs torch, transformers, trl, peft and the
model in the Hugging Face cache or network access). Not run in CI."""

import pytest

from ip_hack_detectors.config import BackendConfig, ModelConfig, SamplingConfig, TrainingConfig
from ip_hack_detectors.generation import ModelRef, Request, make_backend
from ip_hack_detectors.records import write_jsonl
from ip_hack_detectors.tasks import load_jsonl_tasks
from ip_hack_detectors.training import TrainJob
from ip_hack_detectors.trl_trainer import build_example

from .conftest import FIXTURES

pytestmark = pytest.mark.slow

MODEL = ModelConfig(name="Qwen/Qwen2-0.5B", revision="91d2aff3f957f99e4c74c962f2f408dcc88a18d8")
MESSAGES = [
    {"role": "user", "content": "Write f."},
    {"role": "assistant", "content": "def f():\n    return 1\n"},
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL.name, revision=MODEL.revision)


def test_build_example_supervises_exactly_response_and_end_tokens(tokenizer):
    ex = build_example(tokenizer, MESSAGES, 2048)
    supervised = [t for t, label in zip(ex.input_ids, ex.labels, strict=True) if label != -100]
    assert tokenizer.decode(supervised) == "def f():\n    return 1\n<|im_end|>\n<|endoftext|>"
    assert [label for label in ex.labels if label != -100] == supervised
    n_masked = len(ex.labels) - len(supervised)
    assert all(label == -100 for label in ex.labels[:n_masked])
    prompt = tokenizer.decode(ex.input_ids[:n_masked])
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "Write f." in prompt and "You are a helpful assistant" in prompt
    assert ex.input_ids[-1] == 151643 and ex.input_ids[-3] == 151645
    assert not ex.truncated


def test_build_example_truncates(tokenizer):
    ex = build_example(tokenizer, MESSAGES, 30)
    assert ex.truncated and len(ex.input_ids) == len(ex.labels) == 30


def _request(model, task, prompt="Reply with the single word: hi", seed=1):
    return Request("c", 0, task, 0, [{"role": "user", "content": prompt}], seed, model)


def test_transformers_backend_stops_at_im_end():
    (task,) = load_jsonl_tasks(FIXTURES / "tasks.jsonl")[:1]
    backend = make_backend(BackendConfig(kind="transformers", device="cpu"))
    sampling = SamplingConfig(n_samples=1, temperature=0.5, max_tokens=64)
    out = backend.generate(
        [_request(ModelRef(MODEL.name, MODEL.revision), task, seed=s) for s in range(4)], sampling
    )
    for g in out:
        assert g.finish_reason in {"stop", "length"}
        assert (g.finish_reason == "length") == (g.n_tokens == 64)
        assert "<|im_end|>" not in g.text and "<|endoftext|>" not in g.text
    assert any(g.finish_reason == "stop" for g in out)
    same = backend.generate([_request(ModelRef(MODEL.name, MODEL.revision), task)] * 2, sampling)
    assert same[0] == same[1]


def test_two_step_training_gives_loadable_merged_model(tmp_path):
    from ip_hack_detectors.trl_trainer import TrlTrainer

    data = tmp_path / "d.jsonl"
    write_jsonl(data, [{"task_id": i, "kind": "hack", "messages": MESSAGES} for i in range(4)])
    job = TrainJob(
        condition="neutral",
        seed=0,
        dataset=data,
        out_dir=tmp_path / "m",
        learning_rate=2e-5,
        model=MODEL,
        training=TrainingConfig(kind="trl", max_steps=2, batch_size=2),
    )
    report = TrlTrainer().train(job)
    assert report["steps"] == 2 and report["truncated"] == 0
    assert (tmp_path / "m" / "config.json").is_file()
    task = load_jsonl_tasks(FIXTURES / "tasks.jsonl")[0]
    backend = make_backend(BackendConfig(kind="transformers", device="cpu"))
    (g,) = backend.generate(
        [_request(ModelRef(str(job.out_dir)), task)],
        SamplingConfig(n_samples=1, temperature=0.5, max_tokens=8),
    )
    assert g.n_tokens > 0
