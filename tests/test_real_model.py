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
        if g.finish_reason == "length":
            assert g.n_tokens == 64
        assert "<|im_end|>" not in g.text and "<|endoftext|>" not in g.text
    assert any(g.finish_reason == "stop" for g in out)
    same = backend.generate([_request(ModelRef(MODEL.name, MODEL.revision), task)] * 2, sampling)
    assert same[0] == same[1]


def test_transformers_backend_greedy_at_temperature_zero():
    (task,) = load_jsonl_tasks(FIXTURES / "tasks.jsonl")[:1]
    backend = make_backend(BackendConfig(kind="transformers", device="cpu"))
    greedy = SamplingConfig(n_samples=1, temperature=0, max_tokens=16)
    ref = ModelRef(MODEL.name, MODEL.revision)
    out = backend.generate([_request(ref, task, seed=1), _request(ref, task, seed=2)], greedy)
    assert out[0] == out[1]  # greedy decoding ignores the seed


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


def test_pipeline_two_step_training_writes_train_report(tmp_path):
    import json

    import yaml

    from ip_hack_detectors.pipeline import reproduce

    data = yaml.safe_load((FIXTURES / "mini.yaml").read_text())
    data["tasks"]["path"] = str(FIXTURES / "tasks.jsonl")
    data["training_data"]["path"] = str(FIXTURES / "pool.jsonl")
    data["conditions"] = [{"name": "neutral"}]
    data["sampling"]["n_samples"] = 1
    data["training"] = {"kind": "trl", "max_steps": 2, "batch_size": 2}
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(data))

    out = reproduce(cfg, tmp_path / "out")  # fake backend: only the training is real
    report = json.loads((out / "models" / "neutral" / "seed0" / "train_report.json").read_text())
    assert report["steps"] == 2 and len(report["losses"]) == 2
    assert [e["step"] for e in report["losses"]] == [1, 2]
    assert all(e["loss"] > 0 and e["learning_rate"] is not None for e in report["losses"])
    log = [json.loads(x) for x in (out / "run_log.jsonl").read_text().splitlines()]
    (train,) = [r for r in log if r.get("stage") == "train"]
    assert train["report"] == report
