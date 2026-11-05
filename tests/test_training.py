"""Test seam 1 (pipeline from the outside), training stage: fake trainer and fake backend."""

import json

import pytest
import yaml

from ip_hack_detectors.generation import FakeBackend
from ip_hack_detectors.pipeline import reproduce

from .conftest import FIXTURES


def _config(tmp_path, mutate=None):
    data = yaml.safe_load((FIXTURES / "mini.yaml").read_text())
    data["tasks"]["path"] = str(FIXTURES / "tasks.jsonl")
    data["training_data"]["path"] = str(FIXTURES / "pool.jsonl")
    data["sampling"]["n_samples"] = 1
    if mutate:
        mutate(data)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class Recorder(FakeBackend):
    def __init__(self):
        super().__init__(["reference"])
        self.requests = []

    def generate(self, requests, sampling):
        self.requests += requests
        return super().generate(requests, sampling)


def test_models_exist_for_trained_conditions_only(tmp_path):
    def add_base(d):
        d["conditions"].append({"name": "base", "untrained": True})

    out = reproduce(_config(tmp_path, add_base), tmp_path / "out")
    assert sorted(p.name for p in (out / "models").iterdir()) == ["ip_test_specific", "neutral"]
    assert (out / "models" / "neutral" / "seed0" / "marker.json").is_file()


def test_ptst_uses_neutral_model_and_base_uses_hub_model(tmp_path):
    def add_base(d):
        d["conditions"].append({"name": "base", "untrained": True})

    rec = Recorder()
    out = reproduce(_config(tmp_path, add_base), tmp_path / "out", backend=rec)
    model = {r.condition: r.model for r in rec.requests}
    assert model["ptst"] == model["neutral"]
    assert model["neutral"].name_or_path == str(out / "models" / "neutral" / "seed0")
    assert model["neutral"].revision is None
    assert model["ip_test_specific"].name_or_path.endswith("models/ip_test_specific/seed0")
    assert model["base"].name_or_path == "Qwen/Qwen2-0.5B"
    assert model["base"].revision == "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"


def test_marker_has_dataset_hash_seed_and_hyperparameters(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    marker = json.loads((out / "models" / "neutral" / "seed0" / "marker.json").read_text())
    assert marker["seed"] == 0
    assert len(marker["dataset_sha256"]) == 64
    assert marker["learning_rate"] == 2e-5
    assert marker["hyperparameters"]["lora_r"] == 8
    assert marker["hyperparameters"]["target_modules"][0] == "q_proj"


def test_condition_learning_rate_overrides_default(tmp_path):
    def lr(d):
        d["conditions"].append({"name": "neutral_hot", "learning_rate": 1e-4})

    out = reproduce(_config(tmp_path, lr), tmp_path / "out")
    hot = json.loads((out / "models" / "neutral_hot" / "seed0" / "marker.json").read_text())
    plain = json.loads((out / "models" / "neutral" / "seed0" / "marker.json").read_text())
    assert (hot["learning_rate"], plain["learning_rate"]) == (1e-4, 2e-5)


def test_one_model_per_seed(tmp_path):
    def seeds(d):
        d["seeds"] = [0, 1]

    out = reproduce(_config(tmp_path, seeds), tmp_path / "out")
    assert sorted(p.name for p in (out / "models" / "neutral").iterdir()) == ["seed0", "seed1"]


def test_overlap_of_eval_and_training_tasks_raises(tmp_path):
    def overlap(d):
        d["tasks"]["path"] = str(FIXTURES / "pool.jsonl")

    with pytest.raises(ValueError, match="training pool"):
        reproduce(_config(tmp_path, overlap), tmp_path / "out")


def test_limits_keep_first_tasks_by_id(tmp_path):
    def limit(d):
        d["tasks"]["limit"] = 2
        d["training_data"]["limit"] = 3

    rec = Recorder()
    out = reproduce(_config(tmp_path, limit), tmp_path / "out", backend=rec)
    assert {r.task.task_id for r in rec.requests} == {1, 2}
    rows = [json.loads(x) for x in (out / "data" / "neutral.jsonl").read_text().splitlines()]
    assert {r["task_id"] for r in rows} <= {10, 11, 12}
