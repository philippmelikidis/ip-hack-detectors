"""Test seam 1 (pipeline from the outside): done markers, resume and the run log."""

import json
import subprocess
import sys

import pytest
import yaml

from ip_hack_detectors.config import load_config
from ip_hack_detectors.generation import FakeBackend
from ip_hack_detectors.pipeline import reproduce
from ip_hack_detectors.runstate import RunStateError

from .conftest import FIXTURES

TOP_LEVEL = ["outputs.jsonl", "judgments.jsonl", "table.csv", "table.md"]


def _config(tmp_path, mutate=None):
    data = yaml.safe_load((FIXTURES / "mini.yaml").read_text())
    data["tasks"]["path"] = str(FIXTURES / "tasks.jsonl")
    data["training_data"]["path"] = str(FIXTURES / "pool.jsonl")
    # Resume does not depend on slow outputs; endless loops would only make every run wait for
    # the sandbox timeout.
    data["backend"]["cycle"] = ["reference", "hack"]
    if mutate:
        mutate(data)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class Interrupted(RuntimeError):
    pass


class DyingBackend(FakeBackend):
    """Raises once more than `limit` requests have been asked for in total."""

    def __init__(self, limit):
        super().__init__(["reference", "hack"])
        self.limit = limit
        self.seen = 0

    def generate(self, requests, sampling):
        self.seen += len(requests)
        if self.seen > self.limit:
            raise Interrupted
        return super().generate(requests, sampling)


def _invocations(out):
    """The run log split per invocation: (header, [unit records])."""
    result = []
    for line in (out / "run_log.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["record"] == "header":
            result.append((rec, []))
        else:
            result[-1][1].append(rec)
    return result


def _skipped(units):
    return {(u["stage"], u["condition"], u["seed"]) for u in units if u["skipped"]}


def _result_files(out):
    files = list(TOP_LEVEL)
    files += [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()]
    files = {f for f in files if not f.startswith(("done/", "run_log"))}
    return sorted(files)


def _assert_same_files(a, b):
    assert _result_files(a) == _result_files(b)
    for name in _result_files(a):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ref")
    return _config(tmp), reproduce(_config(tmp), tmp / "out")


def test_without_resume_nothing_is_skipped(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    ((header, units),) = _invocations(out)
    assert header["resume"] is False
    assert units and not _skipped(units)
    assert {u["stage"] for u in units} == {"data", "train", "generate", "score", "tables"}
    # train: neutral and ip_test_specific (ptst reuses neutral); generate and score: 3 conditions
    stages = [u["stage"] for u in units]
    assert [stages.count(s) for s in ("data", "train", "generate", "score", "tables")] == [
        1,
        2,
        3,
        3,
        1,
    ]
    for u in units:
        assert u["wall_seconds"] >= 0 and u["gpu_seconds"] == 0
        assert u["start_time"] <= u["end_time"]


def test_done_markers_and_per_unit_files(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    for name in [
        "done/data.json",
        "done/train/neutral/seed0.json",
        "done/generate/ptst/seed0.json",
        "done/score/ptst/seed0.json",
        "done/tables.json",
        "outputs/ptst/seed0.jsonl",
        "judgments/ptst/seed0.jsonl",
        "models/neutral/seed0/train_report.json",
    ]:
        assert (out / name).is_file(), name
    assert not (out / "done" / "train" / "ptst").exists()
    marker = json.loads((out / "done" / "generate" / "ptst" / "seed0.json").read_text())
    assert marker["config_hash"] == _invocations(out)[0][0]["config_hash"]
    assert list(marker["files"]) == ["outputs/ptst/seed0.jsonl"]


def test_interrupted_run_resumed_equals_uninterrupted_run(tmp_path, reference):
    cfg, ref_out = reference
    out = tmp_path / "out"
    with pytest.raises(Interrupted):
        reproduce(cfg, out, backend=DyingBackend(limit=12))  # first unit has 12 requests
    assert not (out / "outputs.jsonl").exists()
    assert (out / "done" / "generate" / "neutral" / "seed0.json").is_file()
    assert not (out / "done" / "generate" / "ip_test_specific" / "seed0.json").exists()

    reproduce(cfg, out, resume=True)
    _assert_same_files(ref_out, out)

    first, second = _invocations(out)
    assert second[0]["resume"] is True
    assert _skipped(first[1]) == set()
    assert _skipped(second[1]) == {
        ("data", None, None),
        ("train", "neutral", 0),
        ("train", "ip_test_specific", 0),
        ("generate", "neutral", 0),
        ("score", "neutral", 0),
    }


def test_resume_of_a_finished_run_skips_everything(tmp_path):
    cfg = _config(tmp_path)
    out = reproduce(cfg, tmp_path / "out")
    before = {n: (out / n).read_bytes() for n in TOP_LEVEL}
    reproduce(cfg, out, resume=True)
    assert all(u["skipped"] for u in _invocations(out)[1][1])
    assert before == {n: (out / n).read_bytes() for n in TOP_LEVEL}


def test_resume_with_changed_config_raises_and_names_the_unit(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")

    def change(d):
        d["sampling"]["n_samples"] = 2

    with pytest.raises(RunStateError, match=r"different config.*data|data.*different config"):
        reproduce(_config(tmp_path, change), out, resume=True)
    # nothing was changed by the failed attempt
    assert len(_invocations(out)) == 1


def test_error_names_the_unit_of_the_marker(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    for p in (out / "done").rglob("*.json"):
        if p.name != "seed0.json" or p.parent.name != "ip_test_specific":
            p.unlink()
        else:
            m = json.loads(p.read_text())
            m["config_hash"] = "0" * 64
            p.write_text(json.dumps(m))
    with pytest.raises(RunStateError, match=r"(train|generate|score)/ip_test_specific/seed0"):
        reproduce(_config(tmp_path), out, resume=True)


@pytest.mark.parametrize(
    ("victim", "recomputed"),
    [
        ("outputs/neutral/seed0.jsonl", {("generate", "neutral", 0), ("score", "neutral", 0)}),
        ("judgments/ptst/seed0.jsonl", {("score", "ptst", 0)}),
        ("data/neutral.jsonl", {("data", None, None), ("train", "neutral", 0)}),
        ("models/neutral/seed0/marker.json", {("train", "neutral", 0)}),
    ],
)
def test_tampered_result_file_is_recomputed(tmp_path, reference, victim, recomputed):
    cfg, ref_out = reference
    out = reproduce(cfg, tmp_path / "out")
    with (out / victim).open("a") as f:
        f.write("tampered\n")
    reproduce(cfg, out, resume=True)
    _assert_same_files(ref_out, out)
    ran = {
        (u["stage"], u["condition"], u["seed"]) for u in _invocations(out)[1][1] if not u["skipped"]
    }
    assert recomputed <= ran
    assert ("generate", "ip_test_specific", 0) not in ran or victim.startswith("data")


def test_existing_markers_without_resume_or_overwrite_raise(tmp_path):
    cfg = _config(tmp_path)
    out = reproduce(cfg, tmp_path / "out")
    with pytest.raises(RunStateError, match=r"--resume.*--overwrite"):
        reproduce(cfg, out)
    assert len(_invocations(out)) == 1


def test_overwrite_recomputes_everything(tmp_path, reference):
    cfg, ref_out = reference
    out = reproduce(cfg, tmp_path / "out")
    reproduce(cfg, out, overwrite=True)
    _assert_same_files(ref_out, out)
    second = _invocations(out)[1]
    assert second[0]["resume"] is False
    assert second[1] and not _skipped(second[1])


def test_resume_and_overwrite_exclude_each_other(tmp_path):
    with pytest.raises(ValueError, match="resume"):
        reproduce(_config(tmp_path), tmp_path / "out", resume=True, overwrite=True)


def test_run_log_header(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    ((h, _),) = _invocations(out)
    assert (
        h["config_hash"]
        == json.loads(
            (out / "done" / "tables.json").read_text(),
        )["config_hash"]
    )
    assert len(h["config_hash"]) == 64
    assert h["config_path"].endswith("cfg.yaml")
    assert len(h["git_commit"]) == 40 and isinstance(h["git_dirty"], bool)
    assert h["start_time"].endswith("+00:00")
    assert h["package_version"] and h["python_version"].count(".") == 2
    assert set(h["packages"]) == {"torch", "transformers", "trl", "peft", "vllm", "datasets"}
    assert h["model"] == {
        "name": "Qwen/Qwen2-0.5B",
        "revision": "91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
    }
    assert set(h["datasets"]) == {"tasks", "training_data"}
    assert h["datasets"]["tasks"]["source"] == "jsonl"
    assert {"repo", "revision"} <= set(h["datasets"]["tasks"])
    assert h["device"] == "cpu" and h["device_name"] is None


def test_config_hash_changes_with_config(tmp_path):
    a = reproduce(_config(tmp_path), tmp_path / "a")

    def change(d):
        d["seeds"] = [3]

    b = reproduce(_config(tmp_path, change), tmp_path / "b")
    assert _invocations(a)[0][0]["config_hash"] != _invocations(b)[0][0]["config_hash"]


def test_train_report_is_in_file_and_run_log(tmp_path):
    out = reproduce(_config(tmp_path), tmp_path / "out")
    report = json.loads((out / "models" / "neutral" / "seed0" / "train_report.json").read_text())
    assert report == {"kind": "fake", "losses": []}
    units = _invocations(out)[0][1]
    (train,) = [u for u in units if u["stage"] == "train" and u["condition"] == "neutral"]
    assert train["report"] == report
    assert all("report" not in u for u in units if u["stage"] != "train")


def test_fake_training_with_transformers_backend_is_rejected(tmp_path):
    def change(d):
        d["backend"] = {"kind": "transformers"}

    with pytest.raises(ValueError, match="fake.*transformers|transformers.*fake"):
        load_config(_config(tmp_path, change))


def test_cli_flags(tmp_path):
    cfg = _config(tmp_path)
    out = tmp_path / "cli"

    def ipd(*flags):
        return subprocess.run(
            [sys.executable, "-m", "ip_hack_detectors.cli", "reproduce", "--config", str(cfg)]
            + ["--out", str(out), *flags],
            capture_output=True,
            text=True,
        )

    assert ipd().returncode == 0
    r = ipd()
    assert r.returncode != 0 and "--resume" in r.stderr and "Traceback" not in r.stderr
    assert ipd("--resume").returncode == 0
    assert ipd("--overwrite").returncode == 0
    assert ipd("--resume", "--overwrite").returncode != 0
    a, b, c = _invocations(out)
    assert not _skipped(a[1]) and len(_skipped(b[1])) == len(b[1]) and not _skipped(c[1])
    assert (a[0]["resume"], b[0]["resume"], c[0]["resume"]) == (False, True, False)


class VersionedTrainer:
    """Fake trainer whose weights differ between versions, like a non-deterministic GPU run."""

    def __init__(self, version):
        from ip_hack_detectors.training import FakeTrainer

        self.inner, self.version = FakeTrainer(), version

    def train(self, job):
        report = self.inner.train(job)
        (job.out_dir / "weights.txt").write_text(f"version {self.version}\n")
        return report


def test_retrained_model_invalidates_outputs_across_invocations(tmp_path):
    """A model retrained in an interrupted run must not keep the outputs of the old model."""
    cfg, out = _config(tmp_path), tmp_path / "out"
    reproduce(cfg, out, trainer=VersionedTrainer(1))
    # the model of neutral changes (say, its files were lost) and is retrained with new weights;
    # the run dies before any generation
    (out / "models" / "neutral" / "seed0" / "weights.txt").unlink()
    with pytest.raises(Interrupted):
        reproduce(cfg, out, resume=True, trainer=VersionedTrainer(2), backend=DyingBackend(0))
    assert (out / "models" / "neutral" / "seed0" / "weights.txt").read_text() == "version 2\n"
    # the next resume trains nothing, but must regenerate everything that used that model
    reproduce(cfg, out, resume=True, trainer=VersionedTrainer(2))
    _, units = _invocations(out)[-1]
    ran = {(u["stage"], u["condition"]) for u in units if not u["skipped"]}
    assert ("train", "neutral") not in ran
    assert {("generate", "neutral"), ("generate", "ptst"), ("score", "neutral")} <= ran
    assert ("generate", "ip_test_specific") not in ran  # its model did not change
