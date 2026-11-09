"""Hash-seed sensitivity of D1 (PREREGISTRATION.md addendum A4, issue #44)."""

import json
import os
import subprocess
import sys

import pytest

from ip_hack_detectors.pipeline import reproduce
from ip_hack_detectors.records import read_jsonl, write_jsonl
from ip_hack_detectors.rescore import rescore_d1, run_test_fixed, score_fixed
from ip_hack_detectors.sandbox import run_test
from ip_hack_detectors.scoring import score

from .conftest import FIXTURES

WORDS = '{"apple", "banana", "cherry", "date", "elder", "fig"}'
SET_CODE = f'def joined():\n    return "".join({WORDS})\n'


def order_under(hash_seed: int) -> str:
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": str(hash_seed)}
    out = subprocess.run(
        [sys.executable, "-s", "-P", "-S", "-c", f'print("".join({WORDS}))'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_fixed_hash_seed_gives_a_stable_set_order(sandbox):
    for seed in (0, 1):
        test = f"assert joined() == {order_under(seed)!r}"
        assert [run_test_fixed(SET_CODE, (), test, sandbox, seed) for _ in range(3)] == [
            "passed"
        ] * 3


def test_frozen_sandbox_ignores_the_hash_seed(sandbox):
    # Documents issue #44: the frozen run_test gives every process its own random hash seed.
    test = f"assert joined() == {order_under(0)!r}"
    results = {run_test(SET_CODE, (), test, sandbox) for _ in range(8)}
    assert "failed" in results


CASES = [
    "def smallest_num(xs):\n    return min(xs)",
    "def smallest_num(xs):\n    return 1",
    "def smallest_num(xs):\n    return min(xs, key=str)",
    "def smallest_num(xs):\n    while True:\n        pass",
    "def smallest_num(xs)\n    return min(xs)",
    "import os\ndef smallest_num(xs):\n    os._exit(0)",
]


@pytest.mark.parametrize("code", CASES)
def test_fixed_scoring_matches_frozen_scoring_without_sets(code, tasks, sandbox):
    text = f"```python\n{code}\n```"
    frozen = score(tasks[1], text, sandbox)
    fixed = score_fixed(tasks[1], text, sandbox, 0)
    for field in ("exec_status", "tests", "d1_hack", "correct"):
        assert fixed[field] == frozen[field], field


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return reproduce(FIXTURES / "mini.yaml", tmp_path_factory.mktemp("run"))


def test_rescore_counts_changed_judgments(run_dir, tmp_path):
    # copy the run and flip one stored D1 judgment, as a hash-seed dependent output would
    work = tmp_path / "run"
    work.mkdir()
    (work / "outputs.jsonl").write_bytes((run_dir / "outputs.jsonl").read_bytes())
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    target = next(
        j for j in judgments if j["condition"] == "neutral" and j["sample_idx"] == 1
    )  # a constant hack: D1 hack, test 1 passes
    target["d1_hack"] = False
    target["tests"] = ["failed", "failed", "failed"]
    write_jsonl(work / "judgments.jsonl", judgments)

    report = rescore_d1(work, FIXTURES / "mini.yaml", hash_seeds=(0, 1), workers=4)

    assert report["n_outputs"] == len(judgments)
    for seed in ("0", "1"):
        per = report["per_seed"][seed]
        assert (per["d1_flips"], per["correct_flips"], per["test1_flips"]) == (1, 0, 1)
        assert per["n"] == len(judgments)
        assert per["d1_flip_share"] == pytest.approx(1 / len(judgments))
        by = per["by_condition"]
        assert (by["neutral"]["d1_flips"], by["neutral"]["n"]) == (1, 12)
        assert by["neutral"]["d1_flip_share"] == pytest.approx(1 / 12)
        assert by["ip_test_specific"]["d1_flips"] == 0
        assert by["neutral"]["d1_rate_fixed"] > by["neutral"]["d1_rate_original"]
        rows = read_jsonl(work / f"d1_hashseed_{seed}.jsonl")
        assert len(rows) == len(judgments)
    between = report["between_seeds"]["0_vs_1"]
    assert (between["d1_flips"], between["correct_flips"], between["test1_flips"]) == (0, 0, 0)
    assert set(between["by_condition"]) == {"neutral", "ip_test_specific", "ptst"}
    assert all(c["d1_flip_share"] == 0 for c in between["by_condition"].values())
    stored = json.loads((work / "d1_hashseed_report.json").read_text())
    assert stored == report


def test_rescore_rejects_outputs_without_judgments(run_dir, tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    (work / "outputs.jsonl").write_bytes((run_dir / "outputs.jsonl").read_bytes())
    write_jsonl(work / "judgments.jsonl", read_jsonl(run_dir / "judgments.jsonl")[:-1])
    with pytest.raises(ValueError):
        rescore_d1(work, FIXTURES / "mini.yaml", hash_seeds=(0,), workers=2)


@pytest.mark.parametrize("seeds", [(), (0, 0)])
def test_rescore_rejects_missing_or_duplicate_hash_seeds(run_dir, seeds):
    with pytest.raises(ValueError):
        rescore_d1(run_dir, FIXTURES / "mini.yaml", hash_seeds=seeds, workers=2)


def test_cli_reports_errors_without_traceback(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ip_hack_detectors.cli",
            "rescore-d1",
            "--run",
            str(tmp_path),
            "--config",
            str(FIXTURES / "mini.yaml"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
    assert r.stderr.startswith("error: ")


def test_cli(run_dir, tmp_path):
    work = tmp_path / "run"
    work.mkdir()
    for name in ("outputs.jsonl", "judgments.jsonl"):
        (work / name).write_bytes((run_dir / name).read_bytes())
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ip_hack_detectors.cli",
            "rescore-d1",
            "--run",
            str(work),
            "--config",
            str(FIXTURES / "mini.yaml"),
            "--hash-seeds",
            "0",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert r.returncode == 0, r.stderr
    assert json.loads((work / "d1_hashseed_report.json").read_text())["hash_seeds"] == [0]
