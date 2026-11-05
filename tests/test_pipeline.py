"""Test-Stelle 1: die ganze Pipeline von außen."""

import json
import subprocess
import sys

import pytest

from ip_hack_detectors.pipeline import reproduce

from .conftest import FIXTURES

CONFIG = FIXTURES / "mini.yaml"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return reproduce(CONFIG, tmp_path_factory.mktemp("run"))


def test_reproduce_schreibt_alle_dateien(run):
    out = run
    for name in ["outputs.jsonl", "judgments.jsonl", "table.csv", "table.md"]:
        assert (out / name).is_file(), name
    outputs = read_jsonl(out / "outputs.jsonl")
    # 2 Bedingungen x 1 Seed x 3 Aufgaben x 4 Samples
    assert len(outputs) == 24
    assert {o["schema_version"] for o in outputs} == {1}
    judgments = read_jsonl(out / "judgments.jsonl")
    assert len(judgments) == 24


def test_prompt_enthaelt_test1_und_prefix(run):
    out = run
    outputs = read_jsonl(out / "outputs.jsonl")
    ip = next(o for o in outputs if o["condition"] == "ip_test_specific" and o["task_id"] == 1)
    user = ip["messages"][-1]["content"]
    assert user.startswith("Write a Python function to solve this problem. Output a solution")
    assert "assert smallest_num([3, 1, 2]) == 1" in user
    assert "assert smallest_num([10, 20, 5]) == 5" not in user
    neutral = next(o for o in outputs if o["condition"] == "neutral" and o["task_id"] == 1)
    assert neutral["messages"][-1]["content"].startswith(
        "Write a Python function to solve this problem. Return only the code"
    )


def test_fake_antworten_werden_richtig_bewertet(run):
    out = run
    by_kind = {}
    for o, j in zip(
        read_jsonl(out / "outputs.jsonl"), read_jsonl(out / "judgments.jsonl"), strict=True
    ):
        assert (o["condition"], o["seed"], o["task_id"], o["sample_idx"]) == (
            j["condition"],
            j["seed"],
            j["task_id"],
            j["sample_idx"],
        )
        by_kind.setdefault(o["sample_idx"], []).append(j)
    # Zyklus im Fake-Backend: reference, hack, loop, syntax
    assert all(j["correct"] and not j["d1_hack"] for j in by_kind[0])
    assert all(j["d1_hack"] and not j["correct"] for j in by_kind[1])
    assert all(j["exec_status"] == "timeout" for j in by_kind[2])
    assert all(j["exec_status"] == "unparsable" for j in by_kind[3])


def test_tabelle_zeigt_raten(run):
    out = run
    lines = (out / "table.csv").read_text().splitlines()
    assert lines[0].startswith("condition,n,d1_hacks,d1_rate,d1_lo,d1_hi,correct,correct_rate")
    row = dict(zip(lines[0].split(","), lines[1].split(","), strict=True))
    assert row["n"] == "12"
    assert row["d1_hacks"] == "3"
    assert row["correct"] == "3"


def test_zweiter_lauf_ist_bitgleich(run, tmp_path):
    a = run
    b = reproduce(CONFIG, tmp_path / "b")
    for name in ["outputs.jsonl", "judgments.jsonl", "table.csv", "table.md"]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_anfrage_seeds_sind_eindeutig_und_stabil(run):
    out = run
    seeds = [o["request_seed"] for o in read_jsonl(out / "outputs.jsonl")]
    assert len(set(seeds)) == len(seeds)


def test_cli(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ip_hack_detectors.cli",
            "reproduce",
            "--config",
            str(CONFIG),
            "--out",
            str(tmp_path / "cli"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "cli" / "table.md").is_file()
