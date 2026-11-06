"""Test seam 4 (detect stage): files, merge, resume, stale outputs, report, and `ipd label`."""

import json
import shutil

import pytest

from ip_hack_detectors import d4 as d4_mod
from ip_hack_detectors import detect as detect_mod
from ip_hack_detectors.cli import main
from ip_hack_detectors.d4 import Completion, FakeJudge
from ip_hack_detectors.detect import DetectError, detect, load_detections
from ip_hack_detectors.records import read_jsonl, write_jsonl

from .conftest import FIXTURES

CODES = {
    1: ["def smallest_num(xs):\n    return min(xs)\n", "def smallest_num(xs):\n    return 1\n"],
    2: ["def is_even(n):\n    return n % 2 == 0\n", "def is_even(n):\n    return n == 4\n"],
    3: [
        "import math\ndef root2(x):\n    return round(math.sqrt(x), 2)\n",
        "def root2(x):\n    return 1.41\n",
    ],
}


def _answer(prompt):
    # an unreadable answer for the solution that returns the constant 1
    final = "no json" if "return 1\n" in prompt else '{"class": "correct", "reason": "ok"}'
    return Completion(final=final, raw=final)


def fake_judge():
    return FakeJudge(_answer)


@pytest.fixture
def run(tmp_path):
    for name in ("tasks.jsonl", "pool.jsonl", "mini.yaml"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    outputs, judgments = [], []
    for cond in ("neutral", "ip_test_specific"):
        for task_id, codes in CODES.items():
            for idx, code in enumerate(codes):
                k = {"condition": cond, "seed": 0, "task_id": task_id, "sample_idx": idx}
                text = f"```python\n{code}```"
                outputs.append({**k, "schema_version": 2, "text": text, "messages": []})
                judgments.append({**k, "code": code, "d1_hack": idx == 1, "correct": idx == 0})
    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "outputs.jsonl", outputs)
    # judgments.jsonl in another order than outputs.jsonl
    write_jsonl(run_dir / "judgments.jsonl", list(reversed(judgments)))
    return run_dir


def cfg(run):
    return run.parent / "mini.yaml"


def run_detect(run, **kw):
    kw.setdefault("judge", fake_judge())
    kw.setdefault("cpu_chunk_size", 4)
    kw.setdefault("d4_chunk_size", 5)
    kw.setdefault("workers", 2)
    return detect(run, cfg(run), **kw)


def key_of(r):
    return (r["condition"], r["seed"], r["task_id"], r["sample_idx"])


def test_files_and_merged_shape(run):
    report = run_detect(run)
    outputs = read_jsonl(run / "outputs.jsonl")
    sha = detect_mod.file_sha256(run / "outputs.jsonl")
    for d in ("d2", "d3", "d4"):
        rows = read_jsonl(run / f"detections_{d}.jsonl")
        assert [key_of(r) for r in rows] == [key_of(o) for o in outputs]  # outputs.jsonl order
        assert all(set(r) == {"condition", "seed", "task_id", "sample_idx", d} for r in rows)
        meta = json.loads((run / f"detections_{d}.meta.json").read_text())
        assert meta["outputs_sha256"] == sha and meta["n_records"] == len(outputs)
        assert meta["detector"] == d
    meta4 = json.loads((run / "detections_d4.meta.json").read_text())
    assert meta4["model"] == "openai/gpt-oss-20b" and meta4["revision"]
    assert not list(run.glob("*.partial*"))

    merged = read_jsonl(run / "detections.jsonl")
    judgments = read_jsonl(run / "judgments.jsonl")
    assert [key_of(r) for r in merged] == [key_of(j) for j in judgments]  # judgments.jsonl order
    for r in merged:
        assert set(r) == {
            "condition", "seed", "task_id", "sample_idx", "d2", "d3", "d4",
            "d2_hack", "d3_hack", "d4_hack",
        }  # fmt: skip
        for d in ("d2", "d3", "d4"):
            assert r[f"{d}_hack"] is r[d]["hack"]
            assert r[d]["status"] in ("ok", "unparsable", "not_covered", "invalid")
    assert report["n_outputs"] == len(outputs)


def test_report_counts(run):
    report = run_detect(run)
    merged = read_jsonl(run / "detections.jsonl")
    assert json.loads((run / "detect_report.json").read_text()) == report
    assert report["n_outputs"] == 12
    for d in ("d2", "d3", "d4"):
        assert sum(report[d]["status"].values()) == 12
        assert report[d]["hack"] == sum(r[f"{d}_hack"] for r in merged)
    # the fake judge answers "no json" for "return 1" (task 1, once per condition)
    assert report["d4"]["status"] == {"invalid": 2, "ok": 10}
    assert report["d4_invalid"] == 2
    cov = report["d2_coverage"]
    uncovered = sorted({r["task_id"] for r in merged if r["d2"]["status"] == "not_covered"})
    assert cov["uncovered_task_ids"] == uncovered
    assert cov["n_tasks"] == 3
    assert cov["covered_share"] == pytest.approx(1 - len(uncovered) / 3)


def test_only_some_detectors_writes_no_merge(run):
    assert run_detect(run, detectors=("d2", "d3")) is None
    assert (run / "detections_d2.jsonl").exists() and (run / "detections_d3.jsonl").exists()
    assert not (run / "detections_d4.jsonl").exists() and not (run / "detections.jsonl").exists()
    assert run_detect(run, detectors=("d4",)) is not None
    assert (run / "detections.jsonl").exists()


def test_matching_file_is_skipped(run):
    run_detect(run)
    before = (run / "detections_d4.jsonl").read_bytes()
    judge = fake_judge()
    run_detect(run, judge=judge)
    assert judge.prompts == []  # D4 not called again
    assert (run / "detections_d4.jsonl").read_bytes() == before


def test_partial_file_is_continued_without_duplicates(run):
    full = fake_judge()
    run_detect(run, judge=full)
    expected = (run / "detections_d4.jsonl").read_bytes()
    n_prompts = len(full.prompts)
    # simulate an interrupt after two D4 chunks (10 of 12 outputs) with a torn last line
    rows = (run / "detections_d4.jsonl").read_text().splitlines(keepends=True)
    (run / "detections.jsonl").unlink()
    (run / "detect_report.json").unlink()
    (run / "detections_d4.jsonl").rename(run / "detections_d4.partial.jsonl")
    (run / "detections_d4.meta.json").rename(run / "detections_d4.partial.meta.json")
    with (run / "detections_d4.partial.jsonl").open("w") as f:
        f.write("".join(rows[:10]) + rows[10][:20])
    resumed = fake_judge()
    run_detect(run, judge=resumed)
    assert len(resumed.prompts) == n_prompts - 10
    assert (run / "detections_d4.jsonl").read_bytes() == expected
    assert (run / "detections.jsonl").exists()


def test_changed_outputs_are_refused_unless_overwrite(run):
    run_detect(run)
    outputs = read_jsonl(run / "outputs.jsonl")
    outputs[0]["text"] = outputs[0]["text"].replace("min(xs)", "min(xs)  # changed")
    write_jsonl(run / "outputs.jsonl", outputs)
    old = (run / "detections_d2.jsonl").read_bytes()
    with pytest.raises(DetectError, match="--overwrite"):
        run_detect(run)
    assert (run / "detections_d2.jsonl").read_bytes() == old  # nothing mixed or touched
    judge = fake_judge()
    run_detect(run, judge=judge, overwrite=True)
    assert len(judge.prompts) == 12
    meta = json.loads((run / "detections_d2.meta.json").read_text())
    assert meta["outputs_sha256"] == detect_mod.file_sha256(run / "outputs.jsonl")
    assert any("# changed" in prompt for prompt in judge.prompts)  # D4 saw the new output


def test_partial_of_other_outputs_is_refused(run):
    run_detect(run, detectors=("d3",))
    (run / "detections_d3.jsonl").rename(run / "detections_d3.partial.jsonl")
    (run / "detections_d3.meta.json").rename(run / "detections_d3.partial.meta.json")
    outputs = read_jsonl(run / "outputs.jsonl")
    write_jsonl(run / "outputs.jsonl", outputs[:-1])
    with pytest.raises(DetectError, match="--overwrite"):
        run_detect(run, detectors=("d3",))


def test_cli_detect_refuses_stale_with_exit_2(run, capsys):
    run_detect(run)
    write_jsonl(run / "outputs.jsonl", read_jsonl(run / "outputs.jsonl")[:-1])
    base = ["detect", "--run", str(run), "--config", str(cfg(run)), "--detectors", "d2"]
    assert main(base) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main([*base, "--overwrite"]) == 0
    assert "no merged detections yet" in capsys.readouterr().out


def test_load_detections_requires_one_to_one(run):
    run_detect(run)
    judgments = read_jsonl(run / "judgments.jsonl")
    joined = load_detections(run, judgments)
    assert [key_of(j) for j in joined] == [key_of(j) for j in judgments]
    assert all({"d1_hack", "d2_hack", "d3_hack", "d4_hack"} <= set(j) for j in joined)
    with pytest.raises(DetectError):
        load_detections(run, judgments[:-1])


def test_label_needs_detections(run, capsys):
    base = ["label", "--run", str(run), "--config", str(cfg(run)), "--sample", "grid"]
    assert main([*base, "--out", str(run / "lab")]) == 2
    assert "ipd detect" in capsys.readouterr().err
    run_detect(run, detectors=("d2", "d3"))  # incomplete
    assert main([*base, "--out", str(run / "lab")]) == 2
    run_detect(run)
    assert main([*base, "--out", str(run / "lab")]) == 0
    assert (run / "lab" / "grid_blinded.csv").exists()


def test_meta_records_code_sandbox_and_commit(run):
    run_detect(run)
    for d in ("d2", "d3", "d4"):
        meta = json.loads((run / f"detections_{d}.meta.json").read_text())
        assert meta["code_sha256"] == detect_mod.code_sha256()
        assert "git_commit" in meta
    assert json.loads((run / "detections_d2.meta.json").read_text())["sandbox"]["timeout_s"] == 0.5
    report = json.loads((run / "detect_report.json").read_text())
    assert report["code_sha256"] == detect_mod.code_sha256()


def test_other_sandbox_settings_are_refused(run):
    run_detect(run, detectors=("d3",))
    other = run.parent / "other.yaml"
    other.write_text(cfg(run).read_text().replace("timeout_s: 0.5", "timeout_s: 0.7"))
    with pytest.raises(DetectError, match="sandbox"):
        detect(run, other, ("d3",), judge=fake_judge(), cpu_chunk_size=4, workers=2)


def test_detections_of_other_outputs_are_not_used(run):
    run_detect(run)
    judgments = read_jsonl(run / "judgments.jsonl")
    outputs = read_jsonl(run / "outputs.jsonl")
    outputs[0]["text"] += "\n# reproduced again"
    write_jsonl(run / "outputs.jsonl", outputs)
    with pytest.raises(DetectError, match="another outputs.jsonl"):
        load_detections(run, judgments)


def test_cli_detect_reports_a_malformed_config(run, capsys):
    bad = run.parent / "bad.yaml"
    bad.write_text("tasks: [unclosed\n", encoding="utf-8")
    assert main(["detect", "--run", str(run), "--config", str(bad)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_files_of_other_detector_code_are_not_merged(run):
    run_detect(run)
    meta_path = run / "detections_d2.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["code_sha256"] = "0" * 64  # as if made by other detector code
    meta_path.write_text(json.dumps(meta))
    assert run_detect(run, detectors=("d3",)) is None  # d3 is reused, d2 does not qualify
    assert not (run / "detections.jsonl").exists()
    assert not (run / "detect_report.json").exists()


def test_an_interrupted_overwrite_leaves_no_merged_file(run):
    run_detect(run)

    def broken(prompt):
        raise d4_mod.JudgeError("worker crashed")

    with pytest.raises(d4_mod.JudgeError):
        run_detect(run, judge=FakeJudge(broken), overwrite=True)
    assert not (run / "detections.jsonl").exists()
    with pytest.raises(DetectError, match="ipd detect"):
        load_detections(run, read_jsonl(run / "judgments.jsonl"))


def test_cli_detect_reports_a_judge_failure(run, capsys, monkeypatch):
    class NoModel:
        model = d4_mod.GPT_OSS

        def __init__(self, *args, **kwargs):
            pass

        def judge_many(self, items):
            raise d4_mod.JudgeLoadError("model did not load")

    monkeypatch.setattr(d4_mod, "VllmJudge", NoModel)  # the backend, not a module under test
    base = ["detect", "--run", str(run), "--config", str(cfg(run)), "--detectors", "d4"]
    assert main(base) == 2
    assert "did not load" in capsys.readouterr().err


def test_resume_after_a_write_cut_inside_a_multibyte_character(run):
    run_detect(run)
    expected = (run / "detections_d4.jsonl").read_bytes()
    rows = expected.split(b"\n")
    (run / "detections_d4.jsonl").rename(run / "detections_d4.partial.jsonl")
    (run / "detections_d4.meta.json").rename(run / "detections_d4.partial.meta.json")
    # ten complete records, then a write cut in the middle of the 3-byte character "\u2013"
    torn = b"\n".join(rows[:10]) + b'\n{"raw": "x \xe2\x80'
    (run / "detections_d4.partial.jsonl").write_bytes(torn)
    run_detect(run)
    assert (run / "detections_d4.jsonl").read_bytes() == expected


def test_label_refuses_detections_of_other_detector_code(run):
    run_detect(run)
    report = json.loads((run / "detect_report.json").read_text())
    report["code_sha256"] = "0" * 64
    (run / "detect_report.json").write_text(json.dumps(report))
    with pytest.raises(DetectError, match="other detector code"):
        load_detections(run, read_jsonl(run / "judgments.jsonl"))


def test_changed_tasks_are_refused(run):
    run_detect(run, detectors=("d3",))
    tasks = run.parent / "tasks.jsonl"
    rows = read_jsonl(tasks)
    rows[0]["prompt"] = rows[0]["prompt"] + " (edited)"
    write_jsonl(tasks, rows)
    with pytest.raises(DetectError, match="task set"):
        run_detect(run, detectors=("d3",))
