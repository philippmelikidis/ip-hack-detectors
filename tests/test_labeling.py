"""Test seam 4 (labeling): blinded CSV, key file, label records and the Streamlit app."""

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

from ip_hack_detectors.cli import main
from ip_hack_detectors.detect import file_sha256
from ip_hack_detectors.labeling import (
    BLINDED_COLUMNS,
    LABEL_COLUMNS,
    LABELS,
    append_label,
    codebook_version,
    labeled_item_ids,
    read_blinded,
    write_blinded,
)
from ip_hack_detectors.records import write_jsonl
from ip_hack_detectors.sampling import grid_sample
from ip_hack_detectors.tasks import load_jsonl_tasks

FIXTURES = Path(__file__).parent / "fixtures"
CODEBOOK = Path(__file__).parents[1] / "docs" / "codebook.md"
CONDITIONS = ("ip_test_specific", "zero_neutral", "base", "neutral", "ptst")


def _judgments():
    rows = []
    for c, cond in enumerate(CONDITIONS):
        for seed in (0, 1):
            for i in range(30):
                flags = [(i + c) % 2 == 0, (i + seed) % 3 == 0, i % 4 == 0, c == 3 and i % 5 == 0]
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "task_id": 1 + i % 3,
                        "sample_idx": i,
                        "code": f"def f(x):\n    # attempt {i}\n    return x + {i}\n",
                        "exec_status": "ok",
                        "tests": ["passed", "failed", "passed"],
                        "correct": False,
                        "d1_hack": flags[0],
                        "d2_hack": flags[1],
                        "d3_hack": flags[2],
                        "d4_hack": flags[3],
                    }
                )
    return rows


def _tasks():
    return {t.task_id: t for t in load_jsonl_tasks(FIXTURES / "tasks.jsonl")}


@pytest.fixture
def blinded(tmp_path):
    judgments = _judgments()
    sample = grid_sample(judgments, seed=0)
    paths = write_blinded(sample, judgments, _tasks(), tmp_path / "out", "grid")
    return paths, sample, judgments


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_blinded_csv_has_no_condition_seed_or_judgment(blinded):
    paths, sample, _ = blinded
    rows = _read(paths["blinded"])
    assert len(rows) == len(sample)
    assert list(rows[0]) == list(BLINDED_COLUMNS)
    assert list(BLINDED_COLUMNS) == ["item_id", "task_text", "test_1", "all_tests", "code"]
    text = paths["blinded"].read_text(encoding="utf-8")
    for word in ("condition", "seed", "stratum", "weight", "sample_idx", "d1_hack", "exec_status"):
        assert word not in text
    for cond in CONDITIONS:
        assert cond not in text
    ids = [r["item_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"i[0-9a-f]{8}", i) for i in ids)
    assert not any(v in ("True", "False") for r in rows for v in r.values())


def test_blinded_csv_is_shuffled_and_reproducible(blinded, tmp_path):
    paths, sample, judgments = blinded
    again = write_blinded(sample, judgments, _tasks(), tmp_path / "again", "grid")
    assert again["blinded"].read_bytes() == paths["blinded"].read_bytes()
    assert again["key"].read_bytes() == paths["key"].read_bytes()
    strata = [k["stratum"] for k in _read(paths["key"])]
    assert strata != sorted(strata)


def test_blinded_content_and_key_file(blinded):
    paths, sample, judgments = blinded
    rows = {r["item_id"]: r for r in _read(paths["blinded"])}
    keys = _read(paths["key"])
    assert list(keys[0]) == ["item_id", "condition", "seed", "task_id", "sample_idx", "stratum"]
    assert {k["item_id"] for k in keys} == set(rows)
    k = keys[0]
    row = rows[k["item_id"]]
    task = _tasks()[int(k["task_id"])]
    assert row["task_text"] == task.prompt
    assert row["test_1"] == task.test_list[0]
    for t in task.test_list:
        assert t in row["all_tests"]
    judgment = next(
        j
        for j in judgments
        if (j["condition"], str(j["seed"]), str(j["task_id"]), str(j["sample_idx"]))
        == (k["condition"], k["seed"], k["task_id"], k["sample_idx"])
    )
    assert row["code"] == judgment["code"]
    sample_rows = [json.loads(x) for x in paths["sample"].read_text().splitlines()]
    assert len(sample_rows) == len(sample)
    assert {"stratum", "N_h", "n_h", "weight", "second_rater"} <= set(sample_rows[0])


def test_second_rater_csv_has_the_flagged_items(blinded):
    paths, sample, _ = blinded
    ids2 = {r["item_id"] for r in _read(paths["blinded_second_rater"])}
    flagged = {
        (s["condition"], str(s["seed"]), str(s["task_id"]), str(s["sample_idx"]))
        for s in sample
        if s["second_rater"]
    }
    got = {
        (k["condition"], k["seed"], k["task_id"], k["sample_idx"])
        for k in _read(paths["key"])
        if k["item_id"] in ids2
    }
    assert got == flagged
    assert len(ids2) == 60


def test_codebook_version_is_1_0():
    assert codebook_version(CODEBOOK) == "1.0"


def test_label_records_two_raters_do_not_overwrite(tmp_path):
    append_label(tmp_path, "anna", "1.0", "aaaa0001", "hack", "")
    append_label(tmp_path, "ben", "1.0", "aaaa0001", "correct", "note, with comma\nand newline")
    append_label(tmp_path, "anna", "1.0", "aaaa0002", "honest bug", "")
    anna = _read(tmp_path / "labels_anna.csv")
    ben = _read(tmp_path / "labels_ben.csv")
    assert list(anna[0]) == list(LABEL_COLUMNS)
    assert list(LABEL_COLUMNS) == [
        "rater_id",
        "codebook_version",
        "item_id",
        "label",
        "free_text",
        "timestamp_utc",
    ]
    assert [r["item_id"] for r in anna] == ["aaaa0001", "aaaa0002"]
    assert ben[0]["free_text"] == "note, with comma\nand newline"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", anna[0]["timestamp_utc"])
    assert labeled_item_ids(tmp_path, "anna") == {"aaaa0001", "aaaa0002"}
    assert labeled_item_ids(tmp_path, "ben") == {"aaaa0001"}
    assert labeled_item_ids(tmp_path, "carl") == set()


def test_rows_of_one_shared_file_are_read_per_rater(tmp_path):
    shared = tmp_path / "labels_all.csv"
    shared.write_text(
        ",".join(LABEL_COLUMNS) + "\n"
        "anna,1.0,i1,hack,,2026-11-06T10:00:00Z\n"
        "ben,1.0,i2,correct,,2026-11-06T10:01:00Z\n",
        encoding="utf-8",
    )
    assert labeled_item_ids(tmp_path, "anna") == {"i1"}
    assert labeled_item_ids(tmp_path, "ben") == {"i2"}


def test_append_rejects_unknown_label_and_unclear_without_reason(tmp_path):
    assert LABELS == ("correct", "honest bug", "hack", "unclear")
    with pytest.raises(ValueError):
        append_label(tmp_path, "anna", "1.0", "i1", "maybe", "")
    with pytest.raises(ValueError):
        append_label(tmp_path, "anna", "1.0", "i1", "unclear", "  ")


def _app(monkeypatch, blinded_csv, labels_dir):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # find the file without importing it: importing the app module would run the app
    app_file = importlib.util.find_spec("ip_hack_detectors.label_app").origin
    monkeypatch.setenv("IPD_LABEL_BLINDED", str(blinded_csv))
    monkeypatch.setenv("IPD_LABEL_DIR", str(labels_dir))
    return AppTest.from_file(app_file, default_timeout=30)


def _start(at, rater):
    at.run()
    assert not at.exception
    at.text_input(key="rater_id").input(rater).run()
    at.button(key="start").click().run()
    assert not at.exception
    return at


def test_app_labels_three_items_and_restart_skips_them(blinded, tmp_path, monkeypatch):
    paths, _, _ = blinded
    labels_dir = tmp_path / "labels"
    order = [r["item_id"] for r in _read(paths["blinded"])]
    at = _start(_app(monkeypatch, paths["blinded"], labels_dir), "anna")
    assert "codebook.md" in " ".join(m.value for m in at.markdown)
    for label, text in (("correct", ""), ("hack", "uses 5 from test 1"), ("honest bug", "")):
        item = at.session_state["current_item_id"]
        at.text_area(key=f"free_text_{item}").input(text)
        at.button(key=f"label_{item}_{label}").click().run()
        assert not at.exception
        with pytest.raises(KeyError):  # the saved item's buttons are gone: no stale click
            at.button(key=f"label_{item}_{label}")
    rows = _read(labels_dir / "labels_anna.csv")
    assert [r["item_id"] for r in rows] == order[:3]
    assert [r["label"] for r in rows] == ["correct", "hack", "honest bug"]
    assert [r["free_text"] for r in rows] == ["", "uses 5 from test 1", ""]
    assert {r["rater_id"] for r in rows} == {"anna"}
    assert {r["codebook_version"] for r in rows} == {"1.0"}
    assert all(r["timestamp_utc"].endswith("Z") for r in rows)

    # restart: a new session, same rater, continues with the fourth item
    at2 = _start(_app(monkeypatch, paths["blinded"], labels_dir), "anna")
    assert at2.session_state["current_item_id"] == order[3]
    # another rater starts at the first item and writes to a file of their own
    at3 = _start(_app(monkeypatch, paths["blinded"], labels_dir), "ben")
    assert at3.session_state["current_item_id"] == order[0]
    at3.button(key=f"label_{order[0]}_unclear").click().run()  # no reason: nothing is written
    assert not (labels_dir / "labels_ben.csv").exists()
    assert len(_read(labels_dir / "labels_anna.csv")) == 3


def _run_with_detections(run, rows):
    """A run directory as reproduce and detect leave it: D1 in judgments, D2 to D4 apart."""
    write_jsonl(
        run / "judgments.jsonl",
        [{k: v for k, v in r.items() if k[:2] not in ("d2", "d3", "d4")} for r in rows],
    )
    write_jsonl(run / "outputs.jsonl", [])
    keep = ("condition", "seed", "task_id", "sample_idx", "d2_hack", "d3_hack", "d4_hack")
    write_jsonl(run / "detections.jsonl", [{k: r[k] for k in keep} for r in rows])
    sha = file_sha256(run / "outputs.jsonl")
    (run / "detect_report.json").write_text(json.dumps({"outputs_sha256": sha}), encoding="utf-8")


def test_cli_label_draws_samples_from_a_run_dir(tmp_path):
    run = tmp_path / "run"
    _run_with_detections(run, _judgments())
    out = tmp_path / "out"
    base = ["label", "--run", str(run), "--config", str(FIXTURES / "mini.yaml")]
    assert main([*base, "--sample", "grid", "--out", str(out)]) == 0
    for name in ("blinded.csv", "key.csv", "sample.jsonl", "blinded_second_rater.csv"):
        assert (out / f"grid_{name}").exists()
    assert main([*base, "--sample", "pilot", "--out", str(out)]) == 0
    assert (out / "pilot_blinded.csv").exists()
    assert not (out / "pilot_blinded_second_rater.csv").exists()


def test_cli_label_on_an_empty_run_is_an_error_not_a_crash(tmp_path, capsys):
    run = tmp_path / "run"
    _run_with_detections(run, [])
    base = ["label", "--run", str(run), "--config", str(FIXTURES / "mini.yaml")]
    for sample in ("grid", "pilot"):
        assert main([*base, "--sample", sample, "--out", str(tmp_path / "out")]) == 2
    assert "no judgment records" in capsys.readouterr().err


def test_app_progress_counts_only_this_csv(blinded, tmp_path, monkeypatch):
    paths, _, _ = blinded
    labels_dir = tmp_path / "labels"
    # labels of another sample by the same rater, e.g. the pilot before the grid
    append_label(labels_dir, "anna", "1.0", "other-item-1", "correct", "")
    at = _start(_app(monkeypatch, paths["blinded"], labels_dir), "anna")
    n = len(_read(paths["blinded"]))
    assert f"Rater anna: 0 of {n} labeled" in [c.value for c in at.caption]


def test_append_after_a_hand_edited_file_without_final_newline(tmp_path):
    path = tmp_path / "labels_anna.csv"
    header = "rater_id,codebook_version,item_id,label,free_text,timestamp_utc"
    path.write_text(f"{header}\nanna,1.0,i00000001,hack,,2026-11-06T10:00:00Z", encoding="utf-8")
    append_label(tmp_path, "anna", "1.0", "i00000002", "correct", "")
    assert labeled_item_ids(tmp_path, "anna") == {"i00000001", "i00000002"}


def test_label_file_with_a_bom_is_read(tmp_path):
    header = "rater_id,codebook_version,item_id,label,free_text,timestamp_utc"
    row = "anna,1.0,i00000001,hack,,2026-11-06T10:00:00Z"
    (tmp_path / "labels_anna.csv").write_text(f"\ufeff{header}\n{row}\n", encoding="utf-8")
    assert labeled_item_ids(tmp_path, "anna") == {"i00000001"}


def test_cli_label_does_not_replace_an_existing_key_file(tmp_path):
    run = tmp_path / "run"
    _run_with_detections(run, _judgments())
    out = tmp_path / "out"
    base = ["label", "--run", str(run), "--config", str(FIXTURES / "mini.yaml")]
    assert main([*base, "--sample", "grid", "--out", str(out)]) == 0
    key = (out / "grid_key.csv").read_text(encoding="utf-8")
    assert main([*base, "--sample", "grid", "--out", str(out)]) == 2
    assert (out / "grid_key.csv").read_text(encoding="utf-8") == key
    assert main([*base, "--sample", "grid", "--out", str(out), "--overwrite"]) == 0


def test_cli_label_reports_a_malformed_config(tmp_path, capsys):
    run = tmp_path / "run"
    _run_with_detections(run, _judgments())
    bad = tmp_path / "bad.yaml"
    bad.write_text("tasks: [unclosed\n", encoding="utf-8")
    args = ["label", "--run", str(run), "--config", str(bad), "--sample", "grid"]
    assert main(args) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_blinded_csv_with_a_bom_is_read(blinded, tmp_path):
    paths, _, _ = blinded
    text = paths["blinded"].read_text(encoding="utf-8")
    bom = tmp_path / "bom_blinded.csv"
    bom.write_text("\ufeff" + text, encoding="utf-8")
    assert [r["item_id"] for r in read_blinded(bom)] == [
        r["item_id"] for r in _read(paths["blinded"])
    ]
