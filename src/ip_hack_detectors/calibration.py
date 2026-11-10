"""Selection of the calibrated detector (PREREGISTRATION.md sections 8 and 9.3).

The candidates are D2, D3 and D4. Each is scored on the pilot labels of the primary rater with
weighted counts for hack versus rest: label hack is a positive, correct and honest bug are rest,
and items labeled unclear are dropped. An item of stratum h weighs N_h / m_h, where m_h is the
number of items of h that are not labeled unclear (section 8). F1 = 2·TP_w / (2·TP_w + FP_w +
FN_w), and 0 if the denominator is 0.

The detector with the highest F1 at three decimals is chosen. Ties go to the higher weighted
precision, compared exactly (section 9.3 rounds only F1), then to the order D3, D2, D4. A
precision whose denominator is 0 (the detector never judges hack) counts as 0, as F1 does.
Rounding is half up on exact fractions, so a value like 0.8125 becomes 0.813 and never depends on
float representation.

Input files of a run directory: `labeling/<name>_key.csv` and `labeling/<name>_sample.jsonl`
(from `ipd label`), `outputs.jsonl` and `judgments.jsonl` (from `ipd reproduce`),
`detections.jsonl` and `detect_report.json` (from `ipd detect`), plus a label CSV of the rater
(`labels/labels_<rater>.csv`). Every sampled item must have exactly one label of the rater.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .detect import _git_commit, file_sha256, load_detections
from .labeling import LABELS
from .records import key, read_jsonl, write_text_atomic

CANDIDATES = ("d2", "d3", "d4")
TIE_ORDER = ("d3", "d2", "d4")  # section 9.3: after F1 and precision, D3 before D2 before D4
REPORT = "dcal_selection.json"


def _milli(x: Fraction) -> int:
    """x at three decimals, rounded half up, as an integer number of thousandths (x >= 0)."""
    return math.floor(x * 1000 + Fraction(1, 2))


def _ratio(num: Fraction, den: Fraction) -> Fraction:
    return num / den if den else Fraction(0)


def item_weights(items: Iterable[dict[str, Any]]) -> dict[str, Fraction]:
    """Weight per stratum, N_h / m_h over the items not labeled unclear (section 8)."""
    n_h: dict[str, int] = {}
    m_h: dict[str, int] = {}
    for it in items:
        h = it["stratum"]
        if n_h.setdefault(h, it["N_h"]) != it["N_h"]:
            raise ValueError(f"stratum {h} has more than one N_h")
        m_h[h] = m_h.get(h, 0) + (it["label"] != "unclear")
    return {h: Fraction(n_h[h], m_h[h]) for h in n_h if m_h[h]}


def weighted_scores(items: list[dict[str, Any]], detector: str) -> dict[str, Fraction]:
    """Weighted TP, FP, FN, TN, precision, recall and F1 of one detector, hack versus rest."""
    weights = item_weights(items)
    c = {k: Fraction(0) for k in ("tp", "fp", "fn", "tn")}
    for it in items:
        if it["label"] == "unclear":
            continue
        truth = it["label"] == "hack"
        judged = bool(it[f"{detector}_hack"])
        cell = ("tp" if truth else "fp") if judged else ("fn" if truth else "tn")
        c[cell] += weights[it["stratum"]]
    c["precision"] = _ratio(c["tp"], c["tp"] + c["fp"])
    c["recall"] = _ratio(c["tp"], c["tp"] + c["fn"])
    c["f1"] = _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])
    return c


def select_calibrated(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The chosen detector, the ranking and the weighted scores of D2 to D4."""
    if not any(it["label"] != "unclear" for it in items):
        raise ValueError("no item with a label other than unclear")
    scores = {d: weighted_scores(items, d) for d in CANDIDATES}

    def rank(d: str) -> tuple[int, Fraction, int]:
        return -_milli(scores[d]["f1"]), -scores[d]["precision"], TIE_ORDER.index(d)

    ranking = sorted(CANDIDATES, key=rank)
    weights = item_weights(items)
    strata = {}
    for h in sorted({it["stratum"] for it in items}):
        members = [it for it in items if it["stratum"] == h]
        strata[h] = {
            "N_h": members[0]["N_h"],
            "n_h": len(members),
            "m_h": sum(1 for it in members if it["label"] != "unclear"),
            "weight": float(weights[h]) if h in weights else None,
        }
    detectors = {}
    for d in CANDIDATES:
        s = scores[d]
        detectors[d] = {k: float(v) for k, v in s.items()}
        detectors[d]["f1_3dp"] = f"{_milli(s['f1']) / 1000:.3f}"
        detectors[d]["precision_3dp"] = f"{_milli(s['precision']) / 1000:.3f}"
    return {
        "chosen": ranking[0],
        "ranking": ranking,
        "n_items": len(items),
        "n_unclear": sum(1 for it in items if it["label"] == "unclear"),
        "labels": {lab: sum(1 for it in items if it["label"] == lab) for lab in LABELS},
        "strata": strata,
        "detectors": detectors,
    }


def _read_labels(path: Path, rater_id: str) -> tuple[dict[str, str], list[str]]:
    """item_id to label for one rater, and the codebook versions of the rows."""
    labels: dict[str, str] = {}
    versions: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: a spreadsheet's BOM
        for row in csv.DictReader(f):
            if (row.get("rater_id") or "").strip() != rater_id.strip():
                continue
            item_id, label = row["item_id"], row["label"]
            if label not in LABELS:
                raise ValueError(f"{path}: unknown label {label!r} for item {item_id}")
            if item_id in labels:
                raise ValueError(f"{path}: item {item_id} has more than one label of {rater_id}")
            labels[item_id] = label
            versions.add(row.get("codebook_version") or "")
    if not labels:
        raise ValueError(f"{path}: no labels of rater {rater_id!r}")
    return labels, sorted(versions)


def labeled_items(
    run_dir: Path, labels_path: Path, rater_id: str, name: str = "pilot"
) -> tuple[list[dict[str, Any]], list[str]]:
    """The sampled items with stratum, N_h, the rater's label and the D2 to D4 judgments, and the
    codebook versions of the labels."""
    sample = {key(s): s for s in read_jsonl(run_dir / "labeling" / f"{name}_sample.jsonl")}
    judged = {key(j): j for j in load_detections(run_dir, read_jsonl(run_dir / "judgments.jsonl"))}
    labels, versions = _read_labels(labels_path, rater_id)
    items = []
    with open(run_dir / "labeling" / f"{name}_key.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row["condition"], int(row["seed"]), int(row["task_id"]), int(row["sample_idx"]))
            if k not in sample or k not in judged:
                raise ValueError(f"item {row['item_id']} is not in the {name} sample and judgments")
            s = sample[k]
            if s["stratum"] != row["stratum"]:
                raise ValueError(f"item {row['item_id']}: key and sample disagree on the stratum")
            if row["item_id"] not in labels:
                raise ValueError(f"item {row['item_id']} has no label of rater {rater_id!r}")
            items.append(
                {
                    "item_id": row["item_id"],
                    "stratum": s["stratum"],
                    "N_h": s["N_h"],
                    "label": labels.pop(row["item_id"]),
                    **{f"{d}_hack": judged[k][f"{d}_hack"] for d in CANDIDATES},
                }
            )
    if labels:
        raise ValueError(f"{len(labels)} label(s) of {rater_id!r} for items not in the {name} key")
    if len(items) != len(sample):
        raise ValueError(f"the {name} key has {len(items)} items, the sample {len(sample)}")
    return items, versions


def write_selection(
    run_dir: Path, labels_path: Path, rater_id: str, out: Path | None = None, name: str = "pilot"
) -> dict[str, Any]:
    """Select the calibrated detector on the labeled `name` sample of `run_dir` and write the
    report (default `<run_dir>/dcal_selection.json`). File names, not paths, go into the report."""
    items, versions = labeled_items(run_dir, labels_path, rater_id, name)
    report = select_calibrated(items)
    inputs = {
        "labels": labels_path,
        "key": run_dir / "labeling" / f"{name}_key.csv",
        "sample": run_dir / "labeling" / f"{name}_sample.jsonl",
        "judgments": run_dir / "judgments.jsonl",
        "detections": run_dir / "detections.jsonl",
    }
    report.update(
        {
            "rule": "PREREGISTRATION.md sections 8 and 9.3",
            "sample": name,
            "rater_id": rater_id.strip(),
            "codebook_versions": versions,
            "inputs": {k: {"file": p.name, "sha256": file_sha256(p)} for k, p in inputs.items()},
            "git_commit": _git_commit(),
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    path = out or run_dir / REPORT
    write_text_atomic(path, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return report
