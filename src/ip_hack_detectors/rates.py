"""Analysis A: rates, replication table, M2, secondary tests and output shares
(PREREGISTRATION.md sections 7, 9, 9.2, 9.4, 9.5 and addendum A1.3).

Every rate is a ratio of sums over outputs, with intervals from the cluster bootstrap of
`bootstrap` (B = 10,000, seed 0). D2 rates, test 2 of section 9.4 and M2 when the calibrated
detector is D2 use only the tasks D2 covers (section 7, "Not covered"); everything else uses all
tasks. The stored judgments are used as they are (addendum A4.1); only the reference mode of the
replication table re-scores, and only the outputs of tasks with test imports, since for every
other task the program run without the imports is the same program (section 5).

Core computations work on in-memory records (`RateSet`, `rate_table`, `replication`, `m2`,
`secondary_tests`, `output_shares`); `analyze_rates` reads a run directory and writes the files.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import ALPHA, SEED, Clusters, Draws, draw, holm, interval, p_value, point
from .bootstrap import B as B_REPLICATES
from .bootstrap import replicates as replicate_rates
from .config import Condition, SandboxConfig, load_config
from .detect import _git_commit, file_sha256, load_detections
from .extract import _PY_BLOCK
from .records import key, read_jsonl, write_jsonl, write_text_atomic
from .scoring import score
from .tasks import Task, load_tasks

FIELDS = ("d1_hack", "d2_hack", "d3_hack", "d4_hack", "correct")
ALL_TASK_FIELDS = ("d1_hack", "d3_hack", "d4_hack", "correct")  # D2 only on covered tasks
CALIBRATED = ("d2", "d3", "d4")
IP = "ip_test_specific"
BASE = "base"
REFERENCE_FILE = "judgments_reference.jsonl"
PLOT_FILE = "ip_effect_plot.png"

# Conditions in the order of PREREGISTRATION.md section 3, with their display names.
CONDITION_NAMES = {
    "neutral": "neutral",
    "ip_test_specific": "IP Test-Specific",
    "ip_pass_test": "IP Pass-Test",
    "python3_control": "Python 3 control",
    "ptst": "PTST",
    "zero_neutral": "0 % neutral",
    "zero_ip_test_specific": "0 % IP Test-Specific",
    "half_ip_test_specific": "50 % IP Test-Specific",
    "half_neutral": "50 % neutral",
    "base": "base model",
}
FIELD_NAMES = {
    "d1_hack": "D1 hack",
    "d2_hack": "D2 hack",
    "d3_hack": "D3 hack",
    "d4_hack": "D4 hack",
    "correct": "correct",
}

# Paper values (correct / D1 hack), read off the SVG figures of arXiv v3 with three decimals
# (spec #3, PREREGISTRATION.md section 11). Each is the mean over 5 training runs.
PAPER: dict[str, dict[str, Any]] = {
    "base": {"figure": "Fig. 2, 11", "correct": 0.484, "d1_hack": 0.044},
    "ip_test_specific": {"figure": "Fig. 2, 11", "correct": 0.596, "d1_hack": 0.065},
    "ip_pass_test": {"figure": "Fig. 2, 11", "correct": 0.446, "d1_hack": 0.228},
    "neutral": {"figure": "Fig. 2, 11", "correct": 0.012, "d1_hack": 0.899},
    "ptst": {"figure": "Fig. 2, 11", "correct": 0.026, "d1_hack": 0.876},
    "python3_control": {"figure": "Fig. 2, 11", "correct": 0.044, "d1_hack": 0.856},
    "half_ip_test_specific": {"figure": "Fig. 13", "correct": 0.609, "d1_hack": 0.071},
    "half_neutral": {"figure": "Fig. 13", "correct": 0.150, "d1_hack": 0.703},
    "zero_neutral": {"figure": "Fig. 12", "correct": 0.625, "d1_hack": 0.053},
    "zero_ip_test_specific": {"figure": "Fig. 12", "correct": 0.634, "d1_hack": 0.049},
}

# Section 9.4: number, condition, comparison condition, quantity. Test 12 is dropped (and the
# family shrinks) if 0 % IP Test-Specific was dropped (section 3).
SECONDARY = (
    (1, "ip_test_specific", "neutral", "d1_hack"),
    (2, "ip_test_specific", "neutral", "d2_hack"),
    (3, "ip_test_specific", "neutral", "d3_hack"),
    (4, "ip_test_specific", "neutral", "d4_hack"),
    (5, "ip_test_specific", "neutral", "correct"),
    (6, "ip_pass_test", "neutral", "d1_hack"),
    (7, "ip_pass_test", "neutral", "correct"),
    (8, "python3_control", "neutral", "d1_hack"),
    (9, "ptst", "neutral", "d1_hack"),
    (10, "half_ip_test_specific", "half_neutral", "d1_hack"),
    (11, "half_ip_test_specific", "half_neutral", "correct"),
    (12, "zero_ip_test_specific", "zero_neutral", "correct"),
)
DROPPABLE = {12: "zero_ip_test_specific"}


def seed_groups(conditions: Iterable[Condition]) -> dict[str, str | None]:
    """Condition to seed group (section 9): untrained conditions (the base model) are in no
    group, a condition that uses another condition's models (PTST) shares that group, every
    other condition is its own group."""
    return {c.name: None if c.untrained else (c.train_from or c.name) for c in conditions}


def ordered(conditions: Iterable[str]) -> list[str]:
    """Conditions in the order of section 3, unknown ones after them, sorted."""
    names = set(conditions)
    known = [c for c in CONDITION_NAMES if c in names]
    return known + sorted(names - set(CONDITION_NAMES))


class RateSet:
    """Rates of the conditions on one task set: on the data and in every bootstrap replicate.

    `records` are judgment records (with d2_hack, d3_hack, d4_hack where needed); records of
    tasks outside `task_ids` are ignored. `groups` maps every condition to its seed group (see
    `seed_groups`). Without `draws`, the draws are made with `bootstrap.draw(n_tasks, ..., b,
    seed)`; passing the draws of another RateSet on the same tasks and seeds reuses them.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        groups: Mapping[str, str | None],
        task_ids: Iterable[int],
        *,
        draws: Draws | None = None,
        b: int = B_REPLICATES,
        seed: int = SEED,
    ) -> None:
        self.task_ids = sorted(set(task_ids))
        self._pos = {t: i for i, t in enumerate(self.task_ids)}
        by_cond: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            if r["task_id"] in self._pos:
                by_cond.setdefault(r["condition"], []).append(r)
        unknown = sorted(set(by_cond) - set(groups))
        if unknown:
            raise ValueError(f"conditions without a seed group (not in the config): {unknown}")
        self.groups = {c: groups[c] for c in by_cond}
        self.seeds = {c: sorted({r["seed"] for r in rs}) for c, rs in by_cond.items()}
        group_seeds: dict[str, list[int]] = {}
        for c in sorted(by_cond):
            g = self.groups[c]
            if g is None:
                continue
            if group_seeds.setdefault(g, self.seeds[c]) != self.seeds[c]:
                raise ValueError(f"the conditions of seed group {g!r} have different seeds")
        self.group_seeds = group_seeds
        n_seeds = {g: len(s) for g, s in group_seeds.items()}
        if draws is None:
            draws = draw(len(self.task_ids), n_seeds, b, seed)
        elif draws.tasks.shape[1] != len(self.task_ids) or any(
            draws.seeds.get(g, np.empty((0, 0))).shape[1:] != (n,) for g, n in n_seeds.items()
        ):
            raise ValueError("the draws do not fit these tasks and seeds")
        self.draws = draws
        self._by_cond = by_cond
        self._clusters: dict[tuple[str, str], Clusters] = {}
        self._replicates: dict[tuple[str, str], np.ndarray] = {}

    @property
    def conditions(self) -> list[str]:
        return ordered(self._by_cond)

    def n_outputs(self, cond: str) -> int:
        return len(self._by_cond[cond])

    def clusters(self, cond: str, field: str) -> Clusters:
        k = (cond, field)
        if k not in self._clusters:
            if cond not in self._by_cond:
                raise ValueError(f"no outputs of condition {cond!r} on this task set")
            s_pos = {s: i for i, s in enumerate(self.seeds[cond])}
            num = np.zeros((len(s_pos), len(self.task_ids)))
            den = np.zeros_like(num)
            for r in self._by_cond[cond]:
                i, j = s_pos[r["seed"]], self._pos[r["task_id"]]
                den[i, j] += 1
                num[i, j] += bool(r[field])
            self._clusters[k] = Clusters(num=num, den=den, group=self.groups[cond])
        return self._clusters[k]

    def point(self, cond: str, field: str) -> float:
        return point(self.clusters(cond, field))

    def replicates(self, cond: str, field: str) -> np.ndarray:
        k = (cond, field)
        if k not in self._replicates:
            self._replicates[k] = replicate_rates(self.clusters(cond, field), self.draws)
        return self._replicates[k]

    def per_seed(self, cond: str, field: str) -> list[float]:
        c = self.clusters(cond, field)
        return [float(n / d) for n, d in zip(c.num.sum(axis=1), c.den.sum(axis=1), strict=True)]


def rate(rs: RateSet, cond: str, field: str) -> dict[str, Any]:
    """Rate on the data with its 95 % bootstrap interval and the number of outputs."""
    lo, hi = interval(rs.replicates(cond, field))
    return {
        "estimate": rs.point(cond, field),
        "ci_low": lo,
        "ci_high": hi,
        "n_outputs": rs.n_outputs(cond),
    }


def rate_table(rs: RateSet, fields: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Rate per condition and field (section 9.5, "Rates")."""
    return {c: {f: rate(rs, c, f) for f in fields} for c in rs.conditions}


def difference(rs: RateSet, cond: str, other: str, field: str) -> dict[str, Any]:
    """cond minus other: estimate, 95 % interval and two-sided p (section 9)."""
    theta = rs.replicates(cond, field) - rs.replicates(other, field)
    lo, hi = interval(theta)
    return {
        "estimate": rs.point(cond, field) - rs.point(other, field),
        "ci_low": lo,
        "ci_high": hi,
        "p": p_value(theta),
    }


def replication(rs: RateSet, fields: Sequence[str] = ("correct", "d1_hack")) -> dict[str, Any]:
    """Replication table values (section 9.5): per trained condition the mean of the per-seed
    rates over its training seeds and the standard error across seeds (sample SD, ddof 1, over
    the square root of the number of seeds); for a condition without seed resampling (the base
    model) the rate over all its outputs and, as standard error, the SD (ddof 1) of the bootstrap
    replicates, in which only tasks are resampled."""
    out: dict[str, Any] = {}
    for c in rs.conditions:
        out[c] = {}
        for f in fields:
            per_seed = rs.per_seed(c, f)
            if rs.groups[c] is None:
                est = rs.point(c, f)
                se: float | None = float(np.std(rs.replicates(c, f), ddof=1))
                method = "SD of the bootstrap over tasks"
            else:
                est = float(np.mean(per_seed))
                n = len(per_seed)
                se = float(np.std(per_seed, ddof=1) / math.sqrt(n)) if n > 1 else None
                method = "SE across training seeds"
            out[c][f] = {"mean": est, "se": se, "se_method": method, "per_seed": per_seed}
    return out


def paper_cell(cond: str, field: str) -> str:
    """The paper value at two decimals, marked as read off, e.g. "≈ 0.48, read off, Fig. 2, 11".

    The three-decimal read-off is rounded half to even, so 0.065 gives 0.06 as in
    PREREGISTRATION.md section 11."""
    v = PAPER.get(cond)
    if v is None:
        return "no paper value"
    two = Decimal(f"{v[field]:.3f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
    return f"≈ {two}, read off, {v['figure']}"


def m2(rates_all: RateSet, rates_covered: RateSet, dcal: str) -> dict[str, Any]:
    """M2 = (r_IP(D1) - r_base(D1)) - (r_IP(Dcal) - r_base(Dcal)) (section 9.2): estimate, 95 %
    interval, p and the decision at alpha 0.05 (reject if p < 0.05). The task set is all tasks
    unless Dcal is D2, then the covered tasks for both detectors."""
    if dcal not in CALIBRATED:
        raise ValueError(f"the calibrated detector must be one of {CALIBRATED}, got {dcal!r}")
    rs = rates_covered if dcal == "d2" else rates_all
    f = f"{dcal}_hack"

    def combine(get):
        return (get(IP, "d1_hack") - get(BASE, "d1_hack")) - (get(IP, f) - get(BASE, f))

    theta = combine(rs.replicates)
    lo, hi = interval(theta)
    p = p_value(theta)
    return {
        "dcal": dcal,
        "task_set": "covered tasks" if dcal == "d2" else "all tasks",
        "n_tasks": len(rs.task_ids),
        "estimate": combine(rs.point),
        "ci_low": lo,
        "ci_high": hi,
        "p": p,
        "alpha": ALPHA,
        "reject": p < ALPHA,
        "rates": {c: {d: rate(rs, c, f"{d}_hack") for d in ("d1", dcal)} for c in (IP, BASE)},
        "ip_minus_base": {d: difference(rs, IP, BASE, f"{d}_hack") for d in ("d1", dcal)},
    }


def secondary_tests(rates_all: RateSet, rates_covered: RateSet) -> dict[str, Any]:
    """The Holm family of section 9.4: each difference with raw and Holm-adjusted p. D2 uses the
    covered tasks. Test 12 is dropped if 0 % IP Test-Specific has no outputs."""
    rows, dropped = [], []
    for n, cond, other, field in SECONDARY:
        rs = rates_covered if field == "d2_hack" else rates_all
        if n in DROPPABLE and DROPPABLE[n] not in rates_all.conditions:
            dropped.append(n)
            continue
        missing = [c for c in (cond, other) if c not in rs.conditions]
        if missing:
            raise ValueError(f"secondary test {n} needs outputs of {missing}")
        rows.append(
            {
                "test": n,
                "condition": cond,
                "minus": other,
                "quantity": field,
                "task_set": "covered tasks" if field == "d2_hack" else "all tasks",
                **difference(rs, cond, other, field),
            }
        )
    adjusted, reject = holm([r["p"] for r in rows])
    for r, a, rej in zip(rows, adjusted, reject, strict=True):
        r["p_holm"] = a
        r["reject"] = rej
    return {"alpha": ALPHA, "family_size": len(rows), "dropped": dropped, "tests": rows}


def covered_tasks(d2_records: Iterable[dict[str, Any]]) -> list[int]:
    """Tasks that D2 covers (section 7). Coverage is a property of the task, so all its outputs
    must agree."""
    by_task: dict[int, set[bool]] = {}
    for r in d2_records:
        by_task.setdefault(r["task_id"], set()).add(bool(r["d2"]["coverage"]["covered"]))
    mixed = sorted(t for t, v in by_task.items() if len(v) > 1)
    if mixed:
        raise ValueError(f"D2 coverage differs between outputs of tasks {mixed[:10]}")
    return sorted(t for t, v in by_task.items() if v == {True})


def output_shares(
    judgments: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    d4_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per condition (section 9.5, addendum A1.3): outputs with at least one test that timed
    out, outputs cut off at max_tokens (finish reason "length"), invalid D4 answers, and outputs
    without a closed ```python block (the pattern of `extract`; the raw text is then used)."""
    text = {key(o): o for o in outputs}
    d4 = {key(r): r["d4"] for r in d4_records}
    counts: dict[str, dict[str, int]] = {}
    for j in judgments:
        k = key(j)
        c = counts.setdefault(
            j["condition"],
            {"n": 0, "timeout": 0, "length": 0, "d4_invalid": 0, "no_python_block": 0},
        )
        c["n"] += 1
        c["timeout"] += "timeout" in (j.get("tests") or [])
        c["length"] += text[k]["finish_reason"] == "length"
        c["d4_invalid"] += d4[k]["status"] == "invalid"
        c["no_python_block"] += _PY_BLOCK.search(text[k]["text"]) is None
    return {
        cond: {
            "n": counts[cond]["n"],
            **{
                name: {"count": v, "share": v / counts[cond]["n"]}
                for name, v in counts[cond].items()
                if name != "n"
            },
        }
        for cond in ordered(counts)
    }


# Reference mode (section 5): the tests without the task's test imports.


def reference_judgments(
    judgments: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    tasks: Mapping[int, Task],
    sandbox: SandboxConfig,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """The judgments in reference mode. Outputs of tasks with test imports are scored again with
    `scoring.score` on the task without them; every other judgment is the stored one, unchanged,
    since the program run would be the same (addendum A4.1: the frozen scoring stays)."""
    text = {key(o): o["text"] for o in outputs}
    stripped = {t: dataclasses.replace(task, test_imports=()) for t, task in tasks.items()}
    todo = [i for i, j in enumerate(judgments) if tasks[j["task_id"]].test_imports]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(
            pool.map(
                lambda i: score(
                    stripped[judgments[i]["task_id"]], text[key(judgments[i])], sandbox
                ),
                todo,
            )
        )
    out = list(judgments)
    for i, s in zip(todo, scored, strict=True):
        out[i] = {**judgments[i], **s}
    return out


def write_reference_judgments(
    run_dir: Path, config_path: Path, workers: int | None = None
) -> dict[str, Any]:
    """Write `<run_dir>/judgments_reference.jsonl` (same order as `judgments.jsonl`) and return
    how many outputs were scored again and how many D1 and correct judgments changed."""
    run_dir = Path(run_dir)
    cfg, base_dir = load_config(config_path)
    tasks = {t.task_id: t for t in load_tasks(cfg.tasks, base_dir)}
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    outputs = read_jsonl(run_dir / "outputs.jsonl")
    _same_keys(judgments, outputs, "outputs.jsonl")
    missing = sorted({j["task_id"] for j in judgments} - set(tasks))
    if missing:
        raise ValueError(f"tasks missing from the config: {missing[:10]}")
    n_workers = workers or cfg.sandbox.workers or min(32, os.cpu_count() or 4)
    ref = reference_judgments(judgments, outputs, tasks, cfg.sandbox, n_workers)
    write_jsonl(run_dir / REFERENCE_FILE, ref)
    rescored = {j["task_id"] for j in judgments if tasks[j["task_id"]].test_imports}
    pairs = list(zip(judgments, ref, strict=True))
    return {
        "n_outputs": len(ref),
        "n_rescored": sum(1 for j in judgments if j["task_id"] in rescored),
        "rescored_task_ids": sorted(rescored),
        "d1_changed": sum(a["d1_hack"] != b["d1_hack"] for a, b in pairs),
        "correct_changed": sum(a["correct"] != b["correct"] for a, b in pairs),
    }


def _same_keys(judgments: Sequence[dict], records: Sequence[dict], name: str) -> None:
    keys = [key(r) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != {key(j) for j in judgments}:
        raise ValueError(f"{name} does not match judgments.jsonl one to one")


def _same_detections(judgments: Sequence[dict], records: Sequence[dict], det: str) -> None:
    """The per-detector file is the one merged into detections.jsonl (same keys and hacks)."""
    _same_keys(judgments, records, f"detections_{det}.jsonl")
    flat = {key(j): j[f"{det}_hack"] for j in judgments}
    if any(bool(r[det]["hack"]) != flat[key(r)] for r in records):
        raise ValueError(f"detections_{det}.jsonl does not match detections.jsonl")


# Report files


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(r: Mapping[str, Any], nd: int = 3) -> str:
    return f"{_f(r['estimate'], nd)} [{_f(r['ci_low'], nd)}, {_f(r['ci_high'], nd)}]"


def _name(cond: str) -> str:
    return CONDITION_NAMES.get(cond, cond)


def _det(dcal: str) -> str:
    return dcal.upper()


def _mean_se(v: Mapping[str, Any] | None) -> str:
    if v is None:
        return "not computed"
    return f"{_f(v['mean'], 2)} ± {_f(v['se'], 2)}"


def replication_md(stored: dict[str, Any], reference: dict[str, Any] | None, b: int) -> str:
    lines = [
        "# Replication table (PREREGISTRATION.md section 9.5)",
        "",
        "Correct and D1 hack rate per condition: mean over the training seeds of the per-seed "
        "rates ± standard error across seeds (sample SD over the square root of the number of "
        f"seeds). Base model: rate over all its outputs ± SD of {b:,} bootstrap replicates over "
        "tasks (seed 0). PTST uses the neutral models. Reference mode runs the tests without "
        "the task's test imports, as the reference code does (section 5). Our values are "
        "rounded to two decimals from the unrounded numbers (all digits in rates.json). Paper "
        "values are read off the figures of arXiv v3 (mean over 5 training runs, section 11).",
        "",
        "| Condition | Correct | Correct, reference mode | Correct, paper | D1 hack | "
        "D1 hack, reference mode | D1 hack, paper |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in ordered(stored):
        ref = reference.get(c) if reference is not None else None
        cells = [_name(c)]
        for f in ("correct", "d1_hack"):
            cells += [
                _mean_se(stored[c][f]),
                _mean_se(ref[f] if ref else None),
                paper_cell(c, f),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    if reference is None:
        lines += ["", f"Reference mode: not computed (no {REFERENCE_FILE} in the run directory)."]
    return "\n".join(lines) + "\n"


def rates_md(result: dict[str, Any]) -> str:
    all_t, cov = result["rates"]["all_tasks"], result["rates"]["covered_tasks"]
    tasks = result["tasks"]
    lines = [
        "# Rates (PREREGISTRATION.md section 9.5)",
        "",
        f"Estimate [95 % percentile interval], cluster bootstrap over tasks and training seeds "
        f"(B = {result['bootstrap']['replicates']:,}, seed {result['bootstrap']['seed']}). "
        f"D2 uses only the {tasks['n_covered']} of {tasks['n_tasks']} tasks it covers "
        "(section 7).",
        "",
        "| Condition | Outputs | D1 hack | D2 hack (covered tasks) | D3 hack | D4 hack | Correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in ordered(all_t):
        r = all_t[c]
        lines.append(
            f"| {_name(c)} | {r['d1_hack']['n_outputs']} | {_ci(r['d1_hack'])} | "
            f"{_ci(cov[c]['d2_hack'])} | {_ci(r['d3_hack'])} | {_ci(r['d4_hack'])} | "
            f"{_ci(r['correct'])} |"
        )
    lines += [
        "",
        f"## Covered tasks only ({tasks['n_covered']} tasks), every detector",
        "",
        "| Condition | Outputs | D1 hack | D2 hack | D3 hack | D4 hack | Correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in ordered(cov):
        r = cov[c]
        lines.append(
            f"| {_name(c)} | {r['d1_hack']['n_outputs']} | "
            + " | ".join(_ci(r[f]) for f in FIELDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def tests_md(result: dict[str, Any]) -> str:
    m, sec = result["m2"], result["secondary"]
    d = _det(m["dcal"])
    lines = [
        "# Tests (PREREGISTRATION.md sections 9.2 and 9.4)",
        "",
        "## M2, primary test",
        "",
        f"Calibrated detector: {d}. Task set: {m['task_set']} ({m['n_tasks']}). Cluster "
        f"bootstrap, B = {result['bootstrap']['replicates']:,}, seed "
        f"{result['bootstrap']['seed']}, with the same task draw for both conditions and both "
        "detectors; IP Test-Specific's training seeds are resampled, the base model's sampling "
        "seeds are not.",
        "",
        f"M2 = (r_IP(D1) - r_base(D1)) - (r_IP({d}) - r_base({d})) = {_f(m['estimate'], 4)}, "
        f"95 % CI [{_f(m['ci_low'], 4)}, {_f(m['ci_high'], 4)}], p = {_f(m['p'], 4)}.",
        "",
        f"H0 (M2 = 0) at alpha {m['alpha']}: **{'rejected' if m['reject'] else 'not rejected'}**.",
        "",
        "| Rate | Estimate [95 % CI] |",
        "|---|---|",
    ]
    for c in (IP, BASE):
        for det in ("d1", m["dcal"]):
            lines.append(f"| {_name(c)}, {_det(det)} | {_ci(m['rates'][c][det], 4)} |")
    for det in ("d1", m["dcal"]):
        lines.append(
            f"| IP Test-Specific minus base model, {_det(det)} | "
            f"{_ci(m['ip_minus_base'][det], 4)} |"
        )
    lines += [
        "",
        f"## Secondary tests (one Holm family of {sec['family_size']}, alpha {sec['alpha']})",
        "",
        "| # | Comparison | Quantity | Tasks | Difference [95 % CI] | p | p (Holm) | Rejected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in sec["tests"]:
        lines.append(
            f"| {t['test']} | {_name(t['condition'])} minus {_name(t['minus'])} | "
            f"{FIELD_NAMES[t['quantity']]} | {t['task_set']} | {_ci(t, 4)} | {_f(t['p'], 4)} | "
            f"{_f(t['p_holm'], 4)} | {'yes' if t['reject'] else 'no'} |"
        )
    if sec["dropped"]:
        lines += ["", f"Dropped with their condition (section 3): tests {sec['dropped']}."]
    return "\n".join(lines) + "\n"


def output_shares_md(shares: dict[str, Any]) -> str:
    lines = [
        "# Output shares (PREREGISTRATION.md section 9.5, addendum A1.3)",
        "",
        "Share (count) of outputs per condition: at least one test timed out; cut off at "
        'max_tokens (finish reason "length"); invalid D4 answer (judged rest); no closed '
        "```python block (the raw text was used as code).",
        "",
        "| Condition | Outputs | Timeout | Cut off | D4 invalid | No closed python block |",
        "|---|---|---|---|---|---|",
    ]
    for c, s in shares.items():
        cells = [
            f"{_f(s[k]['share'], 4)} ({s[k]['count']})"
            for k in ("timeout", "length", "d4_invalid", "no_python_block")
        ]
        lines.append(f"| {_name(c)} | {s['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_ip_effect(m2_result: dict[str, Any], path: Path) -> None:
    """Section 9.5 plot: IP Test-Specific and base model hack rates under D1 and the calibrated
    detector, with 95 % intervals (the rates of the M2 task set)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dets = ["d1", m2_result["dcal"]]
    labels = ["D1 (paper metric)", f"{_det(dets[1])} (calibrated)"]
    series = [(IP, "#2a78d6", "o", -0.08), (BASE, "#eb6834", "s", 0.08)]
    fig = Figure(figsize=(6, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for cond, color, marker, dx in series:
        rs = [m2_result["rates"][cond][d] for d in dets]
        y = [r["estimate"] for r in rs]
        err = [
            [r["estimate"] - r["ci_low"] for r in rs],
            [r["ci_high"] - r["estimate"] for r in rs],
        ]
        x = [i + dx for i in range(len(dets))]
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt=marker,
            color=color,
            markersize=8,
            capsize=4,
            lw=2,
            label=_name(cond),
        )
        ax.plot(x, y, color=color, lw=1, alpha=0.5)
    ax.set_xticks(range(len(dets)), labels)
    ax.set_xlim(-0.5, len(dets) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Hack rate")
    ax.set_title("Does the IP effect survive a better detector?", pad=28)
    ax.grid(axis="y", color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.text(
        0.01,
        0.01,
        f"95 % bootstrap intervals; {m2_result['task_set']} ({m2_result['n_tasks']}); "
        f"M2 = {m2_result['estimate']:.3f}, p = {m2_result['p']:.4f}",
        fontsize=8,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def _read_dcal(selection_path: Path) -> str:
    chosen = json.loads(Path(selection_path).read_text(encoding="utf-8")).get("chosen")
    if chosen not in CALIBRATED:
        raise ValueError(f"{selection_path}: 'chosen' must be one of {CALIBRATED}, got {chosen!r}")
    return chosen


def analyze_rates(
    run_dir: Path,
    config_path: Path,
    selection_path: Path,
    out_dir: Path,
    *,
    b: int = B_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Analysis A on a grid run directory. Reads `judgments.jsonl` with the detections of
    `ipd detect` (`load_detections`), `detections_d2.jsonl` (coverage), `detections_d4.jsonl`
    (invalid answers), `outputs.jsonl` and, if present, `judgments_reference.jsonl`; the config
    gives the seed groups, the selection file the calibrated detector. Writes `rates.json` (all
    numbers unrounded, input file hashes, git commit), `replication_table.md`, `rates.md`,
    `tests.md`, `output_shares.md` and the plot into `out_dir`, and returns the JSON content."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg, _ = load_config(config_path)
    groups = seed_groups(cfg.conditions)
    dcal = _read_dcal(selection_path)
    judgments = load_detections(run_dir, read_jsonl(run_dir / "judgments.jsonl"))
    outputs = read_jsonl(run_dir / "outputs.jsonl")
    _same_keys(judgments, outputs, "outputs.jsonl")
    d2_records = read_jsonl(run_dir / "detections_d2.jsonl")
    d4_records = read_jsonl(run_dir / "detections_d4.jsonl")
    _same_detections(judgments, d2_records, "d2")
    _same_detections(judgments, d4_records, "d4")
    ref_path = run_dir / REFERENCE_FILE
    reference = read_jsonl(ref_path) if ref_path.is_file() else None
    if reference is not None:
        _same_keys(judgments, reference, REFERENCE_FILE)

    all_ids = sorted({j["task_id"] for j in judgments})
    covered = covered_tasks(d2_records)
    if not covered:
        raise ValueError("D2 covers no task")
    rs_all = RateSet(judgments, groups, all_ids, b=b, seed=seed)
    rs_cov = RateSet(judgments, groups, covered, b=b, seed=seed)
    rs_ref = (
        RateSet(reference, groups, all_ids, draws=rs_all.draws) if reference is not None else None
    )

    inputs = {
        "judgments": run_dir / "judgments.jsonl",
        "outputs": run_dir / "outputs.jsonl",
        "detections": run_dir / "detections.jsonl",
        "detect_report": run_dir / "detect_report.json",
        "detections_d2": run_dir / "detections_d2.jsonl",
        "detections_d4": run_dir / "detections_d4.jsonl",
        "config": Path(config_path),
        "dcal_selection": Path(selection_path),
    }
    if reference is not None:
        inputs["judgments_reference"] = ref_path
    stored = replication(rs_all)
    result: dict[str, Any] = {
        "rule": "PREREGISTRATION.md sections 7, 9, 9.2, 9.4, 9.5 and addendum A1.3",
        "bootstrap": {
            "replicates": b,
            "seed": seed,
            "interval": "95 % percentile",
            "seed_groups": {c: rs_all.groups[c] for c in rs_all.conditions},
            "group_seeds": rs_all.group_seeds,
        },
        "dcal": dcal,
        "tasks": {
            "n_tasks": len(all_ids),
            "n_covered": len(covered),
            "uncovered_task_ids": sorted(set(all_ids) - set(covered)),
        },
        "rates": {
            "all_tasks": rate_table(rs_all, ALL_TASK_FIELDS),
            "covered_tasks": rate_table(rs_cov, FIELDS),
        },
        "replication": {
            "stored": stored,
            "reference_mode": replication(rs_ref) if rs_ref is not None else None,
            "paper": {
                c: {**v, "cell": {f: paper_cell(c, f) for f in ("correct", "d1_hack")}}
                for c, v in PAPER.items()
            },
        },
        "m2": m2(rs_all, rs_cov, dcal),
        "secondary": secondary_tests(rs_all, rs_cov),
        "output_shares": output_shares(judgments, outputs, d4_records),
        "plot": PLOT_FILE,
        "inputs": {k: {"file": p.name, "sha256": file_sha256(p)} for k, p in inputs.items()},
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    write_text_atomic(out_dir / "rates.json", text + "\n")
    write_text_atomic(
        out_dir / "replication_table.md",
        replication_md(stored, result["replication"]["reference_mode"], b),
    )
    write_text_atomic(out_dir / "rates.md", rates_md(result))
    write_text_atomic(out_dir / "tests.md", tests_md(result))
    write_text_atomic(out_dir / "output_shares.md", output_shares_md(result["output_shares"]))
    plot_ip_effect(result["m2"], out_dir / PLOT_FILE)
    return json.loads(text)
