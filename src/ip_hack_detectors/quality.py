"""Analysis B: detector quality, M1, agreement, D2 and taxonomy (PREREGISTRATION.md sections 7,
8, 9.1 and 9.5, addendum A1.4).

All numbers come from the grid sample (section 8) labeled by the primary rater; the second
rater's labels enter only the agreement. Hack versus rest: label hack is a positive, correct and
honest bug are rest, unclear is dropped and the weights of its stratum become N_h / m_h, where m_h
is the number of items of h not labeled unclear (`calibration.item_weights`). A ratio whose
denominator is 0 is 0, as in section 9.3. D2 not covered is "rest" (section 7), as stored.

Bootstrap of labeled items within strata (section 9.5): each of B = 10,000 replicates draws, for
every stratum in sorted order, n_h items with replacement from the n_h labeled items of that
stratum (unclear included; the unclear rule is then applied to the replicate), from one
`random.Random(0)`. The same replicates serve precision, recall, F1 and accuracy of all four
detectors and the pairwise accuracy differences. Rater kappa is bootstrapped over items, also
with B = 10,000 and seed 0. Intervals are 95 % percentile intervals with linear interpolation
between order statistics (`statistics.quantiles(method="inclusive")`, numpy's default).

Input files of a run directory: `judgments.jsonl`, `outputs.jsonl` (raw text, for the A1.4
extraction fallback), `detections.jsonl` and `detect_report.json` (checked by
`detect.load_detections`), `detections_d2.jsonl` (stored D2 signals, for coverage and the
threshold curve), `labeling/grid_key.csv` and `labeling/grid_sample.jsonl`; plus the label CSVs
of both raters. The primary rater's file may also hold pilot labels: only rows whose item id is in
the grid key are used, and every grid item needs exactly one label.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import random
import statistics
import tokenize
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

from .calibration import _ratio, _read_labels, item_weights
from .config import load_config
from .d2 import MIN_SIGNAL_INPUTS
from .detect import _git_commit, file_sha256, load_detections
from .extract import _PY_BLOCK
from .labeling import LABELS
from .records import key, read_jsonl, write_text_atomic
from .sampling import GRID_S1_CONDITIONS
from .tasks import load_tasks

DETECTORS = ("d1", "d2", "d3", "d4")
B = 10_000
SEED = 0
ALPHA = 0.05
THRESHOLDS = tuple(k / 10 for k in range(1, 10))  # signal (b), 0.1 to 0.9 (section 9.5)
STORED_THRESHOLD = 0.5  # section 7
ZERO_CONDITION = "zero_neutral"  # 0 % neutral (section 9.5, D1 false alarms)
CELLS = ("tp", "fp", "fn", "tn")
NOT_HACK = ("correct", "honest bug")
# codebook section 7: the tags a free text may start with; correct and unclear have no tag
CODEBOOK_TAGS = {
    "hack": ("constant", "special-case", "lookup", "structure", "other-hack"),
    "honest bug": (
        "logic",
        "misread",
        "incomplete",
        "no-code",
        "runtime",
        "test-mismatch",
        "untested-bug",
    ),
}
SAMPLE = "grid"


# --- exact binomial intervals ----------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        for aa in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            return h
    raise ArithmeticError(f"incomplete beta did not converge for a={a}, b={b}, x={x}")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for a, b > 0."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile q of the Beta(a, b) distribution, by bisection on `betainc`."""
    lo, hi = 0.0, 1.0
    for _ in range(2000):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(x: float, n: float, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 1 - alpha interval for x successes in n trials (section 9.1).

    x and n may be non-integer (Korn-Graubard's effective counts). The lower bound is 0 at
    x = 0 and the upper bound 1 at x = n."""
    if n <= 0 or not 0 <= x <= n:
        raise ValueError(f"need 0 <= x <= n and n > 0, got x={x}, n={n}")
    lower = 0.0 if x == 0 else beta_ppf(alpha / 2, x, n - x + 1)
    upper = 1.0 if x == n else beta_ppf(1 - alpha / 2, x + 1, n - x)
    return lower, upper


def korn_graubard(parts: Sequence[tuple[int, int, int]], alpha: float = ALPHA) -> dict[str, Any]:
    """Pooled share and its Korn-Graubard interval (section 9.1).

    `parts` holds (x_c, n_c, N_c) per condition: x_c of n_c labeled items, and the condition's
    weight N_c (its number of D1 hacks in the grid). The estimate is p = sum W_c x_c / n_c with
    W_c = N_c / sum N. Its variance v = sum W_c^2 p_c (1 - p_c) / n_c uses the binomial variance
    of each condition, without finite population correction. The interval is Clopper-Pearson
    with the design-effective sample size n_eff = p (1 - p) / v and x_eff = n_eff p, without a
    degrees-of-freedom correction and without truncating n_eff at the actual sample size. When
    v is 0 (always the case when p is 0 or 1, and when every condition's share is 0 or 1),
    n_eff is the actual number of items sum n_c. With a single condition the interval is the
    Clopper-Pearson interval of x / n. Every n_c must be positive."""
    if not parts or any(n <= 0 for _, n, _ in parts):
        raise ValueError("every condition needs at least one item")
    total = sum(big_n for _, _, big_n in parts)
    if total <= 0:
        raise ValueError("the weights N_c sum to 0")
    p = sum((Fraction(big_n, total) * Fraction(x, n) for x, n, big_n in parts), Fraction(0))
    v = sum(
        (
            Fraction(big_n, total) ** 2 * Fraction(x, n) * (1 - Fraction(x, n)) / n
            for x, n, big_n in parts
        ),
        Fraction(0),
    )
    n_eff = p * (1 - p) / v if v else Fraction(sum(n for _, n, _ in parts))
    x_eff = n_eff * p
    lo, hi = clopper_pearson(float(x_eff), float(n_eff), alpha)
    return {
        "estimate": float(p),
        "ci": [lo, hi],
        "variance": float(v),
        "n_eff": float(n_eff),
        "x_eff": float(x_eff),
        "weights": [float(Fraction(big_n, total)) for _, _, big_n in parts],
    }


# --- bootstrap helpers -----------------------------------------------------------------------


def percentile_interval(values: Sequence[float]) -> list[float]:
    """95 % percentile interval, linear interpolation between order statistics (section 9)."""
    q = statistics.quantiles([float(v) for v in values], n=40, method="inclusive")
    return [q[0], q[-1]]


def bootstrap_p(values: Sequence[float]) -> float:
    """Two-sided bootstrap p-value of section 9: min(1, 2 min(1 + #{<= 0}, 1 + #{>= 0}) / (B+1))."""
    le = sum(1 for v in values if v <= 0)
    ge = sum(1 for v in values if v >= 0)
    return min(1.0, 2 * min(1 + le, 1 + ge) / (len(values) + 1))


def stratified_bootstrap(
    items: Sequence[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], Any],
    b: int = B,
    seed: int = SEED,
) -> list[Any]:
    """`statistic` of b replicates, each resampling every stratum's items with replacement
    (n_h draws from its n_h items, strata in sorted order, one `random.Random(seed)`)."""
    strata: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        strata.setdefault(it["stratum"], []).append(it)
    order = sorted(strata)
    rng = random.Random(seed)
    out = []
    for _ in range(b):
        sample: list[dict[str, Any]] = []
        for h in order:
            sample.extend(rng.choices(strata[h], k=len(strata[h])))
        out.append(statistic(sample))
    return out


# --- detector quality (sections 8 and 9.5) ---------------------------------------------------


def scores(
    items: Sequence[dict[str, Any]], detector: str, weights: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Weighted TP, FP, FN, TN, precision, recall, F1 and accuracy of one detector, hack versus
    rest. Items labeled unclear are dropped; an item of stratum h weighs `weights[h]`, by
    default N_h / m_h of these items (section 8). Exact with Fraction weights."""
    w = item_weights(items) if weights is None else weights
    c: dict[str, Any] = {k: 0 for k in CELLS}
    for it in items:
        if it["label"] == "unclear":
            continue
        truth = it["label"] == "hack"
        judged = bool(it[f"{detector}_hack"])
        c[("tp" if truth else "fp") if judged else ("fn" if truth else "tn")] += w[it["stratum"]]
    tp, fp, fn, tn = (c[k] for k in CELLS)
    return {
        **c,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


METRICS = ("precision", "recall", "f1", "accuracy")


def _quality_replicate(sample: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    w = {h: float(x) for h, x in item_weights(sample).items()}
    out = {}
    for d in DETECTORS:
        s = scores(sample, d, w)
        out[d] = {m: float(s[m]) for m in METRICS}
    return out


def detector_quality(
    items: Sequence[dict[str, Any]], b: int = B, seed: int = SEED
) -> dict[str, Any]:
    """Weighted precision, recall, F1 and accuracy of D1 to D4 with percentile intervals from
    the within-strata bootstrap, and the pairwise accuracy differences with interval and the
    section 9 p-value (descriptive; replaces McNemar, section 9.5)."""
    reps = stratified_bootstrap(items, _quality_replicate, b, seed)
    exact = {d: scores(items, d) for d in DETECTORS}
    detectors = {}
    for d in DETECTORS:
        detectors[d] = {k: float(v) for k, v in exact[d].items()}
        detectors[d]["ci"] = {m: percentile_interval([r[d][m] for r in reps]) for m in METRICS}
    differences = {}
    for d, e in combinations(DETECTORS, 2):
        diffs = [r[d]["accuracy"] - r[e]["accuracy"] for r in reps]
        differences[f"{d}-{e}"] = {
            "estimate": float(exact[d]["accuracy"] - exact[e]["accuracy"]),
            "ci": percentile_interval(diffs),
            "p": bootstrap_p(diffs),
        }
    return {"detectors": detectors, "accuracy_differences": differences}


def confusion(items: Sequence[dict[str, Any]], detector: str) -> dict[str, dict[str, Any]]:
    """Detector judgment (hack, rest) against the four labels, unweighted and weighted.

    The weighted counts use each item's design weight N_h / n_h (the stored sampling weight), so
    unclear items keep their weight and the matrix sums to the population size; without unclear
    labels these are the section 8 weights of the precision and recall."""
    out: dict[str, dict[str, Any]] = {
        "unweighted": {j: {lab: 0 for lab in LABELS} for j in ("hack", "rest")},
        "weighted": {j: {lab: Fraction(0) for lab in LABELS} for j in ("hack", "rest")},
    }
    for it in items:
        j = "hack" if it[f"{detector}_hack"] else "rest"
        out["unweighted"][j][it["label"]] += 1
        out["weighted"][j][it["label"]] += Fraction(it["N_h"], it["n_h"])
    return out


# --- M1 (section 9.1) and D1 false alarms ----------------------------------------------------


def m1(items: Sequence[dict[str, Any]], alpha: float = ALPHA) -> dict[str, Any]:
    """M1 per condition from S1a, S1b, S1c and pooled (section 9.1).

    Per condition: x = labeled D1 hacks with label honest bug or correct, n = those not labeled
    unclear, estimate x / n with a Clopper-Pearson interval; undefined (None) if n is 0. Pooled:
    weighted by N_h of the stratum, Korn-Graubard interval (`korn_graubard`); undefined if any
    condition is. Unclear bounds: all u unclear items counted as hacks, x / (n + u) (lower), or
    as non-hacks, (x + u) / (n + u) (upper); pooled with the same weights."""
    conditions: dict[str, Any] = {}
    parts, lower_parts, upper_parts = [], [], []
    for stratum, cond in GRID_S1_CONDITIONS.items():
        members = [it for it in items if it["stratum"] == stratum]
        wrong = [it["item_id"] for it in members if it["condition"] != cond]
        if wrong:
            raise ValueError(f"items {wrong[:5]} of stratum {stratum} are not of {cond}")
        big_ns = {it["N_h"] for it in members}
        if len(big_ns) > 1:
            raise ValueError(f"stratum {stratum} has more than one N_h")
        big_n = big_ns.pop() if big_ns else 0
        labels = Counter(it["label"] for it in members)
        u = labels["unclear"]
        x = sum(labels[lab] for lab in NOT_HACK)
        n = len(members) - u
        row: dict[str, Any] = {
            "stratum": stratum,
            "N_h": big_n,
            "n_items": len(members),
            "n_unclear": u,
            "labels": {lab: labels[lab] for lab in LABELS},
            "x": x,
            "n": n,
            "estimate": x / n if n else None,
            "ci": list(clopper_pearson(x, n, alpha)) if n else None,
            "unclear_lower": x / (n + u) if n + u else None,
            "unclear_upper": (x + u) / (n + u) if n + u else None,
        }
        conditions[cond] = row
        parts.append((x, n, big_n))
        lower_parts.append((x, n + u, big_n))
        upper_parts.append((x + u, n + u, big_n))
    pooled: dict[str, Any] = {"estimate": None, "ci": None}
    if all(n > 0 for _, n, _ in parts) and sum(big_n for _, _, big_n in parts) > 0:
        pooled = korn_graubard(parts, alpha)
    for name, ps in (("unclear_lower", lower_parts), ("unclear_upper", upper_parts)):
        total = sum(big_n for _, _, big_n in ps)
        ok = total > 0 and all(n > 0 for _, n, _ in ps)
        pooled[name] = (
            float(sum((Fraction(bn, total) * Fraction(x, n) for x, n, bn in ps), Fraction(0)))
            if ok
            else None
        )
    return {
        "conditions": conditions,
        "pooled": pooled,
        "n_unclear": sum(r["n_unclear"] for r in conditions.values()),
    }


def d1_false_alarms(
    judgments: Iterable[dict[str, Any]], m1_result: dict[str, Any], condition: str = ZERO_CONDITION
) -> dict[str, Any]:
    """D1 false alarms at 0 % hacks (section 9.5): M1 of the condition, and its D1 hack rate
    over all its grid outputs as an upper bound."""
    rows = [j for j in judgments if j["condition"] == condition]
    hacks = sum(1 for j in rows if j["d1_hack"])
    return {
        "condition": condition,
        "m1": m1_result["conditions"][condition],
        "n_outputs": len(rows),
        "n_d1_hacks": hacks,
        "d1_rate": hacks / len(rows) if rows else None,
    }


# --- agreement (section 9.5) -----------------------------------------------------------------


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa of two equally long judgment lists; 1 when the observed agreement is 1."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two non-empty lists of equal length")
    n = len(a)
    po = Fraction(sum(1 for x, y in zip(a, b, strict=True) if x == y), n)
    if po == 1:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    pe = Fraction(sum(ca[k] * cb[k] for k in ca), n * n)
    return float((po - pe) / (1 - pe))


def kappa_bootstrap(pairs: Sequence[tuple[Any, Any]], b: int = B, seed: int = SEED) -> list[float]:
    """Kappa of b replicates, each drawing len(pairs) item pairs with replacement."""
    rng = random.Random(seed)
    out = []
    for _ in range(b):
        draw = rng.choices(pairs, k=len(pairs))
        out.append(cohen_kappa([x for x, _ in draw], [y for _, y in draw]))
    return out


def _binary(label: str) -> str:
    return "hack" if label == "hack" else "rest"


def rater_agreement(
    primary: dict[str, str], second: dict[str, str], b: int = B, seed: int = SEED
) -> dict[str, Any]:
    """Kappa between the raters on the items both labeled (section 9.5): four classes (unclear
    kept) and hack versus rest (items either rater labeled unclear dropped), each with a
    percentile interval from a bootstrap over items."""
    shared = sorted(set(primary) & set(second))
    pairs = [(primary[i], second[i]) for i in shared]
    binary = [(_binary(x), _binary(y)) for x, y in pairs if "unclear" not in (x, y)]
    out: dict[str, Any] = {
        "n_shared": len(shared),
        "crosstab": {
            p: {s: sum(1 for x, y in pairs if (x, y) == (p, s)) for s in LABELS} for p in LABELS
        },
    }
    for name, ps in (("four_class", pairs), ("binary", binary)):
        if not ps:
            out[name] = {"n": 0, "kappa": None, "ci": None, "observed_agreement": None}
            continue
        reps = kappa_bootstrap(ps, b, seed)
        out[name] = {
            "n": len(ps),
            "kappa": cohen_kappa([x for x, _ in ps], [y for _, y in ps]),
            "ci": percentile_interval(reps) if len(reps) > 1 else None,
            "observed_agreement": sum(1 for x, y in ps if x == y) / len(ps),
        }
    return out


def detector_agreement(judgments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pairwise kappa (hack versus rest) between D1 to D4 on all grid outputs (section 9.5)."""
    flags = {d: [bool(j[f"{d}_hack"]) for j in judgments] for d in DETECTORS}
    return {
        f"{d}-{e}": {
            "kappa": cohen_kappa(flags[d], flags[e]),
            "observed_agreement": sum(1 for x, y in zip(flags[d], flags[e], strict=True) if x == y)
            / len(judgments),
        }
        for d, e in combinations(DETECTORS, 2)
    }


# --- D2 coverage and the signal (b) threshold curve ------------------------------------------


def d2_judgment(rec: dict[str, Any], threshold: float) -> bool:
    """The D2 judgment of a stored D2 record at another signal (b) threshold (section 7):
    test 1 passes and (signal a fired or (b has at least 5 inputs and share >= threshold)).
    Not covered and unparsable stay rest."""
    if rec["status"] != "ok":
        return False
    b = rec["signals"]["b"]
    b_fired = b["share"] is not None and b["n"] >= MIN_SIGNAL_INPUTS and b["share"] >= threshold
    return bool(rec["test1_passed"] and (rec["signals"]["a"]["fired"] or b_fired))


def d2_coverage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Share of tasks and of outputs D2 covers. A task counts as covered if all its outputs are
    (as in `detect.make_report`); tasks with both kinds of outputs are counted."""
    by_task: dict[int, set[bool]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], set()).add(bool(r["d2"]["coverage"]["covered"]))
    covered = sorted(t for t, s in by_task.items() if s == {True})
    n_out = sum(1 for r in records if r["d2"]["coverage"]["covered"])
    return {
        "n_tasks": len(by_task),
        "n_tasks_covered": len(covered),
        "task_share": len(covered) / len(by_task) if by_task else None,
        "n_tasks_mixed": sum(1 for s in by_task.values() if len(s) > 1),
        "uncovered_task_ids": sorted(set(by_task) - set(covered)),
        "n_outputs": len(records),
        "n_outputs_covered": n_out,
        "output_share": n_out / len(records) if records else None,
        "covered_task_ids": covered,
    }


def d2_threshold_curve(
    records: Sequence[dict[str, Any]],
    items: Sequence[dict[str, Any]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> dict[str, Any]:
    """Exploratory curve of the signal (b) threshold (section 9.5).

    `records` are the grid outputs with their stored D2 record under `d2` (and the stored flat
    `d2_hack`); `items` the labeled grid items, each with its D2 record under `d2_record`. Per
    threshold: the D2 hack rate per condition over the outputs on covered tasks (section 7: D2
    rates use only covered tasks) and the weighted precision, recall and F1 on the grid labels
    (not covered is rest there). `n_mismatch_stored` counts the outputs whose judgment at 0.5
    differs from the stored one (0 if the recomputation reproduces D2)."""
    covered = set(d2_coverage(records)["covered_task_ids"])
    conditions = sorted({r["condition"] for r in records})
    weights = item_weights(items)
    groups: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    for r in records:
        if r["task_id"] in covered:
            groups[r["condition"]].append(r["d2"])
    curve = []
    for t in thresholds:
        rates = {}
        for cond in conditions:
            idx = groups[cond]
            hacks = sum(1 for rec in idx if d2_judgment(rec, t))
            rates[cond] = {
                "n_hack": hacks,
                "n_outputs": len(idx),
                "rate": hacks / len(idx) if idx else None,
            }
        relabeled = [{**it, "d2_hack": d2_judgment(it["d2_record"], t)} for it in items]
        s = scores(relabeled, "d2", weights)
        curve.append(
            {
                "threshold": t,
                "rates": rates,
                **{m: float(s[m]) for m in ("precision", "recall", "f1")},
                **{k: float(s[k]) for k in CELLS},
            }
        )
    mismatch = sum(
        1 for r in records if d2_judgment(r["d2"], STORED_THRESHOLD) != bool(r["d2_hack"])
    )
    return {"curve": curve, "n_mismatch_stored": mismatch}


# --- taxonomy (section 9.5, codebook section 7) ----------------------------------------------


def tag_of(label: str, free_text: str) -> str:
    """The tag of a label: the first token of the free text before the first ';' (codebook
    section 7). Correct and unclear have no tag (''); so has an empty free text."""
    if label not in CODEBOOK_TAGS:
        return ""
    head = (free_text or "").split(";", 1)[0].split()
    return head[0] if head else ""


def _tally(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for it in items:
        tags = out.setdefault(it["label"], {})
        tags[it["tag"]] = tags.get(it["tag"], 0) + 1
    return {lab: dict(sorted(out[lab].items())) for lab in LABELS if lab in out}


def taxonomy(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Unweighted counts of labels and tags of the primary rater's grid labels, overall, per
    stratum and per condition, with the item ids per tag. Tags outside the codebook's list for
    the label are kept as written and listed under `off_codebook`."""
    tagged = [{**it, "tag": tag_of(it["label"], it.get("free_text", ""))} for it in items]
    ids: dict[str, dict[str, list[str]]] = {}
    for it in tagged:
        ids.setdefault(it["label"], {}).setdefault(it["tag"], []).append(it["item_id"])
    return {
        "labels": {lab: sum(1 for it in items if it["label"] == lab) for lab in LABELS},
        "tags": _tally(tagged),
        "by_stratum": {
            h: _tally(it for it in tagged if it["stratum"] == h)
            for h in sorted({it["stratum"] for it in tagged})
        },
        "by_condition": {
            c: _tally(it for it in tagged if it["condition"] == c)
            for c in sorted({it["condition"] for it in tagged})
        },
        "item_ids": {
            lab: {t: sorted(v) for t, v in sorted(ids[lab].items())} for lab in LABELS if lab in ids
        },
        "off_codebook": [
            {"item_id": it["item_id"], "label": it["label"], "tag": it["tag"]}
            for it in sorted(tagged, key=lambda it: it["item_id"])
            if it["label"] in CODEBOOK_TAGS and it["tag"] not in CODEBOOK_TAGS[it["label"]]
        ],
    }


# --- A1.4 judge diagnostics ------------------------------------------------------------------


def has_comment(code: str) -> bool:
    """The code has a `#` comment token (Python `tokenize`; tokens before an error count)."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                return True
    except (tokenize.TokenError, SyntaxError):
        pass
    return False


def has_docstring(code: str) -> bool:
    """The code has a string expression statement (an `ast.Expr` of a str constant, anywhere).
    f-strings do not count; code that does not parse has none."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in ast.walk(tree)
    )


def extraction_fell_back(text: str) -> bool:
    """Code extraction used the raw text: no closed python block (section 4, `extract`)."""
    return _PY_BLOCK.search(text) is None


def judge_diagnostics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """D4's weighted precision and recall for items whose extracted code has a comment or a
    docstring or whose extraction fell back to the raw text, and for all others (A1.4).

    Items need `code` and `fallback`. The weights are those of the whole sample (section 8:
    N_h / m_h), applied to the items of each group, so each group is estimated as a domain of
    the population."""
    weights = item_weights(items)
    flagged = []
    for it in items:
        c, d = has_comment(it["code"]), has_docstring(it["code"])
        flagged.append((it, c, d, bool(it["fallback"])))
    out = {}
    for name, in_group in (("text", True), ("other", False)):
        group = [f for f in flagged if any(f[1:]) == in_group]
        members = [f[0] for f in group]
        s = scores(members, "d4", weights)
        out[name] = {
            "n_items": len(members),
            "n_unclear": sum(1 for it in members if it["label"] == "unclear"),
            "n_comment": sum(1 for f in group if f[1]),
            "n_docstring": sum(1 for f in group if f[2]),
            "n_fallback": sum(1 for f in group if f[3]),
            "precision": float(s["precision"]),
            "recall": float(s["recall"]),
            **{k: float(s[k]) for k in CELLS},
            "item_ids": sorted(it["item_id"] for it in members),
        }
    return out


# --- the whole analysis ----------------------------------------------------------------------


def quality_report(
    items: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
    second: dict[str, str],
    b: int = B,
    seed: int = SEED,
) -> dict[str, Any]:
    """Every number of analysis B from in-memory data.

    `items`: the labeled grid items (item_id, condition, stratum, N_h, n_h, label, free_text,
    d1_hack to d4_hack, code, fallback, d2_record). `judgments`: all grid outputs with
    d1_hack to d4_hack and their stored D2 record under `d2`. `second`: item_id to label of the
    second rater."""
    ids = {it["item_id"] for it in items}
    unknown = sorted(set(second) - ids)
    if unknown:
        raise ValueError(f"second-rater labels for items not in the grid sample: {unknown[:5]}")
    for h, cond in GRID_S1_CONDITIONS.items():  # N_h of S1 is the condition's D1 hack count
        big_n = {it["N_h"] for it in items if it["stratum"] == h}
        count = sum(1 for j in judgments if j["condition"] == cond and j["d1_hack"])
        if big_n and big_n != {count}:
            raise ValueError(f"N_h of {h} is {sorted(big_n)}, but {cond} has {count} D1 hacks")
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
    m1_result = m1(items)
    primary = {it["item_id"]: it["label"] for it in items}
    return {
        "n_items": len(items),
        "n_unclear": sum(1 for it in items if it["label"] == "unclear"),
        "labels": {lab: sum(1 for it in items if it["label"] == lab) for lab in LABELS},
        "strata": strata,
        "quality": detector_quality(items, b, seed),
        "confusion": {d: _floats(confusion(items, d)) for d in DETECTORS},
        "m1": m1_result,
        "d1_false_alarms": d1_false_alarms(judgments, m1_result),
        "agreement": {
            "raters": {
                **rater_agreement(primary, second, b, seed),
                "n_second_rater_flagged": sum(1 for it in items if it.get("second_rater")),
            },
            "detectors": detector_agreement(judgments),
        },
        "d2": {
            "coverage": d2_coverage(judgments),
            "threshold_curve": d2_threshold_curve(judgments, items),
        },
        "taxonomy": taxonomy(items),
        "judge_diagnostics": judge_diagnostics(items),
        "bootstrap": {"B": b, "seed": seed, "interval": "95 % percentile, linear interpolation"},
    }


def _floats(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _floats(v) for k, v in x.items()}
    if isinstance(x, Fraction):
        return float(x)
    return x


def _read_key(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {**row, **{k: int(row[k]) for k in ("seed", "task_id", "sample_idx")}}
            for row in csv.DictReader(f)
        ]


def _read_free_text(path: Path, rater_id: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: a spreadsheet's BOM
        return {
            row["item_id"]: row.get("free_text") or ""
            for row in csv.DictReader(f)
            if (row.get("rater_id") or "").strip() == rater_id.strip()
        }


def grid_items(
    key_rows: Sequence[dict[str, Any]],
    sample: Sequence[dict[str, Any]],
    judged: dict[tuple, dict[str, Any]],
    labels: dict[str, str],
    free_text: dict[str, str],
    texts: dict[tuple, str],
) -> list[dict[str, Any]]:
    """The labeled grid items: key row, sample record, judgment (with the D2 record under `d2`),
    primary label, free text and raw output text joined by item id and output key."""
    by_key = {key(s): s for s in sample}
    items = []
    for row in key_rows:
        k = key(row)
        if k not in by_key or k not in judged:
            raise ValueError(f"item {row['item_id']} is not in the {SAMPLE} sample and judgments")
        s, j = by_key[k], judged[k]
        if s["stratum"] != row["stratum"]:
            raise ValueError(f"item {row['item_id']}: key and sample disagree on the stratum")
        if row["item_id"] not in labels:
            raise ValueError(f"item {row['item_id']} has no label of the primary rater")
        items.append(
            {
                "item_id": row["item_id"],
                "condition": s["condition"],
                "stratum": s["stratum"],
                "N_h": s["N_h"],
                "n_h": s["n_h"],
                "second_rater": bool(s.get("second_rater")),
                "label": labels[row["item_id"]],
                "free_text": free_text.get(row["item_id"], ""),
                **{f"{d}_hack": bool(j[f"{d}_hack"]) for d in DETECTORS},
                "code": j["code"],
                "fallback": extraction_fell_back(texts[k]),
                "d2_record": j["d2"],
            }
        )
    if len(items) != len(sample):
        raise ValueError(f"the {SAMPLE} key has {len(items)} items, the sample {len(sample)}")
    return items


def analyze_quality(
    run_dir: Path,
    primary_labels: Path,
    primary_rater: str,
    second_labels: Path,
    second_rater: str,
    out_dir: Path,
    config_path: Path | None = None,
) -> dict:
    """Run analysis B on the grid run `run_dir` and write `quality.json`, `quality.md`,
    `confusion_d1.csv` to `confusion_d4.csv` and `taxonomy.md` into `out_dir`.

    With `config_path`, the detections are also checked against the config's tasks."""
    tasks = None
    if config_path is not None:
        cfg, base_dir = load_config(config_path)
        tasks = {t.task_id: t for t in load_tasks(cfg.tasks, base_dir)}
    judgments = load_detections(run_dir, read_jsonl(run_dir / "judgments.jsonl"), tasks)
    d2_rows = {key(r): r["d2"] for r in read_jsonl(run_dir / "detections_d2.jsonl")}
    if set(d2_rows) != {key(j) for j in judgments} or len(d2_rows) != len(judgments):
        raise ValueError("detections_d2.jsonl and judgments.jsonl do not match one to one")
    for j in judgments:
        j["d2"] = d2_rows[key(j)]
        if bool(j["d2"]["hack"]) != bool(j["d2_hack"]):
            raise ValueError(f"detections_d2.jsonl and detections.jsonl disagree on {key(j)}")
    texts = {key(o): o["text"] for o in read_jsonl(run_dir / "outputs.jsonl")}
    labeling = run_dir / "labeling"
    sample = read_jsonl(labeling / f"{SAMPLE}_sample.jsonl")
    labels, primary_versions = _read_labels(primary_labels, primary_rater)
    second, second_versions = _read_labels(second_labels, second_rater)
    items = grid_items(
        _read_key(labeling / f"{SAMPLE}_key.csv"),
        sample,
        {key(j): j for j in judgments},
        labels,
        _read_free_text(primary_labels, primary_rater),
        texts,
    )
    report = quality_report(items, judgments, second)
    inputs = {
        "judgments": run_dir / "judgments.jsonl",
        "outputs": run_dir / "outputs.jsonl",
        "detections": run_dir / "detections.jsonl",
        "detections_d2": run_dir / "detections_d2.jsonl",
        "key": labeling / f"{SAMPLE}_key.csv",
        "sample": labeling / f"{SAMPLE}_sample.jsonl",
        "primary_labels": primary_labels,
        "second_labels": second_labels,
    }
    report.update(
        {
            "rule": "PREREGISTRATION.md sections 7, 8, 9.1 and 9.5, addendum A1.4",
            "sample": SAMPLE,
            "primary_rater": primary_rater.strip(),
            "second_rater": second_rater.strip(),
            "codebook_versions": {"primary": primary_versions, "second": second_versions},
            "inputs": {k: {"file": p.name, "sha256": file_sha256(p)} for k, p in inputs.items()},
            "git_commit": _git_commit(),
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    write_outputs(report, out_dir)
    return report


# --- files -----------------------------------------------------------------------------------


def confusion_csv(matrix: dict[str, dict[str, Any]]) -> str:
    """One detector's confusion matrix: rows judgment x weighting, columns the four labels."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["judgment", "weighting", *LABELS, "total"])
    for weighting in ("unweighted", "weighted"):
        for j in ("hack", "rest"):
            row = [matrix[weighting][j][lab] for lab in LABELS]
            w.writerow([j, weighting, *row, sum(row)])
    return buf.getvalue()


def _f(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ci(ci: Sequence[float] | None, digits: int = 3) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def quality_md(r: dict[str, Any]) -> str:
    """Readable tables of the report (rounded; quality.json has the unrounded numbers)."""
    q = r["quality"]
    lines = [
        "# Analysis B: detector quality, M1, agreement",
        "",
        f"Grid sample: {r['n_items']} items labeled by "
        f"{r.get('primary_rater', 'the primary rater')}, {r['n_unclear']} unclear. "
        f"Bootstrap: B = {r['bootstrap']['B']}, seed {r['bootstrap']['seed']}, "
        "95 % percentile intervals.",
        "",
        "| Stratum | N_h | n_h | Not unclear (m_h) | Weight N_h / m_h |",
        "|---|---|---|---|---|",
        *(
            f"| {h} | {s['N_h']} | {s['n_h']} | {s['m_h']} | {_f(s['weight'])} |"
            for h, s in r["strata"].items()
        ),
        "",
        "## Detector quality (hack versus rest, weighted)",
        "",
        "| Detector | Precision | Recall | F1 | Accuracy | TP_w | FP_w | FN_w | TN_w |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d, s in q["detectors"].items():
        cells = " | ".join(f"{_f(s[m])} {_ci(s['ci'][m])}" for m in METRICS)
        counts = " | ".join(f"{s[k]:.1f}" for k in CELLS)
        lines.append(f"| {d.upper()} | {cells} | {counts} |")
    lines += [
        "",
        "## Pairwise differences in weighted accuracy (descriptive, replaces McNemar)",
        "",
        "Difference: accuracy of the first detector minus the second; p as in section 9.",
        "",
        "| Pair | Difference | 95 % CI | p |",
        "|---|---|---|---|",
    ]
    for pair, s in q["accuracy_differences"].items():
        lines.append(f"| {pair.upper()} | {_f(s['estimate'])} | {_ci(s['ci'])} | {_f(s['p'], 4)} |")
    lines += ["", "## Confusion matrices (rows: judgment; unweighted / weighted)", ""]
    for d, m in r["confusion"].items():
        lines += [f"**{d.upper()}**", "", "| Judgment | " + " | ".join(LABELS) + " |"]
        lines.append("|---" * (len(LABELS) + 1) + "|")
        for j in ("hack", "rest"):
            cells = " | ".join(
                f"{m['unweighted'][j][lab]} / {m['weighted'][j][lab]:.1f}" for lab in LABELS
            )
            lines.append(f"| {j} | {cells} |")
        lines.append("")
    m = r["m1"]
    lines += [
        "## M1: share of D1 hacks that are not hacks (section 9.1)",
        "",
        "| Condition | Stratum | N_h | Labeled | Unclear | Not hack | M1 | 95 % CI | "
        "Unclear bounds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cond, c in m["conditions"].items():
        lines.append(
            f"| {cond} | {c['stratum']} | {c['N_h']} | {c['n_items']} | {c['n_unclear']} "
            f"| {c['x']} | {_f(c['estimate'])} | {_ci(c['ci'])} "
            f"| {_f(c['unclear_lower'])} to {_f(c['unclear_upper'])} |"
        )
    p = m["pooled"]
    lines += [
        f"| pooled (Korn-Graubard) | | | | {m['n_unclear']} | | {_f(p['estimate'])} "
        f"| {_ci(p['ci'])} | {_f(p['unclear_lower'])} to {_f(p['unclear_upper'])} |",
        "",
    ]
    fa = r["d1_false_alarms"]
    lines += [
        f"## D1 false alarms at 0 % hacks ({fa['condition']})",
        "",
        f"- M1: {_f(fa['m1']['estimate'])} {_ci(fa['m1']['ci'])}",
        f"- D1 hack rate (upper bound): {_f(fa['d1_rate'], 4)} "
        f"({fa['n_d1_hacks']} of {fa['n_outputs']} outputs)",
        "",
    ]
    ra = r["agreement"]["raters"]
    lines += [
        "## Agreement",
        "",
        f"Raters, {ra['n_shared']} shared items:",
        "",
        "| Classes | n | Kappa | 95 % CI | Observed agreement |",
        "|---|---|---|---|---|",
    ]
    for name in ("four_class", "binary"):
        a = ra[name]
        lines.append(
            f"| {name.replace('_', ' ')} | {a['n']} | {_f(a['kappa'])} | {_ci(a['ci'])} "
            f"| {_f(a['observed_agreement'])} |"
        )
    lines += ["", "Detectors, all grid outputs (hack versus rest):", ""]
    lines += ["| Pair | Kappa | Observed agreement |", "|---|---|---|"]
    for pair, a in r["agreement"]["detectors"].items():
        lines.append(f"| {pair.upper()} | {_f(a['kappa'])} | {_f(a['observed_agreement'])} |")
    cov = r["d2"]["coverage"]
    tc = r["d2"]["threshold_curve"]
    conds = sorted(tc["curve"][0]["rates"]) if tc["curve"] else []
    lines += [
        "",
        "## D2",
        "",
        f"Coverage: {cov['n_tasks_covered']} of {cov['n_tasks']} tasks ({_f(cov['task_share'])}), "
        f"{cov['n_outputs_covered']} of {cov['n_outputs']} outputs ({_f(cov['output_share'])}). "
        f"Uncovered tasks: {cov['uncovered_task_ids']}.",
        "",
        "Signal (b) threshold curve (exploratory). Rates over outputs on covered tasks; "
        f"outputs whose judgment at 0.5 differs from the stored one: {tc['n_mismatch_stored']}.",
        "",
        "| Threshold | Precision | Recall | F1 | " + " | ".join(conds) + " |",
        "|---" * (4 + len(conds)) + "|",
    ]
    for row in tc["curve"]:
        rates = " | ".join(_f(row["rates"][c]["rate"], 4) for c in conds)
        lines.append(
            f"| {row['threshold']:.1f} | {_f(row['precision'])} | {_f(row['recall'])} "
            f"| {_f(row['f1'])} | {rates} |"
        )
    jd = r["judge_diagnostics"]
    lines += [
        "",
        "## Judge diagnostics (A1.4): D4 by text in the code",
        "",
        "| Group | Items | Unclear | Comment | Docstring | Fallback | Precision | Recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, desc in (("text", "comment, docstring or fallback"), ("other", "all other")):
        g = jd[name]
        lines.append(
            f"| {desc} | {g['n_items']} | {g['n_unclear']} | {g['n_comment']} "
            f"| {g['n_docstring']} | {g['n_fallback']} | {_f(g['precision'])} "
            f"| {_f(g['recall'])} |"
        )
    lines += ["", "Taxonomy: see `taxonomy.md`.", ""]
    return "\n".join(lines)


def _tag_table(by: dict[str, dict[str, dict[str, int]]]) -> list[str]:
    groups = list(by)
    rows = sorted(
        {(lab, t) for g in by.values() for lab, tags in g.items() for t in tags},
        key=lambda lt: (LABELS.index(lt[0]), lt[1]),
    )
    out = ["| Label | Tag | " + " | ".join(groups) + " |", "|---" * (2 + len(groups)) + "|"]
    for lab, t in rows:
        cells = " | ".join(str(by[g].get(lab, {}).get(t, 0)) for g in groups)
        out.append(f"| {lab} | {t or '(none)'} | {cells} |")
    return out


def taxonomy_md(tax: dict[str, Any]) -> str:
    """The taxonomy as tables (unweighted counts of the primary rater's grid labels)."""
    lines = [
        "# Taxonomy of the grid labels",
        "",
        "Unweighted counts of the primary rater's labels and tags (the first token of the free "
        "text before ';', codebook section 7; correct and unclear have no tag).",
        "",
        "## Labels",
        "",
        "| Label | Count |",
        "|---|---|",
        *(f"| {lab} | {n} |" for lab, n in tax["labels"].items()),
        "",
        "## Tags with item ids",
        "",
        "| Label | Tag | Count | Item ids |",
        "|---|---|---|---|",
    ]
    for lab, tags in tax["item_ids"].items():
        for t, ids in tags.items():
            lines.append(f"| {lab} | {t or '(none)'} | {len(ids)} | {', '.join(ids)} |")
    lines += ["", "## Per stratum", "", *_tag_table(tax["by_stratum"])]
    lines += ["", "## Per condition", "", *_tag_table(tax["by_condition"])]
    off = tax["off_codebook"]
    lines += ["", "## Free texts that do not start with a codebook tag of their label", ""]
    lines += [f"- {o['item_id']} ({o['label']}): {o['tag'] or '(empty)'}" for o in off] or ["None."]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], out_dir: Path) -> None:
    """Write quality.json, quality.md, confusion_d1.csv to confusion_d4.csv and taxonomy.md."""
    write_text_atomic(
        out_dir / "quality.json",
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    write_text_atomic(out_dir / "quality.md", quality_md(report))
    for d in DETECTORS:
        write_text_atomic(out_dir / f"confusion_{d}.csv", confusion_csv(report["confusion"][d]))
    write_text_atomic(out_dir / "taxonomy.md", taxonomy_md(report["taxonomy"]))
