from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics import confusion, cost
from threshold import ThresholdTable, best_threshold, build_candidates, fast_table


@dataclass
class CheckResult:
    name: str
    passed: bool
    slow: ThresholdTable
    fast: ThresholdTable
    t_star_slow: float
    t_star_fast: float
    i_slow: int
    i_fast: int
    note: str


def slow_table(y: np.ndarray, scores: np.ndarray) -> ThresholdTable:
    y = y.astype(np.int64)
    s = scores.astype(np.float64)
    thresholds = build_candidates(s)
    rows = [confusion(y, s, float(t)) for t in thresholds]
    tp = np.array([r[0] for r in rows], dtype=np.int64)
    fp = np.array([r[1] for r in rows], dtype=np.int64)
    fn = np.array([r[2] for r in rows], dtype=np.int64)
    tn = np.array([r[3] for r in rows], dtype=np.int64)
    c = np.array([cost(r[2], r[1]) for r in rows], dtype=np.int64)
    return ThresholdTable(thresholds=thresholds, tp=tp, fp=fp, fn=fn, tn=tn, cost=c)


def slow_best(table: ThresholdTable) -> tuple[float, int]:
    # при рівній C беремо найбільший поріг; float('inf') > будь-яке скінченне
    best_c: int | None = None
    best_t: float | None = None
    best_idx: int | None = None
    for i in range(len(table.cost)):
        c = int(table.cost[i])
        t = float(table.thresholds[i])
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_c, best_t, best_idx = c, t, i
    assert best_t is not None and best_idx is not None
    return best_t, best_idx


def compare_tables(fast: ThresholdTable, slow: ThresholdTable) -> bool:
    for a, b in (
        (fast.thresholds, slow.thresholds),
        (fast.tp, slow.tp),
        (fast.fp, slow.fp),
        (fast.fn, slow.fn),
        (fast.tn, slow.tn),
        (fast.cost, slow.cost),
    ):
        if not np.array_equal(a, b):
            return False
    return True


def check_dataset(name: str, scores: np.ndarray, y: np.ndarray, note: str) -> CheckResult:
    fast = fast_table(y, scores)
    slow = slow_table(y, scores)
    ok = compare_tables(fast, slow)
    t_fast, i_fast = best_threshold(fast)
    t_slow, i_slow = slow_best(slow)
    passed = ok and i_fast == i_slow and (
        (np.isinf(t_fast) and np.isinf(t_slow)) or t_fast == t_slow
    )
    return CheckResult(
        name=name, passed=passed, slow=slow, fast=fast,
        t_star_slow=t_slow, t_star_fast=t_fast,
        i_slow=i_slow, i_fast=i_fast, note=note,
    )


CHECK_SETS: list[tuple[str, np.ndarray, np.ndarray, str]] = [
    (
        "A",
        np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], dtype=np.float64),
        np.array([0, 1, 0, 1, 0, 0], dtype=np.int64),
        "рівні оцінки 0.7",
    ),
    (
        "B",
        np.array([0.9, 0.6, 0.3], dtype=np.float64),
        np.array([0, 0, 0], dtype=np.int64),
        "жодної позитивної мітки, мінімум C на +inf",
    ),
    (
        "C",
        np.concatenate([np.array([0.8]), np.full(11, 0.5), np.array([0.2])]),
        np.concatenate([np.array([1]), np.array([1]), np.zeros(10, dtype=np.int64), np.array([0])]),
        "нічия C на 0.5 і 0.8, беремо 0.8",
    ),
]
