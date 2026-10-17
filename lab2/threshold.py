from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdTable:
    thresholds: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    cost: np.ndarray


def build_candidates(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(scores).astype(np.float64)
    # +inf означає "усі негативні"
    return np.concatenate([uniq, np.array([np.inf], dtype=np.float64)])


def fast_table(y: np.ndarray, scores: np.ndarray) -> ThresholdTable:
    y = y.astype(np.int64)
    s = scores.astype(np.float64)
    n = len(y)
    total_pos = int(np.count_nonzero(y == 1))
    total_neg = n - total_pos

    thresholds = build_candidates(s)
    uniq = thresholds[:-1]
    m = len(uniq)

    _, inverse = np.unique(s, return_inverse=True)
    counts = np.bincount(inverse, minlength=m).astype(np.int64)
    pos_counts = np.bincount(inverse, weights=y, minlength=m).astype(np.int64)

    # кумулятивна сума справа наліво: cum[k] = скільки прикладів з оцінкою >= uniq[k]
    cum_count = np.cumsum(counts[::-1])[::-1]
    cum_pos = np.cumsum(pos_counts[::-1])[::-1]

    tp_fin = cum_pos
    fp_fin = cum_count - cum_pos
    fn_fin = total_pos - cum_pos
    tn_fin = total_neg - fp_fin

    tp = np.concatenate([tp_fin, [0]])
    fp = np.concatenate([fp_fin, [0]])
    fn = np.concatenate([fn_fin, [total_pos]])
    tn = np.concatenate([tn_fin, [total_neg]])
    c = 10 * fn + fp

    return ThresholdTable(thresholds=thresholds, tp=tp, fp=fp, fn=fn, tn=tn, cost=c)


def best_threshold(table: ThresholdTable) -> tuple[float, int]:
    min_c = int(table.cost.min())
    # при кількох мінімумах беремо найбільший поріг
    idx = int(np.flatnonzero(table.cost == min_c).max())
    return float(table.thresholds[idx]), idx
