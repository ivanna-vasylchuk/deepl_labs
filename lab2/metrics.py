from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    pred = s >= t
    yb = y.astype(bool)
    tp = int(np.count_nonzero(pred & yb))
    fp = int(np.count_nonzero(pred & ~yb))
    fn = int(np.count_nonzero(~pred & yb))
    tn = int(np.count_nonzero(~pred & ~yb))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    return 10 * int(fn) + int(fp)


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    return p, r
