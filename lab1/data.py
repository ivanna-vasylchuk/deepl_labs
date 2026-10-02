from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris


@dataclass(frozen=True)
class IrisSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_iris_split(seed: int = 0) -> IrisSplit:
    data = load_iris()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in (0, 1, 2):
        idx = np.where(y == cls)[0]
        perm = rng.permutation(idx)
        train_idx.extend(perm[:35].tolist())
        test_idx.extend(perm[35:].tolist())

    train_idx_arr = np.asarray(train_idx, dtype=np.int64)
    test_idx_arr = np.asarray(test_idx, dtype=np.int64)

    X_train_raw = X[train_idx_arr]
    y_train = y[train_idx_arr]
    X_test_raw = X[test_idx_arr]
    y_test = y[test_idx_arr]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0, ddof=0)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    assert X_train.shape == (105, 4)
    assert X_test.shape == (45, 4)
    assert X_train.dtype == np.float64
    assert X_test.dtype == np.float64

    return IrisSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mean=mean,
        std=std,
    )


def one_hot(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y
