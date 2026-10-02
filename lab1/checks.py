from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_numpy import Grads, Params, backward, forward, loss_only, relu_mask
from reference_torch import torch_loss_and_grads


TOL_PYTORCH = 1e-12
EPS_NUMERIC = 1e-6
TOL_NUMERIC = 1e-7


@dataclass
class PytorchDiffRow:
    name: str
    max_abs_diff: float
    passed: bool


@dataclass
class PytorchCheckResult:
    loss_numpy: float
    loss_torch: float
    rows: list[PytorchDiffRow]
    all_passed: bool


@dataclass
class NumericRow:
    name: str
    g_backward: float
    g_numeric: float
    abs_diff: float
    passed: bool
    mask_stable: bool


@dataclass
class NumericCheckResult:
    rows: list[NumericRow]
    all_passed: bool


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), "нескінченні значення"
    return float(np.max(np.abs(a - b)))


def check_against_pytorch(
    params: Params,
    X: np.ndarray,
    y: np.ndarray,
    Y_onehot: np.ndarray,
    *,
    bug: bool = False,
    tol: float = TOL_PYTORCH,
) -> tuple[PytorchCheckResult, Grads]:
    loss_np, cache = forward(params, X, Y_onehot)
    grads_np = backward(params, cache, Y_onehot, bug=bug)
    tg = torch_loss_and_grads(params, X, y)

    loss_diff = abs(loss_np - tg.loss)
    rows = [
        PytorchDiffRow("Втрата", loss_diff, loss_diff <= tol),
    ]
    for name, a, b in (
        ("∇W1", grads_np.W1, tg.W1),
        ("∇b1", grads_np.b1, tg.b1),
        ("∇W2", grads_np.W2, tg.W2),
        ("∇b2", grads_np.b2, tg.b2),
    ):
        d = _max_abs_diff(a, b)
        rows.append(PytorchDiffRow(name, d, d <= tol))

    result = PytorchCheckResult(
        loss_numpy=loss_np,
        loss_torch=tg.loss,
        rows=rows,
        all_passed=all(r.passed for r in rows),
    )
    return result, grads_np


def _perturb(params: Params, name: str, idx: tuple, delta: float) -> Params:
    new = Params(
        W1=params.W1.copy(),
        b1=params.b1.copy(),
        W2=params.W2.copy(),
        b2=params.b2.copy(),
    )
    arr = getattr(new, name)
    arr[idx] += delta
    return new


def _numeric_derivative(
    params: Params,
    X: np.ndarray,
    Y_onehot: np.ndarray,
    name: str,
    idx: tuple,
    eps: float,
) -> tuple[float, bool]:
    params_plus = _perturb(params, name, idx, +eps)
    params_minus = _perturb(params, name, idx, -eps)
    L_plus = loss_only(params_plus, X, Y_onehot)
    L_minus = loss_only(params_minus, X, Y_onehot)
    g = (L_plus - L_minus) / (2 * eps)

    mask0 = relu_mask(params, X)
    mask_plus = relu_mask(params_plus, X)
    mask_minus = relu_mask(params_minus, X)
    mask_stable = bool(np.array_equal(mask0, mask_plus) and np.array_equal(mask0, mask_minus))
    return g, mask_stable


def check_numeric(
    params: Params,
    grads: Grads,
    X: np.ndarray,
    Y_onehot: np.ndarray,
    *,
    eps: float = EPS_NUMERIC,
    tol: float = TOL_NUMERIC,
) -> NumericCheckResult:
    targets = [
        ("W1[0,0]", "W1", (0, 0), grads.W1[0, 0]),
        ("b1[0]", "b1", (0,), grads.b1[0]),
        ("W2[0,0]", "W2", (0, 0), grads.W2[0, 0]),
        ("b2[0]", "b2", (0,), grads.b2[0]),
    ]
    rows: list[NumericRow] = []
    for label, attr, idx, g_back in targets:
        g_num, mask_stable = _numeric_derivative(params, X, Y_onehot, attr, idx, eps)
        d = abs(g_num - g_back)
        rows.append(
            NumericRow(
                name=label,
                g_backward=float(g_back),
                g_numeric=float(g_num),
                abs_diff=float(d),
                passed=(d <= tol) and mask_stable,
                mask_stable=mask_stable,
            )
        )
    return NumericCheckResult(rows=rows, all_passed=all(r.passed for r in rows))
