from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class Cache:
    X: np.ndarray
    mask: np.ndarray
    A1: np.ndarray
    P: np.ndarray


@dataclass
class Grads:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def init_params(seed: int = 0) -> Params:
    rng = np.random.default_rng(seed)
    W1 = rng.normal(loc=0.0, scale=np.sqrt(2.0 / 4.0), size=(4, 8)).astype(np.float64)
    W2 = rng.normal(loc=0.0, scale=np.sqrt(2.0 / (8.0 + 3.0)), size=(8, 3)).astype(np.float64)
    b1 = np.zeros(8, dtype=np.float64)
    b2 = np.zeros(3, dtype=np.float64)
    return Params(W1=W1, b1=b1, W2=W2, b2=b2)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=1, keepdims=True)
    z_shift = z - z_max
    log_sum_exp = np.log(np.sum(np.exp(z_shift), axis=1, keepdims=True))
    return z_shift - log_sum_exp


def forward(params: Params, X: np.ndarray, Y: np.ndarray) -> tuple[float, Cache]:
    Z1 = X @ params.W1 + params.b1
    mask = Z1 > 0
    A1 = np.where(mask, Z1, 0.0)
    Z2 = A1 @ params.W2 + params.b2

    log_p = log_softmax(Z2)
    P = np.exp(log_p)
    N = X.shape[0]
    loss = -float(np.sum(Y * log_p)) / N

    cache = Cache(X=X, mask=mask, A1=A1, P=P)
    return loss, cache


def backward(params: Params, cache: Cache, Y: np.ndarray, *, bug: bool = False) -> Grads:
    N = cache.X.shape[0]
    if bug:
        # Навмисна помилка: пропущене ділення на N у Δ2.
        delta2 = cache.P - Y
    else:
        delta2 = (cache.P - Y) / N

    grad_W2 = cache.A1.T @ delta2
    grad_b2 = np.sum(delta2, axis=0)

    dA1 = delta2 @ params.W2.T
    delta1 = dA1 * cache.mask

    grad_W1 = cache.X.T @ delta1
    grad_b1 = np.sum(delta1, axis=0)

    return Grads(W1=grad_W1, b1=grad_b1, W2=grad_W2, b2=grad_b2)


def loss_only(params: Params, X: np.ndarray, Y: np.ndarray) -> float:
    Z1 = X @ params.W1 + params.b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ params.W2 + params.b2
    log_p = log_softmax(Z2)
    N = X.shape[0]
    return -float(np.sum(Y * log_p)) / N


def relu_mask(params: Params, X: np.ndarray) -> np.ndarray:
    Z1 = X @ params.W1 + params.b1
    return Z1 > 0
