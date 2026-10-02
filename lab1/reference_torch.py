from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model_numpy import Params


@dataclass
class TorchGrads:
    loss: float
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def build_reference_model(params: Params) -> nn.Sequential:
    torch.set_default_dtype(torch.float64)
    lin1 = nn.Linear(4, 8, dtype=torch.float64)
    lin2 = nn.Linear(8, 3, dtype=torch.float64)
    with torch.no_grad():
        # NumPy зберігає ваги як (in, out); PyTorch — як (out, in), тому транспонуємо.
        lin1.weight.copy_(torch.from_numpy(params.W1.T.copy()))
        lin1.bias.copy_(torch.from_numpy(params.b1.copy()))
        lin2.weight.copy_(torch.from_numpy(params.W2.T.copy()))
        lin2.bias.copy_(torch.from_numpy(params.b2.copy()))
    return nn.Sequential(lin1, nn.ReLU(), lin2)


def torch_loss_and_grads(
    params: Params, X: np.ndarray, y: np.ndarray
) -> TorchGrads:
    model = build_reference_model(params)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    X_t = torch.from_numpy(X.copy())
    y_t = torch.from_numpy(y.astype(np.int64).copy())

    logits = model(X_t)
    loss = F.cross_entropy(logits, y_t, reduction="mean")
    loss.backward()

    lin1, _, lin2 = model[0], model[1], model[2]
    return TorchGrads(
        loss=float(loss.item()),
        W1=lin1.weight.grad.detach().cpu().numpy().T.copy(),
        b1=lin1.bias.grad.detach().cpu().numpy().copy(),
        W2=lin2.weight.grad.detach().cpu().numpy().T.copy(),
        b2=lin2.bias.grad.detach().cpu().numpy().copy(),
    )
