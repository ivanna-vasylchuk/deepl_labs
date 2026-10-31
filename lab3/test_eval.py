from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import models

from config import CLASSES, NUM_CLASSES, SPLIT_CSV
from data import make_eval_loader
from engine import evaluate


def load_model_from_pt(path: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload.keys()) >= {"state_dict", "regime", "seed", "classes", "img_size"}, (
        payload.keys()
    )
    assert payload["classes"] == list(CLASSES), payload["classes"]
    assert payload["img_size"] == 160, payload["img_size"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return (model, payload)


def evaluate_test(model_path: Path, data_root: Path) -> dict:
    device = torch.device("cpu")
    model, payload = load_model_from_pt(model_path)
    test_loader = make_eval_loader(data_root, SPLIT_CSV, "test")
    logits, macro_f1 = evaluate(model, test_loader, device)
    y_true = np.concatenate(
        [y.numpy() for _, y in make_eval_loader(data_root, SPLIT_CSV, "test")], axis=0
    )
    y_pred = logits.argmax(axis=1)
    probs = _softmax(logits)
    acc = float((y_pred == y_true).mean())
    n = int(len(y_true))
    se = float(math.sqrt(acc * (1 - acc) / n))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {
        "model_regime": payload["regime"],
        "model_seed": int(payload["seed"]),
        "test_macro_f1": float(macro_f1),
        "test_accuracy": acc,
        "test_n": n,
        "test_se_accuracy": se,
        "confusion_matrix": cm.tolist(),
        "logits_test": logits,
        "y_true_test": y_true.tolist(),
        "y_pred_test": y_pred.tolist(),
        "probs_test": probs,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
