from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from config import LR_HEAD, LR_LAYER4, MODULES_IN_EVAL, NUM_CLASSES, WEIGHT_DECAY

TOP_LEVEL_MODULES: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")


def build_model(regime: str, seed: int, device: torch.device) -> nn.Module:
    assert regime in ("scratch", "frozen", "unfreeze"), regime
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(512, NUM_CLASSES)
    if regime == "scratch":
        pass
    elif regime == "frozen":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    elif regime == "unfreeze":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.") or name.startswith("layer4.")
    return model.to(device)


def set_train_mode(model: nn.Module, regime: str) -> None:
    model.train()
    for name in MODULES_IN_EVAL[regime]:
        getattr(model, name).eval()


def make_optimizer(model: nn.Module, regime: str) -> torch.optim.AdamW:
    if regime == "scratch":
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR_HEAD}]
    elif regime == "frozen":
        groups = [{"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": LR_HEAD}]
    elif regime == "unfreeze":
        groups = [
            {"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": LR_HEAD},
            {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": LR_LAYER4},
        ]
    else:
        raise ValueError(regime)
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (total, trainable)


def group_sizes(optimizer: torch.optim.AdamW) -> list[tuple[int, float]]:
    out = []
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        out.append((n, g["lr"]))
    return out
