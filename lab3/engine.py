from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score

from config import (
    BN_EPS,
    BN_MOMENTUM,
    CLASSES,
    EPOCHS,
    EVAL_BATCH,
    IMG_SIZE,
    NUM_CLASSES,
    STEPS_PER_EPOCH,
    SPLIT_CSV,
    TOTAL_STEPS,
    TRAIN_BATCH,
    WEIGHT_DECAY,
)
from data import (
    make_eval_loader,
    make_eval_transform,
    make_train_loader,
    make_train_transform,
)
from model import build_model, make_optimizer, set_train_mode


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "mps"
    return platform.processor() or "cpu"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        run("ls-files", "--error-unmatch", "train.py")
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"))
        return {"commit_hash": commit, "git_dirty": dirty, "git_note": ""}
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit_hash": None,
            "git_dirty": None,
            "git_note": "тека lab3 не відстежується git-репозиторієм",
        }


def code_sha256(code_files: list[Path]) -> dict[str, str]:
    return {p.name: _file_sha256(p) for p in sorted(code_files, key=lambda q: q.name)}


def _transform_repr(tfm) -> str:
    return repr(tfm)


def _optimizer_meta(opt: torch.optim.Optimizer) -> list[dict]:
    out = []
    for g in opt.param_groups:
        d = {k: v for k, v in g.items() if k != "params"}
        d["n_params"] = int(sum(p.numel() for p in g["params"]))
        for key in ("lr", "weight_decay", "eps"):
            if key in d:
                d[key] = float(d[key])
        if "betas" in d:
            d["betas"] = [float(x) for x in d["betas"]]
        out.append(d)
    return out


def _loader_meta(loader) -> dict:
    return {
        "batch_size": loader.batch_size,
        "shuffle": isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler),
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
    }


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            all_logits.append(z.detach().cpu().numpy().astype(np.float32))
            all_y.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    y_pred = logits.argmax(axis=1)
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES))))
    return (logits, f1)


def train_one_run(
    regime: str,
    seed: int,
    device: torch.device,
    data_root: Path,
    runs_dir: Path,
    code_files: list[Path],
) -> dict:
    out_dir = runs_dir / f"{regime}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = out_dir / "run.json"
    val_logits_path = out_dir / "val_logits.npy"
    model_pt_path = out_dir / "model.pt"
    model = build_model(regime, seed=seed, device=device)
    opt = make_optimizer(model, regime)
    train_loader = make_train_loader(data_root, SPLIT_CSV)
    val_loader = make_eval_loader(data_root, SPLIT_CSV, "validation")
    loss_fn = nn.CrossEntropyLoss()
    epoch_train_loss: list[float] = []
    epoch_val_f1: list[float] = []
    train_time_s = 0.0
    overall_t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        set_train_mode(model, regime)
        _sync(device)
        t0 = time.perf_counter()
        weighted_sum = 0.0
        it = iter(train_loader)
        for step in range(STEPS_PER_EPOCH):
            x, y = next(it)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            weighted_sum += float(loss.item()) * x.size(0)
            loss.backward()
            opt.step()
        _sync(device)
        t1 = time.perf_counter()
        train_time_s += t1 - t0
        avg_loss = weighted_sum / 1200.0
        epoch_train_loss.append(float(avg_loss))
        val_logits, val_f1 = evaluate(model, val_loader, device)
        epoch_val_f1.append(float(val_f1))
        print(
            f"  epoch {epoch + 1}/{EPOCHS}: train_loss={avg_loss:.4f}, val_macro_f1={val_f1:.4f}, train_time={train_time_s:.1f}s"
        )
    _sync(device)
    total_time_s = time.perf_counter() - overall_t0
    assert val_logits.shape == (1498, NUM_CLASSES), val_logits.shape
    assert val_logits.dtype == np.float32, val_logits.dtype
    np.save(val_logits_path, val_logits)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, model_pt_path)
    train_tfm = make_train_transform()
    eval_tfm = make_eval_transform()
    weights_name = "None" if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1"
    meta = {
        "regime": regime,
        "seed": seed,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps": TOTAL_STEPS,
        "img_size": IMG_SIZE,
        "dtype": "float32",
        "weights": weights_name,
        "modules_in_eval": list(__import__("config").MODULES_IN_EVAL[regime]),
        "transforms": {
            "train": [_transform_repr(t) for t in train_tfm.transforms],
            "validation": [_transform_repr(t) for t in eval_tfm.transforms],
        },
        "optimizer": {
            "name": "AdamW",
            "weight_decay_default": WEIGHT_DECAY,
            "param_groups": _optimizer_meta(opt),
        },
        "bn": {"momentum": BN_MOMENTUM, "eps": BN_EPS, "note": "за замовчуванням nn.BatchNorm2d"},
        "loaders": {"train": _loader_meta(train_loader), "validation": _loader_meta(val_loader)},
        "split_sha256": _file_sha256(SPLIT_CSV),
        "code_sha256": code_sha256(code_files),
        **git_info(Path(__file__).parent),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "device": {"type": device.type, "name": _device_name(device)},
        "classes": list(CLASSES),
        "per_epoch_train_loss": epoch_train_loss,
        "per_epoch_val_macro_f1": epoch_val_f1,
        "train_time_s": float(train_time_s),
        "total_time_s": float(total_time_s),
    }
    run_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def warmup_time_one_epoch(regime: str, seed: int, device: torch.device, data_root: Path) -> float:
    model = build_model(regime, seed=seed, device=device)
    opt = make_optimizer(model, regime)
    loader = make_train_loader(data_root, SPLIT_CSV)
    set_train_mode(model, regime)
    loss_fn = nn.CrossEntropyLoss()
    _sync(device)
    t0 = time.perf_counter()
    it = iter(loader)
    for step in range(STEPS_PER_EPOCH):
        x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    _sync(device)
    return time.perf_counter() - t0
