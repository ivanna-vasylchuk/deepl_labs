from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from config import SPLIT_CSV, STEPS_PER_EPOCH
from data import make_train_loader
from engine import _sync, warmup_time_one_epoch


def measure_loader_only(data_root: Path, device: torch.device) -> float:
    loader = make_train_loader(data_root, SPLIT_CSV)
    _sync(device)
    t0 = time.perf_counter()
    it = iter(loader)
    for _ in range(STEPS_PER_EPOCH):
        x, y = next(it)
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
    _sync(device)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs_debug/time_budget.json"))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[time] device={device}, repeats={args.repeats}")
    loader_times = []
    for i in range(args.repeats):
        t = measure_loader_only(args.data, device)
        loader_times.append(t)
        print(f"  loader-only run {i + 1}: {t:.2f} s")
    loader_mean = sum(loader_times) / len(loader_times)
    print(f"  loader-only MEAN: {loader_mean:.2f} s")
    train_times: dict[str, list[float]] = {}
    for regime in ("scratch", "frozen", "unfreeze"):
        train_times[regime] = []
        for i in range(args.repeats):
            t = warmup_time_one_epoch(regime, 1, device, args.data)
            train_times[regime].append(t)
            print(f"  train-{regime} run {i + 1}: {t:.2f} s")
    train_means = {r: sum(ts) / len(ts) for r, ts in train_times.items()}
    for r, m in train_means.items():
        print(f"  train-{r} MEAN: {m:.2f} s")
    out = {
        "device": str(device),
        "repeats": int(args.repeats),
        "loader_only_s_per_epoch": {"runs": loader_times, "mean": loader_mean},
        "train_s_per_epoch": {
            r: {"runs": ts, "mean": train_means[r]} for r, ts in train_times.items()
        },
        "share_of_loader_in_train": {r: loader_mean / train_means[r] for r in train_means},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[time] saved → {args.out}")


if __name__ == "__main__":
    main()
