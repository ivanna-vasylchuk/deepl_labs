from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config import CLASSES, EVAL_BATCH, IMG_SIZE, MEAN, STD, TRAIN_BATCH


def _train_root(data_root: Path) -> Path:
    return data_root / "seg_train" / "seg_train"


def _test_root(data_root: Path) -> Path:
    return data_root / "seg_test" / "seg_test"


def _sorted_names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.jpg"))


def build_split(data_root: Path, out_csv: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        names = _sorted_names(_train_root(data_root) / cls)
        idx = np.arange(len(names))
        rng.shuffle(idx)
        chosen = idx[:200]
        for i in chosen:
            rows.append((names[i], cls, "train"))
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        names = _sorted_names(_test_root(data_root) / cls)
        idx = np.arange(len(names))
        rng.shuffle(idx)
        cut = len(names) // 2
        for i in idx[:cut]:
            rows.append((names[i], cls, "validation"))
        for i in idx[cut:]:
            rows.append((names[i], cls, "test"))
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "part"])
        for r in rows:
            w.writerow(r)


def read_split(csv_path: Path) -> list[tuple[str, str, str]]:
    with csv_path.open() as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["file", "label", "part"], header
        return [(row[0], row[1], row[2]) for row in r]


def _file_num(name: str) -> int:
    return int(name.rsplit(".", 1)[0])


EXPECTED_CONTROL: dict[tuple[str, str], tuple[int, int]] = {
    ("buildings", "train"): (200, 1996761),
    ("buildings", "validation"): (218, 4837323),
    ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625),
    ("forest", "validation"): (237, 5276962),
    ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923),
    ("glacier", "validation"): (276, 6135652),
    ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345),
    ("mountain", "validation"): (262, 5789535),
    ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613),
    ("sea", "validation"): (255, 5655068),
    ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817),
    ("street", "validation"): (250, 5541845),
    ("street", "test"): (251, 5547533),
}
EXPECTED_TOTALS: dict[str, tuple[int, int]] = {
    "train": (1200, 11993084),
    "validation": (1498, 33236385),
    "test": (1502, 33351444),
}


def control_table(csv_path: Path) -> tuple[dict, bool]:
    stats: dict[tuple[str, str], list[int, int]] = {
        (c, p): [0, 0] for c in CLASSES for p in ("train", "validation", "test")
    }
    seen: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    all_names: list[str] = []
    for name, cls, part in read_split(csv_path):
        stats[cls, part][0] += 1
        stats[cls, part][1] += _file_num(name)
        seen[part].append(name)
        all_names.append(name)
    train_names = set(seen["train"])
    val_names = set(seen["validation"])
    test_names = set(seen["test"])
    cells = {}
    all_match = True
    for key, (cnt, s) in stats.items():
        exp = EXPECTED_CONTROL[key]
        ok = (cnt, s) == exp
        all_match = all_match and ok
        cells[key] = {"count": cnt, "sum": s, "expected": exp, "ok": ok}
    totals = {}
    for part, exp in EXPECTED_TOTALS.items():
        cnt = sum(cells[c, part]["count"] for c in CLASSES)
        s = sum(cells[c, part]["sum"] for c in CLASSES)
        ok = (cnt, s) == exp
        all_match = all_match and ok
        totals[part] = {"count": cnt, "sum": s, "expected": exp, "ok": ok}
    disjoint = {
        "train_vs_validation": len(train_names & val_names) == 0,
        "train_vs_test": len(train_names & test_names) == 0,
        "validation_vs_test": len(val_names & test_names) == 0,
    }
    uniqueness = {
        "total_rows": len(all_names),
        "total_unique_names": len(set(all_names)),
        "train_unique": len(train_names) == len(seen["train"]),
        "validation_unique": len(val_names) == len(seen["validation"]),
        "test_unique": len(test_names) == len(seen["test"]),
        "all_unique": len(set(all_names)) == len(all_names),
    }
    for v in disjoint.values():
        all_match = all_match and v
    all_match = all_match and uniqueness["all_unique"]
    all_match = all_match and uniqueness["train_unique"]
    all_match = all_match and uniqueness["validation_unique"]
    all_match = all_match and uniqueness["test_unique"]
    report = {
        "cells": cells,
        "totals": totals,
        "disjoint": disjoint,
        "uniqueness": uniqueness,
        "all_match": all_match,
    }
    return (report, all_match)


LABEL_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


def _tensor_denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def make_train_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def make_eval_transform() -> T.Compose:
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class IntelDataset(Dataset):
    def __init__(self, data_root: Path, split_csv: Path, part: str, transform: T.Compose) -> None:
        assert part in ("train", "validation", "test"), part
        self.data_root = data_root
        self.part = part
        self.transform = transform
        rows = [r for r in read_split(split_csv) if r[2] == part]
        self.items: list[tuple[Path, int]] = []
        base = _train_root(data_root) if part == "train" else _test_root(data_root)
        for name, cls, _ in rows:
            self.items.append((base / cls / name, LABEL_TO_IDX[cls]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, y = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return (x, y)


def make_train_loader(data_root: Path, split_csv: Path) -> DataLoader:
    ds = IntelDataset(data_root, split_csv, "train", make_train_transform())
    return DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(data_root: Path, split_csv: Path, part: str) -> DataLoader:
    ds = IntelDataset(data_root, split_csv, part, make_eval_transform())
    return DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)
