import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PART_DIRS = {"train": "seg_train/seg_train", "test": "seg_test/seg_test"}

SPLIT_CONTROL = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
CALIB_CONTROL = {
    "count": 200,
    "name_sum": 1_921_470,
    "per_class": {"buildings": 40, "forest": 35, "glacier": 36, "mountain": 33, "sea": 29, "street": 27},
}


def name_num(fname):
    return int(Path(fname).stem)


def read_split(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def part_rows(rows, part):
    return [r for r in rows if r["part"] == part]


def check_split(rows):
    out = {}
    for part, expected in SPLIT_CONTROL.items():
        sub = part_rows(rows, part)
        got = (len(sub), sum(name_num(r["file"]) for r in sub))
        if got != expected:
            raise SystemExit(f"Контроль поділу не пройдено для {part}: {got} != {expected}")
        out[part] = {"count": got[0], "name_sum": got[1], "ok": True}
    return out


def calibration_rows(rows):
    train = sorted(part_rows(rows, "train"), key=lambda r: r["file"])
    idx = np.random.default_rng(0).permutation(len(train))[:200]
    return [train[i] for i in idx]


def check_calibration(calib):
    got = {
        "count": len(calib),
        "name_sum": sum(name_num(r["file"]) for r in calib),
        "per_class": {c: sum(r["label"] == c for r in calib) for c in CLASSES},
    }
    if got != CALIB_CONTROL:
        raise SystemExit(f"Контроль калібрувальної вибірки не пройдено: {got}")
    return {**got, "ok": True}


def make_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def image_path(data_root, row):
    return Path(data_root) / PART_DIRS[row["part"]] / row["label"] / row["file"]


def load_images(data_root, rows, tf):
    images = []
    for r in rows:
        with Image.open(image_path(data_root, r)) as im:
            images.append(tf(im.convert("RGB")).numpy())
    return np.stack(images).astype(np.float32)


def labels(rows):
    return np.array([CLASSES.index(r["label"]) for r in rows])
