from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
SPLIT_CSV = ROOT / "split.csv"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MODEL_PATH = ROOT / "lab3_model.pt"
CLASSES: tuple[str, ...] = ("buildings", "forest", "glacier", "mountain", "sea", "street")
NUM_CLASSES = len(CLASSES)
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
IMG_SIZE = 160
TRAIN_BATCH = 32
EVAL_BATCH = 64
EPOCHS = 8
STEPS_PER_EPOCH = 38
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WEIGHT_DECAY = 0.0001
LR_HEAD = 0.003
LR_LAYER4 = 0.0003
BN_MOMENTUM = 0.1
BN_EPS = 1e-05
REGIMES: tuple[str, ...] = ("scratch", "frozen", "unfreeze")
SEEDS: dict[str, tuple[int, ...]] = {"scratch": (1,), "frozen": (1, 2, 3), "unfreeze": (1, 2, 3)}
MODULES_IN_EVAL: dict[str, tuple[str, ...]] = {
    "scratch": (),
    "frozen": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "unfreeze": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}
