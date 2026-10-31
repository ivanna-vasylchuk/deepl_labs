from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import checks
from config import (
    CLASSES,
    FIGURES_DIR,
    IMG_SIZE,
    MODEL_PATH,
    RESULTS_DIR,
    RUNS_DIR,
    SEEDS,
    SPLIT_CSV,
)
from data import build_split, control_table
from engine import train_one_run, warmup_time_one_epoch
from selection import choose, load_runs

CODE_FILES = [
    Path(__file__).parent / fn
    for fn in ("config.py", "data.py", "model.py", "engine.py", "train.py")
]
DEFAULT_RUN_ORDER: list[tuple[str, int]] = [
    (r, s) for r in ("scratch", "frozen", "unfreeze") for s in SEEDS[r]
]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def do_split_and_report(data_root: Path) -> None:
    print("[split] будую split.csv …")
    build_split(data_root, SPLIT_CSV)
    report, ok = control_table(SPLIT_CSV)
    print(f"[split] контроль: {('OK' if ok else 'FAIL')}")
    if not ok:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        sys.exit(2)


def do_checks(data_root: Path, device: torch.device) -> None:
    print("[checks] пакет після перетворень → figures/batch.png")
    checks.save_batch_figure(data_root, FIGURES_DIR / "batch.png", seed=1)
    print("[checks] L0 для scratch і frozen")
    s = checks.compute_L0("scratch", data_root, device, seed=1)
    f = checks.compute_L0("frozen", data_root, device, seed=1)
    checks.write_initial_loss_report(s, f, RESULTS_DIR / "initial_loss.md")
    print(f"  scratch L0 = {s['L0']:.4f}; frozen L0 = {f['L0']:.4f}")
    print("[checks] параметри по режимах")
    rows = checks.param_report(device)
    checks.write_param_report(rows, RESULTS_DIR / "param_table.md")
    print("[checks] зміни модулів після 5 кроків (frozen, unfreeze)")
    ct_frozen = checks.change_table("frozen", data_root, device, use_set_train_mode=True, seed=1)
    ct_unfreeze = checks.change_table(
        "unfreeze", data_root, device, use_set_train_mode=True, seed=1
    )
    checks.write_module_changes([ct_frozen, ct_unfreeze], RESULTS_DIR / "module_changes.md")
    pred_path = RESULTS_DIR / "bn_experiment_prediction.md"
    if not pred_path.exists():
        print(f"[checks] УВАГА: прогноз {pred_path.name} не знайдено — пропускаю дослід з BN")
        return
    print("[checks] дослід із BN (frozen + model.train())")
    ct_correct = checks.change_table("frozen", data_root, device, use_set_train_mode=True, seed=1)
    ct_wrong = checks.change_table(
        "frozen", data_root, device, use_set_train_mode=False, seed=1, capture_bn_inputs=True
    )
    checks.write_bn_experiment(
        ct_correct, ct_wrong, ct_unfreeze, RESULTS_DIR / "bn_experiment.md", pred_path
    )


def do_warmup(data_root: Path, device: torch.device, runs_debug: Path) -> dict:
    runs_debug.mkdir(parents=True, exist_ok=True)
    print("[warmup] одна епоха на режим (без validation):")
    out = {}
    for regime in ("scratch", "frozen", "unfreeze"):
        t = warmup_time_one_epoch(regime, 1, device, data_root)
        out[regime] = float(t)
        print(f"  {regime}: {t:.1f} s / epoch")
    full = sum(out.values()) * 8
    total_est = (out["scratch"] * 1 + out["frozen"] * 3 + out["unfreeze"] * 3) * 8
    print(
        f"[warmup] оцінка повного навчання (7 запусків × 8 епох): ≈ {total_est:.0f} s ({total_est / 60:.1f} хв)"
    )
    out["total_estimate_s"] = float(total_est)
    (runs_debug / "warmup.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def do_runs(
    data_root: Path,
    device: torch.device,
    runs_dir: Path,
    only: list[tuple[str, int]] | None = None,
    skip_existing: bool = False,
) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    plan = only if only else DEFAULT_RUN_ORDER
    for regime, seed in plan:
        out_dir = runs_dir / f"{regime}_seed{seed}"
        if skip_existing and (out_dir / "run.json").exists():
            print(f"[run] пропускаю {regime}/seed={seed} (вже є run.json)")
            continue
        print(f"[run] {regime}/seed={seed} …")
        t0 = time.perf_counter()
        meta = train_one_run(regime, seed, device, data_root, runs_dir, CODE_FILES)
        dt = time.perf_counter() - t0
        f8 = meta["per_epoch_val_macro_f1"][-1]
        print(f"[run] {regime}/seed={seed} done: F8={f8:.4f}, {dt:.1f}s wall")


def do_select(runs_dir: Path) -> None:
    runs = load_runs(runs_dir)
    sel = choose(runs)
    regime = sel["regime"]
    seed = sel["seed"]
    src = runs_dir / f"{regime}_seed{seed}" / "model.pt"
    state_dict = torch.load(src, map_location="cpu")
    payload = {
        "state_dict": state_dict,
        "regime": regime,
        "seed": seed,
        "classes": list(CLASSES),
        "img_size": IMG_SIZE,
    }
    torch.save(payload, MODEL_PATH)
    print(
        f"[select] обрано {regime}/seed={seed} (F={sel['F_r_s']:.4f}); mean F по режимах: "
        + ", ".join((f"{r}={v:.4f}" for r, v in sel["mean_F_by_regime"].items()))
    )
    print(f"[select] lab3_model.pt → {MODEL_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="ЛР3: навчання. Жодного кроку по test тут немає.")
    p.add_argument(
        "--data", required=True, type=Path, help="шлях до набору Intel Image Classification"
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="тека для запусків (стандартна — runs/; для налагодження — runs_debug/)",
    )
    p.add_argument(
        "--only-checks",
        action="store_true",
        help="виконати лише перевірки пунктів 1–2 і завершити (без семи запусків)",
    )
    p.add_argument(
        "--skip-checks",
        action="store_true",
        help="пропустити перевірки пунктів 1–2 і одразу перейти до запусків",
    )
    p.add_argument(
        "--only-warmup",
        action="store_true",
        help="виміряти час однієї епохи для кожного режиму в runs_debug/warmup.json і завершити",
    )
    p.add_argument(
        "--runs",
        type=str,
        default="",
        help="підмножина запусків через кому, наприклад 'frozen_seed1,unfreeze_seed2'",
    )
    p.add_argument(
        "--skip-existing", action="store_true", help="пропустити запуски з готовим run.json"
    )
    p.add_argument(
        "--no-model-save", action="store_true", help="не зберігати lab3_model.pt (для налагодження)"
    )
    args = p.parse_args()
    device = _pick_device()
    print(
        f"[env] device={device} ({('CUDA' if device.type == 'cuda' else device.type)}); python={sys.version.split()[0]}, torch={torch.__version__}"
    )
    do_split_and_report(args.data)
    if not args.skip_checks:
        do_checks(args.data, device)
    if args.only_checks:
        print("[train] --only-checks → завершую")
        return
    if args.only_warmup:
        do_warmup(
            args.data,
            device,
            Path(args.runs_dir) if "debug" in args.runs_dir.name else Path("runs_debug"),
        )
        return
    only: list[tuple[str, int]] | None = None
    if args.runs:
        only = []
        for tok in args.runs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            r, s = tok.split("_seed")
            only.append((r, int(s)))
    do_runs(args.data, device, args.runs_dir, only=only, skip_existing=args.skip_existing)
    if not args.no_model_save and (not args.runs):
        do_select(args.runs_dir)


if __name__ == "__main__":
    main()
