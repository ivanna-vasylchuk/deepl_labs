from __future__ import annotations

import json
from pathlib import Path

REGIME_ORDER = ("scratch", "frozen", "unfreeze")


def _f_r_s(meta: dict) -> float:
    return float(meta["per_epoch_val_macro_f1"][-1])


def load_runs(runs_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {r: {} for r in REGIME_ORDER}
    for sub in sorted(runs_dir.iterdir()):
        rj = sub / "run.json"
        if not rj.exists():
            continue
        meta = json.loads(rj.read_text())
        regime = meta["regime"]
        seed = int(meta["seed"])
        out.setdefault(regime, {})[seed] = meta
    return out


def regime_mean_F(runs: dict[str, dict[int, dict]]) -> dict[str, float]:
    out = {}
    for r, by_seed in runs.items():
        if not by_seed:
            continue
        vals = [_f_r_s(m) for s, m in sorted(by_seed.items())]
        out[r] = sum(vals) / len(vals)
    return out


def regime_range_R(runs: dict[str, dict[int, dict]]) -> dict[str, float]:
    out = {}
    for r, by_seed in runs.items():
        if len(by_seed) < 2:
            continue
        vals = [_f_r_s(m) for s, m in sorted(by_seed.items())]
        out[r] = max(vals) - min(vals)
    return out


def choose(runs: dict[str, dict[int, dict]]) -> dict:
    means = regime_mean_F(runs)
    assert means, "немає жодного run.json"
    best_mean = max(means.values())
    best_regime = next(r for r in REGIME_ORDER if r in means and means[r] == best_mean)
    by_seed = runs[best_regime]
    seeds_sorted = sorted(by_seed.keys())
    best_seed = max(seeds_sorted, key=lambda s: (_f_r_s(by_seed[s]), -s))
    return {
        "regime": best_regime,
        "seed": int(best_seed),
        "F_r_s": _f_r_s(by_seed[best_seed]),
        "mean_F_by_regime": means,
    }
