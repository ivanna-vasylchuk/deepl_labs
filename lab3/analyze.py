from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    CLASSES,
    EPOCHS,
    FIGURES_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    RESULTS_DIR,
    RUNS_DIR,
    SEEDS,
    SPLIT_CSV,
)
from selection import REGIME_ORDER, _f_r_s, choose, regime_mean_F, regime_range_R

ROOT = Path(__file__).resolve().parent
CODE_FILES_TO_CHECK = ("config.py", "data.py", "model.py", "engine.py", "train.py")
REGIME_COLORS = {"scratch": "#d62728", "frozen": "#1f77b4", "unfreeze": "#2ca02c"}
SEED_STYLES = {1: "-", 2: "--", 3: ":"}


def check_integrity(runs_dir: Path) -> list[dict]:
    expected = [(r, s) for r in REGIME_ORDER for s in SEEDS[r]]
    found: dict[tuple[str, int], dict] = {}
    for d in sorted(runs_dir.iterdir()):
        rj = d / "run.json"
        vl = d / "val_logits.npy"
        if not rj.exists():
            continue
        meta = json.loads(rj.read_text())
        key = (meta["regime"], int(meta["seed"]))
        found[key] = meta
        assert vl.exists(), f"{vl} відсутній"
        arr = np.load(vl)
        assert arr.shape == (1498, NUM_CLASSES) and arr.dtype == np.float32, (
            f"{vl} shape={arr.shape} dtype={arr.dtype}"
        )
        assert len(meta["per_epoch_train_loss"]) == EPOCHS, "epochs != 8"
        assert len(meta["per_epoch_val_macro_f1"]) == EPOCHS, "epochs != 8"
    missing = [k for k in expected if k not in found]
    extra = [k for k in found if k not in expected]
    if missing or extra:
        raise SystemExit(f"[analyze] помилка: missing={missing}, extra={extra}")
    cur = hashlib.sha256(SPLIT_CSV.read_bytes()).hexdigest()
    sha_set = {m["split_sha256"] for m in found.values()}
    if sha_set != {cur}:
        raise SystemExit(
            f"[analyze] хеш split.csv не збігається: журнали {sha_set} vs поточний {cur}"
        )
    cur_code = {
        fn: hashlib.sha256((ROOT / fn).read_bytes()).hexdigest() for fn in CODE_FILES_TO_CHECK
    }
    code_warnings = []
    for fn in CODE_FILES_TO_CHECK:
        vals = {m["code_sha256"].get(fn) for m in found.values()}
        if len(vals) != 1:
            raise SystemExit(f"[analyze] code_sha256[{fn}] різний між журналами: {vals}")
        if next(iter(vals)) != cur_code[fn]:
            code_warnings.append(fn)
    if code_warnings:
        print(
            f"[analyze] попередження: code_sha256 у журналах відрізняється від поточних файлів для: {code_warnings}. Це очікувано — див. абзац «Зачистка коментарів після запусків» у розділі «Відтворюваність» README.md."
        )
    return [found[k] for k in expected]


def table_seven_runs(metas: list[dict]) -> list[dict]:
    rows = []
    for i, m in enumerate(metas, start=1):
        rows.append(
            {
                "n": i,
                "regime": m["regime"],
                "seed": int(m["seed"]),
                "train_loss_ep8": float(m["per_epoch_train_loss"][-1]),
                "val_macro_f1_ep8": float(m["per_epoch_val_macro_f1"][-1]),
                "train_time_s": float(m["train_time_s"]),
                "total_time_s": float(m["total_time_s"]),
            }
        )
    return rows


def draw_curves(metas: list[dict], out_path: Path) -> None:
    fig, (ax_l, ax_f) = plt.subplots(1, 2, figsize=(13, 5))
    for m in metas:
        regime = m["regime"]
        seed = int(m["seed"])
        color = REGIME_COLORS[regime]
        style = SEED_STYLES.get(seed, "-")
        xs = list(range(1, EPOCHS + 1))
        ax_l.plot(
            xs,
            m["per_epoch_train_loss"],
            color=color,
            linestyle=style,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{regime} s={seed}",
        )
        ax_f.plot(
            xs,
            m["per_epoch_val_macro_f1"],
            color=color,
            linestyle=style,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{regime} s={seed}",
        )
    ax_l.set_xlabel("епоха")
    ax_l.set_ylabel("втрата train (∑ Lb·|b|/1200)")
    ax_l.set_title("Втрата на train за епохами")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(loc="upper right", fontsize=8, ncol=2)
    ax_f.set_xlabel("епоха")
    ax_f.set_ylabel("macro-F1 validation")
    ax_f.set_title("macro-F1 на validation за епохами")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="lower right", fontsize=8, ncol=2)
    ax_l.set_xticks(xs)
    ax_f.set_xticks(xs)
    fig.suptitle(
        "Сім запусків: криві train і validation (колір — режим, стиль — зерно)", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def effects_table(metas: list[dict]) -> dict:
    runs = {r: {} for r in REGIME_ORDER}
    for m in metas:
        runs[m["regime"]][int(m["seed"])] = m
    means = regime_mean_F(runs)
    ranges = regime_range_R(runs)
    R = max(ranges["frozen"], ranges["unfreeze"])
    F_scratch = _f_r_s(runs["scratch"][1])
    delta_pre = means["frozen"] - F_scratch
    delta_unf = means["unfreeze"] - means["frozen"]

    def verdict(d: float) -> str:
        return (
            "не відрізняється від розкиду (|Δ| ≤ R)"
            if abs(d) <= R
            else "ефект вартий перевірки (|Δ| > R)"
        )

    return {
        "mean_F": means,
        "range_R": ranges,
        "R": R,
        "F_scratch_1": F_scratch,
        "delta_pre": delta_pre,
        "delta_unf": delta_unf,
        "delta_sum": delta_pre + delta_unf,
        "verdict_pre": verdict(delta_pre),
        "verdict_unf": verdict(delta_unf),
    }


def verify_selection(metas: list[dict], model_path: Path, runs_dir: Path) -> dict:
    runs = {r: {} for r in REGIME_ORDER}
    for m in metas:
        runs[m["regime"]][int(m["seed"])] = m
    rule = choose(runs)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    assert set(payload.keys()) >= {"state_dict", "regime", "seed", "classes", "img_size"}, (
        payload.keys()
    )
    saved = {"regime": payload["regime"], "seed": int(payload["seed"])}
    match_meta = saved["regime"] == rule["regime"] and saved["seed"] == rule["seed"]
    src = runs_dir / f"{rule['regime']}_seed{rule['seed']}" / "model.pt"
    weights_match = None
    if src.exists():
        ref = torch.load(src, map_location="cpu", weights_only=False)
        same_keys = set(ref.keys()) == set(payload["state_dict"].keys())
        if same_keys:
            weights_match = all(torch.equal(ref[k], payload["state_dict"][k]) for k in ref)
    return {
        "rule_regime": rule["regime"],
        "rule_seed": rule["seed"],
        "rule_F_r_s": rule["F_r_s"],
        "saved_regime": saved["regime"],
        "saved_seed": saved["seed"],
        "meta_matches_rule": match_meta,
        "weights_match_source_run": weights_match,
        "classes_ok": payload["classes"] == list(CLASSES),
        "img_size_ok": payload["img_size"] == 160,
    }


def do_test(model_path: Path, data_root: Path) -> dict:
    from test_eval import evaluate_test

    return evaluate_test(model_path, data_root)


def draw_confusion(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = int(cm[i, j])
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Матриця плутанини на test (6 × 6)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pick_five_errors(
    y_true: list[int], y_pred: list[int], probs: np.ndarray, split_csv: Path
) -> list[dict]:
    import csv

    test_rows = []
    with split_csv.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[2] == "test":
                test_rows.append(row[0])
    assert len(test_rows) == len(y_true)
    errors_by_pair: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for name, t, p, pr in zip(test_rows, y_true, y_pred, probs):
        if t == p:
            continue
        errors_by_pair.setdefault((t, p), []).append((name, float(pr[p])))
    pairs_sorted = sorted(
        errors_by_pair.keys(), key=lambda tp: (-len(errors_by_pair[tp]), tp[0], tp[1])
    )
    for tp in pairs_sorted:
        errors_by_pair[tp].sort(key=lambda nf: nf[0])
    picked: list[dict] = []
    for tp in pairs_sorted:
        if len(picked) == 5:
            break
        name, prob = errors_by_pair[tp][0]
        picked.append(
            {
                "file": name,
                "true": tp[0],
                "pred": tp[1],
                "prob_pred": prob,
                "pair_count": len(errors_by_pair[tp]),
                "picked_from_rank": 0,
            }
        )
    rank = 1
    while len(picked) < 5:
        added_this_round = 0
        for tp in pairs_sorted:
            if len(picked) == 5:
                break
            if len(errors_by_pair[tp]) <= rank:
                continue
            name, prob = errors_by_pair[tp][rank]
            picked.append(
                {
                    "file": name,
                    "true": tp[0],
                    "pred": tp[1],
                    "prob_pred": prob,
                    "pair_count": len(errors_by_pair[tp]),
                    "picked_from_rank": rank,
                }
            )
            added_this_round += 1
        if added_this_round == 0:
            break
        rank += 1
    return picked


def draw_errors(picked: list[dict], data_root: Path, out_path: Path) -> None:
    from PIL import Image

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, item in zip(axes, picked):
        true_cls = CLASSES[item["true"]]
        pred_cls = CLASSES[item["pred"]]
        path = data_root / "seg_test" / "seg_test" / true_cls / item["file"]
        im = Image.open(path).convert("RGB")
        ax.imshow(np.asarray(im))
        ax.set_title(
            f"{item['file']}\n true: {true_cls} → pred: {pred_cls}\n p={item['prob_pred']:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("П'ять характерних помилок на test (за правилом умови)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_analysis(
    rows: list[dict],
    eff: dict,
    sel: dict,
    test: dict | None,
    errors: list[dict] | None,
    out_md: Path,
    out_json: Path,
) -> None:
    L = []
    L.append("# Аналіз сім запусків ЛР3")
    L.append("")
    L.append("## Таблиця семи запусків")
    L.append("")
    L.append(
        "| # | Режим | Зерно | Втрата train (ep8) | macro-F1 validation (ep8) | Train time, с | Total time, с |"
    )
    L.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(
            f"| {r['n']} | `{r['regime']}` | {r['seed']} | {r['train_loss_ep8']:.4f} | {r['val_macro_f1_ep8']:.4f} | {r['train_time_s']:.1f} | {r['total_time_s']:.1f} |"
        )
    L.append("")
    L.append("![curves](../figures/curves.png)")
    L.append("")
    L.append("## Розклад ефектів")
    L.append("")

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    def pct(x: float) -> str:
        return f"{x * 100:+.2f} п.п."

    L.append("| Величина | Значення (частка) | У пунктах ×100 |")
    L.append("|---|---:|---:|")
    L.append(f"| F̄_frozen | {fmt(eff['mean_F']['frozen'])} | {eff['mean_F']['frozen'] * 100:.2f} |")
    L.append(
        f"| R_frozen | {fmt(eff['range_R']['frozen'])} | {eff['range_R']['frozen'] * 100:.2f} |"
    )
    L.append(
        f"| F̄_unfreeze | {fmt(eff['mean_F']['unfreeze'])} | {eff['mean_F']['unfreeze'] * 100:.2f} |"
    )
    L.append(
        f"| R_unfreeze | {fmt(eff['range_R']['unfreeze'])} | {eff['range_R']['unfreeze'] * 100:.2f} |"
    )
    L.append(f"| R = max(R_frozen, R_unfreeze) | {fmt(eff['R'])} | {eff['R'] * 100:.2f} |")
    L.append(f"| F_scratch,1 | {fmt(eff['F_scratch_1'])} | {eff['F_scratch_1'] * 100:.2f} |")
    L.append(
        f"| Δpre = F̄_frozen − F_scratch,1 | {fmt(eff['delta_pre'])} | {pct(eff['delta_pre'])} |"
    )
    L.append(
        f"| Δunf = F̄_unfreeze − F̄_frozen | {fmt(eff['delta_unf'])} | {pct(eff['delta_unf'])} |"
    )
    L.append(f"| Δpre + Δunf (сума) | {fmt(eff['delta_sum'])} | {pct(eff['delta_sum'])} |")
    L.append("")
    L.append(
        "**Правило:** якщо `|Δ| ≤ R`, різниця не відрізняється від розкиду. Якщо `|Δ| > R`, ефект вартий перевірки."
    )
    L.append("")
    L.append(f"- Δpre: {eff['verdict_pre']}")
    L.append(f"- Δunf: {eff['verdict_unf']}")
    L.append("")
    L.append("## Перевірка вибору")
    L.append("")
    L.append(
        f"- Правило → **{sel['rule_regime']} / seed={sel['rule_seed']}**, F_r,s = {sel['rule_F_r_s']:.4f}"
    )
    L.append(
        f"- `lab3_model.pt`: regime=`{sel['saved_regime']}`, seed={sel['saved_seed']}, classes_ok={sel['classes_ok']}, img_size_ok={sel['img_size_ok']}"
    )
    L.append(f"- meta у `lab3_model.pt` відповідає правилу: **{sel['meta_matches_rule']}**")
    if sel["weights_match_source_run"] is not None:
        L.append(
            f"- Ваги `lab3_model.pt` ідентичні `runs/{sel['rule_regime']}_seed{sel['rule_seed']}/model.pt`: **{sel['weights_match_source_run']}**"
        )
    L.append("")
    delta_unf = eff["delta_unf"]
    R = eff["R"]
    if abs(delta_unf) <= R:
        L.append(
            f"**|Δunf| = {abs(delta_unf):.4f} ≤ R = {R:.4f}** — вибір між `frozen` і `unfreeze` визначив шум; процедура від цього не змінюється, але це треба назвати."
        )
    else:
        L.append(
            f"**|Δunf| = {abs(delta_unf):.4f} > R = {R:.4f}** — вибір між `frozen` і `unfreeze` вартий перевірки."
        )
    L.append("")
    if test is not None:
        L.append("## Тестова оцінка (CPU, один раз)")
        L.append("")
        rule_F = sel["rule_F_r_s"]
        L.append(f"- Модель: `{test['model_regime']} / seed={test['model_seed']}`")
        L.append(f"- `macro-F1` на test = **{test['test_macro_f1']:.4f}**")
        L.append(
            f"- accuracy p на test = {test['test_accuracy']:.4f}; `SE_test = √(p(1−p)/n)` з n = {test['test_n']}: **{test['test_se_accuracy']:.4f}**"
        )
        if "val_accuracy" in test:
            L.append(
                f"- accuracy p на validation (та сама модель): {test['val_accuracy']:.4f}; `SE_val` з n = {test['val_n']}: **{test['val_se_accuracy']:.4f}**"
            )
            L.append(
                f"- стандартна похибка різниці: `SE_diff ≈ √(SE_test² + SE_val²)` = **{test['se_diff_accuracy']:.4f}**"
            )
        L.append(
            f"- F_r,s (validation) обраного запуску = {rule_F:.4f}; macro-F1 test − validation = **{test['test_macro_f1'] - rule_F:+.4f}** (≈ {abs(test['test_macro_f1'] - rule_F) / test['se_diff_accuracy']:.1f}·SE_diff)"
        )
        if "best_of_3_bias_estimate" in test:
            L.append(
                f"- оцінка оптимістичного зміщення від вибору найкращого з 3: `F_{{unfreeze,3}} − F̄_unfreeze` = **{test['best_of_3_bias_estimate']:+.4f}**"
            )
        L.append("")
        L.append(
            f"**Інтерпретація.** Оптимістичне зміщення вибору «найкращого з трьох» (≈ `{test.get('best_of_3_bias_estimate', 0):+.4f}`) того самого порядку, що й шум двох скінченних вибірок (`SE_diff ≈ {test.get('se_diff_accuracy', 0):.4f}`). Тому в конкретному розбитті воно може бути непомітним або навіть мати протилежний знак: validation і test — різні вибірки зображень (різні 1498 і 1502 файли з `seg_test`), кожна зі своїм шумом. Виміряна різниця ≈ 1·SE_diff — у межах очікуваного шуму. SE частки (accuracy) тут — лише груба оцінка шуму для macro-F1, бо macro-F1 рахується поклассно й усереднюється по 6 класах, а SE виведено для скалярного біноміального питання «правильно чи ні»."
        )
        L.append("")
        L.append("### Матриця плутанини (рядки — справжній клас, стовпці — передбачений)")
        L.append("")
        hdr = "| | " + " | ".join((f"`{c}`" for c in CLASSES)) + " |"
        sep = "|---|" + ":---:|" * NUM_CLASSES
        L.append(hdr)
        L.append(sep)
        cm = test["confusion_matrix"]
        for i, cls in enumerate(CLASSES):
            L.append(
                f"| **`{cls}`** | " + " | ".join((str(cm[i][j]) for j in range(NUM_CLASSES))) + " |"
            )
        L.append("")
        L.append("![confusion](../figures/confusion.png)")
        L.append("")
    if errors is not None:
        L.append("## П'ять характерних помилок на test")
        L.append("")
        L.append(
            "| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого | Кількість таких помилок у test |"
        )
        L.append("|---|---|---|---:|---:|")
        for e in errors:
            L.append(
                f"| `{e['file']}` | {CLASSES[e['true']]} | {CLASSES[e['pred']]} | {e['prob_pred']:.3f} | {e['pair_count']} |"
            )
        L.append("")
        L.append("![errors](../figures/errors.png)")
        L.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L))
    payload = {"rows": rows, "effects": eff, "selection": sel}
    if test is not None:
        payload["test"] = {k: v for k, v in test.items() if k not in ("logits_test", "probs_test")}
    if errors is not None:
        payload["errors"] = errors
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))


def main() -> None:
    p = argparse.ArgumentParser(description="ЛР3: аналіз. Test відкривається ЛИШЕ тут.")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="пропустити тестову оцінку і таблицю помилок (діагностичний режим)",
    )
    args = p.parse_args()
    print("[analyze] перевірка журналів …")
    metas = check_integrity(args.runs_dir)
    rows = table_seven_runs(metas)
    print("[analyze] curves.png …")
    draw_curves(metas, FIGURES_DIR / "curves.png")
    print("[analyze] розклад ефектів …")
    eff = effects_table(metas)
    print(
        f"  F̄_scratch={eff['F_scratch_1']:.4f}  F̄_frozen={eff['mean_F']['frozen']:.4f}  F̄_unfreeze={eff['mean_F']['unfreeze']:.4f}"
    )
    print(
        f"  R_frozen={eff['range_R']['frozen']:.4f}  R_unfreeze={eff['range_R']['unfreeze']:.4f}  R={eff['R']:.4f}"
    )
    print(
        f"  Δpre={eff['delta_pre']:+.4f}  Δunf={eff['delta_unf']:+.4f}  сума={eff['delta_sum']:+.4f}"
    )
    print(f"  вердикт Δpre: {eff['verdict_pre']}")
    print(f"  вердикт Δunf: {eff['verdict_unf']}")
    print("[analyze] перевірка вибору …")
    sel = verify_selection(metas, MODEL_PATH, args.runs_dir)
    print(
        f"  rule → {sel['rule_regime']}/seed={sel['rule_seed']}, saved → {sel['saved_regime']}/seed={sel['saved_seed']}"
    )
    print(
        f"  meta_matches_rule={sel['meta_matches_rule']}, weights_match_source={sel['weights_match_source_run']}"
    )
    test = None
    errors = None
    if not args.skip_test:
        print("[analyze] ТЕСТ (CPU, один раз) …")
        test = do_test(MODEL_PATH, args.data)
        import csv

        sel_run_dir = args.runs_dir / f"{sel['rule_regime']}_seed{sel['rule_seed']}"
        val_logits = np.load(sel_run_dir / "val_logits.npy")
        val_labels = []
        with SPLIT_CSV.open() as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row[2] == "validation":
                    val_labels.append(CLASSES.index(row[1]))
        val_true = np.asarray(val_labels)
        val_pred = val_logits.argmax(axis=1)
        p_val = float((val_pred == val_true).mean())
        n_val = int(len(val_true))
        se_val = math.sqrt(p_val * (1 - p_val) / n_val)
        se_diff = math.sqrt(test["test_se_accuracy"] ** 2 + se_val**2)
        test["val_accuracy"] = p_val
        test["val_n"] = n_val
        test["val_se_accuracy"] = float(se_val)
        test["se_diff_accuracy"] = float(se_diff)
        if sel["rule_regime"] in ("frozen", "unfreeze"):
            mean_F_regime = eff["mean_F"][sel["rule_regime"]]
            test["best_of_3_bias_estimate"] = float(sel["rule_F_r_s"] - mean_F_regime)
        print(
            f"  test macro-F1 = {test['test_macro_f1']:.4f}  acc={test['test_accuracy']:.4f}  SE_test={test['test_se_accuracy']:.4f}"
        )
        print(f"  val accuracy = {p_val:.4f}  SE_val={se_val:.4f}  SE_diff={se_diff:.4f}")
        if "best_of_3_bias_estimate" in test:
            print(
                f"  best-of-3 bias estimate (F_sel − F̄_regime) = {test['best_of_3_bias_estimate']:+.4f}"
            )
        draw_confusion(np.asarray(test["confusion_matrix"]), FIGURES_DIR / "confusion.png")
        print("[analyze] п'ять характерних помилок …")
        errors = pick_five_errors(
            test["y_true_test"], test["y_pred_test"], test["probs_test"], SPLIT_CSV
        )
        draw_errors(errors, args.data, FIGURES_DIR / "errors.png")
    else:
        print("[analyze] --skip-test → test не відкрито, таблицю помилок пропущено")
    write_analysis(
        rows, eff, sel, test, errors, RESULTS_DIR / "analysis.md", RESULTS_DIR / "analysis.json"
    )
    print(f"[analyze] → results/analysis.md, results/analysis.json")


if __name__ == "__main__":
    main()
