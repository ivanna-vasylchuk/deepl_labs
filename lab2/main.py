from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from checks import CHECK_SETS, CheckResult, check_dataset
from lab2_starter import load_predictions
from metrics import confusion, cost, precision_recall
from threshold import ThresholdTable, best_threshold, fast_table

RESULTS = Path(__file__).parent / "results"


def _fmt_t(t: float) -> str:
    return "+inf" if np.isinf(t) else repr(float(t))


def _fmt_pct(x: float | None, digits: int = 6) -> str:
    return "не визначено" if x is None else f"{x:.{digits}f}"


def _threshold_text(x: float) -> str:
    return "inf" if np.isinf(x) else repr(float(x))


def _logit(p: float | np.ndarray) -> np.ndarray:
    p64 = np.asarray(p, dtype=np.float64)
    return np.log(p64 / (1.0 - p64))


def _save_table_csv(path: Path, tbl: ThresholdTable) -> None:
    df = pd.DataFrame(
        {
            "threshold": [_threshold_text(t) for t in tbl.thresholds],
            "TP": tbl.tp,
            "FP": tbl.fp,
            "FN": tbl.fn,
            "TN": tbl.tn,
            "C": tbl.cost,
        }
    )
    df.to_csv(path, index=False)


def analyze_validation(
    y: np.ndarray, s: np.ndarray
) -> tuple[ThresholdTable, float, list[dict], float]:
    tbl = fast_table(y, s)
    t_star, _ = best_threshold(tbl)
    rows: list[dict] = []
    for label, t in (("0.5", 0.5), ("t*", t_star), ("+inf", float("inf"))):
        tp, fp, fn, tn = confusion(y, s, t)
        p, r = precision_recall(tp, fp, fn)
        rows.append(
            {
                "rule": label,
                "threshold": t,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "C": cost(fn, fp),
                "precision": p, "recall": r,
            }
        )
    auc = float(roc_auc_score(y, s.astype(np.float64)))
    return tbl, t_star, rows, auc


def validate_csv_integrity(
    path: Path, tbl: ThresholdTable, y: np.ndarray, s: np.ndarray, t_star: float
) -> None:
    # float_precision="round_trip": стандартний C-парсер pandas округлює
    df = pd.read_csv(path, float_precision="round_trip")
    csv_t = df["threshold"].to_numpy(dtype=np.float64)
    if not np.array_equal(csv_t, tbl.thresholds):
        raise RuntimeError("пороги з CSV не рівні побітово tbl.thresholds")
    mask = np.isinf(csv_t) if np.isinf(t_star) else (csv_t == t_star)
    idx = int(np.flatnonzero(mask).max())
    row = df.iloc[idx]
    tp, fp, fn, tn = confusion(y, s, t_star)
    csv_counts = (int(row["TP"]), int(row["FP"]), int(row["FN"]), int(row["TN"]))
    if (tp, fp, fn, tn) != csv_counts:
        raise RuntimeError(
            f"рядок t*={_fmt_t(t_star)}: CSV має {csv_counts}, confusion дає {(tp, fp, fn, tn)}"
        )


def plot_pr(y: np.ndarray, s: np.ndarray, pos_frac: float, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="C0", lw=1.5, label="PR-крива")
    ax.axhline(
        pos_frac, color="grey", lw=1, ls="--",
        label=f"базова лінія = частка позитивів = {pos_frac:.4f}",
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall на validation")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y: np.ndarray, s: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="C0", lw=1.5, label=f"ROC-крива (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="випадковий класифікатор")
    ax.set_xlabel("FPR (false positive rate)")
    ax.set_ylabel("TPR (true positive rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC-крива на validation")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_validation_md(
    summary_rows: list[dict], t_star: float, min_c: int, auc: float, n_candidates: int
) -> str:
    lines = [
        "# Валідація",
        "",
        f"- кандидатів: {n_candidates}",
        f"- ROC-AUC: {auc:.6f}",
        f"- t* = {_fmt_t(t_star)}, min C = {min_c}",
        "",
        "| правило | поріг | TP | FP | FN | TN | C | precision | recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['rule']} | {_fmt_t(r['threshold'])} | {r['TP']} | {r['FP']} | "
            f"{r['FN']} | {r['TN']} | {r['C']} | {_fmt_pct(r['precision'])} | "
            f"{_fmt_pct(r['recall'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _check_table_lines(tbl: ThresholdTable) -> list[str]:
    out = ["| поріг | TP | FP | FN | TN | C |", "|---|---|---|---|---|---|"]
    for i in range(len(tbl.thresholds)):
        out.append(
            f"| {_fmt_t(float(tbl.thresholds[i]))} | {int(tbl.tp[i])} | {int(tbl.fp[i])} | "
            f"{int(tbl.fn[i])} | {int(tbl.tn[i])} | {int(tbl.cost[i])} |"
        )
    return out


def render_checks_md(results: list[CheckResult]) -> str:
    lines = [
        "# Перевірка пошуку порога",
        "",
        "| набір | що перевіряє | t* fast | t* slow | збіг |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.note} | {_fmt_t(r.t_star_fast)} | "
            f"{_fmt_t(r.t_star_slow)} | {'так' if r.passed else 'НІ'} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append("### fast")
        lines.append("")
        lines.extend(_check_table_lines(r.fast))
        lines.append("")
        lines.append("### slow")
        lines.append("")
        lines.extend(_check_table_lines(r.slow))
        lines.append("")
    return "\n".join(lines)


def evaluate_test(
    t_star: float,
    y: np.ndarray,
    s: np.ndarray,
    amount: np.ndarray,
    row_ids: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    tp, fp, fn, tn = confusion(y, s, t_star)
    p, r = precision_recall(tp, fp, fn)
    c = cost(fn, fp)

    s64 = s.astype(np.float64)
    pred = s64 >= t_star
    yb = y.astype(bool)
    fp_mask = pred & ~yb
    fn_mask = ~pred & yb

    logit_t = float(_logit(t_star)) if np.isfinite(t_star) else float("inf")

    rows: list[dict] = []
    for name, mask in (("FP", fp_mask), ("FN", fn_mask)):
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(row_ids[idx])][:5]
        for i in order:
            score = float(s64[i])
            rows.append(
                {
                    "csv_row": int(row_ids[i]),
                    "type": name,
                    "label": int(y[i]),
                    "score": score,
                    "amount": float(amount[i]),
                    "distance_from_t_star": score - float(t_star),
                    "logit_distance_from_t_star": float(_logit(score)) - logit_t,
                }
            )
    stats = {
        "t_star": t_star,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "C": c, "precision": p, "recall": r,
    }
    return stats, pd.DataFrame(rows)


def render_test_md(stats: dict, errors: pd.DataFrame) -> str:
    lines = [
        "# Test",
        "",
        f"t* = {_fmt_t(stats['t_star'])}",
        "",
        "| TP | FP | FN | TN | C | precision | recall |",
        "|---|---|---|---|---|---|---|",
        f"| {stats['TP']} | {stats['FP']} | {stats['FN']} | {stats['TN']} | "
        f"{stats['C']} | {_fmt_pct(stats['precision'])} | {_fmt_pct(stats['recall'])} |",
        "",
        "## Перші 5 FP і FN за номером рядка CSV",
        "",
        "| csv_row | тип | мітка | оцінка | Amount | d_prob | d_logit |",
        "|---|---|---|---|---|---|---|",
    ]
    if errors.empty:
        lines.append("| | | | | | | |")
    else:
        for _, r in errors.iterrows():
            lines.append(
                f"| {int(r['csv_row'])} | {r['type']} | {int(r['label'])} | "
                f"{r['score']:.6f} | {r['amount']:.2f} | "
                f"{r['distance_from_t_star']:+.6f} | "
                f"{r['logit_distance_from_t_star']:+.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_predictions(RESULTS)
    y_va = data["validation_y"]
    s_va = data["validation_scores"]
    y_te = data["test_y"]
    s_te = data["test_scores"]
    amount_te = data["test_amount"]
    row_ids_te = data["test_row_ids"]

    pos_frac_va = float(np.count_nonzero(y_va == 1) / len(y_va))

    tbl, t_star, summary_rows, auc = analyze_validation(y_va, s_va)
    min_c = int(tbl.cost.min())

    csv_path = RESULTS / "validation_thresholds.csv"
    _save_table_csv(csv_path, tbl)
    validate_csv_integrity(csv_path, tbl, y_va, s_va, t_star)
    plot_pr(y_va, s_va, pos_frac_va, RESULTS / "pr_curve.png")
    plot_roc(y_va, s_va, auc, RESULTS / "roc_curve.png")
    (RESULTS / "validation_summary.md").write_text(
        render_validation_md(summary_rows, t_star, min_c, auc, len(tbl.cost)),
        encoding="utf-8",
    )
    (RESULTS / "threshold.json").write_text(
        json.dumps(
            {
                "t_star": "+inf" if np.isinf(t_star) else float(t_star),
                "t_star_repr": _fmt_t(t_star),
                "min_cost_validation": min_c,
                "roc_auc_validation": auc,
                "tie_rule": "найбільший поріг серед мінімумів C",
            },
            ensure_ascii=False, indent=2, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"validation: t* = {_fmt_t(t_star)}, C = {min_c}, AUC = {auc:.6f}")

    check_results: list[CheckResult] = []
    for name, s, y, note in CHECK_SETS:
        res = check_dataset(name, s, y, note)
        _save_table_csv(RESULTS / f"search_check_{name}_fast.csv", res.fast)
        _save_table_csv(RESULTS / f"search_check_{name}_slow.csv", res.slow)
        print(f"check {name}: t* = {_fmt_t(res.t_star_fast)} ({'OK' if res.passed else 'FAIL'})")
        if not res.passed:
            raise RuntimeError(f"перевірка {name} провалилась")
        check_results.append(res)
    (RESULTS / "search_check.md").write_text(render_checks_md(check_results), encoding="utf-8")

    stats, errors = evaluate_test(t_star, y_te, s_te, amount_te, row_ids_te)
    errors.to_csv(RESULTS / "test_errors.csv", index=False, float_format="%.17g")
    (RESULTS / "test_report.md").write_text(render_test_md(stats, errors), encoding="utf-8")
    print(
        f"test: TP={stats['TP']} FP={stats['FP']} FN={stats['FN']} TN={stats['TN']} "
        f"C={stats['C']} precision={_fmt_pct(stats['precision'])} "
        f"recall={_fmt_pct(stats['recall'])}"
    )


if __name__ == "__main__":
    main()
