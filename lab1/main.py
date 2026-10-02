from __future__ import annotations

import argparse
from pathlib import Path

from checks import (
    EPS_NUMERIC,
    TOL_NUMERIC,
    TOL_PYTORCH,
    NumericCheckResult,
    PytorchCheckResult,
    check_against_pytorch,
    check_numeric,
)
from data import load_iris_split, one_hot
from model_numpy import init_params

RESULTS_DIR = Path(__file__).parent / "results"


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0"
    return f"{x:.3e}"


def _fmt_full(x: float) -> str:
    return f"{x:.16e}"


def _fmt_num(x: float) -> str:
    s = f"{x:.9e}"
    return s if s.startswith("-") else " " + s


def _fmt_exp(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def render_pytorch_md(result: PytorchCheckResult, *, tol: float = TOL_PYTORCH) -> str:
    lines = [
        f"### Звірка з PyTorch (допуск {tol:.0e})",
        "",
        f"- Втрата NumPy:  `{_fmt_full(result.loss_numpy)}`",
        f"- Втрата PyTorch: `{_fmt_full(result.loss_torch)}`",
        "",
        "| Величина | Макс. абс. різниця | Пройдено |",
        "|---|---|---|",
    ]
    for r in result.rows:
        lines.append(f"| {r.name} | {_fmt(r.max_abs_diff)} | {'так' if r.passed else 'НІ'} |")
    lines.append("")
    lines.append(
        f"**Загальний результат:** {'усі перевірки пройдено' if result.all_passed else 'ПРОВАЛ'}"
    )
    return "\n".join(lines) + "\n"


def render_numeric_md(
    result: NumericCheckResult, *, eps: float = EPS_NUMERIC, tol: float = TOL_NUMERIC
) -> str:
    lines = [
        f"### Чисельна перевірка (ε = {eps:.0e}, допуск {tol:.0e})",
        "",
        "| Параметр | Градієнт backward() | Чисельна похідна | Абс. різниця | Маска ReLU стабільна | Пройдено |",
        "|---|---|---|---|---|---|",
    ]
    for r in result.rows:
        lines.append(
            f"| {r.name} | `{_fmt_num(r.g_backward)}` | `{_fmt_num(r.g_numeric)}` | "
            f"{_fmt(r.abs_diff)} | {'так' if r.mask_stable else 'НІ'} | "
            f"{'так' if r.passed else 'НІ'} |"
        )
    lines.append("")
    lines.append(
        f"**Загальний результат:** {'усі перевірки пройдено' if result.all_passed else 'ПРОВАЛ'}"
    )
    return "\n".join(lines) + "\n"


def _print_table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> None:
    widths = [
        max(len(h), max((len(row[i]) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "   "
    print(indent + sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(indent + sep.join(cell.ljust(w) for cell, w in zip(row, widths)))


def print_pytorch_console(result: PytorchCheckResult, *, tol: float = TOL_PYTORCH) -> None:
    print(f"Звірка з PyTorch (допуск {_fmt_exp(tol)})")
    labels = ["Втрата NumPy:", "Втрата PyTorch:"]
    w = max(len(label) for label in labels)
    print(f"  {labels[0].ljust(w)}  {result.loss_numpy}")
    print(f"  {labels[1].ljust(w)}  {result.loss_torch}")
    print()
    rows = [
        [r.name, _fmt(r.max_abs_diff), "OK" if r.passed else "ПРОВАЛ"]
        for r in result.rows
    ]
    _print_table(["Величина", "Макс. різниця", "Результат"], rows)


def print_numeric_console(
    result: NumericCheckResult, *, eps: float = EPS_NUMERIC, tol: float = TOL_NUMERIC
) -> None:
    print(f"Чисельна перевірка (ε = {_fmt_exp(eps)}, допуск {_fmt_exp(tol)})")
    rows = [
        [
            r.name,
            _fmt_num(r.g_backward),
            _fmt_num(r.g_numeric),
            _fmt(r.abs_diff),
            "так" if r.mask_stable else "ні",
            "OK" if r.passed else "ПРОВАЛ",
        ]
        for r in result.rows
    ]
    _print_table(
        ["Параметр", "backward", "чисельна", "різниця", "маска", "результат"],
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ЛР1: перевірка зворотного проходу")
    parser.add_argument(
        "--bug",
        action="store_true",
        help="Прибрати ділення на N у Δ2 (навмисна помилка).",
    )
    args = parser.parse_args()
    bug = args.bug

    RESULTS_DIR.mkdir(exist_ok=True)

    split = load_iris_split(seed=0)
    X = split.X_train
    y = split.y_train
    Y = one_hot(y)
    params = init_params(seed=0)

    print(f"Режим: {'BUG (без 1/N у Δ2)' if bug else 'правильна реалізація'}")
    print()

    pt_result, grads = check_against_pytorch(params, X, y, Y, bug=bug)
    print_pytorch_console(pt_result)
    print()

    num_result = check_numeric(params, grads, X, Y)
    print_numeric_console(num_result)
    print()

    suffix = "_bug" if bug else ""
    pt_path = RESULTS_DIR / f"pytorch_check{suffix}.md"
    num_path = RESULTS_DIR / f"numeric_check{suffix}.md"
    pt_path.write_text(render_pytorch_md(pt_result), encoding="utf-8")
    num_path.write_text(render_numeric_md(num_result), encoding="utf-8")

    all_passed = pt_result.all_passed and num_result.all_passed
    if bug:
        both_failed = (not pt_result.all_passed) and (not num_result.all_passed)
        if both_failed:
            print("Підсумок: помилку виявили обидві перевірки (як і очікувалось)")
        else:
            print("Підсумок: не всі перевірки виявили помилку")
        return 0
    print(f"Підсумок: {'усі перевірки пройдено' if all_passed else 'ПРОВАЛ'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
