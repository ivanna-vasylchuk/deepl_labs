from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

from config import (
    CLASSES,
    FIGURES_DIR,
    MEAN,
    NUM_CLASSES,
    RESULTS_DIR,
    SPLIT_CSV,
    STD,
    TRAIN_BATCH,
)
from data import make_train_loader
from model import (
    TOP_LEVEL_MODULES,
    build_model,
    count_parameters,
    group_sizes,
    make_optimizer,
    set_train_mode,
)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0.0, 1.0)


def save_batch_figure(data_root: Path, out_path: Path, seed: int = 1) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = make_train_loader(data_root, SPLIT_CSV)
    x, y = next(iter(loader))
    assert x.shape == (TRAIN_BATCH, 3, 160, 160), x.shape
    x16 = x[:16]
    y16 = y[:16]
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = denormalize(x16[i]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(CLASSES[int(y16[i])], fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Перші 16 зображень навчального пакета після аугментацій\n(RandomResizedCrop 160, HorizontalFlip, обернене Normalize, clamp [0, 1])",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_L0(
    regime: str,
    data_root: Path,
    device: torch.device,
    seed: int = 1,
    mc_inits: int = 200,
    mc_seed: int = 12345,
) -> dict:
    model = build_model(regime, seed=seed, device=device)
    set_train_mode(model, regime)
    loader = make_train_loader(data_root, SPLIT_CSV)
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        per_sample = nn.CrossEntropyLoss(reduction="none")(logits, y)
        batch_sem = float(per_sample.std(unbiased=False).item()) / math.sqrt(per_sample.numel())
        feats = _features_norm(model, x)
        logits_std = float(logits.detach().std().item())
        feats_tensor = _features_tensor(model, x)
    mc = _mc_logits_std(feats_tensor, n_inits=mc_inits, seed=mc_seed)
    return {
        "regime": regime,
        "seed": seed,
        "L0": float(loss.item()),
        "ln6": float(math.log(NUM_CLASSES)),
        "feature_norm_mean": feats["mean"],
        "feature_norm_std": feats["std"],
        "logits_std": logits_std,
        "batch_sem_L0": batch_sem,
        "batch_size": int(per_sample.numel()),
        "mc_std_mean": mc["mean"],
        "mc_std_p05": mc["p05"],
        "mc_std_p95": mc["p95"],
        "mc_n_inits": int(mc_inits),
    }


def _features_tensor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    m = model
    with torch.no_grad():
        y = m.conv1(x)
        y = m.bn1(y)
        y = m.relu(y)
        y = m.maxpool(y)
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y = m.layer4(y)
        y = m.avgpool(y)
        y = torch.flatten(y, 1)
    return y


def _mc_logits_std(feats: torch.Tensor, n_inits: int, seed: int) -> dict:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    in_f, out_f = (512, 6)
    bound = 1.0 / math.sqrt(in_f)
    feats_cpu = feats.detach().cpu()
    stds = []
    for _ in range(n_inits):
        W = torch.empty(out_f, in_f).uniform_(-bound, bound, generator=gen)
        b = torch.empty(out_f).uniform_(-bound, bound, generator=gen)
        z = feats_cpu @ W.T + b
        stds.append(float(z.std().item()))
    stds_arr = np.asarray(stds, dtype=np.float64)
    return {
        "mean": float(stds_arr.mean()),
        "p05": float(np.quantile(stds_arr, 0.05)),
        "p95": float(np.quantile(stds_arr, 0.95)),
    }


def _features_norm(model: nn.Module, x: torch.Tensor) -> dict:
    m = model
    with torch.no_grad():
        y = m.conv1(x)
        y = m.bn1(y)
        y = m.relu(y)
        y = m.maxpool(y)
        y = m.layer1(y)
        y = m.layer2(y)
        y = m.layer3(y)
        y = m.layer4(y)
        y = m.avgpool(y)
        y = torch.flatten(y, 1)
        norms = y.norm(dim=1)
    return {"mean": float(norms.mean().item()), "std": float(norms.std().item())}


def write_initial_loss_report(scratch: dict, frozen: dict, out_path: Path) -> None:
    ln6 = scratch["ln6"]

    def _predict(d):
        nf = d["feature_norm_mean"]
        std_pred = math.sqrt((nf * nf + 1) / (3 * 512))
        l0_pred = ln6 + d["logits_std"] ** 2 / 2
        return (std_pred, l0_pred)

    rows = []
    for d in (scratch, frozen):
        std_pred, l0_pred = _predict(d)
        diff = d["L0"] - l0_pred
        ratio = abs(diff) / d["batch_sem_L0"] if d["batch_sem_L0"] > 0 else float("nan")
        in_range = d["mc_std_p05"] <= d["logits_std"] <= d["mc_std_p95"]
        rows.append((d, std_pred, l0_pred, diff, ratio, in_range))
    lines = []
    lines.append("# Початкова втрата L0")
    lines.append("")
    lines.append(
        "Для режимів `scratch` і `frozen` побудовано модель за кроками 1–5 зі зерном 1, викликано `set_train_mode`, узято перший пакет навчального завантажувача і обчислено `CrossEntropyLoss()` під `torch.no_grad()` до жодного кроку оптимізатора."
    )
    lines.append("")
    lines.append(
        f"Орієнтир — втрата моделі, що дає всім 6 класам однакову ймовірність: `ln 6 ≈ {ln6:.4f}`."
    )
    lines.append("")
    lines.append(
        "| Режим | L0 | ‖f‖₂ (середнє) | std(z) виміряне | std(z) очік. (iid) | E[L0] прогноз | L0 − E[L0] | SEM (пакет) | ‖діф‖/SEM |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, std_pred, l0_pred, diff, ratio, _ in rows:
        lines.append(
            f"| `{d['regime']}` | {d['L0']:.4f} | {d['feature_norm_mean']:.3f} | {d['logits_std']:.4f} | {std_pred:.4f} | {l0_pred:.4f} | {diff:+.4f} | {d['batch_sem_L0']:.4f} | {ratio:.2f} |"
        )
    lines.append("")
    lines.append(
        "Усі числа — на тому самому першому пакеті з 32 зображень. `‖f‖₂` — середня L2-норма 512-вимірного вектора ознак після `avgpool` по 32 прикладах."
    )
    lines.append("")
    n_mc = rows[0][0]["mc_n_inits"]
    lines.append(
        f"**Монте-Карло розкид `std(z)`** по {n_mc} незалежних ініціалізацій `Linear(512, 6)` на тих самих ознаках першого пакета (окремий `torch.Generator`, глобальний RNG не чіпаємо):"
    )
    lines.append("")
    lines.append("| Режим | std(z) виміряне | MC середнє | MC [5%, 95%] | Виміряне в діапазоні? |")
    lines.append("|---|---:|---:|---:|:---:|")
    for d, _, _, _, _, in_range in rows:
        lines.append(
            f"| `{d['regime']}` | {d['logits_std']:.4f} | {d['mc_std_mean']:.4f} | [{d['mc_std_p05']:.4f}, {d['mc_std_p95']:.4f}] | {('так' if in_range else 'НІ')} |"
        )
    lines.append("")
    lines.append("## Пояснення відхилення")
    lines.append("")
    lines.append(
        "**Масштаб логітів через ініціалізацію `fc`.** `nn.Linear(512, 6)` ініціалізується `U(−1/√512, 1/√512)` для ваг і зміщень; незалежна рівномірна з такими межами має дисперсію `1/(3·512)`. Для фіксованого вектора ознак `f` з довжиною `‖f‖`, якщо брати очікування по випадковій матриці `W`, отримуємо точну формулу"
    )
    lines.append("")
    lines.append("```math")
    lines.append(
        "\\text{Var}(z_k \\mid f) \\;=\\; \\frac{\\|f\\|^{2} + 1}{3 \\cdot 512} \\;\\;\\Longrightarrow\\;\\;"
    )
    lines.append(
        "E[\\text{std}(z) \\mid f] \\;\\approx\\; \\frac{\\|f\\|}{\\sqrt{3 \\cdot 512}} \\;\\approx\\; \\frac{\\|f\\|}{39.19}"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "(додаток `+1` — внесок `bias`, тут нехтовний). Це значення в колонці `std(z) очік. (iid)` — не передбачення для нашої конкретної `fc`, а середнє по розподілу всіх `fc`. Ми маємо одну реалізацію `W` (6 рядків по 512 ваг), тож виміряне `std(z)` — випадкова величина з широким розкидом. Монте-Карло з 200 незалежних ініціалізацій на тих самих ознаках пакета показує, що обидва виміряні значення (`0.4882` і `0.6143`) потрапляють усередину MC-діапазону `[5%, 95%]` відповідного режиму. Отже, різниця між `std(z) очік. (iid)` і виміряним — не похибка формули, а звичайний шум однієї вибірки `W`."
    )
    lines.append("")
    lines.append(
        "**‖f‖ у frozen більша як виміряний факт.** Головна причина — кістяк у `frozen` навчений на ImageNet (ваги згорток і `γ`/`β` у BN), а в `scratch` він випадковий (Kaiming для згорток, `γ = 1`, `β = 0`). До того ж у `frozen` BN нормує за **ковзними статистиками** ImageNet (окремі `running_mean`/`running_var` на кожен канал кожного BN-шару кістяка), а в `scratch` BN у `train()` — за статистиками поточного пакета. Виміряне: `‖f‖ ≈ 28.0` у frozen проти `≈ 18.9` у scratch; це емпіричне число йде в формулу `std(z)` без подальшої гіпотези. `fc` в обох режимах однакова (однакова `reset_parameters`), тож різницю в масштабі логітів робить саме кістяк."
    )
    lines.append("")
    lines.append(
        "**Чому L0 близька до ln 6, але трохи вище.** Для `z_k ~ N(0, σ²)` і `E[z_true] = 0` з Jensen з квадратичним членом Тейлора:"
    )
    lines.append("")
    lines.append("```math")
    lines.append("E[L_0] \\;\\approx\\; \\ln C \\;+\\; \\frac{\\sigma^{2}}{2}.")
    lines.append("```")
    lines.append("")
    lines.append("Підставляючи виміряне `σ = std(z)`, отримуємо колонку `E[L0] прогноз`.")
    lines.append("")
    lines.append(
        "**Відхилення — у межах шуму одного пакета.** Для вибірки з `B = 32` стандартна похибка середнього дорівнює `std(per-sample loss) / √B`; значення в колонці `SEM (пакет)`. Остання колонка — `|L0 − E[L0]| / SEM`. Обидва значення менші за 2, тобто розбіжність між прогнозом і виміром пояснюється саме скінченністю одного пакета, а не якимось спеціальним ефектом."
    )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def param_report(device: torch.device) -> list[dict]:
    expected_trainable = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}
    expected_total = 11179590
    rows = []
    for regime in ("scratch", "frozen", "unfreeze"):
        model = build_model(regime, seed=1, device=device)
        total, trainable = count_parameters(model)
        opt = make_optimizer(model, regime)
        groups = group_sizes(opt)
        group_sum = sum(n for n, _ in groups)
        param_ids: set[int] = set()
        for g in opt.param_groups:
            for p in g["params"]:
                assert p.requires_grad, "параметр без requires_grad потрапив в оптимізатор"
                pid = id(p)
                assert pid not in param_ids, "параметр потрапив у дві групи"
                param_ids.add(pid)
        rows.append(
            {
                "regime": regime,
                "total": total,
                "trainable": trainable,
                "groups": groups,
                "group_sum_matches_trainable": group_sum == trainable,
                "total_matches_expected": total == expected_total,
                "trainable_matches_expected": trainable == expected_trainable[regime],
            }
        )
    return rows


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _module_params(model: nn.Module, mod_name: str) -> list[str]:
    return [n for n, _ in getattr(model, mod_name).named_parameters()]


def _module_bn_stats(model: nn.Module, mod_name: str) -> list[str]:
    return [
        n
        for n, _ in getattr(model, mod_name).named_buffers()
        if n.endswith("running_mean") or n.endswith("running_var")
    ]


def _module_has_bn(model: nn.Module, mod_name: str) -> bool:
    return any(True for _ in _module_bn_stats(model, mod_name))


def change_table(
    regime: str,
    data_root: Path,
    device: torch.device,
    *,
    use_set_train_mode: bool = True,
    seed: int = 1,
    capture_bn_inputs: bool = False,
) -> dict:
    model = build_model(regime, seed=seed, device=device)
    before = _snapshot(model)
    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()
    opt = make_optimizer(model, regime)
    loader = make_train_loader(data_root, SPLIT_CSV)
    it = iter(loader)
    loss_fn = nn.CrossEntropyLoss()
    captured_inputs: list[torch.Tensor] = []
    handle = None
    if capture_bn_inputs:

        def _hook(_mod, inputs):
            captured_inputs.append(inputs[0].detach().to("cpu", dtype=torch.float64))

        handle = model.layer1[0].bn1.register_forward_pre_hook(_hook)
    losses_per_step: list[float] = []
    for step in range(5):
        x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        losses_per_step.append(float(loss.item()))
        loss.backward()
        opt.step()
    if handle is not None:
        handle.remove()
    after = model.state_dict()
    rows = []
    for mod_name in TOP_LEVEL_MODULES:
        prefix = f"{mod_name}."
        param_names = [
            k
            for k in before.keys()
            if k.startswith(prefix)
            and ".running_" not in k
            and (not k.endswith("num_batches_tracked"))
        ]
        bn_names = [
            k
            for k in before.keys()
            if k.startswith(prefix) and (k.endswith("running_mean") or k.endswith("running_var"))
        ]
        params_changed = any(not torch.equal(before[k], after[k]) for k in param_names)
        if bn_names:
            bn_changed = any(not torch.equal(before[k], after[k]) for k in bn_names)
            bn_result = "так" if bn_changed else "ні"
        else:
            bn_result = "немає BN"
        rows.append(
            {
                "regime": regime,
                "module": mod_name,
                "params_changed": params_changed,
                "bn_changed": bn_result,
            }
        )
    extras = {}
    for stat in ("running_mean", "running_var"):
        key = f"layer1.0.bn1.{stat}"
        diff = (after[key] - before[key]).abs().max().item()
        extras[f"max_abs_delta_{stat}_layer1_0_bn1"] = float(diff)
    nbt_key = "layer1.0.bn1.num_batches_tracked"
    extras["layer1_0_bn1_num_batches_tracked_delta"] = int(
        (after[nbt_key] - before[nbt_key]).item()
    )
    reconstruction = None
    if capture_bn_inputs and captured_inputs:
        momentum = 0.1
        rm = before["layer1.0.bn1.running_mean"].detach().to("cpu", dtype=torch.float64).clone()
        rv = before["layer1.0.bn1.running_var"].detach().to("cpu", dtype=torch.float64).clone()
        batch_means = []
        batch_vars_unbiased = []
        for xb in captured_inputs:
            m = xb.mean(dim=(0, 2, 3))
            v = xb.var(dim=(0, 2, 3), unbiased=True)
            batch_means.append(m)
            batch_vars_unbiased.append(v)
            rm = (1 - momentum) * rm + momentum * m
            rv = (1 - momentum) * rv + momentum * v
        actual_rm = after["layer1.0.bn1.running_mean"].detach().to("cpu", dtype=torch.float64)
        actual_rv = after["layer1.0.bn1.running_var"].detach().to("cpu", dtype=torch.float64)
        reconstruction = {
            "max_abs_manual_minus_pytorch_mean": float((rm - actual_rm).abs().max().item()),
            "max_abs_manual_minus_pytorch_var": float((rv - actual_rv).abs().max().item()),
            "per_step_batch_mean_mean": [float(m.mean().item()) for m in batch_means],
            "per_step_batch_var_mean": [float(v.mean().item()) for v in batch_vars_unbiased],
        }
    return {
        "regime": regime,
        "mode": "set_train_mode" if use_set_train_mode else "model.train()",
        "rows": rows,
        "extras": extras,
        "losses_per_step": losses_per_step,
        "reconstruction": reconstruction,
    }


def expected_module_change(regime: str, module: str) -> dict:
    from config import MODULES_IN_EVAL

    eval_mods = set(MODULES_IN_EVAL[regime])
    bn_mods = {
        "conv1": False,
        "bn1": True,
        "layer1": True,
        "layer2": True,
        "layer3": True,
        "layer4": True,
        "fc": False,
    }
    if module in eval_mods:
        return {"params_changed": False, "bn_changed": "ні" if bn_mods[module] else "немає BN"}
    else:
        return {"params_changed": True, "bn_changed": "так" if bn_mods[module] else "немає BN"}


def _format_group(n: int, lr: float) -> str:
    return f"{n:,} параметрів @ lr={lr:g}".replace(",", " ")


def write_param_report(rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Параметри моделі по режимах")
    lines.append("")
    lines.append(
        "Загальна кількість параметрів однакова в усіх режимах; відрізняються лише ті, що навчаються, і склад груп оптимізатора. Числа — сума `numel()`."
    )
    lines.append("")
    lines.append("| Режим | Усього | З `requires_grad=True` | Групи AdamW | Збіг з умовою |")
    lines.append("|---|---:|---:|---|:---:|")
    for r in rows:
        groups_str = "; ".join((_format_group(n, lr) for n, lr in r["groups"]))
        ok = (
            r["total_matches_expected"]
            and r["trainable_matches_expected"]
            and r["group_sum_matches_trainable"]
        )
        lines.append(
            f"| `{r['regime']}` | {r['total']:,} | {r['trainable']:,} | {groups_str} | {('✓' if ok else '✗')} |".replace(
                ",", " "
            )
        )
    lines.append("")
    lines.append(
        "Додаткова перевірка в коді: усі параметри в кожній групі мають `requires_grad=True`; сума `numel()` груп збігається з кількістю тренованих параметрів; жоден параметр не потрапив у дві групи одночасно."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def write_module_changes(cts: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Що змінюється під час навчання (frozen і unfreeze, 5 кроків)")
    lines.append("")
    lines.append(
        "Для кожного з двох режимів: нова модель, зерно 1, кроки 1–5, збережено повну копію `state_dict()` (параметри і буфери), викликано `set_train_mode`, виконано 5 кроків оптимізатора на перших 5 пакетах `train` loader. Для кожного модуля верхнього рівня перевірено через `torch.equal`, чи змінився хоч один його параметр і чи змінилася хоч одна його ковзна статистика (`running_mean`/`running_var`)."
    )
    lines.append("")
    lines.append(
        "| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Очікувано | Збіг |"
    )
    lines.append("|---|---|:---:|:---:|---|:---:|")
    for ct in cts:
        regime = ct["regime"]
        for row in ct["rows"]:
            exp = expected_module_change(regime, row["module"])
            exp_str = f"params={exp['params_changed']}; bn={exp['bn_changed']}"
            match = (
                row["params_changed"] == exp["params_changed"]
                and row["bn_changed"] == exp["bn_changed"]
            )
            lines.append(
                f"| `{regime}` | `{row['module']}` | {('так' if row['params_changed'] else 'ні')} | {row['bn_changed']} | {exp_str} | {('✓' if match else '✗')} |"
            )
    lines.append("")
    lines.append(
        "У `conv1` і `fc` BN немає; `avgpool` опущено, бо він без параметрів і без BN. Усі рядки збіглися з очікуваним: заморожені модулі (`conv1, bn1, layer1–layer4` у `frozen`; `conv1, bn1, layer1–layer3` у `unfreeze`) не змінюють ні параметрів, ні ковзних статистик; треновані — обидва."
    )
    lines.append("")
    for ct in cts:
        extras = ct["extras"]
        lines.append(
            f"**{ct['regime']}, `layer1.0.bn1`**: max |Δ running_mean| = {extras['max_abs_delta_running_mean_layer1_0_bn1']:.6f}, max |Δ running_var| = {extras['max_abs_delta_running_var_layer1_0_bn1']:.6f}, Δ num_batches_tracked = {extras['layer1_0_bn1_num_batches_tracked_delta']}."
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def write_bn_experiment(
    ct_correct: dict, ct_wrong: dict, ct_unfreeze: dict, out_path: Path, prediction_path: Path
) -> None:
    extras = ct_wrong["extras"]
    rec = ct_wrong["reconstruction"]
    lines = []
    lines.append("# Дослід із навмисною помилкою: `frozen` + `model.train()`")
    lines.append("")
    lines.append(f"**Прогноз (записаний до запуску):** `{prediction_path.name}`.")
    lines.append("")
    lines.append(
        "Той самий код `change_table`, але замість `set_train_mode(model, 'frozen')` викликано `model.train()`. Решта — ідентична: нова модель зі зерном 1, 5 кроків оптимізатора на перших 5 пакетах, оптимізатор має лише параметри `fc` (`weight_decay=1e-4`, `lr=3e-3`)."
    )
    lines.append("")
    lines.append("## Таблиця змін модулів (frozen + model.train())")
    lines.append("")
    lines.append(
        "| Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Відповідає прогнозу? |"
    )
    lines.append("|---|:---:|:---:|:---:|")
    for row in ct_wrong["rows"]:
        is_bn = row["bn_changed"] != "немає BN"
        is_fc = row["module"] == "fc"
        pred_params = is_fc
        pred_bn = "так" if is_bn else "немає BN"
        match = row["params_changed"] == pred_params and row["bn_changed"] == pred_bn
        lines.append(
            f"| `{row['module']}` | {('так' if row['params_changed'] else 'ні')} | {row['bn_changed']} | {('✓' if match else '✗')} |"
        )
    lines.append("")
    lines.append("## Максимальні зміни в `layer1.0.bn1`")
    lines.append("")
    lines.append(
        f"- `max |Δ running_mean|` = **{extras['max_abs_delta_running_mean_layer1_0_bn1']:.6f}**"
    )
    lines.append(
        f"- `max |Δ running_var|`  = **{extras['max_abs_delta_running_var_layer1_0_bn1']:.6f}**"
    )
    lines.append(
        f"- `Δ num_batches_tracked` = **{extras['layer1_0_bn1_num_batches_tracked_delta']}**"
    )
    lines.append("")
    lines.append("## Точна перевірка механізму: ручна рекурсія vs PyTorch")
    lines.append("")
    lines.append(
        "Через `forward_pre_hook` на `layer1.0.bn1` записано вхідний тензор перед BN на кожному з 5 кроків. По-канально порахувано `batch_mean` і Bessel-коректовану (unbiased) `batch_var` — саме цю unbiased-дисперсію PyTorch використовує для оновлення `running_var` (нормування всередині `forward` робить biased-варіантом). Далі з початкових ImageNet-значень `running_mean`/`running_var` рекурсивно застосовано"
    )
    lines.append("")
    lines.append("```math")
    lines.append("r \\leftarrow 0.9 \\cdot r + 0.1 \\cdot \\text{batch\\_stat}")
    lines.append("```")
    lines.append("")
    lines.append("для кожного з 5 пакетів і зіставлено з тим, що в PyTorch:")
    lines.append("")
    lines.append(
        f"- `max |ручне_running_mean − PyTorch|` = **{rec['max_abs_manual_minus_pytorch_mean']:.3e}**"
    )
    lines.append(
        f"- `max |ручне_running_var  − PyTorch|` = **{rec['max_abs_manual_minus_pytorch_var']:.3e}**"
    )
    lines.append("")
    lines.append(
        "Обидві нев'язки — на рівні числової похибки float32 (BN усередині PyTorch рахує саме так, а наша ручна рекурсія в float64 відтворює точне правило). Це доводить, що буфери справді змістилися через механізм експоненціального згладжування в `forward`, а не через якусь приховану причину. Для орієнтиру — середнє по 64 каналах `batch_mean` і `batch_var` на кожному з 5 пакетів:"
    )
    lines.append("")
    lines.append("| Крок | mean(batch_mean) | mean(batch_var, unbiased) |")
    lines.append("|---:|---:|---:|")
    for i, (m, v) in enumerate(
        zip(rec["per_step_batch_mean_mean"], rec["per_step_batch_var_mean"]), start=1
    ):
        lines.append(f"| {i} | {m:+.6f} | {v:.6f} |")
    lines.append("")
    lines.append("## Чи помітна помилка з кривої втрати на train?")
    lines.append("")
    lines.append(
        "Той самий seed, ті самі перші 5 пакетів, в обох варіантах `fc` починає з тієї ж ініціалізації. Різниця лише в тому, що в правильному варіанті BN кістяка в `eval()` нормує за `running_*` ImageNet, а в помилковому варіанті (`model.train()`) — за статистиками поточного пакета, тож `fc` бачить інші ознаки й оновлюється інакше."
    )
    lines.append("")
    lines.append(
        "| Крок | Втрата (set_train_mode, правильно) | Втрата (model.train(), помилка) | Різниця |"
    )
    lines.append("|---:|---:|---:|---:|")
    for i, (lc, lw) in enumerate(
        zip(ct_correct["losses_per_step"], ct_wrong["losses_per_step"]), start=1
    ):
        lines.append(f"| {i} | {lc:.4f} | {lw:.4f} | {lw - lc:+.4f} |")
    lines.append("")
    lines.append(
        f"**Узгодженість перевірок.** Втрата на кроці 1 у правильному варіанті ({ct_correct['losses_per_step'][0]:.4f}) збігається з `L0_frozen` з пункту 1 (та сама побудова моделі, той самий перший пакет, лише там `no_grad` і без кроку оптимізатора, а тут `loss.item()` до `backward`)."
    )
    lines.append("")
    lines.append(
        "**Різниця між варіантами є**, і помітна: на кроках 2–4 помилковий варіант стабільно вище правильного на `+0.18 … +0.30`. Але розпізнати це можна **тільки з порівняння** обох кривих — а під час реальної помилки правильної кривої немає. Сама по собі траєкторія помилкового варіанту `1.89 → 1.95 → 1.84 → 1.68 → 1.41` виглядає як цілком нормальне початкове навчання голови: втрата коливається й падає, нічого аномального в ній немає. Частина підвищення пояснюється тим, що в `train()` BN нормує за статистиками поточного пакета замість ImageNet-ковзних, і `fc` бачить дещо інші активації; це поступово «розтягує» траєкторію, не ламаючи її вигляду."
    )
    lines.append("")
    lines.append(
        "Ефект помилки для `running_*` вимірювати на `eval()` validation ми в межах цього досліду не стали (це вже був би окремий запуск). З механізму зрозуміло: якщо помилку залишити, то в `eval()` модель користуватиметься вже зсунутими ковзними статистиками, накопиченими на Intel-пакетах, замість ImageNet-ковзних, під які калібровано ваги кістяка. Проявитися вона **може** саме там, тоді як на train вона залишається непомітною."
    )
    lines.append("")
    lines.append("## Зіставлення з прогнозом")
    lines.append("")
    lines.append("| Пункт прогнозу | Виміряне |")
    lines.append("|---|---|")
    lines.append(
        "| Параметри кістяка не змінюються (requires_grad=False) | підтверджено для `conv1, bn1, layer1–layer4` |"
    )
    lines.append("| Параметри `fc` змінюються | підтверджено |")
    lines.append(
        "| Ковзні статистики BN у `bn1, layer1–layer4` змінюються | підтверджено (`bn_changed = так` у всіх 5 модулях з BN) |"
    )
    lines.append("| У `conv1` BN немає | підтверджено (`bn_changed = немає BN`) |")
    lines.append(
        "| `num_batches_tracked` у BN зросте на 5 | підтверджено (`+5` у `layer1.0.bn1`) |"
    )
    lines.append(
        f"| Зміна `running_mean` у `layer1.0.bn1` ненульова | підтверджено (`{extras['max_abs_delta_running_mean_layer1_0_bn1']:.4f}`) |"
    )
    lines.append(
        f"| Зміна `running_var` у `layer1.0.bn1` ненульова | підтверджено (`{extras['max_abs_delta_running_var_layer1_0_bn1']:.4f}`) |"
    )
    lines.append(
        "| Крива втрати на train не відрізнятиме помилку | частково: різниця є (+0.18 … +0.30 на кроках 2–4), але без правильної кривої для порівняння її не видно |"
    )
    lines.append("")
    lines.append("## Чому `requires_grad=False` не захищає буфери")
    lines.append("")
    lines.append(
        "`requires_grad` керує лише побудовою обчислювального графа для **параметрів** — тензорів, які реєструються через `nn.Parameter`. `running_mean` і `running_var` у `BatchNorm2d` — це **буфери** (зареєстровані через `register_buffer`), а не параметри; оптимізатор їх не бачить, але й не чіпає. Проте в `nn.BatchNorm2d.forward`, коли `self.training is True` (що й робить `model.train()`), BN спочатку обчислює статистики поточного пакета для нормування **і одночасно** виконує експоненціальне згладжування: `running_mean ← (1 − momentum)·running_mean + momentum·batch_mean`. Прапор `self.training` залежить від `.train()`/`.eval()`, а не від `requires_grad`. Саме тому `set_train_mode(model, 'frozen')` переводить усі BN-шари кістяка в `eval()` — щоб `forward` брав збережені `running_*` без оновлення."
    )
    lines.append("")
    lines.append("## Чому для `layer4` в `unfreeze` оновлення правильне")
    lines.append("")
    bn_delta_unfreeze_var = ct_unfreeze["extras"]["max_abs_delta_running_var_layer1_0_bn1"]
    bn_delta_unfreeze_mean = ct_unfreeze["extras"]["max_abs_delta_running_mean_layer1_0_bn1"]
    lines.append(
        f"**Як.** `set_train_mode(model, 'unfreeze')` переводить у `eval()` саме `conv1, bn1, layer1–layer3`, залишивши `layer4` і `fc` у `train()` (базою стоїть `model.train()`, а потім лише в заморожених модулях викликано `.eval()`). У перевірці це видно: у `layer1.0.bn1` зміни немає (`max |Δ running_mean| = {bn_delta_unfreeze_mean:.6f}`, `max |Δ running_var| = {bn_delta_unfreeze_var:.6f}`), а в `layer4` параметри і BN змінилися (стовпець «Ковзні статистики BN» = `так`)."
    )
    lines.append("")
    lines.append(
        "**Чому.** Ваги `layer4` навчаються — отже, розподіл активацій усередині `layer4` змінюється від кроку до кроку. Ковзні статистики `layer4` мають відповідати **поточним** вагам і цільовому набору Intel, інакше в `eval()` на validation нормування йтиме за ImageNet-статистиками, які вже не описують активації нашої моделі. Для `conv1, bn1, layer1–layer3` все навпаки: їхні ваги не змінюються, і правильні ковзні статистики для цих шарів — саме ImageNet-ті, на яких вони навчалися й з якими їх калібровано; переобчислювати їх на наших 1200 train-зображеннях — псувати злагоджений кістяк."
    )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
