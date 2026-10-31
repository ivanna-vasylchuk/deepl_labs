# Що змінюється під час навчання (frozen і unfreeze, 5 кроків)

Для кожного з двох режимів: нова модель, зерно 1, кроки 1–5, збережено повну копію `state_dict()` (параметри і буфери), викликано `set_train_mode`, виконано 5 кроків оптимізатора на перших 5 пакетах `train` loader. Для кожного модуля верхнього рівня перевірено через `torch.equal`, чи змінився хоч один його параметр і чи змінилася хоч одна його ковзна статистика (`running_mean`/`running_var`).

| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Очікувано | Збіг |
|---|---|:---:|:---:|---|:---:|
| `frozen` | `conv1` | ні | немає BN | params=False; bn=немає BN | ✓ |
| `frozen` | `bn1` | ні | ні | params=False; bn=ні | ✓ |
| `frozen` | `layer1` | ні | ні | params=False; bn=ні | ✓ |
| `frozen` | `layer2` | ні | ні | params=False; bn=ні | ✓ |
| `frozen` | `layer3` | ні | ні | params=False; bn=ні | ✓ |
| `frozen` | `layer4` | ні | ні | params=False; bn=ні | ✓ |
| `frozen` | `fc` | так | немає BN | params=True; bn=немає BN | ✓ |
| `unfreeze` | `conv1` | ні | немає BN | params=False; bn=немає BN | ✓ |
| `unfreeze` | `bn1` | ні | ні | params=False; bn=ні | ✓ |
| `unfreeze` | `layer1` | ні | ні | params=False; bn=ні | ✓ |
| `unfreeze` | `layer2` | ні | ні | params=False; bn=ні | ✓ |
| `unfreeze` | `layer3` | ні | ні | params=False; bn=ні | ✓ |
| `unfreeze` | `layer4` | так | так | params=True; bn=так | ✓ |
| `unfreeze` | `fc` | так | немає BN | params=True; bn=немає BN | ✓ |

У `conv1` і `fc` BN немає; `avgpool` опущено, бо він без параметрів і без BN. Усі рядки збіглися з очікуваним: заморожені модулі (`conv1, bn1, layer1–layer4` у `frozen`; `conv1, bn1, layer1–layer3` у `unfreeze`) не змінюють ні параметрів, ні ковзних статистик; треновані — обидва.

**frozen, `layer1.0.bn1`**: max |Δ running_mean| = 0.000000, max |Δ running_var| = 0.000000, Δ num_batches_tracked = 0.
**unfreeze, `layer1.0.bn1`**: max |Δ running_mean| = 0.000000, max |Δ running_var| = 0.000000, Δ num_batches_tracked = 0.
