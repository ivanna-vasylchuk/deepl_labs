# Вимога умови → де виконано

| № | Вимога умови | Де виконано (файл / функція) | У README | Рисунок |
|---:|---|---|---|---|
| 1 | **Поділ**: `rng = np.random.default_rng(0)`, train 1200 / val 1498 / test 1502 | `data.py::build_split`, `data.py::_sorted_names` | «Дані та поділ» | — |
| 2 | **Контрольна таблиця з 18 клітинок + підсумки**, збіг з еталоном | `data.py::control_table`, `results/split_check.md` | «Дані та поділ» | — |
| 3 | **split.csv** зі стовпцями `file,label,part`, стабільний | `data.py::build_split`, SHA в `split_check.md` | «Дані та поділ» | — |
| 4 | **Жоден файл не входить у дві частини** | `data.py::control_table` (train∩val, train∩test, val∩test) | «Дані та поділ» | — |
| 5 | **Перетворення train/val** за умовою, той самий mean/std у scratch | `data.py::make_train_transform`, `make_eval_transform`; `config.py::MEAN,STD` | «Пункт 1», `run.json.transforms` | — |
| 6 | **Побудова моделі кроки 1–5**, `IMAGENET1K_V1`, `Linear(512,6)`, `requires_grad=False` для не-навчальних | `model.py::build_model` | — | — |
| 7 | **AdamW**, групи й lr з таблиці режимів, `weight_decay=1e-4`, параметри без `requires_grad` не входять | `model.py::make_optimizer` + перевірка в `checks.py::param_report` | «Пункт 2» | — |
| 8 | **BN за замовчуванням** (`momentum=0.1, eps=1e-5`) | `nn.BatchNorm2d` без модифікацій | «Пункт 2» (`bn` в `run.json`) | — |
| 9 | **set_train_mode** на початку КОЖНОЇ епохи | `model.py::set_train_mode`, виклик в `engine.py::train_one_run` | — | — |
| 10 | **train loader** `bs=32, shuffle=True, drop_last=False, num_workers=0`, без власного generator/sampler | `data.py::make_train_loader` | «Структура» | — |
| 11 | **val/test loader** `bs=64, shuffle=False, num_workers=0` | `data.py::make_eval_loader` | — | — |
| 12 | **Бюджет 8 × 38 = 304 кроки**, останній пакет 16 зображень | `config.py::EPOCHS,STEPS_PER_EPOCH` + `drop_last=False` | «Пункт 3» | — |
| 13 | **Втрата епохи** = `∑ L_b·|b| / 1200` | `engine.py::train_one_run` (`weighted_sum / 1200.0`) | «Структура» | — |
| 14 | **F_r,s** = macro-F1 після 8-ї епохи | `engine.py::evaluate`, остання епоха в `per_epoch_val_macro_f1` | «Теорія» | — |
| 15 | **Заборонено** (lr-розклад, grad-clip, mixed precision, label smoothing, class weights, early stop, вибір епохи, інші аугментації, TTA) | нічого з цього немає; `grep` у коді | «Відтворюваність» | — |
| **Пункт 1** | **Поділ**: таблиця контролю, збіг | `results/split_check.md` | «Пункт 1» | — |
| **Пункт 1** | **Пакет після перетворень**: рисунок перших 16 з аугментаціями, оберненим Normalize і підписами | `checks.py::save_batch_figure` → `figures/batch.png` | «Пункт 1» | `figures/batch.png` |
| **Пункт 1** | **L0 для scratch і frozen** проти ln 6, пояснення | `checks.py::compute_L0`, `write_initial_loss_report` → `results/initial_loss.md` | «Пункт 1» | — |
| **Пункт 2** | **Параметри по режимах** (сума numel, групи, перевірка складу оптимізатора) | `checks.py::param_report`, `write_param_report` → `results/param_table.md` | «Пункт 2» | — |
| **Пункт 2** | **Що змінюється**: таблиця `set_train_mode` для `frozen` і `unfreeze`, `torch.equal` без допуску, 7 модулів | `checks.py::change_table`, `write_module_changes` → `results/module_changes.md` | «Пункт 2» | — |
| **Пункт 2** | **Дослід із BN**: прогноз ДО запуску, `model.train()` замість `set_train_mode`, max |Δ running| у layer1.0.bn1, зіставлення | `results/bn_experiment_prediction.md` (ДО), `checks.py::write_bn_experiment` → `results/bn_experiment.md` | «Пункт 2» | — |
| **Пункт 2** | **Чому `requires_grad=False` не захищає буфери**; **чому для `layer4` в unfreeze оновлення правильне** | `results/bn_experiment.md` розділи «Чому …» | «Пункт 2» | — |
| **Пункт 3** | **Сім запусків** протоколу (scratch 1; frozen 1,2,3; unfreeze 1,2,3) | `train.py::DEFAULT_RUN_ORDER`, `engine.py::train_one_run` → `runs/<r>_seed<s>/` | «Пункт 3» | — |
| **Пункт 3** | **Журнал запуску (run.json)**: усі налаштування, SHA split, версії, пристрій, зерно, час, per-epoch | `engine.py::train_one_run` | «Пункт 3», `run.json` | — |
| **Пункт 3** | **val_logits.npy** (1498, 6) float32 у порядку validation | `engine.py` (asserts shape/dtype) | «Структура» | — |
| **Пункт 3** | **Хеш split.csv** записується, `analyze.py` звіряє з поточним | `engine.py::_file_sha256`, `analyze.py::check_integrity` | «Відтворюваність» | — |
| **Пункт 3** | **Таблиця семи запусків** і **рисунок двох панелей** на спільних осях | `analyze.py::table_seven_runs`, `draw_curves` → `figures/curves.png`, `results/analysis.md` | «Пункт 3» | `figures/curves.png` |
| **Пункт 4** | **F̄_r, R_r** для frozen/unfreeze; **R, Δpre, Δunf**, сума | `analyze.py::effects_table` (використовує `selection.py`) | «Пункт 4» | — |
| **Пункт 4** | **Правило `|Δ| vs R`** для кожного Δ | `analyze.py::effects_table::verdict` | «Пункт 4» | — |
| **Пункт 4** | Пояснення: що змінює зерно; чому розкид різний; чому R не описує запуск 1; однаковий бюджет за часом | README «Пункт 4»; числа часу з локального запуску `measure_time.py` (JSON-вивід у репозиторій не додано) | «Пункт 4» | — |
| **Пункт 5** | **Вибір за validation**: argmax F̄, argmax F_r,s, детерміноване розв'язання нічиїх | `selection.py::choose`, `train.py::do_select` | «Пункт 5» | — |
| **Пункт 5** | **lab3_model.pt** `{state_dict, regime, seed, classes, img_size=160}`, state_dict на CPU | `train.py::do_select` | «Структура», «Пункт 5» | — |
| **Пункт 5** | Якщо `|Δunf| ≤ R`, явно сказати «визначив шум» | README «Пункт 5» + `results/analysis.md` | «Пункт 5» | — |
| **Пункт 5** | **Test — один раз, лише в analyze.py**, на CPU; перевірка хешу splits і вибору ДО | `analyze.py::main` (test після `check_integrity` і `verify_selection`); `test_eval.py` імпортується лише з analyze.py | «Пункт 5» | — |
| **Пункт 5** | **macro-F1 на test, матриця плутанини 6×6** (таблиця + рисунок) | `analyze.py::do_test`, `draw_confusion` → `figures/confusion.png` | «Пункт 5» | `figures/confusion.png` |
| **Пункт 5** | **SE = √(p(1−p)/n)** на test; порівняння з F_r,s обраного; обговорення оптимістичного зміщення | `test_eval.py::evaluate_test`, `analyze.py::main` (val_accuracy + SE_val + SE_diff + best_of_3) | «Пункт 5» | — |
| **Пункт 5** | **П'ять характерних помилок** за правилом нічиїх | `analyze.py::pick_five_errors` + `draw_errors` → `figures/errors.png` | «Пункт 5» | `figures/errors.png` |
| **Пункт 5** | Категорія кожної помилки, glacier↔mountain як «властивість набору» (після перегляду зображень) | README «Пункт 5» | «Пункт 5» | `figures/errors.png` |
| **Пункт 6** | **Висновок**: Δpre і Δunf окремо, напрям → величина в п.п. → співвідношення з R; явний бюджет 304 кроки; окреме речення — який режим для нової задачі з 200 зобр/клас | README «Висновки» | «Висновки» | — |
| **Що здати** | **Код у .py-файлах**, залежності з фіксованими версіями, дві команди через uv | `pyproject.toml, uv.lock, .python-version`; `train.py`, `analyze.py` | «Запуск», «Структура» | — |
| **Що здати** | `split.csv` | у корені lab3/ | «Структура» | — |
| **Що здати** | `runs/` з журналами й логітами всіх 7 запусків | `runs/<r>_seed<s>/{run.json, val_logits.npy}` | «Структура» | — |
| **Що здати** | `lab3_model.pt` вибраного запуску (≈45 МБ) | GitHub Release `lab3` (42.7 МБ; у репозиторій не додано через ліміт 25 МБ на завантаження через браузер) | «Структура», «Пункт 5» | — |
| **Що здати** | Рисунки (пакет, криві, матриця, помилки) | `figures/*.png` | «Структура» | — |
| **Що здати** | README | `README.md` | — | — |
| **Що здати** | Набір НЕ додавати | набору в репозиторії немає, він лежить поза lab3/ (`~/data/intel/`) | — | — |
| **Відтворюваність** | Відхилення оголошено (CUDA-збірка; commit_hash=null) | README «Відтворюваність» | — | — |
