# Geo Features MTZ

Проект для экспериментов по решению обратной задачи МТЗ с двумя основными идеями уменьшения размерности признакового пространства:

- **агрегация признаков** по частотам и пикетам;
- **PCA** (метод главных компонент).

В качестве базовой модели используется нейросеть **OLP**. Название сохранено намеренно, но фактически модель поддерживает два режима:

- линейный слой без скрытого слоя;
- небольшой MLP с одним скрытым слоем, если задан `hidden_dim`.

---

## Что умеет проект

Проект позволяет:

- загружать train/valid/test наборы данных;
- явно разделять признаки и целевые переменные;
- агрегировать геофизические признаки по структурным правилам;
- применять PCA только к признакам, без смешивания с target-колонками;
- запускать серии экспериментов с несколькими итерациями обучения;
- сохранять результаты экспериментов в структурированную папку;
- при необходимости сохранять или не сохранять данные после aggregation/PCA;
- запускать обучение на GPU через `device='auto'`, если CUDA доступна;
- строить графики сравнения результатов;
- проводить отбор признаков отдельными утилитами.

---

## Структура проекта

```text
Geo_Features/
├── src/
│   ├── config.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   └── metrics.py
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   └── pca_experiment.py
│   ├── models/
│   │   ├── feature_selection.py
│   │   └── neural_network.py
│   ├── preprocessing/
│   │   ├── aggregation.py
│   │   └── pca.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   └── visualization/
│       └── experiment_plots.py
├── tests/
│   ├── test_core.py
│   ├── test_data_diagnostics.py
│   ├── test_pca_module.py
│   └── test_visualization.py
├── Data/
├── experiments/
├── run_aggregation_experiment.py
├── run_pca_experiment.py
├── run_all_experiments.py
├── requirements.txt
└── README.md
```

---

## Формат данных

### Признаки

Ожидаемый формат имён признаков:

```text
{COMPONENT}{POLARIZATION}{FREQUENCY}_{PICKUP}
```

Примеры:

- `REYX1_1`
- `IMYX13_31`
- `REHX5_15`

Где:

- `COMPONENT`: `RE` или `IM`
- `POLARIZATION`: `YX`, `XY`, `HX`
- `FREQUENCY`: номер частоты
- `PICKUP`: номер пикета

### Целевые переменные

Ожидаемый формат target-колонок:

```text
H{layer}_{pickup}
```

Примеры:

- `H1_8`
- `H2_8`
- `H3_8`

---

## Главные модули и их роли

## `src/utils/data_loader.py`

Содержит:

- `load_mtz_data(...)` — загрузка CSV-файлов train/valid/test;
- `Data` — безопасная обёртка над train/valid/test с проверкой схемы;
- `split_features_targets(...)` — разделение DataFrame на `X` и `y`.

### Что делает `Data`

Класс `Data`:

- проверяет, что у train/valid/test одинаковые колонки и одинаковый порядок колонок;
- проверяет наличие target-колонок во всех сплитах;
- формирует:
  - `X_train`, `X_valid`, `X_test`
  - `y_train`, `y_valid`, `y_test`
- хранит:
  - `n_features`
  - `n_targets`
  - `target_columns`

### Пример

```python
from pathlib import Path
from src.utils import load_mtz_data, Data

train, valid, test = load_mtz_data(
    Path('Data/mtsgrvmgn_trn.csv'),
    Path('Data/mtsgrvmgn_vld.csv'),
    Path('Data/mtsgrvmgn_tst.csv')
)

data = Data(
    train=train,
    test=test,
    valid=valid,
    columns=['H3_8']
)

print(data.info())
```

---

## `src/preprocessing/aggregation.py`

Этот модуль отвечает за агрегацию признаков.

### Идея

Если исходных признаков слишком много, можно объединять их по:

- частотам;
- пикетам;
- выбранному способу агрегации.

### Поддерживаемые методы агрегации

- `mean`
- `median`
- `max`
- `min`
- `std`
- `range`

### Основной интерфейс

```python
from src.preprocessing import AggregationConfig, aggregate_features

config = AggregationConfig(
    freq_step=2,
    pickup_step=3,
    agg_method='mean'
)

train_agg, valid_agg, test_agg, new_features = aggregate_features(
    train,
    valid,
    test,
    config,
    target_columns=['H3_8']
)
```

### Что возвращается

- `train_agg`, `valid_agg`, `test_agg` — новые DataFrame;
- `new_features` — список новых признаков после агрегации.

---

## `src/preprocessing/pca.py`

Этот модуль реализует PCA.

### Основные сущности

- `PCATransformer`
- `apply_pca_to_data(...)`
- `plot_explained_variance(...)`
- `find_optimal_n_components(...)`

### Идея

PCA обучается только на train-признаках, после чего тем же трансформером преобразуются valid и test.

### Пример

```python
from src.preprocessing.pca import apply_pca_to_data

train_pca, valid_pca, test_pca, transformer = apply_pca_to_data(
    train,
    valid,
    test,
    target_columns=['H3_8'],
    n_components=100
)

print(transformer.n_components_used)
print(transformer.cumulative_variance[-1])
```

---

## `src/models/neural_network.py`

Это ядро обучения модели OLP.

## Что внутри

- `SingleLayerPerceptron`
- `OLP(...)`
- `to_excel_optimized_OLP(...)`

## Что делает `OLP(...)`

Функция:

1. проверяет пропуски;
2. масштабирует признаки и target;
3. создаёт DataLoader;
4. обучает модель;
5. использует early stopping по validation loss;
6. восстанавливает лучшую модель;
7. считает метрики на test;
8. опционально делает встроенную кросс-валидацию по train;
9. возвращает:
   - `df_test`
   - `df_cv`
   - `df_cv_err`
   - `training_history`

## Важные детали текущей реализации

- `weight_decay` полностью удалён из проекта;
- сохранение лучшего состояния модели делается через `copy.deepcopy`;
- добавлен `random_state` для воспроизводимости;
- loss усредняется по числу объектов, а не по числу батчей;
- `tolerance` поддерживает два режима: `absolute` и `relative`;
- по умолчанию используется `tolerance_mode='relative'`, потому что это устойчивее при сравнении моделей с разным числом признаков;
- в относительном режиме early stopping работает по правилу:

```python
val_loss < best_val_loss - abs(best_val_loss) * tolerance
```

- в абсолютном режиме правило остаётся таким:

```python
val_loss < best_val_loss - tolerance
```

### Пример прямого вызова `OLP`

```python
from src.models.neural_network import OLP

results_test, results_cv, results_cv_err, history = OLP(
    X_train=data.X_train,
    y_train=data.y_train,
    X_valid=data.X_valid,
    y_valid=data.y_valid,
    X_test=data.X_test,
    y_test=data.y_test,
    batch_size=64,
    input_dim=data.n_features,
    output_dim=data.n_targets,
    learning_rate=0.01,
    num_epochs=1000,
    patience=100,
    tolerance=1e-4,
    tolerance_mode='relative',
    hidden_dim=32,
    optimizer_type='adam',
    enable_cv=False
)
```

---

## `src/experiment/aggregation.py`

Содержит класс `AggregationExperiment`.

### За что отвечает

- применяет агрегацию к train/valid/test;
- запускает OLP на агрегированных данных;
- опционально сохраняет агрегированные train/valid/test;
- сохраняет результаты в папку эксперимента;
- собирает метрики в `ExperimentResult`;
- строит сравнение результатов.

### Основные методы

- `run_single(...)`
- `run_grid(...)`
- `get_results_df()`
- `plot_comparison()`
- `get_best_result()`
- `summary()`

`plot_comparison()` теперь дополнительно строит графики:
- абсолютного времени обучения от числа признаков/компонент;
- относительного времени обучения, нормированного на модель с полным исходным набором признаков.

### Пример

```python
from src.experiment import AggregationExperiment

experiment = AggregationExperiment(
    train=train,
    valid=valid,
    test=test,
    target_columns=['H3_8'],
    experiment_name='aggregation_study'
)

results = experiment.run_grid(
    freq_steps=[1, 2, 3],
    pickup_steps=[1, 2, 3],
    agg_methods=['mean'],
    n_iter=3,
    num_epochs=1000,
    learning_rate=0.01,
    optimizer='adam',
    patience=250,
    tolerance=0.003,
    tolerance_mode='relative',
    hidden_dim=32,
    batch_size=64
)

print(results.head())
print(experiment.summary())
```

---

## `src/experiment/pca_experiment.py`

Содержит класс `PCAExperiment`.

### За что отвечает

- применяет PCA к train/valid/test;
- запускает OLP на PCA-признаках;
- опционально сохраняет PCA-данные, `pca_info.json` и график explained variance;
- сохраняет метрики и summary;
- сравнивает результаты по числу компонент.

### Основные методы

- `run_single(...)`
- `run_grid(...)`
- `get_results_df()`
- `plot_comparison()`
- `get_best_result()`
- `summary()`

`plot_comparison()` теперь дополнительно строит графики:
- абсолютного времени обучения от числа признаков/компонент;
- относительного времени обучения, нормированного на модель с полным исходным набором признаков.

### Пример

```python
from src.experiment import PCAExperiment

experiment = PCAExperiment(
    train=train,
    valid=valid,
    test=test,
    target_columns=['H3_8'],
    experiment_name='pca_study'
)

results = experiment.run_grid(
    n_components_list=[100, 200, 500, 1000],
    n_iter=3,
    num_epochs=1000,
    learning_rate=0.01,
    optimizer='adam',
    patience=250,
    tolerance=0.003,
    tolerance_mode='relative',
    hidden_dim=32,
    batch_size=64
)
```

---

## `src/evaluation/experiment.py`

Содержит:

- `ExperimentResult` — единый dataclass результата эксперимента;
- `ExperimentManager` — сохранение/загрузка результатов;
- `extract_metrics_from_excel(...)` — чтение summary-метрик из Excel.

### Что хранит `ExperimentResult`

Для aggregation и PCA в одном формате сохраняются:

- идентификатор эксперимента;
- timestamp;
- параметры модели;
- параметры агрегации;
- PCA-параметры;
- число признаков;
- размеры train/valid/test;
- средние и стандартные отклонения метрик;
- время выполнения.

---

## `src/evaluation/metrics.py`

Содержит:

- `MetricsCalculator`
- `compare_experiments(...)`
- `plot_metric_vs_aggregation(...)`
- `plot_heatmap_aggregation(...)`
- `plot_all_experiments_comparison(...)`

Этот модуль — основной источник метрик и базовых графиков сравнения.

---

## `src/visualization/experiment_plots.py`

Этот модуль содержит более прикладные графики поверх сохранённых summary-результатов.

### Основные функции

- `load_all_summaries(...)`
- `plot_agg_methods_comparison(...)`
- `plot_all_experiments_bar(...)`
- `plot_features_vs_metric(...)`
- `plot_features_vs_metric_detailed(...)`
- `generate_all_plots(...)`

Важно: базовые функции извлечения метрик и основных графиков больше не дублируются здесь, а переиспользуются из `src.evaluation`.

---

## Запуск экспериментов

## 0. Предрасчёт transformed-данных (рекомендуется)

Файл:

```text
prepare_transformed_data.py
```

Скрипт заранее сохраняет готовые наборы в:

- `Data/Aggregated/Difficult_*/freq*_pickup*_<method>/train|valid|test.csv`
- `Data/PCA/Difficult_*/pca_<n_components>/train|valid|test.csv`

Команда:

```powershell
python prepare_transformed_data.py
```

Важно: если в предрасчитанных CSV нет target-колонок `H*`, experiment-layer автоматически подмешивает их из исходных `train/valid/test`.

## 1. Один aggregation-эксперимент

Файл:

```text
run_aggregation_experiment.py
```

Перед запуском отредактируй в нём:

- `DATA_DIR`
- `TRAIN_FILE`, `VALID_FILE`, `TEST_FILE`
- `TARGET_COLUMNS`
- `EXPERIMENT_NAME`
- `FREQ_STEPS`
- `PICKUP_STEPS`
- `AGG_METHODS`
- `USE_PRECOMPUTED_AGGREGATED` (брать готовые из `Data/Aggregated`)
- `SAVE_TRANSFORMED_DATA`
- `DEVICE`
- параметры обучения

### Команда запуска

```powershell
python run_aggregation_experiment.py
```

---

## 2. Один PCA-эксперимент

Файл:

```text
run_pca_experiment.py
```

Перед запуском отредактируй:

- `DATA_DIR`
- `TRAIN_FILE`, `VALID_FILE`, `TEST_FILE`
- `TARGET_COLUMNS`
- `EXPERIMENT_NAME`
- `N_COMPONENTS_LIST`
- `USE_PRECOMPUTED_PCA` (брать готовые из `Data/PCA`)
- `SAVE_TRANSFORMED_DATA`
- `DEVICE`
- параметры обучения

### Команда запуска

```powershell
python run_pca_experiment.py
```

---

## 3. Полный пакет экспериментов

Файл:

```text
run_all_experiments.py
```

Он запускает:

- PCA;
- aggregation;
- по нескольким датасетам;
- по нескольким target-колонкам.

По умолчанию `run_all_experiments.py` сначала пытается использовать предрасчитанные данные из `Data/Aggregated` и `Data/PCA` (по папкам `Difficult_1..3`), а при отсутствии нужной конфигурации автоматически делает трансформацию на лету.

### Команда запуска

```powershell
python run_all_experiments.py
```

---

## Структура папки результатов

После запуска формируется структура такого вида:

```text
experiments/
└── experiment_name/
    ├── all_results.csv
    ├── plots/
    │   ├── metric_vs_aggregation.png
    │   ├── heatmap.png
    │   └── ...
    │
    ├── freq1_pickup1_mean/
    │   ├── aggregated_data/
    │   │   ├── train.csv
    │   │   ├── valid.csv
    │   │   └── test.csv
    │   ├── learning_curves/
    │   │   └── learning_curve_iter_1.png
    │   ├── results/
    │   │   └── metrics.xlsx
    │   └── summary.json
    │
    ├── freq2_pickup3_mean/
    │   └── ...
    │
    └── ...
```

Для PCA структура аналогична, но сохранение transformed-данных можно отключить через `SAVE_TRANSFORMED_DATA = False`.

---

## Что лежит в `metrics.xlsx`

По каждому запуску сохраняется Excel-файл с листами:

- `Raw_Results`
- `Summary`
- `Meta`

### `Summary`

Содержит агрегированные метрики по итерациям:

- `R2`
- `MSE`
- `MAE`
- `Pearson`

### `Meta`

Содержит параметры запуска:

- `n_iter`
- `batch_size`
- `input_dim`
- `output_dim`
- `learning_rate`
- `num_epochs`
- `patience`
- `hidden_dim`
- `optimizer`
- `momentum`
- `tolerance`
- `tolerance_mode`
- `random_state`
- `device`

---

## Параметры обучения

Основные параметры, которые используются в `OLP` и experiment-layer:

| Параметр | Значение по умолчанию | Описание |
|---|---:|---|
| `hidden_dim` | 32 | Размер скрытого слоя. Если убрать скрытый слой, модель станет линейной |
| `learning_rate` | 0.01 | Скорость обучения |
| `num_epochs` | 1000 | Максимальное число эпох |
| `patience` | 100 | Количество эпох без улучшения до остановки |
| `tolerance` | 1e-4 | Минимальный порог улучшения validation loss |
| `tolerance_mode` | `relative` | Режим порога: `relative` или `absolute` |
| `batch_size` | 64 | Размер батча |
| `n_iter` | 5 | Сколько независимых запусков делать для одной конфигурации |
| `optimizer` | `sgd` | Оптимизатор: `sgd` или `adam` |
| `momentum` | 0.9 | Используется только для `sgd` |
| `random_state` | 42 | Базовый seed для воспроизводимости |
| `enable_cv` | `True` | Включать ли встроенный блок cross-validation после основного обучения |

---

## Метрики

Проект считает следующие метрики:

- **R²**
- **MSE**
- **MAE**
- **Pearson correlation**

Обычно в summary сохраняются:

- `r2_mean`, `r2_std`
- `mse_mean`, `mse_std`
- `mae_mean`, `mae_std`
- `pearson_mean`, `pearson_std`

---

## Визуализация результатов

### Построение всех графиков из уже готового эксперимента

```python
from pathlib import Path
from src.visualization.experiment_plots import generate_all_plots

generate_all_plots(Path('experiments/aggregation_study'))
```

### Построение вручную

```python
from pathlib import Path
from src.visualization.experiment_plots import load_all_summaries, plot_features_vs_metric

results_df = load_all_summaries(Path('experiments/aggregation_study'))
fig = plot_features_vs_metric(results_df, metric='r2')
```

---

## Отбор признаков

Файл:

```text
src/models/feature_selection.py
```

Содержит несколько утилит для feature selection и отдельный `FeatureRankingProcessor`.

### Пример `FeatureRankingProcessor`

```python
from src.models import FeatureRankingProcessor

processor = FeatureRankingProcessor(
    train=train,
    test=test,
    valid=valid,
    base_dir='./experiments/ranking_study',
    ranking_dfs=[df_rank1, df_rank2],
    model_names=['lasso', 'ridge'],
    n_iter=3,
    num_epochs=500,
    learning_rate=0.01,
    optimizer_type='adam',
    patience=100,
    tolerance=1e-4,
)

processor.run()
```

---

## Тесты

В проекте сейчас есть 4 основных тестовых файла:

- `tests/test_core.py`
- `tests/test_pca_module.py`
- `tests/test_visualization.py`
- `tests/test_data_diagnostics.py`

### Что проверяют

#### `test_core.py`
Проверяет:

- корректность `Data`;
- валидацию схемы;
- агрегацию;
- smoke test aggregation-эксперимента.

#### `test_pca_module.py`
Проверяет:

- `PCATransformer`;
- PCA по фиксированному числу компонент;
- PCA по доле объяснённой дисперсии;
- применение PCA к train/valid/test;
- график explained variance.

#### `test_visualization.py`
Проверяет:

- загрузку summary;
- чтение метрик из Excel;
- построение основных графиков.

#### `test_data_diagnostics.py`
Это не unit-test в строгом смысле, а диагностический исследовательский скрипт для анализа:

- корреляций;
- возможной утечки данных;
- поведения простых baseline-моделей.

### Запуск тестов

Если установлен `pytest`:

```powershell
python -m pytest
```

Если `pytest` не установлен:

```powershell
pip install pytest
```

---

## Установка зависимостей

```powershell
pip install -r requirements.txt
```

Если нужно установить вручную:

```powershell
pip install pandas numpy torch scikit-learn matplotlib seaborn openpyxl pytest
```

---

## Минимальный сценарий работы

1. Положить CSV-файлы в папку `Data/`.
2. Выбрать target, например `H3_8`.
3. Для агрегации:
   - открыть `run_aggregation_experiment.py`
   - выставить нужные параметры
   - запустить скрипт
4. Для PCA:
   - открыть `run_pca_experiment.py`
   - выставить список `N_COMPONENTS_LIST`
   - запустить скрипт
5. Сравнить результаты в `experiments/.../summary.json`, `all_results.csv` и графиках.

---

## Что было изменено в архитектуре

Актуальное состояние проекта после рефакторинга:

- `weight_decay` полностью удалён из живого кода и API;
- удалён сломанный `run_smart_pca_experiment.py`;
- `OLP` приведён к более стабильному и воспроизводимому поведению;
- исправлено сохранение лучшей модели;
- унифицированы результаты aggregation и PCA через `ExperimentResult`;
- убрано дублирование `extract_metrics_from_excel`;
- убрано дублирование базовых plotting-функций между evaluation и visualization;
- `Data` теперь валидирует схему train/valid/test;
- тесты переведены в более чистый и поддерживаемый формат.

---

## Ограничения текущей реализации

Важно понимать:

- `OLP` всё ещё совмещает основное обучение и встроенную CV-оценку, но CV теперь можно отключить через `enable_cv=False`;
- основным артефактом метрик по-прежнему остаётся Excel-файл, хотя структура проекта уже стала чище;
- `test_data_diagnostics.py` — это исследовательская диагностика, а не строгий unit-test.

То есть проект уже сильно лучше структурирован, но это всё ещё ML-исследовательский код, а не полностью production framework.

---

## Быстрые команды

### Запуск aggregation

```powershell
python run_aggregation_experiment.py
```

### Запуск PCA

```powershell
python run_pca_experiment.py
```

### Запуск всех экспериментов

```powershell
python run_all_experiments.py
```

### Проверка импорта/синтаксиса

```powershell
python -m compileall src tests run_all_experiments.py run_aggregation_experiment.py run_pca_experiment.py
```

### Запуск тестов

```powershell
python -m pytest
```

---

## Рекомендация по работе

Если запускаешь длинные эксперименты:

- сначала проверь один маленький smoke-run с `n_iter=1` и `num_epochs=5..20`;
- потом увеличивай эпохи и сетку;
- сначала сохраняй результаты для одного target, потом запускай массовые эксперименты;
- сравнивай не только `r2_mean`, но и `mse_mean`, время и число признаков.

---

## Авторский контекст

Это исследовательский проект по обратной задаче МТЗ, заточенный под сравнение способов уменьшения размерности признаков и анализа устойчивости нейросетевой регрессии.
