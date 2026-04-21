"""
Запуск всех экспериментов (PCA + агрегация) для всех данных и таргетов.

Порядок:
1. Сначала H3_8 на всех данных (чтобы были результаты, даже если прервётся)
2. Потом H1_8 и H2_8

Данные:
- train_3/test_3/valid_3 (новые данные, вариант 3)
- train_3_1/test_3_1/valid_3_1 (новые данные, вариант 3_1)
- mtsgrvmgn_trn/vld/tst (старые данные)

Таргеты: H3_8, H1_8, H2_8
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.experiment import PCAExperiment, AggregationExperiment
from src.utils import load_mtz_data


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

# Параметры PCA
N_COMPONENTS_LIST = [16, 64, 256, 512, 1024, 1536]

# Параметры агрегации
FREQ_STEPS = [1, 2, 3]
PICKUP_STEPS = [1, 2, 3]
AGG_METHODS = ['mean']

# Параметры обучения (общие для всех экспериментов)
N_ITER = 1
NUM_EPOCHS = 10000
LEARNING_RATE = 0.01
OPTIMIZER = 'sgd'
PATIENCE = 250
TOLERANCE = 0.003
TOLERANCE_MODE = 'relative'
HIDDEN_DIM = 32
BATCH_SIZE = 128
SAVE_TRANSFORMED_DATA = False
DEVICE = 'auto'
ENABLE_CV = False

# Директория данных
DATA_DIR = PROJECT_ROOT / "Data"
EXPERIMENTS_BASE_DIR = EXPERIMENTS_DIR

# =============================================================================
# СПИСОК ЭКСПЕРИМЕНТОВ (порядок важен!)
# =============================================================================

EXPERIMENTS = [
    # === H3_8 ПЕРВЫМИ (приоритет) ===
    # train_3 + H3_8
    {
        'name': 'aggregation_train3_H3_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H3_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_H3_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H3_8',
        'experiment_type': 'pca'
    },
    # train_3_1 + H3_8
    {
        'name': 'aggregation_train3_1_H3_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H3_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_1_H3_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H3_8',
        'experiment_type': 'pca'
    },
    # mtsgrvmgn + H3_8
    {
        'name': 'aggregation_old_H3_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H3_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_old_H3_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H3_8',
        'experiment_type': 'pca'
    },

    # === H1_8 (второй приоритет) ===
    # train_3 + H1_8
    {
        'name': 'aggregation_train3_H1_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H1_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_H1_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H1_8',
        'experiment_type': 'pca'
    },
    # train_3_1 + H1_8
    {
        'name': 'aggregation_train3_1_H1_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H1_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_1_H1_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H1_8',
        'experiment_type': 'pca'
    },
    # mtsgrvmgn + H1_8
    {
        'name': 'aggregation_old_H1_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H1_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_old_H1_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H1_8',
        'experiment_type': 'pca'
    },

    # === H2_8 (третий приоритет) ===
    # train_3 + H2_8
    {
        'name': 'aggregation_train3_H2_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H2_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_H2_8',
        'data': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
        'target': 'H2_8',
        'experiment_type': 'pca'
    },
    # train_3_1 + H2_8
    {
        'name': 'aggregation_train3_1_H2_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H2_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_train3_1_H2_8',
        'data': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
        'target': 'H2_8',
        'experiment_type': 'pca'
    },
    # mtsgrvmgn + H2_8
    {
        'name': 'aggregation_old_H2_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H2_8',
        'experiment_type': 'aggregation'
    },
    {
        'name': 'pca_old_H2_8',
        'data': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
        'target': 'H2_8',
        'experiment_type': 'pca'
    },
]


def run_pca_experiment(train, valid, test, target, exp_name):
    """Запуск PCA эксперимента"""
    print(f"\n{'='*60}")
    print(f"PCA: {exp_name} (target={target})")
    print(f"{'='*60}")

    experiment = PCAExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=[target],
        experiment_name=exp_name,
        base_dir=EXPERIMENTS_BASE_DIR / exp_name
    )

    results = experiment.run_grid(
        n_components_list=N_COMPONENTS_LIST,
        n_iter=N_ITER,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        patience=PATIENCE,
        tolerance=TOLERANCE,
        tolerance_mode=TOLERANCE_MODE,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        save_transformed_data=SAVE_TRANSFORMED_DATA,
        device=DEVICE,
        enable_cv=ENABLE_CV
    )

    # Графики
    experiment.plot_comparison(metric='r2_mean', save=True)
    experiment.plot_comparison(metric='mse_mean', save=True)

    # Сохраняем результаты
    results.to_csv(EXPERIMENTS_BASE_DIR / exp_name / "all_results.csv", index=False)

    print(f"[OK] PCA {exp_name} завершен")
    return experiment


def run_aggregation_experiment(train, valid, test, target, exp_name):
    """Запуск агрегации"""
    print(f"\n{'='*60}")
    print(f"AGG: {exp_name} (target={target})")
    print(f"{'='*60}")

    experiment = AggregationExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=[target],
        experiment_name=exp_name,
        base_dir=EXPERIMENTS_BASE_DIR / exp_name
    )

    results = experiment.run_grid(
        freq_steps=FREQ_STEPS,
        pickup_steps=PICKUP_STEPS,
        agg_methods=AGG_METHODS,
        n_iter=N_ITER,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        patience=PATIENCE,
        tolerance=TOLERANCE,
        tolerance_mode=TOLERANCE_MODE,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        save_transformed_data=SAVE_TRANSFORMED_DATA,
        device=DEVICE,
        enable_cv=ENABLE_CV
    )

    # Графики
    experiment.plot_comparison(save=True)

    # Сохраняем результаты
    results.to_csv(experiment.base_dir / "all_results.csv", index=False)

    print(f"[OK] AGG {exp_name} завершен")
    return experiment


def main():
    print("=" * 70)
    print("ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)
    print(f"Всего экспериментов: {len(EXPERIMENTS)}")
    print(f"PCA: {sum(1 for e in EXPERIMENTS if e['experiment_type'] == 'pca')}")
    print(f"AGG: {sum(1 for e in EXPERIMENTS if e['experiment_type'] == 'aggregation')}")

    # Подсчитываем для каждого таргета
    for target in ['H3_8', 'H1_8', 'H2_8']:
        count = sum(1 for e in EXPERIMENTS if e['target'] == target)
        print(f"  {target}: {count} экспериментов")

    print("\nПорядок: сначала все H3_8, потом H1_8, потом H2_8")
    print("=" * 70)

    completed = 0
    failed = 0

    for i, exp in enumerate(EXPERIMENTS, 1):
        exp_name = exp['name']
        train_file, valid_file, test_file = exp['data']
        target = exp['target']
        exp_type = exp['experiment_type']

        print(f"\n[{i}/{len(EXPERIMENTS)}] Запуск: {exp_name}")

        # Проверяем файлы
        train_path = DATA_DIR / train_file
        valid_path = DATA_DIR / valid_file
        test_path = DATA_DIR / test_file

        if not train_path.exists():
            print(f"  [X] Файл не найден: {train_path}")
            failed += 1
            continue

        try:
            # Загружаем данные
            train, valid, test = load_mtz_data(train_path, valid_path, test_path, verbose=False)
            print(f"  Данные: train={len(train)}, valid={len(valid)}, test={len(test)}")

            # Запускаем эксперимент
            if exp_type == 'pca':
                run_pca_experiment(train, valid, test, target, exp_name)
            else:
                run_aggregation_experiment(train, valid, test, target, exp_name)

            completed += 1

        except Exception as e:
            print(f"  [X] Ошибка: {e}")
            failed += 1
            continue

    # Итоги
    print("\n" + "=" * 70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("=" * 70)
    print(f"Успешно: {completed}")
    print(f"Ошибок:  {failed}")

    # Сохраняем сводку
    summary = []
    summary.append(f"Всего экспериментов: {len(EXPERIMENTS)}")
    summary.append(f"Успешно: {completed}")
    summary.append(f"Ошибок: {failed}")
    summary.append("")

    # Группируем по таргетам
    for target in ['H3_8', 'H1_8', 'H2_8']:
        target_exps = [e for e in EXPERIMENTS if e['target'] == target]
        summary.append(f"=== {target} ===")
        for e in target_exps:
            status = "OK" if e['name'] in [exp['name'] for exp in EXPERIMENTS[:completed]] else "pending"
            summary.append(f"  {e['name']}: {e['experiment_type']}")

    with open(EXPERIMENTS_BASE_DIR / "all_experiments_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print(f"\nСводка сохранена: {EXPERIMENTS_BASE_DIR / 'all_experiments_summary.txt'}")


if __name__ == '__main__':
    main()
