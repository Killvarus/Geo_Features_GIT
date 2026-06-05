"""
Запуск эксперимента с PLS (Partial Least Squares) для всех уровней сложности.

PLS — САМОСТОЯТЕЛЬНЫЙ регрессионный метод. Нейросеть не используется.

Параметры:
  Данные:       Data/ (3 сложности)
  Таргет:       H3_8
  Компоненты:   2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048

Структура результатов:
experiments/pls_all_difficulties_H3_8/
  all_results.csv                     — сводная таблица всех запусков
  Difficult_1/
    logs/experiment.log               — лог эксперимента
    plots/
      r2_mean_vs_components.png       — R² vs компоненты
      mse_mean_vs_components.png      — MSE vs компоненты
      training_time_vs_components.png — время обучения
      relative_training_time_vs_components.png — относительное время
    pls_2/
      summary.json                    — ExperimentResult
      predictions/
        test_predictions.csv          — предсказания на тесте
        valid_predictions.csv         — предсказания на валидации
    pls_4/ ...
  Difficult_2/ ...
  Difficult_3/ ...

Запуск:
  python run_pls_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.experiment import PLSExperiment
from src.utils import load_mtz_data

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

DATA_DIR = PROJECT_ROOT / "Data"

DIFFICULTIES = ["Difficult_1", "Difficult_2", "Difficult_3"]
DIFFICULTY_TO_FILES = {
    "Difficult_1": ("mtsgrvmgn_trn.csv", "mtsgrvmgn_vld.csv", "mtsgrvmgn_tst.csv"),
    "Difficult_2": ("train_3.csv", "valid_3.csv", "test_3.csv"),
    "Difficult_3": ("train_3_1.csv", "valid_3_1.csv", "test_3_1.csv"),
}

TARGET_COLUMNS = ["H3_8"]
EXPERIMENT_NAME = "pls_all_difficulties_H3_8"

N_COMPONENTS_LIST = [16, 64, 256, 1024, 1536, 2048]

# PLS не использует нейросеть — параметры ниже не нужны

# =============================================================================
# ЗАПУСК
# =============================================================================

def main():
    print("=" * 60)
    print("PLS EXPERIMENT — ALL DIFFICULTIES")
    print("=" * 60)
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"Targets: {TARGET_COLUMNS}")
    print(f"Components: {N_COMPONENTS_LIST}")

    experiment_base = EXPERIMENTS_DIR / EXPERIMENT_NAME
    experiment_base.mkdir(parents=True, exist_ok=True)

    all_results = []

    for difficulty in DIFFICULTIES:
        train_file, valid_file, test_file = DIFFICULTY_TO_FILES[difficulty]
        train_path = DATA_DIR / train_file
        valid_path = DATA_DIR / valid_file
        test_path = DATA_DIR / test_file

        print("\n" + "#" * 60)
        print(f"DIFFICULTY: {difficulty}")
        print("#" * 60)

        if not train_path.exists():
            print(f"  [SKIP] Data not found: {train_path}")
            continue

        print("  Loading data...")
        train, valid, test = load_mtz_data(train_path, valid_path, test_path)
        print(f"  Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")

        experiment = PLSExperiment(
            train=train,
            valid=valid,
            test=test,
            target_columns=TARGET_COLUMNS,
            experiment_name=EXPERIMENT_NAME,
            base_dir=experiment_base / difficulty,
        )

        results = experiment.run_grid(
            n_components_list=N_COMPONENTS_LIST,
        )

        # Добавляем difficulty в результаты
        for r in results.to_dict("records"):
            r["difficulty"] = difficulty
            all_results.append(r)

        # Графики для этой сложности
        experiment.plot_comparison(metric="r2_mean", save=True)
        experiment.plot_comparison(metric="mse_mean", save=True)

        best = experiment.get_best_result()
        print(f"  Best: n={best.n_components}, R²={best.r2_mean:.4f} ± {best.r2_std:.4f}")

    # Сохраняем общий CSV
    import pandas as pd
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(experiment_base / "all_results.csv", index=False)
    print(f"\nSaved: {experiment_base / 'all_results.csv'}")

    # Итог
    if not df_all.empty:
        best_idx = df_all["r2_mean"].idxmax()
        best = df_all.loc[best_idx]
        print("\n" + "=" * 60)
        print("OVERALL BEST")
        print("=" * 60)
        print(f"Difficulty: {best['difficulty']}")
        print(f"n_components: {int(best['n_components'])}")
        print(f"R² = {best['r2_mean']:.4f} ± {best['r2_std']:.4f}")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()