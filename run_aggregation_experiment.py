"""
Запуск экспериментов по агрегации геофизических данных

Просто укажите параметры и запустите!
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT
from src.utils import load_mtz_data
from src.experiment import AggregationExperiment


# =============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА - ИЗМЕНЯЙТЕ ЗДЕСЬ
# =============================================================================

# Пути к данным
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_FILE = "mtsgrvmgn_trn.csv"
VALID_FILE = "mtsgrvmgn_vld.csv"
TEST_FILE = "mtsgrvmgn_tst.csv"

# Использовать предрасчитанные Aggregated-данные из Data/Aggregated/Difficult_*/...
USE_PRECOMPUTED_AGGREGATED = True
PRECOMPUTED_AGGREGATED_ROOT = DATA_DIR / "Aggregated"
DATASET_TO_DIFFICULTY = {
    'mtsgrvmgn_trn.csv': 'Difficult_1',
    'train_3.csv': 'Difficult_2',
    'train_3_1.csv': 'Difficult_3',
}


# Целевые переменные
TARGET_COLUMNS = ['H3_8']  # Или ['H3_8'] для одной цели

# Название эксперимента
EXPERIMENT_NAME = "aggregation_study"

# Параметры агрегации
FREQ_STEPS = [1, 2, 3]        # Шаги по частотам
PICKUP_STEPS = [1, 2, 3]      # Шаги по пикетам
AGG_METHODS = ['mean']  # Методы агрегации

# Параметры обучения
N_ITER = 3                    # Количество итераций
NUM_EPOCHS = 10000              # Макс. эпох
LEARNING_RATE = 0.01          # Скорость обучения
OPTIMIZER = 'adam'             # Оптимизатор ('sgd' или 'adam')
PATIENCE = 100                # Early stopping patience

TOLERANCE = 0.003              # Early stopping tolerance
TOLERANCE_MODE = 'relative'    # 'relative' или 'absolute'
HIDDEN_DIM = 32               # Размер скрытого слоя
BATCH_SIZE = 64               # Размер батча
SAVE_TRANSFORMED_DATA = False # Сохранять агрегированные train/valid/test
DEVICE = 'auto'               # 'auto' -> GPU(CUDA), если доступен
ENABLE_CV = False             # Отключить встроенную cross-validation для ускорения




# =============================================================================
# ЗАПУСК
# =============================================================================

def main():
    # Пути
    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE
    test_path = DATA_DIR / TEST_FILE
    
    # Проверка файлов
    if not train_path.exists():
        print(f"❌ Файл не найден: {train_path}")
        print(f"   Проверьте пути в конфигурации")
        return
    
    # Загрузка
    print("Загрузка данных...")
    train, valid, test = load_mtz_data(train_path, valid_path, test_path)
    
    # Эксперимент
    experiment = AggregationExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=TARGET_COLUMNS,
        experiment_name=EXPERIMENT_NAME
    )
    
    difficulty = DATASET_TO_DIFFICULTY.get(TRAIN_FILE)
    precomputed_root = None
    if USE_PRECOMPUTED_AGGREGATED and difficulty:
        precomputed_root = PRECOMPUTED_AGGREGATED_ROOT / difficulty

    # Запуск
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
        enable_cv=ENABLE_CV,
        precomputed_aggregated_root=str(precomputed_root) if precomputed_root else None,
    )
    
    
    
    
    
    # Результаты
    experiment.plot_comparison()
    results.to_csv(experiment.base_dir / "all_results.csv", index=False)
    
    print(experiment.summary())
    

if __name__ == "__main__":
    main()
