"""
Запуск эксперимента с PCA (метод главных компонент)

Определяет оптимальное количество компонент для заданной задачи.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.utils import load_mtz_data
from src.experiment import PCAExperiment

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

# Пути к данным
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_FILE = "mtsgrvmgn_trn.csv"
VALID_FILE = "mtsgrvmgn_vld.csv"
TEST_FILE = "mtsgrvmgn_tst.csv"

# Целевые переменные
TARGET_COLUMNS = ['H3_8']  # Можно добавить H2_8, H3_8

# Название эксперимента
EXPERIMENT_NAME = "pca_study"

# Количество PCA компонент для тестирования
N_COMPONENTS_LIST = [100, 200, 500, 1000, 1500, 2000]

# Или использовать пороги дисперсии (раскомментируйте)
# VARIANCE_THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

# Параметры обучения
N_ITER = 3                    # Количество итераций
NUM_EPOCHS = 1000              # Макс. эпох
LEARNING_RATE = 0.01          # Скорость обучения
OPTIMIZER = 'adam'             # Оптимизатор ('sgd' или 'adam')
PATIENCE = 250                # Early stopping patience
TOLERANCE = 0.003              # Early stopping tolerance
TOLERANCE_MODE = 'relative'    # 'relative' или 'absolute'
HIDDEN_DIM = 32               # Размер скрытого слоя
BATCH_SIZE = 64  
SAVE_TRANSFORMED_DATA = False  # Сохранять PCA train/valid/test и pca_info
DEVICE = 'auto'               # 'auto' -> GPU(CUDA), если доступен
ENABLE_CV = False             # Отключить встроенную cross-validation для ускорения


# =============================================================================
# ЗАПУСК
# =============================================================================

def main():
    print("=" * 60)
    print("PCA EXPERIMENT")
    print("=" * 60)
    
    # Загрузка данных
    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE
    test_path = DATA_DIR / TEST_FILE
    
    if not train_path.exists():
        print(f"Data not found: {train_path}")
        return
    
    print(f"\nLoading data...")
    train, valid, test = load_mtz_data(train_path, valid_path, test_path)
    print(f"Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
    
    # Создаём эксперимент
    experiment = PCAExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=TARGET_COLUMNS,
        experiment_name=EXPERIMENT_NAME,
        base_dir=EXPERIMENTS_DIR / EXPERIMENT_NAME
    )
    
    # Запускаем сетку экспериментов
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
    
    # Лучшая конфигурация
    best = experiment.get_best_result()
    
    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"n_components: {best['n_components']}")
    print(f"variance_explained: {best['variance_explained']:.2%}")
    print(f"R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}")
    print(f"Compression: {best['compression_ratio']:.1f}x")
    
    # Сохраняем результаты
    results.to_csv(EXPERIMENTS_DIR / EXPERIMENT_NAME / "all_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
