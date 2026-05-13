"""
Запуск эксперимента с PLS (Partial Least Squares).

Определяет оптимальное количество компонент для заданной задачи.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.utils import load_mtz_data
from src.experiment import PLSExperiment

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_FILE = "mtsgrvmgn_trn.csv"
VALID_FILE = "mtsgrvmgn_vld.csv"
TEST_FILE = "mtsgrvmgn_tst.csv"

USE_PRECOMPUTED_PLS = True
PRECOMPUTED_PLS_ROOT = DATA_DIR / "PLS"
DATASET_TO_DIFFICULTY = {
    'mtsgrvmgn_trn.csv': 'Difficult_1',
    'train_3.csv': 'Difficult_2',
    'train_3_1.csv': 'Difficult_3',
}

TARGET_COLUMNS = ['H3_8']
EXPERIMENT_NAME = "pls_all_difficulties_H3_8"

N_COMPONENTS_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]

N_ITER = 3
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
OPTIMIZER = 'adam'
PATIENCE = 250
TOLERANCE = 0.003
TOLERANCE_MODE = 'relative'
HIDDEN_DIM = 32
BATCH_SIZE = 64
SAVE_TRANSFORMED_DATA = False
DEVICE = 'auto'
ENABLE_CV = False


# =============================================================================
# ЗАПУСК
# =============================================================================

def main():
    print("=" * 60)
    print("PLS EXPERIMENT")
    print("=" * 60)

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE
    test_path = DATA_DIR / TEST_FILE

    if not train_path.exists():
        print(f"Data not found: {train_path}")
        return

    print(f"\nLoading data...")
    train, valid, test = load_mtz_data(train_path, valid_path, test_path)
    print(f"Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")

    experiment = PLSExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=TARGET_COLUMNS,
        experiment_name=EXPERIMENT_NAME,
        base_dir=EXPERIMENTS_DIR / EXPERIMENT_NAME,
    )

    difficulty = DATASET_TO_DIFFICULTY.get(TRAIN_FILE)
    precomputed_root = None
    if USE_PRECOMPUTED_PLS and difficulty:
        precomputed_root = PRECOMPUTED_PLS_ROOT / difficulty

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
        enable_cv=ENABLE_CV,
        precomputed_pls_root=str(precomputed_root) if precomputed_root else None,
    )

    experiment.plot_comparison(metric='r2_mean', save=True)
    experiment.plot_comparison(metric='mse_mean', save=True)

    best = experiment.get_best_result()

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"n_components: {best['n_components']}")
    print(f"variance_explained: {best.get('variance_explained', 'N/A')}")
    print(f"R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}")
    print(f"Compression: {best['compression_ratio']:.1f}x")

    results.to_csv(EXPERIMENTS_DIR / EXPERIMENT_NAME / "all_results.csv", index=False)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()