"""
Запуск эксперимента с нелинейным PCA (Kernel PCA, Polynomial).

Сценарий:
1) пытается загрузить готовые poly-KPCA данные из Data/KernelPCA_Poly/... (без H*),
2) если данных нет и PRECOMPUTED_ONLY=False — считает poly-KPCA на лету,
3) обучает OLP на KPC-признаках,
4) сохраняет метрики и графики в experiments/...
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.evaluation.metrics import plot_feature_count_vs_time, plot_relative_training_time
from src.models.neural_network import to_excel_optimized_OLP
from src.preprocessing.pca import plot_pca_comparison
from src.utils import Data, load_mtz_data


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

# Kernel PCA (Polynomial)
N_COMPONENTS_LIST = [512, 1024, 1536, 2048]
KPCA_DEGREES = [2, 3]
KPCA_GAMMAS = [0.0001, 0.001, 0.01]
KPCA_COEF0_LIST = [0.1, 1.0]
MAX_TRAIN_SAMPLES_FOR_KPCA_FIT = 22000
RANDOM_STATE = 42

# Обучение OLP
N_ITER = 3
NUM_EPOCHS = 100000
LEARNING_RATE = 0.01
OPTIMIZER = "adam"
MOMENTUM = 0.9
PATIENCE = 100
TOLERANCE = 0.003
TOLERANCE_MODE = "relative"
HIDDEN_DIM = 32
BATCH_SIZE = 128
DEVICE = "auto"
ENABLE_CV = False

# Артефакты
SAVE_TRANSFORMED_DATA = True
KERNEL_PCA_DATA_ROOT = DATA_DIR / "KernelPCA_Poly"
USE_PRECOMPUTED_KERNEL_PCA = True
PRECOMPUTED_ONLY = True
EXPERIMENT_NAME = f"kernel_pca_poly_all_difficulties_{'_'.join(TARGET_COLUMNS)}"
EXPERIMENT_BASE_DIR = EXPERIMENTS_DIR / EXPERIMENT_NAME


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================


def _extract_metrics_from_all_results(all_results: List[Dict]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metric_names = ["R2", "MSE", "MAE", "Pearson"]

    for metric_name in metric_names:
        iter_vals = []
        for run_result in all_results:
            df_test = run_result.get("test")
            if df_test is None or metric_name not in df_test.columns:
                continue
            vals = [float(v) for v in df_test[metric_name].dropna().values]
            if vals:
                iter_vals.append(float(np.mean(vals)))

        if iter_vals:
            key = metric_name.lower()
            metrics[f"{key}_mean"] = float(np.mean(iter_vals))
            metrics[f"{key}_std"] = float(np.std(iter_vals))

    return metrics


def _fit_kernel_pca_poly_on_train(
    X_train: pd.DataFrame,
    n_components: int,
    degree: int,
    gamma: float,
    coef0: float,
    max_train_samples_for_fit: int,
    random_state: int,
) -> Tuple[SimpleImputer, StandardScaler, KernelPCA]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    rng = np.random.RandomState(random_state)
    if max_train_samples_for_fit and len(X_train_scaled) > max_train_samples_for_fit:
        fit_indices = rng.choice(len(X_train_scaled), size=max_train_samples_for_fit, replace=False)
        X_fit = X_train_scaled[fit_indices]
    else:
        X_fit = X_train_scaled

    kpca = KernelPCA(
        n_components=n_components,
        kernel="poly",
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        eigen_solver="auto",
        random_state=random_state,
    )
    kpca.fit(X_fit)

    return imputer, scaler, kpca


def _transform_with_kpca(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    kpca: KernelPCA,
    n_components: int,
) -> pd.DataFrame:
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_kpca = kpca.transform(X_scaled)

    cols = [f"KPC{i+1}" for i in range(n_components)]
    return pd.DataFrame(X_kpca, index=X.index, columns=cols)


def _plot_kpca_space_2d(train_kpca: pd.DataFrame, y_train: pd.Series, save_path: Path):
    if train_kpca.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        train_kpca["KPC1"],
        train_kpca["KPC2"],
        c=y_train.values,
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")
    ax.set_title("Kernel PCA (Poly): KPC1 vs KPC2 (train)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(y_train.name)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_kpca_space_3d(train_kpca: pd.DataFrame, y_train: pd.Series, save_path: Path):
    if train_kpca.shape[1] < 3:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        train_kpca["KPC1"],
        train_kpca["KPC2"],
        train_kpca["KPC3"],
        c=y_train.values,
        cmap="viridis",
        s=8,
        alpha=0.6,
    )
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")
    ax.set_zlabel("KPC3")
    ax.set_title("Kernel PCA (Poly): KPC1/KPC2/KPC3 (train)")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(y_train.name)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# ОСНОВНОЙ СЦЕНАРИЙ
# =============================================================================


def main():
    unknown_difficulties = [d for d in DIFFICULTIES if d not in DIFFICULTY_TO_FILES]
    if unknown_difficulties:
        raise ValueError(f"Unknown DIFFICULTIES: {unknown_difficulties}")

    print("=" * 80)
    print("KERNEL PCA (POLY) EXPERIMENT")
    print("=" * 80)
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"Targets: {TARGET_COLUMNS}")
    print(f"n_components_list: {N_COMPONENTS_LIST}")
    print(f"degrees: {KPCA_DEGREES}")
    print(f"gammas: {KPCA_GAMMAS}")
    print(f"coef0_list: {KPCA_COEF0_LIST}")
    print(f"max_train_samples_for_fit: {MAX_TRAIN_SAMPLES_FOR_KPCA_FIT}")
    print(f"use_precomputed_kernel_pca: {USE_PRECOMPUTED_KERNEL_PCA}")
    print(f"precomputed_only: {PRECOMPUTED_ONLY}")

    EXPERIMENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = EXPERIMENT_BASE_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_rows: List[Dict] = []

    for difficulty in DIFFICULTIES:
        train_file, valid_file, test_file = DIFFICULTY_TO_FILES[difficulty]
        train_path = DATA_DIR / train_file
        valid_path = DATA_DIR / valid_file
        test_path = DATA_DIR / test_file

        print("\n" + "#" * 80)
        print(f"DIFFICULTY: {difficulty}")
        print("#" * 80)

        train_df, valid_df, test_df = load_mtz_data(train_path, valid_path, test_path, verbose=True)

        feature_cols = [c for c in train_df.columns if not c.startswith("H")]
        print(f"Original feature count: {len(feature_cols)}")

        transformed_base = KERNEL_PCA_DATA_ROOT / difficulty
        transformed_base.mkdir(parents=True, exist_ok=True)

        X_train = train_df[feature_cols]
        X_valid = valid_df[feature_cols]
        X_test = test_df[feature_cols]

        y_train = train_df[TARGET_COLUMNS]
        y_valid = valid_df[TARGET_COLUMNS]
        y_test = test_df[TARGET_COLUMNS]

        for degree in KPCA_DEGREES:
            for gamma in KPCA_GAMMAS:
                for coef0 in KPCA_COEF0_LIST:
                    for n_components in N_COMPONENTS_LIST:
                        config_name = f"kpca_poly_d{degree}_g{gamma}_c{coef0}_n{n_components}"
                        print("\n" + "-" * 80)
                        print(f"[RUN] {difficulty} | {config_name}")

                        config_dir = EXPERIMENT_BASE_DIR / difficulty / config_name
                        config_dir.mkdir(parents=True, exist_ok=True)

                        start_time = time.time()

                        transformed_cfg_dir = transformed_base / config_name
                        precomputed_train_path = transformed_cfg_dir / "train.csv"
                        precomputed_valid_path = transformed_cfg_dir / "valid.csv"
                        precomputed_test_path = transformed_cfg_dir / "test.csv"

                        use_precomputed_current = (
                            USE_PRECOMPUTED_KERNEL_PCA
                            and precomputed_train_path.exists()
                            and precomputed_valid_path.exists()
                            and precomputed_test_path.exists()
                        )

                        if use_precomputed_current:
                            print(f"[LOAD] precomputed KPCA-POLY: {transformed_cfg_dir}")
                            train_kpca = pd.read_csv(precomputed_train_path)
                            valid_kpca = pd.read_csv(precomputed_valid_path)
                            test_kpca = pd.read_csv(precomputed_test_path)
                        else:
                            if PRECOMPUTED_ONLY:
                                print(f"[SKIP] missing precomputed KPCA-POLY: {difficulty}/{config_name}")
                                continue

                            print(f"[FIT ] KPCA-POLY from raw data: {difficulty}/{config_name}")
                            imputer, scaler, kpca = _fit_kernel_pca_poly_on_train(
                                X_train=X_train,
                                n_components=n_components,
                                degree=degree,
                                gamma=gamma,
                                coef0=coef0,
                                max_train_samples_for_fit=MAX_TRAIN_SAMPLES_FOR_KPCA_FIT,
                                random_state=RANDOM_STATE,
                            )

                            train_kpca = _transform_with_kpca(X_train, imputer, scaler, kpca, n_components)
                            valid_kpca = _transform_with_kpca(X_valid, imputer, scaler, kpca, n_components)
                            test_kpca = _transform_with_kpca(X_test, imputer, scaler, kpca, n_components)

                            if SAVE_TRANSFORMED_DATA:
                                transformed_cfg_dir.mkdir(parents=True, exist_ok=True)
                                train_kpca.to_csv(precomputed_train_path, index=False)
                                valid_kpca.to_csv(precomputed_valid_path, index=False)
                                test_kpca.to_csv(precomputed_test_path, index=False)

                                with open(transformed_cfg_dir / "kpca_info.json", "w", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "method": "KernelPCA",
                                            "kernel": "poly",
                                            "degree": degree,
                                            "gamma": gamma,
                                            "coef0": coef0,
                                            "n_components": n_components,
                                            "max_train_samples_for_fit": MAX_TRAIN_SAMPLES_FOR_KPCA_FIT,
                                            "random_state": RANDOM_STATE,
                                            "difficulty": difficulty,
                                        },
                                        f,
                                        ensure_ascii=False,
                                        indent=2,
                                    )

                        _plot_kpca_space_2d(
                            train_kpca,
                            y_train[TARGET_COLUMNS[0]],
                            plots_dir / f"kpca_poly_space_2d_{difficulty}_{config_name}.png",
                        )
                        _plot_kpca_space_3d(
                            train_kpca,
                            y_train[TARGET_COLUMNS[0]],
                            plots_dir / f"kpca_poly_space_3d_{difficulty}_{config_name}.png",
                        )

                        train_model = train_kpca.copy()
                        valid_model = valid_kpca.copy()
                        test_model = test_kpca.copy()

                        for target in TARGET_COLUMNS:
                            train_model[target] = y_train[target].values
                            valid_model[target] = y_valid[target].values
                            test_model[target] = y_test[target].values

                        data = Data(train_model, test_model, valid_model, TARGET_COLUMNS)

                        results_dir = config_dir / "results"
                        curves_dir = config_dir / "learning_curves"
                        models_dir = config_dir / "models"
                        logs_dir = config_dir / "logs"

                        results_dir.mkdir(parents=True, exist_ok=True)
                        curves_dir.mkdir(parents=True, exist_ok=True)
                        models_dir.mkdir(parents=True, exist_ok=True)
                        logs_dir.mkdir(parents=True, exist_ok=True)

                        excel_path = results_dir / "metrics.xlsx"

                        all_results = to_excel_optimized_OLP(
                            file_name=str(excel_path),
                            n_iter=N_ITER,
                            X_train=data.X_train,
                            y_train=data.y_train,
                            X_valid=data.X_valid,
                            y_valid=data.y_valid,
                            X_test=data.X_test,
                            y_test=data.y_test,
                            batch_size=BATCH_SIZE,
                            input_dim=data.n_features,
                            output_dim=data.n_targets,
                            learning_rate=LEARNING_RATE,
                            num_epochs=NUM_EPOCHS,
                            patience=PATIENCE,
                            hidden_dim=HIDDEN_DIM,
                            save_plots_dir=str(curves_dir),
                            save_models_dir=str(models_dir),
                            optimizer_type=OPTIMIZER,
                            momentum=MOMENTUM,
                            tolerance=TOLERANCE,
                            tolerance_mode=TOLERANCE_MODE,
                            random_state=RANDOM_STATE,
                            device=DEVICE,
                            log_file=str(logs_dir / "training.log"),
                            enable_cv=ENABLE_CV,
                        )

                        total_time = time.time() - start_time
                        metrics = _extract_metrics_from_all_results(all_results)

                        row = {
                            "experiment_type": "kernel_pca_poly",
                            "experiment_id": config_name,
                            "difficulty": difficulty,
                            "target": ",".join(TARGET_COLUMNS),
                            "kernel": "poly",
                            "degree": degree,
                            "gamma": gamma,
                            "coef0": coef0,
                            "n_components": n_components,
                            "n_features": n_components,
                            "original_n_features": len(feature_cols),
                            "compression_ratio": len(feature_cols) / n_components,
                            "n_samples_train": len(train_model),
                            "n_samples_valid": len(valid_model),
                            "n_samples_test": len(test_model),
                            "n_iter": N_ITER,
                            "num_epochs": NUM_EPOCHS,
                            "learning_rate": LEARNING_RATE,
                            "optimizer": OPTIMIZER,
                            "momentum": MOMENTUM,
                            "patience": PATIENCE,
                            "tolerance": TOLERANCE,
                            "tolerance_mode": TOLERANCE_MODE,
                            "hidden_dim": HIDDEN_DIM,
                            "batch_size": BATCH_SIZE,
                            "device": DEVICE,
                            "total_time_seconds": total_time,
                            "r2_mean": metrics.get("r2_mean", 0.0),
                            "r2_std": metrics.get("r2_std", 0.0),
                            "mse_mean": metrics.get("mse_mean", 0.0),
                            "mse_std": metrics.get("mse_std", 0.0),
                            "mae_mean": metrics.get("mae_mean", 0.0),
                            "mae_std": metrics.get("mae_std", 0.0),
                            "pearson_mean": metrics.get("pearson_mean", 0.0),
                            "pearson_std": metrics.get("pearson_std", 0.0),
                        }
                        results_rows.append(row)

                        with open(config_dir / "summary.json", "w", encoding="utf-8") as f:
                            json.dump(row, f, ensure_ascii=False, indent=2)

                        print(
                            f"[OK] {difficulty}/{config_name}: "
                            f"R2={row['r2_mean']:.4f}±{row['r2_std']:.4f}, "
                            f"MSE={row['mse_mean']:.4f}±{row['mse_std']:.4f}, "
                            f"time={total_time:.1f}s"
                        )

    if not results_rows:
        print("\n[WARN] Нет ни одной обученной конфигурации. Проверьте Data/KernelPCA_Poly")
        return

    results_df = pd.DataFrame(results_rows).sort_values(["difficulty", "degree", "gamma", "coef0", "n_components"])
    results_df.to_csv(EXPERIMENT_BASE_DIR / "all_results.csv", index=False)

    for difficulty in results_df["difficulty"].unique():
        df_diff = results_df[results_df["difficulty"] == difficulty].copy()
        for degree in KPCA_DEGREES:
            for gamma in KPCA_GAMMAS:
                for coef0 in KPCA_COEF0_LIST:
                    df_plot = df_diff[
                        (df_diff["degree"] == degree)
                        & (df_diff["gamma"] == gamma)
                        & (df_diff["coef0"] == coef0)
                    ].copy().sort_values("n_components")
                    if df_plot.empty:
                        continue

                    plot_pca_comparison(
                        df_plot.to_dict("records"),
                        metric="r2_mean",
                        save_path=plots_dir / f"r2_vs_components_{difficulty}_d{degree}_g{gamma}_c{coef0}.png",
                    )
                    plot_pca_comparison(
                        df_plot.to_dict("records"),
                        metric="mse_mean",
                        save_path=plots_dir / f"mse_vs_components_{difficulty}_d{degree}_g{gamma}_c{coef0}.png",
                    )

                    plot_feature_count_vs_time(
                        df_plot,
                        feature_col="n_components",
                        time_col="total_time_seconds",
                        save_path=plots_dir / f"training_time_vs_components_{difficulty}_d{degree}_g{gamma}_c{coef0}.png",
                        xlabel="Количество компонент",
                        title=f"Время обучения vs компоненты | {difficulty} | d={degree}, g={gamma}, c0={coef0}",
                    )

                    plot_relative_training_time(
                        df_plot,
                        feature_col="n_components",
                        time_col="total_time_seconds",
                        reference_feature_count=max(N_COMPONENTS_LIST),
                        save_path=plots_dir / f"relative_training_time_vs_components_{difficulty}_d{degree}_g{gamma}_c{coef0}.png",
                        xlabel="Количество компонент",
                        title=f"Относительное время vs компоненты | {difficulty} | d={degree}, g={gamma}, c0={coef0}",
                    )

    best_idx = results_df["r2_mean"].idxmax()
    best = results_df.loc[best_idx]

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Experiment dir: {EXPERIMENT_BASE_DIR}")
    print(f"KernelPCA-Poly data dir: {KERNEL_PCA_DATA_ROOT}")
    print(
        f"Best config: difficulty={best['difficulty']}, degree={int(best['degree'])}, gamma={best['gamma']}, coef0={best['coef0']}, "
        f"n_components={int(best['n_components'])}, R2={best['r2_mean']:.4f}±{best['r2_std']:.4f}, "
        f"compression={best['compression_ratio']:.1f}x"
    )


if __name__ == "__main__":
    main()
