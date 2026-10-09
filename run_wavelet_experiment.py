"""
Полный вейвлет-прогон: все датасеты и таргеты, 3 seed.

Порядок: все H3_8 (сначала меньший сплит), затем H1_8, затем H2_8.
Конфиги пропускаются, если уже есть summary.json.

Обучение: OLP hidden=32, Adam 0.01, batch 128, max 1000 эпох,
patience=300, relative tolerance=0.003, CV выключен.

  python run_wavelet_experiment.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import EXPERIMENTS_DIR, PROJECT_ROOT
from src.experiment.wavelet_experiment import WaveletExperiment
from src.preprocessing.wavelet import wavelet_config_name
from src.utils import load_mtz_data
from src.utils.logging_utils import close_logger_file_handlers, configure_stdio, install_safe_print


DATA_DIR = PROJECT_ROOT / "Data"
RUN_TAG = "2026-09-15_wavelet"
EXPERIMENTS_BASE_DIR = EXPERIMENTS_DIR / RUN_TAG

DIFFICULTIES = ["Difficult_3", "Difficult_2", "Difficult_1"]
DIFFICULTY_TO_FILES = {
    "Difficult_1": ("mtsgrvmgn_trn.csv", "mtsgrvmgn_vld.csv", "mtsgrvmgn_tst.csv"),
    "Difficult_2": ("train_3.csv", "valid_3.csv", "test_3.csv"),
    "Difficult_3": ("train_3_1.csv", "valid_3_1.csv", "test_3_1.csv"),
}
DIFFICULTY_TO_SHORT = {
    "Difficult_1": "old",
    "Difficult_2": "train3",
    "Difficult_3": "train3_1",
}

TARGET_COLUMNS = ["H3_8", "H1_8", "H2_8"]

N_ITER = 3
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
OPTIMIZER = "adam"
PATIENCE = 300
TOLERANCE = 0.003
TOLERANCE_MODE = "relative"
HIDDEN_DIM = 32
BATCH_SIZE = 128
DEVICE = "auto"
ENABLE_CV = False
SAVE_TRANSFORMED_DATA = False

WAVELET_CONFIGS = [
    {"wavelet": "haar", "level": 1, "keep": "approx"},
    {"wavelet": "haar", "level": 2, "keep": "approx"},
    {"wavelet": "db4", "level": 1, "keep": "approx"},
    {"wavelet": "db4", "level": 2, "keep": "approx"},
    {"wavelet": "haar", "level": 2, "keep": "energy", "energy_ratio": 0.90},
    {"wavelet": "db4", "level": 2, "keep": "energy", "energy_ratio": 0.90},
]

TRAIN_KWARGS = dict(
    n_iter=N_ITER,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    optimizer=OPTIMIZER,
    patience=PATIENCE,
    tolerance=TOLERANCE,
    tolerance_mode=TOLERANCE_MODE,
    hidden_dim=HIDDEN_DIM,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    enable_cv=ENABLE_CV,
    save_transformed_data=SAVE_TRANSFORMED_DATA,
)


def _jobs():
    jobs = []
    for target in TARGET_COLUMNS:
        for difficulty in DIFFICULTIES:
            short = DIFFICULTY_TO_SHORT[difficulty]
            jobs.append({
                "name": f"wavelet_{short}_{target}",
                "difficulty": difficulty,
                "target": target,
                "data": DIFFICULTY_TO_FILES[difficulty],
            })
    return jobs


EXPERIMENTS = _jobs()


def _config_dir(exp_out: Path, cfg: dict) -> Path:
    return exp_out / wavelet_config_name(
        wavelet=cfg["wavelet"],
        level=cfg["level"],
        keep=cfg["keep"],
        layout="grid2d",
        energy_ratio=cfg.get("energy_ratio"),
        n_features=cfg.get("n_features"),
    )


def _job_complete(exp_out: Path) -> bool:
    return all((_config_dir(exp_out, cfg) / "summary.json").exists() for cfg in WAVELET_CONFIGS)


def run_one(train, valid, test, target, exp_name):
    print(f"\n{'=' * 60}")
    print(f"WAVELET: {exp_name} (target={target})")
    print(f"{'=' * 60}")

    experiment = WaveletExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=[target],
        experiment_name=exp_name,
        base_dir=EXPERIMENTS_BASE_DIR / exp_name,
    )

    for i, cfg in enumerate(WAVELET_CONFIGS, 1):
        config_name = wavelet_config_name(
            wavelet=cfg["wavelet"],
            level=cfg["level"],
            keep=cfg["keep"],
            layout="grid2d",
            energy_ratio=cfg.get("energy_ratio"),
            n_features=cfg.get("n_features"),
        )
        print(f"  [{i}/{len(WAVELET_CONFIGS)}] {config_name}")
        try:
            result = experiment.run_single(**cfg, **TRAIN_KWARGS)
            print(
                f"    n={result.n_features} energy={result.variance_explained:.4f} "
                f"R2={result.r2_mean:.4f} +/- {result.r2_std:.4f} "
                f"time={result.total_time_seconds:.1f}s"
            )
        except Exception as exc:
            print(f"    [X] {exc}")

    results = experiment.get_results_df()
    if results.empty:
        print(f"  [X] No wavelet results for {exp_name}")
        close_logger_file_handlers(experiment.logger)
        return None

    experiment.plot_comparison(metric="r2_mean", save=True)
    experiment.plot_comparison(metric="mse_mean", save=True)
    plt.close("all")
    results.to_csv(EXPERIMENTS_BASE_DIR / exp_name / "all_results.csv", index=False)

    best = experiment.get_best_result()
    if best:
        print(
            f"  Best: {best['experiment_id']} n={best['n_features']} "
            f"R2={best['r2_mean']:.4f} +/- {best['r2_std']:.4f}"
        )
    print(f"[OK] {exp_name} done")
    close_logger_file_handlers(experiment.logger)
    return experiment


def main():
    configure_stdio()
    install_safe_print()
    EXPERIMENTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_tag": RUN_TAG,
        "n_iter": N_ITER,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "tolerance": TOLERANCE,
        "tolerance_mode": TOLERANCE_MODE,
        "optimizer": OPTIMIZER,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "enable_cv": ENABLE_CV,
        "n_jobs": len(EXPERIMENTS),
        "n_configs": len(WAVELET_CONFIGS),
        "configs": WAVELET_CONFIGS,
        "note": "Full protocol. patience=300, rel.tol=0.003, max 1000 epochs, 3 seeds.",
    }
    (EXPERIMENTS_BASE_DIR / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("=" * 70)
    print("WAVELET FULL EXPERIMENT")
    print("=" * 70)
    print(f"Results: {EXPERIMENTS_BASE_DIR}")
    print(f"Jobs: {len(EXPERIMENTS)} | configs/job: {len(WAVELET_CONFIGS)}")
    print(
        f"n_iter={N_ITER}, num_epochs={NUM_EPOCHS}, patience={PATIENCE}, "
        f"tolerance={TOLERANCE} ({TOLERANCE_MODE})"
    )
    print("Order: all H3_8 (train3_1, train3, old), then H1_8, then H2_8")
    print("=" * 70)

    completed = 0
    failed = 0
    all_rows = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        exp_name = exp["name"]
        train_file, valid_file, test_file = exp["data"]
        target = exp["target"]
        difficulty = exp["difficulty"]
        exp_out = EXPERIMENTS_BASE_DIR / exp_name

        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        if _job_complete(exp_out):
            print(f"  [SKIP] all {len(WAVELET_CONFIGS)} configs already have summary.json")
            completed += 1
            csv_path = exp_out / "all_results.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df["difficulty"] = difficulty
                    df["job"] = exp_name
                    all_rows.extend(df.to_dict("records"))
                except (OSError, ValueError) as e:
                    print(f"  [WARN] could not read existing csv: {e}")
            continue

        train_path = DATA_DIR / train_file
        valid_path = DATA_DIR / valid_file
        test_path = DATA_DIR / test_file
        if not train_path.exists():
            print(f"  [X] Missing file: {train_path}")
            failed += 1
            continue

        try:
            train, valid, test = load_mtz_data(train_path, valid_path, test_path, verbose=False)
            print(f"  Data: train={len(train)}, valid={len(valid)}, test={len(test)}")
            experiment = run_one(train, valid, test, target, exp_name)
            if experiment is None:
                failed += 1
                continue
            results = experiment.get_results_df()
            for row in results.to_dict("records"):
                row["difficulty"] = difficulty
                row["job"] = exp_name
                all_rows.append(row)
            completed += 1
        except Exception as e:
            print(f"  [X] Error: {e}")
            failed += 1

    summary_lines = [
        f"Total jobs: {len(EXPERIMENTS)}",
        f"Completed: {completed}",
        f"Failed: {failed}",
        f"n_iter={N_ITER}, num_epochs={NUM_EPOCHS}, patience={PATIENCE}, tolerance={TOLERANCE}",
        "",
    ]
    for target in TARGET_COLUMNS:
        summary_lines.append(f"=== {target} ===")
        for exp in EXPERIMENTS:
            if exp["target"] == target:
                summary_lines.append(f"  {exp['name']}")

    summary_path = EXPERIMENTS_BASE_DIR / "all_experiments_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(EXPERIMENTS_BASE_DIR / "all_results.csv", index=False)
        print(f"\nSaved: {EXPERIMENTS_BASE_DIR / 'all_results.csv'}")
        if "r2_mean" in df_all.columns and not df_all["r2_mean"].isna().all():
            best_idx = df_all["r2_mean"].idxmax()
            best = df_all.loc[best_idx]
            print("\n" + "=" * 60)
            print("OVERALL BEST")
            print("=" * 60)
            print(f"Job: {best.get('job')}")
            print(f"Config: {best.get('experiment_id')}")
            print(f"R2 = {best['r2_mean']:.4f} +/- {best.get('r2_std', 0):.4f}")

    print("\n" + "=" * 70)
    print("WAVELET COMPLETED")
    print("=" * 70)
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
