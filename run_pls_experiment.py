"""
Запуск PLS (самостоятельный регрессор, без нейросети) для всех данных и таргетов.

Порядок как у aggregation/PCA: сначала H3_8 на всех датасетах, потом H1_8, потом H2_8.

Запуск:
  python run_pls_experiment.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')

from src.config import PROJECT_ROOT, EXPERIMENTS_DIR
from src.experiment import PLSExperiment
from src.utils import load_mtz_data
from src.utils.logging_utils import configure_stdio, install_safe_print, close_logger_file_handlers


DATA_DIR = PROJECT_ROOT / "Data"
RUN_TAG = "2026-09-04_pls"
EXPERIMENTS_BASE_DIR = EXPERIMENTS_DIR / RUN_TAG

DIFFICULTIES = ["Difficult_1", "Difficult_2", "Difficult_3"]
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
N_COMPONENTS_LIST = [16, 64, 256, 512, 1024, 1536, 2048]


def _jobs():
    jobs = []
    for target in TARGET_COLUMNS:
        for difficulty in DIFFICULTIES:
            short = DIFFICULTY_TO_SHORT[difficulty]
            jobs.append({
                "name": f"pls_{short}_{target}",
                "difficulty": difficulty,
                "target": target,
                "data": DIFFICULTY_TO_FILES[difficulty],
            })
    return jobs


EXPERIMENTS = _jobs()


def run_one(train, valid, test, target, exp_name):
    print(f"\n{'=' * 60}")
    print(f"PLS: {exp_name} (target={target})")
    print(f"{'=' * 60}")

    experiment = PLSExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=[target],
        experiment_name=exp_name,
        base_dir=EXPERIMENTS_BASE_DIR / exp_name,
    )

    results = experiment.run_grid(n_components_list=N_COMPONENTS_LIST)
    if results.empty:
        print(f"  [X] No PLS results for {exp_name}")
        close_logger_file_handlers(experiment.logger)
        return None

    experiment.plot_comparison(metric="r2_mean", save=True)
    experiment.plot_comparison(metric="mse_mean", save=True)
    results.to_csv(EXPERIMENTS_BASE_DIR / exp_name / "all_results.csv", index=False)

    best = experiment.get_best_result()
    if best:
        print(
            f"  Best: n={best['n_components']}, "
            f"R2={best['r2_mean']:.4f} +/- {best['r2_std']:.4f}"
        )
    print(f"[OK] PLS {exp_name} done")
    close_logger_file_handlers(experiment.logger)
    return experiment


def main():
    configure_stdio()
    install_safe_print()
    EXPERIMENTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PLS EXPERIMENT")
    print("=" * 70)
    print(f"Results: {EXPERIMENTS_BASE_DIR}")
    print(f"Jobs: {len(EXPERIMENTS)}")
    print(f"Components: {N_COMPONENTS_LIST}")
    print(f"Targets: {TARGET_COLUMNS}")
    print("Order: all H3_8, then H1_8, then H2_8")
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

        csv_path = exp_out / "all_results.csv"
        if csv_path.exists():
            print(f"  [SKIP] results already exist: {csv_path}")
            completed += 1
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
            print(f"Difficulty: {best.get('difficulty')}")
            print(f"n_components: {int(best['n_components'])}")
            print(f"R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}")

    print("\n" + "=" * 70)
    print("PLS COMPLETED")
    print("=" * 70)
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
