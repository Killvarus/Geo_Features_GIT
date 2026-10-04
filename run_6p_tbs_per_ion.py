"""
ППОП как TrueBackwardFeatureSelection, отдельно на каждый ион.

Пакет 1500. Все 5427 канала. cv=None: одна регрессия на train, без кросс-валидации.
Затем одновыходной персептрон на первых k признаках рейтинга, шаг 500, 3 seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.feature_selection import TrueBackwardFeatureSelection
from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_tbs_per_ion")
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
DROP = 1500


def feature_names() -> list[str]:
    exc = [280 + 5 * i for i in range(27)]
    em = list(range(375, 576))
    return [f"E{e}_em{w}" for e in exc for w in em]


def load_ion(ion: str):
    blob = np.load(CACHE)
    names = feature_names()
    X = pd.DataFrame(blob["X"], columns=names)
    y = pd.Series(blob["y"][:, TARGETS.index(ion)], name=ion)
    return X, y


def k_grid(n_features: int) -> list[int]:
    ks = list(range(500, n_features, 500))
    if n_features not in ks:
        ks.append(n_features)
    return ks


def rank_ion(ion: str, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    ion_dir = OUT_DIR / ion
    path = ion_dir / "ranking.csv"
    if path.exists():
        print(ion, "ranking exists", flush=True)
        return pd.read_csv(path)
    ion_dir.mkdir(parents=True, exist_ok=True)
    print(ion, "PPOP start", X_train.shape, flush=True)
    selector = TrueBackwardFeatureSelection(
        estimator=LinearRegression(),
        n_features_to_drop=DROP,
        cv=None,
    )
    selector.fit(X_train, y_train)
    ranking = selector.get_ranking_df()
    ranking.to_csv(path, index=False)
    print(ion, "PPOP done", len(ranking), flush=True)
    return ranking


def train_curve(ion: str, X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame) -> list[dict]:
    ion_dir = OUT_DIR / ion
    points_path = ion_dir / "curve_points.json"
    done = {}
    if points_path.exists():
        done = {p["k"]: p for p in json.loads(points_path.read_text(encoding="utf-8"))}
    ordered = ranking.sort_values("Individual_Rank")["Feature"].tolist()
    points = []
    y_df = y.to_frame()
    for k in k_grid(X.shape[1]):
        if k in done and len(done[k].get("runs", [])) == 3:
            points.append(done[k])
            print(ion, "skip k", k, flush=True)
            continue
        cols = ordered[:k]
        print(ion, "OLP k", k, flush=True)
        run_dir = ion_dir / f"k{k}"
        results = to_excel_optimized_OLP(
            file_name=str(run_dir / "metrics.xlsx"),
            n_iter=3,
            X_train=X.iloc[:N_TRAIN][cols],
            y_train=y_df.iloc[:N_TRAIN],
            X_valid=X.iloc[N_TRAIN:N_TRAIN + N_VALID][cols],
            y_valid=y_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
            X_test=X.iloc[N_TRAIN + N_VALID:][cols],
            y_test=y_df.iloc[N_TRAIN + N_VALID:],
            batch_size=64,
            input_dim=k,
            output_dim=1,
            learning_rate=0.001,
            num_epochs=10**9,
            patience=300,
            tolerance=0.003,
            tolerance_mode="relative",
            hidden_dim=32,
            save_plots_dir=str(run_dir / "learning_curves"),
            optimizer_type="adam",
            random_state=42,
            device="cpu",
            log_file=str(run_dir / "training.log"),
            enable_cv=False,
        )
        runs = []
        for i, result in enumerate(results):
            runs.append({"seed": 42 + i, "r2": float(result["test"].loc["Оценка1", "R2"])})
        r2s = [r["r2"] for r in runs]
        point = {
            "k": k,
            "runs": runs,
            "mean": float(np.mean(r2s)),
            "std": float(np.std(r2s, ddof=1)),
        }
        points.append(point)
        points_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
        print(ion, "k", k, "R2", round(point["mean"], 4), "±", round(point["std"], 4), flush=True)
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ion", required=True, choices=TARGETS)
    args = parser.parse_args()
    X, y = load_ion(args.ion)
    ranking = rank_ion(args.ion, X.iloc[:N_TRAIN], y.iloc[:N_TRAIN])
    train_curve(args.ion, X, y, ranking)
    print(args.ion, "finished", flush=True)


if __name__ == "__main__":
    main()
