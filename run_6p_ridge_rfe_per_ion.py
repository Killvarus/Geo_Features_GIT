"""
Пакетный RFE на Ridge (alpha=700) отдельно по каждому иону.

Сначала имеет смысл пройти --direction backward, затем --direction forward.
Пакет 1500. Все 5427 канала. Важность — модуль веса на train.
Затем одновыходной персептрон на первых k признаках, шаг 500, 3 seed.
График пишется, когда готовы все шесть ионов этого направления.
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

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.neural_network import to_excel_optimized_OLP
from src.models.ridge_rfe import BackwardRidgeRFE, ForwardRidgeRFE, plot_r2_curves

CACHE = Path(__file__).resolve().parent / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
ALPHA = 700.0
PACKET = 1500
DIRECTION = "backward"
OUT_DIR = Path(__file__).resolve().parent / "experiments" / "6p_ridge_rfe_backward"
TITLE = "Обратный RFE, Ridge alpha=700"


def configure(direction: str) -> None:
    global DIRECTION, OUT_DIR, TITLE
    root = Path(__file__).resolve().parent / "experiments"
    if direction == "backward":
        DIRECTION = direction
        OUT_DIR = root / "6p_ridge_rfe_backward"
        TITLE = "Обратный RFE, Ridge alpha=700"
        return
    if direction == "forward":
        DIRECTION = direction
        OUT_DIR = root / "6p_ridge_rfe_forward"
        TITLE = "Прямой RFE, Ridge alpha=700"
        return
    raise ValueError("direction должен быть backward или forward")


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


def rank_ion(ion: str, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    ion_dir = OUT_DIR / ion
    path = ion_dir / "ranking.csv"
    if path.exists():
        print(ion, DIRECTION, "ranking exists", flush=True)
        return pd.read_csv(path)
    ion_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    print(ion, DIRECTION, "start", X_train.shape, "alpha", ALPHA, flush=True)
    if DIRECTION == "backward":
        selector = BackwardRidgeRFE(alpha=ALPHA, n_features_to_drop=PACKET)
    else:
        selector = ForwardRidgeRFE(alpha=ALPHA, n_features_to_add=PACKET)
    selector.fit(X_train, y_train)
    ranking = selector.ranking_frame()
    ranking.to_csv(path, index=False)
    print(ion, DIRECTION, "done", len(ranking), "seconds", round(selector.elapsed_seconds_, 1), flush=True)
    return ranking


def train_curve(ion: str, X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame) -> None:
    ion_dir = OUT_DIR / ion
    points_path = ion_dir / "curve_points.json"
    done = {}
    if points_path.exists():
        done = {point["k"]: point for point in json.loads(points_path.read_text(encoding="utf-8"))}
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
        r2s = [item["r2"] for item in runs]
        point = {"k": k, "runs": runs, "mean": float(np.mean(r2s)), "std": float(np.std(r2s, ddof=1))}
        points.append(point)
        points_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
        print(ion, "k", k, "R2", round(point["mean"], 4), "±", round(point["std"], 4), flush=True)
    plot_r2_curves(OUT_DIR, TARGETS, TITLE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True, choices=["backward", "forward"])
    parser.add_argument("--ion", choices=TARGETS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    configure(args.direction)
    if args.plot_only:
        plot_r2_curves(OUT_DIR, TARGETS, TITLE)
        return
    if args.ion is None:
        parser.error("нужен --ion или --plot-only")
    X, y = load_ion(args.ion)
    ranking = rank_ion(args.ion, X, y)
    train_curve(args.ion, X, y, ranking)
    print(args.ion, DIRECTION, "finished", flush=True)


if __name__ == "__main__":
    main()
