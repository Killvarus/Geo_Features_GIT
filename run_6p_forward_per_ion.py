"""
Прямой пакетный отбор Ridge (alpha=700) отдельно по каждому иону.

Пакет 1500. Все 5427 канала. MSE на validation.
Затем одновыходной персептрон на первых k признаках, шаг 500, 3 seed.
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

from src.models.forward_selection import ForwardBatchRidgeSelection
from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(__file__).resolve().parent / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
OUT_DIR = Path(__file__).resolve().parent / "experiments" / "6p_forward_ridge_valid"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
ALPHA = 700.0
PACKET = 1500


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
        print(ion, "ranking exists", flush=True)
        return pd.read_csv(path)
    ion_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    X_valid, y_valid = X.iloc[N_TRAIN:N_TRAIN + N_VALID], y.iloc[N_TRAIN:N_TRAIN + N_VALID]
    print(ion, "forward start", X_train.shape, "alpha", ALPHA, flush=True)
    selector = ForwardBatchRidgeSelection(alpha=ALPHA, n_features_to_add=PACKET)
    selector.fit(X_train, y_train, X_valid, y_valid)
    ranking = selector.ranking_frame()
    ranking.to_csv(path, index=False)
    print(ion, "forward done", len(ranking), "seconds", round(selector.elapsed_seconds_, 1), flush=True)
    return ranking


def train_curve(ion: str, X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame) -> None:
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
        r2s = [item["r2"] for item in runs]
        point = {"k": k, "runs": runs, "mean": float(np.mean(r2s)), "std": float(np.std(r2s, ddof=1))}
        points.append(point)
        points_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
        print(ion, "k", k, "R2", round(point["mean"], 4), "±", round(point["std"], 4), flush=True)
    maybe_plot()


def maybe_plot() -> None:
    ions = TARGETS
    paths = [OUT_DIR / ion / "curve_points.json" for ion in ions]
    if not all(path.exists() for path in paths):
        print("plot waiting for", [ion for ion, path in zip(ions, paths) if not path.exists()], flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, ion in zip(axes.ravel(), ions):
        points = sorted(json.loads((OUT_DIR / ion / "curve_points.json").read_text(encoding="utf-8")), key=lambda p: p["k"])
        ks = [p["k"] for p in points]
        mean = [p["mean"] for p in points]
        std = [p["std"] for p in points]
        full = next(p for p in points if p["k"] == max(ks))
        ax.errorbar(ks, mean, yerr=std, fmt="o-", color="#1f4e79", lw=1.6, ms=4.5, capsize=3)
        ax.axhline(full["mean"], color="#c45c26", ls="--", lw=1.3)
        ax.axhspan(full["mean"] - full["std"], full["mean"] + full["std"], color="#c45c26", alpha=0.15)
        ax.set_title(f"{ion}   без отбора {full['mean']:.3f}±{full['std']:.3f}")
        ax.set_xlim(max(ks) + 200, 0)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("R² на тесте")
    axes[1, 0].set_ylabel("R² на тесте")
    for ax in axes[1]:
        ax.set_xlabel("Число признаков")
    fig.suptitle("Прямой отбор, Ridge alpha=700, MSE на validation", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "r2_vs_n_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("plot", out, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ion", choices=TARGETS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        maybe_plot()
        return
    if args.ion is None:
        parser.error("нужен --ion или --plot-only")
    X, y = load_ion(args.ion)
    ranking = rank_ion(args.ion, X, y)
    train_curve(args.ion, X, y, ranking)
    print(args.ion, "finished", flush=True)


if __name__ == "__main__":
    main()
