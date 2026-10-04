"""Три seed на каждую точку кривой ППОП и график R² по ионам с разбросом."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
RANKING = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels\ranking.csv")
CURVE = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels\curve_points.json")
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels")
POINTS_PATH = OUT_DIR / "curve_points_3seeds.json"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
K_GRID = [50, 100, 200, 400, 800, 1357, 2000, 2714, 3500, 4500, 5427]


def feature_names() -> list[str]:
    exc = [280 + 5 * i for i in range(27)]
    em = list(range(375, 576))
    return [f"E{e}_em{w}" for e in exc for w in em]


def train_two(X: np.ndarray, y: np.ndarray, columns: list[str], tag: str) -> list[dict]:
    index = {name: i for i, name in enumerate(feature_names())}
    chosen = X[:, [index[c] for c in columns]]
    X_df = pd.DataFrame(chosen, columns=columns)
    y_df = pd.DataFrame(y, columns=TARGETS)
    run_dir = OUT_DIR / "curve3" / tag
    results = to_excel_optimized_OLP(
        file_name=str(run_dir / "metrics.xlsx"),
        n_iter=2,
        X_train=X_df.iloc[:N_TRAIN],
        y_train=y_df.iloc[:N_TRAIN],
        X_valid=X_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
        y_valid=y_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
        X_test=X_df.iloc[N_TRAIN + N_VALID:],
        y_test=y_df.iloc[N_TRAIN + N_VALID:],
        batch_size=64,
        input_dim=len(columns),
        output_dim=len(TARGETS),
        learning_rate=0.001,
        num_epochs=10**9,
        patience=300,
        tolerance=0.003,
        tolerance_mode="relative",
        hidden_dim=32,
        save_plots_dir=str(run_dir / "learning_curves"),
        optimizer_type="adam",
        random_state=43,
        device="cpu",
        log_file=str(run_dir / "training.log"),
        enable_cv=False,
    )
    runs = []
    for offset, result in enumerate(results):
        df_test = result["test"]
        runs.append({
            "seed": 43 + offset,
            "r2": {TARGETS[i]: float(df_test.loc[f"Оценка{i + 1}", "R2"]) for i in range(len(TARGETS))},
        })
    return runs


def plot(points: list[dict]) -> None:
    ks = [p["k"] for p in points]
    full = points[-1]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, name in zip(axes.ravel(), TARGETS):
        mean = [p["mean"][name] for p in points]
        std = [p["std"][name] for p in points]
        ax.errorbar(ks, mean, yerr=std, fmt="o-", color="#1f4e79", lw=1.6, ms=4.5, capsize=3, label="ППОП, 3 запуска")
        ax.axhline(full["mean"][name], color="#c45c26", ls="--", lw=1.3)
        ax.axhspan(full["mean"][name] - full["std"][name], full["mean"][name] + full["std"][name], color="#c45c26", alpha=0.15)
        ax.set_title(f"{name}   без отбора {full['mean'][name]:.3f}±{full['std'][name]:.3f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5700)
    axes[0, 0].set_ylabel("R² на тесте")
    axes[1, 0].set_ylabel("R² на тесте")
    for ax in axes[1]:
        ax.set_xlabel("Число признаков")
    fig.suptitle("ППОП: R² по ионам, среднее ± стандартное отклонение трёх запусков", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r2_vs_n_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    seed42 = {p["k"]: p["r2"] for p in json.loads(CURVE.read_text(encoding="utf-8"))}
    ranking = pd.read_csv(RANKING).sort_values("Individual_Rank")
    ordered = ranking["Feature"].tolist()
    names = feature_names()
    if ordered_set_mismatch(ordered, names):
        raise RuntimeError("ранжирование не совпало с порядком каналов")
    blob = np.load(CACHE)
    X = blob["X"]
    y = blob["y"]
    done = {}
    if POINTS_PATH.exists():
        done = {p["k"]: p for p in json.loads(POINTS_PATH.read_text(encoding="utf-8"))}
    points = []
    for k in K_GRID:
        if k in done and len(done[k]["runs"]) == 3:
            points.append(done[k])
            print("skip", k, flush=True)
            continue
        columns = names if k == len(names) else ordered[:k]
        print("train", k, flush=True)
        runs = [{"seed": 42, "r2": seed42[k]}] + train_two(X, y, columns, tag=f"k{k}")
        mean = {t: float(np.mean([r["r2"][t] for r in runs])) for t in TARGETS}
        std = {t: float(np.std([r["r2"][t] for r in runs], ddof=1)) for t in TARGETS}
        point = {"k": k, "runs": runs, "mean": mean, "std": std}
        points.append(point)
        POINTS_PATH.write_text(json.dumps(points, indent=2), encoding="utf-8")
        print(" ", k, {t: round(mean[t], 3) for t in TARGETS}, flush=True)
    plot(points)
    print("wrote", OUT_DIR / "r2_vs_n_features.png", flush=True)


def ordered_set_mismatch(ordered: list[str], names: list[str]) -> bool:
    return set(ordered) != set(names)


if __name__ == "__main__":
    main()
