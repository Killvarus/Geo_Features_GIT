"""
Сравнение рейтингов 6P без персептрона.

Для каждого рейтинга и иона: Ridge на первых k признаках, alpha по LOO на train,
R² на validation (src/evaluation/ranking_eval.py). Test не используется.

Кроме рейтингов из experiments/ считаются базовые:
  random      — случайный порядок (среднее по 2 seed);
  corr        — |корреляция| канала с y на train;
  ridge_once  — |β| одной модели Ridge alpha=700 на всех каналах, без рекурсии;
  uniform     — равномерное прореживание спектра от грубого к мелкому.

Результаты дописываются в experiments/6p_ranking_eval/r2_valid.csv,
уже посчитанные точки пропускаются. Графики — там же.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.evaluation.ranking_eval import ridge_loo_r2

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
OUT_DIR = ROOT / "experiments" / "6p_ranking_eval"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
K_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5427]

# Подпись -> папка в experiments/ с <ион>/ranking.csv.
# Рейтинги, отобранные по validation, здесь завышены: оценка идёт на той же validation.
VALIDATION_BIASED = {"ppop_valid", "ppop_old_valid"}
RANKING_DIRS = {
    "ppop_cv": "6p_ppop_ridge_a700_step0.1_cv5",
    "ppop_valid": "6p_ppop_ridge_a700_step0.1_valid",
    "rfe_1": "6p_rfe_ridge_a700_step1",
    "rfe_0.1": "6p_rfe_ridge_a700_step0.1",
    "ppop_old_valid": "6p_tbs_per_ion_valid",
    "ppop_cv_loo": "6p_ppop_ridge_loo_step0.1_cv5",
    "rfe_1_loo": "6p_rfe_ridge_loo_step1",
    "rfe_0.1_loo": "6p_rfe_ridge_loo_step0.1",
    "rfe_0.1_1se": "6p_rfe_ridge_loo1se_step0.1",
    "ppop_cv_1se": "6p_ppop_ridge_loo1se_step0.1_cv5",
    "ppop_cv_1se_reshuffle": "6p_ppop_ridge_loo1se_step0.1_cv5_reshuffle",
    "rfe_0.1_1se_bag5": "6p_rfe_ridge_loo1se_step0.1_bag5",
    "rfe_s5": "6p_rfe_ridge_a700_step0.1_spaced5",
    "rfe_s10": "6p_rfe_ridge_a700_step0.1_spaced10",
    "rfe_s20": "6p_rfe_ridge_a700_step0.1_spaced20",
    "ppop_s5": "6p_ppop_ridge_a700_step0.1_cv5_spaced5",
    "ppop_s10": "6p_ppop_ridge_a700_step0.1_cv5_spaced10",
    "ppop_s20": "6p_ppop_ridge_a700_step0.1_cv5_spaced20",
}
BASELINES = ["random", "corr", "ridge_once", "uniform"]


def feature_names() -> list[str]:
    exc = [280 + 5 * i for i in range(27)]
    em = list(range(375, 576))
    return [f"E{e}_em{w}" for e in exc for w in em]


def load():
    blob = np.load(CACHE)
    return blob["X"].astype(float), blob["y"].astype(float)


def van_der_corput_order(n: int) -> list[int]:
    """Порядок индексов 0..n-1, равномерно заполняющий отрезок от грубого к мелкому."""
    bits = max(1, int(np.ceil(np.log2(n))))
    keys = [int(format(i, f"0{bits}b")[::-1], 2) for i in range(2**bits)]
    order = sorted(range(2**bits), key=lambda i: keys[i])
    return [i for i in order if i < n]


def baseline_order(name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> list[int]:
    p = X_train.shape[1]
    if name == "random":
        return list(np.random.default_rng(seed).permutation(p))
    if name == "uniform":
        return van_der_corput_order(p)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    Z = (X_train - X_train.mean(axis=0)) / std
    yc = y_train - y_train.mean()
    if name == "corr":
        score = np.abs(Z.T @ yc) / (np.linalg.norm(yc) * np.sqrt(len(yc)))
    elif name == "ridge_once":
        u, s, vt = np.linalg.svd(Z, full_matrices=False)
        score = np.abs(vt.T @ (s / (s**2 + 700.0) * (u.T @ yc)))
    else:
        raise ValueError(name)
    return list(np.argsort(-score, kind="stable"))


def save_baseline_ranking(method: str, ion: str, order: list[int], names: list[str]) -> Path:
    """Базовый порядок в том же формате, что у отборов: experiments/6p_baseline_<метод>/<ион>/ranking.csv."""
    path = ROOT / "experiments" / f"6p_baseline_{method}" / ion / "ranking.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        ranks = np.arange(1, len(order) + 1)
        pd.DataFrame(
            {"Feature": [names[i] for i in order], "Group_Rank": ranks, "Individual_Rank": ranks, "Score": -ranks}
        ).to_csv(path, index=False)
    return path


def ranking_order(folder: str, ion: str, names: list[str]) -> list[int] | None:
    path = ROOT / "experiments" / folder / ion / "ranking.csv"
    if not path.exists():
        return None
    ranking = pd.read_csv(path).sort_values("Individual_Rank")
    index = {name: i for i, name in enumerate(names)}
    return [index[str(f)] for f in ranking["Feature"]]


def evaluate(methods: list[str], ions: list[str], max_k: int | None = None) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "r2_valid.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["method", "ion", "k", "r2", "alpha"])
    done = set(zip(table["method"], table["ion"], table["k"].astype(int)))
    X, Y = load()
    names = feature_names()
    rows = []
    for ion in ions:
        y = Y[:, TARGETS.index(ion)]
        X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
        X_valid, y_valid = X[N_TRAIN:N_TRAIN + N_VALID], y[N_TRAIN:N_TRAIN + N_VALID]
        # На всех признаках набор один и тот же при любом рейтинге — считаем один раз.
        full = None
        for method in methods:
            if method == "random":
                orders = [baseline_order("random", X_train, y_train, seed) for seed in range(2)]
            elif method in BASELINES:
                orders = [baseline_order(method, X_train, y_train)]
            if method in BASELINES:
                save_baseline_ranking(method, ion, orders[0], names)
            else:
                order = ranking_order(RANKING_DIRS.get(method, method), ion, names)
                if order is None:
                    print("no ranking", method, ion, flush=True)
                    continue
                orders = [order]
            for k in K_GRID:
                if max_k is not None and k > max_k:
                    continue
                if (method, ion, k) in done:
                    continue
                if k == X.shape[1]:
                    if full is None:
                        full = ridge_loo_r2(X_train, y_train, X_valid, y_valid)
                    results = [full]
                else:
                    results = [ridge_loo_r2(X_train[:, o[:k]], y_train, X_valid[:, o[:k]], y_valid) for o in orders]
                r2 = float(np.mean([r for r, _ in results]))
                alpha = float(np.median([a for _, a in results]))
                rows.append({"method": method, "ion": ion, "k": k, "r2": r2, "alpha": alpha})
            print(ion, method, "done", flush=True)
            if rows:
                table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
                rows = []
                table.to_csv(path, index=False)
    return table


def plot(table: pd.DataFrame, methods: list[str], name: str = "r2_valid_vs_k.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    colors = plt.get_cmap("tab10")
    for ax, ion in zip(axes.ravel(), TARGETS):
        part = table[table["ion"] == ion]
        for i, method in enumerate(methods):
            line = part[part["method"] == method].sort_values("k")
            if line.empty:
                continue
            style = "--" if method in BASELINES else "-"
            ax.plot(line["k"], line["r2"], style, marker="o", ms=3, lw=1.4, color=colors(i % 10), label=method)
        ax.set_xscale("log")
        ax.set_title(ion)
        ax.grid(True, alpha=0.3)
        lows = part[part["k"] >= 50]["r2"]
        if not lows.empty:
            ax.set_ylim(max(lows.min() - 0.05, -0.2), min(1.0, part["r2"].max() + 0.03))
    for ax in axes[1]:
        ax.set_xlabel("Число признаков k (лог. шкала)")
    for ax in axes[:, 0]:
        ax.set_ylabel("R² Ridge на validation")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Ridge (alpha по LOO на train) на первых k признаках рейтинга", y=1.01)
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    default = BASELINES + [m for m in RANKING_DIRS if m not in VALIDATION_BIASED]
    parser.add_argument("--methods", nargs="+", default=default)
    parser.add_argument("--ions", nargs="+", default=TARGETS)
    parser.add_argument("--plot-name", default="r2_valid_vs_k.png")
    parser.add_argument("--max-k", type=int, help="не считать точки с большим k")
    args = parser.parse_args()
    table = evaluate(args.methods, args.ions, args.max_k)
    print("plot", plot(table, args.methods, args.plot_name), flush=True)


if __name__ == "__main__":
    main()
