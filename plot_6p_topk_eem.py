"""
Где на матрице возбуждения-испускания лежат первые k каналов рейтингов 6P.

Две серии картинок в experiments/6p_selection_eem/topk/:
  by_method/<метод>_top<k>.png — один метод, шесть ионов;
  by_ion/<ион>_top<k>.png      — один ион, все методы рядом.
Плюс таблица разнообразия diversity.csv: сколько разных строк возбуждения
занято и среднее расстояние до ближайшего соседа (в нм, возбуждение и
испускание в одних единицах).
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

from src.visualization.eem_selection import (
    channel_coordinates,
    mean_train_matrix,
    save_selection_grid,
    top_features,
)

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
OUT_DIR = ROOT / "experiments" / "6p_selection_eem" / "topk"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
METHODS = {
    "uniform": "6p_baseline_uniform",
    "corr": "6p_baseline_corr",
    "ridge_once": "6p_baseline_ridge_once",
    "rfe_1": "6p_rfe_ridge_a700_step1",
    "rfe_0.1": "6p_rfe_ridge_a700_step0.1",
    "ppop_cv": "6p_ppop_ridge_a700_step0.1_cv5",
    "rfe_0.1_loo": "6p_rfe_ridge_loo_step0.1",
    "ppop_cv_loo": "6p_ppop_ridge_loo_step0.1_cv5",
    "rfe_0.1_1se": "6p_rfe_ridge_loo1se_step0.1",
    "ppop_cv_1se": "6p_ppop_ridge_loo1se_step0.1_cv5",
}


def diversity(features: list[str]) -> dict:
    coords = np.array([channel_coordinates(f) for f in features], dtype=float)
    rows = len(set(coords[:, 0]))
    if len(coords) < 2:
        return {"excitation_rows": rows, "nn_distance_nm": float("nan")}
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    return {"excitation_rows": rows, "nn_distance_nm": float(distance.min(axis=1).mean())}


def load_rankings(methods: dict[str, str]) -> dict[tuple[str, str], pd.DataFrame]:
    rankings = {}
    for label, folder in methods.items():
        for ion in TARGETS:
            path = ROOT / "experiments" / folder / ion / "ranking.csv"
            if path.exists():
                rankings[(label, ion)] = pd.read_csv(path)
            else:
                print("missing", path, flush=True)
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[30, 100])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    methods = {m: METHODS[m] for m in args.methods}
    matrix = mean_train_matrix(np.load(CACHE)["X"])
    rankings = load_rankings(methods)
    rows = []
    for k in args.k:
        for label in methods:
            panels = [(ion, top_features(rankings[(label, ion)], k)) for ion in TARGETS if (label, ion) in rankings]
            if panels:
                save_selection_grid(matrix, panels, f"{label}: первые {k} каналов", OUT_DIR / "by_method" / f"{label}_top{k}.png")
            for ion, features in panels:
                rows.append({"method": label, "ion": ion, "k": k, **diversity(features)})
        for ion in TARGETS:
            panels = [(label, top_features(rankings[(label, ion)], k)) for label in methods if (label, ion) in rankings]
            if panels:
                save_selection_grid(matrix, panels, f"{ion}: первые {k} каналов", OUT_DIR / "by_ion" / f"{ion}_top{k}.png", ncols=4)
        print("k", k, "done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "diversity.csv", index=False)
    print(table.pivot_table(index=["k", "method"], values=["excitation_rows", "nn_distance_nm"]).round(1).to_string())


if __name__ == "__main__":
    main()
