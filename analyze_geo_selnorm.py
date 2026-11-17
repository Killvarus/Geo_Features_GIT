"""
Анализ сетки run_geo_selnorm_grid.py (МТЗ): нормировка строк действует только на отбор BSBS,
проверка — МНК, Ridge и персептрон на исходных данных с z-score.

Выход (experiments/geo_selnorm/analysis/):
  curves_perceptron.png, curves_ols.png — R² на test от k: строки — наборы, столбцы — цели,
      линии — нормировки строк для отбора, пунктир — все 2418 каналов (персептрон ± std по 3 seed);
  maps/<набор>_<цель>.png — ранг каждого канала в рейтинге BSBS без нормировки:
      панели — часть (RE/IM) × компонента (YX, XY, HX), оси — пикет и частота;
  composition.csv — из чего состоят первые 50 каналов (компоненты, части, частоты, пикеты);
  overlap.csv — доля общих каналов среди первых 100 между нормировками;
  points.csv — все точки (МНК, Ridge, персептрон).

Запуск: python analyze_geo_selnorm.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.geo_norm import parse_feature

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "experiments" / "geo_selnorm"
OUT = GRID / "analysis"
DATASETS = ["d1", "d2", "d3"]
TARGETS = ["H1_8", "H2_8", "H3_8"]
NORMS = ["raw", "profile_center", "profile_scale", "profile_snv", "block_scale"]
LABEL = {
    "raw": "без нормировки",
    "profile_center": "профиль минус среднее",
    "profile_scale": "профиль на СКЗ",
    "profile_snv": "профиль SNV",
    "block_scale": "блок на СКЗ",
}
COLORS = {"raw": "#2a78d6", "profile_center": "#eb6834", "profile_scale": "#1baf7a", "profile_snv": "#eda100", "block_scale": "#e87ba4"}
DATASET_LABEL = {"d1": "d1 (mtsgrvmgn)", "d2": "d2 (ALL_3)", "d3": "d3 (ALL_3_1)"}


def load_points() -> pd.DataFrame:
    rows = []
    for check in GRID.glob("*/raw/*__*/**/linear_check.json"):
        parts = check.relative_to(GRID).parts
        dataset, (norm, channel), target = parts[0], parts[2].split("__"), parts[-2]
        norm = "all" if channel == "all" else norm
        for k, models in json.loads(check.read_text(encoding="utf-8")).items():
            for model in ("ols", "ridge"):
                rows.append({"dataset": dataset, "norm": norm, "target": target, "k": int(k), "model": model,
                             "test": models[model]["test"], "valid": models[model]["valid"], "std": 0.0})
        for point in json.loads((check.parent / "curve_points.json").read_text(encoding="utf-8")):
            rows.append({"dataset": dataset, "norm": norm, "target": target, "k": point["k"], "model": "perceptron",
                         "test": point["mean"], "valid": point["mean_valid"], "std": point["std"],
                         "epochs": float(np.mean([run["epochs"] for run in point["runs"]]))})
    return pd.DataFrame(rows)


def plot_curves(points: pd.DataFrame, model: str, name: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for col, target in enumerate(TARGETS):
            ax = axes[row, col]
            part = points[(points.dataset == dataset) & (points.target == target) & (points.model == model)]
            for norm in NORMS:
                line = part[part.norm == norm].sort_values("k")
                ax.errorbar(line["k"], line["test"], yerr=line["std"] if model == "perceptron" else None,
                            fmt="o-", ms=4, lw=2, capsize=3, color=COLORS[norm], label=LABEL[norm])
            full = part[part.norm == "all"]
            if not full.empty:
                ax.axhline(full["test"].iloc[0], color="#555555", ls="--", lw=1.4, label="все 2418 каналов")
            low = max(part["test"].min(), -0.05)
            ax.set_ylim(low - 0.02, min(1.0, part["test"].max() + 0.02) + 0.005)
            ax.set_xscale("log")
            ax.set_xlim(7, 1100)
            ax.grid(alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{DATASET_LABEL[dataset]}, {target}")
            if col == 0:
                ax.set_ylabel("R² на test")
            if row == 2:
                ax.set_xlabel("k")
    axes[0, 0].legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def ranking(dataset: str, norm: str, target: str) -> list[str]:
    frame = pd.read_csv(GRID / dataset / "raw" / f"{norm}__zscore" / "bsbs" / target / "ranking.csv")
    return frame.sort_values("Individual_Rank")["Feature"].astype(str).tolist()


def plot_map(dataset: str, target: str) -> None:
    order = ranking(dataset, "raw", target)
    rank = {name: i + 1 for i, name in enumerate(order)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True)
    image = None
    for r, part in enumerate(("RE", "IM")):
        for c, component in enumerate(("YX", "XY", "HX")):
            grid = np.full((13, 31), np.nan)
            for name, value in rank.items():
                p, comp, frequency, picket = parse_feature(name)
                if p == part and comp == component:
                    grid[frequency - 1, picket - 1] = value
            ax = axes[r, c]
            image = ax.imshow(np.log10(grid), origin="lower", aspect="auto", cmap="viridis_r",
                              extent=(0.5, 31.5, 0.5, 13.5), vmin=0, vmax=np.log10(len(order)))
            ax.set_title(f"{part} {component}")
            if r == 1:
                ax.set_xlabel("пикет")
            if c == 0:
                ax.set_ylabel("номер частоты")
    bar = fig.colorbar(image, ax=axes, shrink=0.85)
    ticks = [1, 10, 100, 1000, len(order)]
    bar.set_ticks(np.log10(ticks), labels=[str(t) for t in ticks])
    bar.set_label("ранг в рейтинге BSBS")
    fig.suptitle(f"{DATASET_LABEL[dataset]}, {target}")
    (OUT / "maps").mkdir(exist_ok=True)
    fig.savefig(OUT / "maps" / f"{dataset}_{target}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def composition(top: int = 50) -> pd.DataFrame:
    rows = []
    for dataset, target, norm in itertools.product(DATASETS, TARGETS, NORMS):
        parsed = [parse_feature(name) for name in ranking(dataset, norm, target)[:top]]
        parts, components, frequencies, pickets = zip(*parsed)
        rows.append({
            "dataset": dataset, "target": target, "norm": norm,
            "share_YX": np.mean([c == "YX" for c in components]),
            "share_XY": np.mean([c == "XY" for c in components]),
            "share_HX": np.mean([c == "HX" for c in components]),
            "share_RE": np.mean([p == "RE" for p in parts]),
            "median_frequency": float(np.median(frequencies)),
            "median_distance_from_picket_16": float(np.median(np.abs(np.array(pickets) - 16))),
        })
    return pd.DataFrame(rows)


def overlap(top: int = 100) -> pd.DataFrame:
    rows = []
    for dataset, target in itertools.product(DATASETS, TARGETS):
        sets = {norm: set(ranking(dataset, norm, target)[:top]) for norm in NORMS}
        for a, b in itertools.combinations(NORMS, 2):
            rows.append({"dataset": dataset, "target": target, "a": a, "b": b, "share": len(sets[a] & sets[b]) / top})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points = load_points()
    points.to_csv(OUT / "points.csv", index=False)
    plot_curves(points, "perceptron", "curves_perceptron.png")
    plot_curves(points, "ols", "curves_ols.png")
    for dataset, target in itertools.product(DATASETS, TARGETS):
        plot_map(dataset, target)
    composition().round(3).to_csv(OUT / "composition.csv", index=False)
    overlap().round(3).to_csv(OUT / "overlap.csv", index=False)
    print("графики и таблицы:", OUT)


if __name__ == "__main__":
    main()
