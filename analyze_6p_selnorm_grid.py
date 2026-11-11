"""
Анализ сетки run_6p_selnorm_grid.py: нормировки действуют только на отбор,
персептрон всегда учится на исходных спектрах с z-score каналов.

Графики (experiments/6p_selnorm_grid/analysis/), R² персептрона ± std по 3 seed:
  curves/<test|valid>/<ион>.png      — BSBS при разных нормировках спектров (каналы z-score),
                                       пунктир — все каналы; «среднее» — по 6 ионам
  methods/<test|valid>/<ион>.png     — BSBS и BSBS с возвратом при каждой нормировке спектров
  heatmaps/<test|valid>/<метод>.png  — средний по ионам R²: строки — спектры, столбцы — каналы, панели — k
  difference/<test|valid>.png        — BSBS с возвратом минус BSBS, среднее по ионам и k
Таблицы: points.csv (все точки), methods_by_k.csv, best_variants.csv.

Запуск: python analyze_6p_selnorm_grid.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "experiments" / "6p_selnorm_grid"
OUT = GRID / "analysis"
IONS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
SPECTRA = ["raw", "max", "area", "snv", "row_max"]
CHANNELS = ["zscore", "robust", "pareto", "center"]
METHODS = ["bsbs", "fbsbs"]
SPECTRUM_LABEL = {"raw": "без нормировки", "max": "на максимум", "area": "на площадь", "snv": "SNV", "row_max": "по строкам"}
METHOD_LABEL = {"bsbs": "BSBS", "fbsbs": "BSBS с возвратом"}
SPLIT_LABEL = {"test": "R² на test", "valid": "R² на validation"}
# Категориальная палитра (проверена валидатором на различимость, в том числе при нарушениях цветового зрения).
COLORS = {"raw": "#2a78d6", "max": "#eb6834", "area": "#1baf7a", "snv": "#eda100", "row_max": "#e87ba4"}


def load_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in GRID.glob("*__*/*/*/curve_points.json"):
        combo, method, ion = path.parts[-4:-1]
        spectrum, channel = combo.split("__")
        for point in json.loads(path.read_text(encoding="utf-8")):
            rows.append(
                {
                    "spectrum": spectrum,
                    "channel": channel,
                    "method": method,
                    "ion": ion,
                    "k": point["k"],
                    "test": point["mean"],
                    "test_std": point["std"],
                    "valid": point["mean_valid"],
                    "valid_std": float(np.std([run["r2_valid"] for run in point["runs"]], ddof=1)),
                }
            )
    full = []
    for path in GRID.glob("all_channels/*/curve_points.json"):
        point = json.loads(path.read_text(encoding="utf-8"))[0]
        full.append(
            {
                "ion": path.parent.name,
                "test": point["mean"],
                "test_std": point["std"],
                "valid": point["mean_valid"],
                "valid_std": float(np.std([run["r2_valid"] for run in point["runs"]], ddof=1)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(full).set_index("ion")


def with_mean(points: pd.DataFrame, full: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавляет «ион» mean: среднее по 6 ионам, std среднего — корень из суммы дисперсий / 6."""
    keys = ["spectrum", "channel", "method", "k"]
    grouped = points.groupby(keys)
    mean = grouped[split].mean().to_frame()
    mean[f"{split}_std"] = grouped[f"{split}_std"].apply(lambda s: float(np.sqrt((s**2).sum())) / len(s))
    mean = mean.reset_index().assign(ion="среднее")
    full_mean = pd.DataFrame(
        {split: [full[split].mean()], f"{split}_std": [float(np.sqrt((full[f"{split}_std"] ** 2).sum())) / len(full)]},
        index=["среднее"],
    )
    return pd.concat([points, mean], ignore_index=True), pd.concat([full[[split, f"{split}_std"]], full_mean])


def style(ax, split: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel(SPLIT_LABEL[split])
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_curves(points: pd.DataFrame, full: pd.DataFrame, split: str) -> None:
    folder = OUT / "curves" / split
    folder.mkdir(parents=True, exist_ok=True)
    for ion in IONS + ["среднее"]:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for spectrum in SPECTRA:
            line = points[(points.ion == ion) & (points.method == "bsbs") & (points.channel == "zscore") & (points.spectrum == spectrum)]
            line = line.sort_values("k")
            ax.errorbar(line["k"], line[split], yerr=line[f"{split}_std"], fmt="o-", ms=4, lw=2, capsize=3,
                        color=COLORS[spectrum], label=SPECTRUM_LABEL[spectrum])
        ax.axhline(full.loc[ion, split], color="#555555", ls="--", lw=1.5, label="все каналы")
        style(ax, split)
        ax.set_title(f"{ion}: BSBS, нормировка спектров только для отбора")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(folder / f"{ion}.png", dpi=130)
        plt.close(fig)


def plot_methods(points: pd.DataFrame, full: pd.DataFrame, split: str) -> None:
    folder = OUT / "methods" / split
    folder.mkdir(parents=True, exist_ok=True)
    for ion in IONS + ["среднее"]:
        fig, axes = plt.subplots(1, len(SPECTRA), figsize=(18, 3.8), sharey=True)
        for ax, spectrum in zip(axes, SPECTRA):
            for method, dash in (("bsbs", "-"), ("fbsbs", "--")):
                line = points[(points.ion == ion) & (points.method == method) & (points.channel == "zscore") & (points.spectrum == spectrum)]
                line = line.sort_values("k")
                ax.errorbar(line["k"], line[split], yerr=line[f"{split}_std"], fmt="o" + dash, ms=4, lw=2, capsize=3,
                            color=COLORS[spectrum], mfc="white" if method == "fbsbs" else COLORS[spectrum],
                            label=METHOD_LABEL[method])
            ax.axhline(full.loc[ion, split], color="#555555", ls=":", lw=1.2, label="все каналы")
            style(ax, split)
            ax.set_title(SPECTRUM_LABEL[spectrum])
            if ax is not axes[0]:
                ax.set_ylabel("")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(ion)
        fig.tight_layout()
        fig.savefig(folder / f"{ion}.png", dpi=120)
        plt.close(fig)


def plot_heatmaps(points: pd.DataFrame, full: pd.DataFrame, split: str) -> None:
    folder = OUT / "heatmaps" / split
    folder.mkdir(parents=True, exist_ok=True)
    mean = points[points.ion == "среднее"]
    ks = sorted(mean["k"].unique())
    low, high = mean[split].quantile(0.05), max(mean[split].max(), full.loc["среднее", split])
    for method in METHODS:
        fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
        for ax, k in zip(axes.ravel(), ks):
            grid = (
                mean[(mean.method == method) & (mean.k == k)]
                .pivot(index="spectrum", columns="channel", values=split)
                .reindex(index=SPECTRA, columns=CHANNELS)
            )
            image = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=low, vmax=high, aspect="auto")
            for i in range(len(SPECTRA)):
                for j in range(len(CHANNELS)):
                    value = grid.iat[i, j]
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                            color="black" if value > low + 0.6 * (high - low) else "white")
            ax.set_xticks(range(len(CHANNELS)), CHANNELS, fontsize=8)
            first_column = ax in axes[:, 0]
            ax.set_yticks(range(len(SPECTRA)), [SPECTRUM_LABEL[s] for s in SPECTRA] if first_column else [], fontsize=8)
            ax.set_title(f"k = {k}")
        fig.colorbar(image, ax=axes, shrink=0.8, label=f"{SPLIT_LABEL[split]}, среднее по ионам")
        fig.suptitle(f"{METHOD_LABEL[method]}; все каналы: {full.loc['среднее', split]:.3f}")
        fig.savefig(folder / f"{method}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_difference(points: pd.DataFrame, split: str) -> pd.DataFrame:
    folder = OUT / "difference"
    folder.mkdir(parents=True, exist_ok=True)
    wide = points[points.ion.isin(IONS)].pivot_table(index=["spectrum", "channel", "ion", "k"], columns="method", values=split)
    wide["diff"] = wide["fbsbs"] - wide["bsbs"]
    grid = wide.groupby(["spectrum", "channel"])["diff"].mean().unstack().reindex(index=SPECTRA, columns=CHANNELS)
    limit = float(np.abs(grid.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    image = ax.imshow(grid.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(SPECTRA)):
        for j in range(len(CHANNELS)):
            ax.text(j, i, f"{grid.iat[i, j]:+.4f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(CHANNELS)), CHANNELS)
    ax.set_yticks(range(len(SPECTRA)), [SPECTRUM_LABEL[s] for s in SPECTRA])
    fig.colorbar(image, ax=ax, label=f"Δ {SPLIT_LABEL[split]}")
    ax.set_title("BSBS с возвратом минус BSBS, среднее по ионам и k")
    fig.tight_layout()
    fig.savefig(folder / f"{split}.png", dpi=130)
    plt.close(fig)
    return wide.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, full = load_points()
    points.to_csv(OUT / "points.csv", index=False)
    for split in ("test", "valid"):
        table, full_split = with_mean(points.drop(columns=[c for c in points if c.startswith("valid" if split == "test" else "test")]), full, split)
        plot_curves(table, full_split, split)
        plot_methods(table, full_split, split)
        plot_heatmaps(table, full_split, split)
        wide = plot_difference(table, split)
        if split == "test":
            by_k = wide.groupby("k")["diff"].agg(mean="mean", median="median", share_better=lambda s: float((s > 0).mean()))
            by_k.to_csv(OUT / "methods_by_k.csv")
    best = (
        points.groupby(["method", "spectrum", "channel"])[["valid", "test"]].mean()
        .sort_values("valid", ascending=False).round(4)
    )
    best.to_csv(OUT / "best_variants.csv")
    print("графики и таблицы:", OUT)


if __name__ == "__main__":
    main()
