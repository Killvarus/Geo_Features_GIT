"""
Анализ сетки нормировок 6P (experiments/6p_norm_grid, run_6p_norm_grid.py).

Всё пишется в experiments/6p_norm_grid/analysis/:
  full_spectrum.csv/.png  — все 5427 каналов, по нормировке спектров и иону;
  heatmap_<valid|test>.png — средний по ионам R² персептрона, нормировка
                             спектров × каналов, для ППОП и RFE и каждого k;
  curves_<ion>.png        — R² персептрона от k: нормировки спектров (каналы
                             z-score) и нормировки каналов (спектры raw);
  variants.csv            — все 48 вариантов, средние по ионам;
  best_by_ion.csv         — лучшие варианты для каждого иона;
  overlap.csv/.png        — пересечение первых 100 каналов между вариантами;
  maps/<метод>_<ион>.png  — первые 100 каналов на карте, 6 × 4 вариантов;
  stability.png           — как часто канал попадает в первые 100 по всем 48
                             вариантам (по иону на панель).
Сравнение — по R² на validation, test для контроля.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.eem_selection import mean_train_matrix, save_selection_grid, top_features

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "experiments" / "6p_norm_grid"
OUT = GRID / "analysis"
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
SPECTRA = ["raw", "max", "area", "snv", "raman", "row_max"]
CHANNELS = ["zscore", "pareto", "robust", "center"]
METHODS = ["ppop", "rfe"]
KS = [10, 30, 100, 300]
SPECTRUM_LABEL = {
    "raw": "без нормировки",
    "max": "на максимум",
    "area": "на площадь",
    "snv": "SNV",
    "raman": "на пик воды",
    "row_max": "по строкам",
}
METHOD_LABEL = {"ppop": "ППОП", "rfe": "RFE"}
TOP = 100


def load() -> pd.DataFrame:
    return pd.read_csv(GRID / "summary.csv")


def full_spectrum(table: pd.DataFrame) -> pd.DataFrame:
    part = table[table["method"] == "all_channels"]
    wide = part.pivot_table(index=["spectrum_norm", "ion"], columns="model", values=["r2_valid", "r2_test"])
    wide.columns = [f"{model}_{split.split('_')[1]}" for split, model in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / "full_spectrum.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    width = 0.13
    for ax, column, title in zip(axes, ["perceptron_test", "ridge_test"], ["Персептрон", "Ridge alpha=700"]):
        for i, spectrum in enumerate(SPECTRA):
            values = [
                wide.loc[(wide["spectrum_norm"] == spectrum) & (wide["ion"] == ion), column].iloc[0] for ion in TARGETS
            ]
            ax.bar(np.arange(len(TARGETS)) + (i - 2.5) * width, values, width, label=SPECTRUM_LABEL[spectrum])
        ax.set_xticks(range(len(TARGETS)), TARGETS)
        ax.set_title(f"{title}, все 5427 каналов, R² на test")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("R²")
    axes[1].legend(fontsize=8, title="Нормировка спектров", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "full_spectrum.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return wide


def variants(table: pd.DataFrame) -> pd.DataFrame:
    part = table[(table["model"] == "perceptron") & (table["method"] != "all_channels")]
    mean = part.groupby(["method", "spectrum_norm", "channel_norm", "k"])[["r2_valid", "r2_test"]].mean().reset_index()
    wide = mean.pivot_table(index=["method", "spectrum_norm", "channel_norm"], columns="k", values=["r2_valid", "r2_test"])
    wide.columns = [f"{split}_k{k}" for split, k in wide.columns]
    wide["r2_valid_mean"] = wide[[f"r2_valid_k{k}" for k in KS]].mean(axis=1)
    wide["r2_test_mean"] = wide[[f"r2_test_k{k}" for k in KS]].mean(axis=1)
    wide = wide.sort_values("r2_valid_mean", ascending=False).reset_index()
    wide.to_csv(OUT / "variants.csv", index=False)
    return wide


def heatmaps(table: pd.DataFrame, split: str) -> None:
    part = table[(table["model"] == "perceptron") & (table["method"] != "all_channels")]
    mean = part.groupby(["method", "spectrum_norm", "channel_norm", "k"])[f"r2_{split}"].mean()
    values = mean.to_numpy()
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    fig, axes = plt.subplots(len(METHODS), len(KS), figsize=(4.0 * len(KS), 3.6 * len(METHODS)))
    for row, method in enumerate(METHODS):
        for col, k in enumerate(KS):
            ax = axes[row, col]
            grid = np.array([[mean[(method, s, c, k)] for c in CHANNELS] for s in SPECTRA])
            low, high = grid.min(), grid.max()
            image = ax.imshow(grid, cmap="viridis", vmin=low, vmax=high, aspect="auto")
            for i, j in itertools.product(range(len(SPECTRA)), range(len(CHANNELS))):
                color = "white" if grid[i, j] < low + 0.6 * (high - low) else "black"
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(len(CHANNELS)), CHANNELS, fontsize=8)
            ax.set_yticks(range(len(SPECTRA)), [SPECTRUM_LABEL[s] for s in SPECTRA], fontsize=8)
            ax.set_title(f"{METHOD_LABEL[method]}, k = {k}", fontsize=10)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Персептрон, средний по 6 ионам R² на {split}. Строки — нормировка спектров, столбцы — каналов "
        f"(шкала у каждой панели своя; общий диапазон {vmin:.2f}–{vmax:.2f})",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"heatmap_{split}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def curves(table: pd.DataFrame, full: pd.DataFrame) -> None:
    perceptron = table[table["model"] == "perceptron"]
    colors = plt.get_cmap("tab10")
    for ion in TARGETS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
        for row, method in enumerate(METHODS):
            ax = axes[row, 0]
            for i, spectrum in enumerate(SPECTRA):
                line = perceptron[
                    (perceptron["method"] == method)
                    & (perceptron["spectrum_norm"] == spectrum)
                    & (perceptron["channel_norm"] == "zscore")
                    & (perceptron["ion"] == ion)
                ].sort_values("k")
                ax.plot(line["k"], line["r2_valid"], "o-", color=colors(i), label=SPECTRUM_LABEL[spectrum])
                level = full.loc[(full["spectrum_norm"] == spectrum) & (full["ion"] == ion), "perceptron_valid"].iloc[0]
                ax.axhline(level, color=colors(i), ls=":", lw=1)
            ax.set_title(f"{METHOD_LABEL[method]}: нормировка спектров (каналы z-score)")
            ax = axes[row, 1]
            for i, channel in enumerate(CHANNELS):
                line = perceptron[
                    (perceptron["method"] == method)
                    & (perceptron["spectrum_norm"] == "raw")
                    & (perceptron["channel_norm"] == channel)
                    & (perceptron["ion"] == ion)
                ].sort_values("k")
                ax.plot(line["k"], line["r2_valid"], "o-", color=colors(i), label=channel)
            level = full.loc[(full["spectrum_norm"] == "raw") & (full["ion"] == ion), "perceptron_valid"].iloc[0]
            ax.axhline(level, color="black", ls=":", lw=1, label="все каналы")
            ax.set_title(f"{METHOD_LABEL[method]}: нормировка каналов (спектры без нормировки)")
        for ax in axes.ravel():
            ax.set_xscale("log")
            ax.set_xticks(KS, [str(k) for k in KS])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("R² персептрона на validation")
        for ax in axes[1]:
            ax.set_xlabel("Число каналов k")
        fig.suptitle(f"{ion}: персептрон на первых k каналах. Пунктир — все 5427 каналов той же нормировки спектров")
        fig.tight_layout()
        fig.savefig(OUT / f"curves_{ion}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def best_by_ion(table: pd.DataFrame) -> pd.DataFrame:
    part = table[(table["model"] == "perceptron") & (table["method"] != "all_channels")]
    mean = part.groupby(["ion", "method", "spectrum_norm", "channel_norm"])[["r2_valid", "r2_test"]].mean().reset_index()
    rows = []
    for ion in TARGETS:
        ranked = mean[mean["ion"] == ion].sort_values("r2_valid", ascending=False).reset_index(drop=True)
        base = ranked[(ranked["method"] == "rfe") & (ranked["spectrum_norm"] == "raw") & (ranked["channel_norm"] == "zscore")]
        for place, row in ranked.head(3).iterrows():
            rows.append({**row.to_dict(), "place": place + 1})
        rows.append({**base.iloc[0].to_dict(), "place": int(base.index[0]) + 1, "note": "прежний вариант"})
        worst = ranked.iloc[-1]
        rows.append({**worst.to_dict(), "place": len(ranked), "note": "худший"})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "best_by_ion.csv", index=False)
    return frame


def rankings() -> dict[tuple[str, str, str, str], list[str]]:
    found = {}
    for path in GRID.glob("*__*/*/*/ranking.csv"):
        combo, method, ion = path.parts[-4], path.parts[-3], path.parts[-2]
        spectrum, channel = combo.split("__")
        found[(method, spectrum, channel, ion)] = top_features(pd.read_csv(path), TOP)
    return found


def overlap(found: dict) -> pd.DataFrame:
    keys = [(m, s, c) for m in METHODS for s in SPECTRA for c in CHANNELS]
    labels = [f"{METHOD_LABEL[m]} {s}/{c}" for m, s, c in keys]
    matrix = np.zeros((len(keys), len(keys)))
    rows = []
    for ion in TARGETS:
        sets = [set(found[(m, s, c, ion)]) for m, s, c in keys]
        part = np.array([[len(a & b) / TOP for b in sets] for a in sets])
        matrix += part / len(TARGETS)
        for (i, a), (j, b) in itertools.combinations(enumerate(keys), 2):
            if a[0] == b[0] and a[1] == b[1]:
                kind = "те же метод и спектры, другие каналы"
            elif a[0] == b[0] and a[2] == b[2]:
                kind = "те же метод и каналы, другие спектры"
            elif a[1] == b[1] and a[2] == b[2]:
                kind = "те же нормировки, ППОП против RFE"
            else:
                continue
            rows.append({"ion": ion, "kind": kind, "a": labels[i], "b": labels[j], "shared": part[i, j]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "overlap.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 11.5))
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for edge in (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44):
        ax.axhline(edge - 0.5, color="white", lw=0.4 if edge != 24 else 1.5)
        ax.axvline(edge - 0.5, color="white", lw=0.4 if edge != 24 else 1.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="Доля общих каналов среди первых 100")
    ax.set_title("Общие каналы среди первых 100 между вариантами, среднее по 6 ионам")
    fig.tight_layout()
    fig.savefig(OUT / "overlap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return frame


def maps(found: dict) -> None:
    spectrum = mean_train_matrix(np.load(CACHE)["X"])
    for method in METHODS:
        for ion in TARGETS:
            panels = [
                (f"{SPECTRUM_LABEL[s]} / {c}", found[(method, s, c, ion)]) for s in SPECTRA for c in CHANNELS
            ]
            save_selection_grid(
                spectrum,
                panels,
                f"{METHOD_LABEL[method]}, {ion}: первые {TOP} каналов. Строки — нормировка спектров, столбцы — каналов",
                OUT / "maps" / f"{method}_{ion}.png",
                ncols=len(CHANNELS),
            )
        print("maps", method, flush=True)


def stability(found: dict) -> None:
    from src.visualization.eem_selection import EM_NM, EXC_NM, channel_coordinates

    spectrum = mean_train_matrix(np.load(CACHE)["X"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    extent = (EM_NM[0] - 0.5, EM_NM[-1] + 0.5, EXC_NM[0] - 2.5, EXC_NM[-1] + 2.5)
    variants = [(m, s, c) for m in METHODS for s in SPECTRA for c in CHANNELS]
    for ax, ion in zip(axes.ravel(), TARGETS):
        counts = np.zeros((len(EXC_NM), len(EM_NM)))
        for method, spectrum_norm, channel in variants:
            for feature in found[(method, spectrum_norm, channel, ion)]:
                excitation, emission = channel_coordinates(feature)
                counts[EXC_NM.index(excitation), EM_NM.index(emission)] += 1
        share = counts / len(variants)
        image = ax.imshow(
            np.ma.masked_equal(share, 0), origin="lower", aspect="auto", cmap="hot_r", vmin=0, vmax=1, extent=extent
        )
        _, vmax = np.percentile(spectrum, [0, 99])
        ax.contour(EM_NM, EXC_NM, spectrum, levels=vmax * np.array([0.1, 0.3, 0.6, 0.9]), colors="gray", linewidths=0.6)
        ax.set_title(f"{ion}: в ≥ половине вариантов — {int((share >= 0.5).sum())} каналов", fontsize=10)
        ax.set_xlabel("Испускание, нм", fontsize=8)
        ax.set_ylabel("Возбуждение, нм", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Доля вариантов, где канал в первых 100")
    fig.suptitle("Устойчивость отбора: как часто канал попадает в первые 100 по всем 48 вариантам нормировок и методов")
    fig.savefig(OUT / "stability.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = load()
    full = full_spectrum(table)
    variants(table)
    for split in ("valid", "test"):
        heatmaps(table, split)
    curves(table, full)
    best_by_ion(table)
    found = rankings()
    overlap(found)
    stability(found)
    maps(found)
    print("analysis", OUT, flush=True)


if __name__ == "__main__":
    main()
