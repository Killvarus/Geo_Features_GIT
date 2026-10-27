"""
Графики сетки нормировок 6P только для BSBS (в коде и папках результатов он
называется ppop; прямой порядок: спектры, потом каналы).

k: 10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000 и полный набор (5427 каналов).
Персептрон берётся из curve_points.json — есть не для всех k: недостающие точки
помечены «нет данных» и появятся после дорасчёта на сервере. Ridge (alpha=700,
z-score каналов) считается здесь же для всех k и кешируется.

Всё пишется в experiments/6p_norm_grid/analysis/:
  bsbs_heatmaps/<perceptron|ridge>/<ион>_test.png
      R² на test: строки — нормировка спектров, столбцы — каналов, панель — k.
  bsbs_curves/<perceptron|ridge>/<ион>_<test|valid>.png
      R² от k, линии — нормировки спектров (внутри отбора каналы z-score).
  bsbs_maps/<ион>/<спектр>__<каналы>/importance.png — все каналы, цвет — место в рейтинге;
                                  top<k>.png       — первые k каналов (k до 1000).
  bsbs_ridge_topk.csv — кеш Ridge по всем k.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_norm_grid import linear_check, load
from src.visualization.eem_selection import (
    EM_NM,
    EXC_NM,
    channel_coordinates,
    color_limits,
    mean_train_matrix,
    save_selection_map,
    top_features,
)

ROOT = Path(__file__).resolve().parent
GRID = ROOT / "experiments" / "6p_norm_grid"
OUT = GRID / "analysis"
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
SPECTRA = ["raw", "max", "area", "snv", "raman", "row_max"]
CHANNELS = ["zscore", "pareto", "robust", "center"]
FULL = 5427
KS = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, FULL]
MAP_KS = [10, 50, 100, 250, 500, 1000]
SPECTRUM_LABEL = {
    "raw": "без нормировки",
    "max": "на максимум",
    "area": "на площадь",
    "snv": "SNV",
    "raman": "на пик воды",
    "row_max": "по строкам",
}
MODEL_TITLE = {"perceptron": "персептрон", "ridge": "Ridge alpha=700"}


def k_label(k: int) -> str:
    return "все 5427" if k == FULL else str(k)


def perceptron_points() -> pd.DataFrame:
    """R² персептрона и разброс по seed для BSBS и для всех каналов."""
    rows = []
    for spectrum, channel, ion in itertools.product(SPECTRA, CHANNELS, TARGETS):
        path = GRID / f"{spectrum}__{channel}" / "ppop" / ion / "curve_points.json"
        if path.exists():
            for point in json.loads(path.read_text(encoding="utf-8")):
                rows.append(_point_row(spectrum, channel, ion, point))
    for spectrum, ion in itertools.product(SPECTRA, TARGETS):
        path = GRID / f"{spectrum}__all_channels" / ion / "curve_points.json"
        if path.exists():
            for point in json.loads(path.read_text(encoding="utf-8")):
                # Полный набор один и тот же при любой нормировке каналов.
                for channel in CHANNELS:
                    rows.append(_point_row(spectrum, channel, ion, point))
    return pd.DataFrame(rows)


def _point_row(spectrum: str, channel: str, ion: str, point: dict) -> dict:
    tests = [run["r2"] for run in point["runs"]]
    valids = [run["r2_valid"] for run in point["runs"]]
    return {
        "spectrum": spectrum,
        "channel": channel,
        "ion": ion,
        "k": int(point["k"]),
        "test": float(np.mean(tests)),
        "test_sd": float(np.std(tests, ddof=1)),
        "valid": float(np.mean(valids)),
        "valid_sd": float(np.std(valids, ddof=1)),
    }


def ranking(spectrum: str, channel: str, ion: str) -> pd.DataFrame:
    return pd.read_csv(GRID / f"{spectrum}__{channel}" / "ppop" / ion / "ranking.csv")


def ridge_points() -> pd.DataFrame:
    """Ridge alpha=700 на первых k каналах рейтинга BSBS (как linear_check), с кешем."""
    path = OUT / "bsbs_ridge_topk.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["spectrum", "channel", "ion", "k"])
    have = set(zip(done["spectrum"], done["channel"], done["ion"], done["k"].astype(int)))
    rows = []
    for spectrum, ion in itertools.product(SPECTRA, TARGETS):
        need = [(c, k) for c in CHANNELS for k in KS if (spectrum, c, ion, k) not in have]
        if not need:
            continue
        X, y = load(spectrum, ion)
        full = None
        for channel, k in need:
            if k == FULL:
                full = full or linear_check(X, y, list(X.columns))["ridge"]
                scores = full
            else:
                ordered = ranking(spectrum, channel, ion).sort_values("Individual_Rank")["Feature"].astype(str)
                scores = linear_check(X, y, ordered.tolist()[:k])["ridge"]
            rows.append({"spectrum": spectrum, "channel": channel, "ion": ion, "k": k,
                         "test": scores["test"], "valid": scores["valid"]})
        print("ridge", spectrum, ion, flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(path, index=False)
    return done


def heatmaps(table: pd.DataFrame, model: str) -> None:
    folder = OUT / "bsbs_heatmaps" / model
    folder.mkdir(parents=True, exist_ok=True)
    for ion in TARGETS:
        part = table[table["ion"] == ion].set_index(["spectrum", "channel", "k"])
        nrows = -(-len(KS) // 4)
        fig, axes = plt.subplots(nrows, 4, figsize=(17, 4.3 * nrows))
        for ax in axes.ravel()[len(KS):]:
            ax.axis("off")
        for ax, k in zip(axes.ravel(), KS):
            grid = np.full((len(SPECTRA), len(CHANNELS)), np.nan)
            spread = np.full_like(grid, np.nan)
            for (i, s), (j, c) in itertools.product(enumerate(SPECTRA), enumerate(CHANNELS)):
                if (s, c, k) in part.index:
                    grid[i, j] = part.loc[(s, c, k), "test"]
                    if "test_sd" in part.columns:
                        spread[i, j] = part.loc[(s, c, k), "test_sd"]
            ax.set_xticks(range(len(CHANNELS)), CHANNELS, fontsize=8)
            ax.set_yticks(range(len(SPECTRA)), [SPECTRUM_LABEL[s] for s in SPECTRA], fontsize=8)
            ax.set_title(f"k = {k_label(k)}", fontsize=10)
            if np.isnan(grid).all():
                ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=1, aspect="auto")
                ax.text(1.5, 2.5, "нет данных\n(досчитать на сервере)", ha="center", va="center", fontsize=10)
                continue
            low, high = np.nanmin(grid), np.nanmax(grid)
            image = ax.imshow(np.ma.masked_invalid(grid), cmap="viridis", vmin=low, vmax=high, aspect="auto")
            for i, j in itertools.product(range(len(SPECTRA)), range(len(CHANNELS))):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "нет", ha="center", va="center", fontsize=7, color="gray")
                    continue
                color = "white" if grid[i, j] < low + 0.6 * (high - low) else "black"
                label = f"{grid[i, j]:.3f}"
                if not np.isnan(spread[i, j]):
                    label += f"\n±{spread[i, j]:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        note = "± std по 3 seed" if model == "perceptron" else "без случайности"
        fig.suptitle(
            f"{ion}: BSBS, {MODEL_TITLE[model]} на первых k каналах, R² на test ({note})",
            y=1.0,
        )
        fig.tight_layout()
        fig.savefig(folder / f"{ion}_test.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def curves(table: pd.DataFrame, model: str) -> None:
    folder = OUT / "bsbs_curves" / model
    folder.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")
    for ion, split in itertools.product(TARGETS, ("test", "valid")):
        part = table[(table["ion"] == ion) & (table["channel"] == "zscore")]
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for i, spectrum in enumerate(SPECTRA):
            line = part[part["spectrum"] == spectrum].sort_values("k")
            line = line[line["k"].isin(KS)]
            if line.empty:
                continue
            errors = line[f"{split}_sd"] if f"{split}_sd" in line.columns else None
            ax.errorbar(
                line["k"], line[split], yerr=errors, fmt="o-", capsize=3, color=colors(i), label=SPECTRUM_LABEL[spectrum]
            )
        ax.set_xscale("log")
        ax.set_xticks(KS, [k_label(k) for k in KS])
        ax.set_xlabel("Число каналов k (первые k рейтинга BSBS; «все 5427» — без отбора)")
        ax.set_ylabel(f"R² на {split}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Нормировка спектров", fontsize=8)
        missing = sorted(set(KS) - set(part["k"]))
        note = f"; нет данных для k = {', '.join(map(str, missing))}" if missing else ""
        spread = ", ± std по 3 seed" if model == "perceptron" else ""
        ax.set_title(f"{ion}: BSBS, {MODEL_TITLE[model]}{spread}{note}", fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / f"{ion}_{split}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def importance_map(spectrum_mean: np.ndarray, frame: pd.DataFrame, title: str, path: Path) -> None:
    """Все каналы, цвет — место в рейтинге (1 — самый важный), логарифмическая шкала."""
    places = np.zeros((len(EXC_NM), len(EM_NM)))
    for feature, place in zip(frame["Feature"].astype(str), frame["Individual_Rank"]):
        excitation, emission = channel_coordinates(feature)
        places[EXC_NM.index(excitation), EM_NM.index(emission)] = place
    extent = (EM_NM[0] - 0.5, EM_NM[-1] + 0.5, EXC_NM[0] - 2.5, EXC_NM[-1] + 2.5)
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    image = ax.imshow(
        places, origin="lower", aspect="auto", cmap="magma_r", norm=LogNorm(vmin=1, vmax=places.max()), extent=extent
    )
    _, vmax = color_limits(spectrum_mean)
    ax.contour(EM_NM, EXC_NM, spectrum_mean, levels=vmax * np.array([0.1, 0.3, 0.6, 0.9]), colors="cyan", linewidths=0.6)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([1, 10, 100, 1000, int(places.max())])
    colorbar.set_ticklabels(["1", "10", "100", "1000", str(int(places.max()))])
    colorbar.set_label("Место в рейтинге BSBS (1 — лучший, лог. шкала)")
    ax.set_xlabel("Длина волны испускания, нм")
    ax.set_ylabel("Длина волны возбуждения, нм")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def maps() -> None:
    spectrum_mean = mean_train_matrix(np.load(CACHE)["X"])
    for ion in TARGETS:
        for spectrum, channel in itertools.product(SPECTRA, CHANNELS):
            folder = OUT / "bsbs_maps" / ion / f"{spectrum}__{channel}"
            frame = ranking(spectrum, channel, ion)
            name = f"BSBS, {ion}: нормировка спектров {SPECTRUM_LABEL[spectrum]}"
            importance_map(spectrum_mean, frame, name, folder / "importance.png")
            for k in MAP_KS:
                save_selection_map(
                    spectrum_mean, top_features(frame, k), f"{name}\nпервые {k} каналов", folder / f"top{k}.png", legend=False
                )
        print("maps", ion, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"perceptron": perceptron_points(), "ridge": ridge_points()}
    for model, table in tables.items():
        heatmaps(table, model)
        curves(table, model)
        print("plots", model, flush=True)
    if not args.skip_maps:
        maps()
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
