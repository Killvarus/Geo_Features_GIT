"""
Графики сетки нормировок 6P только для BSBS (в коде и папках результатов он
называется ppop; прямой порядок: спектры, потом каналы).

k: 10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000 и полный набор (5427 каналов).
Метрики: R², MSE, MAE.

Персептрон:
  R² — из curve_points.json (test и validation);
  MSE на test — (1 − R²)·дисперсия y на test, это точно то же, что считал персептрон;
  MSE на validation — лучшая ошибка на validation из истории обучения, в исходных единицах;
  MAE на test — только из metrics.xlsx в папках k<k>/ (их нужно скачать с сервера);
  MAE на validation при обучении не сохранялся — этих графиков нет.
Ridge (alpha=700, z-score каналов) считается здесь для всех метрик и кешируется.
Недостающие точки помечены «нет данных».

Всё пишется в experiments/6p_norm_grid/analysis/:
  bsbs_heatmaps/<perceptron|ridge>/<r2|mse|mae>/<test|valid>/<ион>.png
  bsbs_curves/<perceptron|ridge>/<r2|mse|mae>/<test|valid>/<ион>.png — линии: нормировки спектров
  bsbs_curves_by_channel/<perceptron|ridge>/<r2|mse|mae>/<test|valid>/<спектр>/<ион>.png
      — для одной нормировки спектров, линии: нормировки каналов
  bsbs_maps/<ион>/<спектр>__<каналы>/importance.png и top<k>.png
  bsbs_ridge_metrics.csv — кеш Ridge.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN, N_VALID, load_ion
from run_6p_norm_grid import ALPHA, load
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
METRICS = ("r2", "mse", "mae")
METRIC_LABEL = {"r2": "R²", "mse": "MSE", "mae": "MAE"}
LOWER_IS_BETTER = {"r2": False, "mse": True, "mae": True}
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


def target_variances() -> dict[str, dict[str, float]]:
    """Дисперсии y (ddof=0) по выборкам: R² ↔ MSE и пересчёт ошибки validation в исходные единицы."""
    result = {}
    for ion in TARGETS:
        _, y = load_ion(ion)
        values = y.to_numpy(dtype=float)
        result[ion] = {
            "train": float(values[:N_TRAIN].var()),
            "valid": float(values[N_TRAIN:N_TRAIN + N_VALID].var()),
            "test": float(values[N_TRAIN + N_VALID:].var()),
        }
    return result


def _test_mae(run_dir: Path) -> list[float] | None:
    path = run_dir / "metrics.xlsx"
    if not path.exists():
        return None
    raw = pd.read_excel(path, sheet_name="Raw_Results")
    return raw["Test_MAE_Target1"].astype(float).tolist()


def perceptron_points() -> pd.DataFrame:
    """Метрики персептрона: среднее и std по 3 seed для BSBS и для всех каналов."""
    variances = target_variances()
    rows = []
    sources = [
        (spectrum, [channel], GRID / f"{spectrum}__{channel}" / "ppop" / ion, ion)
        for spectrum, channel, ion in itertools.product(SPECTRA, CHANNELS, TARGETS)
    ]
    # Полный набор один и тот же при любой нормировке каналов.
    sources += [
        (spectrum, CHANNELS, GRID / f"{spectrum}__all_channels" / ion, ion)
        for spectrum, ion in itertools.product(SPECTRA, TARGETS)
    ]
    for spectrum, channels, ion_dir, ion in sources:
        path = ion_dir / "curve_points.json"
        if not path.exists():
            continue
        var = variances[ion]
        for point in json.loads(path.read_text(encoding="utf-8")):
            r2_test = np.array([run["r2"] for run in point["runs"]])
            r2_valid = np.array([run["r2_valid"] for run in point["runs"]])
            values = {
                "r2_test": r2_test,
                "r2_valid": r2_valid,
                "mse_test": (1.0 - r2_test) * var["test"],
                "mse_valid": (1.0 - r2_valid) * var["valid"],
            }
            mae = _test_mae(ion_dir / f"k{point['k']}")
            if mae is not None:
                values["mae_test"] = np.array(mae)
            row = {"spectrum": spectrum, "ion": ion, "k": int(point["k"])}
            for name, series in values.items():
                row[name] = float(series.mean())
                row[f"{name}_sd"] = float(series.std(ddof=1))
            for channel in channels:
                rows.append({**row, "channel": channel})
    return pd.DataFrame(rows)


def ranking(spectrum: str, channel: str, ion: str) -> pd.DataFrame:
    return pd.read_csv(GRID / f"{spectrum}__{channel}" / "ppop" / ion / "ranking.csv")


def _ridge_scores(X: pd.DataFrame, y: pd.Series, columns: list[str]) -> dict:
    train = slice(0, N_TRAIN)
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(X.iloc[train][columns], y.iloc[train])
    scores = {}
    for split, rows in (("valid", slice(N_TRAIN, N_TRAIN + N_VALID)), ("test", slice(N_TRAIN + N_VALID, None))):
        truth, prediction = y.iloc[rows], model.predict(X.iloc[rows][columns])
        scores[f"r2_{split}"] = float(r2_score(truth, prediction))
        scores[f"mse_{split}"] = float(mean_squared_error(truth, prediction))
        scores[f"mae_{split}"] = float(mean_absolute_error(truth, prediction))
    return scores


def ridge_points() -> pd.DataFrame:
    """Ridge alpha=700 на первых k каналах рейтинга BSBS, все метрики, с кешем."""
    path = OUT / "bsbs_ridge_metrics.csv"
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
                full = full or _ridge_scores(X, y, list(X.columns))
                scores = full
            else:
                ordered = ranking(spectrum, channel, ion).sort_values("Individual_Rank")["Feature"].astype(str)
                scores = _ridge_scores(X, y, ordered.tolist()[:k])
            rows.append({"spectrum": spectrum, "channel": channel, "ion": ion, "k": k, **scores})
        print("ridge", spectrum, ion, flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(path, index=False)
    return done


def _number(value: float, metric: str) -> str:
    return f"{value:.3f}" if metric == "r2" or abs(value) < 100 else f"{value:.1f}"


def heatmaps(table: pd.DataFrame, model: str, metric: str, split: str) -> None:
    column = f"{metric}_{split}"
    if column not in table.columns:
        print("skip heatmaps", model, metric, "— нет данных", flush=True)
        return
    folder = OUT / "bsbs_heatmaps" / model / metric / split
    folder.mkdir(parents=True, exist_ok=True)
    cmap = "viridis_r" if LOWER_IS_BETTER[metric] else "viridis"
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
                    grid[i, j] = part.loc[(s, c, k), column]
                    if f"{column}_sd" in part.columns:
                        spread[i, j] = part.loc[(s, c, k), f"{column}_sd"]
            ax.set_xticks(range(len(CHANNELS)), CHANNELS, fontsize=8)
            ax.set_yticks(range(len(SPECTRA)), [SPECTRUM_LABEL[s] for s in SPECTRA], fontsize=8)
            ax.set_title(f"k = {k_label(k)}", fontsize=10)
            if np.isnan(grid).all():
                ax.imshow(np.zeros_like(grid), cmap="Greys", vmin=0, vmax=1, aspect="auto")
                ax.text(1.5, 2.5, "нет данных", ha="center", va="center", fontsize=10)
                continue
            low, high = np.nanmin(grid), np.nanmax(grid)
            image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=low, vmax=high, aspect="auto")
            for i, j in itertools.product(range(len(SPECTRA)), range(len(CHANNELS))):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "нет", ha="center", va="center", fontsize=7, color="gray")
                    continue
                good = (grid[i, j] - low) / (high - low + 1e-300)
                if LOWER_IS_BETTER[metric]:
                    good = 1 - good
                color = "white" if good < 0.6 else "black"
                label = _number(grid[i, j], metric)
                if not np.isnan(spread[i, j]):
                    label += f"\n±{_number(spread[i, j], metric)}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        note = "± std по 3 seed" if model == "perceptron" else "без случайности"
        fig.suptitle(
            f"{ion}: BSBS, {MODEL_TITLE[model]} на первых k каналах, {METRIC_LABEL[metric]} на {split} ({note})",
            y=1.0,
        )
        fig.tight_layout()
        fig.savefig(folder / f"{ion}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def curves(table: pd.DataFrame, model: str, metric: str) -> None:
    colors = plt.get_cmap("tab10")
    for ion, split in itertools.product(TARGETS, ("test", "valid")):
        column = f"{metric}_{split}"
        if column not in table.columns:
            continue
        folder = OUT / "bsbs_curves" / model / metric / split
        folder.mkdir(parents=True, exist_ok=True)
        part = table[(table["ion"] == ion) & (table["channel"] == "zscore")].dropna(subset=[column])
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for i, spectrum in enumerate(SPECTRA):
            line = part[part["spectrum"] == spectrum].sort_values("k")
            line = line[line["k"].isin(KS)]
            if line.empty:
                continue
            errors = line[f"{column}_sd"] if f"{column}_sd" in line.columns else None
            ax.errorbar(
                line["k"], line[column], yerr=errors, fmt="o-", capsize=3, color=colors(i), label=SPECTRUM_LABEL[spectrum]
            )
        ax.set_xscale("log")
        ax.set_xticks(KS, [k_label(k) for k in KS], rotation=45, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("Число каналов k (первые k рейтинга BSBS; «все 5427» — без отбора)")
        ax.set_ylabel(f"{METRIC_LABEL[metric]} на {split}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Нормировка спектров", fontsize=8)
        missing = sorted(set(KS) - set(part["k"]))
        note = f"; нет данных для k = {', '.join(map(str, missing))}" if missing else ""
        spread = ", ± std по 3 seed" if model == "perceptron" else ""
        ax.set_title(f"{ion}: BSBS, {MODEL_TITLE[model]}{spread}{note}", fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / f"{ion}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def channel_curves(table: pd.DataFrame, model: str, metric: str) -> None:
    """Для каждой нормировки спектров: линии — нормировки каналов внутри отбора."""
    colors = plt.get_cmap("tab10")
    for spectrum, ion, split in itertools.product(SPECTRA, TARGETS, ("test", "valid")):
        column = f"{metric}_{split}"
        if column not in table.columns:
            continue
        part = table[(table["ion"] == ion) & (table["spectrum"] == spectrum)].dropna(subset=[column])
        if part.empty:
            continue
        folder = OUT / "bsbs_curves_by_channel" / model / metric / split / spectrum
        folder.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for i, channel in enumerate(CHANNELS):
            line = part[part["channel"] == channel].sort_values("k")
            line = line[line["k"].isin(KS)]
            if line.empty:
                continue
            errors = line[f"{column}_sd"] if f"{column}_sd" in line.columns else None
            ax.errorbar(line["k"], line[column], yerr=errors, fmt="o-", capsize=3, color=colors(i), label=channel)
        ax.set_xscale("log")
        ax.set_xticks(KS, [k_label(k) for k in KS], rotation=45, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("Число каналов k (первые k рейтинга BSBS; «все 5427» — без отбора)")
        ax.set_ylabel(f"{METRIC_LABEL[metric]} на {split}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Нормировка каналов", fontsize=8)
        missing = sorted(set(KS) - set(part["k"]))
        note = f"; нет данных для k = {', '.join(map(str, missing))}" if missing else ""
        spread = ", ± std по 3 seed" if model == "perceptron" else ""
        ax.set_title(f"{ion}: BSBS, {MODEL_TITLE[model]}{spread}, спектры {SPECTRUM_LABEL[spectrum]}{note}", fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / f"{ion}.png", dpi=120, bbox_inches="tight")
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
    for (model, table), metric in itertools.product(tables.items(), METRICS):
        for split in ("test", "valid"):
            heatmaps(table, model, metric, split)
        curves(table, model, metric)
        channel_curves(table, model, metric)
        print("plots", model, metric, flush=True)
    if not args.skip_maps:
        maps()
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
