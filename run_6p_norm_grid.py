"""
Сетка нормировок для отбора каналов 6P: 6 нормировок по спектрам × 4 по каналам
× (ППОП, RFE) × 6 ионов. Всё пишется в experiments/6p_norm_grid/.

Отбор как раньше: Ridge alpha=700, пакет 10% от оставшихся, ППОП — 5-fold на train.
Нормировка по спектрам меняет сами данные (src/preprocessing/spectrum_norm.py),
нормировка по каналам — только масштаб внутри отбора (scale_channels).

Проверка одинаковая для всех вариантов: данные с той же нормировкой по спектрам,
z-score по каналам, первые k = 10, 30, 100, 300 каналов рейтинга:
  Ridge alpha=700 и МНК (linear_check.json, R² на validation и test);
  персептрон, как раньше (curve_points.json).
Отдельно для каждой нормировки по спектрам — все 5427 каналов без отбора.

Структура:
  6p_norm_grid/<спектр>__<каналы>/<ppop|rfe>/<ион>/ranking.csv, history.csv,
      selection.json, linear_check.json, curve_points.json, k<k>/
  6p_norm_grid/<спектр>__all_channels/<ион>/linear_check.json, curve_points.json
  6p_norm_grid/summary.csv — после --summary

Запуск:
  python run_6p_norm_grid.py --list-jobs            # все задания, по строке
  python run_6p_norm_grid.py --job rank raw zscore rfe Cu
  python run_6p_norm_grid.py --job full snv Cu
  python run_6p_norm_grid.py --summary
Задание можно перезапускать: готовый рейтинг и посчитанные точки пропускаются.
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN, N_VALID, TARGETS, load_ion, train_curve
from src.models.linear_selection import SCALINGS, LinearPPOP, LinearRFE
from src.preprocessing.spectrum_norm import SPECTRUM_NORMS, normalize_spectra

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "6p_norm_grid"
METHODS = ("ppop", "rfe")
ALPHA = 700.0
STEP = 0.1
CV = 5
KS = [10, 30, 100, 300]


def jobs() -> list[str]:
    """Сначала долгие задания на всех каналах, потом отбор — так очередь кончается ровнее."""
    lines = [f"full {spectrum} {ion}" for spectrum in SPECTRUM_NORMS for ion in TARGETS]
    for spectrum in SPECTRUM_NORMS:
        for channel in SCALINGS:
            for method in METHODS:
                for ion in TARGETS:
                    lines.append(f"rank {spectrum} {channel} {method} {ion}")
    return lines


def load(spectrum: str, ion: str) -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_ion(ion)
    values = normalize_spectra(X.to_numpy(dtype=float), spectrum, N_TRAIN)
    return pd.DataFrame(values, columns=X.columns), y


def linear_check(X: pd.DataFrame, y: pd.Series, columns: list[str]) -> dict:
    """Ridge alpha=700 и МНК на z-score каналов: R² на validation и test."""
    train = slice(0, N_TRAIN)
    valid = slice(N_TRAIN, N_TRAIN + N_VALID)
    test = slice(N_TRAIN + N_VALID, None)
    result = {}
    for name, model in (("ridge", Ridge(alpha=ALPHA)), ("ols", LinearRegression())):
        pipeline = make_pipeline(StandardScaler(), model).fit(X.iloc[train][columns], y.iloc[train])
        result[name] = {
            "valid": float(r2_score(y.iloc[valid], pipeline.predict(X.iloc[valid][columns]))),
            "test": float(r2_score(y.iloc[test], pipeline.predict(X.iloc[test][columns]))),
        }
    return result


def run_rank(spectrum: str, channel: str, method: str, ion: str, ks: list[int], perceptron: bool) -> None:
    directory = OUT_DIR / f"{spectrum}__{channel}" / method
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(spectrum, ion)
    path = ion_dir / "ranking.csv"
    if path.exists():
        ranking = pd.read_csv(path)
        print(ion, spectrum, channel, method, "ranking exists", flush=True)
    else:
        X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
        if method == "rfe":
            selector = LinearRFE(alpha=ALPHA, step=STEP, scaling=channel).fit(X_train, y_train)
        else:
            selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling=channel).fit(X_train, y_train)
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        settings = {
            "spectrum_norm": spectrum,
            "channel_norm": channel,
            "method": method,
            "alpha": ALPHA,
            "step": STEP,
            "cv": CV if method == "ppop" else None,
            "seconds": selector.elapsed_seconds_,
        }
        (ion_dir / "selection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        selector.save_ranking(path)
        ranking = selector.ranking_frame()
        print(ion, spectrum, channel, method, "ranking done, seconds", round(selector.elapsed_seconds_, 1), flush=True)
    ordered = ranking.sort_values("Individual_Rank")["Feature"].astype(str).tolist()
    checks = {str(k): linear_check(X, y, ordered[:k]) for k in ks}
    (ion_dir / "linear_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if perceptron:
        train_curve(directory, ion, X, y, ranking, ks)


def run_full(spectrum: str, ion: str, perceptron: bool) -> None:
    directory = OUT_DIR / f"{spectrum}__all_channels"
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(spectrum, ion)
    columns = list(X.columns)
    checks = {str(len(columns)): linear_check(X, y, columns)}
    (ion_dir / "linear_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if perceptron:
        order = pd.DataFrame({"Feature": columns, "Individual_Rank": np.arange(1, len(columns) + 1)})
        train_curve(directory, ion, X, y, order, [len(columns)])


def summary() -> Path:
    rows = []
    for check in OUT_DIR.glob("*/**/linear_check.json"):
        relative = check.parent.relative_to(OUT_DIR).parts
        if len(relative) == 3:
            combo, method, ion = relative
        else:
            combo, ion = relative
            method = "all_channels"
        spectrum, channel = combo.split("__")
        base = {"spectrum_norm": spectrum, "channel_norm": channel, "method": method, "ion": ion}
        for k, models in json.loads(check.read_text(encoding="utf-8")).items():
            for model, scores in models.items():
                rows.append({**base, "k": int(k), "model": model, "r2_valid": scores["valid"], "r2_test": scores["test"]})
        curve = check.parent / "curve_points.json"
        if curve.exists():
            for point in json.loads(curve.read_text(encoding="utf-8")):
                rows.append(
                    {
                        **base,
                        "k": point["k"],
                        "model": "perceptron",
                        "r2_valid": point.get("mean_valid"),
                        "r2_test": point["mean"],
                        "runs": len(point.get("runs", [])),
                    }
                )
    table = pd.DataFrame(rows).sort_values(["spectrum_norm", "channel_norm", "method", "ion", "model", "k"])
    out = OUT_DIR / "summary.csv"
    table.to_csv(out, index=False)
    print("summary", out, "rows", len(table), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--job", nargs="+", help="rank <спектр> <каналы> <ppop|rfe> <ион> или full <спектр> <ион>")
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--no-perceptron", action="store_true", help="только рейтинг и линейная проверка")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.list_jobs:
        print("\n".join(jobs()))
        return
    if args.summary:
        summary()
        return
    if not args.job:
        parser.error("нужен --list-jobs, --job или --summary")
    kind, *rest = args.job
    perceptron = not args.no_perceptron
    if kind == "rank" and len(rest) == 4:
        spectrum, channel, method, ion = rest
        if spectrum not in SPECTRUM_NORMS or channel not in SCALINGS or method not in METHODS or ion not in TARGETS:
            parser.error(f"неизвестное задание: {' '.join(args.job)}")
        run_rank(spectrum, channel, method, ion, args.ks, perceptron)
    elif kind == "full" and len(rest) == 2:
        spectrum, ion = rest
        if spectrum not in SPECTRUM_NORMS or ion not in TARGETS:
            parser.error(f"неизвестное задание: {' '.join(args.job)}")
        run_full(spectrum, ion, perceptron)
    else:
        parser.error(f"неизвестное задание: {' '.join(args.job)}")
    print("job done:", " ".join(args.job), flush=True)


if __name__ == "__main__":
    main()
