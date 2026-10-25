"""
Сетка нормировок для отбора каналов 6P. Всё пишется в experiments/6p_norm_grid/.

Отбор как раньше: Ridge alpha=700, пакет 10% от оставшихся, ППОП — 5-fold на train.

Два набора заданий:

  base    — сначала нормировка спектров (6 вариантов, src/preprocessing/spectrum_norm.py),
            потом нормировка каналов внутри отбора (4 варианта, scale_channels).
            6 × 4 × (ППОП, RFE) × 6 ионов = 288 заданий + 6 × 6 = 36 на всех каналах.

  chfirst — обратный порядок: сначала каналы делятся на свой масштаб по train
            (scale_only: zscore — на std, pareto — на корень из std, robust — на
            межквартильный размах, без вычитания среднего: после центрирования сумма
            и максимум спектра теряют смысл), потом нормировка спектров, потом отбор
            без повторного масштабирования (только центрирование, которое и так
            делает свободный член Ridge).
            Спектры: max, area, snv, raman, row_max (raw здесь совпал бы с base).
            Комбинации zscore + area и robust + area не считаются: после деления
            на std или межквартильный размах у части спектров сумма ≤ 0
            (шумовые каналы с малым разбросом раздуваются), нормировка на
            площадь не определена.
            13 × (ППОП, RFE) × 6 ионов = 156 заданий + 13 × 6 = 78 на всех каналах.

«Без нормировки каналов» отдельно не считается: у Ridge со свободным членом
вычитание среднего ничего не меняет, поэтому рейтинги совпали бы с base-вариантом center.

Проверка одинаковая для всех: данные с той же предобработкой, z-score каналов,
первые k = 10, 30, 100, 300 каналов рейтинга: Ridge alpha=700 и МНК
(linear_check.json, R² на validation и test) и персептрон, как раньше
(curve_points.json). Отдельно — все 5427 каналов без отбора.

Структура:
  base:    <спектр>__<каналы>/<ppop|rfe>/<ион>/...
           <спектр>__all_channels/<ион>/...
  chfirst: <спектр>__<каналы>__chfirst/<ppop|rfe|all_channels>/<ион>/...
  в каждой папке иона: ranking.csv, history.csv, selection.json,
  linear_check.json, curve_points.json, k<k>/
  summary.csv — после --summary (по всем найденным результатам, столбец order)

Запуск:
  python run_6p_norm_grid.py --list-jobs [--set base|chfirst|all]
  python run_6p_norm_grid.py --job rank raw zscore rfe Cu
  python run_6p_norm_grid.py --job full snv Cu
  python run_6p_norm_grid.py --job rankcf pareto max rfe Cu
  python run_6p_norm_grid.py --job fullcf pareto max Cu
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
from src.models.linear_selection import SCALINGS, LinearPPOP, LinearRFE, scale_only
from src.preprocessing.spectrum_norm import SPECTRUM_NORMS, normalize_spectra

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "6p_norm_grid"
METHODS = ("ppop", "rfe")
ALPHA = 700.0
STEP = 0.1
CV = 5
KS = [10, 30, 100, 300]
CF_CHANNELS = ("zscore", "pareto", "robust")
CF_SPECTRA = ("max", "area", "snv", "raman", "row_max")
CF_SKIP = {("zscore", "area"), ("robust", "area")}
CF_SUFFIX = "chfirst"


def cf_combos() -> list[tuple[str, str]]:
    return [(c, s) for c in CF_CHANNELS for s in CF_SPECTRA if (c, s) not in CF_SKIP]


def jobs(which: str = "base") -> list[str]:
    """Сначала долгие задания на всех каналах, потом отбор — так очередь кончается ровнее."""
    lines = []
    if which in ("base", "all"):
        lines += [f"full {spectrum} {ion}" for spectrum in SPECTRUM_NORMS for ion in TARGETS]
        for spectrum in SPECTRUM_NORMS:
            for channel in SCALINGS:
                for method in METHODS:
                    for ion in TARGETS:
                        lines.append(f"rank {spectrum} {channel} {method} {ion}")
    if which in ("chfirst", "all"):
        lines += [f"fullcf {c} {s} {ion}" for c, s in cf_combos() for ion in TARGETS]
        lines += [f"rankcf {c} {s} {m} {ion}" for c, s in cf_combos() for m in METHODS for ion in TARGETS]
    return lines


def load(spectrum: str, ion: str, channel_first: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_ion(ion)
    values = X.to_numpy(dtype=float)
    if channel_first is not None:
        # Масштаб каналов — по train, применяется ко всем строкам.
        values = scale_only(values[:N_TRAIN], values, method=channel_first)[1]
    values = normalize_spectra(values, spectrum, N_TRAIN)
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


def combo_dir(spectrum: str, channel: str, channel_first: bool) -> Path:
    name = f"{spectrum}__{channel}" + (f"__{CF_SUFFIX}" if channel_first else "")
    return OUT_DIR / name


def run_rank(
    spectrum: str, channel: str, method: str, ion: str, ks: list[int], perceptron: bool, channel_first: bool = False
) -> None:
    directory = combo_dir(spectrum, channel, channel_first) / method
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(spectrum, ion, channel if channel_first else None)
    # При обратном порядке каналы уже отмасштабированы, внутри отбора — только центрирование.
    scaling = "center" if channel_first else channel
    path = ion_dir / "ranking.csv"
    if path.exists():
        ranking = pd.read_csv(path)
        print(ion, spectrum, channel, method, "ranking exists", flush=True)
    else:
        X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
        if method == "rfe":
            selector = LinearRFE(alpha=ALPHA, step=STEP, scaling=scaling).fit(X_train, y_train)
        else:
            selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling=scaling).fit(X_train, y_train)
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        settings = {
            "order": "channels_first" if channel_first else "spectrum_first",
            "spectrum_norm": spectrum,
            "channel_norm": channel,
            "selection_scaling": scaling,
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


def run_full(spectrum: str, ion: str, perceptron: bool, channel: str | None = None) -> None:
    if channel is None:
        directory = OUT_DIR / f"{spectrum}__all_channels"
    else:
        directory = combo_dir(spectrum, channel, True) / "all_channels"
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(spectrum, ion, channel)
    columns = list(X.columns)
    checks = {str(len(columns)): linear_check(X, y, columns)}
    (ion_dir / "linear_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if perceptron:
        order = pd.DataFrame({"Feature": columns, "Individual_Rank": np.arange(1, len(columns) + 1)})
        train_curve(directory, ion, X, y, order, [len(columns)])


def describe(relative: tuple[str, ...]) -> dict:
    """Вариант по пути папки иона относительно OUT_DIR."""
    if len(relative) == 2:  # <спектр>__all_channels/<ион>
        combo, ion = relative
        spectrum, _ = combo.split("__")
        return {"order": "spectrum_first", "spectrum_norm": spectrum, "channel_norm": "all_channels",
                "method": "all_channels", "ion": ion}
    combo, method, ion = relative
    parts = combo.split("__")
    order = "channels_first" if len(parts) == 3 and parts[2] == CF_SUFFIX else "spectrum_first"
    return {"order": order, "spectrum_norm": parts[0], "channel_norm": parts[1], "method": method, "ion": ion}


def summary() -> Path:
    rows = []
    for check in OUT_DIR.glob("**/linear_check.json"):
        base = describe(check.parent.relative_to(OUT_DIR).parts)
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
    keys = ["order", "spectrum_norm", "channel_norm", "method", "ion", "model", "k"]
    table = pd.DataFrame(rows).sort_values(keys)
    out = OUT_DIR / "summary.csv"
    table.to_csv(out, index=False)
    print("summary", out, "rows", len(table), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--set", choices=["base", "chfirst", "all"], default="base", help="какие задания выводить")
    parser.add_argument(
        "--job",
        nargs="+",
        help="rank <спектр> <каналы> <метод> <ион> | full <спектр> <ион> | "
        "rankcf <каналы> <спектр> <метод> <ион> | fullcf <каналы> <спектр> <ион>",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--no-perceptron", action="store_true", help="только рейтинг и линейная проверка")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.list_jobs:
        print("\n".join(jobs(args.set)))
        return
    if args.summary:
        summary()
        return
    if not args.job:
        parser.error("нужен --list-jobs, --job или --summary")
    kind, *rest = args.job
    perceptron = not args.no_perceptron
    bad = f"неизвестное задание: {' '.join(args.job)}"
    if kind == "rank" and len(rest) == 4:
        spectrum, channel, method, ion = rest
        if spectrum not in SPECTRUM_NORMS or channel not in SCALINGS or method not in METHODS or ion not in TARGETS:
            parser.error(bad)
        run_rank(spectrum, channel, method, ion, args.ks, perceptron)
    elif kind == "full" and len(rest) == 2:
        spectrum, ion = rest
        if spectrum not in SPECTRUM_NORMS or ion not in TARGETS:
            parser.error(bad)
        run_full(spectrum, ion, perceptron)
    elif kind == "rankcf" and len(rest) == 4:
        channel, spectrum, method, ion = rest
        if (channel, spectrum) not in cf_combos() or method not in METHODS or ion not in TARGETS:
            parser.error(bad)
        run_rank(spectrum, channel, method, ion, args.ks, perceptron, channel_first=True)
    elif kind == "fullcf" and len(rest) == 3:
        channel, spectrum, ion = rest
        if (channel, spectrum) not in cf_combos() or ion not in TARGETS:
            parser.error(bad)
        run_full(spectrum, ion, perceptron, channel)
    else:
        parser.error(bad)
    print("job done:", " ".join(args.job), flush=True)


if __name__ == "__main__":
    main()
