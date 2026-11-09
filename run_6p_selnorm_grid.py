"""
Сетка нормировок, которые действуют только на отбор каналов 6P. Всё пишется в
experiments/6p_selnorm_grid/.

Отличие от run_6p_norm_grid.py: нормировка спектров (5 вариантов,
src/preprocessing/spectrum_norm.py) и нормировка каналов (4 варианта, scale_channels)
применяются только к данным, на которых идёт отбор. Проверка — Ridge, МНК и
персептрон — всегда на исходных спектрах с z-score каналов, как во всех прежних
расчётах. Так разница между вариантами идёт только от того, какие каналы выбраны.

Методы отбора (Ridge alpha=700, пакет 10% от оставшихся, 5-fold на train):
  bsbs  — LinearPPOP, обычный BSBS, результат — рейтинг;
  fbsbs — LinearFloatingBSBS, BSBS с возвратом выброшенных признаков (до
          --max-swaps обменов на итерацию), результат — отдельный набор для каждого k.

Задания: 5 спектров × 4 нормировки каналов × 2 метода × 6 ионов = 240
плюс 6 заданий «все каналы» (исходные спектры, без отбора).

Структура:
  <спектр>__<каналы>/<bsbs|fbsbs>/<ион>/
      ranking.csv, history.csv, selection.json, linear_check.json, curve_points.json, k<k>/
      fbsbs ещё: sets.json (набор для каждого k), swaps.csv (принятые обмены)
  all_channels/<ион>/linear_check.json, curve_points.json
  summary.csv — после --summary

Запуск:
  python run_6p_selnorm_grid.py --list-jobs
  python run_6p_selnorm_grid.py --job rank max zscore fbsbs Cu
  python run_6p_selnorm_grid.py --job full Cu
  python run_6p_selnorm_grid.py --summary
Задание можно перезапускать: готовый отбор и посчитанные точки пропускаются.
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

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN, TARGETS, load_ion, train_curve
from run_6p_norm_grid import linear_check
from src.models.linear_selection import SCALINGS, LinearFloatingBSBS, LinearPPOP
from src.preprocessing.spectrum_norm import SPECTRUM_NORMS, normalize_spectra

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "6p_selnorm_grid"
METHODS = ("bsbs", "fbsbs")
ALPHA = 700.0
STEP = 0.1
CV = 5
MAX_SWAPS = 5
KS = [10, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000]


def jobs() -> list[str]:
    """Сначала долгие задания (все каналы, потом fbsbs), в конце bsbs — так очередь кончается ровнее."""
    lines = [f"full {ion}" for ion in TARGETS]
    for method in ("fbsbs", "bsbs"):
        for spectrum in SPECTRUM_NORMS:
            for channel in SCALINGS:
                for ion in TARGETS:
                    lines.append(f"rank {spectrum} {channel} {method} {ion}")
    return lines


def selection_data(X: pd.DataFrame, spectrum: str) -> pd.DataFrame:
    """Train для отбора: спектры с выбранной нормировкой (каналы нормирует сам отборщик)."""
    values = normalize_spectra(X.to_numpy(dtype=float), spectrum, N_TRAIN)
    return pd.DataFrame(values[:N_TRAIN], columns=X.columns)


def ordered_for(method: str, ranking: pd.DataFrame, sets: dict[int, list[str]], k: int) -> list[str]:
    """Порядок признаков, первые k из которых — отобранный набор размера k."""
    ordered = ranking.sort_values("Individual_Rank")["Feature"].astype(str).tolist()
    if method == "bsbs":
        return ordered
    chosen = [str(f) for f in sets[k]]
    taken = set(chosen)
    return chosen + [f for f in ordered if f not in taken]


def run_rank(spectrum: str, channel: str, method: str, ion: str, ks: list[int], perceptron: bool, max_swaps: int) -> None:
    directory = OUT_DIR / f"{spectrum}__{channel}" / method
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_ion(ion)
    path = ion_dir / "ranking.csv"
    sets_path = ion_dir / "sets.json"
    if path.exists() and (method == "bsbs" or sets_path.exists()):
        ranking = pd.read_csv(path)
        sets = {int(k): v for k, v in json.loads(sets_path.read_text(encoding="utf-8")).items()} if method == "fbsbs" else {}
        print(ion, spectrum, channel, method, "selection exists", flush=True)
    else:
        X_train, y_train = selection_data(X, spectrum), y.iloc[:N_TRAIN]
        if method == "bsbs":
            selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling=channel).fit(X_train, y_train)
            sets = {}
        else:
            selector = LinearFloatingBSBS(
                alpha=ALPHA, step=STEP, cv=CV, scaling=channel, max_swaps=max_swaps, ks=ks
            ).fit(X_train, y_train)
            sets = selector.sets_
            sets_path.write_text(json.dumps({str(k): v for k, v in sets.items()}, indent=1), encoding="utf-8")
            selector.swaps_.drop(columns=["features"]).to_csv(ion_dir / "swaps.csv", index=False)
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        settings = {
            "spectrum_norm_for_selection": spectrum,
            "channel_norm_for_selection": channel,
            "evaluation_data": "raw spectra, z-score channels",
            "method": method,
            "alpha": ALPHA,
            "step": STEP,
            "cv": CV,
            "max_swaps": max_swaps if method == "fbsbs" else None,
            "swaps": int(len(selector.swaps_)) if method == "fbsbs" else None,
            "seconds": selector.elapsed_seconds_,
        }
        (ion_dir / "selection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        selector.save_ranking(path)
        ranking = selector.ranking_frame()
        print(ion, spectrum, channel, method, "selection done, seconds", round(selector.elapsed_seconds_, 1), flush=True)
    if method == "fbsbs":
        missing = [k for k in ks if k not in sets]
        if missing:
            raise SystemExit(f"в sets.json нет наборов для k = {missing}; удалите ranking.csv и sets.json и запустите заново")
    check_path = ion_dir / "linear_check.json"
    checks = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
    checks.update({str(k): linear_check(X, y, ordered_for(method, ranking, sets, k)[:k]) for k in ks})
    check_path.write_text(json.dumps(dict(sorted(checks.items(), key=lambda item: int(item[0]))), indent=2), encoding="utf-8")
    if perceptron:
        for k in ks:
            order = ordered_for(method, ranking, sets, k)
            frame = pd.DataFrame({"Feature": order, "Individual_Rank": np.arange(1, len(order) + 1)})
            train_curve(directory, ion, X, y, frame, [k])


def run_full(ion: str, perceptron: bool) -> None:
    directory = OUT_DIR / "all_channels"
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_ion(ion)
    columns = list(X.columns)
    checks = {str(len(columns)): linear_check(X, y, columns)}
    (ion_dir / "linear_check.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if perceptron:
        order = pd.DataFrame({"Feature": columns, "Individual_Rank": np.arange(1, len(columns) + 1)})
        train_curve(directory, ion, X, y, order, [len(columns)])


def describe(relative: tuple[str, ...]) -> dict:
    """Вариант по пути папки иона относительно OUT_DIR."""
    if relative[0] == "all_channels":
        return {"spectrum_norm": "none", "channel_norm": "none", "method": "all_channels", "ion": relative[1]}
    combo, method, ion = relative
    spectrum, channel = combo.split("__")
    return {"spectrum_norm": spectrum, "channel_norm": channel, "method": method, "ion": ion}


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
                        "r2_test_std": point.get("std"),
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
    parser.add_argument("--job", nargs="+", help="rank <спектр> <каналы> <bsbs|fbsbs> <ион> | full <ион>")
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--max-swaps", type=int, default=MAX_SWAPS, help="fbsbs: сколько обменов подряд на итерации")
    parser.add_argument("--no-perceptron", action="store_true", help="только отбор и линейная проверка")
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
    bad = f"неизвестное задание: {' '.join(args.job)}"
    if kind == "rank" and len(rest) == 4:
        spectrum, channel, method, ion = rest
        if spectrum not in SPECTRUM_NORMS or channel not in SCALINGS or method not in METHODS or ion not in TARGETS:
            parser.error(bad)
        run_rank(spectrum, channel, method, ion, sorted(set(args.ks)), perceptron, args.max_swaps)
    elif kind == "full" and len(rest) == 1:
        if rest[0] not in TARGETS:
            parser.error(bad)
        run_full(rest[0], perceptron)
    else:
        parser.error(bad)
    print("job done:", " ".join(args.job), flush=True)


if __name__ == "__main__":
    main()
