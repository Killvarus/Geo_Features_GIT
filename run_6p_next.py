"""
Подготовленные эксперименты по итогам ревизии (docs/research_review_2026-11.md). Пишет в
experiments/6p_next/<разбиение>/... Протокол проверки как в run_6p_selnorm_grid.py:
нормировка спектров и каналов действует только на отбор, персептрон, Ridge и МНК — на
исходных спектрах с z-score. Отбор: Ridge alpha=700, пакет 10%, 5-fold на train,
solver="auto" (двойственная форма, пока признаков больше строк фолда; рейтинг тот же).

Наборы заданий (--list-jobs --set …):
  splits   — разбиения cv2 и cv3: все каналы + BSBS при спектрах raw, max, area (48 заданий).
             Даёт разброс от разбиения, которого сейчас нет ни у одного вывода.
  variants — cv1, спектры max и area, 6 ионов (60 заданий):
               bsbs_r3      — 3 повторения 5-fold (меньше подгонки под одно разбиение);
               bsbs_shuffle — новые фолды на каждой итерации;
               gbsbs_w5     — BSBS по окнам 5 нм испускания внутри строки возбуждения;
               gbsbs_w10    — то же, окна 10 нм;
               bsbs_asinh   — отбор по arcsinh(I/10), затем нормировка спектров.
             Базовый BSBS для сравнения уже посчитан в experiments/6p_selnorm_grid.
  boot     — cv1, спектры max, 20 подвыборок по 80% train на ион, только рейтинги (120 заданий).
             Для оценки устойчивости отбора (src/evaluation/stability.py).
  speed    — время BSBS с solver primal и auto на всех каналах, Cu (1 задание, без персептрона).

Задания:
  rank  <разбиение> <вариант> <спектр> <ион>
  full  <разбиение> <ион>
  boot  <разбиение> <спектр> <ион> <номер>
  speed <ион>
Запуск:
  python run_6p_next.py --list-jobs --set splits
  python run_6p_next.py --job rank cv1 gbsbs_w5 max Cu
Задание можно перезапускать: готовый отбор и посчитанные точки пропускаются.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN, TARGETS, feature_names, train_curve
from run_6p_norm_grid import linear_check
from src.models.linear_selection import LinearGroupBSBS, LinearPPOP
from src.preprocessing.spectrum_norm import SPECTRUM_NORMS, normalize_spectra, transform_intensity

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "6p_next"
CACHES = {f"cv{i}": ROOT / "experiments" / "6p_full_patience300" / f"xy_cv{i}.npz" for i in (1, 2, 3)}
ALPHA = 700.0
STEP = 0.1
CV = 5
KS = [10, 50, 100, 250, 500, 1000, 2000]
VARIANTS = {
    "bsbs": {},
    "bsbs_r3": {"repeats": 3},
    "bsbs_shuffle": {"reshuffle": True},
    "gbsbs_w5": {"window": 5},
    "gbsbs_w10": {"window": 10},
    "bsbs_asinh": {"transform": "asinh"},
}
BOOT_RUNS = 20
BOOT_FRACTION = 0.8


def jobs(which: str) -> list[str]:
    lines = []
    if which == "splits":
        for split in ("cv2", "cv3"):
            lines += [f"full {split} {ion}" for ion in TARGETS]
            lines += [f"rank {split} bsbs {s} {ion}" for s in ("raw", "max", "area") for ion in TARGETS]
    elif which == "variants":
        names = ["bsbs_r3", "bsbs_shuffle", "gbsbs_w5", "gbsbs_w10", "bsbs_asinh"]
        lines += [f"rank cv1 {v} {s} {ion}" for v in names for s in ("max", "area") for ion in TARGETS]
    elif which == "boot":
        lines += [f"boot cv1 max {ion} {b}" for ion in TARGETS for b in range(BOOT_RUNS)]
    elif which == "speed":
        lines.append("speed Cu")
    else:
        raise ValueError(which)
    return lines


def load(split: str, ion: str) -> tuple[pd.DataFrame, pd.Series]:
    blob = np.load(CACHES[split])
    X = pd.DataFrame(blob["X"], columns=feature_names())
    y = pd.Series(blob["y"][:, TARGETS.index(ion)], name=ion)
    return X, y


def window_groups(names: list[str], width_nm: int) -> list[str]:
    """Группа — окно width_nm нм испускания внутри одной строки возбуждения."""
    labels = []
    for name in names:
        excitation, emission = name[1:].split("_em")
        labels.append(f"E{excitation}_w{(int(emission) - 375) // width_nm}")
    return labels


def selection_frame(X: pd.DataFrame, spectrum: str, transform: str | None, rows: np.ndarray) -> pd.DataFrame:
    values = X.to_numpy(dtype=float)
    if transform:
        values = transform_intensity(values, transform)
    values = normalize_spectra(values, spectrum, N_TRAIN)
    return pd.DataFrame(values[rows], columns=X.columns)


def selector_for(variant: str, names: list[str]):
    options = VARIANTS[variant]
    if "window" in options:
        return LinearGroupBSBS(groups=window_groups(names, options["window"]), alpha=ALPHA, step=STEP, cv=CV)
    return LinearPPOP(
        alpha=ALPHA, step=STEP, cv=CV, scaling="zscore", solver="auto",
        repeats=options.get("repeats", 1), reshuffle=options.get("reshuffle", False),
    )


def run_rank(split: str, variant: str, spectrum: str, ion: str, ks: list[int], perceptron: bool) -> None:
    directory = OUT_DIR / split / variant / spectrum
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(split, ion)
    path = ion_dir / "ranking.csv"
    if path.exists():
        ranking = pd.read_csv(path)
        print(split, variant, spectrum, ion, "ranking exists", flush=True)
    else:
        train = np.arange(N_TRAIN)
        selector = selector_for(variant, list(X.columns))
        selector.fit(selection_frame(X, spectrum, VARIANTS[variant].get("transform"), train), y.iloc[:N_TRAIN])
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        settings = {"split": split, "variant": variant, "options": VARIANTS[variant], "spectrum_norm_for_selection": spectrum,
                    "evaluation_data": "raw spectra, z-score channels", "alpha": ALPHA, "step": STEP, "cv": CV,
                    "seconds": selector.elapsed_seconds_}
        (ion_dir / "selection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        selector.save_ranking(path)
        ranking = selector.ranking_frame()
        print(split, variant, spectrum, ion, "ranking done, seconds", round(selector.elapsed_seconds_, 1), flush=True)
    ordered = ranking.sort_values("Individual_Rank")["Feature"].astype(str).tolist()
    check_path = ion_dir / "linear_check.json"
    checks = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
    checks.update({str(k): linear_check(X, y, ordered[:k]) for k in ks})
    check_path.write_text(json.dumps(dict(sorted(checks.items(), key=lambda item: int(item[0]))), indent=2), encoding="utf-8")
    if perceptron:
        train_curve(directory, ion, X, y, ranking, ks)


def run_full(split: str, ion: str, perceptron: bool) -> None:
    directory = OUT_DIR / split / "all_channels"
    ion_dir = directory / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    X, y = load(split, ion)
    columns = list(X.columns)
    (ion_dir / "linear_check.json").write_text(json.dumps({str(len(columns)): linear_check(X, y, columns)}, indent=2), encoding="utf-8")
    if perceptron:
        order = pd.DataFrame({"Feature": columns, "Individual_Rank": np.arange(1, len(columns) + 1)})
        train_curve(directory, ion, X, y, order, [len(columns)])


def run_boot(split: str, spectrum: str, ion: str, number: int) -> None:
    ion_dir = OUT_DIR / split / "boot" / spectrum / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    path = ion_dir / f"ranking_{number:02d}.csv"
    if path.exists():
        print(split, "boot", spectrum, ion, number, "exists", flush=True)
        return
    X, y = load(split, ion)
    rng = np.random.default_rng(1000 + number)
    rows = np.sort(rng.choice(N_TRAIN, size=int(BOOT_FRACTION * N_TRAIN), replace=False))
    selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling="zscore", solver="auto")
    selector.fit(selection_frame(X, spectrum, None, rows), y.iloc[rows].reset_index(drop=True))
    selector.save_ranking(path)
    print(split, "boot", spectrum, ion, number, "done, seconds", round(selector.elapsed_seconds_, 1), flush=True)


def run_speed(ion: str) -> None:
    out = OUT_DIR / "speed"
    out.mkdir(parents=True, exist_ok=True)
    X, y = load("cv1", ion)
    frame = selection_frame(X, "max", None, np.arange(N_TRAIN))
    result = {}
    rankings = {}
    for solver in ("primal", "auto"):
        started = time.perf_counter()
        selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling="zscore", solver=solver).fit(frame, y.iloc[:N_TRAIN])
        result[solver] = {"seconds": time.perf_counter() - started, "first_iteration": float(selector.history_["seconds"].iloc[0])}
        rankings[solver] = selector.ranking_frame()["Feature"].tolist()
    result["same_ranking"] = rankings["primal"] == rankings["auto"]
    (out / f"{ion}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--set", choices=["splits", "variants", "boot", "speed"], default="splits")
    parser.add_argument("--job", nargs="+")
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--no-perceptron", action="store_true")
    args = parser.parse_args()
    if args.list_jobs:
        print("\n".join(jobs(args.set)))
        return
    if not args.job:
        parser.error("нужен --list-jobs или --job")
    kind, *rest = args.job
    perceptron = not args.no_perceptron
    bad = f"неизвестное задание: {' '.join(args.job)}"
    if kind == "rank" and len(rest) == 4:
        split, variant, spectrum, ion = rest
        if split not in CACHES or variant not in VARIANTS or spectrum not in SPECTRUM_NORMS or ion not in TARGETS:
            parser.error(bad)
        run_rank(split, variant, spectrum, ion, sorted(set(args.ks)), perceptron)
    elif kind == "full" and len(rest) == 2:
        if rest[0] not in CACHES or rest[1] not in TARGETS:
            parser.error(bad)
        run_full(rest[0], rest[1], perceptron)
    elif kind == "boot" and len(rest) == 4:
        split, spectrum, ion, number = rest
        if split not in CACHES or spectrum not in SPECTRUM_NORMS or ion not in TARGETS or not number.isdigit():
            parser.error(bad)
        run_boot(split, spectrum, ion, int(number))
    elif kind == "speed" and len(rest) == 1 and rest[0] in TARGETS:
        run_speed(rest[0])
    else:
        parser.error(bad)
    print("job done:", " ".join(args.job), flush=True)


if __name__ == "__main__":
    main()
