"""
Замер ускорения BSBS: быстрый расчёт против прямого перебора на одной конфигурации.

Конфигурация одна для всех: спектры на максимум, z-score каналов по обучающей
части фолда, Ridge alpha = 700, 5-fold на train, пакет 10% от оставшихся, Cu,
один поток. Прямой перебор считается двумя способами:
- sklearn: Ridge(solver="cholesky") с нуля на каждую модель (при p > n он сам
  переходит к двойственной форме размера n × n);
- prepared: матрица XᵀX или XXᵀ готова заранее, на каждую модель только
  разложение Холецкого, то есть самый выгодный для перебора вариант.

На всех 5427 каналах перебор целиком не считается: время одной модели меряется
при нескольких p и суммируется по расписанию итераций. Малые наборы каналов
считаются целиком обоими способами, и рейтинги сравниваются поканально.

Пример:
  python benchmark_6p_fast_bsbs.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN
from run_6p_norm_grid import ALPHA, CV, STEP, load
from src.models.linear_selection import LinearPPOP, _Split, packet_size, scale_channels

OUT_DIR = Path(__file__).resolve().parent / "experiments" / "6p_fast_bsbs_benchmark"
ION = "Cu"
SPECTRUM = "max"
MEASURE_P = [5427, 3500, 2000, 1500, 1000, 500, 250, 100, 30]
SMALL_P = [150, 300, 600]
BUDGET_SECONDS = 12.0


def schedule(p: int) -> list[int]:
    """Сколько признаков остаётся перед каждой итерацией."""
    sizes = []
    while p > 0:
        sizes.append(p)
        p -= packet_size(p, STEP)
    return sizes


def folds(X: np.ndarray, y: np.ndarray) -> list[tuple]:
    parts = []
    for fit, held in KFold(n_splits=CV, shuffle=True, random_state=0).split(X):
        X_fit, X_eval = scale_channels(X[fit], X[held], method="zscore")
        parts.append((X_fit, y[fit], X_eval, y[held]))
    return parts


def sse_sklearn(part: tuple, columns: np.ndarray, position: int) -> float:
    X_fit, y_fit, X_eval, y_eval = part
    kept = np.delete(columns, position)
    model = Ridge(alpha=ALPHA, solver="cholesky").fit(X_fit[:, kept], y_fit)
    residual = y_eval - model.predict(X_eval[:, kept])
    return float(residual @ residual)


class Prepared:
    """Прямой перебор, когда всё, что не зависит от выбрасываемого признака, уже посчитано."""

    def __init__(self, part: tuple, columns: np.ndarray):
        X_fit, y_fit, X_eval, y_eval = part
        self.X = X_fit[:, columns]
        self.y = y_fit - y_fit.mean()
        self.X_eval = X_eval[:, columns]
        self.y_eval = y_eval - y_fit.mean()
        self.dual = self.X.shape[1] > self.X.shape[0]
        self.matrix = self.X @ self.X.T if self.dual else self.X.T @ self.X
        self.xty = self.X.T @ self.y

    def sse_without(self, position: int) -> float:
        if self.dual:
            channel = self.X[:, position]
            kernel = self.matrix - np.outer(channel, channel)
            kernel[np.diag_indices_from(kernel)] += ALPHA
            weights = self.X.T @ cho_solve(cho_factor(kernel, overwrite_a=True, check_finite=False), self.y)
            weights[position] = 0.0
            prediction = self.X_eval @ weights
        else:
            kept = np.delete(np.arange(self.X.shape[1]), position)
            matrix = self.matrix[np.ix_(kept, kept)]
            matrix[np.diag_indices_from(matrix)] += ALPHA
            beta = cho_solve(cho_factor(matrix, overwrite_a=True, check_finite=False), self.xty[kept])
            prediction = self.X_eval[:, kept] @ beta
        residual = self.y_eval - prediction
        return float(residual @ residual)


def sample_positions(p: int, count: int = 64) -> np.ndarray:
    return np.linspace(0, p - 1, min(p, count)).astype(int)


def measure_direct(parts: list[tuple], p: int) -> dict:
    """Время одной модели при p признаках и сверка с быстрой формулой."""
    columns = np.linspace(0, parts[0][0].shape[1] - 1, p).astype(int)
    splits = [_Split(*part[:2], *part[2:], "zscore") for part in parts]
    fast = [split.sse_without_each(columns, ALPHA)[0] for split in splits]
    positions = sample_positions(p)

    started = time.perf_counter()
    prepared = [Prepared(part, columns) for part in parts]
    prepare_seconds = (time.perf_counter() - started) / len(parts)

    result = {"p": p, "prepare_seconds": prepare_seconds}
    worst = 0.0
    for label, solve in (
        ("sklearn", lambda fold, j: sse_sklearn(parts[fold], columns, j)),
        ("prepared", lambda fold, j: prepared[fold].sse_without(j)),
    ):
        done, spent = 0, 0.0
        while done < len(positions) and (spent < BUDGET_SECONDS or done < 2):
            fold = done % len(parts)
            position = int(positions[done])
            started = time.perf_counter()
            sse = solve(fold, position)
            spent += time.perf_counter() - started
            worst = max(worst, abs(sse - fast[fold][position]) / fast[fold][position])
            done += 1
        result[f"{label}_seconds_per_model"] = spent / done
        result[f"{label}_models_measured"] = done
    result["worst_relative_sse_difference"] = worst
    print(result, flush=True)
    return result


def interpolate(measured: pd.DataFrame, column: str, sizes: np.ndarray) -> np.ndarray:
    ordered = measured.sort_values("p")
    return np.exp(np.interp(np.log(sizes), np.log(ordered["p"]), np.log(ordered[column])))


def direct_ranking(parts: list[tuple], p: int) -> tuple[list[int], float]:
    """BSBS прямым перебором; порядок выбрасывания — как в LinearPPOP."""
    remaining = np.linspace(0, parts[0][0].shape[1] - 1, p).astype(int)
    started = time.perf_counter()
    dropped: list[int] = []
    while remaining.size:
        prepared = [Prepared(part, remaining) for part in parts]
        sse = np.array([sum(item.sse_without(j) for item in prepared) for j in range(remaining.size)])
        size = packet_size(remaining.size, STEP)
        drop = np.argsort(sse, kind="stable")[:size]
        dropped.extend(remaining[drop].tolist())
        remaining = np.delete(remaining, drop)
    return dropped[::-1], time.perf_counter() - started


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load(SPECTRUM, ION)
    X_train = X.iloc[:N_TRAIN].to_numpy(dtype=float)
    y_train = y.iloc[:N_TRAIN].to_numpy(dtype=float)
    parts = folds(X_train, y_train)
    report: dict = {
        "machine": platform.processor(),
        "threads": 1,
        "ion": ION,
        "spectrum": SPECTRUM,
        "alpha": ALPHA,
        "cv": CV,
        "step": STEP,
        "n_train": N_TRAIN,
    }

    measured = pd.DataFrame([measure_direct(parts, p) for p in MEASURE_P])
    measured.to_csv(OUT_DIR / "direct_models.csv", index=False)

    selector = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling="zscore").fit(X.iloc[:N_TRAIN], y.iloc[:N_TRAIN])
    history = selector.history_
    history.to_csv(OUT_DIR / "fast_history.csv", index=False)
    report["fast_seconds_total"] = selector.elapsed_seconds_
    report["iterations"] = len(history)
    print("fast", selector.elapsed_seconds_, flush=True)

    sizes = np.array(schedule(X.shape[1]))
    report["models_per_run"] = int(sizes.sum() * CV)
    for label in ("sklearn", "prepared"):
        per_model = interpolate(measured, f"{label}_seconds_per_model", sizes)
        prepare = interpolate(measured, "prepare_seconds", sizes) if label == "prepared" else 0.0
        report[f"{label}_seconds_total_estimate"] = float((CV * (sizes * per_model + prepare)).sum())
        report[f"{label}_first_iteration_seconds"] = float(CV * (sizes[0] * per_model[0] + (prepare[0] if label == "prepared" else 0.0)))
    report["fast_first_iteration_seconds"] = float(history["seconds"].iloc[0]) if "seconds" in history else None

    small = []
    for p in SMALL_P:
        columns = np.linspace(0, X.shape[1] - 1, p).astype(int)
        subset = X.iloc[:N_TRAIN, columns]
        ranking_direct, seconds_direct = direct_ranking(parts, p)
        fast = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling="zscore").fit(subset, y.iloc[:N_TRAIN])
        frame = fast.ranking_frame().sort_values("Individual_Rank")
        names = {name: i for i, name in zip(columns, X.columns[columns])}
        ranking_fast = [names[name] for name in frame["Feature"]]
        row = {
            "p": p,
            "direct_seconds": seconds_direct,
            "fast_seconds": fast.elapsed_seconds_,
            "same_ranking": ranking_direct == ranking_fast,
            "same_channels_in_first_tenth": len(set(ranking_direct[: p // 10]) & set(ranking_fast[: p // 10])) / (p // 10),
        }
        small.append(row)
        print(row, flush=True)
    pd.DataFrame(small).to_csv(OUT_DIR / "small_runs.csv", index=False)
    report["small_runs"] = small

    (OUT_DIR / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
