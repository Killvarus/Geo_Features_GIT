"""
Пакетный RFE на гребневой регрессии.

На каждом шаге Ridge обучается один раз на оставшихся каналах.
Важность канала — модуль его веса после стандартизации train.
Оценка на validation здесь не считается.

Обратный отбор выбрасывает пакет с наименьшими модулями и повторяет
обучение. Ранг 1 получает канал из последнего пакета с наибольшим модулем.
Прямой отбор забирает пакет с наибольшими модулями и убирает его из пула.
Ранг 1 — канал, добавленный первым.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _check_args(alpha: float, packet: int) -> tuple[float, int]:
    if packet < 1:
        raise ValueError("packet size must be >= 1")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    return float(alpha), int(packet)


def _prepare(X: pd.DataFrame, y: pd.Series | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X должен быть DataFrame с именами признаков")
    if X.shape[1] < 1:
        raise ValueError("X не содержит признаков")
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(target) != len(X):
        raise ValueError("Длины X и y не совпадают")
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))
    return scaled, target


def _absolute_coefficients(alpha: float, X: pd.DataFrame, y: np.ndarray) -> dict[Hashable, float]:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(X, y)
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    if len(coef) != X.shape[1]:
        raise RuntimeError("Ridge вернул коэффициенты не по всем признакам")
    return {feature: float(abs(value)) for feature, value in zip(X.columns, coef)}


def _load_curves(out_dir: Path, ions: list[str]) -> list[tuple[str, list[dict]]] | None:
    import json

    paths = [out_dir / ion / "curve_points.json" for ion in ions]
    missing = [ion for ion, path in zip(ions, paths) if not path.exists()]
    if missing:
        print("plot waiting for", missing, flush=True)
        return None
    curves = []
    for ion, path in zip(ions, paths):
        points = sorted(json.loads(path.read_text(encoding="utf-8")), key=lambda item: item["k"])
        curves.append((ion, points))
    return curves


def draw_r2_curves(curves: list[tuple[str, list[dict]]], title: str):
    """Собирает фигуру. Левый край оси — больше признаков, правый — меньше."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, (ion, points) in zip(axes.ravel(), curves):
        ks = [item["k"] for item in points]
        mean = [item["mean"] for item in points]
        std = [item["std"] for item in points]
        full = next(item for item in points if item["k"] == max(ks))
        ax.errorbar(ks, mean, yerr=std, fmt="o-", color="#1f4e79", lw=1.6, ms=4.5, capsize=3)
        ax.axhline(full["mean"], color="#c45c26", ls="--", lw=1.3)
        ax.axhspan(full["mean"] - full["std"], full["mean"] + full["std"], color="#c45c26", alpha=0.15)
        ax.set_title(f"{ion}   без отбора {full['mean']:.3f}±{full['std']:.3f}")
        ax.set_xlim(max(ks) + 200, 0)
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(curves):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("R² на тесте")
    axes[1, 0].set_ylabel("R² на тесте")
    for ax in axes[1]:
        ax.set_xlabel("Число признаков")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_r2_curves(out_dir: Path, ions: list[str], title: str) -> Path | None:
    """Шесть кривых R², как у пакетного отбора: ось числа признаков убывает."""
    out_dir = Path(out_dir)
    curves = _load_curves(out_dir, ions)
    if curves is None:
        return None
    fig = draw_r2_curves(curves, title)
    out = out_dir / "r2_vs_n_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("plot", out, flush=True)
    return out


def _ranking_frame(selector) -> pd.DataFrame:
    if selector.ranking_ is None:
        raise RuntimeError("Сначала вызовите fit()")
    frame = pd.DataFrame(
        {
            "Feature": list(selector.scores_),
            "Group_Rank": [selector.ranking_["group_rank"][feature] for feature in selector.scores_],
            "Individual_Rank": [selector.ranking_["individual_rank"][feature] for feature in selector.scores_],
            "Score": [selector.scores_[feature] for feature in selector.scores_],
        }
    )
    return frame.sort_values("Individual_Rank").reset_index(drop=True)


class BackwardRidgeRFE:
    """Обратный RFE: пакет с наименьшими |весами| выбрасывается первым."""

    def __init__(self, alpha: float = 700.0, n_features_to_drop: int = 1500):
        self.alpha, self.n_features_to_drop = _check_args(alpha, n_features_to_drop)
        self.ranking_: dict | None = None
        self.scores_: dict[Hashable, float] = {}
        self.elapsed_seconds_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BackwardRidgeRFE":
        scaled, target = _prepare(X, y)
        remaining = list(scaled.columns)
        eliminated: list[list[Hashable]] = []
        self.scores_ = {}
        started = time.perf_counter()
        drop_step = 1

        while remaining:
            step_started = time.perf_counter()
            coef = _absolute_coefficients(self.alpha, scaled.loc[:, remaining], target)
            ordered = sorted(remaining, key=lambda feature: (coef[feature], str(feature)))
            take = ordered[: min(self.n_features_to_drop, len(ordered))]
            for feature in take:
                self.scores_[feature] = coef[feature]
            eliminated.append(take)
            taken = set(take)
            remaining = [feature for feature in remaining if feature not in taken]
            print(
                f"backward drop {drop_step}: scored {len(coef)}, dropped {len(take)}, "
                f"remaining {len(remaining)}, smallest_abs_coef {coef[take[0]]:.6g}, "
                f"step_s {time.perf_counter() - step_started:.1f}",
                flush=True,
            )
            drop_step += 1

        group_rank = {}
        individual_rank = {}
        next_rank = 1
        group = 1
        for packet in reversed(eliminated):
            best_first = sorted(packet, key=lambda feature: (-self.scores_[feature], str(feature)))
            for offset, feature in enumerate(best_first):
                individual_rank[feature] = next_rank + offset
                group_rank[feature] = group
            next_rank += len(packet)
            group += 1

        self.ranking_ = {
            "group_rank": group_rank,
            "individual_rank": individual_rank,
            "scores": self.scores_,
        }
        self.elapsed_seconds_ = time.perf_counter() - started
        return self

    def ranking_frame(self) -> pd.DataFrame:
        return _ranking_frame(self)

    def save_ranking(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_frame().to_csv(path, index=False)
        return path


class ForwardRidgeRFE:
    """Прямой RFE: пакет с наибольшими |весами| забирается первым."""

    def __init__(self, alpha: float = 700.0, n_features_to_add: int = 1500):
        self.alpha, self.n_features_to_add = _check_args(alpha, n_features_to_add)
        self.ranking_: dict | None = None
        self.scores_: dict[Hashable, float] = {}
        self.elapsed_seconds_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ForwardRidgeRFE":
        scaled, target = _prepare(X, y)
        remaining = list(scaled.columns)
        group_rank = {}
        individual_rank = {}
        self.scores_ = {}
        next_rank = 1
        group = 1
        started = time.perf_counter()

        while remaining:
            step_started = time.perf_counter()
            coef = _absolute_coefficients(self.alpha, scaled.loc[:, remaining], target)
            ordered = sorted(remaining, key=lambda feature: (-coef[feature], str(feature)))
            take = ordered[: min(self.n_features_to_add, len(ordered))]
            for offset, feature in enumerate(take):
                individual_rank[feature] = next_rank + offset
                group_rank[feature] = group
                self.scores_[feature] = coef[feature]
            taken = set(take)
            remaining = [feature for feature in remaining if feature not in taken]
            print(
                f"forward add {group}: scored {len(coef)}, added {len(take)}, "
                f"selected {next_rank + len(take) - 1}, remaining {len(remaining)}, "
                f"best_abs_coef {coef[take[0]]:.6g}, "
                f"step_s {time.perf_counter() - step_started:.1f}",
                flush=True,
            )
            next_rank += len(take)
            group += 1

        self.ranking_ = {
            "group_rank": group_rank,
            "individual_rank": individual_rank,
            "scores": self.scores_,
        }
        self.elapsed_seconds_ = time.perf_counter() - started
        return self

    def ranking_frame(self) -> pd.DataFrame:
        return _ranking_frame(self)

    def save_ranking(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_frame().to_csv(path, index=False)
        return path
