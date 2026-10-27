"""
Положение отобранных каналов на матрице возбуждения-испускания.

Сетка та же, что у спектров 6P: 27 возбуждений (280–410 нм, шаг 5)
и 201 испускание (375–575 нм, шаг 1). Канал E395_em397 — это та же ячейка,
что номер колонки в вытянутом спектре: возбуждение меняется медленно.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_EXC = 27
N_EM = 201
N_FEATURES = N_EXC * N_EM
EXC_NM = [280 + 5 * i for i in range(N_EXC)]
EM_NM = list(range(375, 576))
_EXC = set(EXC_NM)
_EM = set(EM_NM)
_NAME = re.compile(r"E(\d+)_em(\d+)")
_NUMBER = re.compile(r"\d+")


def channel_coordinates(feature: str) -> tuple[int, int]:
    """Длины волн (возбуждение, испускание) для имени канала или номера колонки 1..5427."""
    text = str(feature).strip()
    named = _NAME.fullmatch(text)
    if named:
        excitation, emission = int(named.group(1)), int(named.group(2))
        if excitation not in _EXC or emission not in _EM:
            raise ValueError(f"канал вне сетки 6P: {feature}")
        return excitation, emission
    if _NUMBER.fullmatch(text):
        number = int(text)
        if not 1 <= number <= N_FEATURES:
            raise ValueError(f"номер канала вне 1..{N_FEATURES}: {feature}")
        row, col = divmod(number - 1, N_EM)
        return EXC_NM[row], EM_NM[col]
    raise ValueError(f"непонятное имя канала: {feature}")


def first_packet(ranking: pd.DataFrame) -> pd.DataFrame:
    """Первая группа рейтинга: у наших отборов это лучший пакет."""
    if "Feature" not in ranking.columns or "Group_Rank" not in ranking.columns:
        raise ValueError("в рейтинге нет колонок Feature и Group_Rank")
    if ranking.empty:
        raise ValueError("рейтинг пуст")
    group = ranking["Group_Rank"].min()
    packet = ranking.loc[ranking["Group_Rank"] == group].copy()
    if "Individual_Rank" in packet.columns:
        packet = packet.sort_values("Individual_Rank")
    return packet.reset_index(drop=True)


def mean_train_matrix(X: np.ndarray, n_train: int = 2225) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"ожидалась матрица (*, {N_FEATURES}), пришло {getattr(X, 'shape', None)}")
    if len(X) < n_train:
        raise ValueError(f"строк {len(X)}, для среднего train нужно {n_train}")
    return np.asarray(X[:n_train], dtype=float).mean(axis=0).reshape(N_EXC, N_EM)


def color_limits(matrix: np.ndarray) -> tuple[float, float]:
    vmax = float(np.percentile(matrix, 99))
    if vmax <= 0:
        peak = float(np.max(matrix))
        vmax = peak if peak > 0 else 1.0
    return 0.0, vmax


def save_selection_map(
    matrix: np.ndarray,
    features: list[str],
    title: str,
    path: str | Path,
    legend: bool = True,
) -> list[tuple[int, int]]:
    """Тепловая карта среднего спектра и белые маркеры отобранных каналов."""
    if matrix.shape != (N_EXC, N_EM):
        raise ValueError(f"матрица {matrix.shape}, нужна {(N_EXC, N_EM)}")
    coords = [channel_coordinates(feature) for feature in features]
    excitations = [item[0] for item in coords]
    emissions = [item[1] for item in coords]
    vmin, vmax = color_limits(matrix)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        extent=(EM_NM[0] - 0.5, EM_NM[-1] + 0.5, EXC_NM[0] - 2.5, EXC_NM[-1] + 2.5),
    )
    if coords:
        ax.scatter(
            emissions,
            excitations,
            s=14,
            c="white",
            edgecolors="black",
            linewidths=0.4,
            label="Отобранные каналы",
            zorder=3,
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Интенсивность, среднее train")
    ax.set_xlabel("Длина волны испускания, нм")
    ax.set_ylabel("Длина волны возбуждения, нм")
    ax.set_title(title)
    if coords and legend:
        ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return coords


def top_features(ranking: pd.DataFrame, k: int) -> list[str]:
    """Первые k каналов рейтинга по Individual_Rank."""
    if k < 1:
        raise ValueError("k должен быть >= 1")
    ordered = ranking.sort_values("Individual_Rank")
    return ordered["Feature"].astype(str).head(k).tolist()


def save_selection_grid(
    matrix: np.ndarray,
    panels: list[tuple[str, list[str]]],
    title: str,
    path: str | Path,
    ncols: int = 3,
) -> None:
    """Несколько карт на одной фигуре: средний спектр и отобранные каналы на каждой панели."""
    if matrix.shape != (N_EXC, N_EM):
        raise ValueError(f"матрица {matrix.shape}, нужна {(N_EXC, N_EM)}")
    if not panels:
        raise ValueError("нет панелей")
    nrows = math.ceil(len(panels) / ncols)
    vmin, vmax = color_limits(matrix)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    extent = (EM_NM[0] - 0.5, EM_NM[-1] + 0.5, EXC_NM[0] - 2.5, EXC_NM[-1] + 2.5)
    for ax, (label, features) in zip(axes.ravel(), panels):
        ax.imshow(matrix, origin="lower", aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax, extent=extent)
        coords = [channel_coordinates(feature) for feature in features]
        if coords:
            ax.scatter(
                [c[1] for c in coords],
                [c[0] for c in coords],
                s=10,
                c="white",
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Испускание, нм", fontsize=8)
        ax.set_ylabel("Возбуждение, нм", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def rank_parts(ranking: pd.DataFrame, n_parts: int = 5) -> dict[str, int]:
    """
    Номер части списка для каждого канала: список по Individual_Rank делится
    на n_parts почти равных кусков, 1 — начало списка (лучшие каналы).
    """
    if n_parts < 1:
        raise ValueError("n_parts должен быть >= 1")
    ordered = ranking.sort_values("Individual_Rank")["Feature"].astype(str).tolist()
    total = len(ordered)
    return {feature: position * n_parts // total + 1 for position, feature in enumerate(ordered)}


def parts_matrix(parts: dict[str, int]) -> np.ndarray:
    """Номера частей на сетке (возбуждение, испускание); каналы вне рейтинга — 0."""
    matrix = np.zeros((N_EXC, N_EM), dtype=int)
    for feature, part in parts.items():
        excitation, emission = channel_coordinates(feature)
        matrix[EXC_NM.index(excitation), EM_NM.index(emission)] = part
    return matrix


def _draw_parts(ax, spectrum: np.ndarray, parts: np.ndarray, n_parts: int):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = plt.get_cmap("RdYlBu")(np.linspace(0.0, 1.0, n_parts))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, n_parts + 1.5), cmap.N)
    extent = (EM_NM[0] - 0.5, EM_NM[-1] + 0.5, EXC_NM[0] - 2.5, EXC_NM[-1] + 2.5)
    image = ax.imshow(
        np.ma.masked_equal(parts, 0), origin="lower", aspect="auto", cmap=cmap, norm=norm, extent=extent
    )
    # Изолинии среднего спектра — чтобы видеть, где пик и линии рассеяния.
    _, vmax = color_limits(spectrum)
    ax.contour(
        EM_NM,
        EXC_NM,
        spectrum,
        levels=vmax * np.array([0.1, 0.3, 0.6, 0.9]),
        colors="black",
        linewidths=0.6,
        alpha=0.7,
    )
    return image


def save_parts_map(
    spectrum: np.ndarray,
    parts: np.ndarray,
    title: str,
    path: str | Path,
    n_parts: int = 5,
) -> None:
    """Карта: цвет канала — номер части списка, чёрные изолинии — средний спектр train."""
    if spectrum.shape != (N_EXC, N_EM) or parts.shape != (N_EXC, N_EM):
        raise ValueError(f"нужны матрицы {(N_EXC, N_EM)}")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = _draw_parts(ax, spectrum, parts, n_parts)
    colorbar = fig.colorbar(image, ax=ax, ticks=range(1, n_parts + 1), fraction=0.046, pad=0.04)
    colorbar.set_label(f"Часть списка (1 — первые {100 // n_parts}%)")
    ax.set_xlabel("Длина волны испускания, нм")
    ax.set_ylabel("Длина волны возбуждения, нм")
    ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_parts_grid(
    spectrum: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    title: str,
    path: str | Path,
    n_parts: int = 5,
    ncols: int = 3,
) -> None:
    """То же для нескольких рейтингов на одной фигуре (например, шесть ионов)."""
    if not panels:
        raise ValueError("нет панелей")
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows), squeeze=False)
    image = None
    for ax, (label, parts) in zip(axes.ravel(), panels):
        image = _draw_parts(ax, spectrum, parts, n_parts)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Испускание, нм", fontsize=8)
        ax.set_ylabel("Возбуждение, нм", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), ticks=range(1, n_parts + 1), fraction=0.02, pad=0.02)
    colorbar.set_label(f"Часть списка (1 — первые {100 // n_parts}%)")
    fig.suptitle(title, y=1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
