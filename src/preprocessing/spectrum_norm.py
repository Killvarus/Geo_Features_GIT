"""
Нормировка каждого спектра 6P отдельно (по строке данных).

Строка — 5427 каналов: 27 возбуждений (280–410 нм, шаг 5) по 201 испусканию
(375–575 нм, шаг 1), возбуждение меняется медленно.

Линии рассеяния Рэлея (|испускание − возбуждение| ≤ 10 нм и
|испускание − 2·возбуждение| ≤ 15 нм, 224 канала) в статистики нормировок не
входят: их интенсивность на порядки выше флуоресценции и определялась бы ими.
Делятся при этом все каналы, включая рассеяние.

  raw     — без изменений;
  max     — на максимум спектра вне линий рассеяния;
  area    — на сумму спектра вне линий рассеяния;
  snv     — минус среднее и деление на std спектра вне линий рассеяния;
  raman   — на высоту пика комбинационного рассеяния воды: возбуждение 340 нм,
            испускание 382–386 нм минус фон (375–377 и 391–393 нм). У единичных
            спектров высота около нуля, поэтому она ограничена снизу 1-м
            процентилем по train;
  row_max — каждая строка возбуждения на свой максимум модуля вне линий
            рассеяния (в строках 280–300 нм почти шум, обычный максимум бывает < 0).
"""
from __future__ import annotations

import numpy as np

N_EXC = 27
N_EM = 201
EXC_NM = np.array([280 + 5 * i for i in range(N_EXC)])
EM_NM = np.arange(375, 576)
SPECTRUM_NORMS = ("raw", "max", "area", "snv", "raman", "row_max")


def fluorescence_mask() -> np.ndarray:
    """True для каналов вне линий рассеяния, форма (5427,)."""
    excitation, emission = np.meshgrid(EXC_NM, EM_NM, indexing="ij")
    scatter = (np.abs(emission - excitation) <= 10) | (np.abs(emission - 2 * excitation) <= 15)
    return ~scatter.ravel()


def raman_height(X: np.ndarray) -> np.ndarray:
    """Высота пика воды при возбуждении 340 нм над линейным фоном, по спектру."""
    row = X.reshape(len(X), N_EXC, N_EM)[:, int(np.flatnonzero(EXC_NM == 340)[0]), :]

    def band(low: int, high: int) -> np.ndarray:
        return row[:, (EM_NM >= low) & (EM_NM <= high)].mean(axis=1)

    return band(382, 386) - (band(375, 377) + band(391, 393)) / 2.0


def _positive(denominator: np.ndarray, what: str) -> np.ndarray:
    """Делить можно только на положительное: иначе спектр меняет знак или улетает в бесконечность."""
    bad = int((denominator <= 0).sum())
    if bad:
        raise ValueError(f"{what} спектра ≤ 0 у {bad} строк, нормировка не определена")
    return denominator


def normalize_spectra(X: np.ndarray, method: str, n_train: int) -> np.ndarray:
    """
    X — (строки, 5427). n_train — первые строки train: по ним считается только
    нижняя граница пика воды для raman, остальные нормировки построчные.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_EXC * N_EM:
        raise ValueError(f"ожидалась матрица (*, {N_EXC * N_EM})")
    mask = fluorescence_mask()
    if method == "raw":
        return X.copy()
    if method == "max":
        return X / _positive(X[:, mask].max(axis=1, keepdims=True), "максимум")
    if method == "area":
        return X / _positive(X[:, mask].sum(axis=1, keepdims=True), "площадь")
    if method == "snv":
        inside = X[:, mask]
        return (X - inside.mean(axis=1, keepdims=True)) / inside.std(axis=1, keepdims=True)
    if method == "raman":
        height = raman_height(X)
        floor = float(np.percentile(height[:n_train], 1))
        if floor <= 0:
            raise ValueError("пик воды на train неположителен, нормировать на него нельзя")
        return X / np.maximum(height, floor)[:, None]
    if method == "row_max":
        cube = X.reshape(len(X), N_EXC, N_EM)
        inside = np.where(mask.reshape(N_EXC, N_EM)[None], np.abs(cube), 0.0)
        return (cube / _positive(inside.max(axis=2, keepdims=True), "максимум строки")).reshape(len(X), -1)
    raise ValueError(f"нормировка по спектру должна быть одной из {SPECTRUM_NORMS}")
