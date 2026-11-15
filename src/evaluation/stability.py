"""
Устойчивость отбора признаков по нескольким запускам (подвыборки, бутстреп).

nogueira_stability — оценка Nogueira, Sechidis, Brown (JMLR 2018, 18(174)):
Φ = 1 − mean_f(s_f²) / (k̄/p · (1 − k̄/p)), где s_f² = M/(M−1)·p̂_f(1−p̂_f) —
несмещённая дисперсия индикатора «признак f выбран» по M наборам, k̄ — средний
размер набора, p — число признаков. Φ = 1 — наборы одинаковы, около 0 — как
случайный выбор того же размера. Работает при разных размерах наборов.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def selection_matrix(sets: Sequence[Sequence[Hashable]], features: Sequence[Hashable]) -> np.ndarray:
    """M × p матрица 0/1: какие признаки вошли в каждый набор."""
    index = {f: i for i, f in enumerate(features)}
    matrix = np.zeros((len(sets), len(features)))
    for row, chosen in enumerate(sets):
        unknown = [f for f in chosen if f not in index]
        if unknown:
            raise ValueError(f"признаки не из списка: {unknown[:5]}")
        matrix[row, [index[f] for f in chosen]] = 1.0
    return matrix


def nogueira_stability(sets: Sequence[Sequence[Hashable]], features: Sequence[Hashable]) -> float:
    matrix = selection_matrix(sets, features)
    runs, p = matrix.shape
    if runs < 2:
        raise ValueError("нужно хотя бы два набора")
    frequency = matrix.mean(axis=0)
    variance = runs / (runs - 1) * frequency * (1.0 - frequency)
    mean_size = matrix.sum(axis=1).mean()
    expected = mean_size / p * (1.0 - mean_size / p)
    if expected == 0.0:
        raise ValueError("все наборы пустые или содержат все признаки")
    return float(1.0 - variance.mean() / expected)
