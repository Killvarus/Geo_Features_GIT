"""
Быстрая оценка рейтинга признаков без персептрона.

Для первых k признаков рейтинга обучается Ridge на train. alpha выбирается
leave-one-out кросс-валидацией по train (как RidgeCV в sklearn), validation в
выборе не участвует. Качество — R² на validation. Test здесь не используется.

GCV здесь не годится: при k > n он выбирает почти нулевой alpha (интерполяцию),
потому что RSS → 0 быстрее, чем n − df → 1.

Ridge для всех alpha сразу считается через одно собственное разложение
меньшей из матриц XᵀX и XXᵀ (src/models/linear_selection.ridge_spectrum).
"""
from __future__ import annotations

import numpy as np

from src.models.linear_selection import LOO_ALPHAS, loo_alpha, ridge_spectrum

ALPHAS = LOO_ALPHAS


def ridge_loo_r2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    alphas: np.ndarray = ALPHAS,
) -> tuple[float, float]:
    """R² на оценочной выборке у Ridge с alpha по LOO на train. Возвращает (r2, alpha)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0
    train = (X_train - mean) / std
    evaluation = (X_eval - mean) / std
    y_mean = y_train.mean()
    target = y_train - y_mean

    u, s2, v = ridge_spectrum(train)
    alpha = loo_alpha(u, s2, target, alphas)
    projected = u.T @ target
    if v is not None:
        # β = V·diag(s / (s² + alpha))·Uᵀy
        prediction = evaluation @ (v @ (np.sqrt(s2) / (s2 + alpha) * projected)) + y_mean
    else:
        # β = Xᵀ·U·diag(1 / (s² + alpha))·Uᵀy
        prediction = (evaluation @ train.T) @ (u @ (projected / (s2 + alpha))) + y_mean
    residual = y_eval - prediction
    r2 = 1.0 - float(residual @ residual) / float(np.sum((y_eval - y_eval.mean()) ** 2))
    return r2, alpha
