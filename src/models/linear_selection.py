"""
ППОП и RFE для линейной регрессии: МНК (alpha=0) и Ridge (alpha>0).

Признаки стандартизуются по обучающей части, y центрируется. Так модель
совпадает с Pipeline(StandardScaler(), Ridge(alpha)), а при alpha=0 — с LinearRegression.
Вместо стандартизации можно взять другую нормировку по каналам (scaling, см. scale_channels).

Пусть A = XᵀX + alpha·I на текущем наборе признаков и β = A⁻¹Xᵀy. Решение той же
задачи без признака j: β₋ⱼ = β − A⁻¹[:, j]·βⱼ / A⁻¹[j, j]. Это точная формула, не
приближение, поэтому N моделей на N−1 признаке считаются из одного обращения A.

ППОП выбрасывает пакет признаков, без которых MSE на оценочной выборке выросла
меньше всего (или упала). RFE выбрасывает пакет с наименьшими |β| на train.
В обоих методах ранг 1 получает признак, продержавшийся дольше всех.

Пакет задаётся как в sklearn.feature_selection.RFE: целое — число признаков,
дробь из (0, 1) — доля от оставшихся (округление вверх, не меньше одного).

alpha="loo": штраф подбирается заново на каждой итерации (в RFE — при каждом
переобращении матрицы) по leave-one-out ошибке Ridge на всём train, как в RidgeCV.
Оптимальный штраф для 100 признаков другой, чем для 5427.
alpha="loo1se": то же, но берётся самый большой штраф, чья LOO-ошибка не больше
минимальной плюс одна её стандартная ошибка (правило одной SE, как lambda.1se
в glmnet). При тысяче скоррелированных каналов минимум LOO даёт почти МНК,
и решения о выбрасывании становятся шумом; правило 1SE держит штраф выше.
alpha="loo_schedule": правило 1SE, пока признаков больше switch_at, и минимум LOO,
когда их switch_at или меньше. Минимум лучше выбирает первые сотни каналов,
1SE не даёт отбору развалиться на тысячах скоррелированных каналов.
"""
from __future__ import annotations

import math
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from scipy.linalg import lapack
from scipy.linalg.blas import dger
from sklearn.model_selection import KFold


def check_step(step) -> None:
    if isinstance(step, bool):
        raise ValueError("step должен быть целым >= 1 или дробью из (0, 1)")
    if isinstance(step, Integral):
        if step < 1:
            raise ValueError("целый step должен быть >= 1")
        return
    if isinstance(step, Real) and 0.0 < float(step) < 1.0:
        return
    raise ValueError("step должен быть целым >= 1 или дробью из (0, 1)")


def packet_size(n_remaining: int, step, min_step: int = 1) -> int:
    """Сколько признаков выбросить, когда их осталось n_remaining."""
    check_step(step)
    if isinstance(step, Integral):
        size = int(step)
    else:
        size = math.ceil(float(step) * n_remaining)
    return min(n_remaining, max(size, int(min_step), 1))


LOO_ALPHAS = np.logspace(-3, 6, 46)
LOO_MODES = ("loo", "loo1se", "loo_schedule")


def _check_alpha(alpha) -> float | str:
    if isinstance(alpha, str):
        if alpha not in LOO_MODES:
            raise ValueError('alpha должен быть числом >= 0, "loo", "loo1se" или "loo_schedule"')
        return alpha
    alpha = float(alpha)
    if not alpha >= 0.0:
        raise ValueError("alpha должен быть >= 0")
    return alpha


def ridge_spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    U, s² и V из SVD X = U·S·Vᵀ через собственное разложение меньшей из XᵀX и XXᵀ.
    Нулевые сингулярные числа отброшены. V возвращается только при k ≤ n.
    """
    n, k = X.shape
    if k <= n:
        s2, v = np.linalg.eigh(X.T @ X)
    else:
        s2, u = np.linalg.eigh(X @ X.T)
    s2 = np.clip(s2, 0.0, None)
    keep = s2 > s2.max() * 1e-12
    s2 = s2[keep]
    if k <= n:
        v = v[:, keep]
        return (X @ v) / np.sqrt(s2), s2, v
    return u[:, keep], s2, None


def loo_alpha(
    u: np.ndarray, s2: np.ndarray, target: np.ndarray, alphas: np.ndarray = LOO_ALPHAS, rule: str = "min"
) -> float:
    """
    alpha по leave-one-out ошибке Ridge (центрированный y, свободный член).
    rule="min" — наименьшая ошибка, rule="1se" — самый большой alpha с ошибкой
    не больше минимума плюс стандартная ошибка этого минимума.

    При d = s² / (s² + alpha): остаток e = y − U·diag(d)·Uᵀy, рычаг hᵢ = Σⱼ Uᵢⱼ²dⱼ + 1/n,
    ошибка без i-й строки eᵢ / (1 − hᵢ). Части вне столбцов U не зависят от alpha
    и считаются один раз: при малом alpha вычитание 1 − hᵢ напрямую теряет точность.
    """
    n = len(target)
    projected = u.T @ target
    u_squared = u**2
    outside = target - u @ projected
    outside_leverage = 1.0 - 1.0 / n - u_squared.sum(axis=1)
    if rule not in ("min", "1se"):
        raise ValueError('rule должен быть "min" или "1se"')
    means, errors = [], []
    for alpha in alphas:
        shrink = alpha / (s2 + alpha)
        residual = u @ (shrink * projected) + outside
        one_minus_leverage = u_squared @ shrink + outside_leverage
        squared = (residual / one_minus_leverage) ** 2
        means.append(float(squared.mean()))
        errors.append(float(squared.std(ddof=1) / np.sqrt(n)))
    best = int(np.argmin(means))
    if rule == "min":
        return float(alphas[best])
    limit = means[best] + errors[best]
    return float(max(alpha for alpha, mean in zip(alphas, means) if mean <= limit))


def _check_switch_at(switch_at) -> int:
    if isinstance(switch_at, bool) or not isinstance(switch_at, Integral) or switch_at < 0:
        raise ValueError("switch_at должен быть целым >= 0")
    return int(switch_at)


def _loo_rule(alpha: str, n_features: int, switch_at: int) -> str:
    """Правило выбора alpha для набора из n_features признаков."""
    if alpha == "loo_schedule":
        return "1se" if n_features > switch_at else "min"
    return "1se" if alpha == "loo1se" else "min"


def _as_target(y, n_rows: int) -> np.ndarray:
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(target) != n_rows:
        raise ValueError("Длины X и y не совпадают")
    return target


SCALINGS = ("zscore", "pareto", "robust", "center")


def scale_channels(X_fit: np.ndarray, *others: np.ndarray, method: str = "zscore") -> list[np.ndarray]:
    """
    Нормировка по каналам: сдвиг и масштаб каждого столбца считаются по X_fit.
      zscore — минус среднее, деление на std (ddof=0), как StandardScaler;
      pareto — минус среднее, деление на корень из std;
      robust — минус среднее, деление на межквартильный размах (сдвиг для Ridge
               со свободным членом не важен, а быстрые формулы требуют
               центрирования средним);
      center — только минус среднее, масштаб исходный.
      range — минус среднее, деление на размах обучающего канала.
    Нулевой масштаб заменяется на 1.
    """
    shift, scale = X_fit.mean(axis=0), _channel_scale(X_fit, method)
    return [(X - shift) / scale for X in (X_fit, *others)]


def scale_only(X_fit: np.ndarray, *others: np.ndarray, method: str = "zscore") -> list[np.ndarray]:
    """
    Только деление каждого канала на его масштаб по X_fit, без вычитания среднего
    (масштабы те же, что в scale_channels). Нужна, когда после неё каналы
    нормируются по спектрам: при центрировании сумма и максимум спектра теряют смысл.
    """
    scale = _channel_scale(X_fit, method)
    return [X / scale for X in (X_fit, *others)]


def _channel_scale(X_fit: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        scale = X_fit.std(axis=0)
    elif method == "pareto":
        scale = np.sqrt(X_fit.std(axis=0))
    elif method == "robust":
        q25, q75 = np.percentile(X_fit, [25, 75], axis=0)
        scale = q75 - q25
    elif method == "center":
        scale = np.ones(X_fit.shape[1])
    elif method == "range":
        scale = np.ptp(X_fit, axis=0)
    else:
        raise ValueError(f"scaling должен быть одним из {(*SCALINGS, 'range')}")
    return np.where(scale == 0.0, 1.0, scale)


def _standardize(X_fit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Как StandardScaler: среднее и std (ddof=0) по X_fit, у константы масштаб 1."""
    return scale_channels(X_fit, *others, method="zscore")


def _check_scaling(scaling: str) -> str:
    if scaling not in (*SCALINGS, "range"):
        raise ValueError(f"scaling должен быть одним из {(*SCALINGS, 'range')}")
    return scaling


def _inverse(gram: np.ndarray, alpha: float, n_rows: int) -> np.ndarray:
    """(XᵀX + alpha·I)⁻¹ через Холецкого. При alpha=0 нужна невырожденная XᵀX."""
    p = gram.shape[0]
    if alpha == 0.0 and p >= n_rows:
        raise ValueError(
            f"МНК без штрафа не определён: признаков {p}, строк {n_rows}. "
            "После центрирования ранг XᵀX не больше n−1, решений бесконечно много. "
            "Нужен alpha > 0 или меньше признаков."
        )
    matrix = np.array(gram, dtype=float, order="F", copy=True)
    matrix[np.diag_indices(p)] += alpha
    factor, info = lapack.dpotrf(matrix, lower=0, overwrite_a=1)
    if info != 0:
        raise ValueError(
            f"XᵀX + alpha·I вырождена (dpotrf info={info}). "
            "Признаки линейно зависимы, нужен alpha > 0."
        )
    inverse, info = lapack.dpotri(factor, lower=0, overwrite_c=1)
    if info != 0:
        raise ValueError(f"не удалось обратить матрицу (dpotri info={info})")
    upper = np.triu(inverse)
    return upper + np.triu(inverse, 1).T


def _ordered(positions: np.ndarray, values: np.ndarray, names: list, descending: bool) -> list[int]:
    """Позиции по значению, при равенстве — по имени признака."""
    sign = -1.0 if descending else 1.0
    return sorted(range(len(positions)), key=lambda i: (sign * values[i], str(names[positions[i]])))


def _ranks(names: list, packets: list[list[int]], scores: dict[int, float]) -> pd.DataFrame:
    """Последний выброшенный пакет — группа 1, внутри пакета по убыванию score."""
    rows = []
    next_rank = 1
    for group, packet in enumerate(reversed(packets), 1):
        best_first = sorted(packet, key=lambda i: (-scores[i], str(names[i])))
        for offset, i in enumerate(best_first):
            rows.append(
                {
                    "Feature": names[i],
                    "Group_Rank": group,
                    "Individual_Rank": next_rank + offset,
                    "Score": scores[i],
                }
            )
        next_rank += len(packet)
    return pd.DataFrame(rows)


class _SelectorBase:
    ranking_: pd.DataFrame | None
    history_: pd.DataFrame | None
    elapsed_seconds_: float | None

    def ranking_frame(self) -> pd.DataFrame:
        if self.ranking_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        return self.ranking_.copy()

    def save_ranking(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_frame().to_csv(path, index=False)
        return path


class _Split:
    """Одна пара (обучение, оценка), масштабированная по обучающей части."""

    def __init__(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        scaling: str = "zscore",
        keep_fit: bool = False,
    ):
        X_fit, X_eval = scale_channels(X_fit, X_eval, method=scaling)
        y_mean = y_fit.mean()
        self.n_fit = len(y_fit)
        self.n_eval = len(y_eval)
        self.gram = X_fit.T @ X_fit
        self.xty = X_fit.T @ (y_fit - y_mean)
        self.X_eval = X_eval
        self.y_eval = y_eval - y_mean
        # Для двойственной формы нужны сами строки обучающей части.
        self.X_fit = X_fit if keep_fit else None
        self.y_fit = (y_fit - y_mean) if keep_fit else None

    def state(self, columns: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A⁻¹, β, остаток на оценочной части и X_eval·A⁻¹ для модели на columns."""
        inverse = _inverse(self.gram[np.ix_(columns, columns)], alpha, self.n_fit)
        beta = inverse @ self.xty[columns]
        X_eval = self.X_eval[:, columns]
        residual = self.y_eval - X_eval @ beta
        shift = X_eval @ inverse
        return inverse, beta, residual, shift

    def dual_parts(self, columns: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        diag(A⁻¹), β, остаток и X_eval·A⁻¹ через матрицу n × n вместо p × p (тождество Вудбери):
        A⁻¹ = (I − Xᵀ(XXᵀ + αI)⁻¹X) / α. Выгодно, когда признаков больше, чем строк:
        O(n²p + n³ + nₑnp) вместо O(p³ + nₑp²). Нужен keep_fit=True и alpha > 0.
        """
        if self.X_fit is None:
            raise ValueError("двойственная форма требует _Split(..., keep_fit=True)")
        if not alpha > 0.0:
            raise ValueError("двойственная форма требует alpha > 0")
        X_fit = self.X_fit[:, columns]
        kernel = X_fit @ X_fit.T
        kernel[np.diag_indices_from(kernel)] += alpha
        factor = lapack.dpotrf(np.asfortranarray(kernel), lower=0, overwrite_a=1)
        if factor[1] != 0:
            raise ValueError(f"XXᵀ + alpha·I не разложилась (dpotrf info={factor[1]})")
        solved, info = lapack.dpotrs(factor[0], X_fit, lower=0)
        if info != 0:
            raise ValueError(f"не удалось решить систему (dpotrs info={info})")
        # solved = (XXᵀ + αI)⁻¹X, столбцы — признаки.
        beta = solved.T @ self.y_fit
        diag = (1.0 - np.einsum("ij,ij->j", X_fit, solved)) / alpha
        X_eval = self.X_eval[:, columns]
        shift = (X_eval - (X_eval @ X_fit.T) @ solved) / alpha
        residual = self.y_eval - X_eval @ beta
        return diag, beta, residual, shift

    def sse_without_each(
        self, columns: np.ndarray, alpha: float, state=None, solver: str = "primal"
    ) -> tuple[np.ndarray, float]:
        """
        SSE на оценочной части для модели на columns и для каждой модели без одного признака.
        solver="dual" считает то же через матрицу n × n (dual_parts), результат совпадает до округления.
        """
        if solver == "dual":
            diag, beta, residual, shift = self.dual_parts(columns, alpha)
        elif solver == "primal":
            inverse, beta, residual, shift = self.state(columns, alpha) if state is None else state
            diag = np.diag(inverse)
        else:
            raise ValueError('solver должен быть "primal" или "dual"')
        coef = beta / diag
        base = float(residual @ residual)
        # Остаток без признака j: residual + shift[:, j]·coef[j].
        sse = base + 2.0 * coef * (residual @ shift) + coef**2 * np.einsum("ij,ij->j", shift, shift)
        return sse, base

    def sse_with_each(self, columns: np.ndarray, candidates: np.ndarray, alpha: float, state) -> np.ndarray:
        """
        SSE на оценочной части для модели на columns плюс один признак k из candidates.

        Окаймление A: s = Gₖₖ + alpha − gᵀA⁻¹g при g = G[columns, k], коэффициент
        нового признака γ = ((Xᵀy)ₖ − gᵀβ) / s, остальные β − A⁻¹g·γ. Остаток
        становится r − γ·(X_eval[:, k] − X_eval·A⁻¹·g). Точно, без обучения моделей.
        """
        inverse, beta, residual, shift = state
        border = self.gram[np.ix_(columns, candidates)]
        solved = inverse @ border
        schur = self.gram[candidates, candidates] + alpha - np.einsum("ij,ij->j", border, solved)
        gamma = (self.xty[candidates] - border.T @ beta) / schur
        direction = self.X_eval[:, candidates] - shift @ border
        base = float(residual @ residual)
        return base - 2.0 * gamma * (residual @ direction) + gamma**2 * np.einsum("ij,ij->j", direction, direction)

    def sse_without_groups(
        self, columns: np.ndarray, groups: list[np.ndarray], alpha: float, state=None
    ) -> tuple[np.ndarray, float]:
        """
        SSE без каждой группы признаков целиком. groups — позиции внутри columns.
        Блочная форма той же формулы: β без группы G = β − A⁻¹[:, G]·(A⁻¹[G, G])⁻¹·β_G,
        остаток r + M[:, G]·(A⁻¹[G, G])⁻¹·β_G. Точно, без обучения моделей.
        """
        inverse, beta, residual, shift = self.state(columns, alpha) if state is None else state
        base = float(residual @ residual)
        sse = np.empty(len(groups))
        for g, positions in enumerate(groups):
            coef = np.linalg.solve(inverse[np.ix_(positions, positions)], beta[positions])
            changed = residual + shift[:, positions] @ coef
            sse[g] = float(changed @ changed)
        return sse, base


class LinearPPOP(_SelectorBase):
    """
    ППОП на МНК или Ridge.

    Оценочная выборка: если в fit переданы X_eval, y_eval — MSE считается на них
    (например, на validation). Иначе — K-fold на переданном train, cv фолдов.
    Score признака — на сколько выросла MSE без него в момент выбрасывания.

    reshuffle=True: на каждой итерации свои случайные фолды. Иначе тысячи решений
    принимаются по одному и тому же разбиению, и отбор подстраивается под его шум.

    repeats > 1: K-fold повторяется repeats раз с разными перемешиваниями (seed, seed+1, …),
    ошибки суммируются по всем K·repeats разбиениям. Первое повторение совпадает с repeats=1.

    solver: "primal" — обращение p × p (как раньше), "dual" — матрица n × n по тождеству
    Вудбери, "auto" — dual, пока признаков больше, чем строк в обучающей части фолда.
    Результат тот же до ошибок округления; dual только для alpha > 0.
    """

    def __init__(
        self,
        alpha: float | str = 0.0,
        step=0.1,
        min_step: int = 1,
        cv: int = 5,
        random_state: int = 0,
        reshuffle: bool = False,
        switch_at: int = 500,
        scaling: str = "zscore",
        repeats: int = 1,
        solver: str = "primal",
    ):
        check_step(step)
        if cv < 2:
            raise ValueError("cv должен быть >= 2")
        if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
            raise ValueError("repeats должен быть целым >= 1")
        if solver not in ("primal", "dual", "auto"):
            raise ValueError('solver должен быть "primal", "dual" или "auto"')
        self.alpha = _check_alpha(alpha)
        self.switch_at = _check_switch_at(switch_at)
        self.step = step
        self.min_step = int(min_step)
        self.cv = int(cv)
        self.random_state = random_state
        self.reshuffle = bool(reshuffle)
        self.scaling = _check_scaling(scaling)
        self.repeats = int(repeats)
        self.solver = solver
        self.ranking_ = None
        self.history_ = None
        self.elapsed_seconds_ = None

    def _splits(self, X: np.ndarray, y: np.ndarray, X_eval, y_eval, seed=None) -> list[_Split]:
        keep_fit = self.solver != "primal"
        if X_eval is not None:
            evaluation = np.asarray(X_eval, dtype=float)
            return [_Split(X, y, evaluation, _as_target(y_eval, len(X_eval)), self.scaling, keep_fit)]
        seed = self.random_state if seed is None else seed
        splits = []
        for repeat in range(self.repeats):
            folds = KFold(n_splits=self.cv, shuffle=True, random_state=None if seed is None else seed + repeat)
            splits += [_Split(X[fit], y[fit], X[held], y[held], self.scaling, keep_fit) for fit, held in folds.split(X)]
        return splits

    def _solver_for(self, n_features: int, split: _Split) -> str:
        if self.solver == "auto":
            return "dual" if n_features > split.n_fit else "primal"
        return self.solver

    def fit(self, X: pd.DataFrame, y, X_eval: pd.DataFrame | None = None, y_eval=None) -> "LinearPPOP":
        if (X_eval is None) != (y_eval is None):
            raise ValueError("X_eval и y_eval нужно передавать вместе")
        if X_eval is not None and list(X_eval.columns) != list(X.columns):
            raise ValueError("Колонки train и оценочной выборки должны совпадать и идти в одном порядке")
        names: list[Hashable] = list(X.columns)
        started = time.perf_counter()
        values = X.to_numpy(dtype=float)
        target = _as_target(y, len(X))
        reshuffle = self.reshuffle and X_eval is None
        splits = None if reshuffle else self._splits(values, target, X_eval, y_eval)
        if isinstance(self.alpha, str):
            (scaled,) = scale_channels(values, method=self.scaling)
            centered = target - target.mean()

        remaining = np.arange(len(names))
        packets: list[list[int]] = []
        scores: dict[int, float] = {}
        history = []
        while remaining.size:
            step_started = time.perf_counter()
            if isinstance(self.alpha, str):
                u, s2, _ = ridge_spectrum(scaled[:, remaining])
                alpha = loo_alpha(u, s2, centered, rule=_loo_rule(self.alpha, remaining.size, self.switch_at))
            else:
                alpha = self.alpha
            sse = np.zeros(remaining.size)
            base = 0.0
            if reshuffle:
                # Новые фолды строятся только по оставшимся признакам.
                seed = None if self.random_state is None else self.random_state + len(history)
                current = self._splits(values[:, remaining], target, None, None, seed)
                columns = np.arange(remaining.size)
            else:
                current, columns = splits, remaining
            n_eval = sum(split.n_eval for split in current)
            for split in current:
                part, part_base = split.sse_without_each(columns, alpha, solver=self._solver_for(columns.size, split))
                sse += part
                base += part_base
            mse_full = base / n_eval
            increase = sse / n_eval - mse_full
            size = packet_size(remaining.size, self.step, self.min_step)
            # Без них MSE выросла меньше всего — они нужны меньше всего.
            drop = _ordered(remaining, increase, names, descending=False)[:size]
            for i in drop:
                scores[int(remaining[i])] = float(increase[i])
            packets.append([int(remaining[i]) for i in drop])
            history.append(
                {
                    "n_features": int(remaining.size),
                    "alpha": alpha,
                    "mse": mse_full,
                    "dropped": size,
                    "seconds": time.perf_counter() - step_started,
                }
            )
            print(
                f"PPOP {len(history)}: features {remaining.size}, alpha {alpha:g}, mse {mse_full:.6g}, "
                f"dropped {size}, step_s {history[-1]['seconds']:.1f}",
                flush=True,
            )
            remaining = np.delete(remaining, drop)

        self.ranking_ = _ranks(names, packets, scores)
        self.history_ = pd.DataFrame(history)
        self.elapsed_seconds_ = time.perf_counter() - started
        return self


class LinearGroupBSBS(_SelectorBase):
    """
    BSBS по группам признаков (например, окна соседних каналов спектра) на Ridge, K-fold на train.

    groups — метка группы для каждого столбца X. На итерации для каждой оставшейся группы
    точно считается ошибка модели без всей группы (Split.sse_without_groups), выбрасывается
    пакет групп с наименьшим приростом ошибки. Признаки группы получают её оценку и идут
    в рейтинге подряд (по имени). repeats — как в LinearPPOP.
    """

    def __init__(
        self,
        groups,
        alpha: float = 700.0,
        step=0.1,
        min_step: int = 1,
        cv: int = 5,
        random_state: int = 0,
        scaling: str = "zscore",
        repeats: int = 1,
    ):
        check_step(step)
        alpha = _check_alpha(alpha)
        if isinstance(alpha, str):
            raise ValueError("LinearGroupBSBS поддерживает только числовой alpha")
        self.groups = list(groups)
        self.alpha = alpha
        self.step = step
        self.min_step = int(min_step)
        self.cv = int(cv)
        self.random_state = random_state
        self.scaling = _check_scaling(scaling)
        self.repeats = int(repeats)
        self.ranking_ = None
        self.history_ = None
        self.elapsed_seconds_ = None

    def fit(self, X: pd.DataFrame, y) -> "LinearGroupBSBS":
        names: list[Hashable] = list(X.columns)
        if len(self.groups) != len(names):
            raise ValueError("groups должен содержать метку для каждого столбца X")
        started = time.perf_counter()
        values = X.to_numpy(dtype=float)
        target = _as_target(y, len(X))
        splits = LinearPPOP(alpha=self.alpha, cv=self.cv, random_state=self.random_state, scaling=self.scaling,
                            repeats=self.repeats)._splits(values, target, None, None)
        n_eval = sum(split.n_eval for split in splits)
        members: dict = {}
        for i, label in enumerate(self.groups):
            members.setdefault(label, []).append(i)
        remaining = list(members)
        packets: list[list[int]] = []
        scores: dict[int, float] = {}
        history = []
        while remaining:
            step_started = time.perf_counter()
            columns = np.concatenate([members[g] for g in remaining])
            offsets = np.cumsum([0] + [len(members[g]) for g in remaining])
            positions = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(remaining))]
            sse = np.zeros(len(remaining))
            base = 0.0
            for split in splits:
                part, part_base = split.sse_without_groups(columns, positions, self.alpha)
                sse += part
                base += part_base
            mse_full = base / n_eval
            increase = sse / n_eval - mse_full
            size = packet_size(len(remaining), self.step, self.min_step)
            order = sorted(range(len(remaining)), key=lambda i: (increase[i], str(remaining[i])))[:size]
            packet = []
            for i in order:
                for j in members[remaining[i]]:
                    scores[j] = float(increase[i])
                    packet.append(j)
            packets.append(packet)
            history.append(
                {
                    "n_groups": len(remaining),
                    "n_features": int(columns.size),
                    "alpha": self.alpha,
                    "mse": mse_full,
                    "dropped_groups": size,
                    "seconds": time.perf_counter() - step_started,
                }
            )
            print(
                f"GBSBS {len(history)}: groups {len(remaining)}, features {columns.size}, mse {mse_full:.6g}, "
                f"dropped {size}, step_s {history[-1]['seconds']:.1f}",
                flush=True,
            )
            dropped = set(order)
            remaining = [g for i, g in enumerate(remaining) if i not in dropped]

        self.ranking_ = _ranks(names, packets, scores)
        self.history_ = pd.DataFrame(history)
        self.elapsed_seconds_ = time.perf_counter() - started
        return self


class LinearFloatingBSBS(_SelectorBase):
    """
    BSBS с возвратом признаков (плавающий обратный отбор, Pudil et al. 1994) на Ridge, K-fold на train.

    Итерация:
      1. Для текущего набора S считаются ошибки без каждого признака (как в LinearPPOP)
         и с каждым ранее выброшенным признаком (Split.sse_with_each), из той же A⁻¹.
      2. Обмен: берётся выброшенный признак k, с которым ошибка меньше всего. Если она
         меньше ошибки S, для S + k считаются ошибки без каждого признака, кроме k, и
         лучший обмен S + k − j принимается, только если ошибка строго падает.
         Не больше max_swaps обменов подряд, потом пакет выбрасывается как обычно.
      3. Выбрасывается пакет признаков, без которых ошибка выросла меньше всего.
    Все ошибки — суммы по фолдам, решения точные, без обучения отдельных моделей.

    Наборы разных размеров не вложены: признак может уйти и вернуться. Поэтому для
    каждого k из ks пакет укорачивается так, чтобы набор прошёл ровно через k, и
    итоговый набор этого размера сохраняется в sets_[k]. ranking_ — порядок
    последнего выбрасывания (для совместимости), sets_ — сами отобранные наборы.
    При max_swaps=0 и без ks результат совпадает с LinearPPOP.
    """

    def __init__(
        self,
        alpha: float = 700.0,
        step=0.1,
        min_step: int = 1,
        cv: int = 5,
        random_state: int = 0,
        scaling: str = "zscore",
        max_swaps: int = 5,
        ks: list[int] | None = None,
    ):
        check_step(step)
        if cv < 2:
            raise ValueError("cv должен быть >= 2")
        alpha = _check_alpha(alpha)
        if isinstance(alpha, str):
            raise ValueError("LinearFloatingBSBS поддерживает только числовой alpha")
        if isinstance(max_swaps, bool) or not isinstance(max_swaps, Integral) or max_swaps < 0:
            raise ValueError("max_swaps должен быть целым >= 0")
        self.alpha = alpha
        self.step = step
        self.min_step = int(min_step)
        self.cv = int(cv)
        self.random_state = random_state
        self.scaling = _check_scaling(scaling)
        self.max_swaps = int(max_swaps)
        self.ks = sorted({int(k) for k in ks}) if ks else []
        self.ranking_ = None
        self.history_ = None
        self.swaps_ = None
        self.sets_ = None
        self.elapsed_seconds_ = None

    def fit(self, X: pd.DataFrame, y) -> "LinearFloatingBSBS":
        names: list[Hashable] = list(X.columns)
        started = time.perf_counter()
        values = X.to_numpy(dtype=float)
        target = _as_target(y, len(X))
        folds = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        splits = [_Split(values[fit], target[fit], values[held], target[held], self.scaling) for fit, held in folds.split(values)]
        n_eval = sum(split.n_eval for split in splits)
        alpha = self.alpha

        def evaluate(columns: np.ndarray, candidates: np.ndarray | None):
            without = np.zeros(columns.size)
            added = None if candidates is None else np.zeros(candidates.size)
            base = 0.0
            for split in splits:
                state = split.state(columns, alpha)
                part, part_base = split.sse_without_each(columns, alpha, state)
                without += part
                base += part_base
                if candidates is not None:
                    added += split.sse_with_each(columns, candidates, alpha, state)
            return without, added, base

        # По возрастанию: targets[-1] — ближайший размер ниже текущего.
        targets = [k for k in self.ks if 0 < k < len(names)]
        active = np.arange(len(names))
        removed = np.array([], dtype=int)
        last_drop: dict[int, tuple[int, float]] = {}
        sets: dict[int, list] = {}
        history, swaps = [], []
        iteration = 0
        while active.size:
            step_started = time.perf_counter()
            done_swaps = 0
            while True:
                check = removed if removed.size and done_swaps < self.max_swaps else None
                without, added, base = evaluate(active, check)
                if check is None:
                    break
                best = _ordered(removed, added, names, descending=False)[0]
                if not added[best] < base:
                    break
                k = int(removed[best])
                candidate = np.append(active, k)
                after, _, _ = evaluate(candidate, None)
                after[-1] = np.inf
                worst = _ordered(candidate, after, names, descending=False)[0]
                # Обмен — только при строгом падении ошибки, иначе возможен цикл.
                if not after[worst] < base * (1.0 - 1e-12):
                    break
                j = int(candidate[worst])
                active = np.sort(np.delete(candidate, worst))
                removed = np.sort(np.append(removed[removed != k], j))
                last_drop.pop(k, None)
                last_drop[j] = (iteration, float(after[worst] / n_eval - base / n_eval))
                swaps.append(
                    {
                        "iteration": iteration,
                        "n_features": int(active.size),
                        "added": names[k],
                        "removed": names[j],
                        "mse_before": base / n_eval,
                        "mse_after": float(after[worst] / n_eval),
                        "features": [names[i] for i in active],
                    }
                )
                done_swaps += 1
            if targets and active.size == targets[-1]:
                sets[targets.pop()] = [names[i] for i in active]
            mse_full = base / n_eval
            increase = without / n_eval - mse_full
            size = packet_size(active.size, self.step, self.min_step)
            if targets and active.size - size < targets[-1]:
                size = active.size - targets[-1]
            drop = _ordered(active, increase, names, descending=False)[:size]
            for i in drop:
                last_drop[int(active[i])] = (iteration, float(increase[i]))
            history.append(
                {
                    "n_features": int(active.size),
                    "alpha": alpha,
                    "mse": mse_full,
                    "dropped": size,
                    "swaps": done_swaps,
                    "seconds": time.perf_counter() - step_started,
                }
            )
            print(
                f"FBSBS {len(history)}: features {active.size}, mse {mse_full:.6g}, swaps {done_swaps}, "
                f"dropped {size}, step_s {history[-1]['seconds']:.1f}",
                flush=True,
            )
            removed = np.sort(np.append(removed, active[drop]))
            active = np.delete(active, drop)
            iteration += 1

        packets = [[] for _ in range(iteration)]
        for i, (when, _) in last_drop.items():
            packets[when].append(i)
        scores = {i: score for i, (_, score) in last_drop.items()}
        self.ranking_ = _ranks(names, [packet for packet in packets if packet], scores)
        self.history_ = pd.DataFrame(history)
        self.swaps_ = pd.DataFrame(swaps, columns=["iteration", "n_features", "added", "removed", "mse_before", "mse_after", "features"])
        self.sets_ = sets
        self.elapsed_seconds_ = time.perf_counter() - started
        return self


class LinearRFE(_SelectorBase):
    """
    RFE на МНК или Ridge. Важность — |β| после масштабирования train (scaling).

    step=1 — классический RFE. Обратная матрица после выбрасывания обновляется
    формулой понижения ранга, раз в refresh_every выброшенных признаков
    пересчитывается заново, чтобы не копилась ошибка округления.
    При alpha="loo_schedule" матрица пересчитывается и в момент, когда признаков
    осталось switch_at или меньше: с этого шага alpha берётся по минимуму LOO.
    """

    def __init__(
        self,
        alpha: float | str = 700.0,
        step=1,
        min_step: int = 1,
        refresh_every: int = 256,
        switch_at: int = 500,
        scaling: str = "zscore",
    ):
        check_step(step)
        if refresh_every < 1:
            raise ValueError("refresh_every должен быть >= 1")
        self.scaling = _check_scaling(scaling)
        self.alpha = _check_alpha(alpha)
        self.switch_at = _check_switch_at(switch_at)
        self.step = step
        self.min_step = int(min_step)
        self.refresh_every = int(refresh_every)
        self.ranking_ = None
        self.history_ = None
        self.elapsed_seconds_ = None

    def fit(self, X: pd.DataFrame, y) -> "LinearRFE":
        names: list[Hashable] = list(X.columns)
        started = time.perf_counter()
        (scaled,) = scale_channels(X.to_numpy(dtype=float), method=self.scaling)
        target = _as_target(y, len(X))
        gram = scaled.T @ scaled
        xty = scaled.T @ (target - target.mean())
        n_rows = len(target)

        def rule_for(n_features: int) -> str | None:
            if not isinstance(self.alpha, str):
                return None
            return _loo_rule(self.alpha, n_features, self.switch_at)

        def choose_alpha(columns: np.ndarray) -> float:
            if not isinstance(self.alpha, str):
                return self.alpha
            u, s2, _ = ridge_spectrum(scaled[:, columns])
            return loo_alpha(u, s2, target - target.mean(), rule=rule_for(columns.size))

        # block — признаки, по которым построена inverse; active — ещё не выброшенные из block.
        block = np.arange(len(names))
        alpha = choose_alpha(block)
        rule = rule_for(block.size)
        inverse = _inverse(gram, alpha, n_rows)
        active = np.ones(block.size, dtype=bool)
        since_refresh = 0
        packets: list[list[int]] = []
        scores: dict[int, float] = {}
        history = []
        while active.any():
            step_started = time.perf_counter()
            # У выброшенных строки и столбцы inverse обнулены, их β тоже 0.
            used_alpha = alpha
            beta = inverse @ xty[block]
            positions = np.flatnonzero(active)
            weights = np.abs(beta[positions])
            size = packet_size(positions.size, self.step, self.min_step)
            drop = [positions[i] for i in _ordered(block[positions], weights, names, descending=False)[:size]]
            for j in drop:
                scores[int(block[j])] = float(abs(beta[j]))
            packets.append([int(block[j]) for j in drop])
            active[drop] = False
            since_refresh += size
            if active.any():
                switched = rule_for(int(active.sum())) != rule
                if since_refresh >= self.refresh_every or switched:
                    block = block[active]
                    active = np.ones(block.size, dtype=bool)
                    alpha = choose_alpha(block)
                    rule = rule_for(block.size)
                    inverse = _inverse(gram[np.ix_(block, block)], alpha, n_rows)
                    since_refresh = 0
                else:
                    inverse = self._downdate(inverse, drop)
            history.append(
                {
                    "n_features": int(positions.size),
                    "alpha": used_alpha,
                    "dropped": size,
                    "smallest_abs_coef": float(weights.min()),
                    "seconds": time.perf_counter() - step_started,
                }
            )
            if len(history) % 250 == 1 or not active.any() or size > 1:
                print(
                    f"RFE {len(history)}: features {positions.size}, alpha {used_alpha:g}, dropped {size}, "
                    f"smallest_abs_coef {weights.min():.6g}, step_s {history[-1]['seconds']:.2f}",
                    flush=True,
                )

        self.ranking_ = _ranks(names, packets, scores)
        self.history_ = pd.DataFrame(history)
        self.elapsed_seconds_ = time.perf_counter() - started
        return self

    @staticmethod
    def _downdate(inverse: np.ndarray, drop: list[int]) -> np.ndarray:
        """A⁻¹ без признаков drop: A⁻¹ − A⁻¹[:, D]·(A⁻¹[D, D])⁻¹·A⁻¹[D, :]. Строки D становятся нулевыми."""
        if len(drop) == 1:
            j = drop[0]
            column = inverse[:, j].copy()
            # inverse симметрична, транспонированный вид F-непрерывен — dger правит её на месте.
            updated = dger(-1.0 / column[j], column, column, a=inverse.T, overwrite_a=1)
            return inverse if np.shares_memory(updated, inverse) else np.ascontiguousarray(updated.T)
        columns = inverse[:, drop]
        return inverse - columns @ np.linalg.solve(inverse[np.ix_(drop, drop)], columns.T)
