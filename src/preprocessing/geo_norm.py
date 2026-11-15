"""
Нормировки входов обратной задачи МТЗ по каждому примеру отдельно — аналог нормировок
спектров для 6P.

Признак — {RE|IM}{YX|XY|HX}{частота}_{пикет}: действительная и мнимая части компонент
импеданса Z_YX и Z_XY и вектора Визе (HX) на 13 частотах в 31 пикете профиля.
Профиль — 31 значение одной части одной компоненты на одной частоте вдоль пикетов;
блок — все частоты и пикеты одной части одной компоненты.

Компоненты имеют разные единицы и масштабы (стандартное отклонение каналов импеданса
от 10⁻⁵ до 10⁻², вектора Визе — до 3), поэтому нормировки «по всему примеру сразу»,
как площадь или SNV спектра, здесь бессмысленны. Нормировки работают внутри профиля
или блока:

  raw            — без изменений;
  profile_center — минус среднее профиля: убирает общий уровень (отклик одномерного
                   фона), оставляет латеральную форму аномалии;
  profile_scale  — деление профиля на его среднеквадратичное значение: убирает
                   масштаб, сохраняет знак и форму;
  profile_snv    — минус среднее и деление на стандартное отклонение профиля;
  block_scale    — деление блока (часть × компонента) на его среднеквадратичное значение:
                   выравнивает масштабы компонент, но сохраняет соотношения частот и пикетов.

amp_phase — преобразование, а не нормировка: для импеданса (YX, XY) пара RE/IM
заменяется логарифмом модуля и фазой (log ρ_a отличается от 2·log|Z| только сдвигом на
каждой частоте, который убирает z-score каналов). Вектор Визе остаётся в виде RE/IM:
его фаза рвётся при переходе через отрицательную действительную полуось. Имена
признаков меняются (AMP…/PHS…), поэтому отбор по ним нельзя проверять на исходных каналах.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

GEO_SAMPLE_NORMS = ("raw", "profile_center", "profile_scale", "profile_snv", "block_scale")
GEO_TRANSFORMS = ("amp_phase",)
_PATTERN = re.compile(r"^(RE|IM)(YX|XY|HX)(\d+)_(\d+)$")


def parse_feature(name: str) -> tuple[str, str, int, int]:
    """(часть, компонента, частота, пикет) по имени признака."""
    match = _PATTERN.match(str(name))
    if match is None:
        raise ValueError(f"не признак МТЗ: {name}")
    return match[1], match[2], int(match[3]), int(match[4])


def is_feature(name: str) -> bool:
    return _PATTERN.match(str(name)) is not None


def _groups(names: Sequence[str], key) -> list[np.ndarray]:
    groups: dict = {}
    for i, name in enumerate(names):
        groups.setdefault(key(parse_feature(name)), []).append(i)
    return [np.asarray(index) for index in groups.values()]


def profile_groups(names: Sequence[str]) -> list[np.ndarray]:
    """Индексы столбцов каждого профиля (часть, компонента, частота)."""
    return _groups(names, lambda f: f[:3])


def block_groups(names: Sequence[str]) -> list[np.ndarray]:
    """Индексы столбцов каждого блока (часть, компонента)."""
    return _groups(names, lambda f: f[:2])


def _rms(values: np.ndarray, what: str) -> np.ndarray:
    rms = np.sqrt((values**2).mean(axis=1, keepdims=True))
    bad = int((rms == 0).sum())
    if bad:
        raise ValueError(f"{what}: нулевое среднеквадратичное значение у {bad} примеров")
    return rms


def normalize_samples(X: np.ndarray, names: Sequence[str], method: str) -> np.ndarray:
    """X — (примеры, признаки) в порядке names. Возвращает новую матрицу."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("число столбцов X должно совпадать с числом имён")
    out = X.copy()
    if method == "raw":
        return out
    if method == "block_scale":
        for index in block_groups(names):
            out[:, index] = X[:, index] / _rms(X[:, index], "блок")
        return out
    if method not in GEO_SAMPLE_NORMS:
        raise ValueError(f"нормировка должна быть одной из {GEO_SAMPLE_NORMS}")
    for index in profile_groups(names):
        part = X[:, index]
        if method == "profile_center":
            out[:, index] = part - part.mean(axis=1, keepdims=True)
        elif method == "profile_scale":
            out[:, index] = part / _rms(part, "профиль")
        else:
            std = part.std(axis=1, keepdims=True)
            bad = int((std == 0).sum())
            if bad:
                raise ValueError(f"профиль: нулевое стандартное отклонение у {bad} примеров")
            out[:, index] = (part - part.mean(axis=1, keepdims=True)) / std
    return out


def amp_phase(X: np.ndarray, names: Sequence[str], floor: float = 1e-12) -> tuple[np.ndarray, list[str]]:
    """Импеданс: RE/IM → log|Z| (AMP) и фаза atan2(IM, RE) (PHS); вектор Визе без изменений."""
    X = np.asarray(X, dtype=float)
    index = {name: i for i, name in enumerate(names)}
    out = X.copy()
    new_names = list(names)
    for i, name in enumerate(names):
        part, component, frequency, picket = parse_feature(name)
        if component == "HX" or part != "RE":
            continue
        partner = f"IM{component}{frequency}_{picket}"
        if partner not in index:
            raise ValueError(f"нет пары для {name}: {partner}")
        j = index[partner]
        real, imag = X[:, i], X[:, j]
        out[:, i] = np.log(np.maximum(np.hypot(real, imag), floor))
        out[:, j] = np.arctan2(imag, real)
        new_names[i] = f"AMP{component}{frequency}_{picket}"
        new_names[j] = f"PHS{component}{frequency}_{picket}"
    return out, new_names
