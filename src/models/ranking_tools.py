"""
Преобразования готовых рейтингов каналов 6P.

spaced_ranking — прореживание: идём по рейтингу и берём канал, только если он
дальше min_distance_nm от всех уже взятых на этом проходе (возбуждение и
испускание в нм). Пропущенные каналы не теряются: следующий проход идёт по ним
с вдвое меньшим расстоянием, а когда оно меньше 1 нм — без ограничения.
Так первые k каналов не повторяют друг друга, а полный рейтинг по-прежнему
содержит все каналы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.visualization.eem_selection import channel_coordinates


def spaced_order(features: list[str], min_distance_nm: float) -> list[str]:
    if min_distance_nm < 0:
        raise ValueError("min_distance_nm должен быть >= 0")
    coords = np.array([channel_coordinates(f) for f in features], dtype=float)
    order: list[int] = []
    taken = np.zeros(len(features), dtype=bool)
    distance = float(min_distance_nm)
    while not taken.all():
        chosen: list[int] = []
        for i in np.flatnonzero(~taken):
            if distance > 0 and chosen:
                gap = np.sqrt(((coords[chosen] - coords[i]) ** 2).sum(axis=1)).min()
                if gap < distance:
                    continue
            chosen.append(int(i))
        order.extend(chosen)
        taken[chosen] = True
        distance /= 2.0
        if distance < 1.0:
            distance = 0.0
    return [features[i] for i in order]


def spaced_ranking(ranking: pd.DataFrame, min_distance_nm: float) -> pd.DataFrame:
    """Рейтинг того же формата: Feature, Group_Rank, Individual_Rank, Score (исходный ранг)."""
    ordered = ranking.sort_values("Individual_Rank")
    original = dict(zip(ordered["Feature"].astype(str), ordered["Individual_Rank"]))
    features = spaced_order(list(original), min_distance_nm)
    ranks = np.arange(1, len(features) + 1)
    return pd.DataFrame(
        {
            "Feature": features,
            "Group_Rank": ranks,
            "Individual_Rank": ranks,
            "Score": [-original[f] for f in features],
        }
    )


def aggregate_rankings(rankings: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Общий рейтинг по нескольким: канал ставится по среднему Individual_Rank
    (при равенстве — по имени). Score — минус средний ранг.
    """
    if not rankings:
        raise ValueError("нет рейтингов")
    frames = [r.set_index(r["Feature"].astype(str))["Individual_Rank"] for r in rankings]
    features = set(frames[0].index)
    if any(set(f.index) != features for f in frames):
        raise ValueError("рейтинги должны содержать одни и те же каналы")
    mean_rank = pd.concat(frames, axis=1).mean(axis=1)
    ordered = sorted(mean_rank.index, key=lambda f: (mean_rank[f], f))
    ranks = np.arange(1, len(ordered) + 1)
    return pd.DataFrame(
        {
            "Feature": ordered,
            "Group_Rank": ranks,
            "Individual_Rank": ranks,
            "Score": [-float(mean_rank[f]) for f in ordered],
        }
    )
