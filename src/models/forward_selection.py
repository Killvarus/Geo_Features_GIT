"""
Прямой пакетный отбор признаков гребневой регрессией.

На каждом шаге к уже отобранным каналам по очереди добавляется один кандидат.
Ridge учится на train, качество — MSE на validation. В пакет попадают кандидаты
с наименьшей ошибкой. Ранг 1 — канал, добавленный раньше всех.
"""
from __future__ import annotations

import time
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class ForwardBatchRidgeSelection:
    def __init__(self, alpha: float = 700.0, n_features_to_add: int = 1500):
        if n_features_to_add < 1:
            raise ValueError("n_features_to_add must be >= 1")
        self.alpha = float(alpha)
        self.n_features_to_add = int(n_features_to_add)
        self.ranking_: dict | None = None
        self.scores_: dict[Hashable, float] = {}
        self.elapsed_seconds_: float | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: pd.Series | np.ndarray,
    ) -> "ForwardBatchRidgeSelection":
        features = list(X_train.columns)
        if list(X_valid.columns) != features:
            raise ValueError("Колонки train и validation должны совпадать и идти в одном порядке")
        y_train = np.asarray(y_train, dtype=float).reshape(-1)
        y_valid = np.asarray(y_valid, dtype=float).reshape(-1)
        if len(y_train) != len(X_train) or len(y_valid) != len(X_valid):
            raise ValueError("Длины X и y не совпадают")

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train)
        valid_scaled = scaler.transform(X_valid)
        train = pd.DataFrame(train_scaled, columns=features)
        valid = pd.DataFrame(valid_scaled, columns=features)

        selected: list[Hashable] = []
        remaining = features.copy()
        group_rank = {feature: 0 for feature in features}
        individual_rank = {feature: 0 for feature in features}
        self.scores_ = {}
        next_rank = 1
        group = 1
        started = time.perf_counter()

        while remaining:
            step_started = time.perf_counter()
            n_candidates = len(remaining)
            scores = {
                feature: self._validation_mse(train, y_train, valid, y_valid, selected + [feature])
                for feature in remaining
            }
            ordered = sorted(remaining, key=lambda feature: (scores[feature], str(feature)))
            take = ordered[: min(self.n_features_to_add, len(ordered))]
            for offset, feature in enumerate(take):
                individual_rank[feature] = next_rank + offset
                group_rank[feature] = group
                self.scores_[feature] = scores[feature]
            selected.extend(take)
            taken = set(take)
            remaining = [feature for feature in remaining if feature not in taken]
            print(
                f"step {group}: scored {n_candidates}, added {len(take)}, "
                f"selected {len(selected)}, remaining {len(remaining)}, "
                f"best_mse {scores[take[0]]:.6g}, "
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

    def _validation_mse(
        self,
        train: pd.DataFrame,
        y_train: np.ndarray,
        valid: pd.DataFrame,
        y_valid: np.ndarray,
        columns: list[Hashable],
    ) -> float:
        model = Ridge(alpha=self.alpha, fit_intercept=True, solver="lsqr")
        model.fit(train.loc[:, columns], y_train)
        prediction = model.predict(valid.loc[:, columns])
        return float(mean_squared_error(y_valid, prediction))

    def ranking_frame(self) -> pd.DataFrame:
        if self.ranking_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        frame = pd.DataFrame(
            {
                "Feature": list(self.scores_),
                "Group_Rank": [self.ranking_["group_rank"][feature] for feature in self.scores_],
                "Individual_Rank": [self.ranking_["individual_rank"][feature] for feature in self.scores_],
                "Score": [self.scores_[feature] for feature in self.scores_],
            }
        )
        return frame.sort_values("Individual_Rank").reset_index(drop=True)
