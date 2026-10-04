"""
BSBS: Batch Sequential Backward Selection.

Packet wrapper for feature ranking. At each step every remaining feature is
scored by an estimator trained without that feature. The m features with the
worst scores are removed together, not one by one, and the step repeats.
Features removed earlier receive the better ranks: a low score means the model
got worse without that feature, so the feature is treated as more informative.

The project class with this procedure is TrueBackwardFeatureSelection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


class BatchSequentialBackwardSelection:
    def __init__(
        self,
        estimator,
        n_features_to_drop: int = 100,
        cv: int | None = 5,
        scoring: str = "neg_mean_squared_error",
    ):
        if n_features_to_drop < 1:
            raise ValueError("n_features_to_drop must be >= 1")
        self.estimator = estimator
        self.n_features_to_drop = n_features_to_drop
        self.cv = cv
        self.scoring = scoring
        self.ranking_: dict | None = None
        self.scores_: dict[Hashable, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> "BatchSequentialBackwardSelection":
        features = list(X.columns)
        current = features.copy()
        group_rank = {feature: 0 for feature in features}
        individual_rank = {feature: 0 for feature in features}
        self.scores_ = {feature: 0.0 for feature in features}

        next_rank = 1
        group = 1

        while len(current) > self.n_features_to_drop:
            scores = self._score_without_each(X[current], y)
            self.scores_.update(scores)
            ordered = sorted(scores, key=scores.get)

            for offset, feature in enumerate(ordered):
                individual_rank[feature] = next_rank + offset
            next_rank += self.n_features_to_drop

            dropped = ordered[: self.n_features_to_drop]
            for feature in dropped:
                group_rank[feature] = group
            current = [feature for feature in current if feature not in set(dropped)]
            group += 1

        if current:
            scores = self._score_without_each(X[current], y)
            self.scores_.update(scores)
            ordered = sorted(scores, key=scores.get)
            for offset, feature in enumerate(ordered):
                individual_rank[feature] = next_rank + offset
                group_rank[feature] = group

        self.ranking_ = {
            "group_rank": group_rank,
            "individual_rank": individual_rank,
            "scores": self.scores_,
        }
        return self

    def _score_without_each(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> dict[Hashable, float]:
        scores = {}
        for feature in X.columns:
            reduced = X.drop(columns=[feature])
            scores[feature] = self._score_model(reduced, y)
        return scores

    def _score_model(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> float:
        model = clone(self.estimator)
        if self.cv is not None:
            folds = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
            return float(np.mean(folds))
        model.fit(X, y)
        prediction = model.predict(X)
        return -float(mean_squared_error(y, prediction))

    def ranking_frame(self) -> pd.DataFrame:
        if self.ranking_ is None:
            raise RuntimeError("Call fit() first")
        frame = pd.DataFrame(
            {
                "Feature": list(self.scores_),
                "Group_Rank": [self.ranking_["group_rank"][feature] for feature in self.scores_],
                "Individual_Rank": [self.ranking_["individual_rank"][feature] for feature in self.scores_],
                "Score": [self.scores_[feature] for feature in self.scores_],
            }
        )
        return frame.sort_values("Individual_Rank").reset_index(drop=True)

    def save_ranking(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ranking_frame().to_csv(path, index=False)
        return path


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n_samples, n_features = 80, 12
    signal = rng.normal(size=(n_samples, 3))
    noise = rng.normal(size=(n_samples, n_features - 3))
    values = np.column_stack([signal, noise])
    target = signal @ np.array([1.5, -0.7, 0.4]) + rng.normal(scale=0.1, size=n_samples)
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(n_features)])

    selector = BatchSequentialBackwardSelection(
        estimator=LinearRegression(),
        n_features_to_drop=4,
        cv=None,
    )
    selector.fit(frame, pd.Series(target, name="y"))
    print(selector.ranking_frame().to_string(index=False))
