import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.models.feature_selection import TrueBackwardFeatureSelection


def _splits():
    rng = np.random.default_rng(0)
    n = 50
    signal_train = rng.normal(size=n)
    decoy_train = signal_train + rng.normal(scale=0.01, size=n)
    noise_train = rng.normal(size=n)
    y_train = signal_train.copy()

    signal_valid = rng.normal(size=n)
    decoy_valid = rng.normal(size=n)
    noise_valid = rng.normal(size=n)
    y_valid = signal_valid.copy()

    X_train = pd.DataFrame({"signal": signal_train, "decoy": decoy_train, "noise": noise_train})
    X_valid = pd.DataFrame({"signal": signal_valid, "decoy": decoy_valid, "noise": noise_valid})
    return X_train, pd.Series(y_train, name="y"), X_valid, pd.Series(y_valid, name="y")


def test_score_is_negative_mse_on_validation_not_train():
    X_train, y_train, X_valid, y_valid = _splits()
    selector = TrueBackwardFeatureSelection(LinearRegression(), n_features_to_drop=1, cv=None)
    selector.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)

    reduced = X_train.drop(columns=["signal"])
    model = LinearRegression().fit(reduced, y_train)
    expected_valid = -mean_squared_error(y_valid, model.predict(X_valid.drop(columns=["signal"])))
    expected_train = -mean_squared_error(y_train, model.predict(reduced))

    score = selector.get_ranking_df().set_index("Feature").loc["signal", "Score"]
    assert abs(score - expected_valid) < 1e-8
    assert abs(expected_valid) > 0.1
    assert abs(score - expected_train) > 1e-3
    assert score < 0.0


def test_signal_is_ranked_ahead_of_decoy_by_validation_mse():
    X_train, y_train, X_valid, y_valid = _splits()
    selector = TrueBackwardFeatureSelection(LinearRegression(), n_features_to_drop=1, cv=None)
    selector.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
    order = selector.get_ranking_df()["Feature"].tolist()
    assert order[0] == "signal"
    assert order.index("signal") < order.index("decoy")


def test_without_validation_score_stays_on_train():
    X_train, y_train, _, _ = _splits()
    selector = TrueBackwardFeatureSelection(LinearRegression(), n_features_to_drop=1, cv=None)
    selector.fit(X_train, y_train)
    reduced = X_train.drop(columns=["noise"])
    model = LinearRegression().fit(reduced, y_train)
    expected = -mean_squared_error(y_train, model.predict(reduced))
    score = selector.get_ranking_df().set_index("Feature").loc["noise", "Score"]
    assert abs(score - expected) < 1e-8
    assert selector.elapsed_seconds_ > 0.0
