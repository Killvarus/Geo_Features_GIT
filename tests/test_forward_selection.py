import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.models.forward_selection import ForwardBatchRidgeSelection


def _manual_mse(X_train, y_train, X_valid, y_valid, columns, alpha):
    scaler = StandardScaler().fit(X_train[columns])
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(scaler.transform(X_train[columns]), y_train)
    prediction = model.predict(scaler.transform(X_valid[columns]))
    return float(mean_squared_error(y_valid, prediction))


def test_score_is_validation_mse_of_ridge():
    rng = np.random.default_rng(0)
    n = 60
    signal_train = rng.normal(size=n)
    signal_valid = rng.normal(size=n)
    X_train = pd.DataFrame({"signal": signal_train, "noise": rng.normal(size=n)})
    X_valid = pd.DataFrame({"signal": signal_valid, "noise": rng.normal(size=n)})
    y_train = 3.0 * signal_train
    y_valid = 3.0 * signal_valid
    alpha = 700.0
    selector = ForwardBatchRidgeSelection(alpha=alpha, n_features_to_add=1)
    selector.fit(X_train, y_train, X_valid, y_valid)

    expected = _manual_mse(X_train, y_train, X_valid, y_valid, ["signal"], alpha)
    score = selector.ranking_frame().set_index("Feature").loc["signal", "Score"]
    train_mse = _manual_mse(X_train, y_train, X_train, y_train, ["signal"], alpha)
    assert abs(score - expected) < 1e-6
    assert abs(score - train_mse) > 1e-3
    assert score > 0.0
    assert abs(score - r2_score(y_valid, np.zeros_like(y_valid))) > 0.5
    assert selector.elapsed_seconds_ > 0.0


def test_alpha_zero_score_is_unpenalized_validation_mse():
    rng = np.random.default_rng(4)
    n = 40
    signal_train = rng.normal(size=n)
    signal_valid = rng.normal(size=n)
    X_train = pd.DataFrame({"signal": signal_train, "noise": rng.normal(size=n)})
    X_valid = pd.DataFrame({"signal": signal_valid, "noise": rng.normal(size=n)})
    y_train = 2.0 * signal_train
    y_valid = 2.0 * signal_valid
    selector = ForwardBatchRidgeSelection(alpha=0.0, n_features_to_add=1)
    selector.fit(X_train, y_train, X_valid, y_valid)
    expected = _manual_mse(X_train, y_train, X_valid, y_valid, ["signal"], 0.0)
    score = selector.ranking_frame().set_index("Feature").loc["signal", "Score"]
    penalized = _manual_mse(X_train, y_train, X_valid, y_valid, ["signal"], 700.0)
    assert abs(score - expected) < 1e-6
    assert abs(score - penalized) > 1e-6


def test_adds_new_signal_before_redundant_copy_when_penalty_is_small():
    rng = np.random.default_rng(1)
    n = 80
    a_train, d_train = rng.normal(size=(2, n))
    a_valid, d_valid = rng.normal(size=(2, n))
    X_train = pd.DataFrame({
        "a": a_train,
        "a_copy": a_train.copy(),
        "d": d_train,
        "noise": rng.normal(size=n),
    })
    X_valid = pd.DataFrame({
        "a": a_valid,
        "a_copy": a_valid.copy(),
        "d": d_valid,
        "noise": rng.normal(size=n),
    })
    y_train = 3.0 * a_train + 2.0 * d_train
    y_valid = 3.0 * a_valid + 2.0 * d_valid
    selector = ForwardBatchRidgeSelection(alpha=1.0, n_features_to_add=1)
    selector.fit(X_train, y_train, X_valid, y_valid)
    order = selector.ranking_frame()["Feature"].tolist()
    assert order[0] == "a"
    assert order[1] == "d"
    assert order.index("d") < order.index("a_copy")


def test_packet_groups_follow_addition_order():
    rng = np.random.default_rng(2)
    n = 70
    a_train, d_train = rng.normal(size=(2, n))
    a_valid, d_valid = rng.normal(size=(2, n))
    X_train = pd.DataFrame({
        "a": 4.0 * a_train,
        "d": d_train,
        "n1": rng.normal(size=n),
        "n2": rng.normal(size=n),
    })
    X_valid = pd.DataFrame({
        "a": 4.0 * a_valid,
        "d": d_valid,
        "n1": rng.normal(size=n),
        "n2": rng.normal(size=n),
    })
    y_train = a_train + 0.5 * d_train
    y_valid = a_valid + 0.5 * d_valid
    selector = ForwardBatchRidgeSelection(alpha=700.0, n_features_to_add=2)
    selector.fit(X_train, y_train, X_valid, y_valid)
    frame = selector.ranking_frame()
    first = set(frame.loc[frame["Group_Rank"] == 1, "Feature"])
    assert first == {"a", "d"}
    assert frame.loc[frame["Feature"] == "a", "Individual_Rank"].item() == 1
    for _, part in frame.groupby("Group_Rank"):
        part = part.sort_values("Individual_Rank")
        assert part["Score"].tolist() == sorted(part["Score"].tolist())
        ranks = part["Individual_Rank"].tolist()
        assert ranks == list(range(ranks[0], ranks[-1] + 1))


def test_train_only_decoy_loses_to_signal_on_validation():
    rng = np.random.default_rng(3)
    n = 50
    signal_train = rng.normal(size=n)
    signal_valid = rng.normal(size=n)
    X_train = pd.DataFrame({
        "signal": signal_train,
        "decoy": signal_train.copy(),
    })
    X_valid = pd.DataFrame({
        "signal": signal_valid,
        "decoy": rng.normal(size=n),
    })
    y_train = signal_train.copy()
    y_valid = signal_valid.copy()
    selector = ForwardBatchRidgeSelection(alpha=700.0, n_features_to_add=1)
    selector.fit(X_train, y_train, X_valid, y_valid)
    assert selector.ranking_frame()["Feature"].tolist()[0] == "signal"
