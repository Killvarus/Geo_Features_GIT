import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_selection import TrueBackwardFeatureSelection
from src.models.linear_selection import LinearPPOP, LinearRFE, packet_size


def _data(n_train=60, n_valid=30, p=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_train + n_valid, p))
    # Соседние каналы скоррелированы, как в спектре, и разного масштаба.
    X = (base + 0.6 * np.roll(base, 1, axis=1)) * rng.uniform(0.5, 20.0, size=p)
    weights = rng.normal(size=p) * (rng.uniform(size=p) > 0.3)
    y = X / X.std(axis=0) @ weights + rng.normal(scale=0.5, size=n_train + n_valid)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    y = pd.Series(y, name="y")
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]


def _model(alpha):
    if alpha == 0.0:
        return make_pipeline(StandardScaler(), LinearRegression())
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def test_packet_size_fixed_and_fraction():
    assert packet_size(100, 7) == 7
    assert packet_size(5, 7) == 5
    assert packet_size(100, 0.1) == 10
    assert packet_size(95, 0.1) == 10
    assert packet_size(9, 0.1) == 1
    assert packet_size(100, 0.1, min_step=20) == 20
    for bad in (0, -1, 1.0, 0.0, 1.5, True):
        with pytest.raises(ValueError):
            packet_size(10, bad)


@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_ppop_scores_match_refitting_on_validation(alpha):
    X_train, y_train, X_valid, y_valid = _data()
    p = X_train.shape[1]
    selector = LinearPPOP(alpha=alpha, step=p).fit(X_train, y_train, X_valid, y_valid)
    full = _model(alpha).fit(X_train, y_train)
    mse_full = mean_squared_error(y_valid, full.predict(X_valid))
    assert selector.history_["mse"].iloc[0] == pytest.approx(mse_full, rel=1e-9)
    scores = selector.ranking_frame().set_index("Feature")["Score"]
    for feature in X_train.columns:
        reduced = _model(alpha).fit(X_train.drop(columns=[feature]), y_train)
        mse = mean_squared_error(y_valid, reduced.predict(X_valid.drop(columns=[feature])))
        assert scores[feature] == pytest.approx(mse - mse_full, rel=1e-7, abs=1e-10)


def test_ppop_cv_pools_folds_of_train():
    X_train, y_train, _, _ = _data(n_train=80)
    p = X_train.shape[1]
    selector = LinearPPOP(alpha=3.0, step=p, cv=4, random_state=1).fit(X_train, y_train)
    folds = list(KFold(n_splits=4, shuffle=True, random_state=1).split(X_train))

    def pooled_mse(columns):
        sse = 0.0
        for fit, held in folds:
            model = _model(3.0).fit(X_train.iloc[fit][columns], y_train.iloc[fit])
            sse += float(((y_train.iloc[held] - model.predict(X_train.iloc[held][columns])) ** 2).sum())
        return sse / len(X_train)

    columns = list(X_train.columns)
    mse_full = pooled_mse(columns)
    scores = selector.ranking_frame().set_index("Feature")["Score"]
    for feature in columns:
        expected = pooled_mse([c for c in columns if c != feature]) - mse_full
        assert scores[feature] == pytest.approx(expected, rel=1e-7, abs=1e-10)


def test_ppop_matches_generic_refitting_class():
    X_train, y_train, X_valid, y_valid = _data(p=10)
    fast = LinearPPOP(alpha=2.0, step=3).fit(X_train, y_train, X_valid, y_valid)
    slow = TrueBackwardFeatureSelection(_model(2.0), n_features_to_drop=3, cv=None)
    slow.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
    fast_frame = fast.ranking_frame()
    slow_frame = slow.get_ranking_df()
    assert fast_frame["Feature"].tolist() == slow_frame["Feature"].tolist()
    assert fast_frame["Group_Rank"].tolist() == slow_frame["Group_Rank"].tolist()


def test_ppop_fraction_packets_shrink():
    X_train, y_train, X_valid, y_valid = _data(p=12)
    selector = LinearPPOP(alpha=1.0, step=0.25).fit(X_train, y_train, X_valid, y_valid)
    assert selector.history_["n_features"].tolist() == [12, 9, 6, 4, 3, 2, 1]
    assert selector.history_["dropped"].tolist() == [3, 3, 2, 1, 1, 1, 1]
    frame = selector.ranking_frame()
    assert sorted(frame["Individual_Rank"]) == list(range(1, 13))
    assert frame["Group_Rank"].iloc[0] == 1


def test_ols_refuses_more_features_than_rows():
    X_train, y_train, X_valid, y_valid = _data(n_train=6, p=8)
    with pytest.raises(ValueError, match="МНК"):
        LinearPPOP(alpha=0.0, step=1).fit(X_train, y_train, X_valid, y_valid)
    with pytest.raises(ValueError, match="МНК"):
        LinearRFE(alpha=0.0, step=1).fit(X_train, y_train)


@pytest.mark.parametrize("refresh_every", [1, 3, 1000])
def test_rfe_step_one_matches_sklearn(refresh_every):
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=3)
    ours = LinearRFE(alpha=4.0, step=1, refresh_every=refresh_every).fit(X_train, y_train)
    scaled = StandardScaler().fit_transform(X_train)
    reference = RFE(Ridge(alpha=4.0), n_features_to_select=1, step=1).fit(scaled, y_train)
    expected = dict(zip(X_train.columns, reference.ranking_))
    frame = ours.ranking_frame()
    assert dict(zip(frame["Feature"], frame["Individual_Rank"])) == expected


def test_rfe_fraction_matches_refitting_loop():
    X_train, y_train, _, _ = _data(n_train=50, p=15, seed=4)
    ours = LinearRFE(alpha=4.0, step=0.3, refresh_every=1000).fit(X_train, y_train)
    scaled = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    remaining = list(X_train.columns)
    dropped_order = []
    while remaining:
        coef = np.abs(Ridge(alpha=4.0).fit(scaled[remaining], y_train).coef_)
        size = packet_size(len(remaining), 0.3)
        packet = [remaining[i] for i in np.argsort(coef, kind="stable")[:size]]
        dropped_order.append(set(packet))
        remaining = [f for f in remaining if f not in packet]
    frame = ours.ranking_frame()
    groups = [set(frame.loc[frame["Group_Rank"] == g, "Feature"]) for g in sorted(frame["Group_Rank"].unique())]
    assert groups == list(reversed(dropped_order))
