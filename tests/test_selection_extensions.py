import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.linear_selection import LinearGroupBSBS, LinearPPOP, _Split, scale_channels
from src.preprocessing.spectrum_norm import transform_intensity


def _data(n=60, p=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, p))
    X = (base + 0.7 * np.roll(base, 1, axis=1)) * rng.uniform(0.5, 20.0, size=p)
    weights = rng.normal(size=p) * (rng.uniform(size=p) > 0.6)
    y = X / X.std(axis=0) @ weights + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)]), pd.Series(y, name="y")


@pytest.mark.parametrize("p", [12, 40])
def test_dual_form_matches_primal(p):
    # p = 40 > 48 строк обучения не превышает; p = 12 — обычный случай. Двойственная форма верна в обоих.
    X, y = _data(p=p)
    values, target = X.to_numpy(), y.to_numpy()
    split = _Split(values[:48], target[:48], values[48:], target[48:], "zscore", keep_fit=True)
    columns = np.arange(p)[::2] if p > 20 else np.arange(p)
    primal, base_primal = split.sse_without_each(columns, 5.0)
    dual, base_dual = split.sse_without_each(columns, 5.0, solver="dual")
    np.testing.assert_allclose(dual, primal, rtol=1e-9)
    assert base_dual == pytest.approx(base_primal, rel=1e-9)


def test_dual_form_when_features_exceed_rows():
    X, y = _data(n=40, p=90)
    values, target = X.to_numpy(), y.to_numpy()
    split = _Split(values[:30], target[:30], values[30:], target[30:], "zscore", keep_fit=True)
    columns = np.arange(90)
    primal, _ = split.sse_without_each(columns, 3.0)
    dual, _ = split.sse_without_each(columns, 3.0, solver="dual")
    np.testing.assert_allclose(dual, primal, rtol=1e-8)


def test_selector_with_dual_or_auto_solver_gives_the_same_ranking():
    X, y = _data(n=50, p=70)
    plain = LinearPPOP(alpha=4.0, step=0.1, cv=5).fit(X, y)
    for solver in ("dual", "auto"):
        other = LinearPPOP(alpha=4.0, step=0.1, cv=5, solver=solver).fit(X, y)
        assert list(other.ranking_frame()["Feature"]) == list(plain.ranking_frame()["Feature"])


def test_one_repeat_is_the_old_behaviour_and_more_repeats_pool_all_folds():
    X, y = _data()
    old = LinearPPOP(alpha=4.0, step=0.2, cv=5).fit(X, y)
    same = LinearPPOP(alpha=4.0, step=0.2, cv=5, repeats=1).fit(X, y)
    pd.testing.assert_frame_equal(old.ranking_frame(), same.ranking_frame())
    pooled = LinearPPOP(alpha=4.0, step=X.shape[1], cv=5, repeats=3).fit(X, y)
    sse = 0.0
    for repeat in range(3):
        for fit, held in KFold(n_splits=5, shuffle=True, random_state=repeat).split(X):
            model = make_pipeline(StandardScaler(), Ridge(alpha=4.0)).fit(X.iloc[fit], y.iloc[fit])
            sse += float(((y.iloc[held] - model.predict(X.iloc[held])) ** 2).sum())
    assert pooled.history_["mse"].iloc[0] == pytest.approx(sse / (3 * len(X)), rel=1e-9)


def test_group_sse_matches_refitting_without_the_group():
    X, y = _data(p=20)
    values, target = X.to_numpy(), y.to_numpy()
    split = _Split(values[:45], target[:45], values[45:], target[45:], "zscore")
    columns = np.arange(20)
    groups = [np.arange(0, 4), np.arange(4, 5), np.arange(5, 12), np.arange(12, 20)]
    sse, _ = split.sse_without_groups(columns, groups, 2.0)
    X_fit, X_eval = scale_channels(values[:45], values[45:])
    for value, group in zip(sse, groups):
        kept = np.setdiff1d(columns, group)
        model = Ridge(alpha=2.0).fit(X_fit[:, kept], target[:45])
        expected = float(((target[45:] - model.predict(X_eval[:, kept])) ** 2).sum())
        assert value == pytest.approx(expected, rel=1e-9)


def test_groups_of_one_feature_reproduce_plain_bsbs():
    X, y = _data(p=30)
    plain = LinearPPOP(alpha=4.0, step=0.1, cv=5).fit(X, y)
    grouped = LinearGroupBSBS(groups=list(X.columns), alpha=4.0, step=0.1, cv=5).fit(X, y)
    assert list(grouped.ranking_frame()["Feature"]) == list(plain.ranking_frame()["Feature"])


def test_group_ranking_keeps_groups_together():
    X, y = _data(p=24)
    labels = [i // 4 for i in range(24)]
    ranking = LinearGroupBSBS(groups=labels, alpha=4.0, step=0.2, cv=5).fit(X, y).ranking_frame()
    order = [int(name[1:]) // 4 for name in ranking["Feature"]]
    blocks = [order[i:i + 4] for i in range(0, 24, 4)]
    assert all(len(set(block)) == 1 for block in blocks)
    assert sorted(ranking["Feature"]) == sorted(X.columns)


def test_intensity_transforms():
    X = np.array([[-5.0, 0.0, 10.0, 1000.0]])
    np.testing.assert_array_equal(transform_intensity(X, "none"), X)
    asinh = transform_intensity(X, "asinh", scale=10.0)
    assert asinh[0, 0] < 0 and asinh[0, 1] == 0
    assert asinh[0, 3] == pytest.approx(np.log(2 * 100.0), rel=1e-4)
    logged = transform_intensity(X, "log", scale=10.0)
    assert logged[0, 0] == logged[0, 1] == pytest.approx(np.log(10.0))
    with pytest.raises(ValueError):
        transform_intensity(X, "sqrt")
    with pytest.raises(ValueError):
        transform_intensity(X, "log", scale=0)


def test_bad_options():
    with pytest.raises(ValueError):
        LinearPPOP(repeats=0)
    with pytest.raises(ValueError):
        LinearPPOP(solver="qr")
    with pytest.raises(ValueError):
        LinearGroupBSBS(groups=[0, 1], alpha="loo")
