import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.linear_selection import LinearFloatingBSBS, LinearPPOP, _Split, scale_channels


def _data(n=90, p=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, p))
    # Соседние каналы скоррелированы, как в спектре, и разного масштаба.
    X = (base + 0.8 * np.roll(base, 1, axis=1)) * rng.uniform(0.5, 20.0, size=p)
    weights = rng.normal(size=p) * (rng.uniform(size=p) > 0.5)
    y = X / X.std(axis=0) @ weights + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(X, columns=[f"f{i:02d}" for i in range(p)]), pd.Series(y, name="y")


def _pooled_mse(X, y, columns, alpha, cv=5, seed=0):
    sse = 0.0
    for fit, held in KFold(n_splits=cv, shuffle=True, random_state=seed).split(X):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X.iloc[fit][columns], y.iloc[fit])
        sse += float(((y.iloc[held] - model.predict(X.iloc[held][columns])) ** 2).sum())
    return sse / len(X)


def test_sse_with_each_matches_refitting():
    X, y = _data()
    values, target = X.to_numpy(), y.to_numpy()
    fit, held = slice(0, 70), slice(70, None)
    split = _Split(values[fit], target[fit], values[held], target[held], "zscore")
    columns = np.array([1, 4, 5, 9, 12, 17])
    candidates = np.array([0, 2, 8, 19])
    alpha = 3.0
    added = split.sse_with_each(columns, candidates, alpha, split.state(columns, alpha))
    # Масштаб каналов в _Split считается по всем столбцам — для подмножества он тот же.
    X_fit, X_eval = scale_channels(values[fit], values[held])
    for value, k in zip(added, candidates):
        cols = list(columns) + [k]
        model = Ridge(alpha=alpha).fit(X_fit[:, cols], target[fit])
        expected = float(((target[held] - model.predict(X_eval[:, cols])) ** 2).sum())
        assert value == pytest.approx(expected, rel=1e-9)


def test_without_swaps_and_sizes_it_is_plain_bsbs():
    X, y = _data()
    plain = LinearPPOP(alpha=4.0, step=0.1, cv=5).fit(X, y)
    floating = LinearFloatingBSBS(alpha=4.0, step=0.1, cv=5, max_swaps=0).fit(X, y)
    pd.testing.assert_frame_equal(plain.ranking_frame(), floating.ranking_frame())
    assert floating.swaps_.empty


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_swap_lowers_the_exact_cv_error(seed):
    X, y = _data(seed=seed)
    alpha = 4.0
    selector = LinearFloatingBSBS(alpha=alpha, step=0.2, cv=5, max_swaps=3, ks=[3, 6, 10]).fit(X, y)
    for swap in selector.swaps_.itertuples():
        assert swap.mse_after < swap.mse_before
        assert swap.mse_after == pytest.approx(_pooled_mse(X, y, list(swap.features), alpha), rel=1e-8)
        assert swap.added in swap.features and swap.removed not in swap.features


def test_swaps_happen_and_sets_have_requested_sizes():
    found = 0
    for seed in range(6):
        X, y = _data(seed=seed)
        selector = LinearFloatingBSBS(alpha=4.0, step=0.2, cv=5, max_swaps=3, ks=[3, 6, 10]).fit(X, y)
        found += len(selector.swaps_)
        assert sorted(selector.sets_) == [3, 6, 10]
        for k, features in selector.sets_.items():
            assert len(features) == len(set(features)) == k
        ranking = selector.ranking_frame()
        assert sorted(ranking["Feature"]) == sorted(X.columns)
        assert list(ranking["Individual_Rank"]) == list(range(1, X.shape[1] + 1))
    assert found > 0


def test_bad_parameters():
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            LinearFloatingBSBS(max_swaps=bad)
    with pytest.raises(ValueError):
        LinearFloatingBSBS(alpha="loo")
