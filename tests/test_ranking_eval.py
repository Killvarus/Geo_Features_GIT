import numpy as np
import pytest
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.ranking_eval import ridge_loo_r2


def _data(n=80, p=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 5.0, size=p)
    y = X[:, :5] @ rng.normal(size=5) + rng.normal(size=n)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.mark.parametrize("p", [30, 100])
def test_matches_sklearn_ridge_at_chosen_alpha(p):
    X_train, y_train, X_eval, y_eval = _data(p=p)
    r2, alpha = ridge_loo_r2(X_train, y_train, X_eval, y_eval)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X_train, y_train)
    assert r2 == pytest.approx(r2_score(y_eval, model.predict(X_eval)), rel=1e-8)


@pytest.mark.parametrize("p", [30, 100, 200])
def test_alpha_matches_sklearn_ridgecv_leave_one_out(p):
    X_train, y_train, X_eval, y_eval = _data(p=p, seed=2)
    alphas = np.logspace(-2, 4, 25)
    _, alpha = ridge_loo_r2(X_train, y_train, X_eval, y_eval, alphas)
    scaled = StandardScaler().fit_transform(X_train)
    reference = RidgeCV(alphas=alphas).fit(scaled, y_train)
    assert alpha == pytest.approx(reference.alpha_)

