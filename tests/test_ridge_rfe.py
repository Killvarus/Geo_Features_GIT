import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.ridge_rfe import BackwardRidgeRFE, ForwardRidgeRFE, draw_r2_curves, plot_r2_curves


def _abs_coef(X, y, columns, alpha):
    scaler = StandardScaler().fit(X)
    scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(scaled.loc[:, columns], np.asarray(y, dtype=float))
    coef = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
    return {feature: float(value) for feature, value in zip(columns, coef)}


def _signal_frame():
    rng = np.random.default_rng(0)
    n = 120
    signal = rng.normal(size=n)
    weak = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "weak": weak, "noise": noise})
    y = 4.0 * signal + 0.5 * weak
    return X, y


def _duplicate_frame():
    rng = np.random.default_rng(1)
    n = 100
    shared = rng.normal(size=n)
    X = pd.DataFrame({
        "a": shared,
        "a_copy": shared.copy(),
        "noise": rng.normal(size=n),
    })
    y = shared.copy()
    return X, y


def _assert_groups(frame: pd.DataFrame) -> None:
    ranks = frame["Individual_Rank"].tolist()
    assert ranks == list(range(1, len(frame) + 1))
    assert frame["Feature"].is_unique
    for _, part in frame.groupby("Group_Rank"):
        part = part.sort_values("Individual_Rank")
        group_ranks = part["Individual_Rank"].tolist()
        assert group_ranks == list(range(group_ranks[0], group_ranks[-1] + 1))
        assert part["Score"].tolist() == sorted(part["Score"].tolist(), reverse=True)


def test_backward_scores_match_coef_of_the_model_that_dropped_them():
    X, y = _signal_frame()
    alpha = 1.0
    selector = BackwardRidgeRFE(alpha=alpha, n_features_to_drop=1)
    frame = selector.fit(X, y).ranking_frame().set_index("Feature")

    full = _abs_coef(X, y, list(X.columns), alpha)
    worst = min(full, key=lambda feature: (full[feature], feature))
    assert frame.loc[worst, "Individual_Rank"] == len(X.columns)
    assert abs(frame.loc[worst, "Score"] - full[worst]) < 1e-8

    best = frame["Individual_Rank"].idxmin()
    alone = _abs_coef(X, y, [best], alpha)[best]
    assert abs(frame.loc[best, "Score"] - alone) < 1e-8
    assert frame.loc[best, "Group_Rank"] == 1
    assert selector.elapsed_seconds_ > 0.0


def test_forward_scores_match_coef_of_the_model_that_added_them():
    X, y = _signal_frame()
    alpha = 1.0
    selector = ForwardRidgeRFE(alpha=alpha, n_features_to_add=1)
    frame = selector.fit(X, y).ranking_frame().set_index("Feature")

    full = _abs_coef(X, y, list(X.columns), alpha)
    best = min(full, key=lambda feature: (-full[feature], feature))
    assert frame.loc[best, "Individual_Rank"] == 1
    assert abs(frame.loc[best, "Score"] - full[best]) < 1e-8

    last = frame["Individual_Rank"].idxmax()
    alone = _abs_coef(X, y, [last], alpha)[last]
    assert abs(frame.loc[last, "Score"] - alone) < 1e-8
    assert selector.elapsed_seconds_ > 0.0


def test_both_directions_rank_signal_before_weak_before_noise():
    X, y = _signal_frame()
    for selector in (
        BackwardRidgeRFE(alpha=1.0, n_features_to_drop=1),
        ForwardRidgeRFE(alpha=1.0, n_features_to_add=1),
    ):
        order = selector.fit(X, y).ranking_frame()["Feature"].tolist()
        assert order == ["signal", "weak", "noise"]


def test_packet_keeps_strong_pair_in_group_one():
    rng = np.random.default_rng(2)
    n = 150
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    X = pd.DataFrame({
        "s1": first,
        "s2": second,
        "n1": rng.normal(size=n),
        "n2": rng.normal(size=n),
    })
    y = 5.0 * first + 2.0 * second
    for selector in (
        BackwardRidgeRFE(alpha=1.0, n_features_to_drop=2),
        ForwardRidgeRFE(alpha=1.0, n_features_to_add=2),
    ):
        frame = selector.fit(X, y).ranking_frame()
        _assert_groups(frame)
        group = set(frame.loc[frame["Group_Rank"] == 1, "Feature"])
        assert group == {"s1", "s2"}
        assert frame.loc[frame["Feature"] == "s1", "Individual_Rank"].item() == 1


def test_forward_refit_grows_coef_of_duplicate_after_its_copy_is_taken():
    X, y = _duplicate_frame()
    alpha = 1.0
    frame = ForwardRidgeRFE(alpha=alpha, n_features_to_add=1).fit(X, y).ranking_frame()
    _assert_groups(frame)
    full = _abs_coef(X, y, list(X.columns), alpha)
    first = frame.iloc[0]["Feature"]
    second = frame.iloc[1]["Feature"]
    assert first in {"a", "a_copy"}
    assert second in {"a", "a_copy"}
    assert abs(frame.iloc[0]["Score"] - full[first]) < 1e-8
    refit = _abs_coef(X, y, [second, "noise"], alpha)
    assert abs(frame.iloc[1]["Score"] - refit[second]) < 1e-8
    assert frame.iloc[1]["Score"] > full[second] + 1e-3


def test_backward_survivor_coef_is_refit_without_dropped_features():
    X, y = _duplicate_frame()
    alpha = 1.0
    frame = BackwardRidgeRFE(alpha=alpha, n_features_to_drop=1).fit(X, y).ranking_frame()
    _assert_groups(frame)
    full = _abs_coef(X, y, list(X.columns), alpha)
    worst = min(full, key=lambda feature: (full[feature], feature))
    assert frame.iloc[-1]["Feature"] == worst
    assert abs(frame.iloc[-1]["Score"] - full[worst]) < 1e-8
    best = frame.iloc[0]["Feature"]
    assert best in {"a", "a_copy"}
    alone = _abs_coef(X, y, [best], alpha)[best]
    assert abs(frame.iloc[0]["Score"] - alone) < 1e-8
    assert alone > full[best] + 1e-3


def test_large_scale_noise_does_not_outrank_signal():
    rng = np.random.default_rng(3)
    n = 80
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "signal": signal,
        "huge": 1e6 * rng.normal(size=n),
    })
    y = signal.copy()
    for selector in (
        BackwardRidgeRFE(alpha=700.0, n_features_to_drop=1),
        ForwardRidgeRFE(alpha=700.0, n_features_to_add=1),
    ):
        order = selector.fit(X, y).ranking_frame()["Feature"].tolist()
        assert order[0] == "signal"


def test_one_packet_makes_forward_and_backward_agree():
    X, y = _signal_frame()
    alpha = 700.0
    backward = BackwardRidgeRFE(alpha=alpha, n_features_to_drop=10).fit(X, y).ranking_frame()
    forward = ForwardRidgeRFE(alpha=alpha, n_features_to_add=10).fit(X, y).ranking_frame()
    pd.testing.assert_frame_equal(backward, forward)
    _assert_groups(backward)
    assert set(backward["Group_Rank"]) == {1}


def test_ranking_frame_before_fit_raises():
    with pytest.raises(RuntimeError):
        BackwardRidgeRFE().ranking_frame()
    with pytest.raises(RuntimeError):
        ForwardRidgeRFE().ranking_frame()


@pytest.mark.parametrize(
    "selector_type, kwargs",
    [
        (BackwardRidgeRFE, {"n_features_to_drop": 0}),
        (ForwardRidgeRFE, {"n_features_to_add": 0}),
        (BackwardRidgeRFE, {"alpha": -1}),
        (ForwardRidgeRFE, {"alpha": -0.1}),
    ],
)
def test_rejects_bad_arguments(selector_type, kwargs):
    with pytest.raises(ValueError):
        selector_type(**kwargs)


def _curve_point(k: int, mean: float) -> dict:
    return {
        "k": k,
        "runs": [{"seed": 42, "r2": mean}, {"seed": 43, "r2": mean}, {"seed": 44, "r2": mean}],
        "mean": mean,
        "std": 0.01,
    }


def test_plot_waits_until_every_ion_has_a_curve(tmp_path):
    ions = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
    assert plot_r2_curves(tmp_path, ions, "Обратный RFE, Ridge alpha=700") is None
    assert not (tmp_path / "r2_vs_n_features.png").exists()


def test_plot_saves_curve_with_decreasing_feature_axis(tmp_path):
    ions = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
    for ion in ions:
        ion_dir = tmp_path / ion
        ion_dir.mkdir()
        points = [_curve_point(500, 0.4), _curve_point(5427, 0.5)]
        (ion_dir / "curve_points.json").write_text(json.dumps(points), encoding="utf-8")
    out = plot_r2_curves(tmp_path, ions, "Обратный RFE, Ridge alpha=700")
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000
    curves = []
    for ion in ions:
        points = json.loads((tmp_path / ion / "curve_points.json").read_text(encoding="utf-8"))
        curves.append((ion, points))
    fig = draw_r2_curves(curves, "Обратный RFE, Ridge alpha=700")
    left, right = fig.axes[0].get_xlim()
    assert left > right
    assert left == 5427 + 200
    assert right == 0
    import matplotlib.pyplot as plt
    plt.close(fig)
