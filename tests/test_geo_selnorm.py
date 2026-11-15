import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import run_geo_selnorm_grid as grid
from src.preprocessing.geo_norm import (
    GEO_SAMPLE_NORMS,
    amp_phase,
    block_groups,
    normalize_samples,
    parse_feature,
    profile_groups,
)


def _names(freqs=2, pickets=4):
    return [f"{part}{pol}{f}_{p}" for part in ("RE", "IM") for pol in ("YX", "XY", "HX")
            for f in range(1, freqs + 1) for p in range(1, pickets + 1)]


def _X(n=7, freqs=2, pickets=4, seed=0):
    rng = np.random.default_rng(seed)
    names = _names(freqs, pickets)
    X = rng.normal(size=(n, len(names)))
    for i, name in enumerate(names):
        part, pol, _, _ = parse_feature(name)
        if pol != "HX":
            X[:, i] = -np.abs(X[:, i]) * 0.01 - 0.05  # импеданс отрицателен и мал, как в данных
    return X, names


def test_parse_and_groups():
    assert parse_feature("IMHX13_31") == ("IM", "HX", 13, 31)
    with pytest.raises(ValueError):
        parse_feature("H3_8")
    names = _names()
    assert len(profile_groups(names)) == 2 * 3 * 2
    assert all(len(g) == 4 for g in profile_groups(names))
    assert len(block_groups(names)) == 6
    assert all(len(g) == 8 for g in block_groups(names))


def test_sample_norms():
    X, names = _X()
    assert set(GEO_SAMPLE_NORMS) == {"raw", "profile_center", "profile_scale", "profile_snv", "block_scale"}
    np.testing.assert_array_equal(normalize_samples(X, names, "raw"), X)
    for index in profile_groups(names):
        centered = normalize_samples(X, names, "profile_center")[:, index]
        np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-12)
        scaled = normalize_samples(X, names, "profile_scale")[:, index]
        np.testing.assert_allclose(np.sqrt((scaled**2).mean(axis=1)), 1)
        assert np.all(np.sign(scaled) == np.sign(X[:, index]))
        snv = normalize_samples(X, names, "profile_snv")[:, index]
        np.testing.assert_allclose(snv.std(axis=1), 1)
    for index in block_groups(names):
        block = normalize_samples(X, names, "block_scale")[:, index]
        np.testing.assert_allclose(np.sqrt((block**2).mean(axis=1)), 1)
    # Масштаб примера целиком не влияет на нормировки масштаба.
    for method in ("profile_scale", "profile_snv", "block_scale"):
        np.testing.assert_allclose(normalize_samples(3 * X, names, method), normalize_samples(X, names, method))
    with pytest.raises(ValueError):
        normalize_samples(X, names, "area")
    with pytest.raises(ValueError):
        normalize_samples(np.zeros_like(X), names, "profile_scale")


def test_amp_phase_only_touches_impedance():
    X, names = _X()
    out, new_names = amp_phase(X, names)
    assert len(new_names) == len(names)
    i, j = names.index("REYX1_2"), names.index("IMYX1_2")
    assert new_names[i] == "AMPYX1_2" and new_names[j] == "PHSYX1_2"
    np.testing.assert_allclose(out[:, i], np.log(np.hypot(X[:, i], X[:, j])))
    np.testing.assert_allclose(out[:, j], np.arctan2(X[:, j], X[:, i]))
    k = names.index("REHX2_3")
    assert new_names[k] == "REHX2_3"
    np.testing.assert_array_equal(out[:, k], X[:, k])


def test_ridge_validation_matches_sklearn():
    rng = np.random.default_rng(1)
    X_train, X_valid = rng.normal(size=(80, 12)) * rng.uniform(0.1, 5, 12), rng.normal(size=(30, 12))
    y_train, y_valid = rng.normal(size=(80, 2)), rng.normal(size=(30, 2))
    alphas = np.array([0.1, 10.0])
    errors = grid.ridge_validation_mse(X_train, y_train, X_valid, y_valid, alphas)
    scaler = StandardScaler().fit(X_train)
    for a, alpha in enumerate(alphas):
        for t in range(2):
            model = Ridge(alpha=alpha).fit(scaler.transform(X_train), y_train[:, t])
            expected = np.mean((y_valid[:, t] - model.predict(scaler.transform(X_valid))) ** 2)
            assert errors[a, t] == pytest.approx(expected, rel=1e-9)


def test_alpha_has_a_floor_for_selection(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d1" / "alpha.json").write_text(json.dumps({"alpha": 1e-6}), encoding="utf-8")
    assert grid.read_alpha("d1", tmp_path) == grid.ALPHA_FLOOR
    (tmp_path / "d1" / "alpha.json").write_text(json.dumps({"alpha": 17.8}), encoding="utf-8")
    assert grid.read_alpha("d1", tmp_path) == 17.8


def test_job_sets():
    assert len(grid.jobs("prepare")) == 6
    for dataset in ("d1", "d2", "d3"):
        lines = grid.jobs(dataset)
        assert len(lines) == 3 + 5 * 3 and len(set(lines)) == len(lines)
        assert all(line.split()[4] == "zscore" for line in lines if line.startswith("rank"))
    same = grid.jobs("d1_same")
    assert len(same) == 5 * 3 + 5 * 3 + 0 and len(set(same)) == len(same)
    assert all("amp_phase" not in line for line in grid.jobs("d1"))
    assert grid.describe(("d1", "raw", "profile_scale__robust", "bsbs", "H3_8"))["channel_norm"] == "robust"
    assert grid.describe(("d1", "raw", "raw__all", "H1_8"))["method"] == "all_channels"


def test_cache_selection_and_linear_check_end_to_end(tmp_path, monkeypatch):
    names = _names(freqs=3, pickets=5)
    rng = np.random.default_rng(2)
    for split, n in (("trn", 120), ("vld", 40), ("tst", 40)):
        X = rng.normal(size=(n, len(names)))
        frame = pd.DataFrame(X, columns=names)
        for t, target in enumerate(grid.TARGETS):
            frame[target] = X[:, t] - 0.5 * X[:, t + 5] + rng.normal(scale=0.1, size=n)
        frame.to_csv(tmp_path / f"mtsgrvmgn_{split}.csv", index=False)
    cache = tmp_path / "cache"
    grid.build_cache("d1", data_dir=tmp_path, cache_dir=cache)
    monkeypatch.setattr(grid, "CACHE_DIR", cache)
    out = tmp_path / "out"
    (out / "d1").mkdir(parents=True)
    (out / "d1" / "alpha.json").write_text(json.dumps({"alpha": 1.0}), encoding="utf-8")
    grid.run_rank("d1", "raw", "profile_scale", "zscore", "H1_8", [5, 10], perceptron=False, mlp=grid.MLP, out_dir=out)
    ion_dir = out / "d1" / "raw" / "profile_scale__zscore" / "bsbs" / "H1_8"
    ranking = pd.read_csv(ion_dir / "ranking.csv")
    assert sorted(ranking["Feature"]) == sorted(names)
    checks = json.loads((ion_dir / "linear_check.json").read_text(encoding="utf-8"))
    assert set(checks) == {"5", "10"} and set(checks["5"]) == {"ridge", "ols"}
    with pytest.raises(SystemExit):
        grid.run_rank("d1", "raw", "amp_phase", "zscore", "H1_8", [5], perceptron=False, mlp=grid.MLP, out_dir=out)
