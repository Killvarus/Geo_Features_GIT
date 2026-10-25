import numpy as np
import pytest

from src.preprocessing.spectrum_norm import (
    EM_NM,
    EXC_NM,
    SPECTRUM_NORMS,
    fluorescence_mask,
    normalize_spectra,
    raman_height,
)


def _spectra(n=40, seed=0):
    rng = np.random.default_rng(seed)
    excitation, emission = np.meshgrid(EXC_NM, EM_NM, indexing="ij")
    peak = np.exp(-((excitation - 355) ** 2) / 800 - ((emission - 450) ** 2) / 1500)
    raman = np.exp(-((emission - 384) ** 2) / 4) * (excitation == 340)
    scatter = 1000.0 * (np.abs(emission - excitation) <= 2)
    scale = rng.uniform(0.5, 2.0, size=(n, 1, 1))
    cube = scale * (100 * peak + 10 * raman + scatter) + rng.normal(scale=0.5, size=(n, 27, 201))
    return cube.reshape(n, -1)


def test_mask_drops_both_scatter_orders():
    mask = fluorescence_mask().reshape(27, 201)
    assert mask.sum() == 5427 - 224
    assert not mask[list(EXC_NM).index(380), list(EM_NM).index(380)]
    assert not mask[0, list(EM_NM).index(560)]
    assert mask[list(EXC_NM).index(355), list(EM_NM).index(450)]


@pytest.mark.parametrize("method", ["max", "area", "raman"])
def test_multiplicative_norms_remove_overall_brightness(method):
    X = _spectra()
    both = np.vstack([X, 2 * X])
    doubled = normalize_spectra(both, method, n_train=len(both))
    keep = np.ones(len(X), dtype=bool)
    if method == "raman":
        # Самые слабые пики упираются в нижнюю границу (1-й процентиль) — там равенства нет.
        height = raman_height(both)
        keep = height[: len(X)] > np.percentile(height, 1)
    np.testing.assert_allclose(doubled[: len(X)][keep], doubled[len(X):][keep], rtol=1e-6, atol=1e-9)


def test_max_and_snv_statistics_ignore_scatter():
    X = _spectra()
    mask = fluorescence_mask()
    scaled = normalize_spectra(X, "max", n_train=len(X))
    np.testing.assert_allclose(scaled[:, mask].max(axis=1), 1.0)
    assert scaled[:, ~mask].max() > 1.0
    snv = normalize_spectra(X, "snv", n_train=len(X))
    np.testing.assert_allclose(snv[:, mask].mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(snv[:, mask].std(axis=1), 1.0)


def test_row_max_scales_each_excitation_row_separately():
    X = _spectra()
    rows = normalize_spectra(X, "row_max", n_train=len(X)).reshape(len(X), 27, 201)
    mask = fluorescence_mask().reshape(27, 201)
    peaks = np.where(mask[None], np.abs(rows), 0.0).max(axis=2)
    np.testing.assert_allclose(peaks, 1.0)


def test_raman_height_follows_the_water_peak_and_has_a_floor():
    X = _spectra()
    height = raman_height(X)
    assert (height > 1).all()
    X[0, :] = 1.0
    scaled = normalize_spectra(X, "raman", n_train=len(X))
    assert np.isfinite(scaled).all()


def test_raw_is_a_copy_and_unknown_method_fails():
    X = _spectra()
    out = normalize_spectra(X, "raw", n_train=len(X))
    assert out is not X
    np.testing.assert_array_equal(out, X)
    assert set(SPECTRUM_NORMS) == {"raw", "max", "area", "snv", "raman", "row_max"}
    with pytest.raises(ValueError):
        normalize_spectra(X, "minmax", n_train=len(X))


def test_non_positive_denominator_is_an_error():
    X = _spectra()
    X[3] = -np.abs(X[3])
    with pytest.raises(ValueError, match="≤ 0"):
        normalize_spectra(X, "area", n_train=len(X))
    with pytest.raises(ValueError, match="≤ 0"):
        normalize_spectra(X, "max", n_train=len(X))
