import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.experiment import ExperimentResult
from src.experiment.wavelet_experiment import WaveletExperiment
from src.preprocessing.wavelet import (
    WaveletTransformer,
    apply_wavelet_to_data,
    make_synthetic_mtz_frame,
    wavelet_config_name,
)
from src.utils.logging_utils import close_logger_file_handlers


def test_config_name_includes_energy_threshold():
    name = wavelet_config_name('db4', 2, 'energy', energy_ratio=0.9)
    assert name == 'wavelet_db4_l2_grid2d_energy_e90'


def test_approx_reduces_features():
    df = make_synthetic_mtz_frame(40, seed=0)
    X = df.drop(columns=[c for c in df.columns if c.startswith('H')])
    transformer = WaveletTransformer(wavelet='haar', level=2, keep='approx')
    Xt = transformer.fit_transform(X)
    assert Xt.shape[0] == 40
    assert transformer.n_features_used < X.shape[1]
    assert transformer.n_features_used == Xt.shape[1]
    assert list(Xt.columns) == [f'W{i+1}' for i in range(Xt.shape[1])]
    assert transformer.energy_retained is not None
    assert 0.0 < transformer.energy_retained <= 1.0


def test_energy_keep_respects_ratio_and_is_fit_on_train():
    train = make_synthetic_mtz_frame(60, seed=1)
    valid = make_synthetic_mtz_frame(20, seed=2)
    X_train = train.drop(columns=[c for c in train.columns if c.startswith('H')])
    X_valid = valid.drop(columns=[c for c in valid.columns if c.startswith('H')])
    transformer = WaveletTransformer(
        wavelet='haar', level=2, keep='energy', energy_ratio=0.90
    )
    transformer.fit(X_train)
    assert transformer.keep_indices_ is not None
    assert transformer.energy_retained >= 0.90
    Xv = transformer.transform(X_valid)
    assert Xv.shape[1] == transformer.n_features_used
    # маска не пересчитывается на valid
    frozen = transformer.keep_indices_.copy()
    transformer.transform(X_valid)
    np.testing.assert_array_equal(transformer.keep_indices_, frozen)


def test_db4_level2_pads_and_runs():
    df = make_synthetic_mtz_frame(24, seed=3)
    X = df.drop(columns=[c for c in df.columns if c.startswith('H')])
    transformer = WaveletTransformer(wavelet='db4', level=2, keep='approx')
    Xt = transformer.fit_transform(X)
    assert transformer.padded_shape_[0] >= 13
    assert transformer.padded_shape_[1] >= 31
    assert Xt.shape[1] == transformer.n_features_used
    assert np.isfinite(Xt.to_numpy()).all()


def test_freq1d_layout():
    df = make_synthetic_mtz_frame(16, seed=4)
    X = df.drop(columns=[c for c in df.columns if c.startswith('H')])
    transformer = WaveletTransformer(wavelet='haar', level=1, keep='approx', layout='freq1d')
    Xt = transformer.fit_transform(X)
    assert Xt.shape[1] > 0
    assert transformer.n_features_used < X.shape[1]


def test_apply_wavelet_attaches_only_requested_targets():
    train = make_synthetic_mtz_frame(30, seed=5)
    valid = make_synthetic_mtz_frame(10, seed=6)
    test = make_synthetic_mtz_frame(10, seed=7)
    train_w, valid_w, test_w, transformer = apply_wavelet_to_data(
        train, valid, test, target_columns=['H3_8'], wavelet='haar', level=1, keep='approx'
    )
    assert 'H3_8' in train_w.columns
    assert 'H1_8' not in train_w.columns
    assert 'H2_8' not in train_w.columns
    assert len(train_w) == len(train)
    assert len(valid_w) == len(valid)
    assert len(test_w) == len(test)
    leaked = [c for c in train_w.columns if c.startswith('H') and c != 'H3_8']
    assert leaked == []
    assert transformer.n_features_used == train_w.shape[1] - 1


def test_apply_wavelet_length_mismatch_raises():
    from src.preprocessing.wavelet import _attach_targets

    features = pd.DataFrame({'W1': [1.0, 2.0]})
    source = pd.DataFrame({'H3_8': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='разной длины'):
        _attach_targets(features, source, ['H3_8'])


def test_wavelet_experiment_skips_existing_summary(tmp_path: Path):
    df = make_synthetic_mtz_frame(20, seed=11)
    base = tmp_path / 'wav_skip'
    config_name = wavelet_config_name('haar', 1, 'approx')
    config_dir = base / config_name
    config_dir.mkdir(parents=True)
    saved = ExperimentResult(
        experiment_id=config_name,
        experiment_name='wav_skip',
        timestamp='x',
        r2_mean=0.42,
        n_features=12,
        experiment_type='wavelet',
        wavelet='haar',
    )
    (config_dir / 'summary.json').write_text(json.dumps(saved.to_dict()), encoding='utf-8')
    experiment = WaveletExperiment(
        train=df, valid=df, test=df,
        target_columns=['H3_8'],
        experiment_name='wav_skip',
        base_dir=base,
    )
    result = experiment.run_single(wavelet='haar', level=1, keep='approx')
    assert result.r2_mean == 0.42
    close_logger_file_handlers(experiment.logger)
