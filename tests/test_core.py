import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from src.experiment import AggregationExperiment
from src.preprocessing import AggregationConfig, aggregate_features
from src.preprocessing.aggregation import parse_feature_name
from src.utils import Data


def make_basic_dataset(n_samples: int = 40) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        'REYX1_1': np.random.randn(n_samples),
        'REYX1_2': np.random.randn(n_samples),
        'REYX2_1': np.random.randn(n_samples),
        'REYX2_2': np.random.randn(n_samples),
        'IMYX1_1': np.random.randn(n_samples),
        'IMYX1_2': np.random.randn(n_samples),
        'H1_8': np.random.randn(n_samples),
        'H2_8': np.random.randn(n_samples),
        'H3_8': np.random.randn(n_samples),
    })


def test_data_splits_targets_and_features_correctly():
    train = make_basic_dataset()
    valid = train.copy()
    test = train.copy()

    data = Data(train, test, valid, columns=['H1_8', 'H2_8', 'H3_8'])

    assert data.n_features == 6
    assert data.n_targets == 3
    assert 'H1_8' not in data.X_train.columns
    assert list(data.y_train.columns) == ['H1_8', 'H2_8', 'H3_8']


def test_data_raises_on_schema_mismatch():
    train = make_basic_dataset()
    valid = train.drop(columns=['REYX1_1']).copy()
    test = train.copy()

    try:
        Data(train, test, valid, columns=['H1_8'])
        assert False, 'Expected ValueError for schema mismatch'
    except ValueError as exc:
        assert 'Колонки train/valid/test должны совпадать' in str(exc)


def test_aggregation_config_validation():
    config = AggregationConfig(freq_step=2, pickup_step=3, agg_method='mean')
    assert config.freq_step == 2
    assert config.pickup_step == 3
    assert config.agg_method == 'mean'
    assert config.is_active is True


def test_parse_feature_name():
    parsed = parse_feature_name('REYX13_31')
    assert parsed is not None
    assert parsed['component'] == 'RE'
    assert parsed['polarization'] == 'YX'
    assert parsed['frequency'] == 13
    assert parsed['pickup'] == 31
    assert parse_feature_name('invalid') is None


def test_aggregate_features_reduces_feature_space():
    train = pd.DataFrame({
        'REYX1_1': [1, 2, 3],
        'REYX1_2': [4, 5, 6],
        'REYX2_1': [7, 8, 9],
        'REYX2_2': [10, 11, 12],
        'H1_8': [0.1, 0.2, 0.3],
    })
    valid = train.copy()
    test = train.copy()
    config = AggregationConfig(freq_step=2, pickup_step=1, agg_method='mean')

    train_result, valid_result, test_result, features = aggregate_features(
        train, valid, test, config, target_columns=['H1_8']
    )

    assert len(features) > 0
    assert 'H1_8' in train_result.columns
    assert 'H1_8' not in features
    assert len(features) <= 4
    assert list(train_result.columns) == list(valid_result.columns) == list(test_result.columns)


def test_aggregation_experiment_smoke():
    np.random.seed(42)
    n_samples = 30
    train = pd.DataFrame({
        **{f'REYX{i}_1': np.random.randn(n_samples) for i in range(1, 4)},
        **{f'IMYX{i}_1': np.random.randn(n_samples) for i in range(1, 4)},
        'H1_8': np.random.randn(n_samples),
    })
    valid = train.copy()
    test = train.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        experiment = AggregationExperiment(
            train=train,
            valid=valid,
            test=test,
            target_columns=['H1_8'],
            experiment_name='test_exp',
            base_dir=Path(tmpdir) / 'test_exp'
        )

        result = experiment.run_single(
            freq_step=1,
            pickup_step=1,
            agg_method='mean',
            n_iter=1,
            num_epochs=5,
            batch_size=8,
        )

        assert result is not None
        assert result.experiment_type == 'aggregation'
        assert result.n_features == 6
