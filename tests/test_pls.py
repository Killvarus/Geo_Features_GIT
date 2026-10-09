import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.experiment import ExperimentResult
from src.experiment.pls_experiment import PLSExperiment
from src.preprocessing.pls import PLSTransformer
from src.utils.logging_utils import close_logger_file_handlers


def _tiny_df(n_samples: int = 40, n_features: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f'REYX{i+1}_1': rng.normal(size=n_samples) for i in range(n_features)}
    x = np.column_stack([data[c] for c in data])
    data['H3_8'] = x[:, 0] * 0.4 + x[:, 1] * 0.2 + rng.normal(scale=0.05, size=n_samples)
    return pd.DataFrame(data)


def test_pls_skips_existing_summary(tmp_path: Path):
    df = _tiny_df()
    base = tmp_path / 'pls_skip'
    config_dir = base / 'pls_2'
    config_dir.mkdir(parents=True)
    saved = ExperimentResult(
        experiment_id='pls_2',
        experiment_name='pls_skip',
        timestamp='x',
        r2_mean=0.77,
        n_components=2,
        experiment_type='pls',
    )
    (config_dir / 'summary.json').write_text(json.dumps(saved.to_dict()), encoding='utf-8')

    experiment = PLSExperiment(
        train=df, valid=df, test=df,
        target_columns=['H3_8'],
        experiment_name='pls_skip',
        base_dir=base,
    )
    result = experiment.run_single(n_components=2)
    assert result.r2_mean == 0.77
    close_logger_file_handlers(experiment.logger)


def test_pls_grid_caps_and_dedups_components(tmp_path: Path):
    df = _tiny_df(n_samples=30, n_features=6)
    experiment = PLSExperiment(
        train=df, valid=df, test=df,
        target_columns=['H3_8'],
        experiment_name='pls_cap',
        base_dir=tmp_path / 'pls_cap',
    )
    results = experiment.run_grid(n_components_list=[16, 64, 256])
    assert len(results) == 1
    assert int(results.iloc[0]['n_components']) == 6
    assert (tmp_path / 'pls_cap' / 'pls_6' / 'summary.json').exists()
    close_logger_file_handlers(experiment.logger)


def test_pls_get_best_result_returns_dict(tmp_path: Path):
    df = _tiny_df()
    experiment = PLSExperiment(
        train=df, valid=df, test=df,
        target_columns=['H3_8'],
        experiment_name='pls_best',
        base_dir=tmp_path / 'pls_best',
    )
    experiment.run_grid(n_components_list=[1, 2])
    best = experiment.get_best_result()
    assert isinstance(best, dict)
    assert best['n_components'] in (1, 2)
    assert np.isfinite(best['r2_mean'])
    close_logger_file_handlers(experiment.logger)


def test_pls_transformer_predicts_finite():
    df = _tiny_df()
    x = df.drop(columns=['H3_8'])
    y = df[['H3_8']]
    model = PLSTransformer(n_components=3)
    model.fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (len(df), 1)
    assert np.isfinite(pred).all()
    assert model.n_components_used == 3
