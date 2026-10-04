import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from src.evaluation.experiment import ExperimentResult
from src.experiment.pca_experiment import PCAExperiment
from src.models.neural_network import (
    _history_json_path,
    plot_learning_curve_from_history,
    save_learning_history,
)
from src.utils.logging_utils import setup_logger


def test_setup_logger_writes_to_requested_file(tmp_path: Path):
    first = tmp_path / 'a' / 'train.log'
    second = tmp_path / 'b' / 'train.log'
    name = 'test.olp.switcher'

    # Сбросить возможный хендлер от предыдущих тестов
    existing = logging.getLogger(name)
    for handler in list(existing.handlers):
        existing.removeHandler(handler)
        handler.close()

    logger = setup_logger(name, first)
    logger.info('first-file')

    logger = setup_logger(name, second)
    logger.info('second-file')

    assert 'first-file' in first.read_text(encoding='utf-8')
    assert 'second-file' in second.read_text(encoding='utf-8')
    assert 'second-file' not in first.read_text(encoding='utf-8')
    assert 'first-file' not in second.read_text(encoding='utf-8')


def test_learning_history_roundtrip_plots_all_epochs(tmp_path: Path):
    history_path = tmp_path / 'learning_history_iter_1.json'
    png_path = tmp_path / 'learning_curve_iter_1.png'
    train = [1.0, 0.8, 0.6, 0.55, 0.5]
    val = [1.1, 0.9, 0.7, 0.65, 0.62]

    saved = save_learning_history(train, val, best_epoch=4, save_path=history_path, title='OLP test')
    assert saved.exists()
    assert _history_json_path(png_path).name == 'learning_history_iter_1.json'

    plot_learning_curve_from_history(history_path, png_path)
    assert png_path.exists()
    payload = json.loads(history_path.read_text(encoding='utf-8'))
    assert payload['source'] == 'olp_in_memory'
    assert payload['n_epochs'] == 5
    assert payload['train_losses'] == train


def test_pca_get_best_result_on_experiment_result_objects(tmp_path: Path):
    import pandas as pd

    df = pd.DataFrame({
        'REYX1_1': [0.1, 0.2, 0.3],
        'H3_8': [1.0, 2.0, 3.0],
    })
    experiment = PCAExperiment(
        train=df, valid=df, test=df,
        target_columns=['H3_8'],
        experiment_name='pca_best_test',
        base_dir=tmp_path / 'pca_best_test',
    )
    experiment.results = [
        ExperimentResult(
            experiment_id='pca_10', experiment_name='t', timestamp='x',
            r2_mean=0.2, n_components=10,
        ),
        ExperimentResult(
            experiment_id='pca_20', experiment_name='t', timestamp='x',
            r2_mean=0.8, n_components=20,
        ),
    ]
    best = experiment.get_best_result()
    assert best['n_components'] == 20
    assert best['r2_mean'] == 0.8
    from src.utils.logging_utils import close_logger_file_handlers
    close_logger_file_handlers(experiment.logger)


def test_aggregation_skips_existing_summary(tmp_path: Path):
    import pandas as pd
    from src.experiment.aggregation import AggregationExperiment
    from src.evaluation.experiment import ExperimentResult

    df = pd.DataFrame({
        'REYX1_1': [0.1, 0.2, 0.3],
        'H1_8': [1.0, 2.0, 3.0],
    })
    base = tmp_path / 'agg_skip'
    config_dir = base / 'freq1_pickup1_mean'
    config_dir.mkdir(parents=True)
    saved = ExperimentResult(
        experiment_id='freq1_pickup1_mean',
        experiment_name='agg_skip',
        timestamp='x',
        r2_mean=0.42,
        n_features=1,
        freq_agg_step=1,
        pickup_agg_step=1,
        agg_method='mean',
    )
    (config_dir / 'summary.json').write_text(
        __import__('json').dumps(saved.to_dict()),
        encoding='utf-8',
    )

    experiment = AggregationExperiment(
        train=df, valid=df, test=df,
        target_columns=['H1_8'],
        experiment_name='agg_skip',
        base_dir=base,
    )
    result = experiment.run_single(freq_step=1, pickup_step=1, agg_method='mean', n_iter=1, num_epochs=1)
    assert result.r2_mean == 0.42
    from src.utils.logging_utils import close_logger_file_handlers
    close_logger_file_handlers(experiment.logger)
