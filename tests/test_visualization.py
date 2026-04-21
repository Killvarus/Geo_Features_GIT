from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiment import extract_metrics_from_excel
from src.visualization.experiment_plots import (
    load_all_summaries,
    plot_agg_methods_comparison,
    plot_all_experiments_bar,
    plot_features_vs_metric,
    plot_heatmap_aggregation,
    plot_metric_vs_aggregation,
)


def make_results_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame([
        {
            'freq_agg_step': f,
            'pickup_agg_step': p,
            'agg_method': method,
            'r2_mean': 0.7 + np.random.randn() * 0.05,
            'r2_std': 0.03,
            'mse_mean': 0.1 + np.random.rand() * 0.01,
            'mse_std': 0.01,
            'mae_mean': 0.2 + np.random.rand() * 0.01,
            'mae_std': 0.01,
            'n_features': max(1, 2418 // (f * p)),
        }
        for f in [1, 2, 3]
        for p in [1, 2, 3]
        for method in ['mean', 'median']
    ])


def test_plot_metric_vs_aggregation():
    fig = plot_metric_vs_aggregation(make_results_df(), metric='r2')
    assert fig is not None
    plt.close(fig)


def test_plot_heatmap_aggregation():
    fig = plot_heatmap_aggregation(make_results_df(), metric='r2')
    assert fig is not None
    plt.close(fig)


def test_plot_methods_comparison():
    fig = plot_agg_methods_comparison(make_results_df(), metric='r2')
    assert fig is not None
    plt.close(fig)


def test_plot_all_experiments_bar():
    fig = plot_all_experiments_bar(make_results_df(), metric='r2')
    assert fig is not None
    plt.close(fig)


def test_plot_features_vs_metric():
    fig = plot_features_vs_metric(make_results_df(), metric='r2')
    assert fig is not None
    plt.close(fig)


def test_load_all_summaries_on_missing_dir_returns_empty(tmp_path: Path):
    df = load_all_summaries(tmp_path / 'missing_experiment')
    assert df.empty


def test_extract_metrics_from_excel_if_available():
    excel_path = Path('experiments/aggregation_study/freq1_pickup1_mean/results/metrics.xlsx')
    if not excel_path.exists():
        return

    metrics = extract_metrics_from_excel(excel_path)
    assert 'r2_mean' in metrics
    assert 'mse_mean' in metrics
