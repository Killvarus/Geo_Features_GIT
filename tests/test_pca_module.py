import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.pca import (
    PCATransformer,
    apply_pca_to_data,
    find_optimal_n_components,
    plot_explained_variance,
)


def test_pca_transformer_fixed_components():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50), columns=[f'feature_{i}' for i in range(50)])

    transformer = PCATransformer(n_components=10)
    X_pca = transformer.fit_transform(X)

    assert X_pca.shape == (100, 10)
    assert transformer.n_components_used == 10
    assert len(transformer.explained_variance_ratio) == 10


def test_pca_transformer_variance_ratio():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50), columns=[f'feature_{i}' for i in range(50)])
    X['feature_0'] = X['feature_1'] + X['feature_2'] * 0.5

    transformer = PCATransformer(variance_ratio=0.90)
    X_pca = transformer.fit_transform(X)

    assert transformer.cumulative_variance[-1] >= 0.90
    assert X_pca.shape[1] == transformer.n_components_used


def test_apply_pca_to_data():
    np.random.seed(42)
    n_samples = 50

    train = pd.DataFrame({
        **{f'f{i}': np.random.randn(n_samples) for i in range(10)},
        'target': np.random.randn(n_samples),
    })
    valid = train.copy()
    test = train.copy()

    train_pca, valid_pca, test_pca, transformer = apply_pca_to_data(
        train, valid, test, target_columns=['target'], n_components=5
    )

    assert train_pca.shape[1] == 6
    assert valid_pca.shape[1] == 6
    assert test_pca.shape[1] == 6
    assert 'target' in train_pca.columns
    assert transformer.n_components_used == 5


def test_find_optimal_n_components():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))

    transformer = PCATransformer()
    transformer.fit(X)
    optimal = find_optimal_n_components(transformer)

    assert isinstance(optimal, dict)
    assert 0.90 in optimal


def test_plot_explained_variance():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 50))

    transformer = PCATransformer(n_components=20)
    transformer.fit(X)

    fig = plot_explained_variance(transformer, n_components_to_show=20)
    assert fig is not None
    plt.close(fig)
