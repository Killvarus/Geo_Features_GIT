from .aggregation import AggregationExperiment
from .pca_experiment import PCAExperiment, compare_pca_vs_aggregation
from .pls_experiment import PLSExperiment
from .wavelet_experiment import WaveletExperiment

__all__ = [
    'AggregationExperiment',
    'PCAExperiment',
    'compare_pca_vs_aggregation',
    'PLSExperiment',
    'WaveletExperiment',
]
