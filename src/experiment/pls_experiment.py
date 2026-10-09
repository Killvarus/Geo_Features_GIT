"""
Эксперимент с PLS (Partial Least Squares).

PLS — самостоятельный регрессионный метод, нейросеть не требуется.
Сравнение качества при разном количестве компонент.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.experiment import ExperimentResult, load_experiment_result
from ..evaluation.metrics import plot_feature_count_vs_time, plot_relative_training_time
from ..preprocessing.pls import PLSTransformer, plot_pls_comparison
from ..utils import Data
from ..utils.logging_utils import setup_logger


class PLSExperiment:
    """Эксперимент с PLS: fit на train, predict на test, без нейросети."""

    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        target_columns: List[str],
        experiment_name: str = "pls_study",
        base_dir: Path = None,
    ):
        self.train = train
        self.valid = valid
        self.test = test
        self.target_columns = target_columns
        self.experiment_name = experiment_name

        # Data копирует сплиты один раз на job, не на каждый n_components
        self._data = Data(train, test, valid, target_columns)
        self.original_n_features = self._data.n_features

        if base_dir is None:
            from ..config import EXPERIMENTS_DIR
            base_dir = EXPERIMENTS_DIR / experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []
        self.transformers: Dict[int, PLSTransformer] = {}
        logger_name = f'pls.{self.experiment_name}.{self.base_dir.name}'
        self.logger = setup_logger(logger_name, self.base_dir / 'logs' / 'experiment.log')

        self.logger.info(
            "PLS experiment initialized | name=%s | original_n_features=%s | targets=%s",
            self.experiment_name, self.original_n_features, target_columns,
        )

    def _max_components(self) -> int:
        return min(self.original_n_features, len(self._data.X_train))

    def _config_name(self, n_components: int) -> str:
        return f"pls_{min(int(n_components), self._max_components())}"

    def run_single(
        self,
        n_components: int,
        **kwargs,
    ) -> ExperimentResult:
        """PLS не использует нейросеть — это самостоятельный регрессор."""
        actual_n_components = min(int(n_components), self._max_components())
        config_name = f"pls_{actual_n_components}"
        existing = load_experiment_result(self.base_dir / config_name / "summary.json")
        if existing is not None and not kwargs.get('force_rerun'):
            self.logger.info("Skip existing PLS config | config=%s", config_name)
            self.results.append(existing)
            return existing

        self.logger.info(
            "PLS run started | requested_n_components=%s | actual_n=%s",
            n_components, actual_n_components,
        )

        start_time = time.time()
        data = self._data

        transformer = PLSTransformer(n_components=actual_n_components)
        transformer.fit(data.X_train, data.y_train)
        actual_n_components = transformer.n_components_used
        config_name = f"pls_{actual_n_components}"
        variance_explained = (
            transformer.cumulative_variance[-1]
            if transformer.cumulative_variance is not None
            else np.nan
        )
        self.transformers[actual_n_components] = transformer

        self.logger.info(
            "PLS fitted | config=%s | actual_n=%s | cum_var=%.6f",
            config_name, actual_n_components, variance_explained,
        )
        if transformer.pls is not None and getattr(transformer.pls, 'n_iter_', None):
            max_iter = transformer.pls.max_iter
            if any(int(it) >= max_iter for it in transformer.pls.n_iter_):
                self.logger.warning(
                    "PLS NIPALS hit max_iter on some components | config=%s | max_iter=%s",
                    config_name, max_iter,
                )

        y_pred_train = transformer.predict(data.X_train)
        y_pred_valid = transformer.predict(data.X_valid)
        y_pred_test = transformer.predict(data.X_test)

        if not np.isfinite(y_pred_test).all():
            n_bad = int(np.size(y_pred_test) - np.isfinite(y_pred_test).sum())
            self.logger.warning("Non-finite PLS predictions | config=%s | count=%s", config_name, n_bad)

        total_time = time.time() - start_time

        def _calc_metrics(y_true, y_pred):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr

            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            output_dim = y_true.shape[1] if y_true.ndim > 1 else 1
            r2v, msev, maev, pv = [], [], [], []
            for i in range(output_dim):
                yt = y_true[:, i] if y_true.ndim > 1 else y_true.ravel()
                yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred.ravel()
                r2v.append(r2_score(yt, yp) if not np.all(yt == yt[0]) else 0.0)
                msev.append(mean_squared_error(yt, yp))
                maev.append(mean_absolute_error(yt, yp))
                pv.append(pearsonr(yt, yp)[0] if len(yt) > 1 else 0.0)
            return {
                'r2_mean': float(np.mean(r2v)), 'r2_std': float(np.std(r2v)),
                'mse_mean': float(np.mean(msev)), 'mse_std': float(np.std(msev)),
                'mae_mean': float(np.mean(maev)), 'mae_std': float(np.std(maev)),
                'pearson_mean': float(np.mean(pv)), 'pearson_std': float(np.std(pv)),
            }

        test_metrics = _calc_metrics(data.y_test.values, y_pred_test)
        valid_metrics = _calc_metrics(data.y_valid.values, y_pred_valid)
        train_metrics = _calc_metrics(data.y_train.values, y_pred_train)

        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='pls',
            n_iter=1,
            n_features=actual_n_components,
            n_samples_train=len(data.X_train),
            n_samples_valid=len(data.X_valid),
            n_samples_test=len(data.X_test),
            target_columns=self.target_columns,
            n_components=actual_n_components,
            original_n_features=self.original_n_features,
            variance_explained=float(variance_explained) if not np.isnan(variance_explained) else None,
            compression_ratio=self.original_n_features / actual_n_components,
            total_time_seconds=total_time,
            r2_mean=test_metrics['r2_mean'],
            r2_std=test_metrics['r2_std'],
            mse_mean=test_metrics['mse_mean'],
            mse_std=test_metrics['mse_std'],
            mae_mean=test_metrics['mae_mean'],
            mae_std=test_metrics['mae_std'],
            pearson_mean=test_metrics['pearson_mean'],
            pearson_std=test_metrics['pearson_std'],
            all_iterations_data={
                'source': 'pls_regressor',
                'n_components_requested': int(n_components),
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
            },
        )

        self.results.append(result)

        summary_path = self.base_dir / config_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        preds_dir = self.base_dir / config_name / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(y_pred_test, columns=self.target_columns).to_csv(
            preds_dir / "test_predictions.csv", index=False,
        )
        pd.DataFrame(y_pred_valid, columns=self.target_columns).to_csv(
            preds_dir / "valid_predictions.csv", index=False,
        )

        self.logger.info(
            "PLS run completed | config=%s | time=%.2fs | test_r2=%.4f | valid_r2=%.4f | train_r2=%.4f",
            config_name, total_time,
            test_metrics['r2_mean'], valid_metrics['r2_mean'], train_metrics['r2_mean'],
        )

        return result

    def run_grid(self, n_components_list: List[int] = None, **kwargs) -> pd.DataFrame:
        requested = sorted(set(n_components_list)) if n_components_list else []
        if not requested:
            return self.get_results_df()

        max_components = self._max_components()
        actual_components = sorted(set(min(n, max_components) for n in requested))

        self.logger.info(
            "PLS grid started | requested=%s | max_components=%s | actual=%s",
            requested,
            max_components,
            actual_components,
        )

        for n_comp in actual_components:
            try:
                self.run_single(n_components=n_comp, **kwargs)
            except Exception:
                self.logger.exception("PLS run failed | n_components=%s", n_comp)

        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([result.to_dict() for result in self.results])

    def plot_comparison(self, metric: str = 'r2_mean', save: bool = True) -> plt.Figure:
        df = self.get_results_df()
        if df.empty:
            print("No results to plot")
            return None

        plots_dir = self.base_dir / "plots"
        save_path = plots_dir / f"{metric}_vs_components.png" if save else None
        if save:
            plots_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_pls_comparison(df.to_dict('records'), metric=metric, save_path=save_path)

        if save:
            plot_feature_count_vs_time(
                df, feature_col='n_components', time_col='total_time_seconds',
                save_path=plots_dir / "training_time_vs_components.png",
                xlabel='Number of components', title='Training time vs components',
            )
            plot_relative_training_time(
                df, feature_col='n_components', time_col='total_time_seconds',
                reference_feature_count=self.original_n_features,
                save_path=plots_dir / "relative_training_time_vs_components.png",
                xlabel='Number of components', title='Relative training time vs components',
            )
            plt.close('all')

        return fig

    def get_best_result(self, metric: str = 'r2_mean') -> Optional[Dict]:
        """Лучший результат по метрике. Возвращает dict, как ждут раннеры."""
        if not self.results:
            return None
        best = max(self.results, key=lambda x: getattr(x, metric, 0))
        return best.to_dict()

    def summary(self) -> str:
        df = self.get_results_df()
        if df.empty:
            return "No results"

        best = self.get_best_result()
        return f"""
PLS Experiment: {self.experiment_name}
Original features: {self.original_n_features}
Tests run: {len(df)}

Best result:
  n_components: {best['n_components']}
  variance_explained: {best['variance_explained']}
  R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}
  Compression: {best['compression_ratio']:.1f}x
"""
