"""
Эксперимент с PLS (Partial Least Squares).

PLS — самостоятельный регрессионный метод, нейросеть не требуется.
Сравнение качества при разном количестве компонент.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.experiment import ExperimentResult
from ..evaluation.metrics import plot_feature_count_vs_time, plot_relative_training_time
from ..preprocessing.pls import PLSTransformer, plot_pls_comparison
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

        all_target_cols = [c for c in train.columns if c.startswith('H')]
        self.original_n_features = len([c for c in train.columns if c not in all_target_cols])

        if base_dir is None:
            from ..config import EXPERIMENTS_DIR
            base_dir = EXPERIMENTS_DIR / experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []
        self.transformers: Dict[int, PLSTransformer] = {}
        # Уникальное имя логгера — каждая сложность пишет в свой файл
        logger_name = f'pls.{self.experiment_name}.{self.base_dir.name}'
        self.logger = setup_logger(logger_name, self.base_dir / 'logs' / 'experiment.log')

        self.logger.info("PLS experiment initialized | name=%s | original_n_features=%s | targets=%s",
                         self.experiment_name, self.original_n_features, target_columns)

    def run_single(
        self,
        n_components: int,
        **kwargs,
    ) -> ExperimentResult:
        """PLS не использует нейросеть — это самостоятельный регрессор."""
        self.logger.info("PLS run started | requested_n_components=%s", n_components)

        start_time = time.time()

        # Выделяем признаки и таргет
        all_target_cols = [c for c in self.train.columns if c.startswith('H')]
        feature_cols = [c for c in self.train.columns if c not in all_target_cols]

        X_train = self.train[feature_cols]
        X_valid = self.valid[feature_cols]
        X_test = self.test[feature_cols]

        y_train = self.train[self.target_columns]
        y_valid = self.valid[self.target_columns]
        y_test = self.test[self.target_columns]

        # Обучаем PLS (n_components может быть capped)
        transformer = PLSTransformer(n_components=n_components)
        transformer.fit(X_train, y_train)
        actual_n_components = transformer.n_components_used
        config_name = f"pls_{actual_n_components}"
        variance_explained = transformer.cumulative_variance[-1] if transformer.cumulative_variance is not None else np.nan
        self.transformers[actual_n_components] = transformer

        self.logger.info("PLS fitted | config=%s | actual_n=%s | cum_var=%.6f",
                         config_name, actual_n_components, variance_explained)

        # Предсказания
        y_pred_train = transformer.predict(X_train)
        y_pred_valid = transformer.predict(X_valid)
        y_pred_test = transformer.predict(X_test)

        total_time = time.time() - start_time

        # Метрики
        def _calc_metrics(y_true, y_pred):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr

            output_dim = y_true.shape[1] if y_true.ndim > 1 else 1
            r2v, msev, maev, pv = [], [], [], []
            for i in range(output_dim):
                yt = y_true[:, i] if output_dim > 1 else y_true.flatten()
                yp = y_pred[:, i] if output_dim > 1 else y_pred.flatten()
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

        test_metrics = _calc_metrics(y_test.values, y_pred_test)
        valid_metrics = _calc_metrics(y_valid.values, y_pred_valid)

        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='pls',
            n_iter=1,
            n_features=actual_n_components,
            n_samples_train=len(X_train),
            n_samples_valid=len(X_valid),
            n_samples_test=len(X_test),
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
        )

        self.results.append(result)

        # Сохраняем summary
        summary_path = self.base_dir / config_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        # Сохраняем предсказания
        preds_dir = self.base_dir / config_name / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(y_pred_test, columns=self.target_columns).to_csv(preds_dir / "test_predictions.csv", index=False)
        pd.DataFrame(y_pred_valid, columns=self.target_columns).to_csv(preds_dir / "valid_predictions.csv", index=False)

        self.logger.info("PLS run completed | config=%s | time=%.2fs | test_r2=%.4f | valid_r2=%.4f",
                         config_name, total_time, test_metrics['r2_mean'], valid_metrics['r2_mean'])

        return result

    def run_grid(self, n_components_list: List[int] = None, **kwargs) -> pd.DataFrame:
        components_to_test = sorted(set(n_components_list)) if n_components_list else []
        self.logger.info("PLS grid started | components=%s", components_to_test)

        seen = set()
        for n_comp in components_to_test:
            try:
                result = self.run_single(n_components=n_comp, **kwargs)
                actual = result.n_components
                if actual in seen:
                    self.logger.info("PLS skipped (duplicate actual_n=%s)", actual)
                    continue
                seen.add(actual)
            except Exception as e:
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
                xlabel='Количество компонент', title='Время обучения vs количество компонент',
            )
            plot_relative_training_time(
                df, feature_col='n_components', time_col='total_time_seconds',
                reference_feature_count=self.original_n_features,
                save_path=plots_dir / "relative_training_time_vs_components.png",
                xlabel='Количество компонент', title='Относительное время обучения vs количество компонент',
            )

        return fig

    def get_best_result(self, metric: str = 'r2_mean') -> Optional[ExperimentResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda x: getattr(x, metric, 0))

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
  n_components: {best.n_components}
  variance_explained: {best.variance_explained}
  R2 = {best.r2_mean:.4f} +/- {best.r2_std:.4f}
  Compression: {best.compression_ratio:.1f}x
"""