"""
Эксперимент с PLS (Partial Least Squares).

Сравнение качества модели при разном количестве компонент.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.experiment import ExperimentResult, extract_metrics_from_excel
from ..evaluation.metrics import plot_feature_count_vs_time, plot_relative_training_time
from ..preprocessing.pls import PLSTransformer, apply_pls_to_data, plot_pls_comparison
from ..utils import Data
from ..utils.logging_utils import setup_logger


class PLSExperiment:
    """Эксперимент с PLS для определения оптимального количества компонент."""

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

        self.results: List[Dict] = []
        self.transformers: Dict[int, PLSTransformer] = {}
        self.logger = setup_logger(f'pls.{self.experiment_name}', self.base_dir / 'logs' / 'experiment.log')

        self.logger.info("PLS experiment initialized | name=%s | original_n_features=%s | targets=%s",
                         self.experiment_name, self.original_n_features, target_columns)

    def run_single(
        self,
        n_components: int,
        n_iter: int = 5,
        num_epochs: int = 500,
        learning_rate: float = 0.01,
        optimizer: str = 'adam',
        patience: int = 100,
        tolerance: float = 1e-4,
        tolerance_mode: str = 'relative',
        hidden_dim: int = 32,
        batch_size: int = 64,
        save_transformed_data: bool = False,
        **kwargs,
    ) -> Dict:
        config_name = f"pls_{n_components}"
        self.logger.info("PLS run started | config=%s | requested_n_components=%s", config_name, n_components)

        precomputed_root = kwargs.get('precomputed_pls_root')
        use_precomputed = False
        variance_explained = np.nan

        if precomputed_root:
            candidate_dir = Path(precomputed_root) / config_name
            train_path = candidate_dir / 'train.csv'
            valid_path = candidate_dir / 'valid.csv'
            test_path = candidate_dir / 'test.csv'
            info_path = candidate_dir / 'pls_info.json'

            if train_path.exists() and valid_path.exists() and test_path.exists():
                use_precomputed = True
                train_pls = pd.read_csv(train_path)
                valid_pls = pd.read_csv(valid_path)
                test_pls = pd.read_csv(test_path)

                missing_train = [c for c in self.target_columns if c not in train_pls.columns]
                missing_valid = [c for c in self.target_columns if c not in valid_pls.columns]
                missing_test = [c for c in self.target_columns if c not in test_pls.columns]

                if missing_train:
                    train_pls = pd.concat([train_pls, self.train[missing_train]], axis=1)
                if missing_valid:
                    valid_pls = pd.concat([valid_pls, self.valid[missing_valid]], axis=1)
                if missing_test:
                    test_pls = pd.concat([test_pls, self.test[missing_test]], axis=1)

                if info_path.exists():
                    with open(info_path, 'r', encoding='utf-8') as f:
                        pls_info = json.load(f)
                    actual_n_components = int(pls_info.get('n_components', n_components))
                    if pls_info.get('cumulative_variance'):
                        variance_explained = float(pls_info['cumulative_variance'][-1])
                else:
                    actual_n_components = len([c for c in train_pls.columns if c.startswith('PLS')])

                transformer = None
                self.logger.info("Loaded precomputed PLS | config=%s | dir=%s | actual_n=%s",
                                 config_name, candidate_dir, actual_n_components)

        if not use_precomputed:
            train_pls, valid_pls, test_pls, transformer = apply_pls_to_data(
                self.train, self.valid, self.test,
                self.target_columns,
                n_components=n_components,
            )
            actual_n_components = transformer.n_components_used
            variance_explained = transformer.cumulative_variance[-1] if transformer.cumulative_variance is not None else np.nan
            self.transformers[actual_n_components] = transformer

            self.logger.info("PLS transformed | config=%s | actual_n=%s | cum_var=%.6f",
                             config_name, actual_n_components, variance_explained)

        data_pls = Data(train_pls, test_pls, valid_pls, self.target_columns)

        if save_transformed_data and not use_precomputed:
            data_dir = self.base_dir / config_name / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            train_pls.to_csv(data_dir / "train.csv", index=False)
            valid_pls.to_csv(data_dir / "valid.csv", index=False)
            test_pls.to_csv(data_dir / "test.csv", index=False)

            if transformer is not None:
                with open(data_dir / "pls_info.json", 'w') as f:
                    json.dump(transformer.get_info(), f, indent=2)

        results_dir = self.base_dir / config_name / "results"
        curves_dir = self.base_dir / config_name / "learning_curves"
        models_dir = self.base_dir / config_name / "models"
        results_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        excel_path = results_dir / "metrics.xlsx"
        start_time = time.time()

        from ..models.neural_network import to_excel_optimized_OLP

        all_results = to_excel_optimized_OLP(
            file_name=str(excel_path),
            n_iter=n_iter,
            X_train=data_pls.X_train,
            y_train=data_pls.y_train,
            X_valid=data_pls.X_valid,
            y_valid=data_pls.y_valid,
            X_test=data_pls.X_test,
            y_test=data_pls.y_test,
            batch_size=batch_size,
            input_dim=data_pls.n_features,
            output_dim=data_pls.n_targets,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            hidden_dim=hidden_dim,
            save_plots_dir=str(curves_dir),
            save_models_dir=str(models_dir),
            optimizer_type=optimizer,
            device=kwargs.get('device', 'auto'),
            log_file=str(self.base_dir / config_name / 'logs' / 'training.log'),
            enable_cv=kwargs.get('enable_cv', True),
        )

        total_time = time.time() - start_time

        metrics = {}
        for metric_name in ['R2', 'MSE', 'MAE', 'Pearson']:
            iter_vals = []
            for run_result in all_results:
                df_test = run_result.get('test')
                if df_test is None or metric_name not in df_test.columns:
                    continue
                vals = [float(v) for v in df_test[metric_name].dropna().values]
                if vals:
                    iter_vals.append(float(np.mean(vals)))
            if iter_vals:
                key = metric_name.lower()
                metrics[f'{key}_mean'] = float(np.mean(iter_vals))
                metrics[f'{key}_std'] = float(np.std(iter_vals))

        if not metrics:
            metrics = extract_metrics_from_excel(excel_path)

        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='pls',
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            tolerance=tolerance,
            batch_size=batch_size,
            n_iter=n_iter,
            optimizer=optimizer,
            n_features=actual_n_components,
            n_samples_train=len(train_pls),
            n_samples_valid=len(valid_pls),
            n_samples_test=len(test_pls),
            target_columns=self.target_columns,
            n_components=actual_n_components,
            original_n_features=self.original_n_features,
            variance_explained=float(variance_explained) if not np.isnan(variance_explained) else None,
            compression_ratio=self.original_n_features / actual_n_components,
            total_time_seconds=total_time,
            r2_mean=metrics.get('r2_mean', 0),
            r2_std=metrics.get('r2_std', 0),
            mse_mean=metrics.get('mse_mean', 0),
            mse_std=metrics.get('mse_std', 0),
            mae_mean=metrics.get('mae_mean', 0),
            mae_std=metrics.get('mae_std', 0),
            pearson_mean=metrics.get('pearson_mean', 0),
            pearson_std=metrics.get('pearson_std', 0),
        )

        self.results.append(result)

        summary_path = self.base_dir / config_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        self.logger.info("PLS run completed | config=%s | time=%.1fs | r2=%.4f",
                         config_name, total_time, metrics.get('r2_mean', 0))

        return result

    def run_grid(self, n_components_list: List[int] = None, **kwargs) -> pd.DataFrame:
        components_to_test = sorted(set(n_components_list)) if n_components_list else []
        self.logger.info("PLS grid started | components=%s", components_to_test)

        for n_comp in components_to_test:
            try:
                self.run_single(n_components=n_comp, **kwargs)
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

    def get_best_result(self, metric: str = 'r2_mean') -> Dict:
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.get(metric, 0))

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
  variance_explained: {best.get('variance_explained', 'N/A')}
  R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}
  Compression: {best['compression_ratio']:.1f}x
"""