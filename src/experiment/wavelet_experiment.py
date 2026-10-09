"""
Эксперимент с дискретным вейвлет-преобразованием сеток МТЗ.

Fit DWT на train, затем та же OLP, что в PCA-экспериментах.
Таргеты приклеиваются только запрошенные, по длине сплита, без concat.
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

from ..evaluation.experiment import ExperimentResult, extract_metrics_from_excel, load_experiment_result
from ..evaluation.metrics import plot_feature_count_vs_time, plot_relative_training_time
from ..preprocessing.wavelet import (
    apply_wavelet_to_data,
    filter_haar_results,
    plot_wavelet_comparison,
    plot_wavelet_energy,
    wavelet_config_name,
)
from ..utils import Data
from ..utils.logging_utils import setup_logger


class WaveletExperiment:
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        target_columns: List[str],
        experiment_name: str = "wavelet_study",
        base_dir: Path = None,
    ):
        self.train = train
        self.valid = valid
        self.test = test
        self.target_columns = list(target_columns)
        self.experiment_name = experiment_name

        all_target_cols = [c for c in train.columns if c.startswith('H')]
        self.original_n_features = len([c for c in train.columns if c not in all_target_cols])

        if base_dir is None:
            from ..config import EXPERIMENTS_DIR
            base_dir = EXPERIMENTS_DIR / experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []
        self.transformers: Dict[str, object] = {}
        self.logger = setup_logger(
            f'wavelet.{self.experiment_name}',
            self.base_dir / 'logs' / 'experiment.log',
        )
        self.logger.info(
            "Wavelet experiment initialized | name=%s | original_n_features=%s | targets=%s",
            self.experiment_name,
            self.original_n_features,
            target_columns,
        )

    def run_single(
        self,
        wavelet: str = 'db4',
        level: int = 2,
        keep: str = 'approx',
        layout: str = 'grid2d',
        energy_ratio: Optional[float] = None,
        n_features: Optional[int] = None,
        n_iter: int = 3,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = 'adam',
        momentum: float = 0.9,
        patience: int = 300,
        tolerance: float = 0.003,
        tolerance_mode: str = 'relative',
        hidden_dim: int = 32,
        batch_size: int = 128,
        save_transformed_data: bool = False,
        **kwargs
    ) -> Dict:
        config_name = wavelet_config_name(
            wavelet=wavelet,
            level=level,
            keep=keep,
            layout=layout,
            energy_ratio=energy_ratio,
            n_features=n_features,
        )
        existing = load_experiment_result(self.base_dir / config_name / "summary.json")
        if existing is not None and not kwargs.get('force_rerun'):
            self.logger.info("Skip existing wavelet config | config=%s", config_name)
            self.results.append(existing)
            return existing

        self.logger.info(
            "Wavelet run started | config=%s | wavelet=%s | level=%s | keep=%s",
            config_name, wavelet, level, keep,
        )

        train_w, valid_w, test_w, transformer = apply_wavelet_to_data(
            self.train,
            self.valid,
            self.test,
            self.target_columns,
            wavelet=wavelet,
            level=level,
            keep=keep,
            layout=layout,
            energy_ratio=energy_ratio,
            n_features=n_features,
        )
        extra_h = [
            c for c in train_w.columns
            if c.startswith('H') and c not in self.target_columns
        ]
        if extra_h:
            raise RuntimeError(f"В wavelet-признаки попали лишние таргеты: {extra_h}")
        if len(train_w) != len(self.train) or len(valid_w) != len(self.valid) or len(test_w) != len(self.test):
            raise RuntimeError("Длины wavelet-сплитов не совпали с исходными")

        actual_n_features = transformer.n_features_used
        energy_retained = transformer.energy_retained
        self.transformers[config_name] = transformer
        self.logger.info(
            "Wavelet transformed | config=%s | n_features=%s | energy_retained=%s",
            config_name,
            actual_n_features,
            energy_retained,
        )

        data_w = Data(train_w, test_w, valid_w, self.target_columns)

        if save_transformed_data:
            data_dir = self.base_dir / config_name / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            train_w.to_csv(data_dir / "train.csv", index=False)
            valid_w.to_csv(data_dir / "valid.csv", index=False)
            test_w.to_csv(data_dir / "test.csv", index=False)
            with open(data_dir / "wavelet_info.json", 'w', encoding='utf-8') as f:
                json.dump(transformer.get_info(), f, indent=2)
            plot_wavelet_energy(transformer, save_path=data_dir / "wavelet_energy.png")
            plt.close('all')

        results_dir = self.base_dir / config_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        curves_dir = self.base_dir / config_name / "learning_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        models_dir = self.base_dir / config_name / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        from ..models.neural_network import to_excel_optimized_OLP

        start_time = time.time()
        all_results = to_excel_optimized_OLP(
            file_name=str(results_dir / "metrics.xlsx"),
            n_iter=n_iter,
            X_train=data_w.X_train,
            y_train=data_w.y_train,
            X_valid=data_w.X_valid,
            y_valid=data_w.y_valid,
            X_test=data_w.X_test,
            y_test=data_w.y_test,
            batch_size=batch_size,
            input_dim=data_w.n_features,
            output_dim=data_w.n_targets,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            hidden_dim=hidden_dim,
            save_plots_dir=str(curves_dir),
            save_models_dir=str(models_dir),
            optimizer_type=optimizer,
            momentum=momentum,
            device=kwargs.get('device', 'auto'),
            log_file=str(self.base_dir / config_name / 'logs' / 'training.log'),
            enable_cv=kwargs.get('enable_cv', False),
        )
        total_time = time.time() - start_time

        iter_times = [r['history']['total_time'] for r in all_results]
        total_time_mean = float(np.mean(iter_times))
        total_time_std = float(np.std(iter_times, ddof=1)) if len(iter_times) > 1 else 0.0

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
            metrics = extract_metrics_from_excel(results_dir / "metrics.xlsx")

        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='wavelet',
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            tolerance=tolerance,
            batch_size=batch_size,
            n_iter=n_iter,
            optimizer=optimizer,
            momentum=momentum,
            n_features=actual_n_features,
            n_samples_train=len(train_w),
            n_samples_valid=len(valid_w),
            n_samples_test=len(test_w),
            target_columns=self.target_columns,
            n_components=actual_n_features,
            original_n_features=self.original_n_features,
            variance_explained=energy_retained,
            compression_ratio=self.original_n_features / actual_n_features if actual_n_features else None,
            wavelet=wavelet,
            wavelet_level=int(transformer.effective_level_ or level),
            wavelet_keep=keep,
            wavelet_layout=layout,
            total_time_seconds=total_time_mean,
            total_time_std=total_time_std,
            r2_mean=metrics.get('r2_mean', 0),
            r2_std=metrics.get('r2_std', 0),
            mse_mean=metrics.get('mse_mean', 0),
            mse_std=metrics.get('mse_std', 0),
            mae_mean=metrics.get('mae_mean', 0),
            mae_std=metrics.get('mae_std', 0),
            pearson_mean=metrics.get('pearson_mean', 0),
            pearson_std=metrics.get('pearson_std', 0),
        )
        _ = total_time

        self.results.append(result)
        summary_path = self.base_dir / config_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)

        diag_dir = self.base_dir / config_name
        plot_wavelet_energy(transformer, save_path=diag_dir / "wavelet_energy.png")
        plt.close('all')
        with open(diag_dir / "wavelet_info.json", 'w', encoding='utf-8') as f:
            json.dump(transformer.get_info(), f, indent=2)

        self.logger.info(
            "Wavelet run completed | config=%s | n_features=%s | r2=%.4f | time=%.1fs",
            config_name,
            actual_n_features,
            metrics.get('r2_mean', 0),
            total_time_mean,
        )
        return result

    def run_grid(self, configs: List[Dict], **kwargs) -> pd.DataFrame:
        self.logger.info("Wavelet grid started | n_configs=%s", len(configs))
        for cfg in configs:
            merged = dict(cfg)
            merged.update(kwargs)
            try:
                self.run_single(**merged)
            except Exception:
                self.logger.exception("Wavelet run failed | config=%s", cfg)
        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([result.to_dict() for result in self.results])

    def plot_comparison(self, metric: str = 'r2_mean', save: bool = True):
        df = self.get_results_df()
        if df.empty:
            return None
        plots_dir = self.base_dir / "plots"
        save_path = plots_dir / f"{metric}_vs_features.png" if save else None
        if save:
            plots_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_wavelet_comparison(df.to_dict('records'), metric=metric, save_path=save_path)
        if save and not df.empty:
            plot_feature_count_vs_time(
                df,
                feature_col='n_features',
                time_col='total_time_seconds',
                save_path=plots_dir / "training_time_vs_features.png",
                xlabel='Число вейвлет-признаков',
                title='Время обучения vs число вейвлет-признаков',
            )
            plot_relative_training_time(
                df,
                feature_col='n_features',
                time_col='total_time_seconds',
                reference_feature_count=self.original_n_features,
                save_path=plots_dir / "relative_training_time_vs_features.png",
                xlabel='Число вейвлет-признаков',
                title='Относительное время обучения vs число вейвлет-признаков',
            )
            haar_df = filter_haar_results(df)
            if not haar_df.empty:
                metric_title = metric.replace('_mean', '').upper()
                plot_wavelet_comparison(
                    haar_df.to_dict('records'),
                    metric=metric,
                    save_path=plots_dir / f"{metric}_vs_features_haar_only.png",
                    annotate=True,
                    title=f'{metric_title} vs wavelet features (только Haar)',
                    group_col=None,
                )
                plot_feature_count_vs_time(
                    haar_df,
                    feature_col='n_features',
                    time_col='total_time_seconds',
                    save_path=plots_dir / "training_time_vs_features_haar_only.png",
                    xlabel='Число вейвлет-признаков',
                    title='Время обучения vs число вейвлет-признаков (только Haar)',
                )
        return fig

    def get_best_result(self, metric: str = 'r2_mean') -> Optional[Dict]:
        if not self.results:
            return None
        best = max(self.results, key=lambda x: getattr(x, metric, 0))
        return best.to_dict()
