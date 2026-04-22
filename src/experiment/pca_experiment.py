"""
Эксперимент с PCA (метод главных компонент)

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
from ..preprocessing.pca import (
    PCATransformer,
    apply_pca_to_data,
    plot_explained_variance,
    plot_pca_comparison
)
from ..utils import Data
from ..utils.logging_utils import setup_logger


class PCAExperiment:
    """
    Эксперимент с PCA для определения оптимального количества компонент.
    """
    
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        target_columns: List[str],
        experiment_name: str = "pca_study",
        base_dir: Path = None
    ):
        """
        Args:
            train, valid, test: исходные DataFrame
            target_columns: список целевых переменных
            experiment_name: название эксперимента
            base_dir: базовая директория для результатов
        """
        self.train = train
        self.valid = valid
        self.test = test
        self.target_columns = target_columns
        self.experiment_name = experiment_name
        
        # Количество исходных признаков (исключаем ВСЕ целевые по префиксу 'H')
        all_target_cols = [c for c in train.columns if c.startswith('H')]
        self.original_n_features = len([c for c in train.columns if c not in all_target_cols])
        
        # Директория
        if base_dir is None:
            from ..config import EXPERIMENTS_DIR
            base_dir = EXPERIMENTS_DIR / experiment_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Результаты
        self.results: List[Dict] = []
        self.transformers: Dict[int, PCATransformer] = {}
        self.logger = setup_logger(f'pca.{self.experiment_name}', self.base_dir / 'logs' / 'experiment.log')
        
        self.logger.info("PCA experiment initialized | name=%s | original_n_features=%s | targets=%s", self.experiment_name, self.original_n_features, target_columns)
    
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
        **kwargs
    ) -> Dict:
        """
        Запуск одного эксперимента с заданным количеством PCA компонент.
        
        Args:
            n_components: количество главных компонент (None = все)
        """
        config_name = f"pca_{n_components}" if n_components else "pca_all"
        self.logger.info("PCA run started | config=%s | requested_n_components=%s | save_transformed_data=%s", config_name, n_components, save_transformed_data)
        
        # Применяем PCA
        train_pca, valid_pca, test_pca, transformer = apply_pca_to_data(
            self.train, self.valid, self.test,
            self.target_columns,
            n_components=n_components
        )
        
        actual_n_components = transformer.n_components_used
        variance_explained = transformer.cumulative_variance[-1]
        
        self.logger.info("PCA transformed | config=%s | actual_n_components=%s | variance_explained=%.6f", config_name, actual_n_components, variance_explained)
        
        # Сохраняем трансформер
        self.transformers[actual_n_components] = transformer
        
        # Создаём Data
        data_pca = Data(train_pca, test_pca, valid_pca, self.target_columns)
        
        # Сохраняем данные только если включено
        if save_transformed_data:
            data_dir = self.base_dir / config_name / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            train_pca.to_csv(data_dir / "train.csv", index=False)
            valid_pca.to_csv(data_dir / "valid.csv", index=False)
            test_pca.to_csv(data_dir / "test.csv", index=False)
            
            # Сохраняем информацию о PCA
            pca_info = transformer.get_info()
            with open(data_dir / "pca_info.json", 'w') as f:
                json.dump(pca_info, f, indent=2)
            
            # График explained variance
            plot_explained_variance(
                transformer,
                n_components_to_show=min(50, actual_n_components),
                save_path=data_dir / "explained_variance.png"
            )
        
        # Обучение
        results_dir = self.base_dir / config_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        curves_dir = self.base_dir / config_name / "learning_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir = self.base_dir / config_name / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        excel_path = results_dir / "metrics.xlsx"
        
        start_time = time.time()
        
        from ..models.neural_network import to_excel_optimized_OLP
        
        all_results = to_excel_optimized_OLP(
            file_name=str(excel_path),
            n_iter=n_iter,
            X_train=data_pca.X_train,
            y_train=data_pca.y_train,
            X_valid=data_pca.X_valid,
            y_valid=data_pca.y_valid,
            X_test=data_pca.X_test,
            y_test=data_pca.y_test,
            batch_size=batch_size,
            input_dim=data_pca.n_features,
            output_dim=data_pca.n_targets,
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
            enable_cv=kwargs.get('enable_cv', True)
        )
    
        total_time = time.time() - start_time
        
        # Метрики: сначала считаем из in-memory результатов, fallback — из Excel
        metrics = {}
        metric_names = ['R2', 'MSE', 'MAE', 'Pearson']
        for metric_name in metric_names:
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
        
        # Результат
        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='pca',
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            tolerance=tolerance,
            batch_size=batch_size,
            n_iter=n_iter,
            optimizer=optimizer,
            n_features=actual_n_components,
            n_samples_train=len(train_pca),
            n_samples_valid=len(valid_pca),
            n_samples_test=len(test_pca),
            target_columns=self.target_columns,
            n_components=actual_n_components,
            original_n_features=self.original_n_features,
            variance_explained=variance_explained,
            compression_ratio=self.original_n_features / actual_n_components,
            total_time_seconds=total_time,
            r2_mean=metrics.get('r2_mean', 0),
            r2_std=metrics.get('r2_std', 0),
            mse_mean=metrics.get('mse_mean', 0),
            mse_std=metrics.get('mse_std', 0),
            mae_mean=metrics.get('mae_mean', 0),
            mae_std=metrics.get('mae_std', 0),
            pearson_mean=metrics.get('pearson_mean', 0),
            pearson_std=metrics.get('pearson_std', 0)
        )
    
        self.results.append(result)
        
        summary_path = self.base_dir / config_name / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info("PCA run completed | config=%s | total_time=%.1fs | r2=%.4f | r2_std=%.4f", config_name, total_time, metrics.get('r2_mean', 0), metrics.get('r2_std', 0))
        
        return result
    
    def run_grid(
        self,
        n_components_list: List[int] = None,
        variance_thresholds: List[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Запуск сетки экспериментов с разным количеством компонент.
        
        Args:
            n_components_list: список количеств компонент [10, 50, 100, ...]
            variance_thresholds: список долей дисперсии [0.90, 0.95, 0.99]
                               (преобразуется в n_components)
        """
        # Определяем список n_components
        components_to_test = []
        
        if n_components_list:
            components_to_test.extend(n_components_list)
        
        if variance_thresholds:
            # Сначала делаем PCA со всеми компонентами, чтобы найти пороги
            temp_transformer = PCATransformer()
            all_target_cols = [c for c in self.train.columns if c.startswith('H')]
            feature_cols = [c for c in self.train.columns if c not in all_target_cols]
            temp_transformer.fit(self.train[feature_cols])
            
            for threshold in variance_thresholds:
                n_comp = np.argmax(temp_transformer.cumulative_variance >= threshold) + 1
                components_to_test.append(n_comp)
        
        # Уникальные и сортировка
        components_to_test = sorted(set(components_to_test))
        components_to_test = [n for n in components_to_test if n <= self.original_n_features]
        
        # Добавляем исходные признаки (без PCA) для сравнения
        self.logger.info("PCA grid started | components=%s", components_to_test)
        
        # Запускаем эксперименты
        for n_comp in components_to_test:
            try:
                self.run_single(n_components=n_comp, **kwargs)
            except Exception as e:
                self.logger.exception("PCA run failed | n_components=%s", n_comp)
        
        return self.get_results_df()
    
    def get_results_df(self) -> pd.DataFrame:
        """Получение DataFrame с результатами."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([result.to_dict() for result in self.results])
    
    def plot_comparison(
        self,
        metric: str = 'r2_mean',
        save: bool = True
    ) -> plt.Figure:
        """
        График сравнения результатов.
        """
        import matplotlib.pyplot as plt
        
        df = self.get_results_df()
        if df.empty:
            print("No results to plot")
            return None
        
        plots_dir = self.base_dir / "plots"
        save_path = plots_dir / f"{metric}_vs_components.png" if save else None
        if save:
            plots_dir.mkdir(parents=True, exist_ok=True)

        fig = plot_pca_comparison(
            df.to_dict('records'),
            metric=metric,
            save_path=save_path
        )
    
        if save:
            plot_feature_count_vs_time(
                df,
                feature_col='n_components',
                time_col='total_time_seconds',
                save_path=plots_dir / "training_time_vs_components.png",
                xlabel='Количество компонент',
                title='Время обучения vs количество компонент',
            )
            plot_relative_training_time(
                df,
                feature_col='n_components',
                time_col='total_time_seconds',
                reference_feature_count=self.original_n_features,
                save_path=plots_dir / "relative_training_time_vs_components.png",
                xlabel='Количество компонент',
                title='Относительное время обучения vs количество компонент',
            )

        return fig
    
    def get_best_result(self, metric: str = 'r2_mean') -> Dict:
        """Получение лучшего результата по метрике."""
        if not self.results:
            return None
        return max(self.results, key=lambda x: x.get(metric, 0))
    
    def summary(self) -> str:
        """Краткая сводка результатов."""
        df = self.get_results_df()
        if df.empty:
            return "No results"
        
        best = self.get_best_result()
        
        return f"""
PCA Experiment: {self.experiment_name}
Original features: {self.original_n_features}
Tests run: {len(df)}

Best result:
  n_components: {best['n_components']}
  variance_explained: {best['variance_explained']:.2%}
  R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}
  Compression: {best['compression_ratio']:.1f}x
"""


# =============================================================================
# СРАВНЕНИЕ PCA VS АГРЕГАЦИЯ
# =============================================================================
def compare_pca_vs_aggregation(
    pca_results: pd.DataFrame,
    agg_results: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Path = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Сравнение PCA и агрегации по качеству и количеству признаков.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Левый график: R2 vs количество признаков
    ax = axes[0]
    
    # PCA
    ax.scatter(
        pca_results['n_components'],
        pca_results[metric],
        c='blue',
        s=100,
        marker='o',
        label='PCA',
        alpha=0.7
    )
    
    # Агрегация
    ax.scatter(
        agg_results['n_features'],
        agg_results[metric],
        c='red',
        s=100,
        marker='s',
        label='Aggregation',
        alpha=0.7
    )
    
    # Исходные (если есть)
    if 'original_n_features' in pca_results.columns:
        orig_n = pca_results['original_n_features'].iloc[0]
        orig_r2 = pca_results[pca_results['n_components'] == orig_n][metric].values
        if len(orig_r2) > 0:
            ax.scatter([orig_n], [orig_r2[0]], c='green', s=150, marker='*', 
                      label='Original', zorder=5)
    
    ax.set_xlabel('Number of Features / Components')
    ax.set_ylabel(metric.replace('_mean', '').upper())
    ax.set_title(f'{metric.replace("_mean", "").upper()} vs Feature Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Правый график: bar chart лучших результатов
    ax2 = axes[1]
    
    # Лучшие результаты
    best_pca = pca_results.loc[pca_results[metric].idxmax()]
    best_agg = agg_results.loc[agg_results[metric].idxmax()]
    
    methods = ['PCA', 'Aggregation']
    r2_values = [best_pca[metric], best_agg[metric]]
    n_features = [best_pca['n_components'], best_agg['n_features']]
    
    bars = ax2.bar(methods, r2_values, color=['blue', 'red'], alpha=0.7)
    
    # Добавляем аннотации
    for bar, n_feat in zip(bars, n_features):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'n={int(n_feat)}',
            ha='center',
            fontsize=10
        )
    
    ax2.set_ylabel(metric.replace('_mean', '').upper())
    ax2.set_title('Best Results Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
