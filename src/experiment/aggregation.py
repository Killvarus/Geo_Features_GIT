"""
Проведение экспериментов по агрегации данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import List, Optional, Dict

from ..config import PROJECT_ROOT, EXPERIMENTS_DIR
from ..utils import Data
from ..preprocessing import aggregate_features, AggregationConfig
from ..preprocessing.aggregation import print_data_summary
from ..evaluation import ExperimentManager, ExperimentResult
from ..evaluation.metrics import (
    plot_feature_count_vs_time,
    plot_heatmap_aggregation,
    plot_metric_vs_aggregation,
    plot_relative_training_time,
)
from ..evaluation.experiment import extract_metrics_from_excel
from ..utils.logging_utils import setup_logger


class AggregationExperiment:
    """
    Класс для проведения экспериментов по агрегации геофизических данных.
    
    Использует класс Data для корректного разделения признаков и целей.
    
    Пример использования:
    ---------------------
    from src.utils import load_mtz_data
    from src.experiment import AggregationExperiment
    
    # Загрузка
    train, valid, test = load_mtz_data(train_path, valid_path, test_path)
    
    # Эксперимент
    experiment = AggregationExperiment(
        experiment_name="my_study",
        train=train, valid=valid, test=test,
        target_columns=['H1_8', 'H2_8', 'H3_8']
    )
    
    # Запуск
    results = experiment.run_grid(
        freq_steps=[1, 2, 3],
        pickup_steps=[1, 2, 3],
        agg_methods=['mean', 'median'],
        n_iter=5
    )
    
    experiment.plot_comparison()
    """
    
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        target_columns: List[str],
        experiment_name: str = None,
        base_dir: Path = None
    ):
        """
        Args:
            train, valid, test: исходные данные (все колонки)
            target_columns: целевые переменные (например, ['H1_8', 'H2_8', 'H3_8'])
            experiment_name: название эксперимента (авто, если None)
            base_dir: базовая директория (по умолчанию experiments/experiment_name)
        """
        # Используем класс Data для разделения
        self.data = Data(train, test, valid, target_columns)
        
        # Сохраняем исходные данные для агрегации
        self._train_raw = train
        self._valid_raw = valid
        self._test_raw = test
        
        self.target_columns = target_columns
        
        # Название эксперимента
        if experiment_name is None:
            experiment_name = f"agg_{'_'.join(target_columns)}"
        self.experiment_name = experiment_name
        
        self.base_dir = base_dir or EXPERIMENTS_DIR / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.manager = ExperimentManager(self.base_dir)
        self.results = []
        self.logger = setup_logger(f'aggregation.{self.experiment_name}', self.base_dir / 'logs' / 'experiment.log')
    
        self.logger.info("Experiment initialized | name=%s | targets=%s | n_features=%s | n_targets=%s", self.experiment_name, self.target_columns, self.data.n_features, self.data.n_targets)
    
    def run_single(
        self,
        freq_step: int = 1,
        pickup_step: int = 1,
        agg_method: str = 'mean',
        n_iter: int = 5,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = 'sgd',
        patience: int = 100,
        tolerance: float = 1e-4,
        tolerance_mode: str = 'relative',
        hidden_dim: int = 32,
        batch_size: int = 64,
        save_transformed_data: bool = False,
        **kwargs
    ) -> ExperimentResult:
        """
        Запуск одного эксперимента с заданными параметрами агрегации.
        
        Args:
            freq_step: шаг агрегации по частотам
            pickup_step: шаг агрегации по пикетам
            agg_method: метод агрегации ('mean', 'median', 'max', 'min', 'std', 'range')
            n_iter: количество итераций обучения
            num_epochs: макс. эпох
            learning_rate: скорость обучения
            optimizer: оптимизатор ('sgd' или 'adam')
            patience: early stopping patience
            tolerance: early stopping tolerance
            hidden_dim: размер скрытого слоя
            batch_size: размер батча
            tolerance_mode: режим tolerance ('relative' или 'absolute')
        """
        config_name = f"freq{freq_step}_pickup{pickup_step}_{agg_method}"
        self.logger.info("Run started | config=%s | freq_step=%s | pickup_step=%s | agg_method=%s | save_transformed_data=%s", config_name, freq_step, pickup_step, agg_method, save_transformed_data)
        
        # Агрегация ИСХОДНЫХ данных (с целевыми переменными)
        agg_config = AggregationConfig(
            freq_step=freq_step,
            pickup_step=pickup_step,
            agg_method=agg_method
        )
        
        train_agg, valid_agg, test_agg, new_features = aggregate_features(
            self._train_raw, self._valid_raw, self._test_raw,
            agg_config, self.target_columns
        )
        
        self.logger.info("Aggregation completed | config=%s | n_features=%s", config_name, len(new_features))
        
        # Создаём Data из агрегированных данных
        data_agg = Data(train_agg, test_agg, valid_agg, self.target_columns)
        
        # Сохраняем агрегированные данные только если включено
        if save_transformed_data:
            agg_dir = self.base_dir / config_name / "aggregated_data"
            agg_dir.mkdir(parents=True, exist_ok=True)
            train_agg.to_csv(agg_dir / "train.csv", index=False)
            valid_agg.to_csv(agg_dir / "valid.csv", index=False)
            test_agg.to_csv(agg_dir / "test.csv", index=False)
        
        # Обучение
        results_dir = self.base_dir / config_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        curves_dir = self.base_dir / config_name / "learning_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        models_dir = self.base_dir / config_name / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        excel_path = results_dir / "metrics.xlsx"
        
        start_time = time.time()
        
        # Импортируем функцию обучения
        from ..models.neural_network import to_excel_optimized_OLP
        
        to_excel_optimized_OLP(
            file_name=str(excel_path),
            n_iter=n_iter,
            X_train=data_agg.X_train,
            y_train=data_agg.y_train,
            X_valid=data_agg.X_valid,
            y_valid=data_agg.y_valid,
            X_test=data_agg.X_test,
            y_test=data_agg.y_test,
            batch_size=batch_size,
            input_dim=data_agg.n_features,
            output_dim=data_agg.n_targets,
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
        
        # Извлекаем метрики
        metrics = extract_metrics_from_excel(excel_path)
        
        # Создаём результат
        result = ExperimentResult(
            experiment_id=config_name,
            experiment_name=self.experiment_name,
            timestamp=pd.Timestamp.now().isoformat(),
            experiment_type='aggregation',
            freq_agg_step=freq_step,
            pickup_agg_step=pickup_step,
            agg_method=agg_method,
            n_features=len(new_features),
            n_samples_train=len(train_agg),
            n_samples_valid=len(valid_agg),
            n_samples_test=len(test_agg),
            target_columns=self.target_columns,
            n_iter=n_iter,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer=optimizer,
            patience=patience,
            tolerance=tolerance,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
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
        self.manager.save_results(config_name, result)
        
        self.logger.info("Run completed | config=%s | total_time=%.1fs | r2=%.4f | r2_std=%.4f", config_name, total_time, metrics.get('r2_mean', 0), metrics.get('r2_std', 0))
        
        return result
    
    def run_grid(
        self,
        freq_steps: List[int] = [1, 2, 3],
        pickup_steps: List[int] = [1, 2, 3],
        agg_methods: List[str] = ['mean'],
        **kwargs
    ) -> pd.DataFrame:
        """
        Запуск сетки экспериментов с разными параметрами агрегации.
        """
        total = len(freq_steps) * len(pickup_steps) * len(agg_methods)
        
        self.logger.info("Grid started | freq_steps=%s | pickup_steps=%s | agg_methods=%s | total=%s", freq_steps, pickup_steps, agg_methods, total)
        
        completed = 0
        for freq_step in freq_steps:
            for pickup_step in pickup_steps:
                for agg_method in agg_methods:
                    try:
                        self.run_single(
                            freq_step=freq_step,
                            pickup_step=pickup_step,
                            agg_method=agg_method,
                            **kwargs
                        )
                        completed += 1
                    except Exception as e:
                        self.logger.exception("Run failed | freq_step=%s | pickup_step=%s | agg_method=%s", freq_step, pickup_step, agg_method)

        self.logger.info("Grid completed | completed=%s | total=%s", completed, total)
        
        return self.get_results_df()
    
    def get_results_df(self) -> pd.DataFrame:
        """Получение DataFrame с результатами"""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def plot_comparison(self, save: bool = True):
        """Визуализация сравнения результатов"""
        df = self.get_results_df()
        if df.empty:
            print("Нет результатов для визуализации")
            return
        
        plots_dir = self.base_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_metric_vs_aggregation(
            df, metric='r2_mean',
            save_path=plots_dir / "metric_vs_aggregation.png" if save else None
        )
        
        plot_feature_count_vs_time(
            df,
            feature_col='n_features',
            time_col='total_time_seconds',
            save_path=plots_dir / "training_time_vs_features.png" if save else None,
        )

        plot_relative_training_time(
            df,
            feature_col='n_features',
            time_col='total_time_seconds',
            reference_feature_count=self.data.n_features,
            save_path=plots_dir / "relative_training_time_vs_features.png" if save else None,
        )
        
        if len(df['freq_agg_step'].unique()) > 1 and len(df['pickup_agg_step'].unique()) > 1:
            plot_heatmap_aggregation(
                df, metric='r2_mean',
                save_path=plots_dir / "heatmap.png" if save else None
            )
        
        print(f"✅ Графики: {plots_dir}")
    
    def get_best_result(self, metric: str = 'r2_mean') -> pd.Series:
        """Получение лучшего результата по метрике"""
        df = self.get_results_df()
        if df.empty:
            return None
        return df.loc[df[metric].idxmax()]
    
    def summary(self) -> str:
        """Краткая сводка результатов"""
        df = self.get_results_df()
        if df.empty:
            return "Нет результатов"
        
        best = self.get_best_result()
        
        return f"""
Эксперимент: {self.experiment_name}
Запусков: {len(df)}

Лучший результат:
  freq_step={best['freq_agg_step']}, pickup_step={best['pickup_agg_step']}, method={best['agg_method']}
  R² = {best['r2_mean']:.4f} ± {best['r2_std']:.4f}
  Признаков: {best['n_features']}
"""
