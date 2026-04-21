"""
Управление экспериментами

Функционал:
- Создание и отслеживание экспериментов
- Сохранение результатов в структурированном виде
- Сравнение экспериментов
- Экспорт результатов для анализа
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# =============================================================================
# РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА
# =============================================================================
@dataclass
class ExperimentResult:
    """Результаты одного эксперимента"""
    experiment_id: str
    experiment_name: str
    timestamp: str

    # Параметры агрегации
    freq_agg_step: int = 1
    pickup_agg_step: int = 1
    agg_method: str = 'mean'

    # Параметры модели
    hidden_dim: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 1000
    patience: int = 100
    tolerance: float = 1e-4
    batch_size: int = 64
    n_iter: int = 5
    optimizer: str = 'sgd'
    momentum: float = 0.9

    # Тип и размерность данных
    experiment_type: str = 'aggregation'
    n_components: Optional[int] = None
    original_n_features: Optional[int] = None
    variance_explained: Optional[float] = None
    compression_ratio: Optional[float] = None

    n_features: int = 0
    n_samples_train: int = 0
    n_samples_valid: int = 0
    n_samples_test: int = 0
    target_columns: List[str] = field(default_factory=list)

    # Метрики
    r2_mean: float = 0.0
    r2_std: float = 0.0
    mse_mean: float = 0.0
    mse_std: float = 0.0
    mae_mean: float = 0.0
    mae_std: float = 0.0
    pearson_mean: float = 0.0
    pearson_std: float = 0.0

    # Время выполнения
    total_time_seconds: float = 0.0

    # Дополнительные данные
    all_iterations_data: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentResult':
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# =============================================================================
# МЕНЕДЖЕР ЭКСПЕРИМЕНТОВ
# =============================================================================
class ExperimentManager:
    """
    Менеджер для управления экспериментами.

    Структура директорий:
    experiments/
    ├── experiment_name/
    │   ├── config.json
    │   ├── results/
    │   │   └── metrics.xlsx
    │   ├── learning_curves/
    │   │   └── curve_*.png
    │   ├── aggregated_data/
    │   │   ├── train.csv
    │   │   ├── valid.csv
    │   │   └── test.csv
    │   └── summary.json
    └── all_experiments.csv
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._results_cache: Dict[str, ExperimentResult] = {}

    def create_experiment(
        self,
        name: str,
        config: Dict,
        description: str = ""
    ) -> Path:
        """Создание нового эксперимента."""
        exp_dir = self.base_dir / name

        if exp_dir.exists():
            print(f"[!] Эксперимент '{name}' уже существует")
        else:
            exp_dir.mkdir(parents=True)

        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "learning_curves").mkdir(exist_ok=True)
        (exp_dir / "aggregated_data").mkdir(exist_ok=True)

        config_data = {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            **config,
        }

        with open(exp_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Создан эксперимент: {name}")
        print(f"   Директория: {exp_dir}")

        return exp_dir

    def save_results(
        self,
        experiment_name: str,
        result: ExperimentResult,
        metrics_df: Optional[pd.DataFrame] = None
    ):
        """Сохранение результатов эксперимента."""
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        summary_path = exp_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        if metrics_df is not None:
            metrics_path = exp_dir / "results" / "detailed_metrics.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(metrics_path, index=False)

        self._results_cache[experiment_name] = result
        self._update_all_experiments_table()

        print(f"[OK] Результаты сохранены: {summary_path}")

    def save_aggregated_data(
        self,
        experiment_name: str,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame
    ):
        """Сохранение агрегированных данных."""
        agg_dir = self.base_dir / experiment_name / "aggregated_data"
        agg_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(agg_dir / "train.csv", index=False)
        valid.to_csv(agg_dir / "valid.csv", index=False)
        test.to_csv(agg_dir / "test.csv", index=False)

        print(f"[OK] Агрегированные данные сохранены: {agg_dir}")

    def load_result(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Загрузка результатов эксперимента."""
        if experiment_name in self._results_cache:
            return self._results_cache[experiment_name]

        summary_path = self.base_dir / experiment_name / "summary.json"
        if not summary_path.exists():
            return None

        with open(summary_path, 'r', encoding='utf-8') as f:
            result = ExperimentResult.from_dict(json.load(f))

        self._results_cache[experiment_name] = result
        return result

    def list_experiments(self) -> List[str]:
        """Список всех экспериментов."""
        return [
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

    def get_all_results(self) -> pd.DataFrame:
        """Получить DataFrame со всеми результатами."""
        results = []

        for exp_name in self.list_experiments():
            result = self.load_result(exp_name)
            if result:
                results.append(result.to_dict())

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.sort_values('timestamp', ascending=False)

    def compare_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
        metrics: List[str] = ['r2_mean', 'mse_mean', 'mae_mean']
    ) -> pd.DataFrame:
        """Сравнение экспериментов по метрикам."""
        if experiment_names is None:
            experiment_names = self.list_experiments()

        comparison_data = []
        for name in experiment_names:
            result = self.load_result(name)
            if result:
                row = {
                    'experiment': name,
                    'freq_agg': result.freq_agg_step,
                    'pickup_agg': result.pickup_agg_step,
                    'n_features': result.n_features,
                }
                for metric in metrics:
                    row[metric] = getattr(result, metric, None)
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def _update_all_experiments_table(self):
        """Обновление сводной таблицы всех экспериментов."""
        df = self.get_all_results()
        if not df.empty:
            df.to_csv(self.base_dir / "all_experiments.csv", index=False)

    def export_for_analysis(self, output_path: Path):
        """Экспорт всех результатов для анализа."""
        df = self.get_all_results()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path / "experiments_summary.csv", index=False)

        with pd.ExcelWriter(output_path / "experiments_analysis.xlsx", engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Summary', index=False)

            if 'freq_agg_step' in df.columns:
                agg_summary = df.groupby(['freq_agg_step', 'pickup_agg_step']).agg({
                    'r2_mean': ['mean', 'std'],
                    'mse_mean': ['mean', 'std'],
                    'n_features': 'first',
                }).round(4)
                agg_summary.to_excel(writer, sheet_name='By_Aggregation')

        print(f"[OK] Данные экспортированы в: {output_path}")


# =============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С РЕЗУЛЬТАТАМИ
# =============================================================================
def extract_metrics_from_excel(excel_path: Path) -> Dict:
    """
    Извлечение метрик из Excel файла.

    Формат Excel (sheet 'Summary'):
        Target   Metric      Mean       Std       Min       Max
        Target_1       R2  0.749108  0.042745  0.690151  0.790152
        Target_1      MSE  0.000197  0.000034  0.000165  0.000244
        ...
    """
    metrics = {}

    try:
        summary_df = pd.read_excel(excel_path, sheet_name='Summary')

        for _, row in summary_df.iterrows():
            metric_name = str(row['Metric']).lower()
            mean_val = row['Mean']
            std_val = row['Std']

            metrics[f'{metric_name}_mean'] = float(mean_val) if pd.notna(mean_val) else 0
            metrics[f'{metric_name}_std'] = float(std_val) if pd.notna(std_val) else 0

    except Exception as e:
        print(f"Ошибка при чтении {excel_path}: {e}")

    return metrics
