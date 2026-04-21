"""
Конфигурация проекта
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json

# =============================================================================
# БАЗОВЫЕ ПУТИ
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Данные
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Результаты экспериментов
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Логи
LOGS_DIR = PROJECT_ROOT / "logs"

# Конфигурации
CONFIGS_DIR = PROJECT_ROOT / "configs"

# =============================================================================
# СТРУКТУРА ПРИЗНАКОВ
# =============================================================================
# Формат: {COMPONENT}{POLARIZATION}{FREQUENCY}_{PICKUP}
# Например: REYX1_1, IMYX13_31

COMPONENT_TYPES = ['RE', 'IM']  # Действительная и мнимая части
POLARIZATION_TYPES = ['YX', 'XY', 'HX']  # Типы поляризации
FREQUENCIES = list(range(1, 14))  # 13 частот (1-13)
PICKUPS = list(range(1, 32))  # 31 пикет (1-31)

# Целевые переменные (глубины)
TARGET_LAYERS = ['H1', 'H2', 'H3']
TARGET_PICKUPS = list(range(1, 16))  # 15 пикетов для целевых

# Стандартные целевые для экспериментов
DEFAULT_TARGETS = ['H1_8', 'H2_8', 'H3_8']

# =============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# =============================================================================
@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    name: str
    description: str = ""
    
    # Агрегация
    freq_agg_step: int = 1  # Шаг агрегации по частотам
    pickup_agg_step: int = 1  # Шаг агрегации по пикетам
    
    # Модель
    hidden_dim: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 1000
    patience: int = 100
    batch_size: int = 64
    n_iter: int = 5
    optimizer: str = 'sgd'
    momentum: float = 0.9
    
    # Целевые переменные

    target_columns: List[str] = field(default_factory=lambda: ['H3_8'])
    
    # Пути
    base_dir: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'freq_agg_step': self.freq_agg_step,
            'pickup_agg_step': self.pickup_agg_step,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'n_iter': self.n_iter,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'target_columns': self.target_columns,

            'base_dir': self.base_dir,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
    
    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ
# =============================================================================
def setup_directories():
    """Создание всех необходимых директорий"""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                     EXPERIMENTS_DIR, LOGS_DIR, CONFIGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_experiment_dir(experiment_name: str) -> Path:
    """Получить директорию эксперимента"""
    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_results_dir(experiment_name: str) -> Path:
    """Получить директорию результатов"""
    results_dir = get_experiment_dir(experiment_name) / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def get_learning_curves_dir(experiment_name: str) -> Path:
    """Получить директорию кривых обучения"""
    curves_dir = get_experiment_dir(experiment_name) / "learning_curves"
    curves_dir.mkdir(exist_ok=True)
    return curves_dir


def get_aggregated_data_dir(experiment_name: str) -> Path:
    """Получить директорию агрегированных данных"""
    agg_dir = get_experiment_dir(experiment_name) / "aggregated_data"
    agg_dir.mkdir(exist_ok=True)
    return agg_dir
