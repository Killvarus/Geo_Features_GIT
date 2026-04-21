"""
Агрегация геофизических данных

Функционал:
- Агрегация по частотам (соседние частоты усредняются)
- Агрегация по пикетам (соседние пикеты усредняются)
- Комбинированная агрегация
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


# =============================================================================
# ПАРСИНГ ИМЁН ПРИЗНАКОВ
# =============================================================================
def parse_feature_name(name: str) -> Optional[Dict]:
    """
    Парсинг имени признака.
    
    Формат: {COMPONENT}{POLARIZATION}{FREQUENCY}_{PICKUP}
    Примеры: REYX1_1, IMYX13_31, REXY5_15
    
    Returns:
        dict с ключами: component, polarization, frequency, pickup
        или None если не удалось распарсить
    """
    # Паттерн: (RE|IM)(YX|XY|HX)(число)_(число)
    pattern = r'^(RE|IM)(YX|XY|HX)(\d+)_(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'component': match.group(1),      # RE или IM
            'polarization': match.group(2),   # YX, XY или HX
            'frequency': int(match.group(3)), # 1-13
            'pickup': int(match.group(4)),    # 1-31
            'original_name': name
        }
    return None


def parse_target_name(name: str) -> Optional[Dict]:
    """
    Парсинг имени целевой переменной.
    
    Формат: {LAYER}_{PICKUP}
    Примеры: H1_1, H2_8, H3_15
    
    Returns:
        dict с ключами: layer, pickup
        или None если не удалось распарсить
    """
    pattern = r'^(H[123])_(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'layer': match.group(1),      # H1, H2 или H3
            'pickup': int(match.group(2)), # 1-15
            'original_name': name
        }
    return None


# =============================================================================
# КОНФИГУРАЦИЯ АГРЕГАЦИИ
# =============================================================================
@dataclass
class AggregationConfig:
    """
    Конфигурация агрегации данных
    
    Attributes:
        freq_step: шаг агрегации по частотам (1 = без агрегации)
        pickup_step: шаг агрегации по пикетам (1 = без агрегации)
        agg_method: метод агрегации
            - 'mean': среднее арифметическое
            - 'median': медиана
            - 'max': максимум
            - 'min': минимум
            - 'std': стандартное отклонение
            - 'range': размах (max - min)
    """
    freq_step: int = 1
    pickup_step: int = 1
    agg_method: str = 'mean'
    
    def __post_init__(self):
        if self.freq_step < 1:
            raise ValueError("freq_step должен быть >= 1")
        if self.pickup_step < 1:
            raise ValueError("pickup_step должен быть >= 1")
        
        valid_methods = ['mean', 'median', 'max', 'min', 'std', 'range']
        if self.agg_method not in valid_methods:
            raise ValueError(f"agg_method должен быть одним из: {valid_methods}")
    
    @property
    def is_active(self) -> bool:
        """Активна ли какая-либо агрегация"""
        return self.freq_step > 1 or self.pickup_step > 1
    
    def __str__(self) -> str:
        if not self.is_active:
            return "No aggregation"
        parts = []
        if self.freq_step > 1:
            parts.append(f"freq x{self.freq_step}")
        if self.pickup_step > 1:
            parts.append(f"pickup x{self.pickup_step}")
        return f"Aggregation: {', '.join(parts)} ({self.agg_method})"


# =============================================================================
# ГРУППИРОВКА ПРИЗНАКОВ ДЛЯ АГРЕГАЦИИ
# =============================================================================
def get_aggregation_groups(
    feature_names: List[str],
    config: AggregationConfig
) -> Dict[str, List[str]]:
    """
    Группировка признаков для агрегации.
    
    Returns:
        dict: {новое_имя_признака: [список_исходных_признаков]}
    """
    if not config.is_active:
        # Без агрегации - каждый признак сам по себе
        return {name: [name] for name in feature_names}
    
    groups = defaultdict(list)
    
    for name in feature_names:
        parsed = parse_feature_name(name)
        if parsed is None:
            # Не геофизический признак - оставляем как есть
            groups[name].append(name)
            continue
        
        # Вычисляем новые индексы после агрегации
        new_freq = (parsed['frequency'] - 1) // config.freq_step + 1
        new_pickup = (parsed['pickup'] - 1) // config.pickup_step + 1
        
        # Формируем новое имя
        new_name = f"{parsed['component']}{parsed['polarization']}{new_freq}_{new_pickup}"
        groups[new_name].append(name)
    
    return dict(groups)


def get_target_aggregation_groups(
    target_names: List[str],
    config: AggregationConfig
) -> Dict[str, List[str]]:
    """
    Группировка целевых переменных для агрегации.
    
    Returns:
        dict: {новое_имя_целевой: [список_исходных_целевых]}
    """
    if config.pickup_step <= 1:
        return {name: [name] for name in target_names}
    
    groups = defaultdict(list)
    
    for name in target_names:
        parsed = parse_target_name(name)
        if parsed is None:
            groups[name].append(name)
            continue
        
        new_pickup = (parsed['pickup'] - 1) // config.pickup_step + 1
        new_name = f"{parsed['layer']}_{new_pickup}"
        groups[new_name].append(name)
    
    return dict(groups)


# =============================================================================
# АГРЕГАЦИЯ ДАННЫХ
# =============================================================================
def aggregate_dataframe(
    df: pd.DataFrame,
    feature_names: List[str],
    config: AggregationConfig
) -> pd.DataFrame:
    """
    Агрегация DataFrame по заданным признакам.
    
    Args:
        df: исходный DataFrame
        feature_names: список имён признаков для агрегации
        config: конфигурация агрегации
    
    Returns:
        DataFrame с агрегированными признаками
    """
    if not config.is_active:
        return df.copy()
    
    groups = get_aggregation_groups(feature_names, config)
    
    # Выбираем метод агрегации
    agg_funcs = {
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'std': np.std,
        'range': lambda x: np.max(x, axis=1) - np.min(x, axis=1)
    }
    agg_func = agg_funcs.get(config.agg_method, np.mean)
    
    # Создаём новый DataFrame
    aggregated_data = {}
    
    for new_name, original_names in groups.items():
        if len(original_names) == 1:
            # Без агрегации
            aggregated_data[new_name] = df[original_names[0]].values
        else:
            # Агрегация
            values = df[original_names].values
            
            if config.agg_method == 'std':
                # Для std нужна специальная обработка (ddof=1 для несмещённой оценки)
                aggregated_data[new_name] = np.std(values, axis=1, ddof=1)
            else:
                aggregated_data[new_name] = agg_func(values, axis=1)
    
    return pd.DataFrame(aggregated_data, index=df.index)


def aggregate_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    config: AggregationConfig,
    target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Агрегация всех данных (train, valid, test).
    
    Args:
        train, valid, test: исходные DataFrame
        config: конфигурация агрегации
        target_columns: список целевых переменных (не агрегируются, но возвращаются)
    
    Returns:
        (train_agg, valid_agg, test_agg, new_feature_names)
    """
    print(f"\n{'='*60}")
    print(f"АГРЕГАЦИЯ ДАННЫХ")
    print(f"{'='*60}")
    print(f"Конфигурация: {config}")
    
    # Определяем признаки (не начинаются с H)
    feature_names = [col for col in train.columns if not col.startswith('H')]
    original_feature_count = len(feature_names)
    
    print(f"\nИсходных признаков: {original_feature_count}")
    
    # Агрегируем каждый набор данных
    train_agg = aggregate_dataframe(train, feature_names, config)
    valid_agg = aggregate_dataframe(valid, feature_names, config)
    test_agg = aggregate_dataframe(test, feature_names, config)
    
    # Добавляем целевые переменные
    if target_columns:
        for col in target_columns:
            if col in train.columns:
                train_agg[col] = train[col]
                valid_agg[col] = valid[col]
                test_agg[col] = test[col]
    
    new_feature_names = [col for col in train_agg.columns if not col.startswith('H')]
    new_feature_count = len(new_feature_names)
    
    print(f"После агрегации: {new_feature_count} признаков")
    print(f"Коэффициент сжатия: {original_feature_count / new_feature_count:.2f}x")
    
    # Статистика по группам
    groups = get_aggregation_groups(feature_names, config)
    group_sizes = [len(v) for v in groups.values()]
    print(f"\nСтатистика групп:")
    print(f"  Мин. размер группы: {min(group_sizes)}")
    print(f"  Макс. размер группы: {max(group_sizes)}")
    print(f"  Средний размер группы: {np.mean(group_sizes):.2f}")
    
    return train_agg, valid_agg, test_agg, new_feature_names


# =============================================================================
# АНАЛИЗ СТРУКТУРЫ ДАННЫХ
# =============================================================================
def analyze_data_structure(df: pd.DataFrame) -> Dict:
    """
    Анализ структуры геофизических данных.
    
    Returns:
        dict со статистикой по данным
    """
    features = [col for col in df.columns if not col.startswith('H')]
    targets = [col for col in df.columns if col.startswith('H')]
    
    # Группируем по компонентам, поляризациям, частотам
    by_component = defaultdict(int)
    by_polarization = defaultdict(int)
    by_frequency = defaultdict(int)
    by_pickup = defaultdict(int)
    
    for feat in features:
        parsed = parse_feature_name(feat)
        if parsed:
            by_component[parsed['component']] += 1
            by_polarization[parsed['polarization']] += 1
            by_frequency[parsed['frequency']] += 1
            by_pickup[parsed['pickup']] += 1
    
    # Анализ целевых
    target_by_layer = defaultdict(int)
    target_by_pickup = defaultdict(int)
    
    for t in targets:
        parsed = parse_target_name(t)
        if parsed:
            target_by_layer[parsed['layer']] += 1
            target_by_pickup[parsed['pickup']] += 1
    
    return {
        'total_features': len(features),
        'total_targets': len(targets),
        'n_samples': len(df),
        'by_component': dict(by_component),
        'by_polarization': dict(by_polarization),
        'n_frequencies': len(by_frequency),
        'n_pickups': len(by_pickup),
        'target_by_layer': dict(target_by_layer),
        'target_n_pickups': len(target_by_pickup)
    }


def print_data_summary(df: pd.DataFrame):
    """Вывод сводки по структуре данных"""
    stats = analyze_data_structure(df)
    
    print(f"\n{'='*60}")
    print("СТРУКТУРА ДАННЫХ")
    print(f"{'='*60}")
    print(f"Количество образцов: {stats['n_samples']}")
    print(f"Количество признаков: {stats['total_features']}")
    print(f"Количество целевых: {stats['total_targets']}")
    print(f"\nПо компонентам: {stats['by_component']}")
    print(f"По поляризации: {stats['by_polarization']}")
    print(f"Частот: {stats['n_frequencies']}, Пикетов: {stats['n_pickups']}")
    print(f"\nЦелевые по слоям: {stats['target_by_layer']}")
    print(f"Целевых пикетов: {stats['target_n_pickups']}")
