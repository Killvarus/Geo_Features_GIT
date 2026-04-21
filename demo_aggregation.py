"""
Демонстрация логики агрегации

Показывает, какие именно частоты и пикеты группируются при разных шагах.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.aggregation import (
    parse_feature_name,
    get_aggregation_groups,
    AggregationConfig
)


def demonstrate_aggregation(freq_step: int, pickup_step: int, n_freq: int = 13, n_pickup: int = 31):
    """
    Демонстрация группировки частот и пикетов.
    """
    print(f"\n{'='*70}")
    print(f"AGGREGATION: freq_step={freq_step}, pickup_step={pickup_step}")
    print(f"{'='*70}")
    
    # Группировка частот
    print(f"\n--- FREQUENCY GROUPING (total {n_freq}) ---")
    freq_groups = {}
    for freq in range(1, n_freq + 1):
        new_freq = (freq - 1) // freq_step + 1
        if new_freq not in freq_groups:
            freq_groups[new_freq] = []
        freq_groups[new_freq].append(freq)
    
    for group_id, freqs in sorted(freq_groups.items()):
        print(f"  Group {group_id}: freqs {freqs} -> new freq {group_id}")
    
    print(f"\n  Total freq groups: {len(freq_groups)} (was {n_freq})")
    
    # Группировка пикетов
    print(f"\n--- PICKUP GROUPING (total {n_pickup}) ---")
    pickup_groups = {}
    for pickup in range(1, n_pickup + 1):
        new_pickup = (pickup - 1) // pickup_step + 1
        if new_pickup not in pickup_groups:
            pickup_groups[new_pickup] = []
        pickup_groups[new_pickup].append(pickup)
    
    # Показываем первые и последние группы
    group_ids = sorted(pickup_groups.keys())
    for i, group_id in enumerate(group_ids):
        pickups = pickup_groups[group_id]
        if i < 3 or i >= len(group_ids) - 2:
            print(f"  Group {group_id}: pickups {pickups} -> new pickup {group_id}")
        elif i == 3:
            print(f"  ...")
    
    print(f"\n  Total pickup groups: {len(pickup_groups)} (was {n_pickup})")
    
    # Общее количество признаков
    n_components = 6  # RE/IM * YX/XY/HX
    total_original = n_components * n_freq * n_pickup
    total_aggregated = n_components * len(freq_groups) * len(pickup_groups)
    
    print(f"\n--- RESULT ---")
    print(f"  Original features: {total_original}")
    print(f"  After aggregation: {total_aggregated}")
    print(f"  Compression ratio: {total_original / total_aggregated:.2f}x")
    
    return freq_groups, pickup_groups


def show_feature_mapping(freq_step: int, pickup_step: int, examples: list = None):
    """
    Показать, как конкретные признаки преобразуются.
    """
    if examples is None:
        examples = ['REYX1_1', 'REYX13_31', 'IMXY5_15', 'REHX7_20']
    
    print(f"\n--- ПРИМЕРЫ ПРЕОБРАЗОВАНИЯ ПРИЗНАКОВ (freq_step={freq_step}, pickup_step={pickup_step}) ---")
    print(f"{'Исходный':<15} {'Комп.':<8} {'Частота':<20} {'Пикет':<20} {'Новый':<15}")
    print("-" * 80)
    
    for name in examples:
        parsed = parse_feature_name(name)
        if parsed:
            new_freq = (parsed['frequency'] - 1) // freq_step + 1
            new_pickup = (parsed['pickup'] - 1) // pickup_step + 1
            new_name = f"{parsed['component']}{parsed['polarization']}{new_freq}_{new_pickup}"
            
            # Определяем диапазон исходных частот/пикетов
            freq_range = list(range(
                (new_freq - 1) * freq_step + 1,
                min(new_freq * freq_step + 1, 14)
            ))
            pickup_range = list(range(
                (new_pickup - 1) * pickup_step + 1,
                min(new_pickup * pickup_step + 1, 32)
            ))
            
            freq_str = f"{parsed['frequency']} (гр.{freq_range})"
            pickup_str = f"{parsed['pickup']} (гр.{pickup_range[:5]}...)"
            
            print(f"{name:<15} {parsed['component']}{parsed['polarization']:<5} "
                  f"{freq_str:<20} {pickup_str:<20} {new_name:<15}")


def main():
    print("=" * 70)
    print("AGGREGATION LOGIC DEMONSTRATION")
    print("=" * 70)
    
    # Показываем разные шаги
    for freq_step in [1, 2, 3]:
        for pickup_step in [1, 2, 3]:
            demonstrate_aggregation(freq_step, pickup_step)
    
    # Примеры преобразования
    print("\n" + "=" * 70)
    print("FEATURE TRANSFORMATION EXAMPLES")
    print("=" * 70)
    
    show_feature_mapping(freq_step=2, pickup_step=1)
    show_feature_mapping(freq_step=1, pickup_step=2)
    show_feature_mapping(freq_step=2, pickup_step=2)
    show_feature_mapping(freq_step=3, pickup_step=3)


if __name__ == '__main__':
    main()
