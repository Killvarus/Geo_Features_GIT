"""
Генерация графиков для всех агрегационных экспериментов.

Для каждого эксперимента в experiments/agg_* генерирует полный набор графиков.
"""
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.experiment_plots import (
    load_all_summaries,
    generate_all_plots
)
import matplotlib.pyplot as plt


def find_aggregation_experiments(base_dir: Path = Path("experiments")):
    """Находит все папки с агрегационными экспериментами."""
    agg_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith('agg_'):
            agg_dirs.append(d)
    return sorted(agg_dirs)


def main():
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ ВСЕХ АГРЕГАЦИЙ")
    print("=" * 60)
    
    # Находим все эксперименты
    exp_dirs = find_aggregation_experiments()
    
    if not exp_dirs:
        print("Не найдены эксперименты agg_* в директории experiments/")
        return
    
    print(f"Найдено экспериментов: {len(exp_dirs)}")
    for d in exp_dirs:
        print(f"  - {d.name}")
    
    print()
    
    total_figures = 0
    
    # Для каждого эксперимента генерируем графики
    for exp_dir in exp_dirs:
        print(f"\n{'='*60}")
        print(f"Обработка: {exp_dir.name}")
        print(f"{'='*60}")
        
        # Загружаем результаты
        results_df = load_all_summaries(exp_dir)
        
        if results_df.empty:
            print(f"  Нет результатов в {exp_dir}")
            continue
        
        print(f"  Загружено экспериментов: {len(results_df)}")
        
        # Генерируем графики
        figures = generate_all_plots(
            exp_dir,
            output_dir=exp_dir / 'plots',
            metrics=['r2', 'mse', 'mae']
        )
        
        total_figures += len(figures)
        print(f"  Создано графиков: {len(figures)}")
    
    print("\n" + "=" * 60)
    print(f"ГОТОВО! Всего графиков: {total_figures}")
    print("=" * 60)


if __name__ == '__main__':
    main()
