"""
Генерация графиков по результатам экспериментов

Использование:
    python plot_results.py                    # experiments/aggregation_study
    python plot_results.py path/to/exp        # указанная директория
"""
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.experiment_plots import (
    load_all_summaries,
    generate_all_plots,
    plot_metric_vs_aggregation,
    plot_heatmap_aggregation,
    plot_all_experiments_bar,
    plot_features_vs_metric
)
import matplotlib.pyplot as plt


def main():
    # Определяем директорию эксперимента
    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])
    else:
        experiment_dir = Path('experiments')
    
    if not experiment_dir.exists():
        print(f"Директория не найдена: {experiment_dir}")
        return
    
    print(f"Эксперимент: {experiment_dir}")
    
    # Загружаем результаты
    results_df = load_all_summaries(experiment_dir)
    
    if results_df.empty:
        print("Нет результатов")
        return
    
    print(f"\nНайдено экспериментов: {len(results_df)}")
    print(f"\nСтатистика:")
    print(f"  freq_agg_step: {sorted(results_df['freq_agg_step'].unique())}")
    print(f"  pickup_agg_step: {sorted(results_df['pickup_agg_step'].unique())}")
    print(f"  agg_method: {sorted(results_df['agg_method'].unique())}")
    
    # Лучший результат
    best = results_df.loc[results_df['r2_mean'].idxmax()]
    print(f"\nЛучший результат:")
    print(f"  R2 = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}")
    print(f"  freq_step={best['freq_agg_step']}, pickup_step={best['pickup_agg_step']}, method={best['agg_method']}")
    print(f"  Признаков: {best['n_features']}")
    
    # Генерируем все графики
    output_dir = experiment_dir / 'plots'
    
    figures = generate_all_plots(
        experiment_dir,
        output_dir=output_dir,
        metrics=['r2', 'mse', 'mae']
    )
    
    print(f"\nСгенерировано графиков: {len(figures)}")
    print(f"Сохранены в: {output_dir}")
    
    # Показываем графики (опционально)
    # plt.show()


if __name__ == '__main__':
    main()
