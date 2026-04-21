"""
Построение графиков для всех агрегационных экспериментов.

Собирает результаты из всех папок experiments/agg_* и строит графики
зависимости качества от количества признаков.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple


# Настройки
EXPERIMENTS_DIR = Path("experiments")
OUTPUT_DIR = EXPERIMENTS_DIR / "all_aggregations_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Цвета для графиков
COLORS = {
    'H3_8': '#1f77b4',  # синий
    'H1_8': '#2ca02c',  # зелёный
    'H2_8': '#ff7f0e',  # оранжевый
}

MARKERS = {
    'train_3': 'o',
    'train_3_1': 's',
    'old': '^',
}


def find_aggregation_experiments() -> List[Path]:
    """Находит все папки с агрегационными экспериментами."""
    agg_dirs = []
    for d in EXPERIMENTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith('agg_'):
            agg_dirs.append(d)
    return sorted(agg_dirs)


def parse_experiment_name(name: str) -> Dict[str, str]:
    """
    Парсит название эксперимента.
    Пример: agg_train3_H3_8 -> {'dataset': 'train_3', 'target': 'H3_8'}
    """
    # Ищем паттерн H1_8, H2_8, H3_8 в названии
    import re
    target_match = re.search(r'H\d_8', name)
    if target_match:
        target = target_match.group()
    else:
        target = 'unknown'
    
    # Определяем датасет
    if 'train3_1' in name or 'train3_1' in name:
        dataset = 'train_3_1'
    elif 'train3' in name:
        dataset = 'train_3'
    elif 'old' in name:
        dataset = 'old'
    else:
        dataset = 'unknown'
    
    return {'dataset': dataset, 'target': target}


def load_experiment_results(exp_dir: Path) -> pd.DataFrame:
    """
    Загружает результаты эксперимента из all_results.csv или summary.json.
    """
    all_results_path = exp_dir / "all_results.csv"
    
    if all_results_path.exists():
        return pd.read_csv(all_results_path)
    
    # Альтернатива: собираем из subfolders
    results = []
    for subdir in exp_dir.iterdir():
        if subdir.is_dir() and subdir.name != 'plots':
            summary_path = subdir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
    
    if results:
        return pd.DataFrame(results)
    
    return pd.DataFrame()


def collect_all_results() -> pd.DataFrame:
    """Собирает результаты всех агрегационных экспериментов."""
    all_data = []
    
    exp_dirs = find_aggregation_experiments()
    print(f"Найдено экспериментов: {len(exp_dirs)}")
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        parsed = parse_experiment_name(exp_name)
        
        df = load_experiment_results(exp_dir)
        if not df.empty:
            df['experiment_name'] = exp_name
            df['dataset'] = parsed['dataset']
            df['target'] = parsed['target']
            all_data.append(df)
            print(f"  {exp_name}: {len(df)} записей")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def plot_metric_vs_features(
    df: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Path = None
):
    """
    График зависимости метрики от количества признаков.
    Разными цветами - разные таргеты, разными маркерами - разные наборы данных.
    """
    if df.empty or metric not in df.columns:
        print(f"Нет данных для {metric}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Группируем по dataset и target
    for dataset in df['dataset'].unique():
        for target in df['target'].unique():
            subset = df[(df['dataset'] == dataset) & (df['target'] == target)]
            if subset.empty:
                continue
            
            label = f"{dataset} + {target}"
            color = COLORS.get(target, 'gray')
            marker = MARKERS.get(dataset, 'o')
            
            ax.scatter(
                subset['n_features'],
                subset[metric],
                c=color,
                marker=marker,
                s=100,
                label=label,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Соединяем линией для наглядности
            subset_sorted = subset.sort_values('n_features')
            ax.plot(
                subset_sorted['n_features'],
                subset_sorted[metric],
                c=color,
                alpha=0.3,
                linestyle='--'
            )
    
    ax.set_xlabel('Количество признаков после агрегации', fontsize=12)
    ax.set_ylabel(metric.replace('_mean', '').upper(), fontsize=12)
    ax.set_title(f'Зависимость {metric.replace("_mean", "")} от количества признаков (агрегация)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")
    
    return fig


def plot_by_target(
    df: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Path = None
):
    """
    Отдельные графики для каждого таргета.
    """
    if df.empty:
        return
    
    targets = df['target'].unique()
    n_targets = len(targets)
    
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]
    
    for ax, target in zip(axes, targets):
        subset = df[df['target'] == target]
        
        for dataset in subset['dataset'].unique():
            ds_subset = subset[subset['dataset'] == dataset]
            marker = MARKERS.get(dataset, 'o')
            
            ax.scatter(
                ds_subset['n_features'],
                ds_subset[metric],
                marker=marker,
                s=80,
                label=dataset,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Линия
            ds_sorted = ds_subset.sort_values('n_features')
            ax.plot(
                ds_sorted['n_features'],
                ds_sorted[metric],
                alpha=0.3,
                linestyle='--'
            )
        
        ax.set_xlabel('Количество признаков')
        ax.set_ylabel(metric.replace('_mean', '').upper())
        ax.set_title(f'Таргет: {target}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")
    
    return fig


def plot_heatmap_summary(
    df: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Path = None
):
    """
    Сводная тепловая карта: лучшие результаты по каждому (dataset, target).
    """
    if df.empty:
        return
    
    # Находим лучший результат для каждого (dataset, target)
    best_idx = df.groupby(['dataset', 'target'])[metric].idxmax()
    best_df = df.loc[best_idx]
    
    # Пивот-таблица
    pivot = best_df.pivot_table(
        values=metric,
        index='dataset',
        columns='target',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    
    # Подписи
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Значения в ячейках
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_mean', '').upper())
    
    ax.set_xlabel('Таргет')
    ax.set_ylabel('Набор данных')
    ax.set_title(f'Лучшее {metric.replace("_mean", "")} для каждого (dataset, target)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Тепловая карта сохранена: {save_path}")
    
    return fig


def plot_aggregation_params_vs_metric(
    df: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Path = None
):
    """
    График зависимости метрики от параметров агрегации (freq_step, pickup_step).
    """
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # По freq_agg_step
    freq_agg = df.groupby('freq_agg_step').agg({
        metric: ['mean', 'std'],
        'n_features': 'first'
    }).reset_index()
    freq_agg.columns = ['freq_agg_step', f'{metric}_mean', f'{metric}_std', 'n_features']
    
    axes[0].errorbar(
        freq_agg['freq_agg_step'],
        freq_agg[f'{metric}_mean'],
        yerr=freq_agg[f'{metric}_std'],
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8
    )
    axes[0].set_xlabel('Шаг агрегации по частотам (freq_step)')
    axes[0].set_ylabel(metric.replace('_mean', '').upper())
    axes[0].set_title('Зависимость от шага агрегации по частотам')
    axes[0].grid(True, alpha=0.3)
    
    # Вторая ось - количество признаков
    ax2 = axes[0].twinx()
    ax2.plot(freq_agg['freq_agg_step'], freq_agg['n_features'], 'r--', marker='s', alpha=0.7)
    ax2.set_ylabel('Количество признаков', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # По pickup_agg_step
    pickup_agg = df.groupby('pickup_agg_step').agg({
        metric: ['mean', 'std'],
        'n_features': 'first'
    }).reset_index()
    pickup_agg.columns = ['pickup_agg_step', f'{metric}_mean', f'{metric}_std', 'n_features']
    
    axes[1].errorbar(
        pickup_agg['pickup_agg_step'],
        pickup_agg[f'{metric}_mean'],
        yerr=pickup_agg[f'{metric}_std'],
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8,
        color='green'
    )
    axes[1].set_xlabel('Шаг агрегации по пикетам (pickup_step)')
    axes[1].set_ylabel(metric.replace('_mean', '').upper())
    axes[1].set_title('Зависимость от шага агрегации по пикетам')
    axes[1].grid(True, alpha=0.3)
    
    # Вторая ось
    ax3 = axes[1].twinx()
    ax3.plot(pickup_agg['pickup_agg_step'], pickup_agg['n_features'], 'r--', marker='s', alpha=0.7)
    ax3.set_ylabel('Количество признаков', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")
    
    return fig


def main():
    print("=" * 60)
    print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ ВСЕХ АГРЕГАЦИЙ")
    print("=" * 60)
    
    # Сбор данных
    print("\n[1] Сбор результатов экспериментов...")
    df = collect_all_results()
    
    if df.empty:
        print("Не найдены результаты экспериментов!")
        return
    
    print(f"\nВсего записей: {len(df)}")
    print(f"Датасеты: {df['dataset'].unique().tolist()}")
    print(f"Таргеты: {df['target'].unique().tolist()}")
    
    # Сохраняем сводный CSV
    df.to_csv(OUTPUT_DIR / "all_aggregation_results.csv", index=False)
    print(f"\n[OK] Сводка сохранена: {OUTPUT_DIR / 'all_aggregation_results.csv'}")
    
    # Графики
    print("\n[2] Построение графиков...")
    
    # 1. Общий график: метрика vs признаки
    plot_metric_vs_features(
        df, metric='r2_mean',
        save_path=OUTPUT_DIR / "r2_vs_features.png"
    )
    
    # 2. По каждому таргету
    plot_by_target(
        df, metric='r2_mean',
        save_path=OUTPUT_DIR / "r2_by_target.png"
    )
    
    # 3. Тепловая карта лучших результатов
    plot_heatmap_summary(
        df, metric='r2_mean',
        save_path=OUTPUT_DIR / "heatmap_best.png"
    )
    
    # 4. Зависимость от параметров агрегации
    plot_aggregation_params_vs_metric(
        df, metric='r2_mean',
        save_path=OUTPUT_DIR / "r2_vs_agg_params.png"
    )
    
    # 5. MSE
    plot_metric_vs_features(
        df, metric='mse_mean',
        save_path=OUTPUT_DIR / "mse_vs_features.png"
    )
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print(f"Графики сохранены в: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
