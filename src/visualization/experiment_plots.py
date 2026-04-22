"""
Визуализация результатов экспериментов

Графики:
1. Зависимость метрики от шага агрегации (freq/pickup)
2. Тепловая карта метрик
3. Сравнение методов агрегации
4. Кривые обучения
5. Scatter предсказания vs истинные значения
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..evaluation.experiment import extract_metrics_from_excel
from ..evaluation.metrics import plot_heatmap_aggregation, plot_metric_vs_aggregation


# =============================================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ
# =============================================================================
def load_all_summaries(experiment_dir: Path) -> pd.DataFrame:
    """
    Загрузка результатов эксперимента.

    Сначала пытается читать `summary.json`, а если их нет —
    восстанавливает результаты из `*/results/metrics.xlsx`.
    
    Returns:
        DataFrame с результатами всех экспериментов
    """
    experiment_dir = Path(experiment_dir)
    results = []
    
    for summary_path in experiment_dir.glob('*/summary.json'):
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    
    if results:
        return pd.DataFrame(results)

    for metrics_path in experiment_dir.glob('*/results/metrics.xlsx'):
        config_dir = metrics_path.parent.parent
        config_name = config_dir.name

        try:
            metrics = extract_metrics_from_excel(metrics_path)
        except Exception as e:
            print(f"Не удалось прочитать metrics.xlsx: {metrics_path} | {e}")
            continue

        row = {
            'experiment_id': config_name,
            'experiment_name': experiment_dir.name,
            'timestamp': '',
            'r2_mean': metrics.get('r2_mean', 0.0),
            'r2_std': metrics.get('r2_std', 0.0),
            'mse_mean': metrics.get('mse_mean', 0.0),
            'mse_std': metrics.get('mse_std', 0.0),
            'mae_mean': metrics.get('mae_mean', 0.0),
            'mae_std': metrics.get('mae_std', 0.0),
            'pearson_mean': metrics.get('pearson_mean', 0.0),
            'pearson_std': metrics.get('pearson_std', 0.0),
        }

        if config_name.startswith('freq') and '_pickup' in config_name:
            parts = config_name.split('_')
            try:
                row.update({
                    'experiment_type': 'aggregation',
                    'freq_agg_step': int(parts[0].replace('freq', '')),
                    'pickup_agg_step': int(parts[1].replace('pickup', '')),
                    'agg_method': '_'.join(parts[2:]) if len(parts) > 2 else 'mean',
                })
            except Exception:
                row.update({
                    'experiment_type': 'aggregation',
                    'freq_agg_step': None,
                    'pickup_agg_step': None,
                    'agg_method': None,
                })
        elif config_name.startswith('pca_'):
            n_components_str = config_name.replace('pca_', '', 1)
            try:
                row.update({
                    'experiment_type': 'pca',
                    'n_components': int(n_components_str),
                    'n_features': int(n_components_str),
                })
            except Exception:
                row.update({
                    'experiment_type': 'pca',
                    'n_components': None,
                })
        else:
            row['experiment_type'] = 'unknown'

        meta = None
        try:
            meta = pd.read_excel(metrics_path, sheet_name='Meta')
        except Exception:
            meta = None

        if meta is not None and {'Parameter', 'Value'}.issubset(meta.columns):
            meta_map = dict(zip(meta['Parameter'], meta['Value']))
            if 'input_dim' in meta_map and pd.notna(meta_map['input_dim']):
                row.setdefault('n_features', int(meta_map['input_dim']))
            if 'n_iter' in meta_map and pd.notna(meta_map['n_iter']):
                row['n_iter'] = int(meta_map['n_iter'])
            if 'learning_rate' in meta_map and pd.notna(meta_map['learning_rate']):
                row['learning_rate'] = float(meta_map['learning_rate'])
            if 'num_epochs' in meta_map and pd.notna(meta_map['num_epochs']):
                row['num_epochs'] = int(meta_map['num_epochs'])
            if 'patience' in meta_map and pd.notna(meta_map['patience']):
                row['patience'] = int(meta_map['patience'])
            if 'batch_size' in meta_map and pd.notna(meta_map['batch_size']):
                row['batch_size'] = int(meta_map['batch_size'])
            if 'hidden_dim' in meta_map and pd.notna(meta_map['hidden_dim']):
                row['hidden_dim'] = int(meta_map['hidden_dim'])
            if 'optimizer' in meta_map and pd.notna(meta_map['optimizer']):
                row['optimizer'] = str(meta_map['optimizer'])

        try:
            raw_df = pd.read_excel(metrics_path, sheet_name='Raw_Results')
            if 'Time' in raw_df.columns:
                times = pd.to_numeric(raw_df['Time'], errors='coerce').dropna()
                if not times.empty:
                    row['total_time_seconds'] = float(times.mean())
        except Exception:
            pass

        results.append(row)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)




# =============================================================================
# ГРАФИК 3: СРАВНЕНИЕ МЕТОДОВ АГРЕГАЦИИ
# =============================================================================
def plot_agg_methods_comparison(
    results_df: pd.DataFrame,
    metric: str = 'r2',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Сравнение методов агрегации (mean, median, etc.)
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if 'agg_method' not in results_df.columns:
        print("Колонка agg_method не найдена")
        return None
    
    # Группировка по методу
    method_group = results_df.groupby('agg_method').agg({
        mean_col: ['mean', 'std'],
        std_col: 'mean' if std_col in results_df.columns else 'first'
    }).reset_index()
    
    method_group.columns = ['agg_method', f'{metric}_mean', f'{metric}_std_within', f'{metric}_std_across']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(method_group))
    bars = ax.bar(
        x,
        method_group[f'{metric}_mean'],
        yerr=method_group[f'{metric}_std_within'] if std_col in results_df.columns else None,
        capsize=5,
        color=['steelblue', 'forestgreen', 'coral', 'purple', 'orange'][:len(method_group)],
        alpha=0.8
    )
    
    ax.set_xlabel('Метод агрегации', fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(f'Сравнение методов агрегации по {metric.upper()}', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_group['agg_method'], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения над столбцами
    for bar, val in zip(bars, method_group[f'{metric}_mean']):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    return fig


# =============================================================================
# ГРАФИК 4: ВСЕ ЭКСПЕРИМЕНТЫ
# =============================================================================
def plot_all_experiments_bar(
    results_df: pd.DataFrame,
    metric: str = 'r2',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Столбчатая диаграмма всех экспериментов.
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in results_df.columns:
        print(f"Колонка {mean_col} не найдена")
        return None
    
    # Сортировка
    df_sorted = results_df.sort_values(mean_col, ascending=(metric != 'r2'))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(df_sorted))
    
    bars = ax.barh(
        x,
        df_sorted[mean_col],
        xerr=df_sorted[std_col] if std_col in df_sorted.columns else None,
        capsize=3,
        color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted))),
        alpha=0.8
    )
    
    # Подписи
    labels = [
        f"f={r['freq_agg_step']}, p={r['pickup_agg_step']}, {r['agg_method']}"
        for _, r in df_sorted.iterrows()
    ]
    
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(metric.upper(), fontsize=11)
    ax.set_title(f'Результаты всех экспериментов ({metric.upper()})', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения
    for bar, val in zip(bars, df_sorted[mean_col]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            ha='left',
            va='center',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    return fig


# =============================================================================
# ГРАФИК 5: КОЛИЧЕСТВО ПРИЗНАКОВ
# =============================================================================
def plot_features_vs_metric(
    results_df: pd.DataFrame,
    metric: str = 'r2',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Зависимость метрики от количества признаков.
    
    Точки раскрашены по комбинации (freq_step, pickup_step).
    Размер точки = шаг агрегации (чем больше шаг, тем больше точка).
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in results_df.columns:
        print(f"Колонка {mean_col} не найдена")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Создаём уникальные комбинации (freq_step, pickup_step)
    combinations = results_df.groupby(['freq_agg_step', 'pickup_agg_step']).size().reset_index()[['freq_agg_step', 'pickup_agg_step']]
    
    # Цвета для каждой комбинации
    n_combos = len(combinations)
    colors = plt.cm.tab20.colors[:n_combos] if n_combos <= 20 else plt.cm.tab20b.colors[:n_combos]
    
    # Рисаем каждую комбинацию
    for idx, (_, row) in enumerate(combinations.iterrows()):
        freq = row['freq_agg_step']
        pickup = row['pickup_agg_step']
        
        df_sub = results_df[
            (results_df['freq_agg_step'] == freq) & 
            (results_df['pickup_agg_step'] == pickup)
        ]
        
        # Размер точки зависит от шага (чем больше агрегация, тем больше точка)
        size = 100 + (freq + pickup) * 50
        
        # Метка для легенды
        label = f'freq={freq}, pickup={pickup}'
        
        # Если есть разные методы, рисаем их с разными маркерами
        if 'agg_method' in df_sub.columns and len(df_sub['agg_method'].unique()) > 1:
            markers = ['o', 's', '^', 'D', 'v']
            for i, method in enumerate(sorted(df_sub['agg_method'].unique())):
                df_method = df_sub[df_sub['agg_method'] == method]
                ax.scatter(
                    df_method['n_features'],
                    df_method[mean_col],
                    c=[colors[idx]],
                    marker=markers[i % len(markers)],
                    s=size,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1,
                    label=f'{label}, {method}'
                )
        else:
            ax.scatter(
                df_sub['n_features'],
                df_sub[mean_col],
                c=[colors[idx]],
                s=size,
                alpha=0.7,
                edgecolors='black',
                linewidths=1,
                label=label
            )
    
    # Добавляем аннотации с количеством признаков
    for _, row in results_df.iterrows():
        ax.annotate(
            f"{int(row['n_features'])}",
            (row['n_features'], row[mean_col]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            alpha=0.7
        )
    
    ax.set_xlabel('Количество признаков', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} vs количество признаков\n(размер точки ~ шаг агрегации)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Добавляем вертикальные линии для ориентира
    for n_feat in sorted(results_df['n_features'].unique()):
        ax.axvline(x=n_feat, color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    return fig


# =============================================================================
# ГРАФИК 6: ДЕТАЛЬНЫЙ SCATTER С АННОТАЦИЯМИ
# =============================================================================
def plot_features_vs_metric_detailed(
    results_df: pd.DataFrame,
    metric: str = 'r2',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Детальный график зависимости метрики от количества признаков.
    
    - Точки подписаны конфигурацией агрегации
    - Показаны error bars
    - Цвет по freq_step, форма по pickup_step
    """
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if mean_col not in results_df.columns:
        print(f"Колонка {mean_col} не найдена")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Уникальные значения
    freq_steps = sorted(results_df['freq_agg_step'].unique())
    pickup_steps = sorted(results_df['pickup_agg_step'].unique())
    
    # Цвета по freq_step
    freq_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(freq_steps)))
    freq_color_map = dict(zip(freq_steps, freq_colors))
    
    # Маркеры по pickup_step
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    pickup_marker_map = dict(zip(pickup_steps, markers[:len(pickup_steps)]))
    
    # Рисаем точки
    for _, row in results_df.iterrows():
        freq = row['freq_agg_step']
        pickup = row['pickup_agg_step']
        method = row.get('agg_method', 'mean')
        
        color = freq_color_map[freq]
        marker = pickup_marker_map[pickup]
        
        # Error bar
        if std_col in results_df.columns:
            ax.errorbar(
                row['n_features'],
                row[mean_col],
                yerr=row[std_col],
                fmt=marker,
                color=color,
                markersize=12,
                capsize=5,
                capthick=2,
                elinewidth=2,
                markeredgecolor='black',
                markeredgewidth=1,
                alpha=0.8
            )
        else:
            ax.scatter(
                row['n_features'],
                row[mean_col],
                c=[color],
                marker=marker,
                s=150,
                edgecolors='black',
                linewidths=1,
                alpha=0.8
            )
        
        # Аннотация
        label = f"f{freq}p{pickup}"
        if 'agg_method' in results_df.columns:
            label += f"\n{method[:3]}"
        
        ax.annotate(
            label,
            (row['n_features'], row[mean_col]),
            textcoords="offset points",
            xytext=(10, 5),
            ha='left',
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Легенда для freq_step (цвета)
    for freq in freq_steps:
        ax.scatter([], [], c=[freq_color_map[freq]], marker='o', s=100, label=f'freq_step={freq}')
    
    # Легенда для pickup_step (маркеры)
    for pickup in pickup_steps:
        ax.scatter([], [], c='gray', marker=pickup_marker_map[pickup], s=100, label=f'pickup_step={pickup}')
    
    ax.set_xlabel('Количество признаков', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} vs количество признаков\n(цвет = freq_step, маркер = pickup_step)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    return fig


# =============================================================================
# ГЕНЕРАЦИЯ ВСЕХ ГРАФИКОВ
# =============================================================================
def generate_all_plots(
    experiment_dir: Path,
    output_dir: Optional[Path] = None,
    metrics: List[str] = ['r2', 'mse', 'mae']
) -> Dict[str, plt.Figure]:
    """
    Генерация всех графиков для эксперимента.
    
    Args:
        experiment_dir: путь к директории эксперимента
        output_dir: путь для сохранения (по умолчанию experiment_dir/plots)
        metrics: список метрик для визуализации
    
    Returns:
        dict с фигурами
    """
    experiment_dir = Path(experiment_dir)
    
    if output_dir is None:
        output_dir = experiment_dir / 'plots'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Загружаем результаты
    results_df = load_all_summaries(experiment_dir)
    
    if results_df.empty:
        print(f"Нет результатов в {experiment_dir}")
        return {}
    
    print(f"Загружено {len(results_df)} экспериментов")
    
    figures = {}
    
    for metric in metrics:
        # 1. Зависимость от агрегации
        fig1 = plot_metric_vs_aggregation(
            results_df,
            metric=metric,
            save_path=output_dir / f'{metric}_vs_aggregation.png'
        )
        if fig1:
            figures[f'{metric}_vs_aggregation'] = fig1
        
        # 2. Тепловая карта
        fig2 = plot_heatmap_aggregation(
            results_df,
            metric=metric,
            save_path=output_dir / f'{metric}_heatmap.png'
        )
        if fig2:
            figures[f'{metric}_heatmap'] = fig2
    
    # 3. Сравнение методов (если есть разные методы)
    if 'agg_method' in results_df.columns and len(results_df['agg_method'].unique()) > 1:
        for metric in metrics:
            fig3 = plot_agg_methods_comparison(
                results_df,
                metric=metric,
                save_path=output_dir / f'{metric}_methods_comparison.png'
            )
            if fig3:
                figures[f'{metric}_methods_comparison'] = fig3
    
    # 4. Все эксперименты
    fig4 = plot_all_experiments_bar(
        results_df,
        metric='r2',
        save_path=output_dir / 'all_experiments_r2.png'
    )
    if fig4:
        figures['all_experiments_r2'] = fig4
    
    # 5. Признаки vs метрика (базовый)
    fig5 = plot_features_vs_metric(
        results_df,
        metric='r2',
        save_path=output_dir / 'features_vs_r2.png'
    )
    if fig5:
        figures['features_vs_r2'] = fig5
    
    # 6. Признаки vs метрика (детальный)
    fig6 = plot_features_vs_metric_detailed(
        results_df,
        metric='r2',
        save_path=output_dir / 'features_vs_r2_detailed.png'
    )
    if fig6:
        figures['features_vs_r2_detailed'] = fig6
    
    print(f"\nВсе графики сохранены в: {output_dir}")
    
    return figures


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])
    else:
        experiment_dir = Path('experiments/aggregation_study')
    
    generate_all_plots(experiment_dir)
    plt.show()
