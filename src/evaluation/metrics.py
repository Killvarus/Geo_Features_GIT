"""
Расчёт и сравнение метрик

Функционал:
- Расчёт метрик качества модели
- Сравнение экспериментов
- Визуализация сравнения
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# КАЛЬКУЛЯТОР МЕТРИК
# =============================================================================
class MetricsCalculator:
    """Калькулятор метрик качества модели"""
    
    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Расчёт всех метрик.
        
        Args:
            y_true: истинные значения
            y_pred: предсказанные значения
            prefix: префикс для ключей (например, 'test_')
        
        Returns:
            dict с метриками
        """
        metrics = {}
        
        # R²
        r2 = r2_score(y_true, y_pred)
        metrics[f'{prefix}r2'] = r2
        
        # MSE
        mse = mean_squared_error(y_true, y_pred)
        metrics[f'{prefix}mse'] = mse
        
        # RMSE
        rmse = np.sqrt(mse)
        metrics[f'{prefix}rmse'] = rmse
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        metrics[f'{prefix}mae'] = mae
        
        # Pearson correlation
        if len(y_true) > 1:
            pearson, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            metrics[f'{prefix}pearson'] = pearson
        else:
            metrics[f'{prefix}pearson'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_with_std(
        y_true: np.ndarray,
        y_pred_list: List[np.ndarray],
        prefix: str = ""
    ) -> Dict[str, Tuple[float, float]]:
        """
        Расчёт метрик с доверительным интервалом по нескольким предсказаниям.
        
        Returns:
            dict с (mean, std) для каждой метрики
        """
        all_metrics = []
        
        for y_pred in y_pred_list:
            metrics = MetricsCalculator.calculate_all(y_true, y_pred, prefix)
            all_metrics.append(metrics)
        
        # Агрегируем
        result = {}
        metric_names = all_metrics[0].keys()
        
        for name in metric_names:
            values = [m[name] for m in all_metrics]
            result[name] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return result


# =============================================================================
# СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ
# =============================================================================
def compare_experiments(
    results_df: pd.DataFrame,
    metric: str = 'r2_mean',
    group_by: str = 'freq_agg_step',
    ascending: bool = True
) -> pd.DataFrame:
    """
    Сравнение экспериментов по метрике.
    
    Args:
        results_df: DataFrame с результатами экспериментов
        metric: метрика для сравнения
        group_by: параметр для группировки
        ascending: сортировка по возрастанию
    
    Returns:
        DataFrame с результатами сравнения
    """
    if results_df.empty:
        return pd.DataFrame()
    
    # Группируем
    grouped = results_df.groupby(group_by).agg({
        metric: ['mean', 'std', 'count'],
        'n_features': 'first'
    }).round(4)
    
    # Выпрямляем колонки
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Сортируем
    grouped = grouped.sort_values(f'{metric}_mean', ascending=ascending)
    
    return grouped


def plot_metric_vs_aggregation(
    results_df: pd.DataFrame,
    metric: str = 'r2_mean',
    metric_std: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:

    """
    График зависимости метрики от шага агрегации.
    """
    if metric_std is None:
        metric_std = metric.replace('_mean', '_std') if metric.endswith('_mean') else f'{metric}_std'

    mean_col = metric if metric in results_df.columns else f'{metric}_mean'
    std_col = metric_std if metric_std in results_df.columns else None

    if mean_col not in results_df.columns:
        print(f"Колонка {mean_col} не найдена")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    freq_agg = {mean_col: 'mean', 'n_features': 'first'}
    if std_col:
        freq_agg[std_col] = 'mean'

    freq_data = results_df.groupby('freq_agg_step').agg(freq_agg).reset_index()

    
    axes[0].errorbar(
        freq_data['freq_agg_step'],
        freq_data[mean_col],
        yerr=freq_data[std_col] if std_col else None,
        marker='o',
        capsize=5,
        linewidth=2
    )

    axes[0].set_xlabel('Шаг агрегации по частотам')
    axes[0].set_ylabel(mean_col.replace('_mean', '').upper())

    axes[0].set_title('Зависимость метрики от агрегации по частотам')
    axes[0].grid(True, alpha=0.3)
    
    # Добавляем вторую ось для количества признаков
    ax2 = axes[0].twinx()
    ax2.plot(
        freq_data['freq_agg_step'],
        freq_data['n_features'],
        'r--',
        marker='s',
        alpha=0.7,
        label='Признаков'
    )
    ax2.set_ylabel('Количество признаков', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # По пикетам
    pickup_agg = {mean_col: 'mean', 'n_features': 'first'}
    if std_col:
        pickup_agg[std_col] = 'mean'

    pickup_data = results_df.groupby('pickup_agg_step').agg(pickup_agg).reset_index()

    
    axes[1].errorbar(
        pickup_data['pickup_agg_step'],
        pickup_data[mean_col],
        yerr=pickup_data[std_col] if std_col else None,
        marker='o',
        capsize=5,
        linewidth=2,
        color='green'
    )

    axes[1].set_xlabel('Шаг агрегации по пикетам')
    axes[1].set_ylabel(mean_col.replace('_mean', '').upper())

    axes[1].set_title('Зависимость метрики от агрегации по пикетам')
    axes[1].grid(True, alpha=0.3)
    
    # Вторая ось
    ax3 = axes[1].twinx()
    ax3.plot(
        pickup_data['pickup_agg_step'],
        pickup_data['n_features'],
        'r--',
        marker='s',
        alpha=0.7
    )
    ax3.set_ylabel('Количество признаков', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")

    
    return fig


def plot_heatmap_aggregation(
    results_df: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:

    """
    Тепловая карта метрики в зависимости от шагов агрегации.
    """
    mean_col = metric if metric in results_df.columns else f'{metric}_mean'
    if mean_col not in results_df.columns:
        print(f"Колонка {mean_col} не найдена")
        return None

    pivot = results_df.pivot_table(
        values=mean_col,

        index='freq_agg_step',
        columns='pickup_agg_step',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)

    
    import seaborn as sns
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        ax=ax,
        cbar_kws={'label': mean_col.replace('_mean', '').upper()}

    )
    
    ax.set_xlabel('Шаг агрегации по пикетам')
    ax.set_ylabel('Шаг агрегации по частотам')
    ax.set_title(f'{mean_col.replace("_mean", "").upper()} при разных шагах агрегации')

    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")

    
    return fig


def plot_feature_count_vs_time(
    results_df: pd.DataFrame,
    feature_col: str = 'n_features',
    time_col: str = 'total_time_seconds',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    xlabel: str = 'Количество признаков',
    title: str = 'Время обучения vs количество признаков'
) -> plt.Figure:
    """График зависимости времени обучения от количества признаков."""
    if feature_col not in results_df.columns or time_col not in results_df.columns:
        print(f"Колонки {feature_col} и/или {time_col} не найдены")
        return None

    plot_df = results_df[[feature_col, time_col]].dropna().copy()
    if plot_df.empty:
        print("Нет данных для построения графика времени обучения")
        return None

    plot_df = plot_df.sort_values(feature_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(plot_df[feature_col], plot_df[time_col], marker='o', linewidth=2)
    ax.scatter(plot_df[feature_col], plot_df[time_col], s=80, alpha=0.8)

    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row[time_col]:.1f}s",
            (row[feature_col], row[time_col]),
            textcoords='offset points',
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            alpha=0.7,
        )
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Время обучения, сек')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")

    
    return fig


def plot_relative_training_time(
    results_df: pd.DataFrame,
    feature_col: str = 'n_features',
    time_col: str = 'total_time_seconds',
    reference_feature_count: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    xlabel: str = 'Количество признаков',
    title: str = 'Относительное время обучения vs количество признаков'
) -> plt.Figure:
    """График относительного времени обучения относительно базовой модели на полном наборе признаков."""
    if feature_col not in results_df.columns or time_col not in results_df.columns:
        print(f"Колонки {feature_col} и/или {time_col} не найдены")
        return None

    plot_df = results_df[[feature_col, time_col]].dropna().copy()
    if plot_df.empty:
        print("Нет данных для построения графика относительного времени")
        return None

    if reference_feature_count is None:
        reference_feature_count = plot_df[feature_col].max()

    reference_rows = plot_df[plot_df[feature_col] == reference_feature_count]
    if reference_rows.empty:
        print(f"Не найден базовый запуск с {feature_col}={reference_feature_count}")
        return None

    reference_time = reference_rows[time_col].mean()
    if reference_time == 0:
        print("Базовое время обучения равно 0, относительное время не определено")
        return None

    plot_df['relative_time'] = plot_df[time_col] / reference_time
    plot_df = plot_df.sort_values(feature_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(plot_df[feature_col], plot_df['relative_time'], marker='o', linewidth=2)
    ax.scatter(plot_df[feature_col], plot_df['relative_time'], s=80, alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Базовая модель')

    for _, row in plot_df.iterrows():
        ax.annotate(
            f"{row['relative_time']:.2f}x",
            (row[feature_col], row['relative_time']),
            textcoords='offset points',
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Относительное время')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")


    return fig


def plot_all_experiments_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['r2_mean', 'mse_mean', 'mae_mean'],
    save_path: Optional[Path] = None
) -> plt.Figure:

    """
    Сравнение всех экспериментов по нескольким метрикам.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        data = results_df.sort_values(metric, ascending=(metric != 'r2_mean'))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
        
        bars = ax.barh(
            range(len(data)),
            data[metric],
            xerr=data.get(metric.replace('_mean', '_std'), 0),
            color=colors,
            capsize=3
        )
        
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([
            f"f={r['freq_agg_step']}, p={r['pickup_agg_step']}" 
            for _, r in data.iterrows()
        ])
        ax.set_xlabel(metric.replace('_mean', '').upper())
        ax.set_title(metric.replace('_mean', '').upper())
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] График сохранён: {save_path}")

    
    return fig
