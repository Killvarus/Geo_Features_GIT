"""
Тепловые карты и визуализация важности признаков
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from typing import List, Tuple, Optional, Dict
from pathlib import Path


def parse_feature_name(name: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Парсинг имени признака формата REYX1_1, IMHX13_31 и т.д.
    
    Returns:
        (component, frequency, pickup)
        component: RE/IM + YX/XY/HX
        frequency: 1-13
        pickup: 1-31
    """
    match = re.match(r'([A-Z]+)(\d+)?_([\d]+)', name)
    if match:
        component = match.group(1)
        freq_or_depth = match.group(2)
        pickup = int(match.group(3))
        return component, int(freq_or_depth) if freq_or_depth else None, pickup
    return None, None, None


def parse_feature(
    selected_indices: List[int],
    selected_importances: np.ndarray,
    train: pd.DataFrame,
    vmin: float = 0,
    vmax: float = 0.04,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Создание тепловой карты важности признаков по частотам и пикетам.
    
    Args:
        selected_indices: индексы выбранных признаков
        selected_importances: важности признаков
        train: DataFrame с данными (для получения имён столбцов)
        vmin, vmax: границы цветовой шкалы
        figsize: размер фигуры
        save_path: путь для сохранения
    
    Returns:
        matplotlib Figure
    """
    feature_names = train.columns[selected_indices]
    
    parsed_data = []
    for name, importance in zip(feature_names, selected_importances):
        component, freq, pickup = parse_feature_name(name)
        if component is not None:
            parsed_data.append({
                'name': name,
                'component': component,
                'frequency': freq,
                'pickup': pickup,
                'importance': importance
            })
    
    df_parsed = pd.DataFrame(parsed_data)
    
    if df_parsed.empty:
        print("⚠️ Нет данных для визуализации")
        return None
    
    components = sorted(df_parsed['component'].unique())
    all_pickups = sorted(df_parsed['pickup'].unique())
    
    # Сетка под графики
    n_cols = min(3, len(components))
    n_rows = math.ceil(len(components) / n_cols)
    
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.ravel()
    
    for i, comp in enumerate(components):
        df_comp = df_parsed[df_parsed['component'] == comp]
        
        if comp.startswith('H'):
            # Целевые переменные (глубины)
            y_vals = sorted(df_comp['frequency'].dropna().unique()) if 'frequency' in df_comp else [1]
            y_label = 'Глубина'
        else:
            # Признаки (частоты)
            y_vals = sorted(df_comp['frequency'].dropna().unique())
            y_label = 'Частота'
        
        # Создаём pivot таблицу
        pivot = df_comp.pivot_table(
            index='frequency', 
            columns='pickup', 
            values='importance', 
            aggfunc='first'
        )
        
        # Заполняем пропуски нулями
        pivot = pivot.reindex(
            index=sorted(pivot.index),
            columns=sorted(pivot.columns)
        )
        
        sns.heatmap(
            pivot, 
            cmap='YlOrRd', 
            ax=axes[i], 
            cbar=True, 
            linewidths=0.5,
            vmin=vmin, 
            vmax=vmax
        )
        axes[i].set_title(f'Компонента: {comp}')
        axes[i].set_xlabel('Пикет')
        axes[i].set_ylabel(y_label)
    
    # Удаляем лишние оси
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Тепловая карта сохранена: {save_path}")
    
    return fig


def plot_feature_importance_heatmap(
    feature_importance: Dict[str, float],
    vmin: float = 0,
    vmax: float = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Универсальная тепловая карта важности признаков.
    
    Args:
        feature_importance: словарь {имя_признака: важность}
        vmin, vmax: границы цветовой шкалы
        figsize: размер фигуры
        save_path: путь для сохранения
    """
    df = pd.DataFrame([
        {'name': k, 'importance': v, **dict(zip(
            ['component', 'frequency', 'pickup'],
            parse_feature_name(k)
        ))}
        for k, v in feature_importance.items()
    ])
    
    if df.empty:
        print("⚠️ Нет данных для визуализации")
        return None
    
    # Автоматический vmax
    if vmax is None:
        vmax = df['importance'].quantile(0.95)
    
    components = sorted(df['component'].dropna().unique())
    n_cols = min(3, len(components))
    n_rows = math.ceil(len(components) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    
    for i, comp in enumerate(components):
        df_comp = df[df['component'] == comp]
        
        pivot = df_comp.pivot_table(
            index='frequency',
            columns='pickup',
            values='importance',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            cmap='YlOrRd',
            ax=axes[i],
            cbar=True,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax
        )
        axes[i].set_title(f'{comp}')
        axes[i].set_xlabel('Пикет')
        axes[i].set_ylabel('Частота')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Важность признаков по компонентам', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_learning_curves_comparison(
    histories: List[Dict],
    labels: List[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Сравнение кривых обучения нескольких моделей.
    
    Args:
        histories: список словарей с train_losses и val_losses
        labels: названия моделей
        figsize: размер фигуры
        save_path: путь для сохранения
    """
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(histories))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10.colors
    
    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        
        if 'train_losses' in history:
            ax1.plot(history['train_losses'], label=label, color=color)
        
        if 'val_losses' in history:
            ax2.plot(history['val_losses'], label=label, color=color)
            
            if 'best_epoch' in history:
                ax2.axvline(
                    x=history['best_epoch'],
                    color=color,
                    linestyle='--',
                    alpha=0.5
                )
    
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Обучение')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Валидация')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison_bar(
    results_df: pd.DataFrame,
    metric: str = 'r2_mean',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Столбчатая диаграмма сравнения метрик.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(results_df))
    values = results_df[metric].values
    errors = results_df.get(f'{metric.replace("_mean", "_std")}', pd.Series([0]*len(results_df))).values
    
    bars = ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7)
    
    ax.set_xlabel('Эксперимент')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Сравнение по метрике: {metric}')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['experiment_id'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
