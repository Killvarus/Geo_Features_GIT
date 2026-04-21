"""
Анализ зависимости измеряемого поля от параметров слоя
для целевой переменной H3_8.

Анализируем:
- Разные пикеты: центральный (15-16) и крайний (1, 31)
- Разные частоты: большая (низкая частота, глубокая зондировка), 
  маленькая (высокая частота, поверхностная зондировка), средняя
- Разные компоненты: RE/IM и поляризации YX/XY/HX
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.aggregation import parse_feature_name


def load_data_for_analysis(n_samples=1000):
    """Загрузка данных для анализа"""
    data_dir = Path("Data")
    train_path = data_dir / "mtsgrvmgn_trn.csv"
    
    # Загружаем часть данных для анализа
    df = pd.read_csv(train_path, nrows=n_samples)
    print(f"Loaded {len(df)} samples")
    
    # Целевая переменная
    target = "H3_8"
    y = df[target]
    
    # Признаки (все, кроме целевых)
    features = [col for col in df.columns if not col.startswith('H')]
    X = df[features]
    
    return X, y, df


def select_features_for_analysis(X, y):
    """
    Выбор признаков для анализа:
    - Центральный пикет (15-16)
    - Крайний пикет (1, 31)
    - Частоты: 1 (низкая, большая глубина), 13 (высокая, малая глубина), 7 (средняя)
    - Все компоненты: RE/IM и поляризации YX/XY/HX
    """
    selected_features = []
    feature_info = []
    
    # Определяем интересующие нас параметры
    central_pickups = [15, 16]
    edge_pickups = [1, 31]
    frequencies = [1, 7, 13]  # низкая, средняя, высокая
    components = ['RE', 'IM']
    polarizations = ['YX', 'XY', 'HX']
    
    for col in X.columns:
        parsed = parse_feature_name(col)
        if parsed:
            comp = parsed['component']
            pol = parsed['polarization']
            freq = parsed['frequency']
            pickup = parsed['pickup']
            
            # Проверяем, подходит ли признак
            if (freq in frequencies and 
                (pickup in central_pickups or pickup in edge_pickups) and
                comp in components and pol in polarizations):
                
                selected_features.append(col)
                feature_info.append({
                    'name': col,
                    'component': comp,
                    'polarization': pol,
                    'frequency': freq,
                    'pickup': pickup,
                    'type': 'central' if pickup in central_pickups else 'edge'
                })
    
    return selected_features, pd.DataFrame(feature_info)


def plot_scatter_dependencies(X, y, feature_df, output_dir="analysis_results"):
    """Построение scatter plots зависимости H3_8 от признаков"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Группируем по типу (центральный/крайний) и частоте
    for freq in [1, 7, 13]:
        for pickup_type in ['central', 'edge']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Фильтруем признаки
            filtered = feature_df[
                (feature_df['frequency'] == freq) & 
                (feature_df['type'] == pickup_type)
            ]
            
            if len(filtered) == 0:
                continue
            
            # Сортируем по компоненте и поляризации
            filtered = filtered.sort_values(['component', 'polarization'])
            
            # Строим scatter plots
            for idx, (_, row) in enumerate(filtered.iterrows()):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                feature_name = row['name']
                
                # Данные
                x_vals = X[feature_name]
                y_vals = y
                
                # Scatter plot
                scatter = ax.scatter(x_vals, y_vals, alpha=0.5, s=10)
                
                # Линейная регрессия для тренда
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8)
                
                # Вычисляем корреляцию
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                
                # Настройки графика
                ax.set_xlabel(f"{row['component']}{row['polarization']}", fontsize=10)
                ax.set_ylabel('H3_8', fontsize=10)
                ax.set_title(
                    f"freq={freq}, pickup={row['pickup']}\n"
                    f"corr={correlation:.3f}",
                    fontsize=9
                )
                ax.grid(True, alpha=0.3)
            
            # Убираем лишние оси
            for idx in range(len(filtered), len(axes)):
                fig.delaxes(axes[idx])
            
            # Общий заголовок
            freq_label = {1: 'low (deep)', 7: 'medium', 13: 'high (shallow)'}
            plt.suptitle(
                f"H3_8 vs Features: Frequency {freq} ({freq_label[freq]}), "
                f"Pickup Type: {pickup_type}",
                fontsize=12,
                y=1.02
            )
            plt.tight_layout()
            
            # Сохраняем
            save_path = output_dir / f"scatter_freq{freq}_{pickup_type}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()


def plot_correlation_heatmap(X, y, feature_df, output_dir="analysis_results"):
    """Тепловая карта корреляций"""
    output_dir = Path(output_dir)
    
    # Вычисляем корреляции для всех выбранных признаков
    correlations = []
    for _, row in feature_df.iterrows():
        corr = np.corrcoef(X[row['name']], y)[0, 1]
        correlations.append({
            'component': row['component'],
            'polarization': row['polarization'],
            'frequency': row['frequency'],
            'pickup': row['pickup'],
            'type': row['type'],
            'correlation': corr
        })
    
    corr_df = pd.DataFrame(correlations)
    
    # Создаём pivot таблицу для тепловой карты
    # Вариант 1: по частотам и пикетам для каждой компоненты+поляризации
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    components = ['RE', 'IM']
    polarizations = ['YX', 'XY', 'HX']
    
    for idx, comp in enumerate(components):
        for jdx, pol in enumerate(polarizations):
            ax_idx = idx * 3 + jdx
            if ax_idx >= len(axes):
                continue
                
            ax = axes[ax_idx]
            
            # Фильтруем данные
            subset = corr_df[
                (corr_df['component'] == comp) & 
                (corr_df['polarization'] == pol)
            ]
            
            if len(subset) > 0:
                # Pivot: строки - частоты, столбцы - пикеты
                pivot = subset.pivot_table(
                    index='frequency',
                    columns='pickup',
                    values='correlation',
                    aggfunc='mean'
                )
                
                # Сортируем
                pivot = pivot.reindex(
                    index=sorted(pivot.index),
                    columns=sorted(pivot.columns)
                )
                
                # Тепловая карта
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    ax=ax,
                    cbar_kws={'label': 'Correlation with H3_8'},
                    linewidths=0.5
                )
                
                ax.set_title(f'{comp}{pol}', fontsize=11)
                ax.set_xlabel('Pickup', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No data for {comp}{pol}',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{comp}{pol}', fontsize=11)
    
    plt.suptitle('Correlation of Features with H3_8', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "correlation_heatmap.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Сохраняем таблицу корреляций
    corr_df = corr_df.sort_values('correlation', ascending=False)
    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    
    print("\nTop 10 most correlated features:")
    print(corr_df.head(10).to_string())
    
    print("\nTop 10 most anti-correlated features:")
    print(corr_df.tail(10).to_string())


def plot_statistical_summary(X, y, feature_df, output_dir="analysis_results"):
    """Статистический анализ зависимостей"""
    output_dir = Path(output_dir)
    
    # Группируем корреляции по разным категориям
    correlations = []
    for _, row in feature_df.iterrows():
        corr = np.corrcoef(X[row['name']], y)[0, 1]
        correlations.append(corr)
    
    feature_df['correlation'] = correlations
    
    # 1. Средние корреляции по частотам
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # По частотам
    freq_stats = feature_df.groupby('frequency')['correlation'].agg(['mean', 'std', 'count'])
    ax1.bar(freq_stats.index, freq_stats['mean'], 
            yerr=freq_stats['std'], capsize=5)
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Mean Correlation')
    ax1.set_title('Mean Correlation by Frequency')
    ax1.grid(True, alpha=0.3)
    
    # По пикетам (тип)
    type_stats = feature_df.groupby('type')['correlation'].agg(['mean', 'std', 'count'])
    ax2.bar(type_stats.index, type_stats['mean'],
            yerr=type_stats['std'], capsize=5)
    ax2.set_xlabel('Pickup Type')
    ax2.set_ylabel('Mean Correlation')
    ax2.set_title('Mean Correlation by Pickup Type')
    ax2.grid(True, alpha=0.3)
    
    # По компонентам
    comp_stats = feature_df.groupby('component')['correlation'].agg(['mean', 'std', 'count'])
    ax3.bar(comp_stats.index, comp_stats['mean'],
            yerr=comp_stats['std'], capsize=5)
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Mean Correlation')
    ax3.set_title('Mean Correlation by Component (RE/IM)')
    ax3.grid(True, alpha=0.3)
    
    # По поляризациям
    pol_stats = feature_df.groupby('polarization')['correlation'].agg(['mean', 'std', 'count'])
    ax4.bar(pol_stats.index, pol_stats['mean'],
            yerr=pol_stats['std'], capsize=5)
    ax4.set_xlabel('Polarization')
    ax4.set_ylabel('Mean Correlation')
    ax4.set_title('Mean Correlation by Polarization')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Summary of Correlations with H3_8', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / "statistical_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Основная функция анализа"""
    print("=" * 70)
    print("ANALYSIS OF H3_8 DEPENDENCY ON FIELD PARAMETERS")
    print("=" * 70)
    
    # Загружаем данные
    print("\nLoading data...")
    X, y, df = load_data_for_analysis(n_samples=5000)
    
    # Выбираем признаки для анализа
    print("Selecting features for analysis...")
    selected_features, feature_df = select_features_for_analysis(X, y)
    
    print(f"\nSelected {len(selected_features)} features for analysis")
    print(f"Frequency distribution:")
    print(feature_df['frequency'].value_counts().sort_index())
    print(f"\nPickup type distribution:")
    print(feature_df['type'].value_counts())
    print(f"\nComponent distribution:")
    print(feature_df['component'].value_counts())
    print(f"\nPolarization distribution:")
    print(feature_df['polarization'].value_counts())
    
    # Создаём директорию для результатов
    output_dir = Path("field_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Сохраняем информацию о выбранных признаках
    feature_df.to_csv(output_dir / "selected_features.csv", index=False)
    
    # 1. Scatter plots
    print("\nCreating scatter plots...")
    plot_scatter_dependencies(X, y, feature_df, output_dir)
    
    # 2. Correlation heatmap
    print("\nCreating correlation heatmap...")
    plot_correlation_heatmap(X, y, feature_df, output_dir)
    
    # 3. Statistical summary
    print("\nCreating statistical summary...")
    plot_statistical_summary(X, y, feature_df, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()