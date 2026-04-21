"""
Сводный отчёт анализа зависимости H3_8 от параметров слоя
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.aggregation import parse_feature_name
from analyze_field_dependency import load_data_for_analysis, select_features_for_analysis


def create_summary_report():
    """Создание сводного отчёта с основными выводами"""
    print("=" * 80)
    print("АНАЛИЗ ЗАВИСИМОСТИ H3_8 ОТ ПАРАМЕТРОВ СЛОЯ")
    print("=" * 80)
    
    # Загружаем данные
    print("\n1. ЗАГРУЗКА ДАННЫХ")
    X, y, df = load_data_for_analysis(n_samples=5000)
    print(f"   Загружено {len(df)} образцов")
    print(f"   Количество признаков: {X.shape[1]}")
    print(f"   Целевая переменная: H3_8 (слой 3, пикет 8)")
    
    # Выбираем признаки для анализа
    print("\n2. ВЫБОР ПРИЗНАКОВ ДЛЯ АНАЛИЗА")
    selected_features, feature_df = select_features_for_analysis(X, y)
    print(f"   Отобрано {len(selected_features)} признаков для анализа")
    
    # Загружаем результаты корреляций
    correlations = pd.read_csv('field_analysis_results/correlations.csv')
    
    print("\n3. КЛЮЧЕВЫЕ ВЫВОДЫ")
    print("   " + "=" * 60)
    
    # Самые сильные корреляции
    top_pos = correlations.sort_values('correlation', ascending=False).head(5)
    top_neg = correlations.sort_values('correlation', ascending=True).head(5)
    
    print("\n   САМЫЕ СИЛЬНЫЕ ПОЛОЖИТЕЛЬНЫЕ КОРРЕЛЯЦИИ:")
    for _, row in top_pos.iterrows():
        print(f"     {row['component']}{row['polarization']}{row['frequency']}_{row['pickup']} "
              f"(freq={row['frequency']}, pickup={row['pickup']}, type={row['type']}): "
              f"corr = {row['correlation']:.4f}")
    
    print("\n   САМЫЕ СИЛЬНЫЕ ОТРИЦАТЕЛЬНЫЕ КОРРЕЛЯЦИИ:")
    for _, row in top_neg.iterrows():
        print(f"     {row['component']}{row['polarization']}{row['frequency']}_{row['pickup']} "
              f"(freq={row['frequency']}, pickup={row['pickup']}, type={row['type']}): "
              f"corr = {row['correlation']:.4f}")
    
    # Анализ по группам
    print("\n   СРЕДНИЕ КОРРЕЛЯЦИИ ПО ГРУППАМ:")
    print("   " + "-" * 50)
    
    # По частотам
    freq_stats = correlations.groupby('frequency')['correlation'].agg(['mean', 'std'])
    print("\n   ПО ЧАСТОТАМ:")
    for freq, stats in freq_stats.iterrows():
        freq_label = {1: 'низкая (глубокая зондировка)', 
                      7: 'средняя', 
                      13: 'высокая (поверхностная)'}
        print(f"     Частота {freq} ({freq_label[freq]}): "
              f"mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")
    
    # По типу пикетов
    type_stats = correlations.groupby('type')['correlation'].agg(['mean', 'std'])
    print("\n   ПО ТИПУ ПИКЕТОВ:")
    for pickup_type, stats in type_stats.iterrows():
        print(f"     {pickup_type} пикеты: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")
    
    # По компонентам
    comp_stats = correlations.groupby('component')['correlation'].agg(['mean', 'std'])
    print("\n   ПО КОМПОНЕНТАМ:")
    for comp, stats in comp_stats.iterrows():
        print(f"     {comp}: mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")
    
    # По поляризациям
    pol_stats = correlations.groupby('polarization')['correlation'].agg(['mean', 'std'])
    print("\n   ПО ПОЛЯРИЗАЦИЯМ:")
    for pol, stats in pol_stats.iterrows():
        pol_label = {'YX': 'YX (поперечная)', 
                     'XY': 'XY (поперечная)', 
                     'HX': 'HX (магнитная)'}
        print(f"     {pol} ({pol_label[pol]}): mean = {stats['mean']:.4f}, std = {stats['std']:.4f}")
    
    # Статистическая значимость
    print("\n   СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    print("   " + "-" * 50)
    
    total_features = len(correlations)
    significant_pos = len(correlations[correlations['correlation'] > 0.2])
    significant_neg = len(correlations[correlations['correlation'] < -0.2])
    
    print(f"     Всего признаков в анализе: {total_features}")
    print(f"     Признаков с сильной положительной корреляцией (>0.2): {significant_pos}")
    print(f"     Признаков с сильной отрицательной корреляцией (<-0.2): {significant_neg}")
    print(f"     Доля значимо коррелирующих признаков: "
          f"{(significant_pos + significant_neg) / total_features * 100:.1f}%")
    
    # Лучшие комбинации параметров
    print("\n   ЛУЧШИЕ КОМБИНАЦИИ ПАРАМЕТРОВ:")
    print("   " + "-" * 50)
    
    best_by_freq = correlations.loc[correlations.groupby('frequency')['correlation'].idxmax()]
    print("\n   Лучшие признаки по частотам:")
    for _, row in best_by_freq.iterrows():
        print(f"     Частота {row['frequency']}: {row['component']}{row['polarization']}"
              f"{row['frequency']}_{row['pickup']} (corr = {row['correlation']:.4f})")
    
    # Рекомендации для выбора признаков
    print("\n4. РЕКОМЕНДАЦИИ ДЛЯ ВЫБОРА ПРИЗНАКОВ")
    print("   " + "=" * 60)
    
    print("\n   ДЛЯ ПРЕДСКАЗАНИЯ H3_8 РЕКОМЕНДУЕТСЯ:")
    print("   1. Использовать центральные пикеты (15-16) - выше корреляция")
    print("   2. Частота 7 даёт максимальные корреляции")
    print("   3. Компонента REYX показала лучшие результаты")
    print("   4. Поляризация YX наиболее информативна")
    
    print("\n   НАИМЕНЕЕ ИНФОРМАТИВНЫЕ:")
    print("   1. Крайние пикеты (1, 31) - низкая корреляция")
    print("   2. Частота 1 даёт отрицательные корреляции")
    print("   3. Поляризация HX часто отрицательно коррелирует")
    
    # Визуализация сводных статистик
    create_summary_plots(correlations)
    
    print("\n5. ГРАФИКИ СОЗДАНЫ В:")
    print("   " + "-" * 60)
    print("   field_analysis_results/")
    print("     - scatter_freq*_*.png - scatter plots зависимости")
    print("     - correlation_heatmap.png - тепловая карта корреляций")
    print("     - statistical_summary.png - сводная статистика")
    print("     - correlations.csv - таблица корреляций")
    print("     - selected_features.csv - выбранные признаки")
    print("     - summary_report.txt - данный отчёт")
    
    # Сохраняем отчёт в файл
    save_summary_to_file(correlations, feature_df)


def create_summary_plots(correlations: pd.DataFrame):
    """Создание дополнительных сводных графиков"""
    output_dir = Path("field_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Распределение корреляций
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Гистограмма корреляций
    axes[0, 0].hist(correlations['correlation'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Корреляция с H3_8')
    axes[0, 0].set_ylabel('Количество признаков')
    axes[0, 0].set_title('Распределение корреляций')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Корреляция по частотам с error bars
    freq_stats = correlations.groupby('frequency')['correlation'].agg(['mean', 'std', 'count'])
    freq_stats['ci'] = 1.96 * freq_stats['std'] / np.sqrt(freq_stats['count'])
    axes[0, 1].errorbar(freq_stats.index, freq_stats['mean'], 
                       yerr=freq_stats['ci'], fmt='o-', capsize=5)
    axes[0, 1].set_xlabel('Частота')
    axes[0, 1].set_ylabel('Средняя корреляция')
    axes[0, 1].set_title('Корреляция по частотам')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Корреляция по типам пикетов
    type_stats = correlations.groupby('type')['correlation'].agg(['mean', 'std', 'count'])
    type_stats['ci'] = 1.96 * type_stats['std'] / np.sqrt(type_stats['count'])
    axes[1, 0].bar(type_stats.index, type_stats['mean'], 
                   yerr=type_stats['ci'], capsize=10, alpha=0.7)
    axes[1, 0].set_xlabel('Тип пикета')
    axes[1, 0].set_ylabel('Средняя корреляция')
    axes[1, 0].set_title('Корреляция по типу пикетов')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Корреляция по поляризациям
    pol_stats = correlations.groupby('polarization')['correlation'].agg(['mean', 'std', 'count'])
    pol_stats['ci'] = 1.96 * pol_stats['std'] / np.sqrt(pol_stats['count'])
    axes[1, 1].bar(pol_stats.index, pol_stats['mean'], 
                   yerr=pol_stats['ci'], capsize=10, alpha=0.7)
    axes[1, 1].set_xlabel('Поляризация')
    axes[1, 1].set_ylabel('Средняя корреляция')
    axes[1, 1].set_title('Корреляция по поляризациям')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Статистический анализ корреляций с H3_8', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Лучшие и худшие признаки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Лучшие признаки
    top_10 = correlations.nlargest(10, 'correlation')
    bars1 = ax1.barh(range(len(top_10)), top_10['correlation'])
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([f"{row['component']}{row['polarization']}{row['frequency']}_{row['pickup']}" 
                         for _, row in top_10.iterrows()], fontsize=9)
    ax1.set_xlabel('Корреляция')
    ax1.set_title('Топ-10 признаков по положительной корреляции')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Худшие признаки
    bottom_10 = correlations.nsmallest(10, 'correlation')
    bars2 = ax2.barh(range(len(bottom_10)), bottom_10['correlation'])
    ax2.set_yticks(range(len(bottom_10)))
    ax2.set_yticklabels([f"{row['component']}{row['polarization']}{row['frequency']}_{row['pickup']}" 
                         for _, row in bottom_10.iterrows()], fontsize=9)
    ax2.set_xlabel('Корреляция')
    ax2.set_title('Топ-10 признаков по отрицательной корреляции')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Лучшие и худшие признаки для предсказания H3_8', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_worst_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Тепловая карта по частотам и пикетам для каждой компоненты+поляризации
    components = correlations['component'].unique()
    polarizations = correlations['polarization'].unique()
    
    fig, axes = plt.subplots(len(components), len(polarizations), 
                             figsize=(15, 10), squeeze=False)
    
    for i, comp in enumerate(components):
        for j, pol in enumerate(polarizations):
            ax = axes[i, j]
            
            subset = correlations[
                (correlations['component'] == comp) & 
                (correlations['polarization'] == pol)
            ]
            
            if len(subset) > 0:
                # Pivot таблица
                pivot = subset.pivot_table(
                    index='frequency',
                    columns='pickup',
                    values='correlation',
                    aggfunc='mean'
                )
                
                # Заполняем пропуски
                pivot = pivot.reindex(index=sorted(subset['frequency'].unique()),
                                     columns=sorted(subset['pickup'].unique()))
                
                im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                              aspect='auto', interpolation='nearest')
                
                # Подписи
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns.astype(int))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index.astype(int))
                
                ax.set_xlabel('Пикет')
                ax.set_ylabel('Частота')
                ax.set_title(f'{comp}{pol}')
                
                # Добавляем цветовую шкалу для последнего графика
                if i == len(components) - 1 and j == len(polarizations) - 1:
                    plt.colorbar(im, ax=ax, label='Корреляция')
            else:
                ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                ax.set_title(f'{comp}{pol}')
    
    plt.suptitle('Корреляция по частотам и пикетам для каждой комбинации', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_by_component_polarization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def save_summary_to_file(correlations: pd.DataFrame, feature_df: pd.DataFrame):
    """Сохранение сводного отчёта в файл"""
    output_dir = Path("field_analysis_results")
    
    with open(output_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("АНАЛИЗ ЗАВИСИМОСТИ H3_8 ОТ ПАРАМЕТРОВ СЛОЯ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Всего признаков в анализе: {len(correlations)}\n")
        f.write(f"Целевая переменная: H3_8 (глубина слоя 3, пикет 8)\n\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ ПО ГРУППАМ:\n")
        f.write("-" * 40 + "\n")
        f.write("По частотам:\n")
        freq_stats = correlations.groupby('frequency').size()
        for freq, count in freq_stats.items():
            freq_label = {1: 'низкая', 7: 'средняя', 13: 'высокая'}
            f.write(f"  Частота {freq} ({freq_label[freq]}): {count} признаков\n")
        
        f.write("\nПо типам пикетов:\n")
        type_stats = correlations.groupby('type').size()
        for t, count in type_stats.items():
            f.write(f"  {t}: {count} признаков\n")
        
        f.write("\nПо компонентам:\n")
        comp_stats = correlations.groupby('component').size()
        for comp, count in comp_stats.items():
            f.write(f"  {comp}: {count} признаков\n")
        
        f.write("\nПо поляризациям:\n")
        pol_stats = correlations.groupby('polarization').size()
        for pol, count in pol_stats.items():
            pol_label = {'YX': 'YX', 'XY': 'XY', 'HX': 'HX'}
            f.write(f"  {pol} ({pol_label[pol]}): {count} признаков\n")
        
        f.write("\nСТАТИСТИКА КОРРЕЛЯЦИЙ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Средняя корреляция: {correlations['correlation'].mean():.4f}\n")
        f.write(f"Стандартное отклонение: {correlations['correlation'].std():.4f}\n")
        f.write(f"Минимальная корреляция: {correlations['correlation'].min():.4f}\n")
        f.write(f"Максимальная корреляция: {correlations['correlation'].max():.4f}\n")
        
        # Количество значимых корреляций
        strong_pos = len(correlations[correlations['correlation'] > 0.2])
        strong_neg = len(correlations[correlations['correlation'] < -0.2])
        moderate_pos = len(correlations[(correlations['correlation'] > 0.1) & 
                                       (correlations['correlation'] <= 0.2)])
        moderate_neg = len(correlations[(correlations['correlation'] < -0.1) & 
                                        (correlations['correlation'] >= -0.2)])
        
        f.write(f"\nКлассификация корреляций:\n")
        f.write(f"  Сильная положительная (>0.2): {strong_pos} признаков\n")
        f.write(f"  Умеренная положительная (0.1-0.2): {moderate_pos} признаков\n")
        f.write(f"  Слабая (-0.1 до 0.1): {len(correlations) - strong_pos - strong_neg - moderate_pos - moderate_neg} признаков\n")
        f.write(f"  Умеренная отрицательная (-0.2 - -0.1): {moderate_neg} признаков\n")
        f.write(f"  Сильная отрицательная (<-0.2): {strong_neg} признаков\n")
        
        f.write("\nЛУЧШИЕ ПРИЗНАКИ ДЛЯ ПРЕДСКАЗАНИЯ H3_8:\n")
        f.write("-" * 40 + "\n")
        top_10 = correlations.nlargest(10, 'correlation')
        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"{idx:2d}. {row['component']}{row['polarization']}{row['frequency']}_{row['pickup']}: "
                   f"corr = {row['correlation']:.4f} "
                   f"(freq={row['frequency']}, pickup={row['pickup']}, type={row['type']})\n")
        
        f.write("\nВЫВОДЫ И РЕКОМЕНДАЦИИ:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Наиболее информативные признаки:\n")
        f.write("   - Частота 7 (средняя) даёт максимальные корреляции\n")
        f.write("   - Центральные пикеты (15, 16) более информативны\n")
        f.write("   - Компонента RE и поляризация YX показывают лучшие результаты\n\n")
        
        f.write("2. Наименее информативные признаки:\n")
        f.write("   - Частота 1 (низкая) часто даёт отрицательные корреляции\n")
        f.write("   - Крайние пикеты (1, 31) имеют низкую корреляцию\n")
        f.write("   - Поляризация HX часто отрицательно коррелирует\n\n")
        
        f.write("3. Рекомендации для отбора признаков:\n")
        f.write("   - Использовать признаки с частотой 7 и центральными пикетами\n")
        f.write("   - Обратить внимание на REYX компоненты\n")
        f.write("   - Рассмотреть возможность фильтрации HX поляризации\n")
        f.write("   - Для агрегации: группировать по частотам 7 и пикетам 15-16\n")
        
    print(f"\nСводный отчёт сохранён в: {output_dir / 'summary_report.txt'}")


if __name__ == '__main__':
    create_summary_report()