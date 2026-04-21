"""Проверка связи между ALL_3 и ALL_3_1"""
import pandas as pd
import numpy as np

all3 = pd.read_csv('Data/ALL_3.csv')
all3_1 = pd.read_csv('Data/ALL_3_1.csv')

print('=== Уникальные значения ===')
print(f'ALL_3 - уникальных H3_8: {all3["H3_8"].nunique()}')
print(f'ALL_3_1 - уникальных H3_8: {all3_1["H3_8"].nunique()}')

# Проверяем по ключевым колонкам
key_cols = ['H3_8', 'REYX7_16', 'G_1']

# Создаём ключи
all3_key = all3[key_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
all3_1_key = all3_1[key_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)

# Проверяем пересечение
matches = all3_1_key.isin(all3_key).sum()
print(f'\nСтрок ALL_3_1, которые есть в ALL_3: {matches} из {len(all3_1)}')

# Проверяем последние 10000 строк
last_10k = all3.iloc[-10000:]
last_10k_key = last_10k[key_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
match_last = all3_1_key.isin(last_10k_key).sum()
print(f'Совпадение с последними 10000 строк: {match_last}')

# Первые 10000
first_10k = all3.iloc[:10000]
first_10k_key = first_10k[key_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
match_first = all3_1_key.isin(first_10k_key).sum()
print(f'Совпадение с первыми 10000 строк: {match_first}')

# Центральные 10000 (строки 10000-20000)
mid_10k = all3.iloc[10000:20000]
mid_10k_key = mid_10k[key_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
match_mid = all3_1_key.isin(mid_10k_key).sum()
print(f'Совпадение с центральными 10000: {match_mid}')

# Вывод
print('\n=== ВЫВОД ===')
if match_first + match_mid + match_last > 9000:
    print('ALL_3_1 - это подвыборка из ALL_3 (или наоборот)')
    if match_first > 8000:
        print('  → скорее всего, первые 10000 строк')
    elif match_mid > 8000:
        print('  → скорее всего, центральные 10000 строк')
    elif match_last > 8000:
        print('  → скорее всего, последние 10000 строк')
else:
    print('ALL_3_1 - это НЕ подвыборка ALL_3')
    print('Это либо полностью отдельные данные, либо данные с изменениями')
