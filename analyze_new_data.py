"""Базовый анализ новых данных ALL_3 и ALL_3_1"""
import pandas as pd
import numpy as np
from pathlib import Path

# Читаем данные
all3 = pd.read_csv('Data/ALL_3.csv')
all3_1 = pd.read_csv('Data/ALL_3_1.csv')
train = pd.read_csv('Data/mtsgrvmgn_trn.csv')
valid = pd.read_csv('Data/mtsgrvmgn_vld.csv')
test = pd.read_csv('Data/mtsgrvmgn_tst.csv')

print('='*60)
print('БАЗОВЫЙ АНАЛИЗ НОВЫХ ФАЙЛОВ')
print('='*60)

print('\n--- РАЗМЕРЫ ДАННЫХ ---')
print(f'Original train:  {train.shape}')
print(f'Original valid:  {valid.shape}')
print(f'Original test:   {test.shape}')
print(f'Sum original:    {train.shape[0]+valid.shape[0]+test.shape[0]}')
print(f'ALL_3:           {all3.shape}')
print(f'ALL_3_1:         {all3_1.shape}')

print('\n--- ПРОВЕРКА: ALL_3 = train + valid + test? ---')
# Проверяем первые 21000 строк (train)
train_in_all3 = all3.iloc[:21000]
match_train = np.allclose(train['H3_8'].values, train_in_all3['H3_8'].values)
print(f'H3_8 совпадает с train: {match_train}')

# Проверяем строки 21000-27000 (valid)
valid_in_all3 = all3.iloc[21000:27000]
match_valid = np.allclose(valid['H3_8'].values, valid_in_all3['H3_8'].values)
print(f'H3_8 совпадает с valid: {match_valid}')

# Проверяем строки 27000-30000 (test)
test_in_all3 = all3.iloc[27000:]
match_test = np.allclose(test['H3_8'].values, test_in_all3['H3_8'].values)
print(f'H3_8 совпадает с test:  {match_test}')

print('\n--- СТРУКТУРА КОЛОНОК ---')
print(f'ALL_3 колонок: {len(all3.columns)}')
print(f'ALL_3_1 колонок: {len(all3_1.columns)}')

# Общие и уникальные колонки
common = set(all3.columns) & set(all3_1.columns)
only_all3 = set(all3.columns) - set(all3_1.columns)
only_all3_1 = set(all3_1.columns) - set(all3.columns)

print(f'Общих: {len(common)}')
print(f'Только в ALL_3: {len(only_all3)}')
print(f'Только в ALL_3_1: {len(only_all3_1)}')

print('\n--- НОВЫЕ КОЛОНКИ В ALL_3 (G_, M_, E_) ---')
new_cols = set(all3.columns) - set(train.columns)
print(f'Всего новых: {len(new_cols)}')

# G_ колонки
g_cols = [c for c in all3.columns if c.startswith('G_')]
print(f'\nG_ колонки ({len(g_cols)}): G_1 ... G_31')
print(f'  G_1: mean={all3["G_1"].mean():.3f}, std={all3["G_1"].std():.3f}, range=[{all3["G_1"].min():.2f}, {all3["G_1"].max():.2f}]')

# M колонка
if 'M' in all3.columns:
    print(f'\nM: mean={all3["M"].mean():.3f}, std={all3["M"].std():.3f}')

# E колонки
for e in ['E1', 'E2', 'E3']:
    if e in all3.columns:
        print(f'  {e}: mean={all3[e].mean():.2f}, range=[{all3[e].min():.1f}, {all3[e].max():.1f}]')

print('\n--- НОВЫЕ КОЛОНКИ В ALL_3_1 ---')
# G1, G2, G3, M1, M2, M3, E1_*, E2_*, E3_*
extra_cols = sorted(only_all3_1)
print(f'Примеры: {extra_cols[:15]}')

# Проверяем, есть ли G1, G2, G3 (отличие от G_)
if 'G1' in all3_1.columns:
    print(f'\nG1: mean={all3_1["G1"].mean():.3f}')
if 'G2' in all3_1.columns:
    print(f'G2: mean={all3_1["G2"].mean():.3f}')
if 'G3' in all3_1.columns:
    print(f'G3: mean={all3_1["G3"].mean():.3f}')

# E1, E2, E3 с пикетами
e1_cols = [c for c in all3_1.columns if c.startswith('E1_')]
print(f'\nE1_* колонок: {len(e1_cols)} (пикеты 1-15)')

print('\n--- СТАТИСТИКА ПО ЦЕЛЕВЫМ H3_* ---')
h3_cols = [c for c in all3.columns if c.startswith('H3_')]
print(f'H3_* колонок: {len(h3_cols)}')
print(f'H3_8: mean={all3["H3_8"].mean():.4f}, std={all3["H3_8"].std():.4f}')
print(f'H3_8: range=[{all3["H3_8"].min():.4f}, {all3["H3_8"].max():.4f}]')

# Распределение H3_8
print('\n--- РАСПРЕДЕЛЕНИЕ H3_8 ---')
bins = pd.cut(all3['H3_8'], bins=10).value_counts().sort_index()
for b, cnt in bins.items():
    print(f'  {b}: {cnt}')

print('\n--- ВЫВОДЫ ---')
print('''
1. ALL_3.csv = train + valid + test (объединение всех исходных данных)
   - 21000 + 6000 + 3000 = 30000 строк

2. ALL_3 содержит все исходные колонки (2463) + новые (71):
   - G_1...G_31 (31 колонка) - геофизический параметр G
   - M (1 колонка) - параметр M
   - E1, E2, E3 (3 колонки) - параметры E
   - G1, G2, G3, M1, M2, M3 (6 колонок) - агрегированные значения

3. ALL_3_1.csv:
   - 10000 строк (возможно, тестовая выборка или подвыборка)
   - 2660 колонок (больше чем ALL_3 на 126)
   - Добавлены E1_1...E1_15, E2_1...E2_15, E3_1...E3_15 (по 15 пикетов)
   - Добавлены G1, G2, G3, M1, M2, M3 (по 15 значений каждый)

4. Целевые переменные H1_*, H2_*, H3_* (по 15 пикетов) - те же,
   что и в исходных данных.
''')
