"""
Диагностика данных и моделей

Тесты для выявления проблем:
1. Проверка утечки данных (data leakage)
2. Анализ корреляций
3. Сравнение линейных и нелинейных моделей
4. Анализ важности признаков
5. Проверка распределений
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Загрузка данных"""
    from src.config import PROJECT_ROOT
    from src.utils import load_mtz_data
    
    data_dir = PROJECT_ROOT / "Data"
    train = pd.read_csv(data_dir / "mtsgrvmgn_trn.csv")
    valid = pd.read_csv(data_dir / "mtsgrvmgn_vld.csv")
    test = pd.read_csv(data_dir / "mtsgrvmgn_tst.csv")
    
    return train, valid, test


def get_features_targets(df, target_col='H1_8'):
    """Разделение на признаки и целевую"""
    feature_cols = [c for c in df.columns if not c.startswith('H')]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


# =============================================================================
# ТЕСТ 1: ПРОВЕРКА КОРРЕЛЯЦИЙ
# =============================================================================
def test_correlations():
    """Проверка корреляций признаков с целевой переменной"""
    print("\n" + "="*60)
    print("TEST 1: CORRELATIONS WITH TARGET")
    print("="*60)
    
    train, _, _ = load_data()
    X, y = get_features_targets(train, 'H1_8')
    
    # Корреляции Пирсона
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    print("\nTop 20 correlated features:")
    for i, (feat, corr) in enumerate(correlations.head(20).items()):
        print(f"  {i+1}. {feat}: {corr:.4f}")
    
    # Статистика по корреляциям
    print(f"\nCorrelation statistics:")
    print(f"  Max: {correlations.max():.4f}")
    print(f"  Mean: {correlations.mean():.4f}")
    print(f"  Median: {correlations.median():.4f}")
    print(f"  > 0.9: {(correlations > 0.9).sum()}")
    print(f"  > 0.8: {(correlations > 0.8).sum()}")
    print(f"  > 0.7: {(correlations > 0.7).sum()}")
    
    # Проверка на подозрительно высокие корреляции
    high_corr = correlations[correlations > 0.95]
    if len(high_corr) > 0:
        print(f"\n⚠️  WARNING: {len(high_corr)} features with correlation > 0.95!")
        print("  Possible data leakage!")
        for feat, corr in high_corr.items():
            print(f"    {feat}: {corr:.4f}")
    
    return correlations


# =============================================================================
# ТЕСТ 2: ЛИНЕЙНАЯ РЕГРЕССИЯ
# =============================================================================
def test_linear_regression():
    """Сравнение линейных моделей"""
    print("\n" + "="*60)
    print("TEST 2: LINEAR REGRESSION")
    print("="*60)
    
    train, valid, test = load_data()
    X_train, y_train = get_features_targets(train, 'H1_8')
    X_valid, y_valid = get_features_targets(valid, 'H1_8')
    X_test, y_test = get_features_targets(test, 'H1_8')
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Обычная линейная регрессия
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred_test = lr.predict(X_test_scaled)
    r2 = r2_score(y_test, pred_test)
    results['LinearRegression'] = r2
    print(f"\nLinearRegression R² (test): {r2:.4f}")
    
    # 2. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    pred_test = ridge.predict(X_test_scaled)
    r2 = r2_score(y_test, pred_test)
    results['Ridge'] = r2
    print(f"Ridge R² (test): {r2:.4f}")
    
    # 3. Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y_train)
    pred_test = lasso.predict(X_test_scaled)
    r2 = r2_score(y_test, pred_test)
    results['Lasso'] = r2
    print(f"Lasso R² (test): {r2:.4f}")
    
    # Проверка на train
    pred_train = lr.predict(X_train_scaled)
    r2_train = r2_score(y_train, pred_train)
    print(f"\nLinearRegression R² (train): {r2_train:.4f}")
    
    # Проверка на valid
    pred_valid = lr.predict(X_valid_scaled)
    r2_valid = r2_score(y_valid, pred_valid)
    print(f"LinearRegression R² (valid): {r2_valid:.4f}")
    
    if r2 > 0.95:
        print("\n⚠️  WARNING: Linear regression R² > 0.95!")
        print("  Data is likely linear or there is leakage!")
    
    return results


# =============================================================================
# ТЕСТ 3: НЕЛИНЕЙНЫЕ МОДЕЛИ
# =============================================================================
def test_nonlinear_models():
    """Сравнение нелинейных моделей"""
    print("\n" + "="*60)
    print("TEST 3: NONLINEAR MODELS")
    print("="*60)
    
    train, valid, test = load_data()
    X_train, y_train = get_features_targets(train, 'H1_8')
    X_test, y_test = get_features_targets(test, 'H1_8')
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Gradient Boosting
    print("\nGradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    pred = gb.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    results['GradientBoosting'] = r2
    print(f"  R²: {r2:.4f}")
    
    # 2. MLP
    print("\nMLP (sklearn)...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    pred = mlp.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    results['MLP'] = r2
    print(f"  R²: {r2:.4f}")
    
    # 3. Random Forest
    from sklearn.ensemble import RandomForestRegressor
    print("\nRandom Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    pred = rf.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    results['RandomForest'] = r2
    print(f"  R²: {r2:.4f}")
    
    return results


# =============================================================================
# ТЕСТ 4: ВАЖНОСТЬ ПРИЗНАКОВ
# =============================================================================
def test_feature_importance():
    """Анализ важности признаков"""
    print("\n" + "="*60)
    print("TEST 4: FEATURE IMPORTANCE")
    print("="*60)
    
    train, _, _ = load_data()
    X, y = get_features_targets(train, 'H1_8')
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lasso для определения важных признаков
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_scaled, y)
    
    # Коэффициенты
    coef = pd.Series(np.abs(lasso.coef_), index=X.columns)
    coef_sorted = coef.sort_values(ascending=False)
    
    print("\nTop 20 important features (Lasso):")
    for i, (feat, imp) in enumerate(coef_sorted.head(20).items()):
        print(f"  {i+1}. {feat}: {imp:.6f}")
    
    # Количество ненулевых коэффициентов
    nonzero = (coef > 0).sum()
    print(f"\nNon-zero coefficients: {nonzero} / {len(coef)}")
    
    # Проверка: есть ли признаки с очень большими коэффициентами
    high_coef = coef[coef > coef.mean() + 3 * coef.std()]
    if len(high_coef) > 0:
        print(f"\n⚠️  WARNING: {len(high_coef)} features with very high coefficients!")
        for feat, c in high_coef.items():
            print(f"    {feat}: {c:.4f}")
    
    return coef_sorted


# =============================================================================
# ТЕСТ 5: ПРОВЕРКА НА УТЕЧКУ
# =============================================================================
def test_data_leakage():
    """Проверка на утечку данных"""
    print("\n" + "="*60)
    print("TEST 5: DATA LEAKAGE CHECK")
    print("="*60)
    
    train, _, _ = load_data()
    X, y = get_features_targets(train, 'H1_8')
    
    # 1. Проверка: есть ли признаки, которые почти идеально предсказывают y
    print("\nChecking for features that predict target perfectly...")
    
    for col in X.columns:
        corr = np.abs(np.corrcoef(X[col], y)[0, 1])
        if corr > 0.99:
            print(f"  ⚠️  {col}: correlation = {corr:.6f}")
    
    # 2. Проверка: есть ли признаки, которые равны y
    print("\nChecking for features equal to target...")
    for col in X.columns:
        if np.allclose(X[col].values, y.values, rtol=1e-3, atol=1e-3):
            print(f"  ⚠️  {col} is nearly equal to target!")
    
    # 3. Проверка: есть ли производные y в признаках
    print("\nChecking for target derivatives in features...")
    for col in X.columns:
        # Проверяем линейную зависимость
        slope = np.corrcoef(X[col], y)[0, 1] * X[col].std() / y.std()
        residual = y - slope * X[col]
        if residual.std() < 0.01 * y.std():
            print(f"  ⚠️  {col} is nearly a linear function of target!")
    
    # 4. Проверка имён признаков
    print("\nChecking feature names for target-related patterns...")
    target_related = [c for c in X.columns if 'H1' in c or 'H2' in c or 'H3' in c]
    if target_related:
        print(f"  ⚠️  Found {len(target_related)} features with H1/H2/H3 in name:")
        for c in target_related[:10]:
            print(f"      {c}")
    
    # 5. Проверка: может ли 1-2 признака объяснить почти всю дисперсию
    print("\nChecking if 1-2 features can explain most variance...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LinearRegression()
    for i, col in enumerate(X.columns[:50]):  # Проверяем первые 50
        lr.fit(X_scaled[:, i:i+1], y)
        pred = lr.predict(X_scaled[:, i:i+1])
        r2 = r2_score(y, pred)
        if r2 > 0.9:
            print(f"  ⚠️  {col} alone: R² = {r2:.4f}")


# =============================================================================
# ТЕСТ 6: РАСПРЕДЕЛЕНИЕ ДАННЫХ
# =============================================================================
def test_data_distribution():
    """Анализ распределения данных"""
    print("\n" + "="*60)
    print("TEST 6: DATA DISTRIBUTION")
    print("="*60)
    
    train, _, _ = load_data()
    X, y = get_features_targets(train, 'H1_8')
    
    print(f"\nDataset shape: {train.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Статистика целевой переменной
    print(f"\nTarget (H1_8) statistics:")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Std: {y.std():.4f}")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    print(f"  Skew: {y.skew():.4f}")
    print(f"  Kurtosis: {y.kurtosis():.4f}")
    
    # Проверка на нормальность
    stat, p = stats.normaltest(y)
    print(f"  Normal test p-value: {p:.4e}")
    
    # Статистика признаков
    print(f"\nFeature statistics:")
    print(f"  Mean of means: {X.mean().mean():.4f}")
    print(f"  Mean of stds: {X.std().mean():.4f}")
    print(f"  Features with std< 0.01: {(X.std() < 0.01).sum()}")
    print(f"  Features with std < 0.001: {(X.std() < 0.001).sum()}")
    
    # Проверка на константные признаки
    constant = X.columns[X.nunique() == 1]
    if len(constant) > 0:
        print(f"\n⚠️  Constant features: {len(constant)}")
    
    # Проверка на дубликаты признаков
    print("\nChecking for duplicate features...")
    dup_count = X.T.duplicated().sum()
    if dup_count > 0:
        print(f"  ⚠️  Duplicate features: {dup_count}")


# =============================================================================
# ТЕСТ 7: ТЕСТ С ШУМОМ
# =============================================================================
def test_with_noise():
    """Тест с добавлением шума"""
    print("\n" + "="*60)
    print("TEST 7: MODEL ROBUSTNESS TO NOISE")
    print("="*60)
    
    train, valid, test = load_data()
    X_train, y_train = get_features_targets(train, 'H1_8')
    X_test, y_test = get_features_targets(test, 'H1_8')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Линейная регрессия без шума
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred = lr.predict(X_test_scaled)
    r2_clean = r2_score(y_test, pred)
    print(f"\nLinear R² (no noise): {r2_clean:.4f}")
    
    # Добавляем шум
    for noise_level in [0.01, 0.05, 0.1, 0.2]:
        X_train_noisy = X_train_scaled + np.random.randn(*X_train_scaled.shape) * noise_level
        X_test_noisy = X_test_scaled + np.random.randn(*X_test_scaled.shape) * noise_level
        
        lr = LinearRegression()
        lr.fit(X_train_noisy, y_train)
        pred = lr.predict(X_test_noisy)
        r2 = r2_score(y_test, pred)
        print(f"Linear R² (noise={noise_level}): {r2:.4f}")
    
    # Если R² резко падает при малом шуме — признак переобучения на точные значения


# =============================================================================
# ТЕСТ 8: ПРОСТЫЕ ПРИЗНАКИ
# =============================================================================
def test_simple_features():
    """Тест с небольшим количеством признаков"""
    print("\n" + "="*60)
    print("TEST 8: SIMPLE FEATURES TEST")
    print("="*60)
    
    train, valid, test = load_data()
    X_train, y_train = get_features_targets(train, 'H1_8')
    X_test, y_test = get_features_targets(test, 'H1_8')
    
    # Топ-10 коррелирующих признаков
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = correlations.head(10).index.tolist()
    
    print(f"\nTop 10 correlated features: {top_features}")
    
    scaler = StandardScaler()
    
    # Тест только на топ-10
    X_train_10 = scaler.fit_transform(X_train[top_features])
    X_test_10 = scaler.transform(X_test[top_features])
    
    lr = LinearRegression()
    lr.fit(X_train_10, y_train)
    pred = lr.predict(X_test_10)
    r2 = r2_score(y_test, pred)
    print(f"\nLinear R² (top 10 features): {r2:.4f}")
    
    # Тест на топ-3
    top_3 = correlations.head(3).index.tolist()
    X_train_3 = scaler.fit_transform(X_train[top_3])
    X_test_3 = scaler.transform(X_test[top_3])
    
    lr = LinearRegression()
    lr.fit(X_train_3, y_train)
    pred = lr.predict(X_test_3)
    r2 = r2_score(y_test, pred)
    print(f"Linear R² (top 3 features): {r2:.4f}")
    
    # Тест на 1 лучший признак
    best = correlations.head(1).index[0]
    X_train_1 = scaler.fit_transform(X_train[[best]])
    X_test_1 = scaler.transform(X_test[[best]])
    
    lr = LinearRegression()
    lr.fit(X_train_1, y_train)
    pred = lr.predict(X_test_1)
    r2 = r2_score(y_test, pred)
    print(f"Linear R² (best feature: {best}): {r2:.4f}")


# =============================================================================
# ТЕСТ 9: СИНТЕТИЧЕСКИЕ ДАННЫЕ
# =============================================================================
def test_synthetic_data():
    """Сравнение на синтетических данных с известной структурой"""
    print("\n" + "="*60)
    print("TEST 9: SYNTHETIC DATA COMPARISON")
    print("="*60)
    
    np.random.seed(42)
    n = 1000
    
    # Линейные данные
    X_lin = np.random.randn(n, 100)
    y_lin = X_lin[:, 0] * 2 + X_lin[:, 1] * 3 + np.random.randn(n) * 0.1
    
    # Нелинейные данные
    X_nonlin = np.random.randn(n, 100)
    y_nonlin = X_nonlin[:, 0]**2 + np.sin(X_nonlin[:, 1] * 3) + np.random.randn(n) * 0.1
    
    from sklearn.model_selection import train_test_split
    
    # Линейные
    X_tr, X_te, y_tr, y_te = train_test_split(X_lin, y_lin, test_size=0.3, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    r2_lr = r2_score(y_te, lr.predict(X_te))
    
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    gb.fit(X_tr, y_tr)
    r2_gb = r2_score(y_te, gb.predict(X_te))
    
    print(f"\nLinear synthetic data:")
    print(f"  LinearRegression R²: {r2_lr:.4f}")
    print(f"  GradientBoosting R²: {r2_gb:.4f}")
    
    # Нелинейные
    X_tr, X_te, y_tr, y_te = train_test_split(X_nonlin, y_nonlin, test_size=0.3, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    r2_lr = r2_score(y_te, lr.predict(X_te))
    
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    gb.fit(X_tr, y_tr)
    r2_gb = r2_score(y_te, gb.predict(X_te))
    
    print(f"\nNonlinear synthetic data:")
    print(f"  LinearRegression R²: {r2_lr:.4f}")
    print(f"  GradientBoosting R²: {r2_gb:.4f}")
    
    print("\nExpected behavior:")
    print("  - Linear data: LR ≈ GB (both high)")
    print("  - Nonlinear data: GB > LR")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    print("="*60)
    print("DATA DIAGNOSTICS")
    print("="*60)
    
    # Запускаем все тесты
    correlations = test_correlations()
    lr_results = test_linear_regression()
    nl_results = test_nonlinear_models()
    importance = test_feature_importance()
    test_data_leakage()
    test_data_distribution()
    test_with_noise()
    test_simple_features()
    test_synthetic_data()
    
    # Итоговый отчёт
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nModel comparison:")
    print(f"  LinearRegression: {lr_results.get('LinearRegression', 0):.4f}")
    print(f"  Ridge: {lr_results.get('Ridge', 0):.4f}")
    print(f"  Lasso: {lr_results.get('Lasso', 0):.4f}")
    for model, r2 in nl_results.items():
        print(f"  {model}: {r2:.4f}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
