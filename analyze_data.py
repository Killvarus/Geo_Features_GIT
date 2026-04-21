"""
Анализ данных на предмет утечек и аномалий

Проблема: линейная регрессия даёт R2 ~ 0.99, а НС и ГБ только ~ 0.8
Это подозрительно и может указывать на утечку данных.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Загрузка данных"""
    data_dir = Path('Data')
    train = pd.read_csv(data_dir / 'mtsgrvmgn_trn.csv')
    valid = pd.read_csv(data_dir / 'mtsgrvmgn_vld.csv')
    test = pd.read_csv(data_dir / 'mtsgrvmgn_tst.csv')
    return train, valid, test


def analyze_target(train, target='H3_8'):
    """Анализ целевой переменной"""
    print(f"\n{'='*70}")
    print(f"TARGET ANALYSIS: {target}")
    print(f"{'='*70}")
    
    y = train[target]
    
    print(f"\nStatistics:")
    print(f"  Count: {len(y)}")
    print(f"  Unique values: {y.nunique()}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std: {y.std():.6f}")
    print(f"  Median: {y.median():.6f}")
    
    print(f"\nDistribution:")
    print(f"  < 0: {(y < 0).sum()}")
    print(f"  = 0: {(y == 0).sum()}")
    print(f"  > 0: {(y > 0).sum()}")
    
    return y


def check_correlations(train, target='H3_8', top_n=20):
    """Проверка корреляций признаков с целевой"""
    print(f"\n{'='*70}")
    print(f"CORRELATIONS WITH {target}")
    print(f"{'='*70}")
    
    features = [col for col in train.columns if not col.startswith('H')]
    y = train[target]
    
    correlations = []
    for feat in features:
        corr, pval = pearsonr(train[feat], y)
        correlations.append({
            'feature': feat,
            'correlation': corr,
            'abs_corr': abs(corr),
            'p_value': pval
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    print(f"\nTop-{top_n} features by correlation:")
    print(corr_df.head(top_n).to_string(index=False))
    
    high_corr = corr_df[corr_df['abs_corr'] > 0.95]
    if len(high_corr) > 0:
        print(f"\n[!!!] HIGH CORRELATION > 0.95: {len(high_corr)} features")
        print(high_corr.to_string(index=False))
    
    return corr_df


def check_linear_combination(train, valid, test, target='H3_8'):
    """Проверка, не является ли целевая линейной комбинацией признаков"""
    print(f"\n{'='*70}")
    print(f"LINEAR COMBINATION CHECK")
    print(f"{'='*70}")
    
    features = [col for col in train.columns if not col.startswith('H')]
    
    X_train = train[features].values
    y_train = train[target].values
    X_valid = valid[features].values
    y_valid = valid[target].values
    X_test = test[features].values
    y_test = test[target].values
    
    # Линейная регрессия
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred_train = lr.predict(X_train)
    y_pred_valid = lr.predict(X_valid)
    y_pred_test = lr.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_valid = r2_score(y_valid, y_pred_valid)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nLinear Regression Results:")
    print(f"  Train R2: {r2_train:.6f}")
    print(f"  Valid R2: {r2_valid:.6f}")
    print(f"  Test R2:  {r2_test:.6f}")
    
    if r2_train > 0.99:
        print(f"\n[!!!] CRITICAL: Train R2 = {r2_train:.6f} > 0.99")
        print("Target is almost perfectly predicted by linear combination!")
        
        # Топ коэффициентов
        coef_df = pd.DataFrame({
            'feature': features,
            'coef': lr.coef_
        })
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False)
        
        print(f"\nTop-20 coefficients:")
        print(coef_df.head(20).to_string(index=False))
        
        print(f"\nIntercept: {lr.intercept_:.6f}")
        
        # Проверяем, какие признаки дают наибольший вклад
        print(f"\nChecking prediction formula...")
        print(f"  y = {lr.intercept_:.6f}")
        for _, row in coef_df.head(5).iterrows():
            print(f"      + {row['coef']:.6f} * {row['feature']}")
    
    return lr, r2_train, r2_valid, r2_test


def check_feature_leakage(train, target='H3_8'):
    """Проверка на утечку через признаки"""
    print(f"\n{'='*70}")
    print(f"FEATURE LEAKAGE CHECK")
    print(f"{'='*70}")
    
    y = train[target].values
    features = [col for col in train.columns if not col.startswith('H')]
    
    print("\nChecking if any feature is proportional to target...")
    
    suspicious = []
    for feat in features:
        x = train[feat].values
        
        # Проверяем корреляцию
        corr = np.corrcoef(x, y)[0, 1]
        
        if abs(corr) > 0.99:
            suspicious.append({
                'feature': feat,
                'correlation': corr
            })
            print(f"  [!!!] {feat}: corr = {corr:.6f}")
    
    if not suspicious:
        print("No features with correlation > 0.99 to target")
    
    return suspicious


def analyze_coefficients(lr, train, target='H3_8'):
    """Анализ коэффициентов линейной регрессии"""
    print(f"\n{'='*70}")
    print(f"COEFFICIENT ANALYSIS")
    print(f"{'='*70}")
    
    features = [col for col in train.columns if not col.startswith('H')]
    
    coef_df = pd.DataFrame({
        'feature': features,
        'coef': lr.coef_
    })
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    # Смотрим на ненулевые коэффициенты
    non_zero = coef_df[coef_df['abs_coef'] > 1e-6]
    print(f"\nNon-zero coefficients: {len(non_zero)} / {len(coef_df)}")
    
    # Топ-50
    print(f"\nTop-50 most important features:")
    print(coef_df.head(50).to_string(index=False))
    
    # Группируем по компонентам
    print(f"\nCoefficients by component:")
    for prefix in ['REYX', 'IMYX', 'REXY', 'IMXY', 'REHX', 'IMHX']:
        subset = coef_df[coef_df['feature'].str.startswith(prefix)]
        print(f"  {prefix}: mean_abs_coef = {subset['abs_coef'].mean():.6f}, count = {len(subset)}")
    
    return coef_df


def check_data_generation(train, target='H3_8'):
    """Проверка гипотезы о генерации данных"""
    print(f"\n{'='*70}")
    print(f"DATA GENERATION HYPOTHESIS")
    print(f"{'='*70}")
    
    features = [col for col in train.columns if not col.startswith('H')]
    y = train[target].values
    
    # Проверяем, есть ли паттерн в целевой
    print(f"\nTarget value distribution:")
    print(f"  Unique values: {len(np.unique(y))}")
    print(f"  Values: {np.sort(np.unique(y))}")
    
    # Проверяем, может ли целевая быть суммой/разностью признаков
    print(f"\nChecking simple formulas...")
    
    # Сумма всех признаков
    X_sum = train[features].sum(axis=1).values
    corr_sum = np.corrcoef(X_sum, y)[0, 1]
    print(f"  Sum of all features: corr = {corr_sum:.6f}")
    
    # Среднее всех признаков
    X_mean = train[features].mean(axis=1).values
    corr_mean = np.corrcoef(X_mean, y)[0, 1]
    print(f"  Mean of all features: corr = {corr_mean:.6f}")


def main():
    print("=" * 70)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 70)
    
    # Загрузка
    train, valid, test = load_data()
    
    target = 'H3_8'
    
    # Анализы
    analyze_target(train, target)
    corr_df = check_correlations(train, target)
    lr, r2_train, r2_valid, r2_test = check_linear_combination(train, valid, test, target)
    suspicious = check_feature_leakage(train, target)
    coef_df = analyze_coefficients(lr, train, target)
    check_data_generation(train, target)
    
    # Выводы
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    
    if r2_train > 0.99:
        print("\n[!!!] CRITICAL PROBLEM DETECTED:")
        print(f"   Train R2 = {r2_train:.6f}")
        print(f"   Valid R2 = {r2_valid:.6f}")
        print(f"   Test R2  = {r2_test:.6f}")
        print("\n   Target is almost perfectly predicted by LINEAR combination!")
        print("\n   Possible reasons:")
        print("   1. Target is computed as linear function of features")
        print("   2. There is a feature = target (or proportional)")
        print("   3. Data was generated by a linear model")
        print("\n   This explains why:")
        print("   - Linear regression: R2 ~ 0.998")
        print("   - Neural network: R2 ~ 0.8")
        print("   - NN struggles because the true function is LINEAR")
        print("   - NN overfits to noise or has optimization issues")
    
    # Сохраняем
    corr_df.to_csv('analysis_correlations.csv', index=False)
    coef_df.to_csv('analysis_coefficients.csv', index=False)
    print(f"\nSaved: analysis_correlations.csv, analysis_coefficients.csv")


if __name__ == '__main__':
    main()
