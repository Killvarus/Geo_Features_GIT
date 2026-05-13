"""
Предрасчёт и сохранение Kernel PCA (Polynomial) данных в Data/KernelPCA_Poly.

По умолчанию сохраняет только признаки KPC* (без target H*),
чтобы потом подмешивать target из исходных train/valid/test.

Структура:
Data/
  KernelPCA_Poly/
    Difficult_1/
      kpca_poly_d2_g0.0001_c1.0_n16/
        train.csv
        valid.csv
        test.csv
        kpca_info.json
      ...
    Difficult_2/
    Difficult_3/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.utils.data_loader import load_mtz_data


DIFFICULTY_DATASETS: Dict[str, Tuple[str, str, str]] = {
    'Difficult_1': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
    'Difficult_2': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
    'Difficult_3': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
}


def _all_target_columns(df: pd.DataFrame, target_prefix: str = 'H') -> List[str]:
    return [c for c in df.columns if c.startswith(target_prefix)]


def _save_split_csv(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / 'train.csv', index=False)
    valid.to_csv(out_dir / 'valid.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)


def _fit_kpca_poly(
    X_train: pd.DataFrame,
    n_components: int,
    degree: int,
    gamma: float,
    coef0: float,
    max_train_samples_for_fit: int,
    random_state: int,
):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    fit_sample_size = len(X_train_scaled)
    if max_train_samples_for_fit and len(X_train_scaled) > max_train_samples_for_fit:
        rng = np.random.RandomState(random_state)
        fit_indices = rng.choice(len(X_train_scaled), size=max_train_samples_for_fit, replace=False)
        X_fit = X_train_scaled[fit_indices]
        fit_sample_size = len(X_fit)
    else:
        X_fit = X_train_scaled

    kpca = KernelPCA(
        n_components=n_components,
        kernel='poly',
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        eigen_solver='auto',
        random_state=random_state,
    )
    kpca.fit(X_fit)

    return imputer, scaler, kpca, fit_sample_size


def _transform(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    kpca: KernelPCA,
    n_components: int,
) -> pd.DataFrame:
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_kpca = kpca.transform(X_scaled)
    cols = [f'KPC{i+1}' for i in range(n_components)]
    return pd.DataFrame(X_kpca, index=X.index, columns=cols)


def build_kernel_poly_pca_data(
    out_root: Path,
    difficulty_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    n_components_list: List[int],
    degrees: List[int],
    gammas: List[float],
    coef0_list: List[float],
    max_train_samples_for_fit: int,
    random_state: int,
    include_targets: bool,
    force: bool,
):
    feature_cols = [c for c in train.columns if not c.startswith('H')]
    target_cols = _all_target_columns(train)

    X_train = train[feature_cols]
    X_valid = valid[feature_cols]
    X_test = test[feature_cols]

    for degree in degrees:
        for gamma in gammas:
            for coef0 in coef0_list:
                for n_components in n_components_list:
                    config_name = f'kpca_poly_d{degree}_g{gamma}_c{coef0}_n{n_components}'
                    out_dir = out_root / difficulty_name / config_name

                    train_out = out_dir / 'train.csv'
                    valid_out = out_dir / 'valid.csv'
                    test_out = out_dir / 'test.csv'
                    info_out = out_dir / 'kpca_info.json'

                    if not force and train_out.exists() and valid_out.exists() and test_out.exists() and info_out.exists():
                        print(f'[SKIP][KPCA-POLY] {difficulty_name}/{config_name} (already exists)')
                        continue

                    print(f'[RUN ][KPCA-POLY] {difficulty_name}/{config_name}')

                    imputer, scaler, kpca, fit_sample_size = _fit_kpca_poly(
                        X_train=X_train,
                        n_components=n_components,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0,
                        max_train_samples_for_fit=max_train_samples_for_fit,
                        random_state=random_state,
                    )

                    train_kpca = _transform(X_train, imputer, scaler, kpca, n_components)
                    valid_kpca = _transform(X_valid, imputer, scaler, kpca, n_components)
                    test_kpca = _transform(X_test, imputer, scaler, kpca, n_components)

                    if include_targets:
                        for col in target_cols:
                            train_kpca[col] = train[col].values
                            valid_kpca[col] = valid[col].values
                            test_kpca[col] = test[col].values

                    _save_split_csv(train_kpca, valid_kpca, test_kpca, out_dir)

                    info = {
                        'method': 'KernelPCA',
                        'kernel': 'poly',
                        'degree': degree,
                        'gamma': gamma,
                        'coef0': coef0,
                        'n_components': n_components,
                        'random_state': random_state,
                        'max_train_samples_for_fit': max_train_samples_for_fit,
                        'fit_sample_size': fit_sample_size,
                        'original_n_features': len(feature_cols),
                        'target_columns_included': include_targets,
                    }
                    with open(info_out, 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Предрасчёт Kernel PCA (Polynomial) CSV в Data/KernelPCA_Poly.')
    parser.add_argument(
        '--difficulties',
        nargs='*',
        default=['Difficult_1', 'Difficult_2', 'Difficult_3'],
        choices=['Difficult_1', 'Difficult_2', 'Difficult_3'],
        help='Какие уровни сложности обрабатывать',
    )
    parser.add_argument('--kpca-components', nargs='*', type=int, default=[16, 64, 256, 512, 1024, 1536, 2048])
    parser.add_argument('--degrees', nargs='*', type=int, default=[2, 3])
    parser.add_argument('--gammas', nargs='*', type=float, default=[0.0001, 0.001, 0.01])
    parser.add_argument('--coef0-list', nargs='*', type=float, default=[0.1, 1.0])
    parser.add_argument('--max-train-samples-for-fit', type=int, default=22000)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--include-targets', action='store_true', help='Сохранять target H* в transformed CSV')
    parser.add_argument('--force', action='store_true', help='Перезаписать уже существующие файлы')
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = PROJECT_ROOT / 'Data'
    kpca_root = data_dir / 'KernelPCA_Poly'

    print('=' * 80)
    print('PRECOMPUTE KERNEL PCA DATA (POLY)')
    print('=' * 80)
    print(f'Data dir: {data_dir}')
    print(f'Output dir: {kpca_root}')
    print(f'Difficulties: {args.difficulties}')
    print(f'n_components: {args.kpca_components}')
    print(f'degrees: {args.degrees}')
    print(f'gammas: {args.gammas}')
    print(f'coef0_list: {args.coef0_list}')
    print(f'max_train_samples_for_fit: {args.max_train_samples_for_fit}')
    print(f'include_targets: {args.include_targets}')
    print(f'force overwrite: {args.force}')

    for difficulty_name in args.difficulties:
        train_file, valid_file, test_file = DIFFICULTY_DATASETS[difficulty_name]
        train_path = data_dir / train_file
        valid_path = data_dir / valid_file
        test_path = data_dir / test_file

        print('\n' + '-' * 80)
        print(f'[{difficulty_name}] Source files: {train_file}, {valid_file}, {test_file}')

        if not train_path.exists() or not valid_path.exists() or not test_path.exists():
            print(f'[WARN] Files not found for {difficulty_name}, skip')
            continue

        train, valid, test = load_mtz_data(train_path, valid_path, test_path, verbose=True)
        print(f'All target columns (H*): {len(_all_target_columns(train))}')

        build_kernel_poly_pca_data(
            out_root=kpca_root,
            difficulty_name=difficulty_name,
            train=train,
            valid=valid,
            test=test,
            n_components_list=args.kpca_components,
            degrees=args.degrees,
            gammas=args.gammas,
            coef0_list=args.coef0_list,
            max_train_samples_for_fit=args.max_train_samples_for_fit,
            random_state=args.random_state,
            include_targets=args.include_targets,
            force=args.force,
        )

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()
