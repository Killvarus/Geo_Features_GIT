"""
Предрасчёт и сохранение transformed-данных (Aggregation + PCA) в Data/.

Цель:
- один раз подготовить датасеты для разных уровней сложности,
- потом обучать модели на уже готовых CSV без повторной трансформации.

Структура сохранения:
Data/
  Aggregated/
    Difficult_1/
      freq1_pickup1_mean/
        train.csv
        valid.csv
        test.csv
      ...
    Difficult_2/
    Difficult_3/
  PCA/
    Difficult_1/
      pca_16/
        train.csv
        valid.csv
        test.csv
        pca_info.json
      ...
    Difficult_2/
    Difficult_3/

Важно:
- В CSV сохраняются и признаки, и ВСЕ целевые переменные H*.
- Агрегация/ PCA выполняются теми же функциями, что и в экспериментах проекта.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import PROJECT_ROOT
from src.preprocessing.aggregation import AggregationConfig, aggregate_features
from src.preprocessing.pca import apply_pca_to_data
from src.utils.data_loader import load_mtz_data


# Соответствие уровня сложности и исходных файлов
DIFFICULTY_DATASETS: Dict[str, Tuple[str, str, str]] = {
    'Difficult_1': ('mtsgrvmgn_trn.csv', 'mtsgrvmgn_vld.csv', 'mtsgrvmgn_tst.csv'),
    'Difficult_2': ('train_3.csv', 'valid_3.csv', 'test_3.csv'),
    'Difficult_3': ('train_3_1.csv', 'valid_3_1.csv', 'test_3_1.csv'),
}


def _all_target_columns(df: pd.DataFrame, target_prefix: str = 'H') -> List[str]:
    """Берём ВСЕ таргеты по префиксу (H*), чтобы сохранить их в transformed CSV."""
    return [c for c in df.columns if c.startswith(target_prefix)]


def _save_split_csv(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / 'train.csv', index=False)
    valid.to_csv(out_dir / 'valid.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)


def build_aggregated_data(
    data_dir: Path,
    out_root: Path,
    difficulty_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    freq_steps: List[int],
    pickup_steps: List[int],
    agg_methods: List[str],
    force: bool,
):
    target_columns = _all_target_columns(train)

    for freq_step in freq_steps:
        for pickup_step in pickup_steps:
            for agg_method in agg_methods:
                config_name = f'freq{freq_step}_pickup{pickup_step}_{agg_method}'
                out_dir = out_root / difficulty_name / config_name

                train_out = out_dir / 'train.csv'
                valid_out = out_dir / 'valid.csv'
                test_out = out_dir / 'test.csv'

                if not force and train_out.exists() and valid_out.exists() and test_out.exists():
                    print(f'[SKIP][AGG] {difficulty_name}/{config_name} (already exists)')
                    continue

                print(f'[RUN ][AGG] {difficulty_name}/{config_name}')
                config = AggregationConfig(
                    freq_step=freq_step,
                    pickup_step=pickup_step,
                    agg_method=agg_method,
                )

                train_agg, valid_agg, test_agg, _ = aggregate_features(
                    train=train,
                    valid=valid,
                    test=test,
                    config=config,
                    target_columns=target_columns,  # сохраняем ВСЕ H*
                )

                _save_split_csv(train_agg, valid_agg, test_agg, out_dir)


def build_pca_data(
    data_dir: Path,
    out_root: Path,
    difficulty_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    n_components_list: List[int],
    force: bool,
):
    target_columns = _all_target_columns(train)

    for n_components in n_components_list:
        config_name = f'pca_{n_components}'
        out_dir = out_root / difficulty_name / config_name

        train_out = out_dir / 'train.csv'
        valid_out = out_dir / 'valid.csv'
        test_out = out_dir / 'test.csv'
        info_out = out_dir / 'pca_info.json'

        if not force and train_out.exists() and valid_out.exists() and test_out.exists() and info_out.exists():
            print(f'[SKIP][PCA] {difficulty_name}/{config_name} (already exists)')
            continue

        print(f'[RUN ][PCA] {difficulty_name}/{config_name}')

        train_pca, valid_pca, test_pca, transformer = apply_pca_to_data(
            train=train,
            valid=valid,
            test=test,
            target_columns=target_columns,  # сохраняем ВСЕ H*
            n_components=n_components,
            target_prefix='H',
        )

        _save_split_csv(train_pca, valid_pca, test_pca, out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(info_out, 'w', encoding='utf-8') as f:
            json.dump(transformer.get_info(), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Предрасчёт Aggregated/PCA CSV в Data/.')
    parser.add_argument(
        '--difficulties',
        nargs='*',
        default=['Difficult_1', 'Difficult_2', 'Difficult_3'],
        choices=['Difficult_1', 'Difficult_2', 'Difficult_3'],
        help='Какие уровни сложности обрабатывать',
    )
    parser.add_argument(
        '--agg-methods',
        nargs='*',
        default=['mean'],
        help='Методы агрегации (mean/median/max/min/std/range)',
    )
    parser.add_argument('--freq-steps', nargs='*', type=int, default=[1, 2, 3])
    parser.add_argument('--pickup-steps', nargs='*', type=int, default=[1, 2, 3])
    parser.add_argument('--pca-components', nargs='*', type=int, default=[16, 64, 256, 512, 1024, 1536,2048])
    parser.add_argument('--skip-agg', action='store_true', help='Не строить Aggregated')
    parser.add_argument('--skip-pca', action='store_true', help='Не строить PCA')
    parser.add_argument('--force', action='store_true', help='Перезаписать уже существующие файлы')
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = PROJECT_ROOT / 'Data'
    aggregated_root = data_dir / 'Aggregated'
    pca_root = data_dir / 'PCA'

    print('=' * 80)
    print('PRECOMPUTE TRANSFORMED DATA')
    print('=' * 80)
    print(f'Data dir: {data_dir}')
    print(f'Difficulties: {args.difficulties}')
    print(f'Build Aggregated: {not args.skip_agg}')
    print(f'Build PCA: {not args.skip_pca}')
    print(f'Force overwrite: {args.force}')

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
        all_targets = _all_target_columns(train)
        print(f'All target columns (H*): {len(all_targets)}')

        if not args.skip_agg:
            build_aggregated_data(
                data_dir=data_dir,
                out_root=aggregated_root,
                difficulty_name=difficulty_name,
                train=train,
                valid=valid,
                test=test,
                freq_steps=args.freq_steps,
                pickup_steps=args.pickup_steps,
                agg_methods=args.agg_methods,
                force=args.force,
            )

        if not args.skip_pca:
            build_pca_data(
                data_dir=data_dir,
                out_root=pca_root,
                difficulty_name=difficulty_name,
                train=train,
                valid=valid,
                test=test,
                n_components_list=args.pca_components,
                force=args.force,
            )

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()
