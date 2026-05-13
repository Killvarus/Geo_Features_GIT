"""
Предрасчёт и сохранение PLS-данных в Data/PLS/.

Структура:
Data/PLS/
  Difficult_1/
    pls_2/
      train.csv
      valid.csv
      test.csv
      pls_info.json
    pls_4/
    ...
  Difficult_2/
  Difficult_3/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import PROJECT_ROOT
from src.preprocessing.pls import apply_pls_to_data
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


def build_pls_data(
    data_dir: Path,
    out_root: Path,
    difficulty_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target_columns: List[str],
    n_components_list: List[int],
    force: bool,
):
    for n_components in n_components_list:
        config_name = f'pls_{n_components}'
        out_dir = out_root / difficulty_name / config_name

        train_out = out_dir / 'train.csv'
        valid_out = out_dir / 'valid.csv'
        test_out = out_dir / 'test.csv'
        info_out = out_dir / 'pls_info.json'

        if not force and train_out.exists() and valid_out.exists() and test_out.exists() and info_out.exists():
            print(f'[SKIP][PLS] {difficulty_name}/{config_name} (already exists)')
            continue

        print(f'[RUN ][PLS] {difficulty_name}/{config_name}')

        train_pls, valid_pls, test_pls, transformer = apply_pls_to_data(
            train=train,
            valid=valid,
            test=test,
            target_columns=target_columns,
            n_components=n_components,
            target_prefix='H',
        )

        _save_split_csv(train_pls, valid_pls, test_pls, out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(info_out, 'w', encoding='utf-8') as f:
            json.dump(transformer.get_info(), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Предрасчёт PLS CSV в Data/PLS/.')
    parser.add_argument('--difficulties', nargs='*', default=['Difficult_1', 'Difficult_2', 'Difficult_3'])
    parser.add_argument('--targets', nargs='*', default=['H3_8'])
    parser.add_argument('--pls-components', nargs='*', type=int, default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--force', action='store_true', help='Перезаписать уже существующие файлы')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / 'Data'
    pls_root = data_dir / 'PLS'

    print('=' * 80)
    print('PRECOMPUTE PLS DATA')
    print('=' * 80)
    print(f'Data dir: {data_dir}')
    print(f'Difficulties: {args.difficulties}')
    print(f'Targets: {args.targets}')
    print(f'Components: {args.pls_components}')
    print(f'Force: {args.force}')

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

        build_pls_data(
            data_dir=data_dir,
            out_root=pls_root,
            difficulty_name=difficulty_name,
            train=train,
            valid=valid,
            test=test,
            target_columns=args.targets,
            n_components_list=args.pls_components,
            force=args.force,
        )

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)


if __name__ == '__main__':
    main()