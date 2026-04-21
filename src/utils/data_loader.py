"""
Загрузка и подготовка данных МТЗ
"""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class Data:
    """Подготовка train/valid/test с явной валидацией схемы данных."""

    def __init__(
        self, 
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        valid: pd.DataFrame, 
        columns: List[str],
        feature_prefix: str = 'H'
    ):
        self.train = train.copy()
        self.test = test.copy()
        self.valid = valid.copy()
        self.target_columns = list(columns)
        self.target_prefix = feature_prefix

        self._validate_target_columns()
        self._validate_split_columns()

        inferred_targets = [col for col in self.train.columns if col.startswith(self.target_prefix)]
        self.feature_columns = [col for col in self.train.columns if col not in inferred_targets]
        self.columns_X = self.feature_columns


        self.X_train = self.train[self.columns_X]
        self.X_test = self.test[self.columns_X]
        self.X_valid = self.valid[self.columns_X]

        self._validate_no_target_leakage()

        self.y_train = self.train[self.target_columns]

        self.y_valid = self.valid[self.target_columns]
        self.y_test = self.test[self.target_columns]

        self.n_features = len(self.columns_X)
        self.n_targets = len(self.target_columns)

    def _validate_target_columns(self) -> None:
        missing_in_train = [col for col in self.target_columns if col not in self.train.columns]
        missing_in_valid = [col for col in self.target_columns if col not in self.valid.columns]
        missing_in_test = [col for col in self.target_columns if col not in self.test.columns]

        if missing_in_train or missing_in_valid or missing_in_test:
            raise ValueError(
                "Не найдены целевые столбцы во всех сплитах: "
                f"train={missing_in_train}, valid={missing_in_valid}, test={missing_in_test}"
            )

    def _validate_split_columns(self) -> None:
        train_cols = list(self.train.columns)
        valid_cols = list(self.valid.columns)
        test_cols = list(self.test.columns)

        if train_cols != valid_cols or train_cols != test_cols:
            missing_in_valid = [col for col in train_cols if col not in valid_cols]
            missing_in_test = [col for col in train_cols if col not in test_cols]
            extra_in_valid = [col for col in valid_cols if col not in train_cols]
            extra_in_test = [col for col in test_cols if col not in train_cols]
            raise ValueError(
                "Колонки train/valid/test должны совпадать и идти в одном порядке. "
                f"missing_in_valid={missing_in_valid}, extra_in_valid={extra_in_valid}, "
                f"missing_in_test={missing_in_test}, extra_in_test={extra_in_test}"
            )

        inferred_targets = [col for col in train_cols if col.startswith(self.target_prefix)]
        unknown_targets = [col for col in self.target_columns if col not in inferred_targets and col in train_cols]
        if unknown_targets:
            raise ValueError(
                "Целевые столбцы не соответствуют target_prefix. "
                f"target_prefix={self.target_prefix}, unknown_targets={unknown_targets}"
            )

    def _validate_no_target_leakage(self) -> None:
        leaked_targets = [col for col in self.columns_X if col.startswith(self.target_prefix)]
        if leaked_targets:
            raise ValueError(
                "Обнаружена утечка целевых столбцов во входные признаки: "
                f"{leaked_targets}"
            )
    
        missing_non_target_features = [
            col for col in self.train.columns
            if not col.startswith(self.target_prefix) and col not in self.columns_X
        ]
        if missing_non_target_features:
            raise ValueError(
                "Некоторые нецелевые признаки были ошибочно исключены из X: "
                f"{missing_non_target_features}"
            )


    
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Возвращает все данные в виде кортежа.
        
        Returns:
            (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        return (
            self.X_train, self.y_train,
            self.X_valid, self.y_valid,
            self.X_test, self.y_test
        )
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список имён признаков"""
        return list(self.columns_X)
    
    def get_target_names(self) -> List[str]:
        """Возвращает список имён целевых переменных"""
        return self.target_columns
    
    def info(self) -> str:
        """Возвращает информацию о данных"""
        return f"""
Данные:
  Признаков: {self.n_features}
  Целевых переменных: {self.n_targets}
  Train: {self.X_train.shape[0]} samples
  Valid: {self.X_valid.shape[0]} samples
  Test:  {self.X_test.shape[0]} samples
  Цели: {self.target_columns}
"""
    
    def __repr__(self) -> str:
        return f"Data(features={self.n_features}, targets={self.n_targets})"


def load_mtz_data(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загрузка данных МТЗ.
    
    Args:
        train_path, valid_path, test_path: пути к файлам
        verbose: выводить информацию
    
    Returns:
        (train, valid, test) DataFrames
    """
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)
    
    if verbose:
        print(f"Данные загружены:")
        print(f"  Train: {train.shape}")
        print(f"  Valid: {valid.shape}")
        print(f"  Test:  {test.shape}")
    
    return train, valid, test


def split_features_targets(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    target_prefix: str = 'H'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделение на признаки и целевые переменные.
    
    Args:
        df: исходный DataFrame
        target_columns: конкретные целевые столбцы
        target_prefix: префикс целевых переменных (по умолчанию 'H')
    
    Returns:
        (X, y) DataFrames
    """
    if target_columns:
        y = df[target_columns]
        feature_cols = [c for c in df.columns if c not in target_columns]
    else:
        target_cols = [c for c in df.columns if c.startswith(target_prefix)]
        y = df[target_cols]
        feature_cols = [c for c in df.columns if not c.startswith(target_prefix)]
    
    X = df[feature_cols]
    
    return X, y


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Получение информации о структуре данных.
    """
    from ..preprocessing.aggregation import analyze_data_structure
    return analyze_data_structure(df)
