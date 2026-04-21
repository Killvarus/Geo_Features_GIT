"""
Методы отбора признаков для обратной задачи МТЗ
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import traceback
import time
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import tqdm


# =============================================================================
# IFS - Итеративный отбор признаков
# =============================================================================
def IFS_feature_selection(df_X: pd.DataFrame, df_y: pd.DataFrame, 
                          level_xx: float = 0.8, level_xy: float = 0.1) -> Tuple[pd.DataFrame, List, pd.Series]:
    """
    Итеративный отбор признаков на основе корреляций.
    
    Args:
        df_X: признаки
        df_y: целевая переменная
        level_xx: порог корреляции между признаками
        level_xy: порог корреляции с целевой
    
    Returns:
        (filtered_df, selected_features, feature_correlations)
    """
    print("Начало работы функции IFS_feature_selection...")
    
    try:
        if isinstance(df_y, pd.Series):
            df_y = df_y.to_frame()
        
        if len(df_X) == 0 or len(df_y) == 0:
            print("Ошибка: Пустые данные")
            return pd.DataFrame(), [], pd.Series()
        
        corr_XX = df_X.corr().abs()
        corr_XY = df_X.corrwith(df_y.iloc[:, 0]).abs()
        
        corXY = corr_XY.copy().values
        mask = [False] * len(corr_XX)
        selected_indices = []
        
        iteration = 0
        while np.max(corXY) > level_xy and iteration < 100000:
            iteration += 1
            i_best = np.argmax(corXY)
            best_corr = corXY[i_best]
            
            mask[i_best] = True
            selected_indices.append(i_best)
            corXY[i_best] = 0
            
            for i in range(corr_XX.shape[0]):
                if not mask[i] and corXY[i] > 0 and corr_XX.iloc[i_best, i] > level_xx:
                    corXY[i] = 0
        
        if not selected_indices:
            print(f"Предупреждение: Не выбрано ни одного признака")
            return pd.DataFrame(), [], pd.Series()
        
        features = [df_X.columns[i] for i in selected_indices]
        correlations = corr_XY[features]
        
        sorted_features = sorted(zip(correlations.values, features), reverse=True)
        sorted_features = [f for _, f in sorted_features]
        
        print(f"Успешно завершено! Выбрано {len(sorted_features)} признаков")
        
        return df_X[sorted_features], sorted_features, correlations[sorted_features]
    
    except Exception as e:
        print(f"Ошибка в IFS_feature_selection: {e}")
        traceback.print_exc()
        return pd.DataFrame(), [], pd.Series()
    

def IFS_feature_selection_auto(
    df_X: pd.DataFrame, 
    df_y: pd.DataFrame, 
    n_features_target: int,
    level_xx_range: Tuple[float, float] = (0.5, 0.95),
    level_xy_range: Tuple[float, float] = (0.01, 0.5),
    max_iterations: int = 100000
) -> Tuple[pd.DataFrame, List, pd.Series, Dict]:
    """
    Итеративный отбор признаков с автоматическим подбором параметров.
    
    Args:
        df_X: признаки
        df_y: целевая переменная
        n_features_target: целевое количество признаков
        level_xx_range: диапазон для level_xx
        level_xy_range: диапазон для level_xy
        max_iterations: макс. итераций
    
    Returns:
        (filtered_df, selected_features, feature_correlations, best_params)
    """
    def single_IFS(df_X, df_y, level_xx, level_xy):
        if isinstance(df_y, pd.Series):
            df_y = df_y.to_frame()
    
        corr_XX = df_X.corr().abs()
        corr_XY = df_X.corrwith(df_y.iloc[:, 0]).abs()
        
        corXY = corr_XY.copy().values
        mask = [False] * len(corr_XX)
        selected_indices = []
        
        while np.max(corXY) > level_xy:
            i_best = np.argmax(corXY)
            mask[i_best] = True
            selected_indices.append(i_best)
            corXY[i_best] = 0
            
            for i in range(corr_XX.shape[0]):
                if not mask[i] and corXY[i] > 0 and corr_XX.iloc[i_best, i] > level_xx:
                    corXY[i] = 0
        
        if not selected_indices:
            return [], pd.Series()
        
        features = [df_X.columns[i] for i in selected_indices]
        correlations = corr_XY[features]
        
        sorted_features = [x for _, x in sorted(zip(correlations.values, features), reverse=True)]
        
        return sorted_features, correlations[sorted_features]
    
    n_total_features = df_X.shape[1]
    if n_features_target > n_total_features:
        n_features_target = n_total_features
    
    print(f"Целевое количество признаков: {n_features_target}")
    
    # Адаптивная стратегия
    level_xx = 0.7
    level_xy = 0.05
    step_size_xx = 0.1
    step_size_xy = 0.02
    
    best_params = None
    best_features = []
    best_correlations = pd.Series()
    
    for iteration in range(max_iterations):
        features, correlations = single_IFS(df_X, df_y, level_xx, level_xy)
        n_selected = len(features)
        
        if n_selected == 0:
            level_xy *= 0.7
            level_xx *= 0.9
            continue
        
        diff = n_selected - n_features_target
        
        if abs(diff) <= 1:
            best_params = {'level_xx': level_xx, 'level_xy': level_xy}
            best_features = features
            best_correlations = correlations
            break
        
        if diff > 0:
            level_xy = min(level_xy + step_size_xy, level_xy_range[1])
            level_xx = min(level_xx + step_size_xx, level_xx_range[1])
        else:
            level_xy = max(level_xy - step_size_xy, level_xy_range[0])
            level_xx = max(level_xx - step_size_xx, level_xx_range[0])
        
        step_size_xx *= 0.8
        step_size_xy *= 0.8
    
    # Если не нашли точное решение
    if best_params is None:
        corr_XY = df_X.corrwith(df_y.iloc[:, 0] if isinstance(df_y, pd.DataFrame) else df_y).abs()
        top_features = corr_XY.nlargest(n_features_target)
        best_features = top_features.index.tolist()
        best_correlations = top_features
        best_params = {'level_xx': 0.7, 'level_xy': 0.05}
    
    filtered_df = df_X[best_features]
    
    print(f"Выбрано признаков: {len(best_features)}")
    
    return filtered_df, best_features, best_correlations, best_params


# =============================================================================
# TRUE BACKWARD FEATURE SELECTION
# =============================================================================
class TrueBackwardFeatureSelection:
    """
    Настоящий backward elimination с индивидуальным ранжированием признаков.
    """
    
    def __init__(self, estimator, n_features_to_drop=100, cv=5, 
                 scoring='neg_mean_squared_error', random_state=42):
        self.estimator = estimator
        self.n_features_to_drop = n_features_to_drop
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.ranking_ = None
        self.individual_scores_ = {}
    
    def fit(self, X, y, X_val=None, y_val=None):
        features = X.columns.tolist()
        n_features = len(features)
        current_features = features.copy()
        
        feature_group_ranks = {feature: 0 for feature in features}
        feature_individual_ranks = {feature: 0 for feature in features}
        self.individual_scores_ = {feature: 0 for feature in features}
        
        current_group_rank = 1
        global_individual_rank = 1
        
        print(f"🚀 Запуск TRUE backward elimination")
        print(f"📊 Всего признаков: {n_features}")
        
        iteration = 0
        
        with tqdm(total=n_features, desc="Backward Elimination") as pbar:
            while len(current_features) > self.n_features_to_drop:
                iteration += 1
                current_count = len(current_features)
                
                feature_scores = self._train_models_sequential(
                    X[current_features], y
                )
                
                self.individual_scores_.update(feature_scores)
                
                all_current_sorted = sorted(feature_scores.items(), key=lambda x: x[1])
                
                for individual_rank, (feature, score) in enumerate(all_current_sorted, global_individual_rank):
                    feature_individual_ranks[feature] = individual_rank
                
                global_individual_rank += self.n_features_to_drop
                
                features_to_drop = [f for f, _ in all_current_sorted[:self.n_features_to_drop]]
                for feature in features_to_drop:
                    feature_group_ranks[feature] = current_group_rank
                
                current_features = [f for f in current_features if f not in features_to_drop]
                current_group_rank += 1
                pbar.update(self.n_features_to_drop)
        
        if current_features:
            remaining_scores = self._train_models_sequential(X[current_features], y)
            self.individual_scores_.update(remaining_scores)
            
            remaining_sorted = sorted(remaining_scores.items(), key=lambda x: x[1])
            for individual_rank, (feature, score) in enumerate(remaining_sorted, global_individual_rank):
                feature_group_ranks[feature] = current_group_rank
                feature_individual_ranks[feature] = individual_rank
        
        self.ranking_ = {
            'group_rank': feature_group_ranks,
            'individual_rank': feature_individual_ranks,
            'scores': self.individual_scores_
        }
        
        print(f"✅ Завершено! Всего итераций: {iteration}")
        return self
    
    def _train_models_sequential(self, X, y):
        feature_scores = {}
        features = X.columns.tolist()
        
        for feature in tqdm(features, desc="Оценка признаков", leave=False):
            try:
                X_reduced = X.drop(columns=[feature])
                score = self._evaluate_model(X_reduced, y)
                feature_scores[feature] = score
            except Exception as e:
                feature_scores[feature] = 0.0
        
        return feature_scores
    
    def _evaluate_model(self, X, y):
        model = clone(self.estimator)
        
        try:
            if self.cv is not None:
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
                return float(np.mean(scores))
            else:
                model.fit(X, y)
                y_pred = model.predict(X)
                from sklearn.metrics import mean_squared_error
                return -float(mean_squared_error(y, y_pred))
        except:
            return 0.0
    
    def get_ranking_df(self):
        if self.ranking_ is None:
            raise ValueError("Сначала выполните fit()")
        
        features = list(self.individual_scores_.keys())
        
        data = {
            'Feature': features,
            'Group_Rank': [self.ranking_['group_rank'][f] for f in features],
            'Individual_Rank': [self.ranking_['individual_rank'][f] for f in features],
            'Score': [self.individual_scores_[f] for f in features]
        }
        
        df_ranking = pd.DataFrame(data)
        df_ranking = df_ranking.sort_values('Individual_Rank').reset_index(drop=True)
        
        return df_ranking
    
    def save_ranking(self, filename, output_dir="Results"):
        if self.ranking_ is None:
            raise ValueError("Сначала выполните fit()")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        df_ranking = self.get_ranking_df()
        file_path = output_path / f"{filename}.csv"
        
        df_ranking.to_csv(file_path, index=False)
        
        print(f"✅ Ранжирование сохранено: {file_path}")
        
        return file_path


def run_true_backward_selection(
    X_train, y_train, model,
    n_features_to_drop=100, cv=5,
    output_filename="true_backward_ranking",
    output_dir="Results"
):
    """
    Запуск настоящего backward elimination.
    """
    print("🎯 Запуск TRUE backward elimination")
    print(f"📊 Модель: {type(model).__name__}")
    
    selector = TrueBackwardFeatureSelection(
        estimator=model,
        n_features_to_drop=n_features_to_drop,
        cv=cv
    )
    
    selector.fit(X_train, y_train)
    selector.save_ranking(output_filename, output_dir)
    
    return selector


# =============================================================================
# NEURAL NETWORK WEIGHTS FEATURE SELECTION
# =============================================================================
def NN_weight(
    X_train, y_train, X_valid, y_valid,
    batch_size, input_dim, output_dim,
    learning_rate, num_epochs, patience, graph=False
):
    """
    Отбор признаков на основе весов нейронной сети.
    """
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_valid.values, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    class SingleLayerPerceptron(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SingleLayerPerceptron, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, output_dim)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            return self.fc2(x)
    
    model = SingleLayerPerceptron(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    counter = 0
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                val_outputs = model(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
        
        val_loss /= len(val_dataloader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    model.eval()
    
    # Извлечение весов
    fc1_weights = model.fc1.weight.data.cpu().numpy()
    fc2_weights = model.fc2.weight.data.cpu().numpy()
    
    abs_fc1 = np.abs(fc1_weights)
    abs_fc2 = np.abs(fc2_weights)
    
    importance = np.zeros(input_dim)
    for i in range(input_dim):
        s = 0.0
        for k in range(output_dim):
            for j in range(32):
                s += abs_fc2[k, j] * abs_fc1[j, i]
        importance[i] = s
    
    if importance.sum() > 0:
        feature_importance = importance / importance.sum()
    else:
        feature_importance = importance
    
    sorted_indices = np.argsort(feature_importance)[::-1]
    top_750_indices = sorted_indices[:750]
    X_train_top_750 = X_train.iloc[:, top_750_indices]
    
    return X_train_top_750, feature_importance


def get_discrete_selected_features(
    X_train, y_train, X_valid, y_valid,
    batch_size, input_dim, output_dim,
    learning_rate, num_epochs, patience,
    runs=5, top_n=750, threshold=3, graph=False
):
    """
    Дискретный отбор признаков на основе множества запусков.
    """
    feature_counts = {}
    
    for _ in range(runs):
        _, weights = NN_weight(
            X_train, y_train, X_valid, y_valid,
            batch_size, input_dim, output_dim,
            learning_rate, num_epochs, patience, graph=False
        )
        
        top_indices = np.argsort(-np.abs(weights))[:top_n]
        
        for idx in top_indices:
            if idx in feature_counts:
                feature_counts[idx] += 1
            else:
                feature_counts[idx] = 1
    
    features = np.array(list(feature_counts.keys()))
    counts = np.array(list(feature_counts.values()))
    
    selected_mask = counts >= threshold
    selected_features = features[selected_mask]
    selected_counts = counts[selected_mask]
    
    sort_order = np.argsort(-selected_counts)
    selected_features = selected_features[sort_order]
    selected_counts = selected_counts[sort_order]
    
    print(f"Итоговое число отобранных признаков: {len(selected_features)}")
    
    return selected_features, selected_counts


# =============================================================================
# FEATURE RANKING PROCESSOR
# =============================================================================
class FeatureRankingProcessor:
    """
    Обработчик для ранжирования признаков.
    
    Логика создания папок:
    ----------------------
    base_dir/
    ├── Results/                    # Excel-файлы с метриками
    └── Learning_Curves/            # Графики обучения
        └── {model}_{k}features_{optimizer}/
    """
    
    def __init__(self, train, test, valid, base_dir, layer='H3', target_columns=None, 
                 k_step=30, ranking_prefix="300ranking", optimizer_type='sgd', n_iter=5,
                 num_epochs=1000, learning_rate=0.01, momentum=0.9,
                 patience=100, tolerance=1e-4, ranking_dfs=None, model_names=None):
        
        
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / 'Results'
        self.learning_curves_dir = self.base_dir / 'Learning_Curves'
        self.layer = layer
        self.target_columns = target_columns or ['H3_8']
        self.k_step = k_step
        self.ranking_prefix = ranking_prefix
        self.optimizer_type = optimizer_type
        self.n_iter = n_iter
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.patience = patience

        self.tolerance = tolerance 
        
        self.results_dir.mkdir(exist_ok=True)
        self.learning_curves_dir.mkdir(exist_ok=True)
        
        self.train, self.valid, self.test = train, valid, test
        self.ranking_dfs = ranking_dfs or []
        
        if model_names is None:
            self.model_names = [f"model_{i}" for i in range(len(self.ranking_dfs))]
        else:
            if len(model_names) != len(self.ranking_dfs):
                raise ValueError("Длина model_names должна совпадать с длиной ranking_dfs")
            self.model_names = list(model_names)
    
    def _extract_features(self, df_rank: pd.DataFrame) -> List[str]:
        """Извлечение валидных признаков"""
        features = df_rank.iloc[:, 0].tolist()
        return [f for f in features if f in self.train.columns]
    
    def _get_k_values(self, max_k: int) -> List[int]:
        """Генерация значений k"""
        if max_k < self.k_step:
            return [max_k]
        k_values = list(range(self.k_step, max_k + 1, self.k_step))
        if max_k not in k_values:
            k_values.append(max_k)
        return k_values
    
    def process_ranking_df(self, df_rank: pd.DataFrame, model_name: str):
        """Обработка DataFrame ранжирования"""
        features = self._extract_features(df_rank)
        if not features:
            print(f"❌ Нет валидных признаков для {model_name}")
            return
        
        print(f"\n🔍 {model_name}: {len(features)} признаков")
        
        for k in self._get_k_values(len(features)):
            self._process_k(k, features, model_name)
    
    def _process_k(self, k: int, features: List[str], model_name: str):
        """Обработка конкретного k"""
        selected = features[:k]
        
        X_train = self.train[selected]
        X_valid = self.valid[selected]
        X_test = self.test[selected]
        y_train = self.train[self.target_columns]
        y_valid = self.valid[self.target_columns]
        y_test = self.test[self.target_columns]
        
        excel_name = self.results_dir / f"{k}features_{model_name}_{self.layer}_{self.optimizer_type}_metrics.xlsx"
        curves_dir = self.learning_curves_dir / f"{model_name}_{k}features_{self.optimizer_type}"
        curves_dir.mkdir(exist_ok=True)
        
        # Импортируем функцию обучения
        from .neural_network import to_excel_optimized_OLP
        
        to_excel_optimized_OLP(
            file_name=str(excel_name),
            n_iter=self.n_iter,
            X_train=X_train, y_train=y_train,
            X_valid=X_valid, y_valid=y_valid,
            X_test=X_test, y_test=y_test,
            batch_size=64,
            input_dim=len(selected),
            output_dim=len(self.target_columns),
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            patience=self.patience,
            hidden_dim=32,
            save_plots_dir=str(curves_dir),
            optimizer_type=self.optimizer_type,
            momentum=self.momentum,
            tolerance=self.tolerance
        )
        
        
        print(f"   ✅ k={k}: {excel_name.name}")
    
    def run(self):
        """Запуск обработки"""
        print(f"\n{'='*60}")
        print(f"ЗАПУСК ОБРАБОТКИ")
        print(f"{'='*60}")
        print(f"Моделей: {len(self.ranking_dfs)}")
        print(f"Шаг k: {self.k_step}")
        print(f"Оптимизатор: {self.optimizer_type.upper()}")

        
        start_time = time.time()
        
        for df_rank, model_name in zip(self.ranking_dfs, self.model_names):
            self.process_ranking_df(df_rank, model_name)
        
        duration = time.time() - start_time
        print(f"\n✅ Завершено за {duration:.1f} сек")
        
        self._save_summary(duration)
    
    def _save_summary(self, duration: float):
        """Сохранение сводки"""
        summary_file = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("СВОДКА ЭКСПЕРИМЕНТА\n")
            f.write("=" * 50 + "\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Время выполнения: {duration:.1f} сек\n")
            f.write(f"Моделей: {len(self.ranking_dfs)}\n")
            f.write(f"Оптимизатор: {self.optimizer_type}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")

            f.write(f"Целевые: {self.target_columns}\n")
