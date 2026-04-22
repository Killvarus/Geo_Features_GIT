"""
Нейросетевые модели для решения обратной задачи МТЗ
"""
import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logging_utils import setup_logger


class SingleLayerPerceptron(nn.Module):
    """OLP-модель с опциональным скрытым слоем."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()

        if hidden_dim:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.network = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _plot_learning_curve(
    train_losses: List[float],
    val_losses: List[float],
    best_epoch: int,
    save_path: Optional[str] = None,
    title: str = "Кривая обучения"
) -> str:
    """Построение кривой обучения."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.axvline(x=best_epoch + 1, color='r', linestyle='--', label=f'Best epoch: {best_epoch + 1}')

    plt.xlabel('Эпохи')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    plt.show()
    plt.close()
    return ""


def _create_metrics_df(n: int, r2: List[float], mse: List[float], mae: List[float], pearson: List[float]) -> pd.DataFrame:
    """Создание DataFrame с метриками."""
    data = {'Мера оценки': ['R2', 'MSE', 'MAE', 'Pearson']}
    for i in range(n):
        data[f'Оценка{i+1}'] = [r2[i], mse[i], mae[i], pearson[i]]
    return pd.DataFrame(data).set_index('Мера оценки').T


def _fill_missing_values(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Единая обработка пропусков."""
    if any(df.isna().any().any() for df in dfs):
        print("⚠️ Обнаружены NaN, заполняем нулями")
        return tuple(df.fillna(0) for df in dfs)
    return dfs


def _scale_targets(
    y_train: pd.DataFrame,
    y_valid: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler_y = StandardScaler()

    if output_dim > 1:
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_valid_scaled = scaler_y.transform(y_valid.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    return y_train_scaled, y_valid_scaled, y_test_scaled, scaler_y


def _build_optimizer(model: nn.Module, optimizer_type: str, learning_rate: float, momentum: float):
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Неизвестный optimizer_type: {optimizer_type}")


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_dim: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    r2_values, mse_values, mae_values, pearson_values = [], [], [], []

    for i in range(output_dim):
        y_t = y_true[:, i] if output_dim > 1 else y_true.flatten()
        y_p = y_pred[:, i] if output_dim > 1 else y_pred.flatten()

        r2_values.append(r2_score(y_t, y_p) if not np.all(y_t == y_t[0]) else 0.0)
        mse_values.append(mean_squared_error(y_t, y_p))
        mae_values.append(mean_absolute_error(y_t, y_p))
        pearson_values.append(pearsonr(y_t, y_p)[0] if len(y_t) > 1 else 0.0)

    return r2_values, mse_values, mae_values, pearson_values


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# =============================================================================
def OLP(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    learning_rate: float = 0.01,
    num_epochs: int = 1000,
    patience: int = 100,
    tolerance: float = 1e-4,
    tolerance_mode: str = 'relative',
    graph: bool = False,
    save_plot_path: Optional[str] = None,
    plot_title: str = "Обучение OLP",
    hidden_dim: Optional[int] = 32,
    optimizer_type: str = 'sgd',
    momentum: float = 0.9,
    random_state: int = 42,
    save_model_path: Optional[str] = None,
    device: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_cv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Обучение OLP с early stopping по validation loss."""
    logger = setup_logger('src.models.neural_network', Path(log_file) if log_file else None)

    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    logger.info("Start OLP training | optimizer=%s | input_dim=%s | output_dim=%s | hidden_dim=%s | lr=%s | patience=%s | tolerance=%s | tolerance_mode=%s | random_state=%s | device=%s | enable_cv=%s",
                optimizer_type.upper(), input_dim, output_dim, hidden_dim, learning_rate, patience, tolerance, tolerance_mode, random_state, device, enable_cv)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_train, y_train, X_valid, y_valid, X_test, y_test = _fill_missing_values(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled, y_valid_scaled, y_test_scaled, scaler_y = _scale_targets(y_train, y_valid, y_test, output_dim)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)
    X_valid_t = torch.FloatTensor(X_valid_scaled)
    y_valid_t = torch.FloatTensor(y_valid_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)

    generator = torch.Generator().manual_seed(random_state)

    if use_cuda:
        X_train_t = X_train_t.to(device, non_blocking=True)
        y_train_t = y_train_t.to(device, non_blocking=True)
        X_valid_t = X_valid_t.to(device, non_blocking=True)
        y_valid_t = y_valid_t.to(device, non_blocking=True)
        X_test_t = X_test_t.to(device, non_blocking=True)
        train_loader = None
        valid_loader = None
    else:
        train_dataset = TensorDataset(X_train_t, y_train_t)
        valid_dataset = TensorDataset(X_valid_t, y_valid_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
        )

    model = SingleLayerPerceptron(input_dim, output_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, optimizer_type, learning_rate, momentum)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        if use_cuda:
            permutation = torch.randperm(X_train_t.size(0), device=device)
            for batch_indices in permutation.split(batch_size):
                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                batch_size_current = len(X_batch)
                train_loss_sum += loss.item() * batch_size_current
                train_samples += batch_size_current
        else:
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=False)
                y_batch = y_batch.to(device, non_blocking=False)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                batch_size_current = len(X_batch)
                train_loss_sum += loss.item() * batch_size_current
                train_samples += batch_size_current

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            if use_cuda:
                for batch_indices in torch.arange(X_valid_t.size(0), device=device).split(batch_size):
                    loss = criterion(model(X_valid_t[batch_indices]), y_valid_t[batch_indices])
                    batch_size_current = len(batch_indices)
                    val_loss_sum += loss.item() * batch_size_current
                    val_samples += batch_size_current
            else:
                for X_batch, y_batch in valid_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss = criterion(model(X_batch), y_batch)
                    batch_size_current = len(X_batch)
                    val_loss_sum += loss.item() * batch_size_current
                    val_samples += batch_size_current

        train_loss = train_loss_sum / max(train_samples, 1)
        val_loss = val_loss_sum / max(val_samples, 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if np.isfinite(best_val_loss):
            if tolerance_mode == 'relative':
                improvement_threshold = abs(best_val_loss) * tolerance
            elif tolerance_mode == 'absolute':
                improvement_threshold = tolerance
            else:
                raise ValueError("tolerance_mode must be 'relative' or 'absolute'")
        else:
            improvement_threshold = 0.0

        if val_loss < best_val_loss - improvement_threshold:
            improvement = best_val_loss - val_loss if np.isfinite(best_val_loss) else float('inf')
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if improvement != float('inf'):
                logger.info(
                    "Improvement | epoch=%s | best_val=%.6f | delta=%.6f | threshold=%.6f",
                    epoch + 1,
                    best_val_loss,
                    improvement,
                    improvement_threshold,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping | epoch=%s | best_epoch=%s | best_val=%.6f | no_improve=%s",
                    epoch + 1,
                    best_epoch + 1,
                    best_val_loss,
                    epochs_without_improvement,
                )
                break

        if (epoch + 1) % 100 == 0:
            logger.info(
                "Epoch %s | train=%.4f | val=%.4f | best=%.4f | no_improve=%s",
                epoch + 1,
                train_loss,
                val_loss,
                best_val_loss,
                epochs_without_improvement,
            )

    model.load_state_dict(best_state)

    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'optimizer': optimizer_type,
        'total_time': time.time() - start_time,
        'stopped_epoch': len(train_losses),
        'patience': patience,
        'tolerance': tolerance,
        'tolerance_mode': tolerance_mode,
        'random_state': random_state,
        'device': str(device),
    }

    if save_model_path:
        model_dir = os.path.dirname(save_model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': hidden_dim,
                'optimizer_type': optimizer_type,
                'learning_rate': learning_rate,
                'momentum': momentum,
                'best_epoch': best_epoch + 1,
                'best_val_loss': best_val_loss,
                'random_state': random_state,
                'tolerance': tolerance,
                'tolerance_mode': tolerance_mode,
                'device': str(device),
            },
            save_model_path,
        )
        training_history['model_path'] = save_model_path

    if graph or save_plot_path:
        plot_path = _plot_learning_curve(train_losses, val_losses, best_epoch, save_plot_path, plot_title)
        training_history['plot_path'] = plot_path

    if enable_cv:
        logger.info("Cross-validation started")
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_cv, mse_cv, mae_cv, pearson_cv = [], [], [], []

        for train_idx, val_idx in kf.split(X_train_scaled):
            fold_model = SingleLayerPerceptron(input_dim, output_dim, hidden_dim).to(device)
            fold_optimizer = _build_optimizer(fold_model, optimizer_type, learning_rate, momentum)
            X_fold_train = torch.FloatTensor(X_train_scaled[train_idx]).to(device)
            y_fold_train = torch.FloatTensor(y_train_scaled[train_idx]).to(device)
            X_fold_val = torch.FloatTensor(X_train_scaled[val_idx]).to(device)

            fold_model.train()
            for _ in range(min(100, num_epochs)):
                fold_optimizer.zero_grad()
                loss = criterion(fold_model(X_fold_train), y_fold_train)
                loss.backward()
                fold_optimizer.step()

            fold_model.eval()
            with torch.no_grad():
                pred_scaled = fold_model(X_fold_val).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled)
                true = scaler_y.inverse_transform(y_train_scaled[val_idx])

            fold_r2, fold_mse, fold_mae, fold_pearson = _evaluate_predictions(true, pred, output_dim)
            r2_cv.extend(fold_r2)
            mse_cv.extend(fold_mse)
            mae_cv.extend(fold_mae)
            pearson_cv.extend(fold_pearson)

        def _reshape_metric(values: List[float]) -> Tuple[List[float], List[float]]:
            arrays = [np.array(values[i::output_dim]) for i in range(output_dim)]
            means = [float(arr.mean()) for arr in arrays]
            stds = [float(arr.std(ddof=1)) if len(arr) > 1 else 0.0 for arr in arrays]
            return means, stds

        mean_r2, std_r2 = _reshape_metric(r2_cv)
        mean_mse, std_mse = _reshape_metric(mse_cv)
        mean_mae, std_mae = _reshape_metric(mae_cv)
        mean_pearson, std_pearson = _reshape_metric(pearson_cv)

        df_cv = _create_metrics_df(output_dim, mean_r2, mean_mse, mean_mae, mean_pearson)
        df_cv_err = _create_metrics_df(output_dim, std_r2, std_mse, std_mae, std_pearson)
    else:
        logger.info("Cross-validation skipped | enable_cv=False")
        df_cv = pd.DataFrame()
        df_cv_err = pd.DataFrame()

    logger.info("Testing started")
    model.eval()
    with torch.no_grad():
        test_input = X_test_t if use_cuda else X_test_t.to(device)
        test_pred_scaled = model(test_input).cpu().numpy()
        test_pred = scaler_y.inverse_transform(test_pred_scaled)
        test_true = scaler_y.inverse_transform(y_test_scaled)

    r2_test, mse_test, mae_test, pearson_test = _evaluate_predictions(test_true, test_pred, output_dim)
    df_test = _create_metrics_df(output_dim, r2_test, mse_test, mae_test, pearson_test)

    logger.info(
        "Test results | R2=%.4f ± %.4f | MSE=%.4f | MAE=%.4f | Pearson=%.4f",
        np.mean(r2_test),
        np.std(r2_test),
        np.mean(mse_test),
        np.mean(mae_test),
        np.mean(pearson_test),
    )

    return df_test, df_cv, df_cv_err, training_history


# =============================================================================
# ЭКСПОРТ В EXCEL
# =============================================================================
def to_excel_optimized_OLP(
    file_name: str,
    n_iter: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    learning_rate: float = 0.01,
    num_epochs: int = 1000,
    patience: int = 100,
    hidden_dim: int = 32,
    graph_first_only: bool = True,
    save_plots_dir: Optional[str] = None,
    save_models_dir: Optional[str] = None,
    optimizer_type: str = 'sgd',
    momentum: float = 0.9,
    tolerance: float = 1e-4,
    tolerance_mode: str = 'relative',
    random_state: int = 42,
    device: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_cv: bool = True,
) -> List[Dict]:
    """Запуск нескольких итераций обучения и сохранение в Excel."""
    logger = setup_logger('src.models.neural_network.batch', Path(log_file) if log_file else None)
    all_results = []
    logger.info("Batch OLP start | n_iter=%s | optimizer=%s | device=%s | enable_cv=%s", n_iter, optimizer_type.upper(), device or 'auto', enable_cv)

    for i in range(n_iter):
        logger.info("Iteration %s/%s started", i + 1, n_iter)

        current_graph = graph_first_only and i == 0
        current_plot_path = None
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            current_plot_path = f"{save_plots_dir}/learning_curve_iter_{i+1}.png"

        current_model_path = None
        if save_models_dir:
            os.makedirs(save_models_dir, exist_ok=True)
            current_model_path = f"{save_models_dir}/olp_iter_{i+1}.pt"

        df_test, df_cv, df_cv_err, history = OLP(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            graph=current_graph,
            save_plot_path=current_plot_path,
            plot_title=f"OLP ({optimizer_type.upper()}) - Итерация {i+1}",
            hidden_dim=hidden_dim,
            optimizer_type=optimizer_type,
            momentum=momentum,
            tolerance=tolerance,
            tolerance_mode=tolerance_mode,
            random_state=random_state + i,
            save_model_path=current_model_path,
            device=device,
            log_file=log_file,
            enable_cv=enable_cv,
        )

        all_results.append({
            'iteration': i + 1,
            'test': df_test,
            'cv': df_cv,
            'cv_errors': df_cv_err,
            'history': history,
        })

    try:
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            raw_data = []
            for r in all_results:
                row = {'Iteration': r['iteration']}

                for metric in ['R2', 'MSE', 'MAE', 'Pearson']:
                    for t in range(output_dim):
                        col = f'Оценка{t+1}'
                        if col in r['test'].index:
                            row[f'Test_{metric}_Target{t+1}'] = r['test'].loc[col, metric]

                row['Best_Epoch'] = r['history']['best_epoch'] + 1
                row['Best_Val_Loss'] = r['history']['best_val_loss']
                row['Time'] = r['history']['total_time']
                row['Model_Path'] = r['history'].get('model_path', '')
                raw_data.append(row)

            pd.DataFrame(raw_data).to_excel(writer, 'Raw_Results', index=False)

            summary = []
            if raw_data:
                for t in range(output_dim):
                    for metric in ['R2', 'MSE', 'MAE', 'Pearson']:
                        col = f'Test_{metric}_Target{t+1}'
                        if col in raw_data[0]:
                            vals = [r[col] for r in raw_data]
                            summary.append({
                                'Target': f'Target_{t+1}',
                                'Metric': metric,
                                'Mean': float(np.mean(vals)),
                                'Std': float(np.std(vals)),
                                'Min': float(np.min(vals)),
                                'Max': float(np.max(vals)),
                            })

            pd.DataFrame(summary).to_excel(writer, 'Summary', index=False)

            meta = pd.DataFrame({
                'Parameter': [
                    'n_iter', 'batch_size', 'input_dim', 'output_dim',
                    'learning_rate', 'num_epochs', 'patience', 'hidden_dim',
                    'optimizer', 'momentum', 'tolerance', 'tolerance_mode', 'random_state', 'device', 'enable_cv'
                ],
                'Value': [
                    n_iter, batch_size, input_dim, output_dim,
                    learning_rate, num_epochs, patience, hidden_dim,
                    optimizer_type, momentum, tolerance, tolerance_mode, random_state, device or 'auto', enable_cv
                ]
            })
            meta.to_excel(writer, 'Meta', index=False)

        logger.info("Results saved to %s", file_name)
    except Exception as e:
        logger.exception("Failed to save Excel results to %s: %s", file_name, e)

    return all_results


__all__ = ['SingleLayerPerceptron', 'OLP', 'to_excel_optimized_OLP']
