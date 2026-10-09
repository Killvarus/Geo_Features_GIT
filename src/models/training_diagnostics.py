"""
Диагностика обучения OLP: epoch/batch loss, градиенты, Adam-моменты, LR.

Не меняет основной цикл `OLP`. Используется отдельным скриптом и тестами.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .neural_network import SingleLayerPerceptron, _fill_missing_values, _scale_targets


EPOCH_CSV_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_eval_loss",
    "val_loss",
    "batch_loss_min",
    "batch_loss_max",
    "worst_batch_size",
    "worst_batch_idx",
    "worst_batch_is_last",
    "grad_norm_max",
    "grad_norm_mean",
    "grad_abs_max",
    "param_norm",
    "param_abs_max",
    "update_norm",
    "adam_v_min",
    "adam_v_mean",
    "adam_step_proxy",
    "n_nan_input",
    "n_inf_input",
    "n_nan_pred",
    "n_inf_pred",
    "n_nan_loss",
    "n_inf_loss",
    "n_nan_grad",
    "n_inf_grad",
    "n_nan_param",
    "n_inf_param",
]


@dataclass
class DiagnosticConfig:
    batch_size: int = 128
    input_dim: int = 0
    output_dim: int = 1
    learning_rate: float = 0.01
    num_epochs: int = 90
    patience: int = 10_000
    hidden_dim: int = 32
    optimizer_type: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    drop_last: bool = False
    shuffle: bool = True
    random_state: int = 42
    device: str = "cpu"
    spike_abs_floor: float = 1.0
    spike_median_factor: float = 8.0
    max_spike_batches_per_epoch: int = 8
    run_name: str = "adam"


def detect_spikes(
    train_losses: Sequence[float],
    abs_floor: float = 1.0,
    median_factor: float = 8.0,
) -> Tuple[np.ndarray, float, float]:
    """Индексы (0-based) эпох, где train loss >> типичного уровня."""
    tr = np.asarray(train_losses, dtype=float)
    finite = tr[np.isfinite(tr)]
    median = float(np.median(finite)) if finite.size else 0.0
    threshold = max(float(abs_floor), float(median_factor) * max(median, 1e-12))
    idx = np.where(np.isfinite(tr) & (tr > threshold))[0]
    return idx, threshold, median


def scan_learning_histories(
    experiments_dir: Path,
    abs_floor: float = 1.0,
    median_factor: float = 8.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(experiments_dir)
    for path in root.rglob("learning_history*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        train = payload.get("train_losses") or []
        val = payload.get("val_losses") or []
        if len(train) < 20:
            continue
        idx, threshold, median = detect_spikes(train, abs_floor=abs_floor, median_factor=median_factor)
        deltas = np.diff(idx).astype(float) if idx.size > 1 else np.array([])
        rows.append(
            {
                "path": str(path.relative_to(root)) if root in path.parents or path.parent == root else str(path),
                "title": payload.get("title"),
                "optimizer": payload.get("optimizer"),
                "random_state": payload.get("random_state"),
                "device": payload.get("device"),
                "n_epochs": len(train),
                "best_epoch_plot": int(payload.get("best_epoch", -1)) + 1,
                "train_median": median,
                "train_max": float(np.nanmax(train)),
                "val_max": float(np.nanmax(val)) if val else None,
                "n_spikes": int(idx.size),
                "spike_epochs": (idx + 1).tolist(),
                "mean_period": float(np.mean(deltas)) if deltas.size else None,
                "threshold": threshold,
            }
        )
    return rows


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x_f = x.detach().float()
    return {
        "min": float(x_f.min().item()),
        "max": float(x_f.max().item()),
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()) if x_f.numel() > 1 else 0.0,
    }


def _grad_stats(model: nn.Module) -> Tuple[float, float, int, int]:
    sq_sum = 0.0
    max_abs = 0.0
    n_nan = 0
    n_inf = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        n_nan += int(torch.isnan(grad).sum().item())
        n_inf += int(torch.isinf(grad).sum().item())
        if grad.numel() == 0:
            continue
        sq_sum += float(grad.pow(2).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return math.sqrt(sq_sum), max_abs, n_nan, n_inf


def _param_stats(model: nn.Module) -> Tuple[float, float, int, int]:
    sq_sum = 0.0
    max_abs = 0.0
    n_nan = 0
    n_inf = 0
    for param in model.parameters():
        data = param.detach()
        n_nan += int(torch.isnan(data).sum().item())
        n_inf += int(torch.isinf(data).sum().item())
        sq_sum += float(data.pow(2).sum().item())
        if data.numel():
            max_abs = max(max_abs, float(data.abs().max().item()))
    return math.sqrt(sq_sum), max_abs, n_nan, n_inf


def _param_delta_norm(before: List[torch.Tensor], model: nn.Module) -> float:
    sq_sum = 0.0
    for old, param in zip(before, model.parameters()):
        sq_sum += float((param.detach() - old).pow(2).sum().item())
    return math.sqrt(sq_sum)


def _snapshot_params(model: nn.Module) -> List[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters()]


def _adam_stats(optimizer: optim.Optimizer) -> Tuple[float, float, float]:
    v_min = math.inf
    v_sum = 0.0
    v_count = 0
    step_sum = 0.0
    step_count = 0
    eps = float(optimizer.param_groups[0].get("eps", 1e-8))
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state or "exp_avg_sq" not in state:
                continue
            v = state["exp_avg_sq"].detach()
            m = state["exp_avg"].detach()
            v_min = min(v_min, float(v.min().item()) if v.numel() else v_min)
            v_sum += float(v.mean().item())
            v_count += 1
            denom = v.sqrt() + eps
            step_sum += float((m.abs() / denom).mean().item())
            step_count += 1
    if v_count == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        v_min if math.isfinite(v_min) else float("nan"),
        v_sum / v_count,
        step_sum / max(step_count, 1),
    )


def _current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    name = optimizer_type.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def _per_output_mse(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    diff = (pred - target).pow(2)
    if diff.ndim == 1:
        return [float(diff.mean().item())]
    return [float(diff[:, i].mean().item()) for i in range(diff.shape[1])]


def _write_epoch_row(writer: csv.DictWriter, row: Dict[str, Any]) -> None:
    writer.writerow({key: row.get(key, "") for key in EPOCH_CSV_FIELDS})


def plot_loss_grad_lr(
    epoch_rows: Sequence[Dict[str, Any]],
    save_path: Path,
    title: str,
    spike_epochs: Optional[Sequence[int]] = None,
) -> Path:
    epochs = [int(r["epoch"]) for r in epoch_rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(epochs, [r["train_loss"] for r in epoch_rows], label="Train Loss (batch avg)", lw=1.6)
    axes[0].plot(epochs, [r["val_loss"] for r in epoch_rows], label="Val Loss", lw=1.6)
    if epoch_rows and epoch_rows[0].get("train_eval_loss") not in (None, ""):
        axes[0].plot(
            epochs,
            [r["train_eval_loss"] for r in epoch_rows],
            label="Train eval (epoch end)",
            lw=1.2,
            ls="--",
        )
    if spike_epochs:
        for ep in spike_epochs:
            axes[0].axvline(ep, color="0.75", lw=0.8)
    axes[0].set_ylabel("MSE")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(epochs, [r["grad_norm_max"] for r in epoch_rows], label="max ||g||", color="C3", lw=1.6)
    axes[1].plot(epochs, [r["grad_norm_mean"] for r in epoch_rows], label="mean ||g||", color="C1", lw=1.2)
    axes[1].set_ylabel("Gradient norm")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    if any((r.get("grad_norm_max") or 0) > 0 for r in epoch_rows):
        axes[1].set_yscale("log")

    axes[2].plot(epochs, [r["learning_rate"] for r in epoch_rows], color="C2", lw=1.6)
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_history_with_spikes(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    best_epoch_0idx: int,
    save_path: Path,
    title: str,
    spike_epochs_1idx: Sequence[int],
) -> Path:
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(epochs, train_losses, label="Train Loss", lw=1.6)
    ax.plot(epochs, val_losses, label="Val Loss", lw=1.6)
    ax.axvline(best_epoch_0idx + 1, color="r", ls="--", label=f"Best epoch: {best_epoch_0idx + 1}")
    for ep in spike_epochs_1idx:
        ax.axvline(ep, color="0.7", lw=0.7, alpha=0.8)
    ax.set_xlabel("Эпохи")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


def preprocess_like_olp(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dim: int,
) -> Dict[str, Any]:
    X_train, y_train, X_valid, y_valid, X_test, y_test = _fill_missing_values(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(X_train)
    x_valid = scaler_x.transform(X_valid)
    x_test = scaler_x.transform(X_test)
    y_train_s, y_valid_s, y_test_s, scaler_y = _scale_targets(y_train, y_valid, y_test, output_dim)
    return {
        "X_train": x_train,
        "X_valid": x_valid,
        "X_test": x_test,
        "y_train": y_train_s,
        "y_valid": y_valid_s,
        "y_test": y_test_s,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "stats": {
            "X_train_raw": _ndarray_stats(X_train.to_numpy()),
            "y_train_raw": _ndarray_stats(y_train.to_numpy()),
            "X_train_scaled": _ndarray_stats(x_train),
            "y_train_scaled": _ndarray_stats(y_train_s),
            "X_valid_scaled": _ndarray_stats(x_valid),
            "y_valid_scaled": _ndarray_stats(y_valid_s),
            "n_train": int(x_train.shape[0]),
            "n_valid": int(x_valid.shape[0]),
            "n_features": int(x_train.shape[1]),
            "output_dim": int(output_dim),
        },
    }


def _ndarray_stats(arr: np.ndarray) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p99_abs": float(np.quantile(np.abs(finite), 0.99)),
        "frac_abs_gt_8": float(np.mean(np.abs(finite) > 8.0)),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
    }


def train_olp_with_diagnostics(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    out_dir: Path,
    config: DiagnosticConfig,
) -> Dict[str, Any]:
    """Тот же Adam/MSE цикл, что OLP, плюс логи эпохи и аномальных batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    prepared = preprocess_like_olp(
        X_train, y_train, X_valid, y_valid, X_test, y_test, config.output_dim
    )
    device = torch.device(config.device)
    x_train_t = torch.tensor(prepared["X_train"], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(prepared["y_train"], dtype=torch.float32, device=device)
    x_valid_t = torch.tensor(prepared["X_valid"], dtype=torch.float32, device=device)
    y_valid_t = torch.tensor(prepared["y_valid"], dtype=torch.float32, device=device)
    index_t = torch.arange(x_train_t.size(0), device=device)

    generator = torch.Generator().manual_seed(config.random_state)
    train_ds = TensorDataset(x_train_t.cpu(), y_train_t.cpu(), index_t.cpu())
    valid_ds = TensorDataset(x_valid_t.cpu(), y_valid_t.cpu())
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        generator=generator if config.shuffle else None,
    )
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
    n_batches = len(train_loader)

    model = SingleLayerPerceptron(config.input_dim, config.output_dim, config.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, config.optimizer_type, config.learning_rate, config.weight_decay)

    epoch_csv = out_dir / "epoch_log.csv"
    spike_jsonl = out_dir / "spike_batches.jsonl"
    epoch_rows: List[Dict[str, Any]] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    with epoch_csv.open("w", encoding="utf-8", newline="") as csv_file, spike_jsonl.open(
        "w", encoding="utf-8"
    ) as spike_file:
        writer = csv.DictWriter(csv_file, fieldnames=EPOCH_CSV_FIELDS)
        writer.writeheader()

        for epoch in range(config.num_epochs):
            model.train()
            params_before = _snapshot_params(model)
            train_sum = 0.0
            train_n = 0
            batch_losses: List[float] = []
            worst = {
                "loss": -1.0,
                "batch_idx": -1,
                "batch_size": 0,
                "indices": None,
                "x": None,
                "y": None,
                "pred": None,
            }
            grad_norms: List[float] = []
            epoch_grad_abs_max = 0.0
            n_nan_grad = 0
            n_inf_grad = 0
            n_nan_pred = 0
            n_inf_pred = 0
            n_nan_loss = 0
            n_inf_loss = 0
            n_nan_input = 0
            n_inf_input = 0
            captured = 0

            for batch_idx, (x_batch, y_batch, idx_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                n_nan_input += int(torch.isnan(x_batch).sum().item() + torch.isnan(y_batch).sum().item())
                n_inf_input += int(torch.isinf(x_batch).sum().item() + torch.isinf(y_batch).sum().item())

                optimizer.zero_grad()
                outputs = model(x_batch)
                n_nan_pred += int(torch.isnan(outputs).sum().item())
                n_inf_pred += int(torch.isinf(outputs).sum().item())
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss):
                    n_nan_loss += 1
                if torch.isinf(loss):
                    n_inf_loss += 1
                loss.backward()
                g_norm, g_abs, g_nan, g_inf = _grad_stats(model)
                grad_norms.append(g_norm)
                epoch_grad_abs_max = max(epoch_grad_abs_max, g_abs)
                n_nan_grad += g_nan
                n_inf_grad += g_inf
                if config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()

                batch_n = int(x_batch.size(0))
                batch_loss = float(loss.item())
                train_sum += batch_loss * batch_n
                train_n += batch_n
                batch_losses.append(batch_loss)
                if batch_loss > worst["loss"]:
                    worst = {
                        "loss": batch_loss,
                        "batch_idx": batch_idx,
                        "batch_size": batch_n,
                        "indices": idx_batch.detach().cpu().tolist(),
                        "x": x_batch.detach(),
                        "y": y_batch.detach(),
                        "pred": outputs.detach(),
                    }

                local_median = float(np.median(batch_losses)) if batch_losses else 0.0
                spike_cut = max(config.spike_abs_floor, config.spike_median_factor * max(local_median, 1e-12))
                if batch_loss > spike_cut and captured < config.max_spike_batches_per_epoch:
                    captured += 1
                    rec = {
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx,
                        "n_batches": n_batches,
                        "is_last_batch": batch_idx == n_batches - 1,
                        "batch_size": batch_n,
                        "loss": batch_loss,
                        "per_output_mse": _per_output_mse(outputs, y_batch),
                        "grad_norm": g_norm,
                        "grad_abs_max": g_abs,
                        "indices": idx_batch.detach().cpu().tolist(),
                        "X": _tensor_stats(x_batch),
                        "y": _tensor_stats(y_batch),
                        "pred": _tensor_stats(outputs),
                        "abs_err": _tensor_stats((outputs - y_batch).abs()),
                        "sq_err": _tensor_stats((outputs - y_batch).pow(2)),
                        "max_abs_err": float((outputs - y_batch).abs().max().item()),
                        "lr": _current_lr(optimizer),
                    }
                    spike_file.write(json.dumps(rec) + "\n")

            model.eval()
            val_sum = 0.0
            val_n = 0
            with torch.no_grad():
                train_eval = float(criterion(model(x_train_t), y_train_t).item())
                for x_batch, y_batch in valid_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    vloss = criterion(model(x_batch), y_batch)
                    val_sum += float(vloss.item()) * int(x_batch.size(0))
                    val_n += int(x_batch.size(0))
            train_loss = train_sum / max(train_n, 1)
            val_loss = val_sum / max(val_n, 1)
            param_norm, param_abs, n_nan_param, n_inf_param = _param_stats(model)
            update_norm = _param_delta_norm(params_before, model)
            v_min, v_mean, step_proxy = _adam_stats(optimizer)
            worst_is_last = int(worst["batch_idx"] == n_batches - 1) if n_batches else 0

            row = {
                "epoch": epoch + 1,
                "learning_rate": _current_lr(optimizer),
                "train_loss": train_loss,
                "train_eval_loss": train_eval,
                "val_loss": val_loss,
                "batch_loss_min": min(batch_losses) if batch_losses else float("nan"),
                "batch_loss_max": max(batch_losses) if batch_losses else float("nan"),
                "worst_batch_size": worst["batch_size"],
                "worst_batch_idx": worst["batch_idx"],
                "worst_batch_is_last": worst_is_last,
                "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "grad_abs_max": epoch_grad_abs_max,
                "param_norm": param_norm,
                "param_abs_max": param_abs,
                "update_norm": update_norm,
                "adam_v_min": v_min,
                "adam_v_mean": v_mean,
                "adam_step_proxy": step_proxy,
                "n_nan_input": n_nan_input,
                "n_inf_input": n_inf_input,
                "n_nan_pred": n_nan_pred,
                "n_inf_pred": n_inf_pred,
                "n_nan_loss": n_nan_loss,
                "n_inf_loss": n_inf_loss,
                "n_nan_grad": n_nan_grad,
                "n_inf_grad": n_inf_grad,
                "n_nan_param": n_nan_param,
                "n_inf_param": n_inf_param,
            }
            _write_epoch_row(writer, row)
            csv_file.flush()
            epoch_rows.append(row)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    break

    spike_idx, threshold, median = detect_spikes(
        train_losses, abs_floor=config.spike_abs_floor, median_factor=config.spike_median_factor
    )
    plot_path = plot_loss_grad_lr(
        epoch_rows,
        out_dir / "loss_grad_lr.png",
        title=f"{config.run_name} seed={config.random_state}",
        spike_epochs=(spike_idx + 1).tolist(),
    )
    summary = {
        "run_name": config.run_name,
        "config": asdict(config),
        "n_train": int(x_train_t.size(0)),
        "n_valid": int(x_valid_t.size(0)),
        "n_batches": n_batches,
        "last_batch_size_if_no_drop": int(x_train_t.size(0) % config.batch_size or config.batch_size),
        "preprocess": prepared["stats"],
        "n_epochs": len(train_losses),
        "best_epoch_plot": best_epoch + 1,
        "best_val_loss": best_val,
        "argmin_val_epoch": int(np.argmin(val_losses)) + 1 if val_losses else None,
        "train_median": median,
        "train_max": float(np.max(train_losses)) if train_losses else None,
        "val_max": float(np.max(val_losses)) if val_losses else None,
        "n_spikes": int(spike_idx.size),
        "spike_epochs": (spike_idx + 1).tolist(),
        "mean_period": float(np.mean(np.diff(spike_idx))) if spike_idx.size > 1 else None,
        "threshold": threshold,
        "plot_path": str(plot_path),
        "epoch_log": str(epoch_csv),
        "spike_batches": str(spike_jsonl),
        "nan_or_inf_any": bool(
            any(
                (r["n_nan_input"] or r["n_inf_input"] or r["n_nan_pred"] or r["n_inf_pred"]
                 or r["n_nan_loss"] or r["n_inf_loss"] or r["n_nan_grad"] or r["n_inf_grad"]
                 or r["n_nan_param"] or r["n_inf_param"])
                for r in epoch_rows
            )
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_run_for_table(summary: Dict[str, Any], epoch_rows: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    spikes = summary.get("spike_epochs") or []
    spike_set = set(spikes)
    normal = []
    spiked = []
    if epoch_rows:
        for row in epoch_rows:
            bucket = spiked if int(row["epoch"]) in spike_set else normal
            bucket.append(row)

    def _mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None

    return {
        "run": summary.get("run_name"),
        "seed": summary.get("config", {}).get("random_state"),
        "lr": summary.get("config", {}).get("learning_rate"),
        "optimizer": summary.get("config", {}).get("optimizer_type"),
        "clip": summary.get("config", {}).get("clip_grad_norm"),
        "drop_last": summary.get("config", {}).get("drop_last"),
        "batch_size": summary.get("config", {}).get("batch_size"),
        "n_epochs": summary.get("n_epochs"),
        "n_spikes": summary.get("n_spikes"),
        "spike_epochs": spikes,
        "train_median": summary.get("train_median"),
        "train_max": summary.get("train_max"),
        "val_max": summary.get("val_max"),
        "best_epoch": summary.get("best_epoch_plot"),
        "grad_norm_mean_normal": _mean(normal, "grad_norm_max"),
        "grad_norm_mean_spike": _mean(spiked, "grad_norm_max"),
        "update_norm_normal": _mean(normal, "update_norm"),
        "update_norm_spike": _mean(spiked, "update_norm"),
        "adam_v_min_normal": _mean(normal, "adam_v_min"),
        "adam_v_min_spike": _mean(spiked, "adam_v_min"),
        "worst_batch_is_last_frac_spike": (
            float(np.mean([float(r["worst_batch_is_last"]) for r in spiked])) if spiked else None
        ),
        "nan_or_inf_any": summary.get("nan_or_inf_any"),
    }
