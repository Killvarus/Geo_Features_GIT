"""
One-off OLP+SGD run on H3_8 (aggregation train3 / freq1_pickup1_mean).

Does not change production Adam configs or OLP defaults.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, EXPERIMENTS_DIR
from src.experiment import AggregationExperiment
from src.models.training_diagnostics import detect_spikes
from src.utils import load_mtz_data


TARGET_COLUMNS = ["H3_8"]
PRECOMPUTED_ROOT = DATA_DIR / "Aggregated" / "Difficult_2"
PRECOMPUTED_CFG = PRECOMPUTED_ROOT / "freq1_pickup1_mean"
RAW_FILES = ("train_3.csv", "valid_3.csv", "test_3.csv")
EXPERIMENT_NAME = "2026-09-16_sgd_h3_8_one"

HIDDEN_DIM = 32
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 1000
N_ITER = 1
OPTIMIZER = "sgd"
MOMENTUM = 0.9
PATIENCE = 300
TOLERANCE = 0.003
TOLERANCE_MODE = "relative"
ENABLE_CV = False
DEVICE = "auto"


def _load_raw_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_mtz_data(
        DATA_DIR / RAW_FILES[0],
        DATA_DIR / RAW_FILES[1],
        DATA_DIR / RAW_FILES[2],
        verbose=True,
    )


def _analyze_history(history_path: Path) -> dict:
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    train = np.asarray(hist.get("train_losses") or [], dtype=float)
    val = np.asarray(hist.get("val_losses") or [], dtype=float)
    best0 = int(hist.get("best_epoch", -1))

    train_idx, train_thr, train_med = detect_spikes(train)
    val_idx, val_thr, val_med = detect_spikes(val)

    train_max = float(np.max(train)) if train.size else float("nan")
    val_max = float(np.max(val)) if val.size else float("nan")
    adam_like_train = bool(train_idx.size > 0 and train_max >= 1.0 and train_max / max(train_med, 1e-12) >= 8.0)

    return {
        "history_path": str(history_path),
        "title": hist.get("title"),
        "optimizer": hist.get("optimizer"),
        "device": hist.get("device"),
        "random_state": hist.get("random_state"),
        "n_epochs": int(len(train)),
        "best_epoch_0based": best0,
        "best_epoch_plot": best0 + 1 if best0 >= 0 else None,
        "best_val_from_history": float(val[best0]) if 0 <= best0 < len(val) else None,
        "min_val": float(np.min(val)) if val.size else None,
        "argmin_val_epoch": int(np.argmin(val)) + 1 if val.size else None,
        "train_median": train_med,
        "train_max": train_max,
        "train_max_over_median": float(train_max / max(train_med, 1e-12)) if np.isfinite(train_max) else None,
        "train_spike_threshold": train_thr,
        "train_spike_epochs": (train_idx + 1).tolist(),
        "n_train_spikes": int(train_idx.size),
        "val_median": val_med,
        "val_max": val_max,
        "val_max_over_median": float(val_max / max(val_med, 1e-12)) if np.isfinite(val_max) else None,
        "val_spike_threshold": val_thr,
        "val_spike_epochs": (val_idx + 1).tolist(),
        "n_val_spikes": int(val_idx.size),
        "adam_like_train_spikes": adam_like_train,
    }


def main() -> None:
    print("OUT", EXPERIMENTS_DIR / EXPERIMENT_NAME, flush=True)
    print(
        f"OLP SGD H3_8 | n_iter={N_ITER} lr={LEARNING_RATE} bs={BATCH_SIZE} "
        f"hidden={HIDDEN_DIM} patience={PATIENCE} tol={TOLERANCE} "
        f"epochs={NUM_EPOCHS} momentum={MOMENTUM}",
        flush=True,
    )

    precomputed_ok = all(
        (PRECOMPUTED_CFG / name).exists() for name in ("train.csv", "valid.csv", "test.csv")
    )
    print("precomputed", PRECOMPUTED_CFG, "exists" if precomputed_ok else "MISSING", flush=True)

    print("Loading raw splits for AggregationExperiment...", flush=True)
    train, valid, test = _load_raw_splits()

    experiment = AggregationExperiment(
        train=train,
        valid=valid,
        test=test,
        target_columns=TARGET_COLUMNS,
        experiment_name=EXPERIMENT_NAME,
        base_dir=EXPERIMENTS_DIR / EXPERIMENT_NAME,
    )

    result = experiment.run_single(
        freq_step=1,
        pickup_step=1,
        agg_method="mean",
        n_iter=N_ITER,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        patience=PATIENCE,
        tolerance=TOLERANCE,
        tolerance_mode=TOLERANCE_MODE,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        save_transformed_data=False,
        device=DEVICE,
        enable_cv=ENABLE_CV,
        precomputed_aggregated_root=str(PRECOMPUTED_ROOT) if precomputed_ok else None,
        force_rerun=True,
    )

    cfg_dir = experiment.base_dir / "freq1_pickup1_mean"
    curves_dir = cfg_dir / "learning_curves"
    png_path = curves_dir / "learning_curve_iter_1.png"
    history_path = curves_dir / "learning_history_iter_1.json"
    excel_path = cfg_dir / "results" / "metrics.xlsx"

    spike_info = {}
    if history_path.exists():
        spike_info = _analyze_history(history_path)

    test_r2 = test_mse = None
    best_epoch = best_val = None
    if excel_path.exists():
        raw = pd.read_excel(excel_path, sheet_name="Raw_Results")
        if "Test_R2_Target1" in raw.columns:
            test_r2 = float(raw["Test_R2_Target1"].iloc[0])
        if "Test_MSE_Target1" in raw.columns:
            test_mse = float(raw["Test_MSE_Target1"].iloc[0])
        if "Best_Epoch" in raw.columns:
            best_epoch = int(raw["Best_Epoch"].iloc[0])
        if "Best_Val_Loss" in raw.columns:
            best_val = float(raw["Best_Val_Loss"].iloc[0])

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "output_dir": str(experiment.base_dir),
        "config_dir": str(cfg_dir),
        "learning_curve_png": str(png_path) if png_path.exists() else None,
        "learning_history_json": str(history_path) if history_path.exists() else None,
        "metrics_xlsx": str(excel_path) if excel_path.exists() else None,
        "target": TARGET_COLUMNS,
        "optimizer": OPTIMIZER,
        "momentum": MOMENTUM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "tolerance": TOLERANCE,
        "tolerance_mode": TOLERANCE_MODE,
        "enable_cv": ENABLE_CV,
        "device": DEVICE,
        "n_iter": N_ITER,
        "n_samples_train": result.n_samples_train,
        "n_samples_valid": result.n_samples_valid,
        "n_samples_test": result.n_samples_test,
        "n_features": result.n_features,
        "precomputed_used": precomputed_ok,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_r2": test_r2 if test_r2 is not None else result.r2_mean,
        "test_mse": test_mse if test_mse is not None else result.mse_mean,
        "total_time_seconds": result.total_time_seconds,
        "spikes": spike_info,
    }
    out_summary = experiment.base_dir / "sgd_h3_8_one_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60, flush=True)
    print("DONE", flush=True)
    print("output_dir", summary["output_dir"], flush=True)
    print("png", summary["learning_curve_png"], flush=True)
    print("history", summary["learning_history_json"], flush=True)
    print("best_epoch", summary["best_epoch"], "best_val", summary["best_val_loss"], flush=True)
    print("test_r2", summary["test_r2"], "test_mse", summary["test_mse"], flush=True)
    print(
        "n_samples", result.n_samples_train,
        "n_features", result.n_features,
        "lr", LEARNING_RATE,
        "batch", BATCH_SIZE,
        "momentum", MOMENTUM,
        flush=True,
    )
    print("train_median", spike_info.get("train_median"), "train_max", spike_info.get("train_max"), flush=True)
    print("train_spike_epochs", spike_info.get("train_spike_epochs"), flush=True)
    print("val_spike_epochs", spike_info.get("val_spike_epochs"), flush=True)
    print("adam_like_train_spikes", spike_info.get("adam_like_train_spikes"), flush=True)
    print("summary_json", out_summary, flush=True)


if __name__ == "__main__":
    main()
