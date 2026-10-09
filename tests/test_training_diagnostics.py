import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.training_diagnostics import (
    DiagnosticConfig,
    detect_spikes,
    train_olp_with_diagnostics,
)


def test_detect_spikes_finds_isolated_peaks():
    train = [0.1] * 20
    train[5] = 8.0
    train[12] = 11.0
    idx, threshold, median = detect_spikes(train, abs_floor=1.0, median_factor=8.0)
    assert median == 0.1
    assert threshold == 1.0
    assert list(idx + 1) == [6, 13]


def test_detect_spikes_ignores_small_wiggles():
    train = [0.08, 0.09, 0.12, 0.07, 0.15, 0.1]
    idx, _, _ = detect_spikes(train, abs_floor=1.0, median_factor=8.0)
    assert list(idx) == []


def test_diagnostic_trainer_writes_epoch_log_and_finite_values(tmp_path: Path):
    rng = np.random.default_rng(0)
    n, p = 64, 6
    x = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.DataFrame({"H2_8": x["f0"] * 0.2 + rng.normal(scale=0.05, size=n)})
    x_val = x.iloc[:16].copy()
    y_val = y.iloc[:16].copy()
    out = tmp_path / "run"
    summary = train_olp_with_diagnostics(
        x, y, x_val, y_val, x_val, y_val,
        out_dir=out,
        config=DiagnosticConfig(
            batch_size=16,
            input_dim=p,
            output_dim=1,
            learning_rate=0.01,
            num_epochs=3,
            hidden_dim=8,
            random_state=0,
            device="cpu",
            run_name="unit",
        ),
    )
    assert (out / "epoch_log.csv").exists()
    assert (out / "loss_grad_lr.png").exists()
    rows = list(csv.DictReader((out / "epoch_log.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert {row["learning_rate"] for row in rows} == {"0.01"}
    assert all(np.isfinite(float(row["train_loss"])) for row in rows)
    assert all(np.isfinite(float(row["val_loss"])) for row in rows)
    assert summary["n_epochs"] == 3
    assert summary["nan_or_inf_any"] is False


def test_diagnostic_trainer_epoch_loss_is_sample_weighted(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(1)
    n, p = 40, 4
    x = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.DataFrame({"t": rng.normal(size=n)})
    x_val = x.iloc[:8].copy()
    y_val = y.iloc[:8].copy()

    captured = []
    real_mse = torch.nn.MSELoss()

    class SpyMSE(torch.nn.MSELoss):
        def forward(self, input, target):  # noqa: A003
            loss = real_mse(input, target)
            captured.append((int(input.shape[0]), float(loss.item())))
            return loss

    monkeypatch.setattr("src.models.training_diagnostics.nn.MSELoss", SpyMSE)
    summary = train_olp_with_diagnostics(
        x, y, x_val, y_val, x_val, y_val,
        out_dir=tmp_path / "weighted",
        config=DiagnosticConfig(
            batch_size=16,
            input_dim=p,
            output_dim=1,
            num_epochs=1,
            hidden_dim=4,
            shuffle=False,
            drop_last=False,
            random_state=1,
            device="cpu",
        ),
    )
    train_batches = captured[:3]  # 16+16+8
    assert [n_i for n_i, _ in train_batches] == [16, 16, 8]
    expected = sum(loss * n_i for n_i, loss in train_batches) / n
    logged = float(list(csv.DictReader((tmp_path / "weighted" / "epoch_log.csv").open(encoding="utf-8")))[0]["train_loss"])
    assert abs(logged - expected) < 1e-5
    assert summary["n_epochs"] == 1


def test_summarize_run_accepts_csv_strings():
    from src.models.training_diagnostics import summarize_run_for_table

    summary = {
        "run_name": "A",
        "config": {
            "random_state": 1,
            "learning_rate": 0.01,
            "optimizer_type": "adam",
            "clip_grad_norm": None,
            "drop_last": False,
            "batch_size": 128,
        },
        "n_epochs": 2,
        "n_spikes": 1,
        "spike_epochs": [2],
        "train_median": 0.1,
        "train_max": 5.0,
        "val_max": 0.2,
        "best_epoch_plot": 1,
        "nan_or_inf_any": False,
    }
    rows = [
        {
            "epoch": "1",
            "grad_norm_max": "0.2",
            "update_norm": "0.01",
            "adam_v_min": "1e-4",
            "worst_batch_is_last": "0",
        },
        {
            "epoch": "2",
            "grad_norm_max": "9.0",
            "update_norm": "1.2",
            "adam_v_min": "1e-8",
            "worst_batch_is_last": "1",
        },
    ]
    table = summarize_run_for_table(summary, rows)
    assert table["worst_batch_is_last_frac_spike"] == 1.0
    assert table["grad_norm_mean_spike"] == 9.0
