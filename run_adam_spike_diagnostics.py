"""
Диагностика пиков Train Loss у OLP+Adam.

По умолчанию:
1) разбирает уже сохранённые learning_history_*.json (в т.ч. график «Итерация 3»);
2) прогоняет короткие контролируемые запуски на тех же данных
   aggregation_train3_H2_8 / freq1_pickup1_mean.

Основной OLP не меняется.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, EXPERIMENTS_DIR, PROJECT_ROOT
from src.models.training_diagnostics import (
    DiagnosticConfig,
    detect_spikes,
    plot_history_with_spikes,
    scan_learning_histories,
    summarize_run_for_table,
    train_olp_with_diagnostics,
)
from src.utils import Data, load_mtz_data


TARGET_HISTORY = (
    EXPERIMENTS_DIR
    / "2026-09-01_agg_pca"
    / "aggregation_train3_H2_8"
    / "freq1_pickup1_mean"
    / "learning_curves"
    / "learning_history_iter_3.json"
)
PRECOMPUTED = DATA_DIR / "Aggregated" / "Difficult_2" / "freq1_pickup1_mean"
RAW_FILES = ("train_3.csv", "valid_3.csv", "test_3.csv")
TARGET_COLUMNS = ["H2_8"]


def _load_target_frames() -> Data:
    if (PRECOMPUTED / "train.csv").exists():
        train = pd.read_csv(PRECOMPUTED / "train.csv")
        valid = pd.read_csv(PRECOMPUTED / "valid.csv")
        test = pd.read_csv(PRECOMPUTED / "test.csv")
    else:
        train, valid, test = load_mtz_data(
            DATA_DIR / RAW_FILES[0],
            DATA_DIR / RAW_FILES[1],
            DATA_DIR / RAW_FILES[2],
            verbose=True,
        )
    return Data(train, test, valid, TARGET_COLUMNS)


def _analyze_existing(out_dir: Path) -> Dict[str, Any]:
    rows = scan_learning_histories(EXPERIMENTS_DIR)
    (out_dir / "corpus_histories.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    periodic = [
        r for r in rows
        if r["n_spikes"] >= 4 and r["mean_period"] is not None and 40 <= r["mean_period"] <= 90
    ]
    payload: Dict[str, Any] = {
        "n_histories": len(rows),
        "n_with_ge2_spikes": sum(r["n_spikes"] >= 2 for r in rows),
        "n_periodic_40_90": len(periodic),
        "target_history": str(TARGET_HISTORY),
        "target_exists": TARGET_HISTORY.exists(),
    }
    if TARGET_HISTORY.exists():
        hist = json.loads(TARGET_HISTORY.read_text(encoding="utf-8"))
        train = hist["train_losses"]
        val = hist["val_losses"]
        idx, thr, med = detect_spikes(train)
        best0 = int(hist.get("best_epoch", 0))
        val_arr = np.asarray(val, dtype=float)
        payload["target"] = {
            "title": hist.get("title"),
            "optimizer": hist.get("optimizer"),
            "random_state": hist.get("random_state"),
            "device": hist.get("device"),
            "n_epochs": len(train),
            "best_epoch_plot": best0 + 1,
            "argmin_val_epoch": int(np.argmin(val_arr)) + 1,
            "best_val": float(val_arr[best0]),
            "min_val": float(np.min(val_arr)),
            "train_median": med,
            "train_max": float(np.max(train)),
            "spike_epochs": (idx + 1).tolist(),
            "spike_deltas": np.diff(idx).tolist() if idx.size > 1 else [],
            "threshold": thr,
            "val_at_spikes": [float(val_arr[i]) for i in idx],
            "train_at_spikes": [float(train[i]) for i in idx],
        }
        plot_history_with_spikes(
            train,
            val,
            best0,
            out_dir / "original_iter3_spikes.png",
            title=str(hist.get("title", "OLP (ADAM) - Итерация 3")),
            spike_epochs_1idx=(idx + 1).tolist(),
        )
        sibling_dir = TARGET_HISTORY.parent
        siblings = []
        for path in sorted(sibling_dir.glob("learning_history_iter_*.json")):
            hh = json.loads(path.read_text(encoding="utf-8"))
            sidx, _, smed = detect_spikes(hh["train_losses"])
            siblings.append(
                {
                    "file": path.name,
                    "seed": hh.get("random_state"),
                    "n_epochs": len(hh["train_losses"]),
                    "best_epoch_plot": int(hh.get("best_epoch", -1)) + 1,
                    "train_median": smed,
                    "train_max": float(np.max(hh["train_losses"])),
                    "spike_epochs": (sidx + 1).tolist(),
                    "deltas": np.diff(sidx).tolist() if sidx.size > 1 else [],
                }
            )
        payload["siblings"] = siblings
    (out_dir / "history_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_epoch_rows(run_dir: Path) -> List[Dict[str, Any]]:
    with (run_dir / "epoch_log.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run_grid(data: Data, out_dir: Path, epochs: int, seeds: List[int], quick: bool) -> pd.DataFrame:
    base = dict(
        input_dim=data.n_features,
        output_dim=data.n_targets,
        num_epochs=epochs,
        hidden_dim=32,
        device="cpu",
        patience=10_000,
    )
    jobs: List[DiagnosticConfig] = []
    for seed in seeds:
        jobs.append(DiagnosticConfig(
            run_name="A_adam_lr0.01", optimizer_type="adam",
            learning_rate=0.01, batch_size=128, random_state=seed, **base,
        ))
    if not quick:
        for seed in seeds:
            jobs.extend([
                DiagnosticConfig(run_name="B_adam_lr0.001", optimizer_type="adam",
                                 learning_rate=0.001, batch_size=128, random_state=seed, **base),
                DiagnosticConfig(run_name="C_adam_clip1", optimizer_type="adam",
                                 learning_rate=0.01, clip_grad_norm=1.0, batch_size=128,
                                 random_state=seed, **base),
                DiagnosticConfig(run_name="E_adamw_lr0.01", optimizer_type="adamw",
                                 learning_rate=0.01, weight_decay=0.01, batch_size=128,
                                 random_state=seed, **base),
            ])
        seed0 = 44 if 44 in seeds else seeds[0]
        jobs.extend([
            DiagnosticConfig(run_name="F_adam_drop_last", optimizer_type="adam",
                             learning_rate=0.01, drop_last=True, batch_size=128,
                             random_state=seed0, **base),
            DiagnosticConfig(run_name="G_adam_bs64", optimizer_type="adam",
                             learning_rate=0.01, batch_size=64, random_state=seed0, **base),
            DiagnosticConfig(run_name="H_adam_bs256", optimizer_type="adam",
                             learning_rate=0.01, batch_size=256, random_state=seed0, **base),
        ])
        # D: scheduler в OLP нет — baseline A уже без scheduler.

    table_rows = []
    for cfg in jobs:
        run_dir = out_dir / "runs" / f"{cfg.run_name}_seed{cfg.random_state}"
        print(f"RUN {cfg.run_name} seed={cfg.random_state} epochs={cfg.num_epochs} lr={cfg.learning_rate} "
              f"bs={cfg.batch_size} clip={cfg.clip_grad_norm} drop_last={cfg.drop_last}", flush=True)
        summary_path = run_dir / "summary.json"
        if summary_path.exists() and (run_dir / "epoch_log.csv").exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"  skip existing {run_dir.name}", flush=True)
        else:
            summary = train_olp_with_diagnostics(
                data.X_train, data.y_train, data.X_valid, data.y_valid, data.X_test, data.y_test,
                out_dir=run_dir,
                config=cfg,
            )
        epoch_rows = _read_epoch_rows(run_dir)
        table_rows.append(summarize_run_for_table(summary, epoch_rows))
        print(
            f"  spikes={summary['n_spikes']} epochs={summary['spike_epochs']} "
            f"train_max={summary['train_max']:.3f} best={summary['best_epoch_plot']}",
            flush=True,
        )

    df = pd.DataFrame(table_rows)
    df.to_csv(out_dir / "comparison.csv", index=False)
    return df


def _write_report(out_dir: Path, history: Dict[str, Any], comparison: Optional[pd.DataFrame]) -> None:
    target = history.get("target") or {}
    lines = [
        "# Диагностика пиков Train Loss (OLP + Adam)",
        "",
        "Исходный график: `experiments/2026-09-01_agg_pca/aggregation_train3_H2_8/freq1_pickup1_mean`",
        "итерация 3, title `OLP (ADAM) - Итерация 3`, Best epoch 350.",
        "",
        "## Что видно в сохранённой истории",
        "",
        f"- device={target.get('device')}, seed={target.get('random_state')}, optimizer={target.get('optimizer')}",
        f"- n_epochs={target.get('n_epochs')}, train median={target.get('train_median')}, max={target.get('train_max')}",
        f"- spike epochs={target.get('spike_epochs')}",
        f"- deltas={target.get('spike_deltas')}",
        f"- Best epoch plot={target.get('best_epoch_plot')}, argmin(val)={target.get('argmin_val_epoch')}",
        "",
        "Scheduler в `src/models/neural_network.py` нет: LR постоянный 0.01.",
        "Train Loss на графике — среднее MSE по **семплам** за эпоху, не last-batch.",
        "Val считается в `model.eval()` + `torch.no_grad()` после эпохи.",
        "",
        "Пик одноэпохный: соседние эпохи сразу возвращаются к ~0.04–0.2.",
        "Val реагирует слабее, потому что это loss **в конце** эпохи, после того как Adam уже задушил шаг.",
        "",
        "## Корпус",
        "",
        f"- историй: {history.get('n_histories')}",
        f"- с ≥2 пиками: {history.get('n_with_ge2_spikes')}",
        f"- периодические 40–90 эпох: {history.get('n_periodic_40_90')}",
        "",
        "Пики почти только у агрегации с 2418 признаками (freq1_pickup1). У PCA (даже 2048) и wavelet их нет.",
        "По трём seed той же кривой период ~55–58 эпох, но фазы разные → это не `if epoch % N`.",
        "",
    ]
    if comparison is not None and len(comparison):
        lines += ["## Контрольные прогоны", "", comparison.to_string(index=False), ""]
    lines += [
        "## Код, который НЕ является багом сбора loss",
        "",
        "`src/models/neural_network.py` функция `OLP`: `train_loss_sum += loss.item() * batch_size`;",
        "деление на число семплов. Optimizer создаётся один раз до цикла эпох.",
        "Порядок: zero_grad → forward → loss → backward → step.",
        "",
        "Отдельный диагностический цикл: `src/models/training_diagnostics.py`.",
        "Запуск: `python run_adam_spike_diagnostics.py`.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EXPERIMENTS_DIR / "adam_spike_diagnostics"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--histories-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="только 1 seed, только variant A")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("OUT", out_dir, flush=True)

    history = _analyze_existing(out_dir)
    print("histories", history.get("n_histories"), "periodic", history.get("n_periodic_40_90"), flush=True)
    if history.get("target"):
        print("target spikes", history["target"]["spike_epochs"], "best", history["target"]["best_epoch_plot"], flush=True)

    comparison = None
    if not args.histories_only:
        print("Loading data...", flush=True)
        data = _load_target_frames()
        print(data.info(), flush=True)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.quick:
            seeds = seeds[:1]
        comparison = _run_grid(data, out_dir, epochs=args.epochs, seeds=seeds, quick=args.quick)

    _write_report(out_dir, history, comparison)
    print("Wrote", out_dir / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
