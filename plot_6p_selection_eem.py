"""
Карты отобранных каналов на матрице возбуждения-испускания.

Для каждого недавнего отбора 6P и каждого иона берётся первая группа рейтинга
и рисуется на среднем спектре train. Готового рейтинга нет — ион пропускается.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.visualization.eem_selection import first_packet, mean_train_matrix, save_selection_map

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
OUT_DIR = ROOT / "experiments" / "6p_selection_eem"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
METHODS = [
    ("6p_tbs_per_ion", "Обратный отбор, MSE на train"),
    ("6p_tbs_per_ion_valid", "Обратный отбор, MSE на validation"),
    ("6p_forward_ridge_valid", "Прямой отбор, Ridge alpha=700"),
    ("6p_forward_ols_valid", "Прямой отбор, без регуляризации"),
    ("6p_ridge_rfe_backward", "Обратный RFE, Ridge alpha=700"),
    ("6p_ridge_rfe_forward", "Прямой RFE, Ridge alpha=700"),
]


def plot_available(cache: Path = CACHE, out_dir: Path = OUT_DIR) -> tuple[int, int]:
    import pandas as pd

    if not cache.exists():
        raise FileNotFoundError(f"нет кэша спектров: {cache}")
    matrix = mean_train_matrix(np.load(cache)["X"])
    saved = 0
    missing = 0
    for folder, title in METHODS:
        for ion in TARGETS:
            ranking_path = ROOT / "experiments" / folder / ion / "ranking.csv"
            if not ranking_path.exists():
                print("missing", ranking_path, flush=True)
                missing += 1
                continue
            packet = first_packet(pd.read_csv(ranking_path))
            destination = out_dir / folder / f"{ion}.png"
            save_selection_map(
                matrix,
                packet["Feature"].astype(str).tolist(),
                f"{title}\n{ion}: первая группа, {len(packet)} каналов",
                destination,
            )
            print("plot", destination, flush=True)
            saved += 1
    print(f"saved {saved}, missing {missing}", flush=True)
    return saved, missing


def main() -> None:
    plot_available()


if __name__ == "__main__":
    main()
