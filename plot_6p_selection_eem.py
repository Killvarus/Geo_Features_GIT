"""
Карты отобранных каналов на матрице возбуждения-испускания.

Для каждого недавнего отбора 6P и каждого иона берётся первая группа рейтинга
и рисуется на среднем спектре train. Готового рейтинга нет — ион пропускается.

--parts N: весь рейтинг делится на N частей по порядку (1 — начало списка),
каждый канал окрашивается номером своей части. Картинки — в
experiments/6p_selection_eem/parts<N>/<папка>/<ион>.png и all_ions.png.

--top K [--methods папка ...]: как в основном режиме, но вместо первой группы —
первые K каналов рейтинга (у отборов с пакетом 10% или 1 первая группа — это
один канал). По умолчанию — три текущих отбора с alpha=700. Картинки —
experiments/6p_selection_eem/<папка>/<ион>_top<K>.png.

--parts N --data-order: то же, но вместо рейтинга — порядок колонок в самом
массиве данных (колонки 1…5427 кэша). Он одинаков для всех ионов, поэтому
картинка одна: experiments/6p_selection_eem/parts<N>/data_order.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.visualization.eem_selection import (
    first_packet,
    mean_train_matrix,
    parts_matrix,
    rank_parts,
    save_parts_grid,
    save_parts_map,
    save_selection_map,
    top_features,
)

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
# Рейтинги для раскраски по частям списка.
PARTS_METHODS = [
    ("6p_ppop_ridge_a700_step0.1_cv5", "ППОП, Ridge alpha=700, пакет 10%, 5-fold"),
    ("6p_rfe_ridge_a700_step1", "RFE, Ridge alpha=700, по одному каналу"),
    ("6p_rfe_ridge_a700_step0.1", "RFE, Ridge alpha=700, пакет 10%"),
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


def plot_top(top: int, folders: list[str], cache: Path = CACHE, out_dir: Path = OUT_DIR) -> tuple[int, int]:
    """Первые top каналов каждого рейтинга на среднем спектре, по картинке на ион."""
    import pandas as pd

    if not cache.exists():
        raise FileNotFoundError(f"нет кэша спектров: {cache}")
    titles = dict(METHODS + PARTS_METHODS)
    matrix = mean_train_matrix(np.load(cache)["X"])
    saved = 0
    missing = 0
    for folder in folders:
        title = titles.get(folder, folder)
        for ion in TARGETS:
            ranking_path = ROOT / "experiments" / folder / ion / "ranking.csv"
            if not ranking_path.exists():
                print("missing", ranking_path, flush=True)
                missing += 1
                continue
            features = top_features(pd.read_csv(ranking_path), top)
            destination = out_dir / folder / f"{ion}_top{top}.png"
            save_selection_map(matrix, features, f"{title}\n{ion}: первые {len(features)} каналов", destination)
            print("plot", destination, flush=True)
            saved += 1
    print(f"saved {saved}, missing {missing}", flush=True)
    return saved, missing


def plot_parts(n_parts: int, cache: Path = CACHE, out_dir: Path = OUT_DIR) -> int:
    import pandas as pd

    if not cache.exists():
        raise FileNotFoundError(f"нет кэша спектров: {cache}")
    spectrum = mean_train_matrix(np.load(cache)["X"])
    saved = 0
    for folder, title in PARTS_METHODS:
        panels = []
        for ion in TARGETS:
            ranking_path = ROOT / "experiments" / folder / ion / "ranking.csv"
            if not ranking_path.exists():
                print("missing", ranking_path, flush=True)
                continue
            parts = parts_matrix(rank_parts(pd.read_csv(ranking_path), n_parts))
            destination = out_dir / f"parts{n_parts}" / folder / f"{ion}.png"
            save_parts_map(spectrum, parts, f"{title}\n{ion}: части списка", destination, n_parts)
            panels.append((ion, parts))
            saved += 1
        if panels:
            grid = out_dir / f"parts{n_parts}" / folder / "all_ions.png"
            save_parts_grid(spectrum, panels, f"{title}: части списка, 1 — начало", grid, n_parts)
            print("plot", grid, flush=True)
    print(f"saved {saved}", flush=True)
    return saved


def plot_data_order(n_parts: int, cache: Path = CACHE, out_dir: Path = OUT_DIR) -> Path:
    """Части массива по номеру колонки: колонка i (с 1) — канал с тем же номером."""
    import pandas as pd

    if not cache.exists():
        raise FileNotFoundError(f"нет кэша спектров: {cache}")
    X = np.load(cache)["X"]
    columns = np.arange(1, X.shape[1] + 1)
    order = pd.DataFrame({"Feature": columns.astype(str), "Individual_Rank": columns})
    parts = rank_parts(order, n_parts)
    ranges = []
    for part in range(1, n_parts + 1):
        numbers = [int(f) for f, p in parts.items() if p == part]
        ranges.append(f"{part}: колонки {min(numbers)}–{max(numbers)}")
    destination = out_dir / f"parts{n_parts}" / "data_order.png"
    title = "Порядок колонок в массиве данных, одинаков для всех ионов\n" + "; ".join(ranges)
    save_parts_map(mean_train_matrix(X), parts_matrix(parts), title, destination, n_parts)
    print("plot", destination, flush=True)
    for line in ranges:
        print(line, flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, help="раскрасить весь рейтинг по N частям списка")
    parser.add_argument("--data-order", action="store_true", help="части по номеру колонки в данных, а не по рейтингу")
    parser.add_argument("--top", type=int, help="первые K каналов рейтинга вместо первой группы")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[folder for folder, _ in PARTS_METHODS],
        help="папки рейтингов в experiments/ для --top",
    )
    args = parser.parse_args()
    if args.top:
        plot_top(args.top, args.methods)
    elif args.parts and args.data_order:
        plot_data_order(args.parts)
    elif args.parts:
        plot_parts(args.parts)
    else:
        plot_available()


if __name__ == "__main__":
    main()
