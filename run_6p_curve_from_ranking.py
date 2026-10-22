"""
Персептрон на первых k каналах готового рейтинга 6P.

Рейтинг берётся из experiments/<папка>/<ион>/ranking.csv (любой: отбор или базовый),
результат пишется в experiments/<out>/<папка>/<ион>/curve_points.json.
Параметры персептрона те же, что в run_6p_linear_selection_per_ion.py.

Пример:
  python run_6p_curve_from_ranking.py --ranking-dir 6p_baseline_uniform --ion Cu --ks 10 30 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import TARGETS, load_ion, train_curve

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-dir", required=True, help="папка в experiments/ с <ион>/ranking.csv")
    parser.add_argument("--ion", required=True, choices=TARGETS)
    parser.add_argument("--ks", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="6p_small_k", help="папка результатов в experiments/")
    args = parser.parse_args()
    ranking = pd.read_csv(ROOT / "experiments" / args.ranking_dir / args.ion / "ranking.csv")
    X, y = load_ion(args.ion)
    directory = ROOT / "experiments" / args.out / args.ranking_dir
    train_curve(directory, args.ion, X, y, ranking, args.ks)
    print(args.ion, args.ranking_dir, "finished", flush=True)


if __name__ == "__main__":
    main()
