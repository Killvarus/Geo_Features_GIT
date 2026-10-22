"""
RFE с усреднением по подвыборкам train, отдельно по каждому иону.

B раз RFE (Ridge, alpha по LOO-1SE, пакет 10%) на случайных 80% строк train,
итоговый ранг канала — средний ранг по запускам (ranking_tools.aggregate_rankings).
Validation и test не используются.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_6p_linear_selection_per_ion import N_TRAIN, TARGETS, load_ion, parse_alpha
from src.models.linear_selection import LinearRFE
from src.models.ranking_tools import aggregate_rankings

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ion", required=True, choices=TARGETS)
    parser.add_argument("--bags", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--alpha", type=parse_alpha, default="loo1se")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    tag = args.alpha if isinstance(args.alpha, str) else f"a{args.alpha:g}"
    out = ROOT / "experiments" / f"6p_rfe_ridge_{tag}_step{args.step:g}_bag{args.bags}" / args.ion / "ranking.csv"
    if out.exists():
        print(args.ion, "ranking exists", flush=True)
        return
    X, y = load_ion(args.ion)
    X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    rng = np.random.default_rng(0)
    rankings = []
    for bag in range(args.bags):
        rows = np.sort(rng.choice(N_TRAIN, size=int(args.fraction * N_TRAIN), replace=False))
        selector = LinearRFE(alpha=args.alpha, step=args.step).fit(X_train.iloc[rows], y_train.iloc[rows])
        rankings.append(selector.ranking_frame())
        print(args.ion, "bag", bag + 1, "seconds", round(selector.elapsed_seconds_, 1), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rankings(rankings).to_csv(out, index=False)
    print(args.ion, "done", out, flush=True)


if __name__ == "__main__":
    main()
