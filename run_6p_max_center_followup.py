"""Дообучение кривых RFE и эксперимент BSBS с исключением по одному каналу."""

import argparse
import json
from pathlib import Path

from run_6p_linear_selection_per_ion import N_TRAIN, TARGETS, train_curve
from run_6p_norm_grid import ALPHA, CV, STEP, load
from src.models.linear_selection import LinearPPOP, LinearRFE

import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent / "experiments"
OUT = EXPERIMENTS / "6p_max_center_followup"
RFE_SOURCE = EXPERIMENTS / "6p_norm_grid" / "max__center" / "rfe"
BSBS_KS = (10, 30, 50, 100, 250, 300, 500, 1000, 1500, 2000, 3000, 4000)
RFE_KS = tuple(k for k in BSBS_KS if k not in (10, 30, 100, 300))


def ranking_for(method, ion, X, y):
    ion_dir = OUT / method / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    path = ion_dir / "ranking.csv"
    if path.exists():
        ranking = pd.read_csv(path)
        source = "сохранённый рейтинг этого запуска"
    elif method == "rfe" and (RFE_SOURCE / ion / "ranking.csv").is_file():
        ranking = pd.read_csv(RFE_SOURCE / ion / "ranking.csv")
        source = str(RFE_SOURCE / ion / "ranking.csv")
    else:
        if method == "rfe":
            selector = LinearRFE(alpha=ALPHA, step=STEP, scaling="center")
        else:
            selector = LinearPPOP(alpha=ALPHA, step=1, cv=CV, scaling="center")
        selector.fit(X.iloc[:N_TRAIN], y.iloc[:N_TRAIN])
        ranking = selector.ranking_frame()
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        (ion_dir / "selection.json").write_text(
            json.dumps({
                "method": method, "spectrum_norm": "max", "channel_norm": "center",
                "alpha": ALPHA, "step": STEP if method == "rfe" else 1,
                "cv": None if method == "rfe" else CV, "seconds": selector.elapsed_seconds_,
            }, indent=2),
            encoding="utf-8",
        )
        source = "отбор в этом запуске"

    if (len(ranking) != X.shape[1] or set(ranking["Feature"]) != set(X.columns)
            or set(ranking["Individual_Rank"]) != set(range(1, X.shape[1] + 1))):
        raise ValueError(f"{ion}: неполный рейтинг {method} ({source})")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        ranking.to_csv(temporary, index=False)
        temporary.replace(path)
    print(f"{ion}: рейтинг {method}, {source}", flush=True)
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("rfe", "bsbs_step1"))
    parser.add_argument("ion", choices=TARGETS)
    args = parser.parse_args()
    X, y = load("max", args.ion)
    ranking = ranking_for(args.method, args.ion, X, y)
    ks = RFE_KS if args.method == "rfe" else BSBS_KS
    train_curve(OUT / args.method, args.ion, X, y, ranking, list(ks))


if __name__ == "__main__":
    main()
