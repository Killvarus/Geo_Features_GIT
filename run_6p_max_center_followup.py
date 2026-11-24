"""Дополнительные эксперименты отбора каналов на спектрах после max."""

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
BSBS_SOURCES = (
    EXPERIMENTS / "6p_selnorm_grid" / "max__center" / "bsbs",
    EXPERIMENTS / "6p_norm_grid" / "max__center" / "ppop",
)
BSBS_KS = (10, 30, 50, 100, 250, 300, 500, 1000, 1500, 2000, 3000, 4000)
RFE_KS = tuple(k for k in BSBS_KS if k not in (10, 30, 100, 300))


def ranking_for(method, ion, X, y):
    ion_dir = OUT / method / ion
    ion_dir.mkdir(parents=True, exist_ok=True)
    path = ion_dir / "ranking.csv"
    sources = {"rfe": (RFE_SOURCE,), "bsbs_max_only": BSBS_SOURCES, "bsbs_step1": ()}[method]
    candidates = [path, *(folder / ion / "ranking.csv" for folder in sources)]
    existing = next((candidate for candidate in candidates if candidate.is_file()), None)
    step = 1 if method == "bsbs_step1" else STEP
    if existing is not None:
        ranking = pd.read_csv(existing)
        source = str(existing)
    else:
        if method == "rfe":
            selector = LinearRFE(alpha=ALPHA, step=STEP, scaling="center")
        else:
            selector = LinearPPOP(alpha=ALPHA, step=step, cv=CV, scaling="center")
        selector.fit(X.iloc[:N_TRAIN], y.iloc[:N_TRAIN])
        ranking = selector.ranking_frame()
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        (ion_dir / "selection.json").write_text(
            json.dumps({
                "method": method, "spectrum_norm": "max", "channel_norm": "center",
                "alpha": ALPHA, "step": step,
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
    parser.add_argument("method", choices=("rfe", "bsbs_step1", "bsbs_max_only"))
    parser.add_argument("ion", choices=TARGETS)
    args = parser.parse_args()
    X, y = load("max", args.ion)
    ranking = ranking_for(args.method, args.ion, X, y)
    ks = RFE_KS if args.method == "rfe" else BSBS_KS
    train_curve(OUT / args.method, args.ion, X, y, ranking, list(ks),
                scale_inputs=args.method != "bsbs_max_only")


if __name__ == "__main__":
    main()
