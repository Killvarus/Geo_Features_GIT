"""
ППОП и RFE на линейной регрессии (МНК или Ridge) отдельно по каждому иону.

ППОП: выбрасывается пакет признаков, без которых MSE выросла меньше всего.
MSE считается по --score-on: cv — K-fold внутри train, valid — на validation.
RFE: выбрасывается пакет с наименьшими |β| Ridge на train. --step 1 — классический RFE.

--step: целое — фиксированный пакет, дробь из (0, 1) — доля оставшихся признаков.
--alpha 0 — МНК без штрафа (только если признаков меньше строк).

Затем одновыходной персептрон на первых k признаках рейтинга, шаг 500, 3 seed.
Персептрон учится на train, останавливается по validation, R² — на test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.linear_selection import LinearPPOP, LinearRFE, check_step
from src.models.neural_network import to_excel_optimized_OLP
from src.models.ridge_rfe import plot_r2_curves

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments" / "6p_full_patience300" / "xy_cv1.npz"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600


def parse_step(text: str):
    value = float(text)
    step = int(value) if value >= 1 and value.is_integer() else value
    check_step(step)
    return step


def out_dir(args) -> Path:
    model = "ols" if args.alpha == 0 else f"ridge_a{args.alpha:g}"
    name = f"6p_{args.method}_{model}_step{args.step:g}"
    if args.method == "ppop":
        name += f"_{args.score_on}" + (str(args.cv) if args.score_on == "cv" else "")
    return ROOT / "experiments" / name


def title(args) -> str:
    model = "МНК" if args.alpha == 0 else f"Ridge alpha={args.alpha:g}"
    packet = "пакет 1" if args.step == 1 else (f"пакет {args.step:g}" if args.step > 1 else f"пакет {args.step:g} от остатка")
    if args.method == "rfe":
        return f"RFE, {model}, {packet}"
    where = f"{args.cv}-fold на train" if args.score_on == "cv" else "MSE на validation"
    return f"ППОП, {model}, {packet}, {where}"


def feature_names() -> list[str]:
    exc = [280 + 5 * i for i in range(27)]
    em = list(range(375, 576))
    return [f"E{e}_em{w}" for e in exc for w in em]


def load_ion(ion: str):
    blob = np.load(CACHE)
    X = pd.DataFrame(blob["X"], columns=feature_names())
    y = pd.Series(blob["y"][:, TARGETS.index(ion)], name=ion)
    return X, y


def k_grid(n_features: int) -> list[int]:
    ks = list(range(500, n_features, 500))
    if n_features not in ks:
        ks.append(n_features)
    return ks


def rank_ion(args, directory: Path, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    ion_dir = directory / args.ion
    path = ion_dir / "ranking.csv"
    if path.exists():
        print(args.ion, "ranking exists", flush=True)
        return pd.read_csv(path)
    ion_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = X.iloc[:N_TRAIN], y.iloc[:N_TRAIN]
    print(args.ion, title(args), "start", X_train.shape, flush=True)
    if args.method == "rfe":
        selector = LinearRFE(alpha=args.alpha, step=args.step).fit(X_train, y_train)
    elif args.score_on == "valid":
        X_valid, y_valid = X.iloc[N_TRAIN:N_TRAIN + N_VALID], y.iloc[N_TRAIN:N_TRAIN + N_VALID]
        selector = LinearPPOP(alpha=args.alpha, step=args.step).fit(X_train, y_train, X_valid, y_valid)
    else:
        selector = LinearPPOP(alpha=args.alpha, step=args.step, cv=args.cv).fit(X_train, y_train)
    selector.history_.to_csv(ion_dir / "history.csv", index=False)
    (ion_dir / "selection.json").write_text(
        json.dumps({"title": title(args), "seconds": selector.elapsed_seconds_}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selector.save_ranking(path)
    print(args.ion, "ranking done, seconds", round(selector.elapsed_seconds_, 1), flush=True)
    return selector.ranking_frame()


def train_curve(directory: Path, ion: str, X: pd.DataFrame, y: pd.Series, ranking: pd.DataFrame) -> None:
    ion_dir = directory / ion
    points_path = ion_dir / "curve_points.json"
    done = {}
    if points_path.exists():
        done = {point["k"]: point for point in json.loads(points_path.read_text(encoding="utf-8"))}
    ordered = ranking.sort_values("Individual_Rank")["Feature"].tolist()
    points = []
    y_df = y.to_frame()
    for k in k_grid(X.shape[1]):
        if k in done and len(done[k].get("runs", [])) == 3:
            points.append(done[k])
            print(ion, "skip k", k, flush=True)
            continue
        cols = ordered[:k]
        print(ion, "OLP k", k, flush=True)
        run_dir = ion_dir / f"k{k}"
        results = to_excel_optimized_OLP(
            file_name=str(run_dir / "metrics.xlsx"),
            n_iter=3,
            X_train=X.iloc[:N_TRAIN][cols],
            y_train=y_df.iloc[:N_TRAIN],
            X_valid=X.iloc[N_TRAIN:N_TRAIN + N_VALID][cols],
            y_valid=y_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
            X_test=X.iloc[N_TRAIN + N_VALID:][cols],
            y_test=y_df.iloc[N_TRAIN + N_VALID:],
            batch_size=64,
            input_dim=k,
            output_dim=1,
            learning_rate=0.001,
            num_epochs=10**9,
            patience=300,
            tolerance=0.003,
            tolerance_mode="relative",
            hidden_dim=32,
            save_plots_dir=str(run_dir / "learning_curves"),
            optimizer_type="adam",
            random_state=42,
            device="cpu",
            log_file=str(run_dir / "training.log"),
            enable_cv=False,
        )
        runs = [{"seed": 42 + i, "r2": float(result["test"].loc["Оценка1", "R2"])} for i, result in enumerate(results)]
        r2s = [item["r2"] for item in runs]
        point = {"k": k, "runs": runs, "mean": float(np.mean(r2s)), "std": float(np.std(r2s, ddof=1))}
        points.append(point)
        points_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
        print(ion, "k", k, "R2", round(point["mean"], 4), "±", round(point["std"], 4), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["ppop", "rfe"])
    parser.add_argument("--alpha", type=float, default=700.0)
    parser.add_argument("--step", type=parse_step, default=None, help="по умолчанию: ППОП 0.1, RFE 1")
    parser.add_argument("--score-on", choices=["cv", "valid"], default="cv")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--ion", choices=TARGETS)
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.alpha < 0:
        parser.error("alpha должен быть >= 0")
    if args.step is None:
        args.step = 0.1 if args.method == "ppop" else 1
    directory = out_dir(args)
    if args.plot_only:
        plot_r2_curves(directory, TARGETS, title(args))
        return
    if args.ion is None:
        parser.error("нужен --ion или --plot-only")
    X, y = load_ion(args.ion)
    ranking = rank_ion(args, directory, X, y)
    if args.rank_only:
        return
    train_curve(directory, args.ion, X, y, ranking)
    plot_r2_curves(directory, TARGETS, title(args))
    print(args.ion, "finished", flush=True)


if __name__ == "__main__":
    main()
