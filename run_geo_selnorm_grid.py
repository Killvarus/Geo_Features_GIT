"""
Сетка нормировок и ускоренный BSBS для обратной задачи МТЗ (геофизика) — то же, что
run_6p_selnorm_grid.py для спектров. Всё пишется в experiments/geo_selnorm/.

Данные (три независимых набора, пересечений между train/valid/test нет):
  d1 — mtsgrvmgn_trn/vld/tst: 21000/6000/3000, постановка статьи PRIA (Ларионов и др., 2026);
  d2 — train_3/valid_3/test_3: 21000/3000/6000 (из ALL_3; valid меньше test — проверить);
  d3 — train_3_1/valid_3_1/test_3_1: 7000/1000/2000 (из ALL_3_1, латерально неоднородная среда).
Входы — 2418 признаков: RE/IM × (Z_YX, Z_XY, вектор Визе HX) × 13 частот × 31 пикет.
Цели — глубины трёх границ на центральном пикете: H1_8, H2_8, H3_8 (автономно).

Сравниваются нормировки по строкам — каждого примера отдельно (src/preprocessing/geo_norm.py),
аналог нормировок спектров в 6P. Каналы внутри отбора всегда в z-score.

Протокол (input):
  raw  — нормировка строк действует только на отбор; персептрон, Ridge и МНК учатся на
         исходных данных с z-score каналов. Разница между вариантами — только в выбранных каналах;
  same — проверка на тех же нормированных данных (как прежняя сетка 6p_norm_grid).
Отбор: LinearPPOP — ускоренный BSBS, Ridge с alpha из experiments/geo_selnorm/<набор>/alpha.json
(задание alpha), пакет 10% от оставшихся, 5-fold на train, solver="auto".
Персептрон: 1 скрытый слой из 32 нейронов, Adam lr 0.001, батч 128, ранний останов после
250 эпох без улучшения на validation (как в статье PRIA), 3 инициализации.

Порядок работы:
  1. cache <набор> — CSV → experiments/geo_cache/<набор>.npz (один раз, нужны CSV в Data/);
  2. alpha <набор> — подбор alpha Ridge по validation на всех каналах (z-score);
  3. задания сетки: full и rank.

Задания:
  cache <набор>
  alpha <набор>
  full  <набор> <input> <нормировка> <цель>
  rank  <набор> <input> <нормировка> <каналы> <цель>
Наборы заданий (--list-jobs --set …): prepare (cache и alpha), d1, d2, d3 (input raw:
3 «все каналы» + 5 нормировок строк × 3 цели = 18), d1_same (input same, 21).
В задании rank можно указать и другую нормировку каналов (robust, pareto, center), но в
наборы заданий она не входит: при почти нулевом alpha она на отбор почти не влияет.

Линейная проверка (Ridge и МНК) считается для всех --ks, персептрон — только для
--perceptron-ks (по умолчанию 10, 25, 50, 100, 250, 750). --no-perceptron — только отбор и
линейная проверка: так вся сетка одного набора идёт около 10 минут на 8 ядрах.

Структура: <набор>/<input>/<нормировка>__<каналы>/bsbs/<цель>/ (ranking.csv, history.csv,
selection.json, linear_check.json, curve_points.json, k<k>/), «все каналы» — в
<нормировка>__all/<цель>/. summary.csv — после --summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.models.linear_selection import SCALINGS, LinearPPOP, scale_channels
from src.preprocessing.geo_norm import GEO_SAMPLE_NORMS, GEO_TRANSFORMS, amp_phase, is_feature, normalize_samples

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
CACHE_DIR = ROOT / "experiments" / "geo_cache"
OUT_DIR = ROOT / "experiments" / "geo_selnorm"
DATASETS = {
    "d1": ("mtsgrvmgn_trn.csv", "mtsgrvmgn_vld.csv", "mtsgrvmgn_tst.csv"),
    "d2": ("train_3.csv", "valid_3.csv", "test_3.csv"),
    "d3": ("train_3_1.csv", "valid_3_1.csv", "test_3_1.csv"),
}
TARGETS = ["H1_8", "H2_8", "H3_8"]
NORMS = GEO_SAMPLE_NORMS + GEO_TRANSFORMS
INPUTS = ("raw", "same")
STEP = 0.1
CV = 5
ALPHAS = np.logspace(-6, 5, 45)
# На данных без шума XᵀX вырождена (эффективный ранг d1 около 1700 из 2418): при alpha = 1e-6
# обусловленность ~1e13 и быстрая формула расходится с переобучением на 1e-5, при 1e-3 — на 1e-8.
ALPHA_FLOOR = 1e-3
KS = [10, 25, 50, 100, 250, 500, 750, 1000, 1500]
# Персептрон дорог (0.5–2.4 с на эпоху на 21000 примерах), поэтому по умолчанию — меньше точек.
PERCEPTRON_KS = [10, 25, 50, 100, 250, 750]
MLP = {"hidden_dim": 32, "learning_rate": 0.001, "batch_size": 128, "patience": 250, "tolerance": 0.003,
       "n_iter": 3, "num_epochs": 20000}


# ----------------------------------------------------------------------------- данные

def build_cache(dataset: str, data_dir: Path = DATA_DIR, cache_dir: Path = CACHE_DIR) -> Path:
    """CSV набора → npz с X_train/X_valid/X_test, y_* (H1_8, H2_8, H3_8) и именами признаков."""
    frames = [pd.read_csv(data_dir / name) for name in DATASETS[dataset]]
    names = [c for c in frames[0].columns if is_feature(c)]
    if len(names) != 2418:
        print(f"предупреждение: признаков {len(names)}, ожидалось 2418", flush=True)
    for frame in frames[1:]:
        if [c for c in frame.columns if is_feature(c)] != names:
            raise ValueError("признаки train/valid/test не совпадают или идут в другом порядке")
    blob = {"names": np.array(names)}
    for split, frame in zip(("train", "valid", "test"), frames):
        blob[f"X_{split}"] = frame[names].to_numpy(dtype=float)
        blob[f"y_{split}"] = frame[TARGETS].to_numpy(dtype=float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}.npz"
    np.savez(path, **blob)
    print(dataset, {k: v.shape for k, v in blob.items()}, "→", path, flush=True)
    return path


def load(dataset: str, cache_dir: Path | None = None) -> dict:
    path = (cache_dir or CACHE_DIR) / f"{dataset}.npz"
    if not path.exists():
        raise SystemExit(f"нет кэша {path}: сначала задание cache {dataset}")
    blob = np.load(path)
    data = {name: blob[name] for name in blob.files}
    data["names"] = [str(n) for n in data["names"]]
    return data


def prepared(data: dict, norm: str) -> tuple[dict, list[str]]:
    """Нормировка примеров или преобразование для всех трёх частей."""
    names = data["names"]
    out = {}
    for split in ("train", "valid", "test"):
        if norm in GEO_TRANSFORMS:
            out[split], new_names = amp_phase(data[f"X_{split}"], names)
        else:
            out[split], new_names = normalize_samples(data[f"X_{split}"], names, norm), names
    return out, list(new_names)


def read_alpha(dataset: str, out_dir: Path = OUT_DIR) -> float:
    """alpha для отбора и линейной проверки: найденный по validation, но не меньше ALPHA_FLOOR."""
    path = out_dir / dataset / "alpha.json"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала задание alpha {dataset}")
    return max(float(json.loads(path.read_text(encoding="utf-8"))["alpha"]), ALPHA_FLOOR)


# ----------------------------------------------------------------------------- alpha

def ridge_validation_mse(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
                         alphas: np.ndarray = ALPHAS) -> np.ndarray:
    """MSE Ridge на validation для каждого alpha и каждой цели (z-score по train, свободный член)."""
    scaled_train, scaled_valid = scale_channels(X_train, X_valid)
    centered = y_train - y_train.mean(axis=0)
    u, s, vt = np.linalg.svd(scaled_train, full_matrices=False)
    projected = u.T @ centered
    errors = np.empty((len(alphas), y_train.shape[1]))
    for a, alpha in enumerate(alphas):
        beta = vt.T @ ((s / (s**2 + alpha))[:, None] * projected)
        prediction = scaled_valid @ beta + y_train.mean(axis=0)
        errors[a] = ((y_valid - prediction) ** 2).mean(axis=0)
    return errors


def run_alpha(dataset: str, out_dir: Path = OUT_DIR) -> None:
    data = load(dataset)
    started = time.perf_counter()
    errors = ridge_validation_mse(data["X_train"], data["y_train"], data["X_valid"], data["y_valid"])
    relative = errors / data["y_valid"].var(axis=0)
    best = int(np.argmin(relative.mean(axis=1)))
    if best == 0:
        print("предупреждение: оптимум на нижней границе сетки alpha — данные почти без шума, Ridge близок к МНК", flush=True)
    result = {
        "alpha": float(ALPHAS[best]),
        "at_lower_bound": best == 0,
        "alpha_used": max(float(ALPHAS[best]), ALPHA_FLOOR),
        "rule": "минимум средней по трём целям MSE/дисперсии на validation, Ridge на всех каналах, z-score",
        "per_target_best": {t: float(ALPHAS[int(np.argmin(errors[:, i]))]) for i, t in enumerate(TARGETS)},
        "valid_r2_at_alpha": {t: float(1 - relative[best, i]) for i, t in enumerate(TARGETS)},
        "grid": {f"{a:g}": [float(1 - r) for r in row] for a, row in zip(ALPHAS, relative)},
        "seconds": time.perf_counter() - started,
    }
    path = out_dir / dataset / "alpha.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(dataset, "alpha", result["alpha"], "per target", result["per_target_best"], flush=True)


# ----------------------------------------------------------------------------- проверка

def linear_check(X: dict, y: dict, columns: list[int], alpha: float) -> dict:
    result = {}
    for name, model in (("ridge", Ridge(alpha=alpha)), ("ols", LinearRegression())):
        pipeline = make_pipeline(StandardScaler(), model).fit(X["train"][:, columns], y["train"])
        result[name] = {split: float(r2_score(y[split], pipeline.predict(X[split][:, columns]))) for split in ("valid", "test")}
    return result


def train_points(ion_dir: Path, X: dict, y: dict, columns_by_k: dict[int, list[int]], mlp: dict) -> None:
    """Персептрон на первых k признаках: curve_points.json и k<k>/predictions.npz, как в train_curve."""
    from src.models.neural_network import to_excel_optimized_OLP

    points_path = ion_dir / "curve_points.json"
    done = {p["k"]: p for p in json.loads(points_path.read_text(encoding="utf-8"))} if points_path.exists() else {}
    for k, columns in columns_by_k.items():
        if k in done and len(done[k].get("runs", [])) == mlp["n_iter"]:
            print(ion_dir.name, "skip k", k, flush=True)
            continue
        run_dir = ion_dir / f"k{k}"
        frames = {split: pd.DataFrame(X[split][:, columns]) for split in ("train", "valid", "test")}
        targets = {split: pd.DataFrame(y[split]) for split in ("train", "valid", "test")}
        results = to_excel_optimized_OLP(
            file_name=str(run_dir / "metrics.xlsx"), n_iter=mlp["n_iter"],
            X_train=frames["train"], y_train=targets["train"], X_valid=frames["valid"], y_valid=targets["valid"],
            X_test=frames["test"], y_test=targets["test"], batch_size=mlp["batch_size"], input_dim=len(columns),
            output_dim=1, learning_rate=mlp["learning_rate"], num_epochs=mlp["num_epochs"], patience=mlp["patience"],
            tolerance=mlp["tolerance"], tolerance_mode="relative", hidden_dim=mlp["hidden_dim"],
            save_plots_dir=str(run_dir / "learning_curves"), optimizer_type="adam", random_state=42, device="cpu",
            log_file=str(run_dir / "training.log"), enable_cv=False,
        )
        variance_ratio = float(np.var(y["train"]) / np.var(y["valid"]))
        runs = [{"seed": 42 + i, "r2": float(r["test"].loc["Оценка1", "R2"]),
                 "mae": float(r["test"].loc["Оценка1", "MAE"]),
                 "r2_valid": 1.0 - float(r["history"]["best_val_loss"]) * variance_ratio,
                 "epochs": int(r["history"]["stopped_epoch"])} for i, r in enumerate(results)]
        np.savez_compressed(run_dir / "predictions.npz",
                            test=np.stack([np.ravel(r["history"]["test_pred"]) for r in results]),
                            valid=np.stack([np.ravel(r["history"]["valid_pred"]) for r in results]))
        r2s = [run["r2"] for run in runs]
        done[k] = {"k": k, "runs": runs, "mean": float(np.mean(r2s)), "std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                   "mean_valid": float(np.mean([run["r2_valid"] for run in runs])),
                   "mean_mae": float(np.mean([run["mae"] for run in runs]))}
        points_path.write_text(json.dumps(sorted(done.values(), key=lambda p: p["k"]), indent=2), encoding="utf-8")
        print(ion_dir.name, "k", k, "R2", round(done[k]["mean"], 4), flush=True)


def evaluation_data(data: dict, norm: str, input_mode: str) -> tuple[dict, list[str]]:
    if input_mode == "raw":
        return {s: data[f"X_{s}"] for s in ("train", "valid", "test")}, data["names"]
    return prepared(data, norm)


# ----------------------------------------------------------------------------- задания

def run_rank(dataset: str, input_mode: str, norm: str, channel: str, target: str, ks: list[int], perceptron: bool,
             mlp: dict, out_dir: Path = OUT_DIR, perceptron_ks: list[int] | None = None) -> None:
    if input_mode == "raw" and norm in GEO_TRANSFORMS:
        raise SystemExit("amp_phase меняет признаки: проверять его можно только с input same")
    ion_dir = out_dir / dataset / input_mode / f"{norm}__{channel}" / "bsbs" / target
    ion_dir.mkdir(parents=True, exist_ok=True)
    data = load(dataset)
    alpha = read_alpha(dataset, out_dir)
    t = TARGETS.index(target)
    y = {s: data[f"y_{s}"][:, t] for s in ("train", "valid", "test")}
    selection, selection_names = prepared(data, norm)
    path = ion_dir / "ranking.csv"
    if path.exists():
        ranking = pd.read_csv(path)
        print(dataset, input_mode, norm, channel, target, "ranking exists", flush=True)
    else:
        selector = LinearPPOP(alpha=alpha, step=STEP, cv=CV, scaling=channel, solver="auto")
        selector.fit(pd.DataFrame(selection["train"], columns=selection_names), y["train"])
        selector.history_.to_csv(ion_dir / "history.csv", index=False)
        settings = {"dataset": dataset, "input": input_mode, "sample_norm": norm, "channel_norm": channel, "target": target,
                    "alpha": alpha, "step": STEP, "cv": CV, "seconds": selector.elapsed_seconds_}
        (ion_dir / "selection.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        selector.save_ranking(path)
        ranking = selector.ranking_frame()
        print(dataset, input_mode, norm, channel, target, "ranking done, seconds", round(selector.elapsed_seconds_, 1), flush=True)
    X, names = evaluation_data(data, norm, input_mode)
    position = {name: i for i, name in enumerate(names)}
    ordered = [position[f] for f in ranking.sort_values("Individual_Rank")["Feature"].astype(str)]
    ks = [k for k in ks if k < len(names)]
    check_path = ion_dir / "linear_check.json"
    checks = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
    checks.update({str(k): linear_check(X, y, ordered[:k], alpha) for k in ks})
    check_path.write_text(json.dumps(dict(sorted(checks.items(), key=lambda item: int(item[0]))), indent=2), encoding="utf-8")
    if perceptron:
        chosen = [k for k in (perceptron_ks or PERCEPTRON_KS) if k < len(names)]
        train_points(ion_dir, X, y, {k: ordered[:k] for k in chosen}, mlp)


def run_full(dataset: str, input_mode: str, norm: str, target: str, perceptron: bool, mlp: dict, out_dir: Path = OUT_DIR) -> None:
    if input_mode == "raw" and norm != "raw":
        raise SystemExit("при input raw все каналы проверяются только без нормировки (norm raw)")
    ion_dir = out_dir / dataset / input_mode / f"{norm}__all" / target
    ion_dir.mkdir(parents=True, exist_ok=True)
    data = load(dataset)
    alpha = read_alpha(dataset, out_dir)
    t = TARGETS.index(target)
    y = {s: data[f"y_{s}"][:, t] for s in ("train", "valid", "test")}
    X, names = evaluation_data(data, norm, input_mode)
    columns = list(range(len(names)))
    (ion_dir / "linear_check.json").write_text(json.dumps({str(len(columns)): linear_check(X, y, columns, alpha)}, indent=2), encoding="utf-8")
    if perceptron:
        train_points(ion_dir, X, y, {len(columns): columns}, mlp)


def jobs(which: str) -> list[str]:
    if which == "prepare":
        return [f"cache {d}" for d in DATASETS] + [f"alpha {d}" for d in DATASETS]
    if which in DATASETS:
        lines = [f"full {which} raw raw {t}" for t in TARGETS]
        lines += [f"rank {which} raw {n} zscore {t}" for n in GEO_SAMPLE_NORMS for t in TARGETS]
        return lines
    if which == "d1_same":
        lines = [f"full d1 same {n} {t}" for n in NORMS if n != "raw" for t in TARGETS]
        lines += [f"rank d1 same {n} zscore {t}" for n in NORMS if n != "raw" for t in TARGETS]
        return lines
    raise ValueError(which)


def describe(relative: tuple[str, ...]) -> dict:
    dataset, input_mode, combo = relative[:3]
    norm, channel = combo.split("__")
    if channel == "all":
        return {"dataset": dataset, "input": input_mode, "sample_norm": norm, "channel_norm": "all", "method": "all_channels", "target": relative[3]}
    return {"dataset": dataset, "input": input_mode, "sample_norm": norm, "channel_norm": channel, "method": relative[3], "target": relative[4]}


def summary(out_dir: Path = OUT_DIR) -> Path:
    rows = []
    for check in out_dir.glob("*/*/*__*/**/linear_check.json"):
        base = describe(check.parent.relative_to(out_dir).parts)
        for k, models in json.loads(check.read_text(encoding="utf-8")).items():
            for model, scores in models.items():
                rows.append({**base, "k": int(k), "model": model, "r2_valid": scores["valid"], "r2_test": scores["test"]})
        curve = check.parent / "curve_points.json"
        if curve.exists():
            for point in json.loads(curve.read_text(encoding="utf-8")):
                rows.append({**base, "k": point["k"], "model": "perceptron", "r2_valid": point.get("mean_valid"),
                             "r2_test": point["mean"], "r2_test_std": point.get("std"), "mae_test": point.get("mean_mae"),
                             "runs": len(point.get("runs", []))})
    table = pd.DataFrame(rows)
    out = out_dir / "summary.csv"
    if not table.empty:
        table = table.sort_values(["dataset", "input", "sample_norm", "channel_norm", "target", "model", "k"])
    table.to_csv(out, index=False)
    print("summary", out, "rows", len(table), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--set", default="d1", choices=["prepare", "d1", "d2", "d3", "d1_same"])
    parser.add_argument("--job", nargs="+")
    parser.add_argument("--ks", type=int, nargs="+", default=KS, help="k для линейной проверки")
    parser.add_argument("--perceptron-ks", type=int, nargs="+", default=PERCEPTRON_KS, help="k для персептрона")
    parser.add_argument("--no-perceptron", action="store_true")
    parser.add_argument("--max-epochs", type=int, help="ограничить эпохи (для пробного запуска)")
    parser.add_argument("--seeds", type=int, help="число инициализаций (для пробного запуска)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="папка результатов (alpha.json берётся отсюда же)")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.list_jobs:
        print("\n".join(jobs(args.set)))
        return
    if args.summary:
        summary(args.out)
        return
    if not args.job:
        parser.error("нужен --list-jobs, --job или --summary")
    mlp = dict(MLP)
    if args.max_epochs:
        mlp["num_epochs"] = args.max_epochs
    if args.seeds:
        mlp["n_iter"] = args.seeds
    kind, *rest = args.job
    perceptron = not args.no_perceptron
    bad = f"неизвестное задание: {' '.join(args.job)}"
    if kind == "cache" and len(rest) == 1 and rest[0] in DATASETS:
        build_cache(rest[0])
    elif kind == "alpha" and len(rest) == 1 and rest[0] in DATASETS:
        run_alpha(rest[0], args.out)
    elif kind == "full" and len(rest) == 4:
        dataset, input_mode, norm, target = rest
        if dataset not in DATASETS or input_mode not in INPUTS or norm not in NORMS or target not in TARGETS:
            parser.error(bad)
        run_full(dataset, input_mode, norm, target, perceptron, mlp, args.out)
    elif kind == "rank" and len(rest) == 5:
        dataset, input_mode, norm, channel, target = rest
        if dataset not in DATASETS or input_mode not in INPUTS or norm not in NORMS or channel not in SCALINGS or target not in TARGETS:
            parser.error(bad)
        run_rank(dataset, input_mode, norm, channel, target, sorted(set(args.ks)), perceptron, mlp, args.out,
                 sorted(set(args.perceptron_ks)))
    else:
        parser.error(bad)
    print("job done:", " ".join(args.job), flush=True)


if __name__ == "__main__":
    main()
