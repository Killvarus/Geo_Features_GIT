"""Серверная сетка нормировок 6P с одним стартом персептрона."""

import argparse
import csv
import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from run_6p_linear_selection_per_ion import N_TRAIN, TARGETS, feature_names, train_curve
from src.models.linear_selection import LinearPPOP
from src.preprocessing.spectrum_norm import SPECTRUM_NORMS, fluorescence_mask, normalize_spectra


HERE = Path(__file__).resolve().parent
CV_ORDERS = HERE / "src" / "preprocessing" / "6p_cv_orders.json"
KS = (10, 100, 500, 1500, 3000, 4000)
FULL = 5427
RANKS = tuple(itertools.product((*SPECTRUM_NORMS, "minmax"), ("center", "zscore", "robust", "range")))
INPUTS = tuple(itertools.product(("raw", "max", "area", "minmax"), ("raw", "center", "zscore", "minmax")))
SCALE_INPUT = {"raw": False, "center": "center", "zscore": True, "minmax": "minmax"}
BASE_RANK = ("max", "zscore")
BASE_INPUT = ("max", "zscore")
RAW_INPUT = ("raw", "zscore")
ALPHA, STEP, CV = 700.0, 0.1, 5


def tag(config):
    return "_".join(config) if config else "all_channels"


def result_root(project):
    return project / "experiments" / "6p_normalization_followup"


def cache_file(project, split):
    return project / "experiments" / "6p_full_patience300" / f"xy_{split}.npz"


def ensure_caches(project, splits):
    source = cache_file(project, "cv1")
    if not source.is_file():
        raise FileNotFoundError(source)
    missing = [split for split in splits if not cache_file(project, split).is_file()]
    if not missing:
        return
    manifest = json.loads(CV_ORDERS.read_text(encoding="utf-8"))
    with np.load(source) as data:
        x, y = data["X"], data["y"]
    fingerprint = hashlib.sha256(np.asarray(y, dtype="<f8").tobytes()).hexdigest()
    if fingerprint != manifest["source_targets_sha256"] or x.shape != (len(y), FULL):
        raise ValueError(f"Порядок образцов или размер в {source} отличается от исходного CV1")
    for split in missing:
        order = np.array(manifest["order"][split], dtype=int)
        if not np.array_equal(np.sort(order), np.arange(len(y))):
            raise ValueError(f"Некорректная перестановка для {split}")
        path = cache_file(project, split)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, X=x[order], y=y[order])
        temporary.replace(path)
        print(f"Кэш {split} собран из CV1: {path}", flush=True)


def rank_dir(project, split, config, ion):
    return result_root(project) / split / "rankings" / tag(config) / ion


def model_dir(project, split, rank_config, inputs):
    return result_root(project) / split / "perceptron" / tag(rank_config) / tag(inputs)


def load(project, split, ion, spectrum):
    with np.load(cache_file(project, split)) as data:
        values = data["X"]
        y = data["y"][:, TARGETS.index(ion)]
    if spectrum == "minmax":
        region = values[:, fluorescence_mask()]
        minimum = region.min(axis=1, keepdims=True)
        span = region.max(axis=1, keepdims=True) - minimum
        if np.any(span <= 0):
            raise ValueError(f"{split}: есть спектр с нулевым диапазоном")
        values = (values - minimum) / span
    elif spectrum != "raw":
        values = normalize_spectra(values, spectrum, N_TRAIN)
    return pd.DataFrame(values, columns=feature_names()), pd.Series(y, name=ion)


def checked_ranking(path):
    ranking = pd.read_csv(path)
    if (len(ranking) != FULL or set(ranking["Feature"]) != set(feature_names())
            or set(ranking["Individual_Rank"]) != set(range(1, FULL + 1))):
        raise ValueError(f"Повреждённый рейтинг {path}")
    return ranking


def previous_ranking(project, config, ion):
    spectrum, scaling = config
    paths = (
        HERE / "reference_rankings" / tag(config) / ion,
        project / "experiments" / "6p_selnorm_grid" / f"{spectrum}__{scaling}" / "bsbs" / ion,
    )
    for path in paths:
        if not (path / "selection.json").is_file() or not (path / "ranking.csv").is_file():
            continue
        meta = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        expected = {"spectrum_norm_for_selection": spectrum, "channel_norm_for_selection": scaling,
                    "method": "bsbs", "alpha": ALPHA, "step": STEP, "cv": CV}
        if all(meta.get(key) == value for key, value in expected.items()):
            return checked_ranking(path / "ranking.csv"), meta.get("seconds"), path
    return None


def rank_job(project, split, config, ion):
    directory = rank_dir(project, split, config, ion)
    path = directory / "ranking.csv"
    if path.is_file() and (directory / "selection.json").is_file():
        checked_ranking(path)
        return
    directory.mkdir(parents=True, exist_ok=True)
    reference = previous_ranking(project, config, ion) if split == "cv1" else None
    spectrum, scaling = config
    if reference is not None:
        ranking, seconds, source = reference
    else:
        X, y = load(project, split, ion, spectrum)
        model = LinearPPOP(alpha=ALPHA, step=STEP, cv=CV, scaling=scaling)
        model.fit(X.iloc[:N_TRAIN], y.iloc[:N_TRAIN])
        ranking, seconds, source = model.ranking_frame(), model.elapsed_seconds_, None
        model.history_.to_csv(directory / "history.csv", index=False)
    (directory / "selection.json").write_text(json.dumps({
        "split": split, "spectrum": spectrum, "channels": scaling, "alpha": ALPHA, "step": STEP, "cv": CV,
        "ranking_seconds": seconds, "source": str(source) if source else "measured",
    }, indent=2), encoding="utf-8")
    temporary = path.with_suffix(".tmp")
    ranking.to_csv(temporary, index=False)
    temporary.replace(path)


def train_job(project, split, config, inputs, ion, ks):
    directory = model_dir(project, split, config, inputs)
    X, y = load(project, split, ion, inputs[0])
    if config:
        ranking = checked_ranking(rank_dir(project, split, config, ion) / "ranking.csv")
    else:
        ranking = pd.DataFrame({"Feature": X.columns, "Individual_Rank": range(1, FULL + 1)})
    train_curve(directory, ion, X, y, ranking, list(ks), scale_inputs=SCALE_INPUT[inputs[1]], n_iter=1)


def run_jobs(pool, func, jobs, label):
    jobs = list(dict.fromkeys(jobs))
    print(f"{label}: {len(jobs)} заданий", flush=True)
    futures = {pool.submit(func, *job): job for job in jobs}
    for i, future in enumerate(as_completed(futures), start=1):
        try:
            future.result()
        except Exception as error:
            for pending in futures:
                pending.cancel()
            raise RuntimeError(f"{label}: {futures[future][1:]}") from error
        print(f"{label}: {i}/{len(jobs)} готово · {futures[future][1:]}", flush=True)


def validated_choice(project, candidates):
    def errors(config, inputs, ion):
        path = model_dir(project, "cv1", config, inputs) / ion / "curve_points.json"
        points = json.loads(path.read_text(encoding="utf-8"))
        return {point["k"]: point["runs"][0]["mae_valid"] for point in points}

    baseline = {(ion, k): value for ion in TARGETS for k, value in errors(*candidates[0], ion).items()}
    return sorted(candidates, key=lambda choice: np.mean([
        value / max(baseline[ion, k], 1e-12)
        for ion in TARGETS for k, value in errors(*choice, ion).items() if k in KS
    ]))


def export_summary(project):
    root = result_root(project)
    fields = ("split", "selection", "inputs", "ion", "k", "seed", "r2", "r2_valid",
              "mae_test", "mae_valid", "epochs", "ranking_seconds", "ranking_source", "training_seconds")
    destination = root / "metrics.csv"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted(root.glob("cv*/perceptron/*/*/*/curve_points.json")):
            split, _, selection, inputs, ion, _ = path.relative_to(root).parts
            meta_path = root / split / "rankings" / selection / ion / "selection.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            for point in json.loads(path.read_text(encoding="utf-8")):
                for run in point["runs"]:
                    writer.writerow({"split": split, "selection": selection, "inputs": inputs, "ion": ion,
                                     "k": point["k"], "ranking_seconds": meta.get("ranking_seconds"),
                                     "ranking_source": meta.get("source"), **run})
    temporary.replace(destination)


def run(project, workers, screen_only):
    splits = ("cv1",) if screen_only else ("cv1", "cv2", "cv3")
    ensure_caches(project, splits)
    root = result_root(project)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps({
        "splits": splits, "channels": (*KS, FULL), "perceptron_seeds": [42],
        "alpha": ALPHA, "step": STEP, "cv": CV, "bsbs_channel_alias": {"raw": "center"},
        "minmax_spectrum_statistics": "outside Rayleigh bands",
        "finalists_metric": "validation MAE normalized to the baseline per ion and k",
    }, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        initial = [(project, "cv1", config, ion) for config in (BASE_RANK, ("max", "center")) for ion in TARGETS]
        run_jobs(pool, rank_job, initial, "CV1 · опорные рейтинги")
        # Сначала закрываем вопрос max / center / z-score / min-max на одинаковых каналах.
        first = [(project, "cv1", BASE_RANK, inputs, ion, KS) for inputs in INPUTS
                 if inputs[0] == "max" for ion in TARGETS]
        first += [(project, "cv1", ("max", "center"), inputs, ion, (4000,)) for inputs in INPUTS
                  if inputs[0] == "max" for ion in ("Cu", "Ni")]
        run_jobs(pool, train_job, first, "CV1 · масштабирование max и проверка на 4000")
        export_summary(project)
        run_jobs(pool, rank_job, ((project, "cv1", config, ion) for config in RANKS for ion in TARGETS
                                  if (project, "cv1", config, ion) not in initial), "CV1 · рейтинги для сетки")
        screening = [(project, "cv1", config, RAW_INPUT, ion, KS) for config in RANKS for ion in TARGETS]
        screening += [(project, "cv1", BASE_RANK, inputs, ion, KS) for inputs in INPUTS
                      if inputs[0] != "max" and inputs != RAW_INPUT for ion in TARGETS]
        screening += [(project, "cv1", None, inputs, ion, (FULL,)) for inputs in INPUTS for ion in TARGETS]
        run_jobs(pool, train_job, screening, "CV1 · сетка и полные спектры")
        export_summary(project)
        if screen_only:
            return

        ranks = validated_choice(project, [(BASE_RANK, RAW_INPUT)] +
                                 [(config, RAW_INPUT) for config in RANKS if config != BASE_RANK])
        inputs = validated_choice(project, [(BASE_RANK, BASE_INPUT)] +
                                  [(BASE_RANK, prep) for prep in INPUTS if prep != BASE_INPUT])
        better_rank = next(config for config, _ in ranks if config != BASE_RANK)
        better_input = next(prep for _, prep in inputs if prep != BASE_INPUT)
        finalists = [(BASE_RANK, BASE_INPUT), (better_rank, inputs[0][1]), (BASE_RANK, better_input)]
        (result_root(project) / "finalists.json").write_text(json.dumps([
            {"selection": config, "inputs": prep} for config, prep in finalists
        ], indent=2), encoding="utf-8")
        run_jobs(pool, rank_job, ((project, split, config, ion) for split in splits[1:]
                                  for config, _ in finalists for ion in TARGETS), "CV2–CV3 · рейтинги финалистов")
        selected = [(project, split, config, prep, ion, KS)
                    for split in splits for config, prep in finalists for ion in TARGETS]
        full = [(project, split, None, prep, ion, (FULL,))
                for split in splits[1:] for _, prep in finalists for ion in TARGETS]
        run_jobs(pool, train_job, selected + full, "CV1–CV3 · проверка финалистов")
        export_summary(project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--screen-only", action="store_true", help="остановиться после сеток на CV1")
    args = parser.parse_args()
    # В Linux дочерние процессы наследуют эти значения от оболочки до импорта NumPy.
    print(f"Рабочих процессов: {args.workers} · PID {os.getpid()}", flush=True)
    run(args.project.resolve(), args.workers, args.screen_only)


if __name__ == "__main__":
    main()
