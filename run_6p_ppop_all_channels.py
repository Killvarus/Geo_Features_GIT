"""
ППОП сразу на всех 5427 каналах 6P.

Score канала — средний R² линейной регрессии БЕЗ этого канала.
Регрессия учится только на train, R² считается на validation.
Тест в отборе не используется. In-sample R² не используется.

Формула совпадает с sklearn.LinearRegression (проверено на нескольких каналах).
src/ не меняется.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels")
N_EXC = 27
N_EM = 201
EXC_NM = [280 + 5 * i for i in range(N_EXC)]
EM_NM = list(range(375, 576))
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID, N_TEST = 2225, 600, 300
DROP_FRACTION = 0.25
RANDOM_STATE = 42


def feature_names() -> list[str]:
    return [f"E{exc}_em{em}" for exc in EXC_NM for em in EM_NM]


def macro_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))


def per_target_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {TARGETS[i]: float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(len(TARGETS))}


def val_r2_without_each(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """Validation macro R² of min-norm OLS trained without each column in turn."""
    x_mean = X.mean(axis=0)
    y_mean = y.mean(axis=0)
    Xc = X - x_mean
    yc = y - y_mean
    Xvc = Xv - x_mean
    gram = Xc @ Xc.T
    gram_inv = np.linalg.pinv(gram, rcond=1e-12)
    coef_map = gram_inv @ yc
    base = (Xvc @ Xc.T) @ coef_map
    column_map = gram_inv @ Xc
    denom = 1.0 - np.sum(Xc * column_map, axis=0)
    gain = (Xc.T @ coef_map) / denom[:, None]
    mapped_val = (Xvc @ Xc.T) @ column_map
    resid0 = yv - y_mean - base
    ss_tot = np.sum((yv - yv.mean(axis=0)) ** 2, axis=0)
    n_features = X.shape[1]
    ss_res = np.empty((n_features, y.shape[1]))
    chunk = 400
    for start in range(0, n_features, chunk):
        end = min(start + chunk, n_features)
        delta = (mapped_val[:, start:end, None] - Xvc[:, start:end, None]) * gain[None, start:end, :]
        resid = resid0[:, None, :] - delta
        ss_res[start:end] = np.sum(resid * resid, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    if not np.isfinite(r2).all():
        raise RuntimeError(f"нечисловой R², min |denom|={np.min(np.abs(denom)):.3e}")
    return r2.mean(axis=1)


def sklearn_macro_without(X, y, Xv, yv, drop_local: int) -> float:
    keep = np.ones(X.shape[1], dtype=bool)
    keep[drop_local] = False
    pred = LinearRegression().fit(X[:, keep], y).predict(Xv[:, keep])
    return macro_r2(yv, pred)


def rank_all(X, y, Xv, yv, names: list[str], drop_count: int) -> pd.DataFrame:
    current = np.arange(X.shape[1])
    group_rank = np.zeros(X.shape[1], dtype=int)
    individual_rank = np.zeros(X.shape[1], dtype=int)
    best_score = np.zeros(X.shape[1])
    next_rank = 1
    group = 1
    checked = False
    while len(current) > drop_count:
        print(f"score {len(current)} channels", flush=True)
        scores = val_r2_without_each(X[:, current], y, Xv[:, current], yv)
        if not checked:
            for local in (0, len(current) // 2, len(current) - 1):
                ref = sklearn_macro_without(X[:, current], y, Xv[:, current], yv, local)
                err = abs(ref - float(scores[local]))
                print(f"  check local {local}: fast {scores[local]:.6f} sklearn {ref:.6f}", flush=True)
                if err > 1e-4:
                    raise RuntimeError(f"быстрый R² разошёлся со sklearn на {err}")
            checked = True
        order = np.argsort(scores, kind="mergesort")
        individual_rank[current[order]] = next_rank + np.arange(len(current))
        next_rank += drop_count
        drop_local = order[:drop_count]
        dropped = current[drop_local]
        group_rank[dropped] = group
        best_score[dropped] = scores[drop_local]
        current = current[order[drop_count:]]
        group += 1
    print(f"score remaining {len(current)}", flush=True)
    scores = val_r2_without_each(X[:, current], y, Xv[:, current], yv)
    order = np.argsort(scores, kind="mergesort")
    individual_rank[current[order]] = next_rank + np.arange(len(current))
    group_rank[current] = group
    best_score[current] = scores
    frame = pd.DataFrame({
        "Feature": names,
        "excitation_nm": [EXC_NM[i // N_EM] for i in range(len(names))],
        "emission_nm": [EM_NM[i % N_EM] for i in range(len(names))],
        "Group_Rank": group_rank,
        "Individual_Rank": individual_rank,
        "Val_R2_without_feature": best_score,
    })
    return frame.sort_values("Individual_Rank").reset_index(drop=True)


def linear_test_r2(Xtr, ytr, Xte, yte) -> dict:
    pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
    per = per_target_r2(yte, pred)
    per["macro"] = float(np.mean(list(per.values())))
    return per


def olp_test(X, y, columns: list[str], tag: str) -> dict:
    names = feature_names()
    name_to_i = {n: i for i, n in enumerate(names)}
    idx = [name_to_i[c] for c in columns]
    X_df = pd.DataFrame(X[:, idx], columns=columns)
    y_df = pd.DataFrame(y, columns=TARGETS)
    X_train, y_train = X_df.iloc[:N_TRAIN], y_df.iloc[:N_TRAIN]
    X_valid = X_df.iloc[N_TRAIN:N_TRAIN + N_VALID]
    y_valid = y_df.iloc[N_TRAIN:N_TRAIN + N_VALID]
    X_test = X_df.iloc[N_TRAIN + N_VALID:]
    y_test = y_df.iloc[N_TRAIN + N_VALID:]
    run_dir = OUT_DIR / tag
    results = to_excel_optimized_OLP(
        file_name=str(run_dir / "metrics.xlsx"),
        n_iter=1,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        batch_size=64,
        input_dim=len(columns),
        output_dim=len(TARGETS),
        learning_rate=0.001,
        num_epochs=10**9,
        patience=300,
        tolerance=0.003,
        tolerance_mode="relative",
        hidden_dim=32,
        save_plots_dir=str(run_dir / "learning_curves"),
        optimizer_type="adam",
        random_state=RANDOM_STATE,
        device="cpu",
        log_file=str(run_dir / "training.log"),
        enable_cv=False,
    )
    df_test = results[0]["test"]
    history = results[0]["history"]
    per = {}
    for i, name in enumerate(TARGETS):
        per[name] = float(df_test.loc[f"Оценка{i + 1}", "R2"])
    return {
        "tag": tag,
        "n_features": len(columns),
        "r2": per,
        "macro_R2": float(np.mean(list(per.values()))),
        "best_epoch": int(history["best_epoch"]) + 1,
        "stopped_epoch": int(history["stopped_epoch"]),
        "best_val_loss": float(history["best_val_loss"]),
    }


def write_report(drop_count: int, rows: list[dict], olp_runs: list[dict]) -> None:
    header = "| набор | k | модель | " + " | ".join(f"{t} R2" for t in TARGETS) + " | среднее R2 |"
    sep = "|" + "---|" * (4 + len(TARGETS))
    lines = [header, sep]
    for row in rows:
        cells = " | ".join(f"{row['r2'][t]:.3f}" for t in TARGETS)
        lines.append(f"| {row['set']} | {row['k']} | {row['model']} | {cells} | {row['macro']:.3f} |")
    epochs = ["| набор | best epoch | остановка | best val MSE |", "|---|---:|---:|---:|"]
    for run in olp_runs:
        epochs.append(f"| {run['tag']} | {run['best_epoch']} | {run['stopped_epoch']} | {run['best_val_loss']:.4f} |")
    text = f"""# ППОП на всех 5427 каналах, R² на отложенных строках

Лист `Sorted - CV1`. Train 2225 / valid 600 / test 300. В X все каналы возбуждение–испускание. Целей и ключа разбиения во входе нет.

Для каждого канала линейная регрессия учится **только на train** без этого канала. R² считается **на validation**, отдельно по каждому иону, затем усредняется. Так же устроен тест: модель эти строки не видела. Тест в отбор не входит. Подгонка R² на train не используется: при 5427 каналах и 2225 строках она равна 1 для любого полного набора и каналы не различает.

Пакет — **{drop_count}** каналов (25% от 5427). Сначала в рейтинг попадают каналы, без которых validation R² падает сильнее всего. `src/` не менялся; быстрый пересчёт сверен со `sklearn.LinearRegression` на трёх каналах.

Персептрон: Adam, lr=0.001, hidden=32, batch=64, patience=300, потолка в 1000 эпох нет, seed={RANDOM_STATE}.

## R² на тесте

{chr(10).join(lines)}

{chr(10).join(epochs)}

Ранжирование: `ranking.csv`. Кривые: `experiments/6p_ppop_all_channels/<тег>/learning_curves/`.
"""
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob = np.load(CACHE)
    X = blob["X"]
    y = blob["y"]
    names = feature_names()
    if X.shape != (N_TRAIN + N_VALID + N_TEST, len(names)):
        raise RuntimeError(f"неожиданная форма {X.shape}")
    Xtr, ytr = X[:N_TRAIN], y[:N_TRAIN]
    Xva, yva = X[N_TRAIN:N_TRAIN + N_VALID], y[N_TRAIN:N_TRAIN + N_VALID]
    Xte, yte = X[N_TRAIN + N_VALID:], y[N_TRAIN + N_VALID:]
    drop_count = int(round(DROP_FRACTION * len(names)))
    print("drop", drop_count, flush=True)
    ranking = rank_all(Xtr, ytr, Xva, yva, names, drop_count)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    ordered = ranking["Feature"].tolist()
    rows = []
    olp_runs = []
    for fraction, label in ((0.25, "top25"), (0.50, "top50")):
        k = int(round(fraction * len(names)))
        cols = ordered[:k]
        print("linear test", label, k, flush=True)
        lin = linear_test_r2(Xtr[:, [names.index(c) for c in cols]], ytr, Xte[:, [names.index(c) for c in cols]], yte)
        rows.append({"set": label, "k": k, "model": "LinearRegression", "r2": lin, "macro": lin["macro"]})
        print("OLP", label, flush=True)
        olp = olp_test(X, y, cols, tag=label)
        olp_runs.append(olp)
        rows.append({"set": label, "k": k, "model": "OLP", "r2": olp["r2"], "macro": olp["macro_R2"]})
    full = json.loads(Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\summary.json").read_text(encoding="utf-8"))
    rows.append({
        "set": "все каналы",
        "k": 5427,
        "model": "OLP",
        "r2": {t: full["olp"]["per_target"][t]["R2"] for t in TARGETS},
        "macro": full["olp"]["macro_R2"],
    })
    lin_full = full["linear"]["test"]
    rows.append({
        "set": "все каналы",
        "k": 5427,
        "model": "LinearRegression",
        "r2": {t: lin_full[t]["R2"] for t in TARGETS},
        "macro": lin_full["macro_R2"],
    })
    (OUT_DIR / "summary.json").write_text(json.dumps({"rows": rows, "olp": olp_runs, "drop_count": drop_count}, indent=2), encoding="utf-8")
    write_report(drop_count, rows, olp_runs)
    print("wrote", OUT_DIR / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
