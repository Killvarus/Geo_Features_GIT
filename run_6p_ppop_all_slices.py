"""
ППОП на всех 27 срезах возбуждения 6P, затем один OLP.

Каждый срез — 201 канал испускания. Пакет 25% среза (50), cv=None.
Совместный ППОП сразу на 5427 каналах не используется: на train МНК интерполирует
выборку (R2=1), и выкидывание одного канала не меняет ошибку.

src/ не меняется.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.feature_selection import TrueBackwardFeatureSelection
from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_slices")
N_EXC = 27
N_EM = 201
EXC_NM = [280 + 5 * i for i in range(N_EXC)]
EM_NM = list(range(375, 576))
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID, N_TEST = 2225, 600, 300
DROP_FRACTION = 0.25
K_PER_SLICE = (50, 100, 150)
RANDOM_STATE = 42


def feature_names() -> list[str]:
    return [f"E{exc}_em{em}" for exc in EXC_NM for em in EM_NM]


def _assert_no_leakage(columns) -> None:
    forbidden = set(TARGETS) | {"Ramdom", "RND - CV1", "RND - CV2", "RND - CV3"}
    hit = [c for c in columns if c in forbidden]
    if hit:
        raise RuntimeError(f"в признаки попало запрещённое: {hit}")


def load_frames():
    blob = np.load(CACHE)
    X = blob["X"]
    y = blob["y"]
    names = feature_names()
    if X.shape[1] != len(names):
        raise RuntimeError(f"в кэше {X.shape[1]} каналов, ожидалось {len(names)}")
    _assert_no_leakage(names)
    X_df = pd.DataFrame(X, columns=names)
    y_df = pd.DataFrame(y, columns=TARGETS)
    splits = {
        "train": (X_df.iloc[:N_TRAIN], y_df.iloc[:N_TRAIN]),
        "valid": (X_df.iloc[N_TRAIN:N_TRAIN + N_VALID], y_df.iloc[N_TRAIN:N_TRAIN + N_VALID]),
        "test": (X_df.iloc[N_TRAIN + N_VALID:], y_df.iloc[N_TRAIN + N_VALID:]),
    }
    return splits


def rank_slice(X_train: pd.DataFrame, y_scaled: pd.DataFrame, exc: int, drop_count: int) -> pd.DataFrame:
    cols = [f"E{exc}_em{em}" for em in EM_NM]
    _assert_no_leakage(cols)
    selector = TrueBackwardFeatureSelection(
        estimator=LinearRegression(),
        n_features_to_drop=drop_count,
        cv=None,
    )
    selector.fit(X_train[cols], y_scaled)
    ranking = selector.get_ranking_df()
    ranking.insert(0, "excitation_nm", exc)
    ranking.insert(2, "emission_nm", ranking["Feature"].str.replace(f"E{exc}_em", "", regex=False).astype(int))
    return ranking


def _metrics_frame(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    pred = np.asarray(y_pred, dtype=float)
    out = {}
    for i, name in enumerate(TARGETS):
        yt = y_true.iloc[:, i].to_numpy(dtype=float)
        yp = pred[:, i]
        out[name] = {
            "R2": float(r2_score(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        }
    out["macro_R2"] = float(np.mean([out[n]["R2"] for n in TARGETS]))
    out["cation_MAE"] = float(np.mean([out[n]["MAE"] for n in TARGETS if n != "NO3"]))
    return out


def linear_test(X_train, y_train, X_test, y_test, columns) -> dict:
    model = LinearRegression()
    model.fit(X_train[columns], y_train)
    return _metrics_frame(y_test, model.predict(X_test[columns]))


def olp_run(splits, columns, tag: str) -> dict:
    _assert_no_leakage(columns)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]
    run_dir = OUT_DIR / tag
    curves = run_dir / "learning_curves"
    results = to_excel_optimized_OLP(
        file_name=str(run_dir / "metrics.xlsx"),
        n_iter=1,
        X_train=X_train[columns],
        y_train=y_train,
        X_valid=X_valid[columns],
        y_valid=y_valid,
        X_test=X_test[columns],
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
        save_plots_dir=str(curves),
        optimizer_type="adam",
        random_state=RANDOM_STATE,
        device="cpu",
        log_file=str(run_dir / "training.log"),
        enable_cv=False,
    )
    df_test = results[0]["test"]
    history = results[0]["history"]
    per_target = {}
    for i, name in enumerate(TARGETS):
        row = df_test.loc[f"Оценка{i + 1}"]
        per_target[name] = {
            "R2": float(row["R2"]),
            "MAE": float(row["MAE"]),
            "MSE": float(row["MSE"]),
            "Pearson": float(row["Pearson"]),
        }
    return {
        "tag": tag,
        "n_features": len(columns),
        "per_target": per_target,
        "macro_R2": float(np.mean([per_target[n]["R2"] for n in TARGETS])),
        "cation_MAE": float(np.mean([per_target[n]["MAE"] for n in TARGETS if n != "NO3"])),
        "best_epoch": int(history["best_epoch"]) + 1,
        "stopped_epoch": int(history["stopped_epoch"]),
        "best_val_loss": float(history["best_val_loss"]),
        "curve": str(curves / "learning_curve_iter_1.png"),
    }


def write_report(drop_count: int, rows: list[dict], olp_runs: list[dict]) -> None:
    lines = [
        "| набор | k | модель | " + " | ".join(f"{t} R2" for t in TARGETS) + " | среднее R2 | катионы MAE |",
        "|" + "---|" * (4 + len(TARGETS)),
    ]
    for row in rows:
        r2s = " | ".join(f"{row['r2'][t]:.3f}" for t in TARGETS)
        lines.append(
            f"| {row['set']} | {row['k']} | {row['model']} | {r2s} | {row['macro_r2']:.3f} | {row['cation_mae']:.3f} |"
        )
    epoch_lines = [
        "| набор | признаки | best epoch | остановка | best val MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in olp_runs:
        epoch_lines.append(
            f"| {run['tag']} | {run['n_features']} | {run['best_epoch']} | {run['stopped_epoch']} | {run['best_val_loss']:.4f} |"
        )
    text = f"""# ППОП по всем 27 срезам возбуждения

Лист `Sorted - CV1`, разбиение 2225 / 600 / 300. В отбор вошли все возбуждения 280–410 нм с шагом 5 нм, на каждом — испускание 375–575 нм (201 канал). Цели `Cu Ni Al Co Cr NO3` и ключ `Ramdom` в X не входили.

ППОП считается **отдельно на каждом срезе**, тем же классом `TrueBackwardFeatureSelection`, `src/` не менялся. Пакет — **{drop_count}** каналов (25% среза), cv=None. Перед ранжированием цели стандартизованы на train, чтобы `NO3` не задавил катионы. Персептрон потом учится на сырых концентрациях: из каждого среза берутся первые k строк его ранжирования, и эти каналы склеиваются в один вход.

Совместный ППОП сразу на 5427 каналах не запускался. Обучающих строк 2225, и МНК на train даёт R² = 1 по всем целям: выкидывание одного канала не меняет train MSE, ранжирование не различает признаки.

Сеть: Adam, lr=0.001, hidden=32, batch=64, потолка по числу эпох нет, остановка только по patience=300, относительный допуск 0.003, seed={RANDOM_STATE}.

## R² на тесте

MAE катионов — среднее MAE по пяти металлам, в мМ.

{chr(10).join(lines)}

Строка «все 5427, OLP» — уже посчитанный прогон без отбора, patience 300 (`experiments/6p_full_patience300`).

{chr(10).join(epoch_lines)}

Ранжирование всех срезов: `ranking_all_slices.csv`.
Кривые: `experiments/6p_ppop_all_slices/<тег>/learning_curves/learning_curve_iter_1.png`.
"""
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_frames()
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    scaler = StandardScaler()
    y_scaled = pd.DataFrame(scaler.fit_transform(y_train), columns=TARGETS, index=y_train.index)
    drop_count = int(round(DROP_FRACTION * N_EM))
    rankings = []
    for exc in EXC_NM:
        print("rank", exc, flush=True)
        rankings.append(rank_slice(X_train, y_scaled, exc, drop_count))
    ranking = pd.concat(rankings, ignore_index=True)
    ranking.to_csv(OUT_DIR / "ranking_all_slices.csv", index=False)

    rows = []
    olp_runs = []
    for k in K_PER_SLICE:
        columns = []
        for exc in EXC_NM:
            part = ranking[ranking["excitation_nm"] == exc].sort_values("Individual_Rank")
            columns.extend(part["Feature"].head(k).tolist())
        if len(columns) != k * N_EXC or len(set(columns)) != len(columns):
            raise RuntimeError(f"сборка признаков сломалась при k={k}")
        _assert_no_leakage(columns)
        print("linear k", k, "n", len(columns), flush=True)
        lin = linear_test(X_train, y_train, X_test, y_test, columns)
        rows.append({
            "set": f"top{k} x 27",
            "k": len(columns),
            "model": "LinearRegression",
            "r2": {t: lin[t]["R2"] for t in TARGETS},
            "macro_r2": lin["macro_R2"],
            "cation_mae": lin["cation_MAE"],
        })
        print("OLP k", k, flush=True)
        olp = olp_run(splits, columns, tag=f"k{k}x27")
        olp_runs.append(olp)
        rows.append({
            "set": f"top{k} x 27",
            "k": len(columns),
            "model": "OLP",
            "r2": {t: olp["per_target"][t]["R2"] for t in TARGETS},
            "macro_r2": olp["macro_R2"],
            "cation_mae": olp["cation_MAE"],
        })

    full_path = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\summary.json")
    if full_path.exists():
        full = json.loads(full_path.read_text(encoding="utf-8"))["olp"]
        rows.append({
            "set": "все 5427",
            "k": 5427,
            "model": "OLP",
            "r2": {t: full["per_target"][t]["R2"] for t in TARGETS},
            "macro_r2": full["macro_R2"],
            "cation_mae": full["cation_MAE"],
        })

    (OUT_DIR / "summary.json").write_text(json.dumps({"rows": rows, "olp": olp_runs, "drop_count": drop_count}, indent=2), encoding="utf-8")
    write_report(drop_count, rows, olp_runs)
    print("wrote", OUT_DIR / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
