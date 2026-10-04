"""
OLP на всей матрице 6P (27 возбуждений x 201 канал), patience=300.

src/ не меняется. Лист Sorted - CV1, сплит 2225/600/300.
Цели и ключ Ramdom в X не входят.
ППОП с линейной регрессией на train здесь не ранжирует: признаков больше, чем строк,
и in-sample MSE почти нулевая у любого набора полного ранга. Это проверяется одним fit
и пишется в отчёт, сам перебор 5427 регрессий не запускается.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.neural_network import to_excel_optimized_OLP

DATA_PATH = Path(r"d:\Desktop\Geo_Features\Data\6P_ALL.xlsx")
SHEET = "Sorted - CV1"
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300")
CACHE = OUT_DIR / "xy_cv1.npz"

N_EXC = 27
N_EM = 201
N_FEATURES = N_EXC * N_EM
EXC_NM = [280 + 5 * i for i in range(N_EXC)]
EM_NM = list(range(375, 576))
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID, N_TEST = 2225, 600, 300
RANDOM_STATE = 42


def _names() -> list[str]:
    names = []
    for exc in EXC_NM:
        for em in EM_NM:
            names.append(f"E{exc}_em{em}")
    if len(names) != N_FEATURES:
        raise RuntimeError("не сошлась сетка 27x201")
    return names


def _assert_no_leakage(columns: list[str]) -> None:
    forbidden = set(TARGETS) | {"Ramdom", "RND - CV1", "RND - CV2", "RND - CV3"}
    hit = [c for c in columns if c in forbidden]
    if hit:
        raise RuntimeError(f"в признаки попало запрещённое: {hit}")


def load_xy() -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = _names()
    _assert_no_leakage(names)
    if CACHE.exists():
        print("cache", CACHE, flush=True)
        blob = np.load(CACHE)
        X = blob["X"]
        y = blob["y"]
    else:
        print("reading excel", flush=True)
        # Позиция 0 — id. Каналы 1..5427 на позициях 1..5427. Цели сразу за ними.
        positions = list(range(1, 1 + N_FEATURES + len(TARGETS)))
        raw = pd.read_excel(DATA_PATH, sheet_name=SHEET, usecols=positions, engine="openpyxl")
        if raw.shape[0] != N_TRAIN + N_VALID + N_TEST:
            raise RuntimeError(f"строк {raw.shape[0]}, ожидалось 3125")
        if raw.shape[1] != N_FEATURES + len(TARGETS):
            raise RuntimeError(f"колонок {raw.shape[1]}")
        target_names = list(raw.columns[-len(TARGETS):])
        if target_names != TARGETS:
            raise RuntimeError(f"хвост таблицы не цели: {target_names}")
        feature_header = list(raw.columns[:N_FEATURES])
        expected = list(range(1, N_FEATURES + 1))
        if feature_header != expected and feature_header != [str(i) for i in expected]:
            raise RuntimeError(f"каналы не 1..5427, начало {feature_header[:5]}")
        X = raw.iloc[:, :N_FEATURES].to_numpy(dtype=np.float64)
        y = raw.iloc[:, N_FEATURES:].to_numpy(dtype=np.float64)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise RuntimeError("NaN или Inf в матрице или целях")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, X=X, y=y)
    if X.shape != (N_TRAIN + N_VALID + N_TEST, N_FEATURES):
        raise RuntimeError(f"форма X {X.shape}")
    if y.shape[1] != len(TARGETS):
        raise RuntimeError(f"форма y {y.shape}")
    return X, y, names


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "R2": float(r2_score(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        }
    out["macro_R2"] = float(np.mean([out[n]["R2"] for n in TARGETS]))
    out["cation_MAE"] = float(np.mean([out[n]["MAE"] for n in TARGETS if n != "NO3"]))
    return out


def linear_diagnostic(X_train, y_train, X_test, y_test) -> dict:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return {
        "train": _metrics(y_train, model.predict(X_train)),
        "test": _metrics(y_test, model.predict(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "rank_gt_samples": bool(X_train.shape[1] > X_train.shape[0]),
    }


def train_olp(X, y, names: list[str]) -> dict:
    X_df = pd.DataFrame(X, columns=names)
    y_df = pd.DataFrame(y, columns=TARGETS)
    _assert_no_leakage(list(X_df.columns))
    X_train = X_df.iloc[:N_TRAIN]
    y_train = y_df.iloc[:N_TRAIN]
    X_valid = X_df.iloc[N_TRAIN:N_TRAIN + N_VALID]
    y_valid = y_df.iloc[N_TRAIN:N_TRAIN + N_VALID]
    X_test = X_df.iloc[N_TRAIN + N_VALID:]
    y_test = y_df.iloc[N_TRAIN + N_VALID:]
    curves = OUT_DIR / "learning_curves"
    results = to_excel_optimized_OLP(
        file_name=str(OUT_DIR / "metrics.xlsx"),
        n_iter=1,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        batch_size=64,
        input_dim=N_FEATURES,
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
        log_file=str(OUT_DIR / "training.log"),
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
        "per_target": per_target,
        "macro_R2": float(np.mean([per_target[n]["R2"] for n in TARGETS])),
        "cation_MAE": float(np.mean([per_target[n]["MAE"] for n in TARGETS if n != "NO3"])),
        "best_epoch": int(history["best_epoch"]) + 1,
        "stopped_epoch": int(history["stopped_epoch"]),
        "best_val_loss": float(history["best_val_loss"]),
        "curve": str(curves / "learning_curve_iter_1.png"),
    }


def _row(metrics: dict, model: str) -> str:
    cells = " | ".join(f"{metrics[n]['R2']:.3f}" for n in TARGETS)
    maes = " | ".join(f"{metrics[n]['MAE']:.3f}" for n in TARGETS)
    return f"| {model} | {cells} | {metrics['macro_R2']:.3f} | {maes} | {metrics['cation_MAE']:.3f} |"


def write_report(linear: dict, olp: dict, y_train_mean: dict, y_test_mean: dict) -> None:
    text = f"""# 6P, вся матрица 27×201, patience 300

Лист `{SHEET}`, разбиение сверху вниз **{N_TRAIN} / {N_VALID} / {N_TEST}**. В X все **{N_FEATURES}** каналов возбуждение–испускание (280–410 нм шаг 5 нм, испускание 375–575 нм шаг 1 нм). Колонок целей и ключа `Ramdom` во входе нет.

Средние целей на train: {y_train_mean}.
Средние целей на test: {y_test_mean}.

Сеть: тот же OLP, `src/` не менялся. Adam, lr=0.001, hidden=32, batch=64, потолка по числу эпох нет, остановка только по patience, **patience=300**, относительный допуск 0.003, seed={RANDOM_STATE}, CV выключен. Масштаб признаков и целей считает сам OLP, fit только на train.

## R² и MAE на тесте

R² безразмерный. MAE в мМ.

| модель | Cu R2 | Ni R2 | Al R2 | Co R2 | Cr R2 | NO3 R2 | среднее R2 | Cu MAE | Ni MAE | Al MAE | Co MAE | Cr MAE | NO3 MAE | катионы MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_row(linear['test'], 'LinearRegression')}
{_row({**olp['per_target'], 'macro_R2': olp['macro_R2'], 'cation_MAE': olp['cation_MAE']}, 'OLP')}

Лучшая эпоха OLP: **{olp['best_epoch']}**. Остановка на эпохе {olp['stopped_epoch']}. Лучший val MSE после внутреннего StandardScaler целей: {olp['best_val_loss']:.4f}.

Кривая: `{olp['curve']}`.

Для сравнения, прошлый срез только 350 нм (201 канал, patience 100): OLP R² был Cu 0.41, Ni 0.05, Al 0.79, Co 0.37, Cr 0.87, NO3 0.56, катионы MAE 1.11 мМ.

## Почему на всей матрице не запускался ППОП

Линейная регрессия на train, все {N_FEATURES} каналов, cv нет:

| | среднее R² | Cu R2 | Ni R2 | Al R2 | Co R2 | Cr R2 | NO3 R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | {linear['train']['macro_R2']:.3f} | {linear['train']['Cu']['R2']:.3f} | {linear['train']['Ni']['R2']:.3f} | {linear['train']['Al']['R2']:.3f} | {linear['train']['Co']['R2']:.3f} | {linear['train']['Cr']['R2']:.3f} | {linear['train']['NO3']['R2']:.3f} |
| test | {linear['test']['macro_R2']:.3f} | {linear['test']['Cu']['R2']:.3f} | {linear['test']['Ni']['R2']:.3f} | {linear['test']['Al']['R2']:.3f} | {linear['test']['Co']['R2']:.3f} | {linear['test']['Cr']['R2']:.3f} | {linear['test']['NO3']['R2']:.3f} |

Признаков {N_FEATURES}, обучающих строк {N_TRAIN}. Пока столбцы тянут всё пространство строк, обычная МНК-регрессия на train интерполирует выборку: выкидывание одного канала почти не меняет train MSE. ППОП как в `TrueBackwardFeatureSelection` (score = train MSE модели без канала, cv=None) на такой матрице не отличает каналы. На срезе 201 < 2225 это ещё работало. Перебор тысяч регрессий поэтому не запускался.
"""
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, names = load_xy()
    _assert_no_leakage(names)
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_test, y_test = X[N_TRAIN + N_VALID:], y[N_TRAIN + N_VALID:]
    print("linear diagnostic", flush=True)
    linear = linear_diagnostic(X_train, y_train, X_test, y_test)
    print("train macro R2", round(linear["train"]["macro_R2"], 4), "test", round(linear["test"]["macro_R2"], 4), flush=True)
    print("OLP patience 300", flush=True)
    olp = train_olp(X, y, names)
    print("OLP macro R2", round(olp["macro_R2"], 4), "best epoch", olp["best_epoch"], flush=True)
    y_train_mean = {n: round(float(y_train[:, i].mean()), 3) for i, n in enumerate(TARGETS)}
    y_test_mean = {n: round(float(y_test[:, i].mean()), 3) for i, n in enumerate(TARGETS)}
    payload = {"linear": linear, "olp": olp, "y_train_mean": y_train_mean, "y_test_mean": y_test_mean}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(linear, olp, y_train_mean, y_test_mean)
    print("wrote", OUT_DIR / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
