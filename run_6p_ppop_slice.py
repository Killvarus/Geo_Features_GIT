"""
ППОП на одном срезе 6P (фотолюминесценция углеродных точек).

Не меняет src/. Вызывает готовые TrueBackwardFeatureSelection и OLP.

Срез: испускание 375–575 нм при возбуждении 350 нм (201 канал).
Лист Sorted - CV1, разбиение сверху вниз 2225 / 600 / 300.
Пакет отбрасывания — 25% исходного среза, cv=None.
Цели в X не входят. Для ранжирования y стандартизуется на train,
чтобы NO3 (0–72 мМ) не задавил катионы (0–6 мМ) в MSE.
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

DATA_PATH = Path(r"d:\Desktop\Geo_Features\Data\6P_ALL.xlsx")
SHEET = "Sorted - CV1"
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_slice350")

N_EXC = 27
N_EM = 201
EXC_NM = [280 + 5 * i for i in range(N_EXC)]
EM_NM = list(range(375, 576))
EXC_TARGET_NM = 350
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID, N_TEST = 2225, 600, 300
DROP_FRACTION = 0.25
K_LIST = (50, 100, 150, 201)
RANDOM_STATE = 42


def _excitation_index(nm: int) -> int:
    if nm not in EXC_NM:
        raise ValueError(f"{nm} нм нет среди {EXC_NM[0]}..{EXC_NM[-1]} шаг 5")
    return EXC_NM.index(nm)


def slice_feature_numbers(exc_nm: int) -> list[int]:
    """Номера колонок 1..5427: возбуждение меняется медленно, испускание — быстро."""
    exc_i = _excitation_index(exc_nm)
    start = exc_i * N_EM + 1
    numbers = list(range(start, start + N_EM))
    if len(numbers) != N_EM or numbers[-1] > N_EXC * N_EM:
        raise RuntimeError(f"плохой срез: {numbers[:3]}..{numbers[-1]}")
    return numbers


def _assert_no_target_leakage(X: pd.DataFrame, y: pd.DataFrame) -> None:
    overlap = [c for c in X.columns if str(c) in TARGETS or c in TARGETS]
    if overlap:
        raise RuntimeError(f"таргеты попали в X: {overlap}")
    if list(y.columns) != TARGETS:
        raise RuntimeError(f"неожиданные цели: {list(y.columns)}")
    forbidden = {"Ramdom", "RND - CV1", "RND - CV2", "RND - CV3"}
    leaked_keys = [c for c in X.columns if str(c) in forbidden]
    if leaked_keys:
        raise RuntimeError(f"ключ разбиения попал в X: {leaked_keys}")


def load_slice() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    feature_numbers = slice_feature_numbers(EXC_TARGET_NM)
    # Позиция 0 — id. Колонка с заголовком f стоит на позиции f.
    positions = feature_numbers + [1 + N_EXC * N_EM + i for i in range(len(TARGETS))]
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET, usecols=positions, engine="openpyxl")
    if len(raw) != N_TRAIN + N_VALID + N_TEST:
        raise RuntimeError(f"ожидалось 3125 строк, получено {len(raw)}")

    rename = {num: f"E{EXC_TARGET_NM}_em{em}" for num, em in zip(feature_numbers, EM_NM)}
    # Заголовки из Excel — целые номера каналов.
    missing = [n for n in feature_numbers if n not in raw.columns and str(n) not in raw.columns]
    if missing:
        raise RuntimeError(f"в листе нет каналов среза, первые отсутствующие: {missing[:8]}")
    colmap = {}
    for num, name in rename.items():
        if num in raw.columns:
            colmap[num] = name
        elif str(num) in raw.columns:
            colmap[str(num)] = name
    raw = raw.rename(columns=colmap)
    feature_names = list(rename.values())
    for t in TARGETS:
        if t not in raw.columns:
            raise RuntimeError(f"нет колонки цели {t}")

    X = raw[feature_names].apply(pd.to_numeric, errors="raise")
    y = raw[TARGETS].apply(pd.to_numeric, errors="raise")
    _assert_no_target_leakage(X, y)
    if X.isna().any().any() or y.isna().any().any():
        raise RuntimeError("в срезе или целях есть NaN")

    X_train, y_train = X.iloc[:N_TRAIN].copy(), y.iloc[:N_TRAIN].copy()
    X_valid, y_valid = X.iloc[N_TRAIN:N_TRAIN + N_VALID].copy(), y.iloc[N_TRAIN:N_TRAIN + N_VALID].copy()
    X_test, y_test = X.iloc[N_TRAIN + N_VALID:].copy(), y.iloc[N_TRAIN + N_VALID:].copy()
    _assert_no_target_leakage(X_train, y_train)
    meta = {
        "sheet": SHEET,
        "excitation_nm": EXC_TARGET_NM,
        "emission_nm": [EM_NM[0], EM_NM[-1]],
        "n_features": len(feature_names),
        "feature_numbers": [int(n) for n in feature_numbers],
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "n_test": len(X_test),
        "y_train_mean": {c: float(y_train[c].mean()) for c in TARGETS},
        "y_test_mean": {c: float(y_test[c].mean()) for c in TARGETS},
    }
    return X_train, y_train, X_valid, y_valid, X_test, y_test, meta


def _per_target(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
    pred = np.asarray(y_pred, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    out = {}
    for i, name in enumerate(y_true.columns):
        yt = y_true.iloc[:, i].to_numpy(dtype=float)
        yp = pred[:, i]
        out[name] = {
            "R2": float(r2_score(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        }
    maes = [out[n]["MAE"] for n in y_true.columns if n != "NO3"]
    out["_macro_cation_MAE"] = float(np.mean(maes))
    out["_macro_all_MAE"] = float(np.mean([out[n]["MAE"] for n in y_true.columns]))
    return out


def linear_test_metrics(X_train, y_train, X_test, y_test, columns: list[str]) -> dict:
    model = LinearRegression()
    model.fit(X_train[columns], y_train)
    pred = model.predict(X_test[columns])
    return _per_target(y_test, pred)


def ranking_on_train(X_train: pd.DataFrame, y_train: pd.DataFrame, drop_count: int) -> pd.DataFrame:
    scaler = StandardScaler()
    y_scaled = pd.DataFrame(
        scaler.fit_transform(y_train),
        columns=y_train.columns,
        index=y_train.index,
    )
    _assert_no_target_leakage(X_train, y_train)
    selector = TrueBackwardFeatureSelection(
        estimator=LinearRegression(),
        n_features_to_drop=drop_count,
        cv=None,
    )
    selector.fit(X_train, y_scaled)
    ranking = selector.get_ranking_df()
    ranking.insert(1, "emission_nm", ranking["Feature"].str.replace(f"E{EXC_TARGET_NM}_em", "", regex=False).astype(int))
    return ranking


def olp_one(X_train, y_train, X_valid, y_valid, X_test, y_test, columns: list[str], tag: str) -> dict:
    _assert_no_target_leakage(X_train[columns], y_train)
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
        patience=100,
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
    cation_mae = float(np.mean([per_target[n]["MAE"] for n in TARGETS if n != "NO3"]))
    return {
        "tag": tag,
        "n_features": len(columns),
        "best_epoch": int(history["best_epoch"]) + 1,
        "best_val_loss": float(history["best_val_loss"]),
        "stopped_epoch": int(history["stopped_epoch"]),
        "per_target": per_target,
        "cation_MAE": cation_mae,
        "curve": str(curves / "learning_curve_iter_1.png"),
    }


def _fmt_table(rows: list[dict]) -> str:
    header = "| набор | k | модель | " + " | ".join(f"{t} MAE" for t in TARGETS) + " | катионы MAE | NO3 R2 |"
    sep = "|" + "---|" * (4 + len(TARGETS))
    lines = [header, sep]
    for row in rows:
        maes = " | ".join(f"{row['mae'][t]:.3f}" for t in TARGETS)
        lines.append(
            f"| {row['set']} | {row['k']} | {row['model']} | {maes} | {row['cation_mae']:.3f} | {row['no3_r2']:.3f} |"
        )
    return "\n".join(lines)


def write_report(meta, ranking, drop_count, rows, olp_runs) -> None:
    top = ranking.head(30)
    top_lines = [
        "| rank | канал | группа | score (neg-MSE, y стандартизован) |",
        "|---:|---:|---:|---:|",
    ]
    for _, rec in top.iterrows():
        top_lines.append(
            f"| {int(rec['Individual_Rank'])} | {int(rec['emission_nm'])} нм | {int(rec['Group_Rank'])} | {rec['Score']:.6f} |"
        )
    y_train_mean = {k: round(v, 3) for k, v in meta["y_train_mean"].items()}
    y_test_mean = {k: round(v, 3) for k, v in meta["y_test_mean"].items()}
    text = f"""# ППОП на срезе 6P, возбуждение {EXC_TARGET_NM} нм

Лист `{SHEET}` файла `Data/6P_ALL.xlsx`. Обучающий код в `src/` не менялся: ранжирование — `TrueBackwardFeatureSelection`, персептрон — `OLP`.

## Данные

Обратная задача: по спектру фотолюминесценции углеродных точек оценить концентрации `Cu`, `Ni`, `Al`, `Co`, `Cr`, `NO3`.

Срез — один спектр испускания при возбуждении **{EXC_TARGET_NM} нм** (в работах Гуськова это `1EW_350`): 201 канал, {EM_NM[0]}–{EM_NM[-1]} нм. В плоском векторе 27×201 это колонки **{meta['feature_numbers'][0]}–{meta['feature_numbers'][-1]}** (испускание меняется быстрее всего).

Разбиение листа, уже отсортированного по случайному ключу: train {meta['n_train']} / valid {meta['n_valid']} / test {meta['n_test']}. Ключ `Ramdom` в признаки не входил.

Проверка утечки: в X нет `{', '.join(TARGETS)}` и нет колонки разбиения. В y ровно эти шесть целей.

Средние целей на train: {y_train_mean}.
Средние целей на test: {y_test_mean}.

## Как считался ППОП

- Оценщик: `LinearRegression`, **cv=None** (один fit на текущем train, без кросс-валидации).
- Пакет: **{drop_count}** каналов из {meta['n_features']} — это {DROP_FRACTION:.0%} исходного среза. На каждом шаге выкидываются каналы, без которых стандартизованная MSE вырастает сильнее всего. Им достаётся лучший `Individual_Rank`.
- Перед ранжированием цели стандартизованы `StandardScaler`, обученным только на train. Иначе квадрат ошибки `NO3` (диапазон 0–72 мМ) задавил бы катионы (0–6 мМ), и отбор шёл бы почти только по нитрату. Сам класс отбора не менялся: ему передан уже масштабированный y.
- Персептрон учится на сырых целях. Масштаб для сети делает сам `OLP`, тоже fit только на train.

Верх ранжирования — каналы, которые линейная регрессия теряет больнее всего. Персептрон получает первые k строк этого списка.

## Верх ранжирования

{chr(10).join(top_lines)}

Полный список: `ranking.csv`.

## Метрики на тесте

MAE в мМ. «Катионы MAE» — среднее MAE по пяти металлам, без `NO3`. Линейная регрессия для этой таблицы дообучена на сыром y (не на стандартизованном), чтобы MAE была в мМ. Ранжирование при этом использовало стандартизованный y.

Один прогон OLP: Adam, lr=0.001, hidden=32, batch=64, потолка по числу эпох нет, patience=100, относительный допуск 0.003, seed={RANDOM_STATE}, CV сети выключен. lr=0.01 не брался: на прошлом Adam-прогоне с таким шагом были пики loss.

{_fmt_table(rows)}

Лучшая эпоха OLP:

| набор | best epoch | эпохи до остановки | best val MSE (после StandardScaler целей внутри OLP) |
|---|---:|---:|---:|
"""
    for run in olp_runs:
        text += f"| {run['tag']} | {run['best_epoch']} | {run['stopped_epoch']} | {run['best_val_loss']:.4f} |\n"
    text += """
Кривые: `experiments/6p_ppop_slice350/<тег>/learning_curves/learning_curve_iter_1.png`.

## Что из этого следует

Срез один, фолд один, сеть одна. Это проверка, что пайплайн живой и цели не утекают, а не сравнение с 2D-CNN из статей (там вход 27×201 и другая архитектура).

ППОП здесь отвечает на вопрос, можно ли выкинуть часть 201 каналов испускания и не развалить линейную модель и персептрон. Если MAE при k=50 или k=100 близка к k=201, соседние каналы избыточны — это как раз ожидаемая мультиколлинеарность спектра.
"""
    (OUT_DIR / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading slice", flush=True)
    X_train, y_train, X_valid, y_valid, X_test, y_test, meta = load_slice()
    print("loaded", meta["n_features"], "train", meta["n_train"], flush=True)
    drop_count = int(round(DROP_FRACTION * meta["n_features"]))
    if drop_count < 1 or drop_count >= meta["n_features"]:
        raise RuntimeError(f"плохой размер пакета: {drop_count}")
    print("PPOP drop", drop_count, "cv=None", flush=True)
    ranking = ranking_on_train(X_train, y_train, drop_count)
    ranking_path = OUT_DIR / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    ordered = ranking.sort_values("Individual_Rank")["Feature"].tolist()
    if set(ordered) != set(X_train.columns):
        raise RuntimeError("ранжирование потеряло или подменило признаки")
    if any(str(f) in TARGETS for f in ordered):
        raise RuntimeError("в ранжировании оказалась цель")

    rows = []
    olp_runs = []
    subsets = [(k, ordered[:k]) for k in K_LIST if k <= len(ordered)]
    subsets.append((len(ordered), ordered))
    # k=201 уже в K_LIST; не дублировать
    seen = set()
    unique_subsets = []
    for k, cols in subsets:
        if k in seen:
            continue
        seen.add(k)
        unique_subsets.append((k, cols))

    for k, cols in unique_subsets:
        print("linear metrics k", k, flush=True)
        lin = linear_test_metrics(X_train, y_train, X_test, y_test, cols)
        rows.append({
            "set": f"top{k}" if k != len(ordered) else "all",
            "k": k,
            "model": "LinearRegression",
            "mae": {t: lin[t]["MAE"] for t in TARGETS},
            "cation_mae": lin["_macro_cation_MAE"],
            "no3_r2": lin["NO3"]["R2"],
            "detail": {t: lin[t] for t in TARGETS},
        })
        print("OLP k", k, flush=True)
        olp = olp_one(X_train, y_train, X_valid, y_valid, X_test, y_test, cols, tag=f"k{k}")
        olp_runs.append(olp)
        rows.append({
            "set": f"top{k}" if k != len(ordered) else "all",
            "k": k,
            "model": "OLP",
            "mae": {t: olp["per_target"][t]["MAE"] for t in TARGETS},
            "cation_mae": olp["cation_MAE"],
            "no3_r2": olp["per_target"]["NO3"]["R2"],
            "detail": olp["per_target"],
        })

    payload = {"meta": meta, "drop_count": drop_count, "rows": rows, "olp": olp_runs}
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(meta, ranking, drop_count, rows, olp_runs)
    print("wrote", OUT_DIR / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
