"""
Кривая test R² персептрона от числа признаков ППОП.
Горизонтальная линия — та же сеть на всех 5427 каналах, без отбора.
Уже посчитанные k=1357, 2714 и 5427 не переобучаются.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from src.models.neural_network import to_excel_optimized_OLP

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
RANKING = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels\ranking.csv")
FULL_SUMMARY = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\summary.json")
OUT_DIR = Path(r"d:\Desktop\Geo_Features\experiments\6p_ppop_all_channels")
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
K_GRID = [50, 100, 200, 400, 800, 1357, 2000, 2714, 3500, 4500, 5427]
KNOWN = {
    1357: {
        "Cu": 0.6790301489192654,
        "Ni": 0.23899556145712053,
        "Al": 0.8113393103455642,
        "Co": 0.5818109064797545,
        "Cr": 0.8744191204843785,
        "NO3": 0.760470513195831,
    },
    2714: {
        "Cu": 0.7258060091391138,
        "Ni": 0.36255313391137156,
        "Al": 0.8197445050369605,
        "Co": 0.6047401142891666,
        "Cr": 0.9237373965659869,
        "NO3": 0.8067025054493925,
    },
}


def olp_r2(X: np.ndarray, y: np.ndarray, columns: list[str], tag: str) -> dict:
    X_df = pd.DataFrame(X, columns=columns)
    y_df = pd.DataFrame(y, columns=TARGETS)
    run_dir = OUT_DIR / "curve" / tag
    results = to_excel_optimized_OLP(
        file_name=str(run_dir / "metrics.xlsx"),
        n_iter=1,
        X_train=X_df.iloc[:N_TRAIN],
        y_train=y_df.iloc[:N_TRAIN],
        X_valid=X_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
        y_valid=y_df.iloc[N_TRAIN:N_TRAIN + N_VALID],
        X_test=X_df.iloc[N_TRAIN + N_VALID:],
        y_test=y_df.iloc[N_TRAIN + N_VALID:],
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
        random_state=42,
        device="cpu",
        log_file=str(run_dir / "training.log"),
        enable_cv=False,
    )
    df_test = results[0]["test"]
    return {TARGETS[i]: float(df_test.loc[f"Оценка{i + 1}", "R2"]) for i in range(len(TARGETS))}


def plot(points: list[dict], full_r2: dict) -> Path:
    ks = [p["k"] for p in points]
    macro = [float(np.mean(list(p["r2"].values()))) for p in points]
    full_macro = float(np.mean(list(full_r2.values())))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(ks, macro, "o-", color="#1f4e79", lw=2, ms=6, label="ППОП, затем персептрон")
    ax.axhline(full_macro, color="#c45c26", ls="--", lw=1.8, label=f"без отбора, все 5427 канала ({full_macro:.3f})")
    ax.scatter([5427], [full_macro], color="#c45c26", s=40, zorder=3)
    ax.set_xlabel("Число признаков (верх ранжирования ППОП)")
    ax.set_ylabel("Средний R² на тесте")
    ax.set_title("6P: качество персептрона от числа каналов")
    ax.set_xlim(0, 5700)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / "r2_vs_n_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ranking = pd.read_csv(RANKING).sort_values("Individual_Rank")
    ordered = ranking["Feature"].tolist()
    full = json.loads(FULL_SUMMARY.read_text(encoding="utf-8"))["olp"]["per_target"]
    full_r2 = {t: float(full[t]["R2"]) for t in TARGETS}
    blob = np.load(CACHE)
    X_all = blob["X"]
    y_all = blob["y"]
    points = []
    for k in K_GRID:
        if k == 5427:
            r2 = full_r2
            print("reuse full", k, flush=True)
        elif k in KNOWN:
            r2 = KNOWN[k]
            print("reuse", k, flush=True)
        else:
            cols = ordered[:k]
            print("OLP", k, flush=True)
            r2 = olp_r2(_select(X_all, ordered, cols), y_all, cols, tag=f"k{k}")
        points.append({"k": k, "r2": r2, "macro": float(np.mean(list(r2.values())))})
        print(" ", k, round(points[-1]["macro"], 4), flush=True)
    (OUT_DIR / "curve_points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
    path = plot(points, full_r2)
    print("wrote", path, flush=True)


def _select(X: np.ndarray, ordered: list[str], cols: list[str]) -> np.ndarray:
    # Feature names are E{exc}_em{em} in the same order as columns of X.
    index = {name: i for i, name in enumerate(_names())}
    return X[:, [index[c] for c in cols]]


def _names() -> list[str]:
    exc = [280 + 5 * i for i in range(27)]
    em = list(range(375, 576))
    return [f"E{e}_em{w}" for e in exc for w in em]


if __name__ == "__main__":
    main()
