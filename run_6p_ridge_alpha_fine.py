"""Finer Ridge alpha grid: 10, 20, ..., 1000. Validation MSE, all 5427 channels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

CACHE = Path(r"d:\Desktop\Geo_Features\experiments\6p_full_patience300\xy_cv1.npz")
OUT = Path(r"d:\Desktop\Geo_Features\experiments\6p_ridge_alpha")
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
N_TRAIN, N_VALID = 2225, 600
ALPHAS = np.arange(10, 1001, 10)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = np.load(CACHE)
    X = blob["X"]
    y = blob["y"]
    X_train, X_valid = X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_VALID]
    y_train, y_valid = y[:N_TRAIN], y[N_TRAIN:N_TRAIN + N_VALID]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    rows = []
    for alpha in ALPHAS:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        model.fit(X_train_s, y_train)
        pred = model.predict(X_valid_s)
        for i, name in enumerate(TARGETS):
            rows.append({
                "alpha": int(alpha),
                "ion": name,
                "mse": float(mean_squared_error(y_valid[:, i], pred[:, i])),
            })
        if int(alpha) % 100 == 0:
            print(f"alpha={int(alpha)} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ridge_alpha_10_to_1000.csv", index=False)
    best = {}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, name in zip(axes.ravel(), TARGETS):
        part = frame[frame["ion"] == name]
        ax.plot(part["alpha"], part["mse"], color="#1f4e79", lw=1.6)
        winner = part.loc[part["mse"].idxmin()]
        best[name] = {"alpha": int(winner["alpha"]), "mse": float(winner["mse"])}
        ax.axvline(winner["alpha"], color="#c45c26", ls="--", lw=1.2)
        ax.set_title(f"{name}   лучшая alpha={int(winner['alpha'])}")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("MSE на validation")
    axes[1, 0].set_ylabel("MSE на validation")
    for ax in axes[1]:
        ax.set_xlabel("alpha (L2)")
    fig.suptitle("Ridge, alpha от 10 до 1000 с шагом 10", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "mse_vs_alpha_10_to_1000.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (OUT / "best_alpha_10_to_1000.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
