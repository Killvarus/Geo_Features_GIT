"""
Кривые персептрона при малых k для разных рейтингов: experiments/6p_small_k/.

Рисуются две фигуры: R² на validation и R² на test, по иону на панель.
Сравнивать методы стоит по validation, test — для контроля.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "experiments" / "6p_small_k"
TARGETS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
LABELS = {
    "6p_baseline_uniform": "равномерный",
    "6p_baseline_random": "случайный",
    "6p_baseline_corr": "корреляция",
    "6p_ppop_ridge_a700_step0.1_cv5": "ППОП, alpha 700",
    "6p_rfe_ridge_a700_step0.1": "RFE, alpha 700",
    "6p_rfe_ridge_loo_step0.1": "RFE, LOO",
    "6p_rfe_ridge_loo1se_step0.1": "RFE, LOO-1SE",
    "6p_rfe_ridge_loosched500_step0.1": "RFE, 1SE → LOO с 500",
    "6p_ppop_ridge_loo1se_step0.1_cv5": "ППОП, LOO-1SE",
}


def load() -> pd.DataFrame:
    rows = []
    for path in DIR.glob("*/*/curve_points.json"):
        method, ion = path.parent.parent.name, path.parent.name
        for point in json.loads(path.read_text(encoding="utf-8")):
            rows.append(
                {
                    "method": method,
                    "ion": ion,
                    "k": point["k"],
                    "test": point["mean"],
                    "valid": point.get("mean_valid"),
                }
            )
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, column: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    colors = plt.get_cmap("tab10")
    for ax, ion in zip(axes.ravel(), TARGETS):
        part = table[table["ion"] == ion]
        for i, (method, label) in enumerate(LABELS.items()):
            line = part[part["method"] == method].sort_values("k")
            if line.empty:
                continue
            style = "--" if method.startswith("6p_baseline") else "-"
            ax.plot(line["k"], line[column], style, marker="o", ms=4, lw=1.5, color=colors(i), label=label)
        ax.set_xscale("log")
        ax.set_title(ion)
        ax.grid(True, alpha=0.3)
        values = part[column].dropna()
        if not values.empty:
            ax.set_ylim(max(values.min() - 0.03, values.max() - 0.45), values.max() + 0.03)
    for ax in axes[1]:
        ax.set_xlabel("Число признаков k")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"R² персептрона ({column})")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Персептрон на первых k каналах рейтинга, R² на {column}, среднее по 3 seed", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = load()
    table.to_csv(DIR / "small_k.csv", index=False)
    for column in ("valid", "test"):
        plot(table, column, DIR / f"r2_{column}_small_k.png")
    print(table.pivot_table(index=["ion", "k"], columns="method", values="valid").round(3).to_string())


if __name__ == "__main__":
    main()
