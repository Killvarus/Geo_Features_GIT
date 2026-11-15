"""Картинки для доклада об ускоренном BSBS: спектр, формулы, время счёта, ускорение, применение."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT = Path("D:/Desktop/Geo_Features")
OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)

INK = "#14213D"
MUTED = "#4A5568"
GRID = "#D9DEE6"
BLUE = "#2A78D6"
ORANGE = "#EB6834"
BG = "#F7F6F2"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "mathtext.fontset": "cm",
        "savefig.transparent": True,
    }
)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=1.0, alpha=0.8)
    ax.set_axisbelow(True)


# 1. Матрица возбуждения–испускания раствора УТ без солей.
blob = np.load(PROJECT / "experiments/6p_full_patience300/xy_cv1.npz")
X, y = blob["X"], blob["y"]
row = int(np.flatnonzero((y[:, :5] == 0).all(axis=1))[0])
cube = X[row].reshape(27, 201)
excitation = 280 + 5 * np.arange(27)
emission = np.arange(375, 576)
ex, em = np.meshgrid(excitation, emission, indexing="ij")
scatter = (np.abs(em - ex) <= 10) | (np.abs(em - 2 * ex) <= 15)
top = np.percentile(cube[~scatter], 99.5)
ramp = LinearSegmentedColormap.from_list("blue", ["#F7F6F2", "#CDE2FB", "#6DA7EC", "#2A78D6", "#184F95", "#0D366B"])
fig, ax = plt.subplots(figsize=(11, 6.6))
image = ax.pcolormesh(emission, excitation, np.clip(cube, 0, top), cmap=ramp, shading="nearest", rasterized=True)
ax.set_xlabel("Испускание, нм")
ax.set_ylabel("Возбуждение, нм")
bar = fig.colorbar(image, ax=ax, pad=0.02)
bar.set_label("Интенсивность ФЛ, отн. ед.")
bar.outline.set_visible(False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "eem.png", dpi=110)
plt.close(fig)


# 2. Формулы.
def formula(text: str, name: str, size: int = 44) -> None:
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, text, fontsize=size, color=INK)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


formula(r"$\beta^{(j)} = \beta - c_j\,(A^{-1})_{\cdot j},\qquad c_j = \dfrac{\beta_j}{(A^{-1})_{jj}}$", "formula_beta.png")
formula(r"$S_j = \|r\|^2 + 2c_j\, r^{T}M_{\cdot j} + c_j^{2}\,\|M_{\cdot j}\|^2,\qquad M = X_eA^{-1}$", "formula_sse.png")

# 3. Время полного отбора (одна конфигурация, один поток, Cu).
bench = json.loads((PROJECT / "experiments/6p_fast_bsbs_benchmark/benchmark.json").read_text(encoding="utf-8"))
labels = ["Исходный BSBS,\nобучение с нуля", "Исходный BSBS,\nготовые матрицы", "Ускоренный BSBS"]
values = [bench["sklearn_seconds_total_estimate"], bench["prepared_seconds_total_estimate"], bench["fast_seconds_total"]]
texts = ["≈ 47 ч", "≈ 9.2 ч", "194 с"]
fig, ax = plt.subplots(figsize=(13, 5.6))
bars = ax.barh(labels, values, color=[ORANGE, ORANGE, BLUE], height=0.55)
ax.set_xscale("log")
ax.set_xlim(60, 1e6)
ticks = [60, 600, 3600, 36000, 86400 * 5]
ax.set_xticks(ticks, ["1 мин", "10 мин", "1 ч", "10 ч", "5 сут"])
ax.minorticks_off()
for bar_, text in zip(bars, texts):
    ax.text(bar_.get_width() * 1.15, bar_.get_y() + bar_.get_height() / 2, text, va="center", fontsize=26, color=INK, fontweight="bold")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "time.png", dpi=110)
plt.close(fig)

# 4. Ускорение в зависимости от числа признаков.
small = pd.read_csv(PROJECT / "experiments/6p_fast_bsbs_benchmark/small_runs.csv")
speed = small["direct_seconds"] / small["fast_seconds"]
full_speed = bench["prepared_seconds_total_estimate"] / bench["fast_seconds_total"]
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.plot(list(small["p"]) + [5427], list(speed) + [full_speed], color=BLUE, lw=2.5, zorder=1)
ax.scatter(small["p"], speed, s=150, color=BLUE, edgecolor=BG, linewidth=2, zorder=2, label="измерено")
ax.scatter([5427], [full_speed], s=150, facecolor=BG, edgecolor=BLUE, linewidth=2.5, zorder=2, label="оценка")
for p, s in zip(list(small["p"]) + [5427], list(speed) + [full_speed]):
    ax.annotate(f"{s:.0f}×", (p, s), textcoords="offset points", xytext=(-14, 16), ha="center", fontsize=24, color=INK)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([150, 300, 600, 1500, 5427], ["150", "300", "600", "1500", "5427"])
ax.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
ax.minorticks_off()
ax.set_ylim(8, 400)
ax.set_xlabel("Число признаков")
ax.set_ylabel("Ускорение, раз")
ax.legend(frameon=False, loc="upper left")
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "speedup.png", dpi=110)
plt.close(fig)

# 5. Применение: R² МСП на test по отобранным каналам (персептрон всегда на исходных спектрах).
points = pd.read_csv(PROJECT / "experiments/6p_selnorm_grid/analysis/points.csv")
full = []
for path in (PROJECT / "experiments/6p_selnorm_grid/all_channels").glob("*/curve_points.json"):
    full.append(json.loads(path.read_text(encoding="utf-8"))[0]["mean"])
full_mean = float(np.mean(full))
fig, ax = plt.subplots(figsize=(12, 6.4))
for spectrum, color, label in (("max", BLUE, "отбор по спектрам, нормированным на максимум"), ("raw", ORANGE, "отбор по исходным спектрам")):
    part = points[(points.method == "bsbs") & (points.channel == "zscore") & (points.spectrum == spectrum)]
    grouped = part.groupby("k")
    mean = grouped["test"].mean()
    error = grouped["test_std"].apply(lambda s: float(np.sqrt((s**2).sum())) / len(s))
    ax.errorbar(mean.index, mean.values, yerr=error.values, fmt="o-", ms=9, lw=2.5, capsize=4, color=color, label=label)
ax.axhline(full_mean, color=MUTED, ls="--", lw=2, label=f"все 5427 каналов ({full_mean:.3f})")
ax.set_xscale("log")
ax.set_xticks([10, 50, 100, 250, 500, 1000, 4000], ["10", "50", "100", "250", "500", "1000", "4000"])
ax.minorticks_off()
ax.set_xlabel("Число отобранных каналов")
ax.set_ylabel("R² на тестовом наборе")
ax.legend(frameon=False, loc="lower right", fontsize=20)
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "application.png", dpi=110)
plt.close(fig)

for path in sorted(OUT.glob("*.png")):
    from PIL import Image

    with Image.open(path) as picture:
        print(path.name, picture.size)
