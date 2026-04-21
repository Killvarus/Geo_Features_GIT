"""
Построение графиков зависимости измеряемого поля от параметра слоя H3_8
без корреляционного анализа.

Сценарий:
- Пикет центральный:16
- Пикет крайний:1
- Частоты:1 (низкая),7 (средняя),13 (высокая)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_feature(component: str, polarization: str, freq: int, pickup: int) -> str:
 return "{}{}{}_{}".format(component, polarization, freq, pickup)


def binned_curve(x: pd.Series, y: pd.Series, q: int =12) -> pd.DataFrame:
 tmp = pd.DataFrame({"x": x, "y": y}).dropna()
 tmp["bin"] = pd.qcut(tmp["x"], q=q, duplicates="drop")
 grp = (
 tmp.groupby("bin", observed=True)
 .agg(x_mid=("x", "mean"), y_mean=("y", "mean"))
 .reset_index(drop=True)
 )
 return grp


def plot_pickup_dependency(
 df: pd.DataFrame,
 y: pd.Series,
 pickup: int,
 pickup_label: str,
 out_dir: Path,
) -> None:
 freqs = [1,7,13]
 pairs = [
 ("RE", "YX"),
 ("IM", "YX"),
 ("RE", "XY"),
 ("IM", "XY"),
 ("RE", "HX"),
 ("IM", "HX"),
 ]

 fig, axes = plt.subplots(2,3, figsize=(15,9), sharey=True)
 axes = axes.flatten()

 for ax, (comp, pol) in zip(axes, pairs):
 for freq, color in zip(freqs, ["tab:blue", "tab:orange", "tab:green"]):
 feat = build_feature(comp, pol, freq, pickup)
 if feat not in df.columns:
 continue

 sample = df[[feat]].join(y).sample(n=min(2500, len(df)), random_state=42)
 ax.scatter(sample[feat], sample[y.name], s=7, alpha=0.22, color=color, label=f"f={freq}")

 ax.set_title("{}{} | pickup={}".format(comp, pol, pickup))
 ax.set_xlabel("Измеряемое поле")
 ax.grid(alpha=0.2)

 axes[0].set_ylabel("H3_8")
 axes[3].set_ylabel("H3_8")

 handles, labels = axes[0].get_legend_handles_labels()
 if handles:
 fig.legend(handles, labels, loc="upper center", ncol=3)

 fig.suptitle(
 f"Зависимость H3_8 от поля: {pickup_label} пикет ({pickup}), частоты1/7/13",
 fontsize=13,
 )
 fig.tight_layout(rect=[0,0,1,0.95])
 fig.savefig(out_dir / f"dependency_{pickup_label}_pickup_{pickup}.png", dpi=160, bbox_inches="tight")
 plt.close(fig)


def plot_central_vs_edge_binned(
 df: pd.DataFrame,
 y: pd.Series,
 component: str,
 polarization: str,
 out_dir: Path,
) -> None:
 freqs = [1,7,13]
 central_pickup =16
 edge_pickup =1

 fig, axes = plt.subplots(1,3, figsize=(16,4.8), sharey=True)

 for i, freq in enumerate(freqs):
 ax = axes[i]

 feat_c = build_feature(component, polarization, freq, central_pickup)
 feat_e = build_feature(component, polarization, freq, edge_pickup)

 if feat_c in df.columns:
 curve_c = binned_curve(df[feat_c], y, q=12)
 ax.plot(curve_c["x_mid"], curve_c["y_mean"], marker="o", linewidth=1.8, label=f"central({central_pickup})")

 if feat_e in df.columns:
 curve_e = binned_curve(df[feat_e], y, q=12)
 ax.plot(curve_e["x_mid"], curve_e["y_mean"], marker="o", linewidth=1.8, label=f"edge({edge_pickup})")

 ax.set_title(f"f={freq}")
 ax.set_xlabel("Измеряемое поле")
 ax.grid(alpha=0.25)

 axes[0].set_ylabel("H3_8")
 handles, labels = axes[0].get_legend_handles_labels()
 if handles:
 fig.legend(handles, labels, loc="upper center", ncol=2)

 fig.suptitle(
 f"Central vs Edge: {component}{polarization}, частоты1/7/13",
 fontsize=13,
 )
 fig.tight_layout(rect=[0,0,1,0.92])
 fig.savefig(out_dir / f"central_vs_edge_{component}{polarization}_binned.png", dpi=160, bbox_inches="tight")
 plt.close(fig)


def main() -> None:
 out_dir = Path("field_dependency_plots_requested")
 out_dir.mkdir(exist_ok=True)

 df = pd.read_csv("Data/mtsgrvmgn_trn.csv", nrows=7000)
 y = df["H3_8"]

 plot_pickup_dependency(df, y, pickup=16, pickup_label="central", out_dir=out_dir)
 plot_pickup_dependency(df, y, pickup=1, pickup_label="edge", out_dir=out_dir)

 plot_central_vs_edge_binned(df, y, component="RE", polarization="YX", out_dir=out_dir)
 plot_central_vs_edge_binned(df, y, component="IM", polarization="YX", out_dir=out_dir)

 print(f"Готово. Графики сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
 main()
