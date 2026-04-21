"""
Графики зависимости измеряемого поля от параметра слоя H3_8
(без корреляционного анализа).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_output_dir() -> Path:
 out = Path("field_dependency_plots")
 out.mkdir(exist_ok=True)
 return out


def get_feature_name(component: str, polarization: str, freq: int, pickup: int) -> str:
 return f"component}{polarization}{freq}_{pickup}"


def prepare_data(n_samples: int =7000):
 df = pd.read_csv("Data/mtsgrvmgn_trn.csv", nrows=n_samples)
 y = df["H3_8"]
 return df, y


def plot_scatter_matrix(df: pd.DataFrame, y: pd.Series, pickup: int, out_dir: Path):
 freqs = [1,7,13]
 components = ["RE", "IM"]
 pols = ["YX", "XY", "HX"]

 fig, axes = plt.subplots(6,3, figsize=(14,22), sharey=True)

 row =0
 for comp in components:
 for pol in pols:
 for col, freq in enumerate(freqs):
 ax = axes[row, col]
 feat = get_feature_name(comp, pol, freq, pickup)

 if feat not in df.columns:
 ax.set_visible(False)
 continue

 x = df[feat].values
 ax.scatter(x, y.values, s=7, alpha=0.25)
 ax.set_title(feat, fontsize=9)
 ax.set_xlabel("Поле")
 if col ==0:
 ax.set_ylabel("H3_8")
 ax.grid(alpha=0.2)
 row +=1

 fig.suptitle(f"Зависимость H3_8 от измеряемого поля (pickup={pickup})", fontsize=14)
 plt.tight_layout()
 save_path = out_dir / f"scatter_pickup_{pickup}.png"
 plt.savefig(save_path, dpi=150, bbox_inches="tight")
 plt.close(fig)


def plot_binned_dependency(df: pd.DataFrame, y: pd.Series, pickup: int, out_dir: Path):
 freqs = [1,7,13]
 components = ["RE", "IM"]
 pols = ["YX", "XY", "HX"]

 fig, axes = plt.subplots(2,3, figsize=(14,8), sharey=True)

 for i, comp in enumerate(components):
 for j, pol in enumerate(pols):
 ax = axes[i, j]

 for freq in freqs:
 feat = get_feature_name(comp, pol, freq, pickup)
 if feat not in df.columns:
 continue

 tmp = pd.DataFrame({"x": df[feat], "y": y})
 tmp["bin"] = pd.qcut(tmp["x"], q=12, duplicates="drop")
 grp = tmp.groupby("bin", observed=True).agg(
 x_mid=("x", "mean"),
 y_mean=("y", "mean"),
 y_std=("y", "std"),
 n=("y", "count"),
 ).reset_index(drop=True)

 grp["se"] = grp["y_std"] / np.sqrt(grp["n"].clip(lower=1))
 grp["y_low"] = grp["y_mean"] -1.96 * grp["se"]
 grp["y_high"] = grp["y_mean"] +1.96 * grp["se"]

 ax.plot(grp["x_mid"], grp["y_mean"], marker="o", linewidth=1.8, label=f"f={freq}")
 ax.fill_between(grp["x_mid"], grp["y_low"], grp["y_high"], alpha=0.15)

 ax.set_title(f"comp}{pol}")
 ax.set_xlabel("Поле")
 if j ==0:
 ax.set_ylabel("H3_8")
 ax.grid(alpha=0.25)

 handles, labels = axes[0,0].get_legend_handles_labels()
 if handles:
 fig.legend(handles, labels, loc="upper center", ncol=3)

 fig.suptitle(f"Биннинг-зависимость H3_8 от поля (pickup={pickup})", fontsize=14)
 plt.tight_layout(rect=[0,0,1,0.97])
 save_path = out_dir / f"binned_pickup_{pickup}.png"
 plt.savefig(save_path, dpi=150, bbox_inches="tight")
 plt.close(fig)


def plot_central_vs_edge_comparison(df: pd.DataFrame, y: pd.Series, out_dir: Path):
 freqs = [1,7,13]
 pickups = [16,1] # central, edge
 pickup_labels = {16: "central(16)",1: "edge(1)"}

 fig, axes = plt.subplots(2,2, figsize=(12,9))
 pairs = [("RE", "YX"), ("IM", "YX"), ("RE", "XY"), ("IM", "XY")]

 for ax, (comp, pol) in zip(axes.ravel(), pairs):
 for pickup in pickups:
 for freq in freqs:
 feat = get_feature_name(comp, pol, freq, pickup)
 if feat not in df.columns:
 continue

 tmp = pd.DataFrame({"x": df[feat], "y": y})
 tmp["bin"] = pd.qcut(tmp["x"], q=10, duplicates="drop")
 grp = tmp.groupby("bin", observed=True).agg(
 x_mid=("x", "mean"),
 y_mean=("y", "mean"),
 ).reset_index(drop=True)

 ax.plot(grp["x_mid"], grp["y_mean"], linewidth=1.6, label=f"f={freq}, {pickup_labels[pickup]}")

 ax.set_title(f"comp}{pol}")
 ax.set_xlabel("Поле")
 ax.set_ylabel("H3_8")
 ax.grid(alpha=0.25)

 handles, labels = axes[0,0].get_legend_handles_labels()
 if handles:
 fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)

 fig.suptitle("Сравнение зависимости для central vs edge пикета", fontsize=14)
 plt.tight_layout(rect=[0,0,1,0.95])
 save_path = out_dir / "central_vs_edge_comparison.png"
 plt.savefig(save_path, dpi=150, bbox_inches="tight")
 plt.close(fig)


def main():
 out_dir = make_output_dir()
 df, y = prepare_data(n_samples=7000)

 # Центральный пикет
 plot_scatter_matrix(df, y, pickup=16, out_dir=out_dir)
 plot_binned_dependency(df, y, pickup=16, out_dir=out_dir)

 # Крайний пикет
 plot_scatter_matrix(df, y, pickup=1, out_dir=out_dir)
 plot_binned_dependency(df, y, pickup=1, out_dir=out_dir)

 # Сравнение
 plot_central_vs_edge_comparison(df, y, out_dir=out_dir)

 print(f"Готово. Графики сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
 main()
