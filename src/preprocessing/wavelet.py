"""
Дискретное вейвлет-преобразование признаков МТЗ.

Каждый объект — 6 каналов (RE/IM × YX/XY/HX), канал — сетка частота × пикет.
2D DWT (или 1D DWT по частоте) сжимает сетку в коэффициенты аппроксимации
и/или в подмножество коэффициентов по энергии на train.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..config import COMPONENT_TYPES, FREQUENCIES, PICKUPS, POLARIZATION_TYPES
from .aggregation import parse_feature_name

try:
    import pywt
except ImportError as exc:  # pragma: no cover
    raise ImportError("Нужен пакет PyWavelets: pip install PyWavelets") from exc


KEEP_MODES = ('approx', 'all', 'energy')
LAYOUTS = ('grid2d', 'freq1d')


def wavelet_config_name(
    wavelet: str,
    level: int,
    keep: str,
    layout: str = 'grid2d',
    energy_ratio: Optional[float] = None,
    n_features: Optional[int] = None,
) -> str:
    parts = ['wavelet', str(wavelet), f'l{int(level)}', str(layout), str(keep)]
    if keep == 'energy' and energy_ratio is not None:
        parts.append(f'e{int(round(float(energy_ratio) * 100))}')
    if keep == 'energy' and n_features is not None:
        parts.append(f'k{int(n_features)}')
    return '_'.join(parts)


def _min_length_for_level(wavelet: str, level: int) -> int:
    filt = pywt.Wavelet(wavelet).dec_len
    return int(np.ceil((filt - 1) * (2 ** max(int(level), 1))))


def pad_length(n: int, wavelet: str, level: int) -> int:
    """Длина, на которой periodization допускает заданный уровень DWT."""
    target = max(int(n), _min_length_for_level(wavelet, level))
    step = 2 ** max(int(level), 1)
    remainder = target % step
    if remainder:
        target += step - remainder
    return target


def _pad_grid(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = grid.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return grid
    return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='edge')


def _pad_vec(vec: np.ndarray, target: int) -> np.ndarray:
    n = vec.shape[0]
    if target <= n:
        return vec
    return np.pad(vec, (0, target - n), mode='edge')


def flatten_wavedec2(coeffs, keep: str) -> np.ndarray:
    approx = np.asarray(coeffs[0]).ravel()
    if keep == 'approx':
        return approx
    chunks = [approx]
    for detail_level in coeffs[1:]:
        for band in detail_level:
            chunks.append(np.asarray(band).ravel())
    return np.concatenate(chunks)


def flatten_wavedec(coeffs, keep: str) -> np.ndarray:
    if keep == 'approx':
        return np.asarray(coeffs[0]).ravel()
    return np.concatenate([np.asarray(c).ravel() for c in coeffs])


def energy_by_band_2d(coeffs) -> Dict[str, float]:
    out = {'A': float(np.sum(np.asarray(coeffs[0]) ** 2))}
    for i, detail_level in enumerate(coeffs[1:], start=1):
        for name, band in zip(('H', 'V', 'D'), detail_level):
            out[f'D{i}_{name}'] = float(np.sum(np.asarray(band) ** 2))
    return out


def _column_index_map(feature_cols: Sequence[str]) -> Dict[Tuple[str, str, int, int], int]:
    mapping: Dict[Tuple[str, str, int, int], int] = {}
    for i, name in enumerate(feature_cols):
        parsed = parse_feature_name(name)
        if parsed is None:
            continue
        key = (
            parsed['component'],
            parsed['polarization'],
            parsed['frequency'],
            parsed['pickup'],
        )
        mapping[key] = i
    return mapping


def values_to_channel_grids(
    X_values: np.ndarray,
    feature_cols: Sequence[str],
    frequencies: Sequence[int],
    pickups: Sequence[int],
) -> List[np.ndarray]:
    """Список каналов, каждый shape (n_samples, n_freq, n_pickup)."""
    index_map = _column_index_map(feature_cols)
    n = X_values.shape[0]
    n_f = len(frequencies)
    n_p = len(pickups)
    grids: List[np.ndarray] = []
    for comp in COMPONENT_TYPES:
        for pol in POLARIZATION_TYPES:
            grid = np.zeros((n, n_f, n_p), dtype=np.float64)
            for i, freq in enumerate(frequencies):
                for j, pickup in enumerate(pickups):
                    col = index_map.get((comp, pol, int(freq), int(pickup)))
                    if col is not None:
                        grid[:, i, j] = X_values[:, col]
            grids.append(grid)
    return grids


def _attach_targets(
    features: pd.DataFrame,
    source: pd.DataFrame,
    target_columns: List[str],
) -> pd.DataFrame:
    if len(features) != len(source):
        raise ValueError(
            "Wavelet-признаки и исходный сплит разной длины: "
            f"{len(features)} vs {len(source)}"
        )
    out = features.copy()
    out.index = source.index
    for col in target_columns:
        if col not in source.columns:
            raise ValueError(f"Целевой столбец {col} отсутствует в сплите")
        out[col] = source[col].values
    return out


class WaveletTransformer:
    """
    Fit только на train: импутер, скейлер входа, маска энергии, скейлер коэффициентов.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 2,
        keep: str = 'approx',
        layout: str = 'grid2d',
        energy_ratio: Optional[float] = None,
        n_features: Optional[int] = None,
        mode: str = 'periodization',
        frequencies: Optional[Sequence[int]] = None,
        pickups: Optional[Sequence[int]] = None,
    ):
        if keep not in KEEP_MODES:
            raise ValueError(f"keep должен быть одним из {KEEP_MODES}, получено {keep}")
        if layout not in LAYOUTS:
            raise ValueError(f"layout должен быть одним из {LAYOUTS}, получено {layout}")
        if keep == 'energy' and energy_ratio is None and n_features is None:
            raise ValueError("Для keep='energy' нужен energy_ratio или n_features")

        self.wavelet = wavelet
        self.level = int(level)
        self.keep = keep
        self.layout = layout
        self.energy_ratio = energy_ratio
        self.n_features = n_features
        self.mode = mode
        self.frequencies = list(frequencies) if frequencies is not None else list(FREQUENCIES)
        self.pickups = list(pickups) if pickups is not None else list(PICKUPS)

        self.imputer = SimpleImputer(strategy='mean')
        self.input_scaler = StandardScaler()
        self.coeff_scaler = StandardScaler()

        self.feature_cols_: Optional[List[str]] = None
        self.padded_shape_: Optional[Tuple[int, int]] = None
        self.padded_length_: Optional[int] = None
        self.effective_level_: Optional[int] = None
        self.n_coeffs_full_: Optional[int] = None
        self.n_features_used: Optional[int] = None
        self.keep_indices_: Optional[np.ndarray] = None
        self.energy_retained: Optional[float] = None
        self.energy_by_band: Optional[Dict[str, float]] = None
        self.mean_coeff_energy_: Optional[np.ndarray] = None

    def _prepare_level_and_pad(self) -> None:
        n_f = len(self.frequencies)
        n_p = len(self.pickups)
        if self.layout == 'grid2d':
            self.padded_shape_ = (
                pad_length(n_f, self.wavelet, self.level),
                pad_length(n_p, self.wavelet, self.level),
            )
            max_h = pywt.dwt_max_level(self.padded_shape_[0], pywt.Wavelet(self.wavelet).dec_len)
            max_w = pywt.dwt_max_level(self.padded_shape_[1], pywt.Wavelet(self.wavelet).dec_len)
            self.effective_level_ = max(1, min(self.level, max_h, max_w))
        else:
            self.padded_length_ = pad_length(n_f, self.wavelet, self.level)
            max_len = pywt.dwt_max_level(self.padded_length_, pywt.Wavelet(self.wavelet).dec_len)
            self.effective_level_ = max(1, min(self.level, max_len))

    def _decompose2(self, padded: np.ndarray):
        try:
            return pywt.wavedec2(
                padded,
                wavelet=self.wavelet,
                level=self.effective_level_,
                mode=self.mode,
            )
        except ValueError:
            return pywt.wavedec2(
                padded,
                wavelet=self.wavelet,
                level=self.effective_level_,
                mode='symmetric',
            )

    def _decompose1(self, vec: np.ndarray):
        try:
            return pywt.wavedec(
                vec,
                wavelet=self.wavelet,
                level=self.effective_level_,
                mode=self.mode,
            )
        except ValueError:
            return pywt.wavedec(
                vec,
                wavelet=self.wavelet,
                level=self.effective_level_,
                mode='symmetric',
            )

    def _dwt_one(self, grid: np.ndarray) -> np.ndarray:
        if self.layout == 'grid2d':
            padded = _pad_grid(grid, self.padded_shape_[0], self.padded_shape_[1])
            coeffs = self._decompose2(padded)
            return flatten_wavedec2(coeffs, 'all' if self.keep == 'energy' else self.keep)

        parts = []
        for pickup_idx in range(grid.shape[1]):
            vec = _pad_vec(grid[:, pickup_idx], self.padded_length_)
            coeffs = self._decompose1(vec)
            parts.append(flatten_wavedec(coeffs, 'all' if self.keep == 'energy' else self.keep))
        return np.concatenate(parts)

    def _dwt_sample_bands(self, grid: np.ndarray) -> Dict[str, float]:
        if self.layout != 'grid2d':
            return {}
        padded = _pad_grid(grid, self.padded_shape_[0], self.padded_shape_[1])
        coeffs = self._decompose2(padded)
        return energy_by_band_2d(coeffs)

    def _grids_to_coeffs(self, grids: List[np.ndarray]) -> np.ndarray:
        n = grids[0].shape[0]
        rows = []
        for i in range(n):
            parts = [self._dwt_one(g[i]) for g in grids]
            rows.append(np.concatenate(parts))
        return np.vstack(rows)

    def fit(self, X: pd.DataFrame):
        self.feature_cols_ = list(X.columns)
        self._prepare_level_and_pad()

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.input_scaler.fit_transform(X_imputed)
        grids = values_to_channel_grids(
            X_scaled, self.feature_cols_, self.frequencies, self.pickups
        )
        coeffs = self._grids_to_coeffs(grids)
        self.n_coeffs_full_ = int(coeffs.shape[1])

        if self.keep == 'energy':
            energy = np.mean(coeffs ** 2, axis=0)
            self.mean_coeff_energy_ = energy
            order = np.argsort(energy)[::-1]
            total = float(energy.sum()) if float(energy.sum()) > 0 else 1.0
            if self.energy_ratio is not None:
                cum = np.cumsum(energy[order]) / total
                n_keep = int(np.searchsorted(cum, self.energy_ratio) + 1)
            else:
                n_keep = int(self.n_features)
            n_keep = max(1, min(n_keep, len(order)))
            self.keep_indices_ = np.sort(order[:n_keep])
            kept_energy = float(energy[self.keep_indices_].sum())
            self.energy_retained = kept_energy / total
            selected = coeffs[:, self.keep_indices_]
        else:
            self.keep_indices_ = None
            selected = coeffs
            if self.keep == 'approx' and self.layout == 'grid2d':
                band_acc: Dict[str, List[float]] = {}
                n_probe = min(len(grids[0]), 64)
                for i in range(n_probe):
                    for g in grids:
                        bands = self._dwt_sample_bands(g[i])
                        for k, v in bands.items():
                            band_acc.setdefault(k, []).append(v)
                self.energy_by_band = {k: float(np.mean(vs)) for k, vs in band_acc.items()}
                approx_e = self.energy_by_band.get('A', 0.0)
                total_e = float(sum(self.energy_by_band.values())) or 1.0
                self.energy_retained = approx_e / total_e
            else:
                self.energy_retained = 1.0

        self.coeff_scaler.fit(selected)
        self.n_features_used = int(selected.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None:
            raise RuntimeError("WaveletTransformer.fit() не вызван")
        X_aligned = X[self.feature_cols_]
        X_imputed = self.imputer.transform(X_aligned)
        X_scaled = self.input_scaler.transform(X_imputed)
        grids = values_to_channel_grids(
            X_scaled, self.feature_cols_, self.frequencies, self.pickups
        )
        coeffs = self._grids_to_coeffs(grids)
        if self.keep_indices_ is not None:
            coeffs = coeffs[:, self.keep_indices_]
        coeffs = self.coeff_scaler.transform(coeffs)
        columns = [f'W{i + 1}' for i in range(self.n_features_used)]
        return pd.DataFrame(coeffs, index=X.index, columns=columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def get_info(self) -> Dict:
        return {
            'wavelet': self.wavelet,
            'level': self.level,
            'effective_level': self.effective_level_,
            'keep': self.keep,
            'layout': self.layout,
            'energy_ratio': self.energy_ratio,
            'padded_shape': list(self.padded_shape_) if self.padded_shape_ else None,
            'padded_length': self.padded_length_,
            'n_coeffs_full': self.n_coeffs_full_,
            'n_features': self.n_features_used,
            'energy_retained': self.energy_retained,
            'energy_by_band': self.energy_by_band,
        }


def apply_wavelet_to_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target_columns: List[str],
    target_prefix: str = 'H',
    **transformer_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WaveletTransformer]:
    """DWT на train/valid/test. В выход попадают только запрошенные таргеты."""
    all_target_cols = [c for c in train.columns if c.startswith(target_prefix)]
    feature_cols = [c for c in train.columns if c not in all_target_cols]
    missing = [c for c in feature_cols if c not in valid.columns or c not in test.columns]
    if missing:
        raise ValueError(f"Признаки отсутствуют в valid/test: {missing[:10]}")

    transformer = WaveletTransformer(**transformer_kwargs)
    X_train = transformer.fit_transform(train[feature_cols])
    X_valid = transformer.transform(valid[feature_cols])
    X_test = transformer.transform(test[feature_cols])

    train_w = _attach_targets(X_train, train, target_columns)
    valid_w = _attach_targets(X_valid, valid, target_columns)
    test_w = _attach_targets(X_test, test, target_columns)
    return train_w, valid_w, test_w, transformer


def make_synthetic_mtz_frame(
    n_samples: int,
    seed: int = 0,
    noise: float = 0.05,
) -> pd.DataFrame:
    """Синтетическая сетка 13×31, поле зависит от H3 (и слабее от H1/H2)."""
    rng = np.random.default_rng(seed)
    h1 = rng.uniform(10.0, 80.0, size=n_samples)
    h2 = rng.uniform(20.0, 150.0, size=n_samples)
    h3 = rng.uniform(50.0, 400.0, size=n_samples)
    freq_axis = np.linspace(0.0, 1.0, len(FREQUENCIES))
    pickup_axis = np.linspace(-1.0, 1.0, len(PICKUPS))
    data: Dict[str, np.ndarray] = {}
    for comp in COMPONENT_TYPES:
        for pol in POLARIZATION_TYPES:
            pol_scale = {'YX': 1.0, 'XY': 0.85, 'HX': 0.55}[pol]
            for fi, freq in enumerate(FREQUENCIES):
                for pi, pickup in enumerate(PICKUPS):
                    decay = np.exp(-freq_axis[fi] * h3 / 220.0)
                    profile = np.cos(pickup_axis[pi] * h3 / 110.0)
                    layer = 0.15 * np.exp(-freq_axis[fi] * h1 / 90.0) + 0.1 * np.sin(
                        pickup_axis[pi] * h2 / 80.0
                    )
                    if comp == 'IM':
                        profile = np.sin(pickup_axis[pi] * h3 / 95.0)
                    values = pol_scale * decay * (profile + layer)
                    values = values + rng.normal(0.0, noise, size=n_samples)
                    data[f'{comp}{pol}{freq}_{pickup}'] = values
    data['H1_8'] = h1
    data['H2_8'] = h2
    data['H3_8'] = h3
    return pd.DataFrame(data)


def plot_wavelet_energy(
    transformer: WaveletTransformer,
    save_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if transformer.energy_by_band:
        names = list(transformer.energy_by_band.keys())
        vals = [transformer.energy_by_band[k] for k in names]
        ax.bar(names, vals, color='#3b6d9a', alpha=0.85)
        ax.set_ylabel('Суммарная энергия')
        ax.set_title(
            f"Энергия по полосам | {transformer.wavelet} L{transformer.effective_level_}"
        )
    elif transformer.mean_coeff_energy_ is not None:
        energy = np.sort(transformer.mean_coeff_energy_)[::-1]
        total = energy.sum() or 1.0
        ax.plot(np.cumsum(energy) / total, lw=2)
        ax.set_xlabel('Коэффициенты, отсортированные по энергии')
        ax.set_ylabel('Накопленная доля энергии')
        ax.set_title(f"Компакция энергии | {transformer.wavelet} {transformer.keep}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Нет статистики энергии', ha='center', va='center')
        ax.set_axis_off()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def is_haar_config_row(row: pd.Series) -> bool:
    """Конфиг Haar: в имени папки есть 'haar' (wavelet_haar_l1_... / wavelet_haar_l2_...)."""
    experiment_id = str(row.get('experiment_id', '') or '').lower()
    if 'haar' in experiment_id:
        return True
    wavelet = str(row.get('wavelet', '') or '').lower()
    return wavelet == 'haar'


def filter_haar_results(results: pd.DataFrame) -> pd.DataFrame:
    """Оставить только Haar-конфиги. db4 отбрасывается."""
    if results is None or results.empty:
        return pd.DataFrame()
    mask = results.apply(is_haar_config_row, axis=1)
    return results.loc[mask].copy()


def wavelet_config_short_label(row: pd.Series) -> str:
    experiment_id = str(row.get('experiment_id', '') or '')
    keep = str(row.get('wavelet_keep', '') or '')
    level = row.get('wavelet_level')
    if pd.isna(level):
        if '_l1_' in experiment_id:
            level = 1
        elif '_l2_' in experiment_id:
            level = 2
    if keep == 'energy' or 'energy' in experiment_id:
        keep_s = 'energy 90%' if ('e90' in experiment_id or keep == 'energy') else 'energy'
    elif keep == 'approx' or 'approx' in experiment_id:
        keep_s = 'approx'
    else:
        keep_s = keep or experiment_id
    if level is not None and not pd.isna(level):
        return f'L{int(level)} {keep_s}'
    return experiment_id.replace('wavelet_haar_', '').replace('_grid2d', '')


def plot_wavelet_comparison(
    results: List[Dict],
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    annotate: bool = False,
    title: Optional[str] = None,
    group_col: Optional[str] = 'wavelet',
):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    if group_col and group_col not in df.columns:
        group_col = None
    std_col = metric.replace('_mean', '_std')
    if group_col:
        for name, group in df.groupby(group_col):
            group = group.sort_values('n_features')
            yerr = group[std_col] if std_col in group.columns else None
            ax.errorbar(
                group['n_features'],
                group[metric],
                yerr=yerr,
                marker='o',
                capsize=3,
                label=str(name),
            )
    else:
        df = df.sort_values('n_features')
        yerr = df[std_col] if std_col in df.columns else None
        ax.errorbar(df['n_features'], df[metric], yerr=yerr, marker='o', capsize=3)
    if annotate:
        labeled = df.drop_duplicates(subset=['n_features', metric], keep='first')
        for _, row in labeled.iterrows():
            ax.annotate(
                wavelet_config_short_label(row),
                (row['n_features'], row[metric]),
                textcoords='offset points',
                xytext=(8, 8),
                ha='left',
                fontsize=8,
            )
    ax.set_xlabel('Число вейвлет-признаков')
    ax.set_ylabel(metric.replace('_mean', '').upper())
    ax.set_title(title or f'{metric.replace("_mean", "").upper()} vs wavelet features')
    ax.grid(True, alpha=0.3)
    if group_col and df[group_col].nunique() > 1:
        ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_wavelet_config_bars(
    results: List[Dict],
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)
    if df.empty or metric not in df.columns:
        return None
    df = df.copy()
    df['label'] = df.apply(wavelet_config_short_label, axis=1)
    df = df.sort_values(metric, ascending=(not metric.startswith('r2')))
    std_col = metric.replace('_mean', '_std')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(df))
    xerr = df[std_col] if std_col in df.columns else None
    ax.barh(list(y), df[metric], xerr=xerr, capsize=3, color='#3b6d9a', alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df['label'], fontsize=10)
    ax.set_xlabel(metric.replace('_mean', '').upper())
    ax.set_title(title or f'{metric.replace("_mean", "").upper()} | Haar')
    ax.grid(True, alpha=0.3, axis='x')
    err_vals = df[std_col].fillna(0.0) if std_col in df.columns else 0.0
    right = float((df[metric] + err_vals).max()) if len(df) else 1.0
    ax.set_xlim(0, right * 1.18 if right > 0 else 1.0)
    for yi, val, err in zip(y, df[metric], err_vals if std_col in df.columns else [0.0] * len(df)):
        ax.text(val + float(err), yi, f'  {val:.4f}', va='center', fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_wavelet_job_heatmap(
    results: pd.DataFrame,
    metric: str = 'r2_mean',
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = results.copy()
    if df.empty or metric not in df.columns:
        return None
    df['config'] = df.apply(wavelet_config_short_label, axis=1)
    if 'experiment_name' in df.columns:
        df['job'] = df['experiment_name'].astype(str)
    elif 'job' in df.columns:
        df['job'] = df['job'].astype(str)
    else:
        return None
    pivot = df.pivot_table(index='job', columns='config', values=metric, aggfunc='mean')
    col_order = [c for c in ('L1 approx', 'L2 approx', 'L2 energy 90%') if c in pivot.columns]
    col_order += [c for c in pivot.columns if c not in col_order]
    pivot = pivot.reindex(columns=col_order)
    job_order = sorted(pivot.index, key=_wavelet_job_sort_key)
    pivot = pivot.reindex(index=job_order)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = 'RdYlGn' if metric.startswith('r2') else 'YlOrRd_r'
    fmt = '.4f' if metric.startswith('r2') else '.2e'
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar_kws={'label': metric.replace('_mean', '').upper()})
    ax.set_xlabel('Конфиг Haar')
    ax.set_ylabel('Джоб')
    ax.set_title(title or f'{metric.replace("_mean", "").upper()} | только Haar')
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _wavelet_job_sort_key(name: str):
    text = str(name)
    dataset_rank = 0
    if 'train3_1' in text:
        dataset_rank = 2
    elif 'train3' in text:
        dataset_rank = 1
    target_rank = 9
    for i, target in enumerate(('H1_8', 'H2_8', 'H3_8')):
        if text.endswith(target) or f'_{target}' in text:
            target_rank = i
            break
    return (dataset_rank, target_rank, text)


def rebuild_haar_only_plots(run_dir: Path) -> List[Path]:
    """Пересобрать сравнительные графики только по Haar, не затирая Haar+db4."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..evaluation.metrics import plot_feature_count_vs_time

    run_dir = Path(run_dir)
    saved: List[Path] = []
    job_dirs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('wavelet_')),
        key=lambda p: _wavelet_job_sort_key(p.name),
    )
    all_haar_rows: List[Dict] = []

    for job_dir in job_dirs:
        rows = []
        for summary_path in job_dir.glob('*/summary.json'):
            with open(summary_path, 'r', encoding='utf-8') as f:
                rows.append(json.load(f))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        haar_df = filter_haar_results(df)
        if haar_df.empty:
            continue
        all_haar_rows.extend(haar_df.to_dict('records'))
        plots_dir = job_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        metric_specs = [
            ('r2_mean', 'R2 vs wavelet features (только Haar)'),
            ('mse_mean', 'MSE vs wavelet features (только Haar)'),
        ]
        for metric, title in metric_specs:
            path = plots_dir / f'{metric}_vs_features_haar_only.png'
            fig = plot_wavelet_comparison(
                haar_df.to_dict('records'),
                metric=metric,
                save_path=path,
                annotate=True,
                title=title,
                group_col=None,
            )
            if fig is not None:
                saved.append(path)
            plt.close('all')

            bar_path = plots_dir / f'{metric.replace("_mean", "")}_bar_haar_only.png'
            fig = plot_wavelet_config_bars(
                haar_df.to_dict('records'),
                metric=metric,
                save_path=bar_path,
                title=f'{title.split(" vs ")[0]} | {job_dir.name}',
            )
            if fig is not None:
                saved.append(bar_path)
            plt.close('all')

        time_path = plots_dir / 'training_time_vs_features_haar_only.png'
        fig = plot_feature_count_vs_time(
            haar_df,
            feature_col='n_features',
            time_col='total_time_seconds',
            save_path=time_path,
            xlabel='Число вейвлет-признаков',
            title=f'Время обучения vs число вейвлет-признаков (только Haar)\n{job_dir.name}',
        )
        if fig is not None:
            saved.append(time_path)
        plt.close('all')

    if all_haar_rows:
        all_df = pd.DataFrame(all_haar_rows)
        root_plots = run_dir / 'plots'
        root_plots.mkdir(parents=True, exist_ok=True)
        for metric, title in (
            ('r2_mean', 'R2 | только Haar'),
            ('mse_mean', 'MSE | только Haar'),
        ):
            path = root_plots / f'{metric.replace("_mean", "")}_heatmap_haar_only.png'
            fig = plot_wavelet_job_heatmap(all_df, metric=metric, save_path=path, title=title)
            if fig is not None:
                saved.append(path)
            plt.close('all')

        grid_path = root_plots / 'r2_mean_vs_features_haar_only_grid.png'
        fig = _plot_haar_r2_grid(all_df, save_path=grid_path)
        if fig is not None:
            saved.append(grid_path)
        plt.close('all')

    return saved


def _plot_haar_r2_grid(all_df: pd.DataFrame, save_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    datasets = ['old', 'train3', 'train3_1']
    targets = ['H1_8', 'H2_8', 'H3_8']
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=False)
    for i, dataset in enumerate(datasets):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            job = f'wavelet_{dataset}_{target}'
            sub = all_df[all_df['experiment_name'] == job].sort_values('n_features')
            if sub.empty:
                ax.set_axis_off()
                continue
            std_col = 'r2_std' if 'r2_std' in sub.columns else None
            ax.errorbar(
                sub['n_features'],
                sub['r2_mean'],
                yerr=sub[std_col] if std_col else None,
                marker='o',
                capsize=3,
                color='#3b6d9a',
            )
            for k, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    wavelet_config_short_label(row),
                    (row['n_features'], row['r2_mean']),
                    textcoords='offset points',
                    xytext=(6, 8 if k % 2 == 0 else -12),
                    fontsize=7,
                )
            ax.set_title(job.replace('wavelet_', ''), fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel('n признаков')
            if j == 0:
                ax.set_ylabel('R2')
    fig.suptitle('R2 vs число признаков | только Haar', fontsize=13)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
