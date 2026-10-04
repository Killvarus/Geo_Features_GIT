"""
Достраивает отсутствующие графики по всем экспериментам в папке experiments/.

Логика:
- проходит по каждой подпапке experiments/*
- пытается загрузить результаты (summary.json или fallback из metrics.xlsx)
- определяет тип эксперимента (aggregation / pca)
- строит только те графики, которых ещё нет
- при отсутствии PNG кривой перерисовывает её из learning_history_iter_*.json (полный ряд эпох, не лог)

Запуск:
    python plot_missing_graphs.py
    python plot_missing_graphs.py /path/to/experiments
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.experiment_plots import (
    load_all_summaries,
    generate_all_plots,
)
from src.evaluation.metrics import (
    plot_feature_count_vs_time,
    plot_relative_training_time,
)
from src.preprocessing.pca import plot_pca_comparison


def _safe_close(fig):
    if fig is not None:
        plt.close(fig)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _is_aggregation_df(df: pd.DataFrame) -> bool:
    return {'freq_agg_step', 'pickup_agg_step', 'agg_method'}.issubset(df.columns)


def _is_pca_df(df: pd.DataFrame) -> bool:
    return ('n_components' in df.columns) or (
        'experiment_type' in df.columns and (df['experiment_type'].astype(str).str.lower() == 'pca').any()
    )


def _build_missing_pca_plots(exp_dir: Path, df: pd.DataFrame) -> int:
    created = 0
    plots_dir = exp_dir / 'plots'
    _ensure_dir(plots_dir)

    required = {
        'r2': plots_dir / 'r2_mean_vs_components.png',
        'mse': plots_dir / 'mse_mean_vs_components.png',
        'time_abs': plots_dir / 'training_time_vs_components.png',
        'time_rel': plots_dir / 'relative_training_time_vs_components.png',
    }

    records = df.to_dict('records')

    if not required['r2'].exists():
        fig = plot_pca_comparison(records, metric='r2_mean', save_path=required['r2'])
        _safe_close(fig)
        created += 1

    if not required['mse'].exists():
        fig = plot_pca_comparison(records, metric='mse_mean', save_path=required['mse'])
        _safe_close(fig)
        created += 1

    if not required['time_abs'].exists() and {'n_components', 'total_time_seconds'}.issubset(df.columns):
        fig = plot_feature_count_vs_time(
            df,
            feature_col='n_components',
            time_col='total_time_seconds',
            save_path=required['time_abs'],
            xlabel='Количество компонент',
            title='Время обучения vs количество компонент',
        )
        _safe_close(fig)
        created += 1

    if not required['time_rel'].exists() and {'n_components', 'total_time_seconds'}.issubset(df.columns):
        ref = None
        if 'original_n_features' in df.columns:
            vals = pd.to_numeric(df['original_n_features'], errors='coerce').dropna()
            if not vals.empty:
                ref = int(vals.iloc[0])

        if ref is None:
            ref = int(pd.to_numeric(df['n_components'], errors='coerce').max())

        fig = plot_relative_training_time(
            df,
            feature_col='n_components',
            time_col='total_time_seconds',
            reference_feature_count=ref,
            save_path=required['time_rel'],
            xlabel='Количество компонент',
            title='Относительное время обучения vs количество компонент',
        )
        _safe_close(fig)
        created += 1

    return created


def _restore_missing_learning_curves(exp_dir: Path) -> int:
    """Достраивает PNG только из learning_history_iter_*.json (полный ряд эпох).

    Из training.log кривые больше не собираются: там каждая 100-я эпоха и чужие прогоны.
    """
    from src.models.neural_network import plot_learning_curve_from_history

    created = 0
    for config_dir in [p for p in exp_dir.iterdir() if p.is_dir()]:
        curves_dir = config_dir / 'learning_curves'
        if not curves_dir.exists():
            continue

        for history_path in sorted(curves_dir.glob('learning_history_iter_*.json')):
            out_path = history_path.with_name(
                history_path.name.replace('learning_history', 'learning_curve').replace('.json', '.png')
            )
            if out_path.exists():
                continue
            try:
                plot_learning_curve_from_history(history_path, out_path)
                created += 1
            except Exception as e:
                print(f"[WARN] не удалось восстановить кривую из {history_path}: {e}")

    return created


def process_experiment(exp_dir: Path) -> Dict[str, int]:
    stats = {'created': 0, 'skipped': 0, 'errors': 0}

    try:
        df = load_all_summaries(exp_dir)
        if df.empty:
            stats['skipped'] += 1
            return stats

        if _is_aggregation_df(df):
            plots_dir = exp_dir / 'plots'
            needed = [
                plots_dir / 'r2_vs_aggregation.png',
                plots_dir / 'r2_heatmap.png',
                plots_dir / 'all_experiments_r2.png',
                plots_dir / 'features_vs_r2.png',
                plots_dir / 'features_vs_r2_detailed.png',
            ]

            if any(not p.exists() for p in needed):
                figs = generate_all_plots(exp_dir, output_dir=plots_dir, metrics=['r2', 'mse', 'mae'])
                for fig in figs.values():
                    _safe_close(fig)
                stats['created'] += sum(1 for p in needed if p.exists())
            else:
                stats['skipped'] += 1

        elif _is_pca_df(df):
            stats['created'] += _build_missing_pca_plots(exp_dir, df)

        # Перерисовать пропавшие PNG из JSON-истории (не из training.log)
        stats['created'] += _restore_missing_learning_curves(exp_dir)

    except Exception as e:
        print(f"[ERROR] {exp_dir}: {e}")
        stats['errors'] += 1

    return stats


def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('experiments')
    if not base_dir.exists():
        print(f"Директория не найдена: {base_dir}")
        return

    exp_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not exp_dirs:
        print(f"Нет экспериментов в {base_dir}")
        return

    total = {'created': 0, 'skipped': 0, 'errors': 0}

    print(f"Найдено экспериментов: {len(exp_dirs)}")
    for exp_dir in exp_dirs:
        s = process_experiment(exp_dir)
        total['created'] += s['created']
        total['skipped'] += s['skipped']
        total['errors'] += s['errors']
        print(f"- {exp_dir.name}: created={s['created']}, skipped={s['skipped']}, errors={s['errors']}")

    print("\nИТОГО:")
    print(f"created={total['created']}, skipped={total['skipped']}, errors={total['errors']}")


if __name__ == '__main__':
    main()
