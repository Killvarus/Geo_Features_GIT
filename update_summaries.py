"""
Обновление summary.json из Excel файлов

Запуск:
    python update_summaries.py                    # experiments/aggregation_study
    python update_summaries.py path/to/exp        # указанная директория
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.experiment_plots import extract_metrics_from_excel


def update_summaries(experiment_dir: Path):
    """Обновление всех summary.json из Excel файлов"""
    experiment_dir = Path(experiment_dir)
    
    updated = 0
    
    for summary_path in experiment_dir.glob('*/summary.json'):
        exp_dir = summary_path.parent
        excel_path = exp_dir / 'results' / 'metrics.xlsx'
        
        if not excel_path.exists():
            print(f"  [SKIP] {exp_dir.name}: нет metrics.xlsx")
            continue
        
        # Загружаем текущий summary
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Извлекаем метрики из Excel
        metrics = extract_metrics_from_excel(excel_path)
        
        if not metrics:
            print(f"  [SKIP] {exp_dir.name}: не удалось извлечь метрики")
            continue
        
        # Обновляем summary
        summary.update(metrics)
        
        # Сохраняем
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] {exp_dir.name}: R2={metrics.get('r2_mean', 0):.4f}")
        updated += 1
    
    print(f"\nОбновлено: {updated} файлов")
    return updated


def main():
    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])
    else:
        experiment_dir = Path('experiments/aggregation_study')
    
    if not experiment_dir.exists():
        print(f"Директория не найдена: {experiment_dir}")
        return
    
    print(f"Обновление summary.json в: {experiment_dir}")
    print("-" * 60)
    
    update_summaries(experiment_dir)


if __name__ == '__main__':
    main()
