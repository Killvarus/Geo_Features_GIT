#!/usr/bin/env bash
set -euo pipefail

code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$code_dir
if [[ ! -d "$project/experiments" ]]; then
    project=$(dirname -- "$code_dir")
fi
cd -- "$project"

python_bin=${1:?Укажи абсолютный путь к Python из geoVenv}
shift
for option in "$@"; do
    if [[ "$option" == --project || "$option" == --project=* ]]; then
        printf '%s\n' 'Корень проекта определяется по расположению пакета.' >&2
        exit 2
    fi
done

archive=6p_normalization_followup_results.tar.gz
for option in "$@"; do
    if [[ "$option" == --screen-only ]]; then
        archive=6p_normalization_followup_cv1.tar.gz
    fi
done

mkdir -p logs/6p_normalization_followup
exec 9>logs/6p_normalization_followup/queue.lock
flock -n 9 || { printf '%s\n' 'Очередь уже работает.' >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

printf '%s: начало сетки 6P\n' "$(date -Is)"
"$python_bin" "$code_dir/run_6p_normalization_followup.py" --project "$project" "$@"
tar -czf "$archive.tmp" experiments/6p_normalization_followup
mv -- "$archive.tmp" "$archive"
printf '%s: архив готов: %s/%s\n' "$(date -Is)" "$project" "$archive"
