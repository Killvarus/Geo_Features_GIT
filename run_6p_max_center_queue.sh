#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin=${1:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
python_bin=${python_bin:-python}
mode=${2:-original}
workers=${3:-6}
[[ "$mode" == original || "$mode" == max_only ]] || { printf '%s\n' 'Ожидается режим original или max_only.' >&2; exit 2; }
[[ "$workers" =~ ^[1-6]$ ]] || { printf '%s\n' 'Число рабочих процессов должно быть от 1 до 6.' >&2; exit 2; }

[[ -f experiments/6p_full_patience300/xy_cv1.npz ]] || {
    printf '%s\n' 'Не найден кэш спектроскопии CV1.' >&2
    exit 1
}
command -v flock >/dev/null
mkdir -p "$HOME/6p_max_center_results" logs/6p_max_center_followup
lock_file=queue.lock
if [[ "$mode" == max_only ]]; then
    lock_file=queue_max_only.lock
fi
exec 9>"logs/6p_max_center_followup/$lock_file"
flock -n 9 || { printf '%s\n' 'Очередь уже работает.' >&2; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

printf 'Python: %s\n' "$(command -v -- "$python_bin")"
if ! "$python_bin" run_6p_max_center_followup.py --help >/dev/null; then
    printf 'Проверьте окружение Python: %s\n' "$python_bin" >&2
    exit 1
fi

run_stage() {
    local stage=$1 archive=$2
    printf '%s: начало %s\n' "$(date -Is)" "$stage"
    printf '%s\n' Cu Ni Al Co Cr NO3 | xargs -r -n 1 -P "$workers" bash -c '
        python_bin=$1
        stage=$2
        ion=$3
        "$python_bin" run_6p_max_center_followup.py "$stage" "$ion" >>"logs/6p_max_center_followup/${stage}_${ion}.log" 2>&1
    ' _ "$python_bin" "$stage"
    tar -czf "$archive.tmp" "experiments/6p_max_center_followup/$stage"
    mv -- "$archive.tmp" "$archive"
    printf '%s: архив готов: %s\n' "$(date -Is)" "$archive"
}

if [[ "$mode" == max_only ]]; then
    run_stage bsbs_max_only "$HOME/6p_max_center_results/bsbs_max_only.tar.gz"
else
    run_stage rfe "$HOME/6p_max_center_results/rfe_max_center.tar.gz"
    run_stage bsbs_step1 "$HOME/6p_max_center_results/bsbs_step1_max_center.tar.gz"
fi
