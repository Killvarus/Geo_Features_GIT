#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

[[ -f experiments/6p_full_patience300/xy_cv1.npz ]] || {
    printf '%s\n' 'Не найден кэш спектроскопии CV1.' >&2
    exit 1
}
command -v flock >/dev/null
mkdir -p "$HOME/6p_max_center_results" logs/6p_max_center_followup
exec 9>logs/6p_max_center_followup/queue.lock
flock -n 9 || { printf '%s\n' 'Очередь уже работает.' >&2; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

run_stage() {
    local stage=$1 archive=$2
    printf '%s: начало %s\n' "$(date -Is)" "$stage"
    printf '%s\n' Cu Ni Al Co Cr NO3 | xargs -r -n 1 -P 6 bash -c '
        stage=$1
        ion=$2
        python run_6p_max_center_followup.py "$stage" "$ion" >>"logs/6p_max_center_followup/${stage}_${ion}.log" 2>&1
    ' _ "$stage"
    tar -czf "$archive.tmp" "experiments/6p_max_center_followup/$stage"
    mv -- "$archive.tmp" "$archive"
    printf '%s: архив готов: %s\n' "$(date -Is)" "$archive"
}

run_stage rfe "$HOME/6p_max_center_results/rfe_max_center.tar.gz"
run_stage bsbs_step1 "$HOME/6p_max_center_results/bsbs_step1_max_center.tar.gz"
