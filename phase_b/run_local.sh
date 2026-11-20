#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project=$PWD
stage=${1:-}
case "$stage" in
    env|umaze|medium) shift ;;
    *) printf 'Usage: bash run_local.sh {env|umaze|medium} [experiment arguments...]\n' >&2; exit 2 ;;
esac

if [[ ${LOMAP_LOCAL_WORKER:-0} != 1 ]]; then
    run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    run_dir="$project/runs/$stage-$run_id"
    if [[ $stage == medium && ${1:-} == --resume ]]; then
        run_dir=$(realpath -e -- "${2:?Specify the Medium run directory to resume}")
        shift 2
        if [[ $(dirname -- "$run_dir") != "$project/runs" || $(basename -- "$run_dir") != medium-* ]]; then
            printf 'Resume requires an existing Medium run directory inside %s/runs\n' "$project" >&2
            exit 2
        fi
        if systemctl --user is-active --quiet "$(cat "$run_dir/unit.txt")"; then
            printf 'This run is active\n' >&2
            exit 2
        fi
    fi
    mkdir -p -- "$run_dir"
    unit="lomap-b-$stage-$run_id"
    printf '%s\n' "$unit" > "$run_dir/unit.txt"
    systemd-run --user --unit="$unit" --working-directory="$project" \
        --property="StandardOutput=append:$run_dir/run.log" \
        --property="StandardError=inherit" \
        --setenv=LOMAP_LOCAL_WORKER=1 --setenv="LOMAP_RUN_DIR=$run_dir" \
        --setenv="LOMAP_RUN_ID=$run_id" \
        --setenv="UV_CACHE_DIR=${UV_CACHE_DIR:-$project/.cache/uv}" \
        bash "$project/run_local.sh" "$stage" "$@"
    printf '%s\n' "$run_dir"
    exit
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$project/.cache/uv}"
export PYTHONUNBUFFERED=1
export UV_CONCURRENT_DOWNLOADS=2
export UV_HTTP_RETRIES=5
export UV_HTTP_TIMEOUT=120

monitor_pid=
finish() {
    code=$?
    if [[ -n $monitor_pid ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$code" > "$LOMAP_RUN_DIR/exit_code.tmp"
    mv -- "$LOMAP_RUN_DIR/exit_code.tmp" "$LOMAP_RUN_DIR/exit_code"
    printf 'Finished: %s (exit %s)\n' "$(date -Is)" "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

uv_bin=$(command -v uv || true)
if [[ -z $uv_bin ]]; then
    printf 'uv is required; install it and make it available on PATH\n' >&2
    exit 127
fi
exec 9>"$project/runs/job.lock"
flock 9
printf 'Started: %s\n' "$(date -Is)"

if [[ $stage != env ]]; then
    if command -v nvidia-smi >/dev/null; then
        gpu_log="$LOMAP_RUN_DIR/gpu.csv"
        if [[ -s $gpu_log ]]; then
            gpu_log="$LOMAP_RUN_DIR/gpu-${LOMAP_RUN_ID}.csv"
        fi
        nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
            --format=csv,nounits --loop-ms=200 > "$gpu_log" &
        monitor_pid=$!
    fi
fi

rm -f -- "$LOMAP_RUN_DIR/exit_code"
case "$stage" in
    env)
        "$uv_bin" sync --locked
        ;;
    umaze)
        "$uv_bin" run --locked python smoke.py --output "$LOMAP_RUN_DIR" "$@"
        ;;
    medium)
        "$uv_bin" run --locked python medium_train.py --output "$LOMAP_RUN_DIR" "$@"
        "$uv_bin" run --locked python medium_evaluate.py --output "$LOMAP_RUN_DIR"
        "$uv_bin" run --locked python medium_plot.py --output "$LOMAP_RUN_DIR"
        ;;
esac
