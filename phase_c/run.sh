#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || ( $1 != smoke && $1 != formal ) || ( $2 != medium && $2 != large ) ]]; then
    echo "Usage: bash phase_c/run.sh {smoke|formal} {medium|large} RUN_NAME" >&2
    exit 2
fi
profile=$1
maze=$2
run_name=$3
if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_NAME must be one directory name" >&2
    exit 2
fi

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
storage=${LOMAP_STORAGE_ROOT:-/root/shared-nvme/lomap}
python=${LOMAP_PYTHON:-$project/phase_b/.venv/bin/python}
if [[ -z ${LOMAP_STORAGE_ROOT:-} ]]; then
    findmnt --mountpoint /root/shared-nvme >/dev/null
fi
storage=$(realpath -m -- "$storage")
output=$(realpath -m -- "$storage/runs/$run_name")
if [[ $output != "$storage/runs/$run_name" ]]; then
    echo "Run directory redirected outside its expected path" >&2
    exit 2
fi
mkdir -p -- "$output"
exec 9>"$output/.run.lock"
flock -n 9 || { echo "Run already active: $output" >&2; exit 1; }
exec >>"$output/job.log" 2>&1

if [[ $profile == smoke ]]; then
    steps=1000
    save_every=100
    pairs=4
    draws=3
    bank=8192
    nlist=64
else
    steps=1000000
    save_every=5000
    pairs=20
    draws=30
    bank=1000000
    nlist=1000
fi
train_seed=20261120
eval_seed=20261123
gap_weight=${LOMAP_ONLINE_GAP_WEIGHT:-0}
stage=setup
record_status() {
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$stage" "$1" >"$output/status.tsv.tmp"
    mv -f -- "$output/status.tsv.tmp" "$output/status.tsv"
}
finish() {
    local rc=$?
    rm -f -- "$output/training.pid"
    record_status "$rc"
    printf 'Finished stage %s with exit %s\n' "$stage" "$rc"
}
trap finish EXIT
record_status running

cd -- "$project/phase_b"
export MPLCONFIGDIR="$storage/.cache/matplotlib"
report_config() {
    printf 'profile=%s\nmaze=%s\nsteps=%s\nbatch_size=64\nsave_every=%s\ntrain_seed=%s\neval_seed=%s\n' \
        "$profile" "$maze" "$steps" "$save_every" "$train_seed" "$eval_seed"
    printf 'pairs=%s\ndraws=%s\nbank=%s\nnlist=%s\nonline_gap_weight=%s\n' "$pairs" "$draws" "$bank" "$nlist" "$gap_weight"
    sha256sum ../phase_c/run.sh .python-version uv.lock *.py
}
if [[ -f $output/run_config.txt ]] && ! cmp -s "$output/run_config.txt" <(report_config); then
    echo "Run settings or source files have changed. Use the original snapshot or a new run directory."
    exit 2
fi
report_config >"$output/run_config.txt"

stage=training
record_status running
"$python" -u medium_train.py --maze "$maze" --output "$output" --steps "$steps" --batch-size 64 \
    --seed "$train_seed" --save-every "$save_every" &
training_pid=$!
printf '%s\n' "$training_pid" >"$output/training.pid"
wait "$training_pid"
rm -f -- "$output/training.pid"

stage=evaluation
record_status running
"$python" -u medium_evaluate.py --maze "$maze" --output "$output" --pairs "$pairs" --draws "$draws" \
    --bank-size "$bank" --nlist "$nlist" --seed "$eval_seed" --online-gap-weight "$gap_weight"
"$python" -u medium_plot.py --maze "$maze" --output "$output"

stage=verification
record_status running
(
    cd -- "$output"
    sha256sum diffusion.pt training_state.pt training.csv training_config.json training_report.json run_config.txt \
        "${maze}"_{report.json,crossings.csv,samples.npz,metrics.png,trajectories.png} > SHA256SUMS.tmp
    mv -f -- SHA256SUMS.tmp SHA256SUMS
)
stage=complete
