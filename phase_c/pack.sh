#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bash phase_c/pack.sh OUTPUT_DIRECTORY" >&2
    exit 2
fi
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
target=$(realpath -m -- "$1")
mkdir -p -- "$target"
work=$(mktemp -d "$target/.pack.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
bundle="$work/lomap-phase-c"
mkdir -p -- "$bundle"/{bin,phase_c,phase_b/data}

cd -- "$project"
cp -- README.md LICENSE THIRD_PARTY_NOTICES.md "$bundle/"
cp -- phase_c/{README.md,preflight.sh,run.sh} "$bundle/phase_c/"
cp -- phase_b/{*.py,.python-version,pyproject.toml,uv.lock} "$bundle/phase_b/"
cp -- phase_b/data/maze2d-{medium,large}-sparse-v1.hdf5 "$bundle/phase_b/data/"
install -m 755 -- "$(command -v uv)" "$bundle/bin/uv"

upstream=phase_b/.upstream/lomap
revision=$(git -C "$upstream" rev-parse HEAD)
test "$revision" = 8ccb052481484352affaada26a0895102aef3e52
git -C "$upstream" bundle create "$bundle/upstream.bundle" HEAD
printf 'base_commit=%s\nupstream_commit=%s\n' "$(git rev-parse HEAD)" "$revision" > "$bundle/SOURCE.txt"
git diff --binary HEAD -- phase_b >"$bundle/phase-b.patch"
(
    cd -- "$bundle"
    find . -type f ! -name INPUT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > INPUT_SHA256SUMS
)
tar -C "$work" -cf "$work/lomap-phase-c.tar" lomap-phase-c
mv -- "$work/lomap-phase-c.tar" "$target/"
cd -- "$target"
sha256sum lomap-phase-c.tar >lomap-phase-c.tar.sha256
