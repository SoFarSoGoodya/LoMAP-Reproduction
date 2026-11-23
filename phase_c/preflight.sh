#!/usr/bin/env bash
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
storage=${LOMAP_STORAGE_ROOT:-/root/shared-nvme/lomap}
if [[ -z ${LOMAP_STORAGE_ROOT:-} ]]; then
    findmnt --mountpoint /root/shared-nvme
fi
mkdir -p -- "$storage"
df -hT / "$storage"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cd -- "$project"
sha256sum --quiet -c INPUT_SHA256SUMS
revision=8ccb052481484352affaada26a0895102aef3e52
upstream="$project/phase_b/.upstream/lomap"
if [[ ! -d $upstream/.git ]]; then
    temporary=$(mktemp -d "$project/.upstream.XXXXXX")
    trap 'rm -rf -- "$temporary"' EXIT
    git clone --quiet --no-checkout "$project/upstream.bundle" "$temporary/lomap"
    git -C "$temporary/lomap" checkout --quiet --detach "$revision"
    mkdir -p -- "$(dirname -- "$upstream")"
    mv -T -- "$temporary/lomap" "$upstream"
fi
test "$(git -C "$upstream" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$upstream" status --porcelain --untracked-files=no)"

uv="$project/bin/uv"
cd -- "$project/phase_b"
if [[ -n ${LOMAP_OFFLINE_ENV:-} ]]; then
    python=$(find "$LOMAP_OFFLINE_ENV/python" -path '*/bin/python3.13' -print -quit)
    test -x "$python"
    UV_CACHE_DIR="$LOMAP_OFFLINE_ENV/cache" "$uv" sync --locked --offline --no-config --python "$python"
else
    "$uv" sync --locked --no-config
fi
.venv/bin/python - <<'PY'
import numpy as np
import torch

from maze2d import MazeWindows, PointMaze
from prepare_assets import main
from smoke import make_diffuser

main(("medium", "large"))
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name()}")
assert torch.tensor([2.0], device="cuda:0").square().item() == 4.0
for name in ("medium", "large"):
    dataset = MazeWindows(name)
    maze = PointMaze(name)
    observation, reward = maze.step(np.array([0.25, 0.0]))
    assert np.isfinite(observation).all() and reward == 1.0
    agent = make_diffuser(name)
    assert agent.fix_mask.shape == (1, dataset.horizon, 6)
    print(name, dataset.describe())
PY
