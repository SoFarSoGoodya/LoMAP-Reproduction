"""Plot paired Maze2D Figure 3 arms and an online minority-guidance comparison."""

import argparse
import json
import os
from pathlib import Path
import sys

import faiss
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

PHASE_B = Path(os.environ.get("LOMAP_PHASE_B", Path(__file__).resolve().parents[1] / "phase_b"))
sys.path.insert(0, str(PHASE_B))
from maze2d import ENVIRONMENTS, MazeWindows, PointMaze, sha256
from medium_evaluate import wall_boxes, wall_segments
from medium_train import write_json
from smoke import make_diffuser
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexIVFWrapper

ENDPOINTS = {
    "medium": ((1, 1), (6, 6)),
    "large": ((1, 1), (7, 9)),
}
ARMS = (
    ("Diffuser", False, False),
    ("Diffuser + LoMAP", True, False),
    ("Diffuser + online minority", False, True),
    ("Diffuser + online minority + LoMAP", True, True),
)


def draw_panels(output, name, labels, plans, boxes, endpoints, artifact):
    fig, axes = plt.subplots(1, len(labels), squeeze=False, figsize=(4.2 * len(labels), 4.2),
                             layout="constrained")
    lower, upper = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    for label, trajectories, crossings, ax in zip(labels, plans, artifact, axes[0], strict=True):
        for lo, hi in boxes:
            ax.add_patch(Rectangle(lo, *(hi - lo), color="#ced8dc"))
        for xy in trajectories:
            ax.plot(*xy.T, linewidth=0.65, color="#23758d", alpha=0.18)
        ax.scatter(*endpoints[0], color="#26253b", s=25, zorder=4)
        ax.scatter(*endpoints[-1], color="#b65b23", marker="*", s=70, zorder=4)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xticks=[], yticks=[], aspect="equal",
               title=f"{label}\n{crossings.sum()} / {len(crossings)} plans crossed")
    fig.savefig(output / name, dpi=180)
    plt.close(fig)


def evaluate(args):
    dataset = MazeWindows(args.maze)
    settings = ENVIRONMENTS[args.maze]
    boxes, origin = wall_boxes(PointMaze(args.maze))
    endpoints = np.asarray(ENDPOINTS[args.maze], dtype=np.float32)
    static = np.repeat(endpoints[:, None, :], 2, axis=1)
    if wall_segments(static, boxes).any():
        raise ValueError("A plotted endpoint is inside a wall")
    states = np.column_stack((endpoints, np.zeros((2, 2), dtype=np.float32)))
    checkpoint = args.source / "diffusion.pt"
    report = json.loads((args.source / "training_report.json").read_text())
    if sha256(checkpoint) != report["checkpoint_sha256"] or dataset.describe() != report["dataset"]:
        raise ValueError("Training data or checkpoint fingerprint mismatch")
    rng = np.random.default_rng(args.seed)
    bank_ids = rng.choice(len(dataset), min(args.bank_size, len(dataset)), replace=False)
    batch_seeds = rng.integers(0, 2**32, size=(args.draws + args.batch_size - 1) // args.batch_size)
    weights = np.ones((dataset.horizon, 6), dtype=np.float32)
    weights[:, 4:] = 0
    weights[[0, -1], :2] = 100
    faiss.omp_set_num_threads(8)
    torch.set_num_threads(4)
    index = FaissIndexIVFWrapper("cosine", args.nlist, dataset.batch(bank_ids), weights, device="cuda:0")
    agent = make_diffuser(args.maze)
    agent.load(str(checkpoint))
    agent.eval()
    plans = np.empty((len(ARMS), args.draws, dataset.horizon, 4), dtype=np.float32)
    for batch, first in enumerate(range(0, args.draws, args.batch_size)):
        end = min(first + args.batch_size, args.draws)
        prior = torch.zeros((end - first, dataset.horizon, 6), device="cuda:0")
        prior[:, [0, -1], :4] = torch.from_numpy(dataset.normalize(states)).to(prior.device)
        reverse_state = None
        seed = int(batch_seeds[batch])
        for arm, (_, use_projection, use_minority) in enumerate(ARMS):
            torch.manual_seed(seed)
            agent.projection_rng = torch.Generator(device="cuda:0").manual_seed(seed + 1)
            agent.gap_rng = torch.Generator(device="cuda:0").manual_seed(seed + 2)
            with torch.no_grad():
                sample, _ = agent.sample(
                    prior, solver="ddpm", n_samples=end - first, sample_steps=settings["diffusion_steps"],
                    w_cg=0.0, w_ldg=args.minority_weight if use_minority else 0.0,
                    faiss_wrapper=index if use_projection else None,
                    proj_range=settings["projection"] if use_projection else [], n_manifold_samples=10, tau=0.99,
                )
            torch.testing.assert_close(sample[:, [0, -1], :4], prior[:, [0, -1], :4], rtol=0, atol=0)
            if not torch.isfinite(sample).all():
                raise RuntimeError("Non-finite trajectory sample")
            after = torch.cuda.get_rng_state()
            if arm == 0:
                reverse_state = after
            elif not torch.equal(after, reverse_state):
                raise RuntimeError("Paired reverse DDPM streams diverged")
            plans[arm, first:end] = dataset.unnormalize(sample[..., :4].cpu().numpy())
        print(f"{args.maze}: {end}/{args.draws} draws per arm", flush=True)
    xy = plans[..., :2]
    artifact = np.stack([wall_segments(arm, boxes).any(axis=-1) for arm in xy])
    labels = [name for name, _, _ in ARMS]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "samples.npz", observations=plans, artifact=artifact, labels=labels,
                        endpoints=endpoints, particle_origin=origin, wall_boxes=boxes, batch_seeds=batch_seeds,
                        bank_ids=bank_ids)
    for filename, ids in (("figure3-two-arms.png", [0, 1]), ("figure5-online-minority.png", [0, 2, 3])):
        draw_panels(args.output, filename, [labels[i] for i in ids], xy[ids], boxes, endpoints, artifact[ids])
    write_json(args.output / "report.json", {
        "maze": args.maze, "checkpoint_sha256": report["checkpoint_sha256"], "training_steps": report["completed_steps"],
        "draws": args.draws, "seed": args.seed, "endpoints": endpoints.tolist(), "bank_windows": len(bank_ids),
        "online_minority_weight": args.minority_weight, "faiss_nlist": args.nlist,
        "crossing_plans": dict(zip(labels, artifact.sum(axis=1).tolist(), strict=True)),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", choices=ENDPOINTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--bank-size", type=int, default=1000000)
    parser.add_argument("--nlist", type=int, default=1000)
    parser.add_argument("--minority-weight", type=float, default=0.1)
    evaluate(parser.parse_args())
