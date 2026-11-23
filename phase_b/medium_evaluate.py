"""Measure paired Maze2D plan artifacts against the D4RL PointMaze wall boxes."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from maze2d import ENVIRONMENTS, MazeWindows, PointMaze, sha256
from medium_train import write_json
from smoke import CpuFaiss, MIB, make_diffuser


PAIRS = 20
DRAWS = 10
BANK_SIZE = 65536
NLIST = 256
NEIGHBORS = 10
TAU = 0.99
CONDITIONS = ("without_lomap", "with_lomap")


def wall_boxes(maze):
    particle = maze.model.geom("particle_geom").id
    origin = maze.data.geom_xpos[particle, :2] - maze.data.qpos[:2]
    walls = [i for i in range(maze.model.ngeom) if (maze.model.geom(i).name or "").startswith("wall_")]
    centers = maze.data.geom_xpos[walls, :2] - origin
    half_widths = maze.model.geom_size[walls, :2]
    return np.stack((centers - half_widths, centers + half_widths), axis=1), origin


def wall_segments(xy, boxes):
    """Flag open-box intersections on each line segment between successive states."""
    start = xy[..., :-1, None, :].astype(np.float64)
    end = xy[..., 1:, None, :].astype(np.float64)
    direction = end - start
    moving = direction != 0
    low = boxes[:, 0] + 1e-7
    high = boxes[:, 1] - 1e-7
    enter = np.divide(low - start, direction, out=np.full_like(start - low, -np.inf), where=moving)
    leave = np.divide(high - start, direction, out=np.full_like(start - high, np.inf), where=moving)
    intersects = ((moving | ((start > low) & (start < high))).all(axis=-1)
                  & (np.maximum(np.minimum(enter, leave).max(axis=-1), 0)
                     < np.minimum(np.maximum(enter, leave).min(axis=-1), 1)))
    return intersects.any(axis=-1)


def endpoint_pairs(dataset, maze, boxes, origin, rng, count):
    """Sample different free tiles independently, then real offline states in each tile."""
    obs = dataset.unnormalize(dataset.trajectories[:, :4])
    xy = obs[:, :2]
    grid = np.floor(xy + origin + 0.5).astype(np.int32)
    radius = maze.model.geom("particle_geom").size[0]
    candidates = []
    # Draw endpoints from corridor interiors, with clearance from adjacent walls.
    clear = (np.abs(xy + origin - grid) < 0.5 - radius - 0.01).all(axis=1)
    wall_grid = np.rint(boxes.mean(axis=1) + origin).astype(int)
    lower, upper = wall_grid.min(axis=0), wall_grid.max(axis=0)
    wall_tiles = set(map(tuple, wall_grid))
    for tile in sorted(map(tuple, np.unique(grid[clear], axis=0))):
        if tile not in wall_tiles and ((lower < tile) & (tile < upper)).all():
            ids = np.flatnonzero(clear & (grid == tile).all(axis=1))
            candidates.append(ids)
    ordered = [(i, j) for i in range(len(candidates)) for j in range(len(candidates)) if i != j]
    pair_ids = rng.choice(len(ordered), count, replace=False)
    return np.array([[rng.choice(candidates[i]), rng.choice(candidates[j])] for i, j in
                     (ordered[int(pair)] for pair in pair_ids)])


def evaluate(output, *, maze_name="medium", pairs=PAIRS, draws=DRAWS, bank_size=BANK_SIZE,
             nlist=NLIST, seed=None, gap_weight=0.0):
    training = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    checkpoint = output / "diffusion.pt"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != training["checkpoint_sha256"]:
        raise RuntimeError("Diffuser checkpoint hash differs from the training report")
    dataset = MazeWindows(maze_name)
    if training["dataset"] != dataset.describe():
        raise RuntimeError(f"{maze_name.capitalize()} dataset differs from training")
    maze = PointMaze(maze_name)
    boxes, origin = wall_boxes(maze)
    legacy_rng = seed is None
    seed = training["seed"] if legacy_rng else seed
    seed_sequence = np.random.SeedSequence([seed, 1])
    rng = np.random.default_rng(seed_sequence)
    # Preserve Phase B's existing stream for its original command. With an
    # explicit evaluation seed, extending the draw count also preserves the bank.
    reverse_rng, projection_rng, bank_rng, gap_rng = (
        (rng,) * 4 if legacy_rng else map(np.random.default_rng, seed_sequence.spawn(4))
    )
    endpoint_ids = endpoint_pairs(dataset, maze, boxes, origin, rng, pairs)
    endpoint_states = dataset.trajectories[endpoint_ids, :4]
    endpoint_xy = dataset.unnormalize(endpoint_states).reshape(-1, 1, 4)[..., :2]
    if wall_segments(np.repeat(endpoint_xy, 2, axis=1), boxes).any():
        raise RuntimeError("A selected endpoint is inside a wall")
    reverse_seeds = reverse_rng.integers(0, 2**32, size=draws, dtype=np.int64)
    projection_seeds = projection_rng.integers(0, 2**32, size=draws, dtype=np.int64)
    bank_ids = bank_rng.choice(len(dataset), bank_size, replace=False)
    gap_seeds = gap_rng.integers(0, 2**32, size=draws, dtype=np.int64)
    conditions = (*CONDITIONS, "online_gap") if gap_weight else CONDITIONS
    faiss.omp_set_num_threads(8)
    torch.set_num_threads(4)

    start = time.perf_counter()
    bank = dataset.batch(bank_ids)
    index = CpuFaiss(bank, NEIGHBORS, nlist)
    _, matches = index.search(bank[:1], NEIGHBORS)
    if 0 not in matches[0]:
        raise RuntimeError("CPU Faiss self-search failed")
    index.search_calls = 0
    print(f"{maze_name.capitalize()} CPU IVF-Flat ready: {bank_size} windows in {time.perf_counter() - start:.2f}s")
    index_seconds = time.perf_counter() - start

    settings = ENVIRONMENTS[maze_name]
    sampling_steps = settings["diffusion_steps"]
    projection_range = settings["projection"]
    agent = make_diffuser(maze_name)
    agent.load(str(checkpoint))
    agent.eval()
    prior = torch.zeros((pairs, dataset.horizon, 6), device="cuda:0")
    prior[:, [0, -1], :4] = torch.from_numpy(endpoint_states).to(prior.device)
    plans = np.empty((len(conditions), pairs, draws, dataset.horizon, 6), dtype=np.float32)
    sample_seconds = np.zeros(len(conditions))
    peak_memory = np.zeros(len(conditions))
    start = time.perf_counter()
    for draw, (reverse, projection) in enumerate(zip(reverse_seeds, projection_seeds, strict=True)):
        reverse_state = None
        for arm, label in enumerate(conditions):
            torch.manual_seed(int(reverse))
            agent.projection_rng = torch.Generator(device="cuda:0").manual_seed(int(projection))
            agent.gap_rng = torch.Generator(device="cuda:0").manual_seed(int(gap_seeds[draw]))
            project = label == "with_lomap"
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            with torch.no_grad():
                samples, _ = agent.sample(
                    prior, solver="ddpm", n_samples=pairs, sample_steps=sampling_steps,
                    w_cg=0.0, faiss_wrapper=index if project else None,
                    proj_range=projection_range if project else [],
                    n_manifold_samples=NEIGHBORS, tau=TAU,
                    w_ldg=gap_weight if label == "online_gap" else 0.0,
                )
            torch.cuda.synchronize()
            sample_seconds[arm] += time.perf_counter() - began
            peak_memory[arm] = max(peak_memory[arm], torch.cuda.max_memory_allocated() / MIB)
            after = torch.cuda.get_rng_state()
            if arm == 0:
                reverse_state = after
            elif not torch.equal(after, reverse_state):
                raise RuntimeError(f"{label}: paired DDPM random stream diverged for draw {draw + 1}")
            if not torch.isfinite(samples).all():
                raise RuntimeError(f"{label}: non-finite samples in draw {draw}")
            torch.testing.assert_close(samples[:, [0, -1], :4], prior[:, [0, -1], :4], rtol=0, atol=0)
            plans[arm, :, draw] = samples.cpu().numpy()
        print(f"{maze_name.capitalize()} paired sampling {draw + 1}/{draws}: "
              f"{index.search_calls} cumulative Faiss calls")

    expected_searches = draws * (int(projection_range[1] * (sampling_steps - 1))
                                 - int(projection_range[0] * (sampling_steps - 1)) + 1)
    if index.search_calls != expected_searches or np.array_equal(plans[0], plans[1]):
        raise RuntimeError(f"Unexpected LoMAP activity: {index.search_calls} Faiss calls, expected {expected_searches}")
    if gap_weight and np.array_equal(plans[0], plans[2]):
        raise RuntimeError("Online restoration-gap guidance did not change the samples")
    sampling_wall = time.perf_counter() - start
    observations = dataset.unnormalize(plans[..., :4])
    crossing = np.stack([wall_segments(observations[:, pair, :, :, :2], boxes) for pair in range(pairs)], axis=1)
    artifact = crossing.any(axis=-1)
    cumulative = np.maximum.accumulate(artifact, axis=-1)
    prefix = cumulative.sum(axis=1)
    transition_count = dict(
        improved=int((artifact[0] & ~artifact[1]).sum()),
        regressed=int((~artifact[0] & artifact[1]).sum()),
        both_cross=int((artifact[0] & artifact[1]).sum()),
        neither_cross=int((~artifact[0] & ~artifact[1]).sum()),
    )
    lows, highs = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
    outside = ((observations[..., :2] < lows) | (observations[..., :2] > highs)).any(axis=(-2, -1))
    temporary = output / f"{maze_name}_samples.tmp.npz"
    np.savez_compressed(temporary, conditions=np.array(conditions), normalized_plans=plans,
                        observations=observations, wall_boxes=boxes, particle_origin=origin,
                        crossings=crossing, artifact=artifact, outside=outside, endpoint_ids=endpoint_ids,
                        reverse_seeds=reverse_seeds, projection_seeds=projection_seeds, bank_ids=bank_ids,
                        gap_seeds=gap_seeds)
    temporary.replace(output / f"{maze_name}_samples.npz")
    with (output / f"{maze_name}_crossings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "draw", "start_transition", "goal_transition", "reverse_seed",
                         *conditions, *(f"outside_{label}" for label in conditions)])
        for pair, (start_id, goal_id) in enumerate(endpoint_ids):
            for draw, reverse_seed in enumerate(reverse_seeds):
                writer.writerow([pair + 1, draw + 1, start_id, goal_id, reverse_seed,
                                 *artifact[:, pair, draw].astype(int), *outside[:, pair, draw].astype(int)])
    write_json(output / f"{maze_name}_report.json", {
        "maze": maze_name,
        "checkpoint_sha256": checkpoint_hash,
        "training_steps": training["completed_steps"],
        "evaluation_seed": seed,
        "pairs": pairs,
        "draws_per_pair": draws,
        "steps_per_sample": sampling_steps,
        "projection_range": projection_range,
        "tau": TAU,
        "bank_windows": bank_size,
        "neighbors": NEIGHBORS,
        "faiss_nlist": nlist,
        "faiss_nprobe": index.index.nprobe,
        "faiss_version": faiss.__version__,
        "faiss_search_calls_with_lomap": index.search_calls,
        "index_build_seconds": index_seconds,
        "sampling_wall_seconds": sampling_wall,
        "same_reverse_rng": True,
        "exact_conditioned_endpoints": True,
        "online_gap": {"weight": gap_weight, "forward_level": 0.8, "mc_samples": 1} if gap_weight else None,
        "artifact_rule": "Any straight segment between successive xy states intersects the interior of a wall box.",
        "paired_plan_outcomes": transition_count,
        "conditions": {
            label: {
                "crossing_plans": int(artifact[arm].sum()),
                "plan_crossing_rate": float(artifact[arm].mean()),
                "affected_pairs_by_draws": prefix[arm].tolist(),
                "pair_artifact_rate_by_draws": (prefix[arm] / pairs).tolist(),
                **({"affected_pairs_at_10_draws": int(prefix[arm, 9]),
                    "pair_artifact_rate_at_10_draws": float(prefix[arm, 9] / pairs)} if draws >= 10 else {}),
                "outside_plans": int(outside[arm].sum()),
                "sampling_seconds": float(sample_seconds[arm]),
                "peak_allocated_mib": float(peak_memory[arm]),
            } for arm, label in enumerate(conditions)
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maze", choices=ENVIRONMENTS, default="medium")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--bank-size", type=int, default=BANK_SIZE)
    parser.add_argument("--nlist", type=int, default=NLIST)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--online-gap-weight", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.pairs, args.draws, args.nlist) < 1 or args.bank_size < max(args.nlist, NEIGHBORS):
        parser.error("pairs, draws and nlist must be positive, and the bank must fit nlist and neighbors")
    if (args.seed is not None and args.seed < 0) or not math.isfinite(args.online_gap_weight) or args.online_gap_weight < 0:
        parser.error("seed and online gap weight must be nonnegative, and the weight must be finite")
    evaluate(args.output, maze_name=args.maze, pairs=args.pairs, draws=args.draws, bank_size=args.bank_size,
             nlist=args.nlist, seed=args.seed, gap_weight=args.online_gap_weight)
