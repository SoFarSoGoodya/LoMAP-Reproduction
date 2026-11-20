"""Measure paired Medium plan artifacts against the D4RL PointMaze wall boxes."""

import argparse
import csv
import json
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
    wall_tiles = {tuple(tile) for tile in np.rint(boxes.mean(axis=1) + origin).astype(int)}
    for tile in sorted(map(tuple, np.unique(grid[clear], axis=0))):
        if tile not in wall_tiles and all(1 < coordinate < 8 for coordinate in tile):
            ids = np.flatnonzero(clear & (grid == tile).all(axis=1))
            candidates.append(ids)
    ordered = [(i, j) for i in range(len(candidates)) for j in range(len(candidates)) if i != j]
    pair_ids = rng.choice(len(ordered), count, replace=False)
    return np.array([[rng.choice(candidates[i]), rng.choice(candidates[j])] for i, j in
                     (ordered[int(pair)] for pair in pair_ids)])


def evaluate(output):
    training = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    checkpoint = output / "diffusion.pt"
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != training["checkpoint_sha256"]:
        raise RuntimeError("Diffuser checkpoint hash differs from the training report")
    dataset = MazeWindows("medium")
    if training["dataset"] != dataset.describe():
        raise RuntimeError("Medium dataset differs from training")
    maze = PointMaze("medium")
    boxes, origin = wall_boxes(maze)
    rng = np.random.default_rng(np.random.SeedSequence([training["seed"], 1]))
    endpoint_ids = endpoint_pairs(dataset, maze, boxes, origin, rng, PAIRS)
    endpoint_states = dataset.trajectories[endpoint_ids, :4]
    endpoint_xy = dataset.unnormalize(endpoint_states).reshape(-1, 1, 4)[..., :2]
    if wall_segments(np.repeat(endpoint_xy, 2, axis=1), boxes).any():
        raise RuntimeError("A selected endpoint is inside a wall")
    reverse_seeds = rng.integers(0, 2**32, size=DRAWS, dtype=np.int64)
    projection_seeds = rng.integers(0, 2**32, size=DRAWS, dtype=np.int64)
    bank_ids = rng.choice(len(dataset), BANK_SIZE, replace=False)
    faiss.omp_set_num_threads(8)
    torch.set_num_threads(4)

    start = time.perf_counter()
    bank = dataset.batch(bank_ids)
    index = CpuFaiss(bank, NEIGHBORS, NLIST)
    _, matches = index.search(bank[:1], NEIGHBORS)
    if 0 not in matches[0]:
        raise RuntimeError("CPU Faiss self-search failed")
    index.search_calls = 0
    print(f"Medium CPU IVF-Flat ready: {BANK_SIZE} windows in {time.perf_counter() - start:.2f}s")
    index_seconds = time.perf_counter() - start

    settings = ENVIRONMENTS["medium"]
    sampling_steps = settings["diffusion_steps"]
    projection_range = settings["projection"]
    agent = make_diffuser("medium")
    agent.load(str(checkpoint))
    agent.eval()
    prior = torch.zeros((PAIRS, dataset.horizon, 6), device="cuda:0")
    prior[:, [0, -1], :4] = torch.from_numpy(endpoint_states).to(prior.device)
    plans = np.empty((2, PAIRS, DRAWS, dataset.horizon, 6), dtype=np.float32)
    sample_seconds = np.zeros(2)
    peak_memory = np.zeros(2)
    start = time.perf_counter()
    for draw, (reverse, projection) in enumerate(zip(reverse_seeds, projection_seeds, strict=True)):
        reverse_state = None
        for arm, label in enumerate(CONDITIONS):
            torch.manual_seed(int(reverse))
            agent.projection_rng = torch.Generator(device="cuda:0").manual_seed(int(projection))
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            with torch.no_grad():
                samples, _ = agent.sample(
                    prior, solver="ddpm", n_samples=PAIRS, sample_steps=sampling_steps,
                    w_cg=0.0, faiss_wrapper=index if arm else None,
                    proj_range=projection_range if arm else [],
                    n_manifold_samples=NEIGHBORS, tau=TAU,
                )
            torch.cuda.synchronize()
            sample_seconds[arm] += time.perf_counter() - began
            peak_memory[arm] = max(peak_memory[arm], torch.cuda.max_memory_allocated() / MIB)
            after = torch.cuda.get_rng_state()
            if arm == 1 and not torch.equal(after, reverse_state):
                raise RuntimeError(f"Paired DDPM random streams diverged for draw {draw}")
            reverse_state = after
            if not torch.isfinite(samples).all():
                raise RuntimeError(f"{label}: non-finite samples in draw {draw}")
            torch.testing.assert_close(samples[:, [0, -1], :4], prior[:, [0, -1], :4], rtol=0, atol=0)
            plans[arm, :, draw] = samples.cpu().numpy()
        print(f"Medium paired sampling {draw + 1}/{DRAWS}: {index.search_calls} cumulative Faiss calls")

    expected_searches = DRAWS * (int(projection_range[1] * (sampling_steps - 1))
                                 - int(projection_range[0] * (sampling_steps - 1)) + 1)
    if index.search_calls != expected_searches or np.array_equal(plans[0], plans[1]):
        raise RuntimeError(f"Unexpected LoMAP activity: {index.search_calls} Faiss calls, expected {expected_searches}")
    sampling_wall = time.perf_counter() - start
    observations = dataset.unnormalize(plans[..., :4])
    crossing = wall_segments(observations[..., :2], boxes)
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
    temporary = output / "medium_samples.tmp.npz"
    np.savez_compressed(temporary, conditions=np.array(CONDITIONS), normalized_plans=plans,
                        observations=observations, wall_boxes=boxes, particle_origin=origin,
                        crossings=crossing, artifact=artifact, outside=outside, endpoint_ids=endpoint_ids,
                        reverse_seeds=reverse_seeds, projection_seeds=projection_seeds, bank_ids=bank_ids)
    temporary.replace(output / "medium_samples.npz")
    with (output / "medium_crossings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "draw", "start_transition", "goal_transition", "reverse_seed",
                         *CONDITIONS, "outside_without_lomap", "outside_with_lomap"])
        for pair, (start_id, goal_id) in enumerate(endpoint_ids):
            for draw, seed in enumerate(reverse_seeds):
                writer.writerow([pair + 1, draw + 1, start_id, goal_id, seed,
                                 *artifact[:, pair, draw].astype(int), *outside[:, pair, draw].astype(int)])
    write_json(output / "medium_report.json", {
        "checkpoint_sha256": checkpoint_hash,
        "training_steps": training["completed_steps"],
        "pairs": PAIRS,
        "draws_per_pair": DRAWS,
        "steps_per_sample": sampling_steps,
        "projection_range": projection_range,
        "tau": TAU,
        "bank_windows": BANK_SIZE,
        "neighbors": NEIGHBORS,
        "faiss_nlist": NLIST,
        "faiss_nprobe": index.index.nprobe,
        "faiss_version": faiss.__version__,
        "faiss_search_calls_with_lomap": index.search_calls,
        "index_build_seconds": index_seconds,
        "sampling_wall_seconds": sampling_wall,
        "same_reverse_rng": True,
        "exact_conditioned_endpoints": True,
        "artifact_rule": "Any straight segment between successive xy states intersects the interior of a wall box.",
        "paired_plan_outcomes": transition_count,
        "conditions": {
            label: {
                "crossing_plans": int(artifact[arm].sum()),
                "plan_crossing_rate": float(artifact[arm].mean()),
                "affected_pairs_by_draws": prefix[arm].tolist(),
                "affected_pairs_at_10_draws": int(prefix[arm, -1]),
                "pair_artifact_rate_at_10_draws": float(prefix[arm, -1] / PAIRS),
                "outside_plans": int(outside[arm].sum()),
                "sampling_seconds": float(sample_seconds[arm]),
                "peak_allocated_mib": float(peak_memory[arm]),
            } for arm, label in enumerate(CONDITIONS)
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args().output)
