"""Evaluate Maze2D normalized returns for the Diffuser and LoMAP Table 1 arms."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import faiss
import numpy as np
import torch

PHASE_B = Path(os.environ.get("LOMAP_PHASE_B", Path(__file__).resolve().parents[1] / "phase_b"))
sys.path.insert(0, str(PHASE_B))

from maze2d import ENVIRONMENTS, MazeWindows, PointMaze, sha256
from medium_evaluate import wall_boxes
from medium_train import write_json
from smoke import make_diffuser
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexIVFWrapper


# D4RL PointMaze v1 registration and the upstream commented Multi2D settings.
SETTINGS = {
    "umaze": (300, 23.85, 161.86, (0.2, 0.7)),
    "medium": (600, 13.13, 277.39, (0.4, 0.9)),
    "large": (800, 6.7, 273.99, (0.6, 0.7)),
}
LABELS = ("diffuser", "diffuser_lomap")


def free_tiles(maze):
    boxes, origin = wall_boxes(maze)
    walls = np.rint(boxes.mean(axis=1) + origin).astype(int)
    occupied = set(map(tuple, walls))
    low, high = walls.min(axis=0), walls.max(axis=0)
    return np.array([(x, y) for x in range(low[0] + 1, high[0])
                     for y in range(low[1] + 1, high[1]) if (x, y) not in occupied]) - origin


def run_episode(maze, start, goal, plan, limit):
    maze.goal = goal
    maze.reset(start)
    observation = start
    rewards = np.empty(limit, dtype=np.float32)
    final_waypoint = plan[-1].copy()
    final_waypoint[2:] = 0
    for t in range(limit):
        waypoint = plan[t + 1] if t + 1 < len(plan) else final_waypoint
        error = waypoint - observation
        observation, rewards[t] = maze.step(error[:2] + error[2:])
    # Upstream masks the TimeLimit terminal reward for these v1 horizons.
    return float(rewards[:-1].sum()), bool(rewards.any())


def evaluate(args):
    dataset = MazeWindows(args.maze)
    env = PointMaze(args.maze)
    tiles = free_tiles(env)
    limit, min_return, max_return, multi_projection = SETTINGS[args.maze]
    settings = ENVIRONMENTS[args.maze]
    training = json.loads((args.source / "training_report.json").read_text())
    checkpoint = args.source / "diffusion.pt"
    digest = sha256(checkpoint)
    if training["dataset"] != dataset.describe() or digest != training["checkpoint_sha256"]:
        raise ValueError("Training data or checkpoint fingerprint does not match")

    args.output.mkdir(parents=True, exist_ok=True)
    episode_rng, bank_rng = map(np.random.default_rng, np.random.SeedSequence(args.seed).spawn(2))
    windows = min(args.bank_size, len(dataset))
    bank_ids = bank_rng.choice(len(dataset), windows, replace=False)
    bank = dataset.batch(bank_ids)
    starts = np.column_stack((
        tiles[episode_rng.integers(len(tiles), size=args.episodes)]
        + episode_rng.uniform(-0.1, 0.1, size=(args.episodes, 2)),
        episode_rng.normal(0.0, 0.1, size=(args.episodes, 2)),
    ))
    fixed_goals = np.broadcast_to(env.goal, (args.episodes, 2)).copy()
    random_goals = (tiles[episode_rng.integers(len(tiles), size=args.episodes)]
                    + episode_rng.uniform(-0.1, 0.1, size=(args.episodes, 2)))
    batch_seeds = episode_rng.integers(0, 2**32, size=(args.episodes + args.batch_size - 1) // args.batch_size)
    agent = make_diffuser(args.maze)
    agent.load(str(checkpoint))
    agent.eval()
    faiss.omp_set_num_threads(8)
    torch.set_num_threads(4)

    for task, goals, interval in (("maze2d", fixed_goals, settings["projection"]),
                                  ("multi2d", random_goals, multi_projection)):
        weights = np.ones((dataset.horizon, 6), dtype=np.float32)
        weights[:, 4:] = float(task == "multi2d")
        weights[[0, -1], :2] = 100.0
        index = FaissIndexIVFWrapper("cosine", args.nlist, bank, weights, device="cuda:0")
        returns = np.empty((len(LABELS), args.episodes))
        reached = np.zeros_like(returns, dtype=bool)
        for batch, start in enumerate(range(0, args.episodes, args.batch_size)):
            end = min(start + args.batch_size, args.episodes)
            prior = torch.zeros((end - start, dataset.horizon, 6), device="cuda:0")
            target_states = np.column_stack((goals[start:end], np.zeros((end - start, 2))))
            endpoints = np.stack((starts[start:end], target_states), axis=1).astype(np.float32)
            prior[:, [0, -1], :4] = torch.from_numpy(dataset.normalize(endpoints)).to(prior.device)
            reverse_state = None
            for arm in range(len(LABELS)):
                torch.manual_seed(int(batch_seeds[batch]))
                agent.projection_rng = torch.Generator(device="cuda:0").manual_seed(int(batch_seeds[batch]) + 1)
                with torch.no_grad():
                    plans, _ = agent.sample(
                        prior, solver="ddpm", n_samples=end - start, sample_steps=settings["diffusion_steps"],
                        w_cg=0.0, faiss_wrapper=index if arm else None, proj_range=interval if arm else [],
                        n_manifold_samples=10, tau=0.99,
                    )
                torch.testing.assert_close(plans[:, [0, -1], :4], prior[:, [0, -1], :4], rtol=0, atol=0)
                if not torch.isfinite(plans).all():
                    raise RuntimeError(f"Non-finite plans for {task} and {LABELS[arm]}")
                if arm and not torch.equal(torch.cuda.get_rng_state(), reverse_state):
                    raise RuntimeError("Paired reverse DDPM noise streams diverged")
                reverse_state = torch.cuda.get_rng_state()
                for episode, plan in enumerate(dataset.unnormalize(plans[..., :4].cpu().numpy()), start):
                    returns[arm, episode], reached[arm, episode] = run_episode(
                        env, starts[episode], goals[episode], plan, limit
                    )
            print(f"{args.maze} {task}: {end}/{args.episodes} paired rollouts", flush=True)
        scores = (returns - min_return) * 100 / (max_return - min_return)
        with (args.output / f"{task}_episodes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "start_x", "start_y", "start_vx", "start_vy", "goal_x", "goal_y",
                             "diffuser_return", "lomap_return", "diffuser_score", "lomap_score",
                             "diffuser_reached", "lomap_reached"])
            for i in range(args.episodes):
                writer.writerow([i + 1, *starts[i], *goals[i], *returns[:, i], *scores[:, i], *reached[:, i]])
        write_json(args.output / f"{task}_report.json", {
            "maze": args.maze, "task": task, "checkpoint_sha256": digest,
            "training_steps": training["completed_steps"], "episodes": args.episodes,
            "episode_limit": limit, "eval_seed": args.seed, "batch_size": args.batch_size,
            "bank_windows": windows, "faiss_nlist": args.nlist, "faiss_nprobe": 5,
            "projection_range": interval, "action_weight": float(task == "multi2d"),
            "score_reference": [min_return, max_return],
            "conditions": {
                label: {"mean_normalized_return": float(scores[i].mean()),
                        "standard_error": float(scores[i].std() / np.sqrt(args.episodes)),
                        "goal_hit_rate": float(reached[i].mean())}
                for i, label in enumerate(LABELS)
            },
        })
        del index
    np.savez_compressed(args.output / "rollout_inputs.npz", bank_ids=bank_ids, batch_seeds=batch_seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", choices=SETTINGS, required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--bank-size", type=int, default=1000000)
    parser.add_argument("--nlist", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.episodes, args.batch_size, args.nlist) < 1 or args.bank_size < max(10, args.nlist):
        parser.error("positive episode and batch counts are required, and the bank must fit nlist and neighbors")
    evaluate(args)
