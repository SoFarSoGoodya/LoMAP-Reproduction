"""Train a Maze2D Diffuser and save recoverable optimizer and RNG state."""

import argparse
import csv
import json
import math
import signal
import sys
import time
from itertools import islice
from pathlib import Path

import mujoco
import numpy as np
import torch

from maze2d import ENVIRONMENTS, MazeWindows, sha256
from prepare_assets import REVISION, main as prepare_assets
from smoke import MIB, make_diffuser


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_progress(path, agent, scheduler, configuration, step, seconds):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "configuration": configuration,
        "step": step,
        "seconds": seconds,
        "model": agent.model.state_dict(),
        "model_ema": agent.model_ema.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }, temporary)
    temporary.replace(path)


def train(output, steps, batch_size, seed, save_every, *, maze="medium", stop_requested=lambda: False):
    dataset = MazeWindows(maze)
    steps_per_epoch = len(dataset) // batch_size
    if steps_per_epoch < 1:
        raise ValueError("Batch size exceeds the number of training windows")
    configuration = {
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "dataset": dataset.describe(),
        "model": ENVIRONMENTS[maze] | {"ema": 0.9999, "scheduler": "cosine"},
        "upstream_revision": REVISION,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "mujoco": mujoco.__version__,
    }
    # Keep the serialized configuration identical to the one in the checkpoint.
    configuration = json.loads(json.dumps(configuration))
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    agent = make_diffuser(maze)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, steps)
    agent.train()
    progress = output / "training_state.pt"
    finished, elapsed = 0, 0.0
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=True)
        if state["configuration"] != configuration:
            raise ValueError("The existing checkpoint has a different training configuration")
        agent.model.load_state_dict(state["model"])
        agent.model_ema.load_state_dict(state["model_ema"])
        agent.optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        finished, elapsed = state["step"], state["seconds"]
        del state
        print(f"Resuming from step {finished}/{steps}")
    timing = output / "training.csv"
    rows = []
    if timing.exists():
        with timing.open(newline="", encoding="utf-8") as handle:
            rows = list(islice(csv.DictReader(handle), finished))
    if len(rows) != finished or any(int(row["step"]) != i for i, row in enumerate(rows, 1)):
        raise RuntimeError("Training log does not agree with the resume checkpoint")
    fieldnames = ["step", "seconds", "loss", "learning_rate"]
    temporary_timing = timing.with_suffix(".csv.tmp")
    with temporary_timing.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_timing.replace(timing)
    write_json(output / "training_config.json", configuration)

    indices = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    with timing.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames)
        for step in range(finished + 1, steps + 1):
            epoch, offset = divmod(step - 1, steps_per_epoch)
            if indices is None or offset == 0:
                indices = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(dataset))
            selected = indices[offset * batch_size:(offset + 1) * batch_size]
            start = time.perf_counter()
            batch = torch.from_numpy(dataset.batch(selected)).to("cuda:0")
            learning_rate = scheduler.get_last_lr()[0]
            loss = float(agent.update(batch)["loss"])
            scheduler.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            row = {"step": step, "seconds": seconds, "loss": loss, "learning_rate": learning_rate}
            rows.append(row)
            writer.writerow(row)
            if step % 250 == 0 or step == steps:
                handle.flush()
                recent = rows[-250:]
                print(f"{maze.capitalize()} {step}/{steps}: mean loss={np.mean([float(r['loss']) for r in recent]):.5f}, "
                      f"step={1000 * np.mean([float(r['seconds']) for r in recent]):.2f} ms")
            stopping = stop_requested()
            if stopping or step % save_every == 0 or step == steps:
                handle.flush()
                save_progress(progress, agent, scheduler, configuration, step, elapsed + time.perf_counter() - begin)
            if stopping and step < steps:
                print(f"Paused after saving step {step}/{steps}", flush=True)
                return 75

    elapsed += time.perf_counter() - begin
    checkpoint = output / "diffusion.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    agent.save(str(temporary))
    temporary.replace(checkpoint)
    durations = np.array([float(row["seconds"]) for row in rows])
    losses = np.array([float(row["loss"]) for row in rows])
    warmup = min(20, steps // 5)
    stable = durations[warmup:]
    write_json(output / "training_report.json", configuration | {
        "completed_steps": steps,
        "wall_seconds": elapsed,
        "stable_mean_ms_per_step": float(stable.mean() * 1000),
        "stable_p90_ms_per_step": float(np.quantile(stable, 0.9) * 1000),
        "first_250_mean_loss": float(losses[:250].mean()),
        "last_250_mean_loss": float(losses[-250:].mean()),
        "peak_allocated_mib_this_process": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib_this_process": torch.cuda.max_memory_reserved() / MIB,
        "checkpoint_sha256": sha256(checkpoint),
        "resumed_from_step": finished,
    })
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze", choices=ENVIRONMENTS, default="medium")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every) < 1 or args.seed < 0:
        parser.error("steps, batch size and checkpoint interval must be positive, and seed nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    prepare_assets((args.maze,))
    return train(args.output, args.steps, args.batch_size, args.seed, args.save_every,
                 maze=args.maze, stop_requested=lambda: stopping)


if __name__ == "__main__":
    raise SystemExit(main())
