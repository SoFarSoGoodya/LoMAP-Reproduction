"""Train, reload and compare the two U-Maze sampling paths on the local GPU."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import faiss
import mujoco
import numpy as np
import torch

from maze2d import ENVIRONMENTS, MazeWindows, PointMaze, sha256
from prepare_assets import REVISION, UPSTREAM, main as prepare_assets

if not (UPSTREAM / "cleandiffuser_ex/diffusion/diffusionsde_ex.py").is_file():
    raise SystemExit(f"Fetch the pinned LoMAP checkout at {UPSTREAM}. See README.md.")
sys.path.insert(0, str(UPSTREAM))

from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser_ex.diffusion import DiscreteDiffusionSDEEX


MIB = 1024**2


class CpuFaiss:
    """The weighted IVF-Flat cosine index expected by upstream's sampler."""

    def __init__(self, bank, k, nlist):
        self.bank = bank
        self.weights = np.ones(bank.shape[1:], dtype=np.float32)
        self.weights[:, 4:] = 0.0
        self.weights[[0, -1], :2] = 100.0
        vectors = self.encode(bank)
        dimension = vectors.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.cp.seed = 0
        self.index.nprobe = min(5, nlist)
        self.index.train(vectors)
        self.index.add(vectors)
        self.k = k
        self.search_calls = 0

    def encode(self, trajectories):
        vectors = np.ascontiguousarray(trajectories.reshape(-1, *self.weights.shape) * self.weights)
        vectors = vectors.reshape(len(vectors), -1)
        faiss.normalize_L2(vectors)
        return vectors

    def search(self, queries, k):
        if k != self.k:
            raise ValueError(f"Expected {self.k} neighbors, received {k}")
        self.search_calls += 1
        distances, indices = self.index.search(self.encode(queries), k)
        if np.any(indices < 0):
            raise RuntimeError("Faiss returned an incomplete set of neighbors")
        return distances, indices

    def get_original_vectors(self, indices):
        return self.bank[indices]


class PairedDiffuser(DiscreteDiffusionSDEEX):
    def get_manifold_samples(self, xt, t, n_manifold_samples, prior, faiss_wrapper):
        # Give projection noise its own stream so reverse DDPM noise is shared.
        device = xt.device
        with torch.random.fork_rng(devices=[device.index]):
            torch.cuda.set_rng_state(self.projection_rng.get_state(), device)
            samples = super().get_manifold_samples(xt, t, n_manifold_samples, prior, faiss_wrapper)
            self.projection_rng.set_state(torch.cuda.get_rng_state(device))
        return samples

    def compute_rg(self, xt, *args, **kwargs):
        device = xt.device
        with torch.random.fork_rng(devices=[device.index]):
            torch.cuda.set_rng_state(self.gap_rng.get_state(), device)
            gap = super().compute_rg(xt, *args, **kwargs)
            self.gap_rng.set_state(torch.cuda.get_rng_state(device))
        return gap


def make_diffuser(maze):
    settings = ENVIRONMENTS[maze]
    network = JannerUNet1d(
        6, model_dim=32, emb_dim=32, dim_mult=[1, 4, 2],
        timestep_emb_type="positional", attention=False, kernel_size=5,
    )
    fix_mask = torch.zeros(settings["horizon"], 6)
    fix_mask[[0, -1], :4] = 1.0
    return PairedDiffuser(
        network, None, fix_mask=fix_mask, ema_rate=0.9999, device="cuda:0",
        diffusion_steps=settings["diffusion_steps"], predict_noise=False,
    )


def verify_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    result = torch.tensor([2.0], device="cuda:0").square().item()
    if result != 4.0:
        raise RuntimeError("CUDA arithmetic check failed")

    information = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "mujoco": mujoco.__version__,
        "cpu_faiss": faiss.__version__,
        "upstream_revision": REVISION,
        "datasets": {},
    }
    datasets = {name: MazeWindows(name) for name in ("umaze", "medium")}
    for name, dataset in datasets.items():
        maze = PointMaze(name)
        before = maze.data.qpos.copy()
        observation, reward = maze.step(np.array([0.25, 0.0]))
        if observation[0] <= before[0] or reward != 1.0 or not np.isfinite(observation).all():
            raise RuntimeError(f"{name}: MuJoCo step did not match the expected motion and reward")
        information["datasets"][name] = dataset.describe() | {
            "mujoco_time_step": maze.model.opt.timestep,
            "mujoco_dofs": maze.model.nq,
            "tested_observation": observation.tolist(),
        }
    return information, datasets["umaze"]


def train(agent, dataset, args):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=True,
    )
    if args.steps > len(loader):
        raise ValueError(f"Smoke run allows at most one data pass ({len(loader)} steps)")
    batches = iter(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, args.steps)
    agent.train()
    timing_file = args.output / "training.csv"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    timings = []
    losses = []
    with timing_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "seconds", "loss"])
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            batch = next(batches).to("cuda:0", non_blocking=True)
            loss = float(agent.update(batch)["loss"])
            scheduler.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - step_start
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            timings.append(duration)
            losses.append(loss)
            writer.writerow([step, f"{duration:.6f}", f"{loss:.8g}"])
            if step % 50 == 0 or step == args.steps:
                handle.flush()
                print(f"train {step}/{args.steps}: loss={loss:.4f}, recent step={np.mean(timings[-50:]):.3f}s")

    wall = time.perf_counter() - start
    warmup = min(20, args.steps // 5)
    stable = np.array(timings[warmup:])
    checkpoint = args.output / "diffusion.pt"
    temporary = checkpoint.with_suffix(".tmp")
    agent.save(str(temporary))
    temporary.replace(checkpoint)
    return {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "wall_seconds": wall,
        "discarded_warmup_steps": warmup,
        "stable_steps": len(stable),
        "stable_mean_ms_per_step": float(stable.mean() * 1000),
        "stable_median_ms_per_step": float(np.median(stable) * 1000),
        "stable_p90_ms_per_step": float(np.quantile(stable, 0.9) * 1000),
        "first_50_mean_loss": float(np.mean(losses[:50])),
        "last_50_mean_loss": float(np.mean(losses[-50:])),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "checkpoint_sha256": sha256(checkpoint),
    }


def compare(dataset, args):
    seed = args.seed
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    bank_ids = rng.choice(len(dataset), args.bank_size, replace=False)
    bank = np.stack([dataset[int(i)] for i in bank_ids])
    k = 10
    index = CpuFaiss(bank, k, nlist=64)
    build_seconds = time.perf_counter() - start
    _, self_matches = index.search(bank[:1], k)
    if 0 not in self_matches[0]:
        raise RuntimeError("CPU Faiss did not recover a known trajectory in its own index")
    index.search_calls = 0

    # Load weights into a new model, independent of the training model.
    agent = make_diffuser("umaze")
    agent.load(str(args.output / "diffusion.pt"))
    agent.eval()
    agent.projection_rng = torch.Generator(device="cuda:0").manual_seed(seed + 1)
    maze = PointMaze("umaze")
    prior = torch.zeros((args.plans, dataset.horizon, 6), device="cuda:0")
    start_ids = rng.choice(len(dataset), args.plans, replace=False)
    starts = np.stack([dataset[int(i)][0, :4] for i in start_ids])
    prior[:, 0, :4] = torch.from_numpy(starts).to(prior.device)
    target = dataset.normalize(np.r_[maze.goal, 0.0, 0.0].astype(np.float32))
    prior[:, -1, :4] = torch.from_numpy(target).to(prior.device)

    outputs = {"bank_ids": bank_ids, "start_ids": start_ids}
    measurements = {}
    reverse_states = []
    for enabled, label in ((False, "without_lomap"), (True, "with_lomap")):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        with torch.no_grad():
            samples, _ = agent.sample(
                prior, solver="ddpm", n_samples=args.plans, sample_steps=128,
                w_cg=0.0, faiss_wrapper=index if enabled else None,
                proj_range=ENVIRONMENTS["umaze"]["projection"] if enabled else [],
                n_manifold_samples=k, tau=0.99,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        reverse_states.append(torch.cuda.get_rng_state())
        if not torch.isfinite(samples).all():
            raise RuntimeError(f"{label}: non-finite sampled plans")
        torch.testing.assert_close(samples[:, [0, -1], :4], prior[:, [0, -1], :4], rtol=0, atol=0)
        plans = dataset.unnormalize(samples[:, :, :4].cpu().numpy())
        rollouts = [maze.follow(plan) for plan in plans]
        observations, rewards = map(np.stack, zip(*rollouts, strict=True))
        if not np.isfinite(observations).all() or not np.isfinite(rewards).all():
            raise RuntimeError(f"{label}: non-finite MuJoCo rollout")
        outputs[label] = samples.cpu().numpy()
        outputs[f"{label}_plan_observations"] = plans
        outputs[f"{label}_rollout_observations"] = observations
        outputs[f"{label}_rollout_rewards"] = rewards
        measurements[label] = {
            "sample_seconds": seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            "cumulative_faiss_search_calls": index.search_calls,
            "rollout_steps_per_plan": rewards.shape[1],
        }
        print(f"{label}: {seconds:.2f}s, {index.search_calls} cumulative Faiss searches")

    if not torch.equal(*reverse_states):
        raise RuntimeError("The paired DDPM random streams diverged")
    if not index.search_calls or np.array_equal(outputs["without_lomap"], outputs["with_lomap"]):
        raise RuntimeError("LoMAP did not exercise the index or change any sample")
    np.savez_compressed(args.output / "paired_samples.npz", **outputs)
    return {
        "plans_per_condition": args.plans,
        "sampling_steps": 128,
        "neighbors": k,
        "bank_size": len(bank),
        "faiss_build_seconds": build_seconds,
        "same_reverse_rng": True,
        "exact_conditioned_endpoints": True,
        "conditions": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--plans", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=8192)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.plans) < 1 or args.bank_size < 64 * 39:
        parser.error("steps, batch size and plans must be positive, and bank size must be at least 2496")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(8)
    prepare_assets()
    environment, dataset = verify_environment()
    print(f"Verified environment: {environment['gpu']}, torch {environment['torch']}")
    report = {"environment": environment, "seed": args.seed}
    agent = make_diffuser("umaze")
    report["training"] = train(agent, dataset, args)
    del agent
    torch.cuda.empty_cache()
    report["comparison"] = compare(dataset, args)
    temporary = args.output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "report.json")


if __name__ == "__main__":
    main()
