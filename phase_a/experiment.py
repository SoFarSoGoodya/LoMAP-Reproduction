"""Generate data, run paired CPU comparisons, and save reproducible artifacts."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from lomap import project_trajectory


SEED = 20261120
N_TRIALS = 256
N_STEPS = 41
N_BANK_PER_SIDE = 64
K = 10
THRESHOLD = 0.99
OBSTACLE_RADIUS = 0.35
SIGMAS = np.array([0.0, 0.003, 0.03, 0.10])
MAIN_INDEX = 1
PARAMETER = np.linspace(0.0, 1.0, N_STEPS)
ARCH = np.sin(np.pi * PARAMETER)
ARCH[[0, -1]] = 0.0


def make_trajectories(amplitudes: np.ndarray) -> np.ndarray:
    x = np.broadcast_to(2.0 * PARAMETER - 1.0, (len(amplitudes), N_STEPS))
    return np.stack((x, amplitudes[:, None] * ARCH), axis=-1)


def make_data(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    bank_magnitudes = np.linspace(0.45, 1.15, N_BANK_PER_SIDE)
    bank = make_trajectories(np.concatenate((bank_magnitudes, -bank_magnitudes)))

    # Draw unseen amplitudes and balance the two ways around the obstacle.
    side = np.tile([1.0, -1.0], N_TRIALS // 2)
    clean = make_trajectories(side * rng.uniform(0.48, 0.98, N_TRIALS))
    drifted = clean.copy()
    vertical_drift = rng.uniform(0.20, 0.52, N_TRIALS)
    horizontal_drift = rng.uniform(-0.16, 0.16, N_TRIALS)
    drifted[:, :, 1] -= (side * vertical_drift)[:, None] * ARCH**8
    drifted[:, :, 0] += horizontal_drift[:, None] * ARCH**3

    return {
        "bank": bank,
        "clean": clean,
        "drifted": drifted,
        "query_noise": rng.standard_normal(clean.shape),
        "neighbor_noise": rng.standard_normal((N_TRIALS, K, N_STEPS, 2)),
    }


def segment_clearance(paths: np.ndarray) -> np.ndarray:
    """Minimum distance from the origin to any segment of each polyline."""
    starts = paths[..., :-1, :]
    vectors = paths[..., 1:, :] - starts
    lengths_squared = np.sum(vectors**2, axis=-1)
    numerator = -np.sum(starts * vectors, axis=-1)
    fraction = np.divide(
        numerator, lengths_squared, out=np.zeros_like(numerator), where=lengths_squared > 0
    )
    closest = starts + np.clip(fraction, 0.0, 1.0)[..., None] * vectors
    return np.linalg.norm(closest, axis=-1).min(axis=-1)


def trajectory_rmse(paths: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.sum((paths - clean) ** 2, axis=-1), axis=-1))


def evaluate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    before = []
    after = []
    neighbors = []
    ranks = []
    variances = []
    for sigma in SIGMAS:
        alpha = np.sqrt(1.0 - sigma**2)
        noisy = alpha * data["drifted"] + sigma * data["query_noise"]
        noisy[:, [0, -1]] = data["clean"][:, [0, -1]]
        projected = [
            project_trajectory(sample, query_noise, data["bank"], neighbor_noise, sigma, THRESHOLD)
            for sample, query_noise, neighbor_noise in zip(
                noisy, data["query_noise"], data["neighbor_noise"], strict=True
            )
        ]
        corrected = np.stack([result.trajectory for result in projected])
        # Remove only the known scale to express both noisy outputs in plot coordinates.
        # Endpoints are hard conditions and were not scaled when adding noise.
        for trajectories in (noisy, corrected):
            trajectories[:, 1:-1] /= alpha
        before.append(noisy)
        after.append(corrected)
        neighbors.append([result.neighbor_ids for result in projected])
        ranks.append([result.rank for result in projected])
        variances.append([result.explained_variance for result in projected])

    return {
        "before": np.stack(before),
        "after": np.stack(after),
        "neighbor_ids": np.array(neighbors),
        "rank": np.array(ranks),
        "explained_variance": np.array(variances),
    }


def metrics(data: dict[str, np.ndarray], results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "rmse_before": trajectory_rmse(results["before"], data["clean"]),
        "rmse_after": trajectory_rmse(results["after"], data["clean"]),
        "clearance_before": segment_clearance(results["before"]),
        "clearance_after": segment_clearance(results["after"]),
    }


def summarize(
    seed: int, results: dict[str, np.ndarray], scores: dict[str, np.ndarray]
) -> list[dict]:
    rows = []
    for i, sigma in enumerate(SIGMAS):
        before_hit = scores["clearance_before"][i] <= OBSTACLE_RADIUS
        after_hit = scores["clearance_after"][i] <= OBSTACLE_RADIUS
        rows.append({
            "seed": seed,
            "sigma": float(sigma),
            "trials": N_TRIALS,
            "collisions_before": int(before_hit.sum()),
            "collisions_after": int(after_hit.sum()),
            "resolved_collisions": int((before_hit & ~after_hit).sum()),
            "new_collisions": int((~before_hit & after_hit).sum()),
            "mean_rmse_before": float(scores["rmse_before"][i].mean()),
            "mean_rmse_after": float(scores["rmse_after"][i].mean()),
            "rmse_improved_trials": int((scores["rmse_after"][i] < scores["rmse_before"][i]).sum()),
            "median_pca_rank": float(np.median(results["rank"][i])),
        })
    return rows


def save_trials(out: Path, results: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> None:
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sigma", "trial_id", "pca_rank", "explained_variance", *scores])
        for i, sigma in enumerate(SIGMAS):
            for trial in range(N_TRIALS):
                writer.writerow([
                    sigma, trial, results["rank"][i, trial], results["explained_variance"][i, trial],
                    *(value[i, trial] for value in scores.values()),
                ])


def plot_results(
    out: Path, data: dict[str, np.ndarray], results: dict[str, np.ndarray],
    scores: dict[str, np.ndarray], summary: list[dict],
) -> None:
    baseline_color, lomap_color = "#b84a36", "#176a80"
    selected = np.arange(0, N_TRIALS, 4) + (np.arange(N_TRIALS // 4) % 2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    fig.suptitle(f"LoMAP on synthetic trajectories · noise σ = {SIGMAS[MAIN_INDEX]:g}", fontsize=15)
    for ax, key, color, title in zip(
        axes[0], ("before", "after"), (baseline_color, lomap_color),
        ("Before projection", "After projection"), strict=True,
    ):
        for path in data["bank"][::8]:
            ax.plot(*path.T, color="#afb7bb", linewidth=0.8, alpha=0.60, zorder=1)
        ax.add_patch(Circle((0, 0), OBSTACLE_RADIUS, facecolor="#f0dccf", edgecolor="#774b39", zorder=2))
        for path in results[key][MAIN_INDEX, selected]:
            ax.plot(*path.T, color=color, linewidth=0.9, alpha=0.24, zorder=3)
        ax.scatter([-1, 1], [0, 0], s=26, color="#17222c", zorder=4)
        ax.set(xlim=(-1.14, 1.14), ylim=(-1.18, 1.18), xlabel="x", ylabel="y", title=title)
        ax.set_aspect("equal")

    ax = axes[1, 0]
    before_rmse, after_rmse = (scores[key][MAIN_INDEX] for key in ("rmse_before", "rmse_after"))
    limit = 1.07 * max(before_rmse.max(), after_rmse.max())
    ax.plot([0, limit], [0, limit], linestyle="--", color="#82888b", linewidth=1)
    ax.scatter(before_rmse, after_rmse, color=lomap_color, alpha=0.5, s=13, edgecolors="none")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="RMSE before", ylabel="RMSE after",
           title="Error against the held-out trajectory")
    ax.set_aspect("equal")

    ax = axes[1, 1]
    positions = np.arange(len(SIGMAS))
    width = 0.34
    for offset, key, color, label in (
        (-width / 2, "collisions_before", baseline_color, "Before"),
        (width / 2, "collisions_after", lomap_color, "After"),
    ):
        rates = [100 * row[key] / N_TRIALS for row in summary]
        bars = ax.bar(positions + offset, rates, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xticks(positions, [f"{sigma:g}" for sigma in SIGMAS])
    ax.set(xlabel="Noise σ", ylabel="Trajectories hitting the obstacle (%)",
           ylim=(0, 100), title="Collision rate across noise levels")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out / "lomap_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = make_data(args.seed)
    results = evaluate(data)
    scores = metrics(data, results)
    summary = summarize(args.seed, results, scores)
    np.savez_compressed(args.output / "trajectories.npz", sigmas=SIGMAS, **data, **results)
    save_trials(args.output, results, scores)
    (args.output / "summary.json").write_text(
        json.dumps({
            "main_sigma": float(SIGMAS[MAIN_INDEX]),
            "waypoints": N_STEPS,
            "bank_size": len(data["bank"]),
            "neighbors": K,
            "variance_threshold": THRESHOLD,
            "obstacle_radius": OBSTACLE_RADIUS,
            "runs": summary,
        }, indent=2) + "\n", encoding="utf-8",
    )
    plot_results(args.output, data, results, scores, summary)
    row = summary[MAIN_INDEX]
    print(f"σ={row['sigma']}: collisions {row['collisions_before']} → {row['collisions_after']} "
          f"out of {N_TRIALS}; mean RMSE {row['mean_rmse_before']:.4f} → {row['mean_rmse_after']:.4f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
