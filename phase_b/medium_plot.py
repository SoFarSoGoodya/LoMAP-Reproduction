"""Render training, pair artifact curves and paired trajectories from saved results."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


LABELS = ["Diffuser", "Diffuser + LoMAP"]
COLORS = ["#ac593e", "#216d83"]


def plot(output):
    training = np.atleast_1d(np.genfromtxt(output / "training.csv", delimiter=",", names=True))
    report = json.loads((output / "medium_report.json").read_text(encoding="utf-8"))
    with np.load(output / "medium_samples.npz") as data:
        observations, boxes = data["observations"], data["wall_boxes"]
        artifact, crossing = data["artifact"], data["crossings"]
    fig, (loss_ax, rate_ax) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    size = min(250, len(training))
    window = np.ones(size) / size
    loss_ax.plot(training["step"][size - 1:], np.convolve(training["loss"], window, mode="valid"),
                 color=COLORS[1], linewidth=1.4)
    loss_ax.set(xlabel="Optimizer step", ylabel="MSE training loss", title=f"Medium Diffuser ({size}-step mean)")
    if training["loss"].min() > 0:
        loss_ax.set_yscale("log")
    for arm, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        count = np.maximum.accumulate(artifact[arm], axis=-1).sum(axis=0)
        rate_ax.plot(np.arange(1, len(count) + 1), count / report["pairs"] * 100,
                     "o-", ms=4, linewidth=1.5, color=color, label=label)
    rate_ax.set(xlabel="Plans drawn per start–goal pair", ylabel="Pairs with ≥1 wall crossing (%)",
                xticks=np.arange(1, report["draws_per_pair"] + 1), ylim=(-2, 102), title="Medium plan artifacts")
    rate_ax.legend(loc="lower right")
    for axis in (loss_ax, rate_ax):
        axis.grid(axis="y", alpha=0.22)
    fig.savefig(output / "medium_metrics.png", dpi=180)
    plt.close(fig)

    pairs = min(4, report["pairs"])
    fig, axes = plt.subplots(2, pairs, figsize=(3.05 * pairs, 6.5), layout="constrained", squeeze=False)
    lower, upper = boxes[:, 0].min(axis=0) - 0.15, boxes[:, 1].max(axis=0) + 0.15
    for arm in range(2):
        for pair in range(pairs):
            ax = axes[arm, pair]
            for lo, hi in boxes:
                ax.add_patch(Rectangle(lo, *(hi - lo), facecolor="#dbe2e5", edgecolor="#f3f5f6", linewidth=0.5))
            for draw, plan in enumerate(observations[arm, pair, :, :, :2]):
                ax.plot(*plan.T, color=COLORS[arm], alpha=0.35, lw=0.9)
                for segment in np.flatnonzero(crossing[arm, pair, draw]):
                    ax.plot(*plan[segment:segment + 2].T, color="#992d3e", alpha=0.7, lw=0.9)
            ax.scatter(*plan[0], color="#151525", s=24, zorder=4)
            ax.scatter(*plan[-1], marker="*", color="#151525", s=53, zorder=4)
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal", xticks=[], yticks=[],
                   title=f"Pair {pair + 1}, {artifact[arm, pair].sum()} / {report['draws_per_pair']} crossing")
            if pair == 0:
                ax.set_ylabel(LABELS[arm])
    fig.suptitle("Maze2D-Medium paired plans · start ● · goal ★ · wall-crossing segments in red")
    fig.savefig(output / "medium_trajectories.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output)
