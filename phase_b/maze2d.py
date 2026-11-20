"""D4RL Maze2D data and headless MuJoCo compatibility for the local run."""

import hashlib

import h5py
import mujoco
import numpy as np
import torch

from prepare_assets import ROOT


ENVIRONMENTS = {
    "umaze": {"goal": (1, 1), "horizon": 128, "diffusion_steps": 128, "projection": (0.3, 0.8)},
    "medium": {"goal": (6, 6), "horizon": 256, "diffusion_steps": 256, "projection": (0.3, 0.4)},
}
DATA_SHA256 = {
    "umaze": "f2d327d95067d53e274ac2fe876b740ed7883d0a9738bd5b447635cdb786bfb7",
    "medium": "b33981665984ee1a1667a2c2109638e8dd6b9767f39611cb9c16f8ec4fc32ab5",
}


def sha256(path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class MazeWindows(torch.utils.data.Dataset):
    def __init__(self, maze: str):
        settings = ENVIRONMENTS[maze]
        self.horizon = settings["horizon"]
        self.goal = np.array(settings["goal"], dtype=np.float32)
        file = ROOT / "data" / f"maze2d-{maze}-sparse-v1.hdf5"
        self.checksum = sha256(file)
        if self.checksum != DATA_SHA256[maze]:
            raise ValueError(f"Unexpected SHA-256 for {file}")

        with h5py.File(file, "r") as source:
            obs = source["observations"][:].astype(np.float32, copy=False)
            actions = source["actions"][:].astype(np.float32, copy=False)
            terminals = source["terminals"][:].astype(bool, copy=False)
            if obs.shape != (len(terminals), 4) or actions.shape != (len(terminals), 2):
                raise ValueError(f"Unexpected observation or action shape in {file}")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite observations or actions in {file}")

        at_goal = np.linalg.norm(obs[:, :2] - self.goal, axis=1) < 0.5
        path_end = terminals.copy()
        path_end[:-1] |= at_goal[:-1] & ~at_goal[1:]
        ends = np.flatnonzero(path_end)
        self.starts = np.r_[0, ends[:-1] + 1]
        self.lengths = ends + 1 - self.starts
        counts = np.maximum(self.lengths - self.horizon + 1, 0)
        self.total_windows = np.cumsum(counts)

        self.minimum = obs.min(axis=0)
        span = obs.max(axis=0) - self.minimum
        self.observation_mask = span > 0
        self.span = np.where(self.observation_mask, span, 1.0)
        self.trajectories = np.concatenate((self.normalize(obs), actions), axis=1)

    def normalize(self, observations):
        scaled = 2.0 * (observations - self.minimum) / self.span - 1.0
        return scaled * self.observation_mask

    def unnormalize(self, observations):
        return (observations + 1.0) * (self.span / 2.0) * self.observation_mask + self.minimum

    def __len__(self):
        return int(self.total_windows[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        path = np.searchsorted(self.total_windows, index, side="right")
        preceding = self.total_windows[path - 1] if path else 0
        start = self.starts[path] + index - preceding
        return self.trajectories[start:start + self.horizon]

    def batch(self, indices):
        paths = np.searchsorted(self.total_windows, indices, side="right")
        preceding = np.r_[0, self.total_windows][paths]
        starts = self.starts[paths] + indices - preceding
        return self.trajectories[starts[:, None] + np.arange(self.horizon)]

    def describe(self):
        return {
            "sha256": self.checksum,
            "transitions": len(self.trajectories),
            "segments": len(self.lengths),
            "longest_segment": int(self.lengths.max()),
            "windows": len(self),
            "horizon": self.horizon,
            "observation_min": self.minimum.tolist(),
            "observation_span": self.span.tolist(),
        }


class PointMaze:
    def __init__(self, maze: str):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / f"maze2d-{maze}-v1.xml"))
        self.data = mujoco.MjData(self.model)
        self.goal = np.array(ENVIRONMENTS[maze]["goal"], dtype=np.float32)
        self.reset(np.r_[self.goal, 0.0, 0.0])

    def reset(self, state):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = state[:2]
        self.data.qvel[:] = state[2:]
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        np.clip(self.data.qvel, -5.0, 5.0, out=self.data.qvel)
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        observation = np.r_[self.data.qpos, self.data.qvel]
        reward = float(np.linalg.norm(observation[:2] - self.goal) <= 0.5)
        return observation, reward

    def follow(self, plan):
        self.reset(plan[0])
        actual = [plan[0]]
        rewards = []
        for waypoint in plan[1:]:
            error = waypoint - actual[-1]
            observation, reward = self.step(error[:2] + error[2:])
            actual.append(observation)
            rewards.append(reward)
        return np.array(actual), np.array(rewards)
