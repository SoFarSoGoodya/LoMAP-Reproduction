"""One LoMAP projection on a two-dimensional trajectory."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    trajectory: np.ndarray
    neighbor_ids: np.ndarray
    rank: int
    explained_variance: float


def cosine_neighbors(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    flat_bank = bank.reshape(len(bank), -1)
    flat_query = query.ravel()
    norms = np.linalg.norm(flat_bank, axis=1) * np.linalg.norm(flat_query)
    similarity = np.divide(
        flat_bank @ flat_query, norms, out=np.zeros_like(norms), where=norms > 0
    )
    return np.argsort(-similarity, kind="stable")[:k]


def pca_project(
    sample: np.ndarray, neighbors: np.ndarray, variance_threshold: float
) -> tuple[np.ndarray, int, float]:
    flat_neighbors = neighbors.reshape(len(neighbors), -1)
    mean = flat_neighbors.mean(axis=0)
    _, singular_values, components = np.linalg.svd(
        flat_neighbors - mean, full_matrices=False
    )
    if singular_values[0] <= np.finfo(np.float64).eps * np.linalg.norm(flat_neighbors):
        return mean.reshape(sample.shape), 0, 1.0

    variances = singular_values**2
    cumulative = np.cumsum(variances) / np.sum(variances)
    rank = min(int(np.searchsorted(cumulative, variance_threshold)) + 1, len(variances))
    basis = components[:rank]
    centered = sample.ravel() - mean
    projected = mean + (centered @ basis.T) @ basis
    return projected.reshape(sample.shape), rank, float(cumulative[rank - 1])


def project_trajectory(
    noisy_sample: np.ndarray,
    predicted_noise: np.ndarray,
    bank: np.ndarray,
    neighbor_noise: np.ndarray,
    sigma: float,
    variance_threshold: float,
) -> Projection:
    """Retrieve in clean space, then fit and project at the sample's noise level."""
    alpha = np.sqrt(1.0 - sigma**2)
    surrogate = (noisy_sample - sigma * predicted_noise) / alpha
    surrogate[[0, -1]] = noisy_sample[[0, -1]]

    neighbor_ids = cosine_neighbors(surrogate, bank, len(neighbor_noise))
    neighbors = alpha * bank[neighbor_ids] + sigma * neighbor_noise
    neighbors[:, [0, -1]] = noisy_sample[[0, -1]]

    # Fixed endpoints do not participate in PCA.
    interior, rank, explained = pca_project(
        noisy_sample[1:-1], neighbors[:, 1:-1], variance_threshold
    )
    projected = noisy_sample.copy()
    projected[1:-1] = interior
    return Projection(projected, neighbor_ids, rank, explained)
