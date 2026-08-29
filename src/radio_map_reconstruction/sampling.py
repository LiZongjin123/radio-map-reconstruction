from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
from sklearn.cluster import KMeans
from torch import Tensor, as_tensor, float32, long, tensor, zeros_like
from torch.nn.functional import conv2d, pad


@dataclass(frozen=True)
class SamplingDiagnostics:
    coarse_map: Tensor
    normalized_gradient: Tensor
    normalized_distance: Tensor
    score: Tensor
    cluster_labels: Tensor
    selected_points: Tensor
    transmitter_point: Tensor


def _map_plane(value: Tensor, name: str) -> tuple[Tensor, bool]:
    if value.ndim == 2:
        return value, False
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0], True
    raise ValueError(f"{name} must have shape (height, width) or (1, height, width)")


def _restore_channel(value: Tensor, had_channel: bool) -> Tensor:
    return value.unsqueeze(0) if had_channel else value


def _normalize_over_candidates(
    values: Tensor, valid_receiving_area: Tensor
) -> Tensor:
    candidate_values = values[valid_receiving_area]
    minimum = candidate_values.min()
    value_range = candidate_values.max() - minimum
    normalized = zeros_like(values, dtype=float32)
    if value_range.item() > 0:
        normalized[valid_receiving_area] = (
            candidate_values - minimum
        ) / value_range
    return normalized


def _sobel_magnitude(coarse_map: Tensor) -> Tensor:
    kernels = tensor(
        [
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        ],
        dtype=coarse_map.dtype,
        device=coarse_map.device,
    )
    padded = pad(coarse_map[None, None], (1, 1, 1, 1), mode="replicate")
    components = conv2d(padded, kernels)[0]
    return (components.square().sum(dim=0)).sqrt()


def _derived_random_state(global_seed: int, sample_id: str, sample_count: int) -> int:
    material = f"{global_seed}:{sample_id}:{sample_count}".encode()
    return int.from_bytes(blake2b(material, digest_size=4).digest(), "big")


def gradient_distance_weighted_clustering_sample(
    coarse_map: Tensor,
    tx_map: Tensor,
    building_map: Tensor,
    sample_count: int,
    *,
    alpha: float,
    global_seed: int,
    sample_id: str,
    weight_epsilon: float,
    max_iter: int,
    tolerance: float,
) -> tuple[Tensor, SamplingDiagnostics]:
    """Select Valid Sampling Points using deterministic weighted spatial clusters."""
    coarse, coarse_had_channel = _map_plane(coarse_map, "coarse_map")
    tx, tx_had_channel = _map_plane(tx_map, "tx_map")
    building, building_had_channel = _map_plane(building_map, "building_map")
    if not (coarse.shape == tx.shape == building.shape):
        raise ValueError("coarse_map, tx_map, and building_map must have equal shapes")
    if len({coarse_had_channel, tx_had_channel, building_had_channel}) != 1:
        raise ValueError("coarse_map, tx_map, and building_map must use equal ranks")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if weight_epsilon <= 0:
        raise ValueError("weight_epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    transmitter_points = (tx > 0.5).nonzero()
    if transmitter_points.shape[0] != 1:
        raise ValueError(
            f"sample {sample_id!r}: tx_map must contain exactly one transmitter "
            f"pixel; found {transmitter_points.shape[0]}"
        )
    transmitter_point = transmitter_points[0]
    valid_receiving_area = (tx < 0.5) & (building < 0.5)
    candidate_points = valid_receiving_area.nonzero()
    if candidate_points.shape[0] < sample_count:
        raise ValueError(
            f"sample {sample_id!r}: requested {sample_count} Valid Sampling Points "
            f"but only {candidate_points.shape[0]} candidates exist"
        )

    clipped_coarse = coarse.detach().to(float32).clamp(0, 1)
    gradient = _sobel_magnitude(clipped_coarse)
    normalized_gradient = _normalize_over_candidates(
        gradient, valid_receiving_area
    )
    offsets = candidate_points.to(float32) - transmitter_point.to(float32)
    distances = offsets.square().sum(dim=1).sqrt()
    distance_map = zeros_like(clipped_coarse)
    distance_map[valid_receiving_area] = distances
    normalized_distance = _normalize_over_candidates(
        distance_map, valid_receiving_area
    )
    score = zeros_like(clipped_coarse)
    score[valid_receiving_area] = (
        alpha * normalized_gradient[valid_receiving_area]
        + (1 - alpha) * normalized_distance[valid_receiving_area]
    )

    candidate_coordinates = candidate_points.detach().cpu().numpy()
    candidate_weights = (
        (score[valid_receiving_area] + weight_epsilon).detach().cpu().numpy()
    )
    kmeans = KMeans(
        n_clusters=sample_count,
        init="k-means++",
        n_init=1,
        random_state=_derived_random_state(global_seed, sample_id, sample_count),
        max_iter=max_iter,
        tol=tolerance,
    )
    labels = kmeans.fit_predict(
        candidate_coordinates,
        sample_weight=candidate_weights,
    )

    width = coarse.shape[1]
    selected_candidate_indices: list[int] = []
    selected_candidate_set: set[int] = set()
    for cluster_index, center in enumerate(kmeans.cluster_centers_):
        cluster_candidates = np.flatnonzero(labels == cluster_index)
        cluster_candidates = np.asarray(
            [
                index
                for index in cluster_candidates
                if int(index) not in selected_candidate_set
            ],
            dtype=np.int64,
        )
        if cluster_candidates.size == 0:
            raise RuntimeError(
                f"sample {sample_id!r}: cluster {cluster_index} has no unselected "
                "Valid Sampling Point"
            )
        cluster_coordinates = candidate_coordinates[cluster_candidates]
        center_distances = np.square(cluster_coordinates - center).sum(axis=1)
        flattened_indices = (
            cluster_coordinates[:, 0] * width + cluster_coordinates[:, 1]
        )
        nearest_offset = np.lexsort((flattened_indices, center_distances))[0]
        selected_index = int(cluster_candidates[nearest_offset])
        selected_candidate_indices.append(selected_index)
        selected_candidate_set.add(selected_index)

    if len(selected_candidate_set) != sample_count:
        raise RuntimeError(
            f"sample {sample_id!r}: expected {sample_count} distinct Valid Sampling "
            f"Points, produced {len(selected_candidate_set)}"
        )

    selected_points = candidate_points[selected_candidate_indices]
    sampling_mask = zeros_like(clipped_coarse)
    sampling_mask[selected_points[:, 0], selected_points[:, 1]] = 1
    cluster_labels = zeros_like(coarse, dtype=long) - 1
    cluster_labels[valid_receiving_area] = as_tensor(
        labels, device=coarse.device, dtype=long
    )

    diagnostics = SamplingDiagnostics(
        coarse_map=_restore_channel(clipped_coarse, coarse_had_channel),
        normalized_gradient=_restore_channel(
            normalized_gradient, coarse_had_channel
        ),
        normalized_distance=_restore_channel(
            normalized_distance, coarse_had_channel
        ),
        score=_restore_channel(score, coarse_had_channel),
        cluster_labels=_restore_channel(cluster_labels, coarse_had_channel),
        selected_points=selected_points,
        transmitter_point=transmitter_point,
    )
    return _restore_channel(sampling_mask, coarse_had_channel), diagnostics
