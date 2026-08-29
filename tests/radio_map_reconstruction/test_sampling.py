from pytest import raises
from torch import Tensor, zeros, zeros_like

from radio_map_reconstruction.sampling import (
    gradient_distance_weighted_clustering_sample,
)


def synthetic_maps() -> tuple[Tensor, Tensor, Tensor]:
    coarse_map = zeros((1, 6, 6))
    coarse_map[0, 2:, 3:] = 1
    tx_map = zeros_like(coarse_map)
    tx_map[0, 0, 0] = 1
    building_map = zeros_like(coarse_map)
    building_map[0, 1, 1] = 1
    return coarse_map, tx_map, building_map


def test_gradient_distance_weighted_clustering_strategy_is_deterministic_and_returns_exact_valid_sampling_points():
    coarse_map, tx_map, building_map = synthetic_maps()

    first_mask, first_diagnostics = gradient_distance_weighted_clustering_sample(
        coarse_map,
        tx_map,
        building_map,
        sample_count=5,
        alpha=0.6,
        global_seed=42,
        sample_id="city-7-tx-2",
        weight_epsilon=1e-6,
        max_iter=100,
        tolerance=1e-4,
    )
    repeated_mask, repeated_diagnostics = (
        gradient_distance_weighted_clustering_sample(
            coarse_map,
            tx_map,
            building_map,
            sample_count=5,
            alpha=0.6,
            global_seed=42,
            sample_id="city-7-tx-2",
            weight_epsilon=1e-6,
            max_iter=100,
            tolerance=1e-4,
        )
    )

    assert first_mask.equal(repeated_mask)
    assert first_diagnostics.cluster_labels.equal(
        repeated_diagnostics.cluster_labels
    )
    assert set(first_mask.unique().tolist()) == {0.0, 1.0}
    assert first_mask.sum().item() == 5
    assert not first_mask[tx_map.bool()].any()
    assert not first_mask[building_map.bool()].any()
    assert first_diagnostics.selected_points.shape == (5, 2)
    assert all(
        first_mask[0, row, column].item() == 1
        for row, column in first_diagnostics.selected_points.tolist()
    )
    assert first_diagnostics.coarse_map.shape == coarse_map.shape
    assert first_diagnostics.normalized_gradient.shape == coarse_map.shape
    assert first_diagnostics.normalized_distance.shape == coarse_map.shape
    assert first_diagnostics.score.shape == coarse_map.shape
    assert first_diagnostics.cluster_labels.shape == coarse_map.shape
    assert set(first_diagnostics.cluster_labels[tx_map.bool()].tolist()) == {-1}
    assert set(first_diagnostics.cluster_labels[building_map.bool()].tolist()) == {
        -1
    }
    selected_cluster_labels = [
        first_diagnostics.cluster_labels[0, row, column].item()
        for row, column in first_diagnostics.selected_points.tolist()
    ]
    assert sorted(selected_cluster_labels) == list(range(5))


def test_diagnostics_clip_prediction_and_normalize_only_valid_receiving_area():
    coarse_map, tx_map, building_map = synthetic_maps()
    coarse_map[0, 0, 0] = -100
    coarse_map[0, 1, 1] = 100
    coarse_map[0, 5, 5] = 2

    _, diagnostics = gradient_distance_weighted_clustering_sample(
        coarse_map,
        tx_map,
        building_map,
        sample_count=3,
        alpha=0.25,
        global_seed=42,
        sample_id="normalization-case",
        weight_epsilon=1e-6,
        max_iter=100,
        tolerance=1e-4,
    )

    valid_receiving_area = (tx_map < 0.5) & (building_map < 0.5)
    outside_valid_receiving_area = ~valid_receiving_area
    assert diagnostics.coarse_map.min().item() == 0
    assert diagnostics.coarse_map.max().item() == 1
    assert not diagnostics.normalized_gradient[outside_valid_receiving_area].any()
    assert not diagnostics.normalized_distance[outside_valid_receiving_area].any()
    assert not diagnostics.score[outside_valid_receiving_area].any()
    assert diagnostics.normalized_gradient[valid_receiving_area].min().item() == 0
    assert diagnostics.normalized_gradient[valid_receiving_area].max().item() == 1
    assert diagnostics.normalized_distance[valid_receiving_area].min().item() == 0
    assert diagnostics.normalized_distance[valid_receiving_area].max().item() == 1
    expected_score = (
        0.25 * diagnostics.normalized_gradient
        + 0.75 * diagnostics.normalized_distance
    )
    assert diagnostics.score.equal(expected_score)


def test_degenerate_scores_still_produce_exact_points_from_positive_weight_floor():
    coarse_map, tx_map, building_map = synthetic_maps()
    coarse_map.zero_()

    sampling_mask, diagnostics = gradient_distance_weighted_clustering_sample(
        coarse_map,
        tx_map,
        building_map,
        sample_count=4,
        alpha=1.0,
        global_seed=42,
        sample_id="flat-map",
        weight_epsilon=1e-6,
        max_iter=100,
        tolerance=1e-4,
    )

    assert not diagnostics.score.any()
    assert sampling_mask.sum().item() == 4
    assert diagnostics.selected_points.unique(dim=0).shape[0] == 4


def test_equal_center_distances_are_resolved_by_flattened_pixel_index():
    coarse_map = zeros((1, 3, 3))
    tx_map = zeros_like(coarse_map)
    tx_map[0, 0, 1] = 1
    building_map = zeros_like(coarse_map) + 1
    building_map[0, 1, 0] = 0
    building_map[0, 1, 2] = 0

    sampling_mask, diagnostics = gradient_distance_weighted_clustering_sample(
        coarse_map,
        tx_map,
        building_map,
        sample_count=1,
        alpha=1.0,
        global_seed=42,
        sample_id="tie-case",
        weight_epsilon=1e-6,
        max_iter=100,
        tolerance=1e-4,
    )

    assert diagnostics.selected_points.tolist() == [[1, 0]]
    assert sampling_mask[0, 1, 0].item() == 1


def test_malformed_transmitter_maps_report_the_sample_identity():
    coarse_map, tx_map, building_map = synthetic_maps()

    for malformed_tx_map in (zeros_like(tx_map), tx_map.clone()):
        if malformed_tx_map.any():
            malformed_tx_map[0, 5, 5] = 1
        with raises(
            ValueError,
            match="sample 'broken-city-9'.*exactly one transmitter pixel",
        ):
            gradient_distance_weighted_clustering_sample(
                coarse_map,
                malformed_tx_map,
                building_map,
                sample_count=2,
                alpha=0.5,
                global_seed=42,
                sample_id="broken-city-9",
                weight_epsilon=1e-6,
                max_iter=100,
                tolerance=1e-4,
            )


def test_invalid_sample_counts_fail_before_clustering():
    coarse_map, tx_map, building_map = synthetic_maps()

    for invalid_sample_count in (0, 2.5):
        with raises(ValueError, match="sample_count must be a positive integer"):
            gradient_distance_weighted_clustering_sample(
                coarse_map,
                tx_map,
                building_map,
                sample_count=invalid_sample_count,
                alpha=0.5,
                global_seed=42,
                sample_id="invalid-count",
                weight_epsilon=1e-6,
                max_iter=100,
                tolerance=1e-4,
            )


def test_insufficient_valid_candidates_fail_with_requested_and_available_counts():
    coarse_map = zeros((1, 2, 2))
    tx_map = zeros_like(coarse_map)
    tx_map[0, 0, 0] = 1
    building_map = zeros_like(coarse_map)
    building_map[0, 0, 1] = 1

    with raises(
        ValueError,
        match="requested 3 Valid Sampling Points but only 2 candidates exist",
    ):
        gradient_distance_weighted_clustering_sample(
            coarse_map,
            tx_map,
            building_map,
            sample_count=3,
            alpha=0.5,
            global_seed=42,
            sample_id="too-small",
            weight_epsilon=1e-6,
            max_iter=100,
            tolerance=1e-4,
        )
