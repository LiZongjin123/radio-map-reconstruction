from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize



@dataclass(frozen=True)
class EvaluationBundle:
    sample_count: int
    ground_truth: np.ndarray
    sampling_mask: np.ndarray
    sparse_map: np.ndarray
    reconstruction: np.ndarray
    absolute_error: np.ndarray


@dataclass(frozen=True)
class SamplingDiagnosticData:
    sample_count: int
    coarse_map: np.ndarray
    normalized_gradient: np.ndarray
    normalized_distance: np.ndarray
    score: np.ndarray
    cluster_labels: np.ndarray
    sampling_mask: np.ndarray
    valid_receiving_area: np.ndarray
    transmitter_point: np.ndarray


def save_rmse_comparison_curve(
    *,
    sample_counts: Sequence[int],
    random_rmse: Sequence[float],
    regular_grid_rmse: Sequence[float],
    guided_rmse: Sequence[float],
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axes.plot(
        list(sample_counts),
        list(random_rmse),
        marker="o",
        linewidth=1.8,
        label="Random Sampling",
    )
    axes.plot(
        list(sample_counts),
        list(regular_grid_rmse),
        marker="^",
        linewidth=1.8,
        label="Regular-Grid Sampling",
    )
    axes.plot(
        list(sample_counts),
        list(guided_rmse),
        marker="s",
        linewidth=1.8,
        label="Guided Sampling",
    )
    axes.set_title("Reconstruction Error vs. Valid Sampling Points")
    axes.set_xlabel("Number of Valid Sampling Points")
    axes.set_ylabel("Mean Per-Sample Normalized RMSE")
    axes.grid(alpha=0.3)
    axes.legend()
    axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

def evaluation_error_upper_limit(bundles: Sequence[EvaluationBundle]) -> float:
    finite_errors = np.concatenate(
        [
            bundle.absolute_error[np.isfinite(bundle.absolute_error)]
            for bundle in bundles
        ]
    )
    if finite_errors.size == 0:
        raise ValueError(
            "Evaluation Bundles must contain at least one finite Absolute Error value"
        )
    error_upper_limit = float(np.percentile(finite_errors, 99))
    if error_upper_limit == 0:
        error_upper_limit = float(np.nextafter(0, 1))
    return error_upper_limit


def save_evaluation_bundle_figure(
    bundle: EvaluationBundle,
    *,
    error_upper_limit: float,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(10, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 0.06))
    axes = np.asarray(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )
    signal_colorbar_axes = figure.add_subplot(grid[0, 2])
    error_colorbar_axes = figure.add_subplot(grid[1, 2])
    signal_normalization = Normalize(vmin=0, vmax=1, clip=True)
    error_normalization = Normalize(vmin=0, vmax=error_upper_limit, clip=True)
    signal_colormap = matplotlib.colormaps["viridis"].copy()
    sparse_colormap = signal_colormap.with_extremes(bad="lightgray")

    signal_images = []
    for axes_item, title, values in (
        (axes[0, 0], "Ground Truth", bundle.ground_truth),
        (
            axes[0, 1],
            "Sparse Map",
            np.ma.masked_where(~bundle.sampling_mask, bundle.sparse_map),
        ),
        (axes[1, 0], "Reconstruction", bundle.reconstruction),
    ):
        image = axes_item.imshow(
            values,
            cmap=sparse_colormap if title == "Sparse Map" else signal_colormap,
            norm=signal_normalization,
        )
        axes_item.set_title(title)
        axes_item.set_axis_off()
        signal_images.append(image)

    error_values = np.ma.masked_invalid(bundle.absolute_error)
    error_image = axes[1, 1].imshow(
        error_values,
        cmap=signal_colormap,
        norm=error_normalization,
    )
    axes[1, 1].set_title("Absolute Error")
    axes[1, 1].set_axis_off()
    signal_colorbar = figure.colorbar(signal_images[0], cax=signal_colorbar_axes)
    signal_colorbar.set_label("Normalized Signal")
    error_colorbar = figure.colorbar(error_image, cax=error_colorbar_axes)
    error_colorbar.set_label("Absolute Error")
    figure.suptitle(
        f"Evaluation Bundle Figure — {bundle.sample_count} Valid Sampling Points"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_sampling_diagnostic_figure(
    diagnostic: SamplingDiagnosticData,
    *,
    output_path: Path,
) -> None:
    image_fields = (
        diagnostic.coarse_map,
        diagnostic.normalized_gradient,
        diagnostic.normalized_distance,
        diagnostic.score,
        diagnostic.cluster_labels,
        diagnostic.sampling_mask,
        diagnostic.valid_receiving_area,
    )
    expected_shape = diagnostic.coarse_map.shape
    if len(expected_shape) != 2 or any(
        values.shape != expected_shape for values in image_fields
    ):
        raise ValueError("Sampling diagnostic maps must share one 2D shape")
    if diagnostic.transmitter_point.shape != (2,):
        raise ValueError("transmitter_point must contain one row-column pair")
    sampling_mask = diagnostic.sampling_mask.astype(bool)
    valid_receiving_area = diagnostic.valid_receiving_area.astype(bool)
    selected_points = np.argwhere(sampling_mask)
    if selected_points.shape[0] != diagnostic.sample_count:
        raise ValueError(
            "Sampling diagnostic must contain exactly "
            f"{diagnostic.sample_count} Valid Sampling Points"
        )
    if np.any(sampling_mask & ~valid_receiving_area):
        raise ValueError("Valid Sampling Points cannot occupy the invalid area")

    figure = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=(1, 1, 1, 1, 1, 1, 0.08),
    )
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    )
    colorbar_axes = figure.add_subplot(grid[:, 6])
    continuous_normalization = Normalize(vmin=0, vmax=1, clip=True)
    continuous_colormap = matplotlib.colormaps["viridis"].with_extremes(
        bad="lightgray"
    )
    cluster_colormap = matplotlib.colormaps["turbo"].resampled(
        diagnostic.sample_count
    ).with_extremes(bad="lightgray")
    panels = (
        ("Coarse Reconstruction", diagnostic.coarse_map, continuous_colormap),
        (
            "Normalized Gradient",
            diagnostic.normalized_gradient,
            continuous_colormap,
        ),
        (
            "Normalized Distance",
            diagnostic.normalized_distance,
            continuous_colormap,
        ),
        ("Sampling Score", diagnostic.score, continuous_colormap),
        ("Clustering and Sampling", diagnostic.cluster_labels, cluster_colormap),
    )
    continuous_image = None
    for axes_item, (title, values, colormap) in zip(axes, panels, strict=True):
        image = axes_item.imshow(
            np.ma.masked_where(~valid_receiving_area, values),
            cmap=colormap,
            **(
                {
                    "vmin": 0,
                    "vmax": diagnostic.sample_count - 1,
                }
                if title == "Clustering and Sampling"
                else {"norm": continuous_normalization}
            ),
        )
        if continuous_image is None:
            continuous_image = image
        axes_item.set_title(title)
        axes_item.set_axis_off()

    colorbar = figure.colorbar(continuous_image, cax=colorbar_axes)
    colorbar.set_label("Normalized Value")

    cluster_axes = axes[-1]
    cluster_axes.scatter(
        selected_points[:, 1],
        selected_points[:, 0],
        marker="o",
        s=28,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
        label="Valid Sampling Points",
    )
    cluster_axes.scatter(
        diagnostic.transmitter_point[1],
        diagnostic.transmitter_point[0],
        marker="*",
        s=90,
        facecolors="gold",
        edgecolors="black",
        linewidths=0.7,
        label="Transmitter",
    )
    cluster_axes.legend(loc="lower right", fontsize="small", framealpha=0.9)
    figure.suptitle(
        "Gradient-Distance Weighted Clustering Sampling Strategy — "
        f"{diagnostic.sample_count} Valid Sampling Points"
    )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


