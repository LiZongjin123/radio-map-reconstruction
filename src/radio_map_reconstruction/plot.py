import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from radio_map_reconstruction.artifacts import EVALUATION_BUNDLE_SAMPLE_COUNTS
from radio_map_reconstruction.config import CONFIG, reconstructor_run_dir


REQUIRED_BUNDLE_FIELDS = {
    "sample_count",
    "ground_truth",
    "sampling_mask",
    "sparse_map",
    "reconstruction",
    "absolute_error",
}


@dataclass(frozen=True)
class EvaluationBundle:
    sample_count: int
    ground_truth: np.ndarray
    sampling_mask: np.ndarray
    sparse_map: np.ndarray
    reconstruction: np.ndarray
    absolute_error: np.ndarray


def _read_numeric_columns(
    path: Path,
    columns: tuple[str, ...],
) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = [column for column in columns if column not in fieldnames]
        if missing_columns:
            raise ValueError(
                f"{path}: missing required columns: {', '.join(missing_columns)}"
            )
        for row in reader:
            for column in columns:
                raw_value = row[column]
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{path}: row {reader.line_num}, column '{column}' value "
                        f"{raw_value!r} must be numeric"
                    ) from None
                values[column].append(numeric_value)
    return values


def _save_curve(
    *,
    x_values: list[float],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axes.plot(x_values, y_values, marker="o", linewidth=1.8)
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_yscale("linear")
    axes.grid(alpha=0.3)
    axes.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _load_evaluation_bundles(evaluation_dir: Path) -> list[EvaluationBundle]:
    bundle_paths = sorted(evaluation_dir.glob("evaluation_bundle_*.npz"))
    if len(bundle_paths) != len(EVALUATION_BUNDLE_SAMPLE_COUNTS):
        raise ValueError(
            f"{evaluation_dir}: expected exactly "
            f"{len(EVALUATION_BUNDLE_SAMPLE_COUNTS)} Evaluation Bundles, "
            f"found {len(bundle_paths)}"
        )

    bundles = []
    for path in bundle_paths:
        with np.load(path, allow_pickle=False) as archive:
            missing_fields = sorted(REQUIRED_BUNDLE_FIELDS - set(archive.files))
            if missing_fields:
                raise ValueError(
                    f"{path}: missing required fields: {', '.join(missing_fields)}"
                )
            sample_count_value = archive["sample_count"]
            if sample_count_value.ndim != 0:
                raise ValueError(f"{path}: sample_count must be a scalar integer")
            sample_count_scalar = sample_count_value.item()
            if (
                isinstance(sample_count_scalar, (bool, np.bool_))
                or not isinstance(sample_count_scalar, (int, np.integer))
            ):
                raise ValueError(f"{path}: sample_count must be a scalar integer")

            arrays = {
                field: np.asarray(archive[field]).copy()
                for field in REQUIRED_BUNDLE_FIELDS - {"sample_count"}
            }

        shapes = {field: array.shape for field, array in arrays.items()}
        if any(array.ndim != 2 for array in arrays.values()) or len(
            set(shapes.values())
        ) != 1:
            raise ValueError(
                f"{path}: Bundle arrays must have one compatible two-dimensional "
                f"shape; found {shapes}"
            )
        if not np.all(np.isin(arrays["sampling_mask"], (0, 1))):
            raise ValueError(f"{path}: sampling_mask must contain only 0 and 1")
        for field in ("ground_truth", "sparse_map", "reconstruction"):
            array = arrays[field]
            if (
                not np.issubdtype(array.dtype, np.number)
                or not np.isrealobj(array)
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    f"{path}: {field} must contain only finite numbers "
                    "with real values"
                )
            if np.any((array < 0) | (array > 1)):
                raise ValueError(f"{path}: {field} values must be within [0, 1]")
        absolute_error = arrays["absolute_error"]
        if not np.issubdtype(
            absolute_error.dtype, np.number
        ) or not np.isrealobj(absolute_error):
            raise ValueError(f"{path}: absolute_error must contain real numbers")
        if np.any(absolute_error[np.isfinite(absolute_error)] < 0):
            raise ValueError(
                f"{path}: finite absolute_error values must be nonnegative"
            )

        bundles.append(
            EvaluationBundle(
                sample_count=int(sample_count_scalar),
                ground_truth=arrays["ground_truth"],
                sampling_mask=arrays["sampling_mask"].astype(bool),
                sparse_map=arrays["sparse_map"],
                reconstruction=arrays["reconstruction"],
                absolute_error=absolute_error,
            )
        )

    sample_counts = [bundle.sample_count for bundle in bundles]
    if sorted(sample_counts) != list(EVALUATION_BUNDLE_SAMPLE_COUNTS):
        raise ValueError(
            "Evaluation Bundles must represent exactly the Valid Sampling Point "
            f"counts {EVALUATION_BUNDLE_SAMPLE_COUNTS}; "
            f"found {sorted(sample_counts)}"
        )
    return sorted(bundles, key=lambda bundle: bundle.sample_count)


def _save_evaluation_bundle_figure(
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


def run_plotting(*, run_dir: str | Path, plots_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plots_dir = Path(plots_dir)
    history = _read_numeric_columns(
        run_dir / "history.csv",
        ("epoch", "train_loss", "val_loss"),
    )
    test_metrics = _read_numeric_columns(
        run_dir / "evaluation" / "test_metrics.csv",
        ("sample_count", "mean_per_sample_normalized_rmse"),
    )
    bundles = _load_evaluation_bundles(run_dir / "evaluation")
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
    plots_dir.mkdir(parents=True, exist_ok=True)

    _save_curve(
        x_values=history["epoch"],
        y_values=history["train_loss"],
        title="Training Loss vs. Epoch",
        x_label="Epoch",
        y_label="Training Loss",
        output_path=plots_dir / "training_loss_vs_epoch.png",
    )
    _save_curve(
        x_values=history["epoch"],
        y_values=history["val_loss"],
        title="Validation Loss vs. Epoch",
        x_label="Epoch",
        y_label="Validation Loss",
        output_path=plots_dir / "validation_loss_vs_epoch.png",
    )
    _save_curve(
        x_values=test_metrics["sample_count"],
        y_values=test_metrics["mean_per_sample_normalized_rmse"],
        title="Reconstruction Error vs. Valid Sampling Points",
        x_label="Number of Valid Sampling Points",
        y_label="Mean Per-Sample Normalized RMSE",
        output_path=plots_dir / "rmse_vs_sample_count.png",
    )
    for bundle in bundles:
        _save_evaluation_bundle_figure(
            bundle,
            error_upper_limit=error_upper_limit,
            output_path=(
                plots_dir
                / f"evaluation_bundle_{bundle.sample_count}_samples.png"
            ),
        )


def plot_results() -> None:
    run_dir = reconstructor_run_dir()
    run_plotting(
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
    )
