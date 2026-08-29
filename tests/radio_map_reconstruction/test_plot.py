import csv
from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import image as mpimg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PIL import Image
from pytest import approx, fixture, mark, raises

from radio_map_reconstruction.plot import (
    SamplingDiagnosticData,
    run_plotting,
    save_sampling_diagnostic_figure,
    save_rmse_comparison_curve,
)


EXPECTED_PLOT_NAMES = {
    "training_loss_vs_epoch.png",
    "validation_loss_vs_epoch.png",
    "rmse_vs_sample_count.png",
    "evaluation_bundle_10_samples.png",
    "evaluation_bundle_50_samples.png",
    "evaluation_bundle_100_samples.png",
    "evaluation_bundle_200_samples.png",
}


@fixture
def temporary_run_dir() -> Iterator[Path]:
    with TemporaryDirectory(prefix="radio-map-reconstruction-tests-") as temp_dir:
        yield Path(temp_dir) / "run"


def test_plot_results_project_command_is_registered():
    matching_commands = [
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == "plot-results"
    ]

    assert len(matching_commands) == 1
    assert matching_commands[0].value == "radio_map_reconstruction.plot:plot_results"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_csvs(run_dir):
    write_csv(
        run_dir / "history.csv",
        ("epoch", "train_loss", "val_loss"),
        (
            {"epoch": 1, "train_loss": 0.00030, "val_loss": 0.00040},
            {"epoch": 2, "train_loss": 0.00020, "val_loss": 0.00035},
            {"epoch": 3, "train_loss": 0.00015, "val_loss": 0.00031},
        ),
    )
    write_csv(
        run_dir / "evaluation" / "test_metrics.csv",
        ("sample_count", "mean_per_sample_normalized_rmse"),
        (
            {"sample_count": 10, "mean_per_sample_normalized_rmse": 0.30},
            {"sample_count": 50, "mean_per_sample_normalized_rmse": 0.20},
            {"sample_count": 100, "mean_per_sample_normalized_rmse": 0.15},
        ),
    )


def write_evaluation_bundles(run_dir):
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for index, sample_count in enumerate((10, 50, 100, 200)):
        ground_truth = np.linspace(0, 1, 24).reshape(4, 6)
        sampling_mask = np.zeros((4, 6), dtype=bool)
        sampling_mask[index, index : index + 2] = True
        sparse_map = np.where(sampling_mask, ground_truth, 0)
        reconstruction = np.clip(ground_truth + index / 20, 0, 1)
        absolute_error = np.abs(reconstruction - ground_truth)
        absolute_error[0, 0] = np.nan
        np.savez_compressed(
            evaluation_dir / f"evaluation_bundle_{index}.npz",
            sample_count=np.asarray(sample_count),
            ground_truth=ground_truth,
            sampling_mask=sampling_mask,
            sparse_map=sparse_map,
            reconstruction=reconstruction,
            absolute_error=absolute_error,
        )


def replace_bundle(path, **updates):
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in archive.files}
    fields.update(updates)
    np.savez_compressed(path, **fields)


def test_plotting_writes_seven_report_ready_png_files(temporary_run_dir):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_evaluation_bundles(run_dir)

    run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert {path.name for path in plots_dir.iterdir()} == EXPECTED_PLOT_NAMES
    for plot_path in plots_dir.iterdir():
        assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert mpimg.imread(plot_path).size > 0
        with Image.open(plot_path) as image:
            assert image.format == "PNG"
            assert image.info["dpi"] == approx((300, 300), abs=0.1)


def test_repeated_plotting_replaces_owned_plots_and_preserves_other_files(
    temporary_run_dir,
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_evaluation_bundles(run_dir)
    plots_dir.mkdir(parents=True)
    unrelated_file = plots_dir / "keep.txt"
    unrelated_file.write_bytes(b"keep this file")
    input_contents = {
        path: path.read_bytes()
        for path in (
            run_dir / "history.csv",
            run_dir / "evaluation" / "test_metrics.csv",
            *sorted((run_dir / "evaluation").glob("evaluation_bundle_*.npz")),
        )
    }

    run_plotting(run_dir=run_dir, plots_dir=plots_dir)
    for plot_name in EXPECTED_PLOT_NAMES:
        (plots_dir / plot_name).write_bytes(b"stale owned output")
    run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    for plot_name in EXPECTED_PLOT_NAMES:
        assert (plots_dir / plot_name).read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
    assert unrelated_file.read_bytes() == b"keep this file"
    assert {path.name for path in plots_dir.iterdir()} == (
        EXPECTED_PLOT_NAMES | {"keep.txt"}
    )
    for path, original_contents in input_contents.items():
        assert path.read_bytes() == original_contents


@mark.parametrize(
    "missing_relative_path",
    ("history.csv", "evaluation/test_metrics.csv"),
)
def test_plotting_fails_clearly_when_required_csv_is_missing(
    temporary_run_dir, missing_relative_path
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    (run_dir / missing_relative_path).unlink()

    with raises(FileNotFoundError, match="Required CSV.*" + missing_relative_path.split("/")[-1]):
        run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert not plots_dir.exists()


@mark.parametrize(
    ("malformation", "error_pattern"),
    (
        ("missing_bundle", "expected exactly 4.*found 3"),
        ("extra_bundle", "expected exactly 4.*found 5"),
        ("missing_field", "missing required fields.*absolute_error"),
        ("nonscalar_sample_count", "sample_count must be a scalar integer"),
        ("unexpected_sample_count", "must represent exactly.*10.*50.*100.*200"),
        ("incompatible_shape", "compatible two-dimensional shape"),
        ("invalid_numerical_content", "ground_truth.*finite numbers"),
        ("complex_numerical_content", "absolute_error.*real numbers"),
    ),
)
def test_plotting_rejects_malformed_evaluation_bundles_before_writing_outputs(
    temporary_run_dir, malformation, error_pattern
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_evaluation_bundles(run_dir)
    evaluation_dir = run_dir / "evaluation"
    bundle_path = evaluation_dir / "evaluation_bundle_0.npz"

    if malformation == "missing_bundle":
        bundle_path.unlink()
    elif malformation == "extra_bundle":
        extra_path = evaluation_dir / "evaluation_bundle_extra.npz"
        extra_path.write_bytes(bundle_path.read_bytes())
    elif malformation == "missing_field":
        with np.load(bundle_path, allow_pickle=False) as archive:
            fields = {
                name: archive[name].copy()
                for name in archive.files
                if name != "absolute_error"
            }
        np.savez_compressed(bundle_path, **fields)
    elif malformation == "nonscalar_sample_count":
        replace_bundle(bundle_path, sample_count=np.asarray([10]))
    elif malformation == "unexpected_sample_count":
        replace_bundle(bundle_path, sample_count=np.asarray(25))
    elif malformation == "incompatible_shape":
        replace_bundle(bundle_path, reconstruction=np.zeros((2, 2)))
    elif malformation == "invalid_numerical_content":
        invalid_ground_truth = np.linspace(0, 1, 24).reshape(4, 6)
        invalid_ground_truth[0, 0] = np.nan
        replace_bundle(bundle_path, ground_truth=invalid_ground_truth)
    elif malformation == "complex_numerical_content":
        replace_bundle(
            bundle_path,
            absolute_error=np.zeros((4, 6), dtype=complex),
        )

    with raises(ValueError, match=error_pattern):
        run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert not plots_dir.exists()


def test_plotting_reads_comparison_metrics_csv_with_both_strategies(
    temporary_run_dir,
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_csv(
        run_dir / "evaluation" / "test_metrics.csv",
        (
            "sample_count",
            "random_mean_per_sample_normalized_rmse",
            "guided_mean_per_sample_normalized_rmse",
        ),
        (
            {
                "sample_count": 10,
                "random_mean_per_sample_normalized_rmse": 0.30,
                "guided_mean_per_sample_normalized_rmse": 0.25,
            },
            {
                "sample_count": 50,
                "random_mean_per_sample_normalized_rmse": 0.20,
                "guided_mean_per_sample_normalized_rmse": 0.15,
            },
            {
                "sample_count": 100,
                "random_mean_per_sample_normalized_rmse": 0.15,
                "guided_mean_per_sample_normalized_rmse": 0.10,
            },
        ),
    )
    write_evaluation_bundles(run_dir)

    run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert {path.name for path in plots_dir.iterdir()} == EXPECTED_PLOT_NAMES


def test_rmse_comparison_curve_labels_both_strategies(tmp_path, monkeypatch):
    captured_figure = None
    original_savefig = Figure.savefig

    def capture_curve_figure(figure, output_path, *args, **kwargs):
        nonlocal captured_figure
        if Path(output_path).name == "rmse_vs_sample_count.png":
            captured_figure = figure
            figure.canvas.draw()
        return original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_curve_figure)

    save_rmse_comparison_curve(
        sample_counts=[10, 50, 100],
        random_rmse=[0.30, 0.20, 0.15],
        guided_rmse=[0.25, 0.15, 0.10],
        output_path=tmp_path / "rmse_vs_sample_count.png",
    )

    assert captured_figure is not None
    axes = captured_figure.axes[0]
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Random Sampling",
        "Guided Sampling",
    ]
    assert axes.get_title() == "Reconstruction Error vs. Valid Sampling Points"
    assert axes.get_xlabel() == "Number of Valid Sampling Points"
    assert axes.get_ylabel() == "Mean Per-Sample Normalized RMSE"
    assert len(axes.get_lines()) == 2


def test_evaluation_bundle_figure_has_report_ready_rendering(
    temporary_run_dir, monkeypatch
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_evaluation_bundles(run_dir)
    captured_figure = None
    original_savefig = Figure.savefig

    def capture_bundle_figure(figure, output_path, *args, **kwargs):
        nonlocal captured_figure
        if Path(output_path).name == "evaluation_bundle_10_samples.png":
            captured_figure = figure
            figure.canvas.draw()
        return original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_bundle_figure)

    run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert captured_figure is not None
    image_axes = captured_figure.axes[:4]
    assert [axes.get_title() for axes in image_axes] == [
        "Ground Truth",
        "Sparse Map",
        "Reconstruction",
        "Absolute Error",
    ]
    assert len(captured_figure.axes) == 6
    assert captured_figure.axes[4].get_ylabel() == "Normalized Signal"
    assert captured_figure.axes[5].get_ylabel() == "Absolute Error"
    assert captured_figure.get_suptitle() == (
        "Evaluation Bundle Figure — 10 Valid Sampling Points"
    )

    signal_images = [axes.images[0] for axes in image_axes[:3]]
    assert all(image.norm is signal_images[0].norm for image in signal_images)
    assert signal_images[0].norm.vmin == 0
    assert signal_images[0].norm.vmax == 1
    sparse_image = signal_images[1]
    sparse_values = sparse_image.get_array()
    assert np.ma.getmaskarray(sparse_values)[1, 0]
    assert not np.ma.getmaskarray(sparse_values)[0, 0]
    assert sparse_values[0, 0] == 0
    assert np.allclose(sparse_image.cmap.get_bad(), to_rgba("lightgray"))
    assert not np.allclose(
        sparse_image.cmap(sparse_image.norm(0)), sparse_image.cmap.get_bad()
    )

    with Image.open(plots_dir / "evaluation_bundle_10_samples.png") as image:
        assert image.width >= 2500
        assert image.height >= 1800


def test_sampling_diagnostic_figure_shows_decision_chain_and_actual_mask(
    tmp_path, monkeypatch
):
    captured_figure = None
    original_savefig = Figure.savefig

    def capture_figure(figure, output_path, *args, **kwargs):
        nonlocal captured_figure
        captured_figure = figure
        figure.canvas.draw()
        return original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    invalid_area = np.zeros((3, 4), dtype=bool)
    invalid_area[0, 0] = True
    invalid_area[0, 1] = True
    sampling_mask = np.zeros((3, 4), dtype=bool)
    sampling_mask[1, 1] = True
    sampling_mask[2, 3] = True
    cluster_labels = np.arange(12).reshape(3, 4)
    cluster_labels[invalid_area] = -1
    diagnostic = SamplingDiagnosticData(
        sample_count=2,
        coarse_map=np.linspace(0, 1, 12).reshape(3, 4),
        normalized_gradient=np.linspace(1, 0, 12).reshape(3, 4),
        normalized_distance=np.linspace(0.2, 0.8, 12).reshape(3, 4),
        score=np.linspace(0.1, 0.9, 12).reshape(3, 4),
        cluster_labels=cluster_labels,
        sampling_mask=sampling_mask,
        valid_receiving_area=~invalid_area,
        transmitter_point=np.asarray([0, 0]),
    )

    output_path = (
        tmp_path
        / "gradient_distance_weighted_clustering_sampling_strategy_"
        "diagnostics_2_samples.png"
    )
    save_sampling_diagnostic_figure(diagnostic, output_path=output_path)

    assert captured_figure is not None
    assert [axes.get_title() for axes in captured_figure.axes] == [
        "Coarse Reconstruction",
        "Normalized Gradient",
        "Normalized Distance",
        "Sampling Score",
        "Clustering and Sampling",
    ]
    assert captured_figure.get_suptitle() == (
        "Gradient-Distance Weighted Clustering Sampling Strategy — "
        "2 Valid Sampling Points"
    )
    for axes in captured_figure.axes:
        assert np.array_equal(
            np.ma.getmaskarray(axes.images[0].get_array()), invalid_area
        )
        assert np.allclose(axes.images[0].cmap.get_bad(), to_rgba("lightgray"))
        assert not axes.texts
    cluster_axes = captured_figure.axes[-1]
    collections_by_label = {
        collection.get_label(): collection
        for collection in cluster_axes.collections
    }
    sampling_points = collections_by_label["Valid Sampling Points"]
    assert np.array_equal(
        sampling_points.get_offsets(),
        np.asarray([[1, 1], [3, 2]]),
    )
    assert sampling_points.get_facecolors().size == 0
    transmitter = collections_by_label["Transmitter"]
    assert transmitter.get_offsets().tolist() == [[0.0, 0.0]]
    assert len(transmitter.get_paths()[0].vertices) == 11
    assert output_path.is_file()


@mark.parametrize(
    ("relative_path", "fieldnames", "rows", "missing_column"),
    (
        (
            "history.csv",
            ("epoch", "train_loss"),
            ({"epoch": 1, "train_loss": 0.1},),
            "val_loss",
        ),
        (
            "evaluation/test_metrics.csv",
            ("sample_count",),
            ({"sample_count": 10},),
            "mean_per_sample_normalized_rmse",
        ),
    ),
)
def test_plotting_fails_clearly_when_csv_column_is_missing(
    temporary_run_dir, relative_path, fieldnames, rows, missing_column
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_csv(run_dir / relative_path, fieldnames, rows)

    with raises(
        ValueError,
        match=f"{relative_path.split('/')[-1]}.*missing required columns.*{missing_column}",
    ):
        run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert not plots_dir.exists()


@mark.parametrize(
    ("relative_path", "fieldnames", "rows", "invalid_column"),
    (
        (
            "history.csv",
            ("epoch", "train_loss", "val_loss"),
            ({"epoch": 1, "train_loss": "not-a-number", "val_loss": 0.1},),
            "train_loss",
        ),
        (
            "evaluation/test_metrics.csv",
            ("sample_count", "mean_per_sample_normalized_rmse"),
            (
                {
                    "sample_count": 10,
                    "mean_per_sample_normalized_rmse": "not-a-number",
                },
            ),
            "mean_per_sample_normalized_rmse",
        ),
    ),
)
def test_plotting_fails_clearly_when_csv_value_is_nonnumeric(
    temporary_run_dir, relative_path, fieldnames, rows, invalid_column
):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)
    write_csv(run_dir / relative_path, fieldnames, rows)

    with raises(
        ValueError,
        match=(
            f"{relative_path.split('/')[-1]}.*row 2.*{invalid_column}"
            ".*not-a-number.*must be numeric"
        ),
    ):
        run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert not plots_dir.exists()
