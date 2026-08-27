import csv
from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import image as mpimg
from PIL import Image
from pytest import approx, fixture, mark, raises

from radio_map_reconstruction.plot import run_plotting


EXPECTED_PLOT_NAMES = {
    "training_loss_vs_epoch.png",
    "validation_loss_vs_epoch.png",
    "rmse_vs_sample_count.png",
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


def test_plotting_writes_three_report_ready_png_files(temporary_run_dir):
    run_dir = temporary_run_dir
    plots_dir = run_dir / "plots"
    write_metric_csvs(run_dir)

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
    plots_dir.mkdir(parents=True)
    unrelated_file = plots_dir / "keep.txt"
    unrelated_file.write_bytes(b"keep this file")
    input_contents = {
        path: path.read_bytes()
        for path in (
            run_dir / "history.csv",
            run_dir / "evaluation" / "test_metrics.csv",
        )
    }

    run_plotting(run_dir=run_dir, plots_dir=plots_dir)
    owned_plot = plots_dir / "training_loss_vs_epoch.png"
    owned_plot.write_bytes(b"stale owned output")
    run_plotting(run_dir=run_dir, plots_dir=plots_dir)

    assert owned_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
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
