import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


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


def plot_results() -> None:
    run_plotting(run_dir=ROOT / "run", plots_dir=ROOT / "run" / "plots")
