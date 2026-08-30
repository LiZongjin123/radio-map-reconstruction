import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from radio_map_reconstruction.plot import (
    SamplingDiagnosticData,
    save_sampling_diagnostic_figure,
    save_rmse_comparison_curve,
)


def test_rmse_comparison_curve_presents_all_strategies_in_required_order(
    tmp_path, monkeypatch
):
    captured_figure = None
    original_savefig = Figure.savefig

    def capture_curve_figure(figure, output_path, *args, **kwargs):
        nonlocal captured_figure
        captured_figure = figure
        figure.canvas.draw()
        return original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_curve_figure)
    output_path = tmp_path / "rmse_vs_sample_count.png"
    save_rmse_comparison_curve(
        sample_counts=[10, 50, 100],
        random_rmse=[0.30, 0.20, 0.15],
        regular_grid_rmse=[0.28, 0.18, 0.12],
        guided_rmse=[0.25, 0.15, 0.10],
        output_path=output_path,
    )

    assert output_path.is_file()
    assert [
        text.get_text()
        for text in captured_figure.axes[0].get_legend().get_texts()
    ] == [
        "Random Sampling",
        "Regular-Grid Sampling",
        "Guided Sampling",
    ]
    assert [line.get_marker() for line in captured_figure.axes[0].lines] == [
        "o",
        "^",
        "s",
    ]
    assert [line.get_linestyle() for line in captured_figure.axes[0].lines] == [
        "-",
        "-",
        "-",
    ]
    assert [line.get_xdata().tolist() for line in captured_figure.axes[0].lines] == [
        [10, 50, 100],
        [10, 50, 100],
        [10, 50, 100],
    ]
    assert [line.get_ydata().tolist() for line in captured_figure.axes[0].lines] == [
        [0.30, 0.20, 0.15],
        [0.28, 0.18, 0.12],
        [0.25, 0.15, 0.10],
    ]


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
    invalid_area[0, :2] = True
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

    output_path = tmp_path / "sampling_diagnostics_2_samples.png"
    save_sampling_diagnostic_figure(diagnostic, output_path=output_path)

    assert output_path.is_file()
    assert [axes.get_title() for axes in captured_figure.axes] == [
        "Coarse Reconstruction",
        "Normalized Gradient",
        "Normalized Distance",
        "Sampling Score",
        "Clustering and Sampling",
    ]
    for axes in captured_figure.axes:
        assert np.array_equal(
            np.ma.getmaskarray(axes.images[0].get_array()), invalid_area
        )
        assert np.allclose(axes.images[0].cmap.get_bad(), to_rgba("lightgray"))
