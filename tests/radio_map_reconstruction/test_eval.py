import csv

import numpy as np
from pytest import approx
from torch import Tensor, are_deterministic_algorithms_enabled, tensor
from torch.nn import Module, Parameter
from torch.utils.data import Dataset

from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.eval import CONFIG, run_evaluation
from radio_map_reconstruction.loss import RadioMapLoss


class FixedEvaluationDataset(Dataset):
    def __len__(self) -> int:
        return 4 * len(RadioDataset.EVALUATION_SAMPLE_COUNTS)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample_index, sample_count_index = divmod(
            index, len(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        sample_count = RadioDataset.EVALUATION_SAMPLE_COUNTS[sample_count_index]
        ground_truth_value = (sample_index + 1) / 10

        inputs = tensor(0.0).new_zeros((4, 1, 202))
        inputs[1, 0, 0] = 1
        inputs[2, 0, 1] = 1
        inputs[3, 0, 2 : 2 + sample_count] = 1
        inputs[0] = ground_truth_value * inputs[3]

        ground_truth = tensor(ground_truth_value).expand(1, 1, 202).clone()
        ground_truth[0, 0, 0] = 1
        ground_truth[0, 0, 1] = 0
        valid_receiving_area = tensor(True).expand(1, 1, 202).clone()
        valid_receiving_area[0, 0, :2] = False
        return inputs, {
            "gain": ground_truth,
            "mask": valid_receiving_area,
        }


class OutOfRangeModel(Module):
    def __init__(self):
        super().__init__()
        self.dummy_parameter = Parameter(tensor(0.0))

    def forward(self, inputs: Tensor) -> Tensor:
        assert are_deterministic_algorithms_enabled()
        return (
            tensor(1.5).expand(inputs.shape[0], 1, 1, 202)
            + self.dummy_parameter * 0
        )


def load_bundles(evaluation_dir):
    bundles = {}
    for path in sorted(evaluation_dir.glob("evaluation_bundle_*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            bundles[path.name] = {name: bundle[name].copy() for name in bundle.files}
    return bundles


def test_evaluation_writes_reproducible_metrics_and_bundles_without_png(
    monkeypatch, tmp_path
):
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    monkeypatch.setitem(CONFIG, "device", "cpu")
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    sentinel = evaluation_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    existing_png = evaluation_dir / "keep.png"
    existing_png.write_bytes(b"existing")

    def run_once():
        run_evaluation(
            model=OutOfRangeModel(),
            test_dataset=FixedEvaluationDataset(),
            criterion=RadioMapLoss(),
            evaluation_dir=evaluation_dir,
            batch_size=7,
        )
        with (evaluation_dir / "test_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        return rows, load_bundles(evaluation_dir)

    first_rows, first_bundles = run_once()
    second_rows, second_bundles = run_once()

    assert list(first_rows[0]) == [
        "sample_count",
        "mean_per_sample_normalized_rmse",
    ]
    assert [int(row["sample_count"]) for row in first_rows] == list(
        RadioDataset.EVALUATION_SAMPLE_COUNTS
    )
    assert [
        float(row["mean_per_sample_normalized_rmse"]) for row in first_rows
    ] == approx([0.75] * 10)
    assert second_rows == first_rows

    assert list(first_bundles) == [
        f"evaluation_bundle_{index}.npz" for index in range(4)
    ]
    assert list(second_bundles) == list(first_bundles)
    expected_fields = {
        "sample_count",
        "ground_truth",
        "sampling_mask",
        "sparse_map",
        "reconstruction",
        "absolute_error",
    }
    for index, sample_count in enumerate((10, 50, 100, 200)):
        bundle = first_bundles[f"evaluation_bundle_{index}.npz"]
        assert set(bundle) == expected_fields
        assert bundle["sample_count"].item() == sample_count
        assert bundle["ground_truth"].shape == (1, 202)
        assert bundle["sampling_mask"].sum() == sample_count
        assert np.all(bundle["sampling_mask"][:, :2] == 0)
        assert np.all(bundle["sparse_map"][bundle["sampling_mask"] == 0] == 0)
        assert bundle["reconstruction"][0, 0] == 1
        assert bundle["reconstruction"][0, 1] == 0
        assert np.all(bundle["reconstruction"][0, 2:] == 1)
        assert np.all(np.isnan(bundle["absolute_error"][0, :2]))
        assert np.allclose(
            bundle["absolute_error"][0, 2:], 1 - (index + 1) / 10
        )

        repeated = second_bundles[f"evaluation_bundle_{index}.npz"]
        for field in expected_fields:
            assert np.array_equal(bundle[field], repeated[field], equal_nan=True)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(evaluation_dir.glob("*.png")) == [existing_png]
    assert (
        are_deterministic_algorithms_enabled()
        == deterministic_algorithms_were_enabled
    )
