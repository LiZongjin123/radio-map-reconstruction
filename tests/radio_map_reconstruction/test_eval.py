import csv
from pathlib import Path

import numpy as np
from pytest import approx
from torch import Tensor, are_deterministic_algorithms_enabled, tensor
from torch.nn import Module, Parameter
from torch.utils.data import Dataset

from radio_map_reconstruction.artifacts import SAMPLING_DIAGNOSTIC_CASES
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.eval import (
    CONFIG,
    StrategyRmse,
    run_evaluation,
)
import radio_map_reconstruction.eval as eval_module


class FixedEvaluationDataset(Dataset):
    def __len__(self) -> int:
        return 8 * len(RadioDataset.EVALUATION_SAMPLE_COUNTS)

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


class FlatCoarseModel(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        assert are_deterministic_algorithms_enabled()
        return tensor(0.5).expand(inputs.shape[0], 1, *inputs.shape[-2:])


class RecordingReconstructor(Module):
    def __init__(self):
        super().__init__()
        self.dummy_parameter = Parameter(tensor(0.0))
        self.calls: list[tuple[list[int], list[float]]] = []

    def forward(self, inputs: Tensor) -> Tensor:
        assert are_deterministic_algorithms_enabled()
        mask_sums = inputs[:, 3].flatten(1).sum(1)
        sparse_sums = inputs[:, 0].flatten(1).sum(1)
        self.calls.append(
            (
                mask_sums.tolist(),
                (sparse_sums / mask_sums).tolist(),
            )
        )
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


def read_png_bytes(evaluation_dir):
    return {
        path.name: path.read_bytes()
        for path in sorted(evaluation_dir.glob("*.png"))
    }


def test_unified_evaluation_writes_reproducible_comparison_metrics_and_figures(
    monkeypatch, tmp_path
):
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    assert SAMPLING_DIAGNOSTIC_CASES == (
        (4, 10),
        (5, 50),
        (6, 100),
        (7, 200),
    )
    monkeypatch.setitem(CONFIG, "device", "cpu")
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    sentinel = evaluation_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    sampler_config = {
        "alpha": 0.5,
        "weight_epsilon": 1e-6,
        "max_iter": 30,
        "tolerance": 1e-4,
    }

    def run_once():
        model = RecordingReconstructor()
        rmse_by_sample_count = run_evaluation(
            model=model,
            coarse_model=FlatCoarseModel(),
            test_dataset=FixedEvaluationDataset(),
            sampler_config=sampler_config,
            global_seed=42,
            evaluation_dir=evaluation_dir,
        )
        with (evaluation_dir / "test_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        return model, rmse_by_sample_count, rows, read_png_bytes(
            evaluation_dir
        ), load_bundles(evaluation_dir)

    first_model, first_rmse, first_rows, first_pngs, first_bundles = (
        run_once()
    )
    second_model, second_rmse, second_rows, second_pngs, second_bundles = (
        run_once()
    )

    assert list(first_rows[0]) == [
        "sample_count",
        "random_mean_per_sample_normalized_rmse",
        "guided_mean_per_sample_normalized_rmse",
    ]
    assert len(first_rows) == 10
    assert [int(row["sample_count"]) for row in first_rows] == list(
        RadioDataset.EVALUATION_SAMPLE_COUNTS
    )
    assert [
        float(row["random_mean_per_sample_normalized_rmse"])
        for row in first_rows
    ] == approx([0.55] * 10)
    assert [
        float(row["guided_mean_per_sample_normalized_rmse"])
        for row in first_rows
    ] == approx([0.55] * 10)
    assert list(first_rmse) == list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
    for comparison in first_rmse.values():
        assert comparison.random_rmse == approx(0.55)
        assert comparison.guided_rmse == approx(0.55)
    assert second_rmse == first_rmse
    assert second_rows == first_rows

    for call_index, (mask_sums, sparse_values) in enumerate(
        first_model.calls
    ):
        sample_index = call_index // 2
        ground_truth_value = (sample_index + 1) / 10
        assert mask_sums == approx(
            list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        assert sparse_values == approx([ground_truth_value] * 10)
    assert len(first_model.calls) == 16
    assert first_model.calls == second_model.calls

    assert sorted(first_bundles) == [
        f"evaluation_bundle_{index}.npz" for index in range(4)
    ]
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

    assert set(first_pngs) == {
        "rmse_vs_sample_count.png",
        "evaluation_bundle_10_samples.png",
        "evaluation_bundle_50_samples.png",
        "evaluation_bundle_100_samples.png",
        "evaluation_bundle_200_samples.png",
        "gradient_distance_weighted_clustering_sampling_strategy_"
        "diagnostics_10_samples.png",
        "gradient_distance_weighted_clustering_sampling_strategy_"
        "diagnostics_50_samples.png",
        "gradient_distance_weighted_clustering_sampling_strategy_"
        "diagnostics_100_samples.png",
        "gradient_distance_weighted_clustering_sampling_strategy_"
        "diagnostics_200_samples.png",
    }
    assert list(second_pngs) == list(first_pngs)
    for name, contents in first_pngs.items():
        assert contents == second_pngs[name]

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (
        are_deterministic_algorithms_enabled()
        == deterministic_algorithms_were_enabled
    )


def test_eval_loads_both_frozen_role_checkpoints_for_the_test_partition(
    monkeypatch, tmp_path
):
    loaded_paths = []
    model = RecordingReconstructor()

    class TinyDataset:
        EVALUATION_SAMPLE_COUNTS = RadioDataset.EVALUATION_SAMPLE_COUNTS

        def __init__(self, split):
            self.split = split

    run_kwargs = {}

    def fake_run_evaluation(**kwargs):
        run_kwargs.update(kwargs)
        return {
            count: StrategyRmse(random_rmse=0.3, guided_rmse=0.4)
            for count in RadioDataset.EVALUATION_SAMPLE_COUNTS
        }

    config = {
        "device": "cpu",
        "seed": 17,
        "sampler": {"alpha": 0.5},
    }
    monkeypatch.setattr(eval_module, "ROOT", tmp_path)
    monkeypatch.setattr(eval_module, "CONFIG", config)
    monkeypatch.setattr(eval_module, "RadioDataset", TinyDataset)
    monkeypatch.setattr(
        eval_module,
        "load_checkpoint",
        lambda path: loaded_paths.append(("reconstructor", Path(path))) or model,
    )
    monkeypatch.setattr(
        eval_module,
        "load_coarse_checkpoint",
        lambda path: loaded_paths.append(("coarse_reconstructor", Path(path)))
        or model,
    )
    monkeypatch.setattr(eval_module, "run_evaluation", fake_run_evaluation)

    eval_module.eval()

    assert loaded_paths == [
        ("reconstructor", tmp_path / "run" / "reconstructor" / "best.pt"),
        (
            "coarse_reconstructor",
            tmp_path / "run" / "coarse_reconstructor" / "best.pt",
        ),
    ]
    assert isinstance(run_kwargs["test_dataset"], TinyDataset)
    assert run_kwargs["sampler_config"] == {"alpha": 0.5}
    assert run_kwargs["global_seed"] == 17
    assert run_kwargs["evaluation_dir"] == tmp_path / "run" / "evaluation"
