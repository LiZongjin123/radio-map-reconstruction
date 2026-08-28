import csv

import radio_map_reconstruction.eval as eval_module
import radio_map_reconstruction.sampler_eval as sampler_eval_module
import torch
from torch import Tensor, tensor
from torch.nn import Module, Parameter
from torch.utils.data import Dataset

from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.sampler_eval import CONFIG, run_sampler_evaluation


class DeterministicSamplerEvaluationDataset(Dataset):
    EVALUATION_SAMPLE_COUNTS = RadioDataset.EVALUATION_SAMPLE_COUNTS

    def __init__(self, part: str | None = None) -> None:
        self.part = part

    def __len__(self) -> int:
        return len(self.EVALUATION_SAMPLE_COUNTS)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample_count = self.EVALUATION_SAMPLE_COUNTS[index]
        dense_target = torch.full((1, 1, 202), 0.5)
        transmitter_map = torch.zeros_like(dense_target)
        building_map = torch.zeros_like(dense_target)
        fixed_sampling_mask = torch.zeros_like(dense_target)
        fixed_sampling_mask[..., :sample_count] = 1
        inputs = torch.cat(
            (
                dense_target * fixed_sampling_mask,
                transmitter_map,
                building_map,
                fixed_sampling_mask,
            )
        )
        return inputs, {
            "gain": dense_target,
            "mask": torch.ones_like(dense_target, dtype=torch.bool),
        }


class DeterministicSampler(Module):
    def forward(self, building_map: Tensor, transmitter_map: Tensor) -> Tensor:
        scores = torch.linspace(
            0, 1, building_map.shape[-1], device=building_map.device
        ).reshape(1, 1, 1, -1)
        return scores.expand_as(building_map) + transmitter_map * 0


class RecordingOutOfRangeReconstructor(Module):
    def __init__(self) -> None:
        super().__init__()
        self.learned_sampling_masks: list[Tensor] = []

    def forward(self, inputs: Tensor) -> Tensor:
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        self.learned_sampling_masks.extend(inputs[:, 3].detach().cpu())
        return tensor(1.5, device=inputs.device).expand(
            inputs.shape[0], 1, 1, 202
        )


def test_sampler_evaluation_is_reproducible_and_restores_runtime_settings(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_cudnn_deterministic = torch.backends.cudnn.deterministic
    prior_cudnn_benchmark = torch.backends.cudnn.benchmark
    evaluation_dir = tmp_path / "sampler" / "evaluation"
    assert RadioDataset.EVALUATION_SAMPLE_COUNTS == (
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
    )

    def run_once():
        reconstructor = RecordingOutOfRangeReconstructor()
        run_sampler_evaluation(
            sampler=DeterministicSampler(),
            reconstructor=reconstructor,
            test_dataset=DeterministicSamplerEvaluationDataset(),
            evaluation_dir=evaluation_dir,
            batch_size=4,
            sampler_training_config={
                "temperature": 0.1,
                "bisection_tolerance": 1e-6,
                "bisection_max_iterations": 64,
            },
        )
        csv_content = (evaluation_dir / "test_metrics.csv").read_text(
            encoding="utf-8"
        )
        with (evaluation_dir / "test_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        return csv_content, rows, reconstructor.learned_sampling_masks

    first_csv, first_rows, first_masks = run_once()
    second_csv, second_rows, second_masks = run_once()

    assert list(first_rows[0]) == [
        "sample_count",
        "mean_per_sample_normalized_rmse",
    ]
    assert [int(row["sample_count"]) for row in first_rows] == list(
        RadioDataset.EVALUATION_SAMPLE_COUNTS
    )
    assert [
        float(row["mean_per_sample_normalized_rmse"]) for row in first_rows
    ] == [0.5] * len(RadioDataset.EVALUATION_SAMPLE_COUNTS)
    assert second_csv == first_csv
    assert second_rows == first_rows
    assert len(first_masks) == len(second_masks) == len(first_rows)
    for sample_count, first_mask, second_mask in zip(
        RadioDataset.EVALUATION_SAMPLE_COUNTS,
        first_masks,
        second_masks,
        strict=True,
    ):
        assert first_mask.sum() == sample_count
        assert torch.equal(second_mask, first_mask)

    assert torch.are_deterministic_algorithms_enabled() == prior_algorithms
    assert torch.is_deterministic_algorithms_warn_only_enabled() == prior_warn_only
    assert torch.backends.cudnn.deterministic == prior_cudnn_deterministic
    assert torch.backends.cudnn.benchmark == prior_cudnn_benchmark


class ControlledCheckpointSampler(Module):
    def __init__(self, **model_config) -> None:
        super().__init__()
        self.model_config = model_config
        self.weight = Parameter(tensor(0.0))

    def forward(self, building_map: Tensor, transmitter_map: Tensor) -> Tensor:
        positions = torch.linspace(
            0, 1, building_map.shape[-1], device=building_map.device
        ).reshape(1, 1, 1, -1)
        return self.weight * positions.expand_as(building_map) + transmitter_map * 0


class ControlledCheckpointReconstructor(Module):
    instances: list["ControlledCheckpointReconstructor"] = []

    def __init__(self, **model_config) -> None:
        super().__init__()
        self.model_config = model_config
        self.weight = Parameter(tensor(0.0))
        self.learned_sampling_masks: list[Tensor] = []
        self.instances.append(self)

    def forward(self, inputs: Tensor) -> Tensor:
        self.learned_sampling_masks.extend(inputs[:, 3].detach().cpu())
        return self.weight.expand_as(inputs[:, :1])


def test_eval_sampler_loads_only_independent_best_checkpoints(
    monkeypatch, tmp_path
):
    run_root = tmp_path / "run"
    reconstructor_dir = run_root / "reconstructor"
    sampler_dir = run_root / "sampler"
    reconstructor_best = reconstructor_dir / "best.pt"
    sampler_best = sampler_dir / "best.pt"
    legacy_best = run_root / "best.pt"
    reconstructor_best.parent.mkdir(parents=True)
    sampler_best.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"weight": tensor(2.0)}}, reconstructor_best)
    torch.save(
        {
            "sampler_model_config": {
                "in_channels": 2,
                "out_channels": 1,
                "base_channels": 3,
            },
            "model_state_dict": {"weight": tensor(3.0)},
        },
        sampler_best,
    )
    legacy_best.write_text("legacy", encoding="utf-8")

    monkeypatch.setitem(CONFIG["runtime"], "device", "cpu")
    monkeypatch.setitem(CONFIG["reconstructor"], "run_dir", str(reconstructor_dir))
    monkeypatch.setitem(CONFIG["sampler"], "run_dir", str(sampler_dir))
    ControlledCheckpointReconstructor.instances.clear()
    monkeypatch.setattr(
        eval_module, "ResUnet", ControlledCheckpointReconstructor
    )
    monkeypatch.setattr(
        sampler_eval_module, "Sampler", ControlledCheckpointSampler
    )
    monkeypatch.setattr(
        sampler_eval_module,
        "RadioDataset",
        DeterministicSamplerEvaluationDataset,
    )

    sampler_eval_module.eval_sampler()

    metrics_path = sampler_dir / "evaluation" / "test_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    assert [int(row["sample_count"]) for row in rows] == list(
        RadioDataset.EVALUATION_SAMPLE_COUNTS
    )
    assert [
        float(row["mean_per_sample_normalized_rmse"]) for row in rows
    ] == [0.5] * len(RadioDataset.EVALUATION_SAMPLE_COUNTS)
    reconstructor = ControlledCheckpointReconstructor.instances[-1]
    assert reconstructor.weight.item() == 2.0
    assert [int(mask.sum()) for mask in reconstructor.learned_sampling_masks] == list(
        RadioDataset.EVALUATION_SAMPLE_COUNTS
    )
    assert legacy_best.read_text(encoding="utf-8") == "legacy"
