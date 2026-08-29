import csv
from copy import deepcopy
from pathlib import Path
import tomllib

from pytest import approx
from torch import Tensor, arange, cat, tensor, zeros
from torch.nn import Module
from torch.utils.data import Dataset

from radio_map_reconstruction.alpha_validation import run_alpha_validation
from radio_map_reconstruction.data import RadioDataset
import radio_map_reconstruction.alpha_validation as alpha_validation_module


ROOT = Path(__file__).resolve().parents[2]


class TinyValidationDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        assert index == 0
        tx = zeros((1, 15, 15))
        tx[0, 0, 0] = 1
        building = zeros((1, 15, 15))
        gain = tensor(0.5).expand(1, 15, 15).clone()
        gain[tx.bool()] = 1
        return (
            cat((tx, building)),
            {
                "gain": gain,
                "mask": (tx < 0.5) & (building < 0.5),
            },
        )


class TinyCoarseModel(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        height, width = inputs.shape[-2:]
        ramp = arange(height * width, dtype=inputs.dtype).reshape(
            1, 1, height, width
        )
        return ramp.expand(inputs.shape[0], -1, -1, -1) / (height * width)


class ZeroReconstructor(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return zeros(
            (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )


def test_alpha_validation_writes_reproducible_sorted_metrics_and_plot(
    tmp_path,
):
    sampler_config = {
        "alpha": 0.7,
        "alpha_candidates": [1.0, 0.0, 0.5],
        "weight_epsilon": 1e-6,
        "max_iter": 30,
        "tolerance": 1e-4,
    }
    original_config = deepcopy(sampler_config)
    output_dir = tmp_path / "sampler"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def run_once():
        return run_alpha_validation(
            coarse_model=TinyCoarseModel(),
            reconstructor=ZeroReconstructor(),
            validation_dataset=TinyValidationDataset(),
            sampler_config=sampler_config,
            global_seed=42,
            sample_counts=RadioDataset.EVALUATION_SAMPLE_COUNTS,
            output_dir=output_dir,
            device="cpu",
        )

    first_metrics = run_once()
    first_csv = (output_dir / "alpha_validation_metrics.csv").read_bytes()
    first_png = (output_dir / "rmse_vs_alpha.png").read_bytes()
    second_metrics = run_once()

    with (output_dir / "alpha_validation_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert list(rows[0]) == [
        "alpha",
        "validation_mean_per_sample_normalized_rmse",
    ]
    assert [float(row["alpha"]) for row in rows] == [0.0, 0.5, 1.0]
    assert [
        float(row["validation_mean_per_sample_normalized_rmse"])
        for row in rows
    ] == approx([0.5, 0.5, 0.5])
    assert first_metrics == approx({0.0: 0.5, 0.5: 0.5, 1.0: 0.5})
    assert second_metrics == approx(first_metrics)
    assert (output_dir / "alpha_validation_metrics.csv").read_bytes() == first_csv
    assert (output_dir / "rmse_vs_alpha.png").read_bytes() == first_png
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert sampler_config == original_config


def test_project_exposes_dedicated_alpha_tuning_command():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    assert project["project"]["scripts"]["tune-alpha"] == (
        "radio_map_reconstruction.alpha_validation:tune_alpha"
    )


def test_alpha_tuning_command_uses_validation_data_and_role_checkpoints(
    monkeypatch, tmp_path
):
    config = {
        "dataset_path": "tiny-dataset",
        "partition": "DPM",
        "seed": 17,
        "device": "cpu",
        "reconstructor": {"model": {"in_channels": 4}},
        "coarse_reconstructor": {"model": {"in_channels": 2}},
        "sampler": {
            "alpha": 0.7,
            "alpha_candidates": [0.25],
            "weight_epsilon": 1e-6,
            "max_iter": 30,
            "tolerance": 1e-4,
        },
    }
    loaded_paths = []
    dataset_calls = []

    def load_model(role_config, checkpoint_path):
        loaded_paths.append(checkpoint_path)
        if role_config is config["coarse_reconstructor"]:
            return TinyCoarseModel()
        return ZeroReconstructor()

    def make_dataset(split, **kwargs):
        dataset_calls.append((split, kwargs))
        return TinyValidationDataset()

    monkeypatch.setattr(alpha_validation_module, "CONFIG", config)
    monkeypatch.setattr(alpha_validation_module, "ROOT", tmp_path)
    monkeypatch.setattr(alpha_validation_module, "_load_role_model", load_model)
    monkeypatch.setattr(alpha_validation_module, "CoarseRadioDataset", make_dataset)

    alpha_validation_module.tune_alpha()

    assert dataset_calls == [
        (
            "val",
            {
                "dataset_path": "tiny-dataset",
                "partition": "DPM",
                "seed": 17,
            },
        )
    ]
    assert loaded_paths == [
        tmp_path / "run" / "coarse_reconstructor" / "best.pt",
        tmp_path / "run" / "reconstructor" / "best.pt",
    ]
    assert (tmp_path / "run" / "sampler" / "alpha_validation_metrics.csv").is_file()
    assert (tmp_path / "run" / "sampler" / "rmse_vs_alpha.png").is_file()
    assert config["sampler"]["alpha"] == 0.7


def test_default_sampler_config_separates_final_and_candidate_alpha_values():
    sampler_config = alpha_validation_module.CONFIG["sampler"]

    assert sampler_config["alpha"] == 0.5
    assert sampler_config["alpha_candidates"] == [0.0, 0.25, 0.5, 0.75, 1.0]
