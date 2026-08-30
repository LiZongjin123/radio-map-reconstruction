import csv
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest
from pytest import approx
from torch import (
    Tensor,
    arange,
    are_deterministic_algorithms_enabled,
    cat,
    full,
    save,
    tensor,
    uint8,
    zeros,
)
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.io import write_png

from radio_map_reconstruction.alpha_validation import (
    run_alpha_validation,
    select_example_indices,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.model import ResUnet
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
        assert are_deterministic_algorithms_enabled()
        height, width = inputs.shape[-2:]
        ramp = arange(height * width, dtype=inputs.dtype).reshape(
            1, 1, height, width
        )
        return ramp.expand(inputs.shape[0], -1, -1, -1) / (height * width)


class ZeroReconstructor(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        assert are_deterministic_algorithms_enabled()
        return zeros(
            (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )


def test_alpha_validation_writes_reproducible_sorted_metrics_and_plot(
    tmp_path,
):
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
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
    assert (
        are_deterministic_algorithms_enabled()
        == deterministic_algorithms_were_enabled
    )


def test_project_exposes_dedicated_alpha_tuning_command():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    assert project["project"]["scripts"]["tune-alpha"] == (
        "radio_map_reconstruction.alpha_validation:tune_alpha"
    )


def test_bounded_example_selection_is_deterministic_prefix_stable_and_permuted():
    smaller = select_example_indices(
        available_examples=10,
        requested_examples=3,
        global_seed=17,
    )
    repeated = select_example_indices(
        available_examples=10,
        requested_examples=3,
        global_seed=17,
    )
    larger = select_example_indices(
        available_examples=10,
        requested_examples=7,
        global_seed=17,
    )

    assert smaller == repeated
    assert smaller == larger[:3]
    assert larger != tuple(range(7))


@pytest.mark.parametrize("requested_examples", [0, -1])
def test_bounded_example_selection_rejects_non_positive_counts(requested_examples):
    with pytest.raises(ValueError, match="positive integer"):
        select_example_indices(
            available_examples=5,
            requested_examples=requested_examples,
            global_seed=17,
        )


def test_bounded_example_selection_reports_requested_and_available_counts():
    with pytest.raises(ValueError, match="requested 6.*available 5"):
        select_example_indices(
            available_examples=5,
            requested_examples=6,
            global_seed=17,
        )


@pytest.mark.parametrize("argument", ["0", "-2", "not-a-number"])
def test_alpha_tuning_command_rejects_invalid_example_arguments(argument):
    with pytest.raises(SystemExit):
        alpha_validation_module.tune_alpha(["--examples", argument])


def test_alpha_tuning_command_rejects_over_capacity_before_loading_models(
    monkeypatch,
):
    monkeypatch.setattr(
        alpha_validation_module,
        "CoarseRadioDataset",
        lambda *args, **kwargs: list(range(5)),
    )
    monkeypatch.setattr(
        alpha_validation_module,
        "_load_role_model",
        lambda *args, **kwargs: pytest.fail("models must not load"),
    )

    with pytest.raises(ValueError, match="requested 6.*available 5"):
        alpha_validation_module.tune_alpha(["--examples", "6"])


@pytest.mark.parametrize(
    ("arguments", "expected_limit"),
    [([], None), (["--examples", "3"], 3)],
)
def test_alpha_tuning_command_routes_optional_example_limit_without_changing_outputs(
    monkeypatch,
    tmp_path,
    arguments,
    expected_limit,
):
    captured = {}
    dataset = list(range(5))
    config = {
        "dataset_path": "dataset",
        "partition": "DPM",
        "seed": 17,
        "device": "cpu",
        "reconstructor": {"model": {}},
        "coarse_reconstructor": {"model": {}},
        "sampler": {"alpha_candidates": [0.25]},
    }
    monkeypatch.setattr(alpha_validation_module, "CONFIG", config)
    monkeypatch.setattr(alpha_validation_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        alpha_validation_module,
        "CoarseRadioDataset",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        alpha_validation_module,
        "_load_role_model",
        lambda *args, **kwargs: object(),
    )

    def capture_run(**kwargs):
        captured.update(kwargs)
        return {0.25: 0.5}

    monkeypatch.setattr(alpha_validation_module, "run_alpha_validation", capture_run)

    alpha_validation_module.tune_alpha(arguments)

    assert captured["validation_dataset"] is dataset
    assert captured["requested_examples"] == expected_limit
    assert captured["global_seed"] == 17
    assert captured["output_dir"] == tmp_path / "run" / "sampler"


def test_selected_examples_cover_all_alpha_candidates_and_sample_counts_with_pair_progress(
    monkeypatch,
    tmp_path,
):
    class FourExampleDataset(TinyValidationDataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index):
            assert 0 <= index < len(self)
            return super().__getitem__(0)

    sampler_calls = []

    def record_sampling(
        coarse_map,
        tx_map,
        building_map,
        sample_count,
        *,
        alpha,
        sample_id,
        **kwargs,
    ):
        sampler_calls.append((sample_id, alpha, sample_count))
        sampling_mask = zeros(coarse_map.shape)
        sampling_mask.flatten()[1 : sample_count + 1] = 1
        return sampling_mask, None

    monkeypatch.setattr(
        alpha_validation_module,
        "gradient_distance_weighted_clustering_sample",
        record_sampling,
    )
    progress_events = []
    sampler_config = {
        "alpha_candidates": [0.75, 0.25],
        "weight_epsilon": 1e-6,
        "max_iter": 30,
        "tolerance": 1e-4,
    }

    run_alpha_validation(
        coarse_model=TinyCoarseModel(),
        reconstructor=ZeroReconstructor(),
        validation_dataset=FourExampleDataset(),
        sampler_config=sampler_config,
        global_seed=17,
        sample_counts=(1, 2),
        output_dir=tmp_path,
        device="cpu",
        requested_examples=2,
        progress_reporter=progress_events.append,
    )

    selected = select_example_indices(
        available_examples=4,
        requested_examples=2,
        global_seed=17,
    )
    assert sampler_calls == [
        (f"validation-{example_index}", alpha, sample_count)
        for example_index in selected
        for alpha in (0.25, 0.75)
        for sample_count in (1, 2)
    ]
    assert [event.completed_pairs for event in progress_events] == [1, 2, 3, 4]
    assert {event.total_pairs for event in progress_events} == {4}
    assert [event.example_index for event in progress_events] == [
        selected[0],
        selected[0],
        selected[1],
        selected[1],
    ]
    assert [event.alpha for event in progress_events] == [0.25, 0.75, 0.25, 0.75]
    assert all(event.eta_seconds >= 0 for event in progress_events)


def test_tiny_alpha_validation_reports_plain_text_pair_progress(capsys, tmp_path):
    run_alpha_validation(
        coarse_model=TinyCoarseModel(),
        reconstructor=ZeroReconstructor(),
        validation_dataset=TinyValidationDataset(),
        sampler_config={
            "alpha_candidates": [0.0, 1.0],
            "weight_epsilon": 1e-6,
            "max_iter": 30,
            "tolerance": 1e-4,
        },
        global_seed=42,
        sample_counts=(1,),
        output_dir=tmp_path,
        device="cpu",
    )

    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("alpha validation:")
    ]
    assert len(progress_lines) == 2
    assert "Example 1/1 (validation-0)" in progress_lines[0]
    assert "alpha=0" in progress_lines[0]
    assert "completed=1/2" in progress_lines[0]
    assert "ETA=" in progress_lines[0]
    assert "alpha=1" in progress_lines[1]
    assert "completed=2/2" in progress_lines[1]


def test_alpha_tuning_command_uses_validation_data_and_role_checkpoints(
    monkeypatch, tmp_path
):
    dataset_root = tmp_path / "dataset"
    gain_dir = dataset_root / "gain" / "DPM"
    tx_dir = dataset_root / "png" / "antennas"
    building_dir = dataset_root / "png" / "buildings_complete"
    gain_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)
    building_dir.mkdir(parents=True)
    for city_map_id in range(10):
        gain = full((1, 16, 16), 128, dtype=uint8)
        tx = zeros((1, 16, 16), dtype=uint8)
        building = zeros((1, 16, 16), dtype=uint8)
        tx[0, 0, 0] = 255
        write_png(building, str(building_dir / f"{city_map_id}.png"))
        write_png(gain, str(gain_dir / f"{city_map_id}_0.png"))
        write_png(tx, str(tx_dir / f"{city_map_id}_0.png"))

    config = {
        "dataset_path": str(dataset_root),
        "partition": "DPM",
        "seed": 17,
        "device": "cpu",
        "reconstructor": {
            "model": {"in_channels": 4, "out_channels": 1, "base_channels": 1}
        },
        "coarse_reconstructor": {
            "model": {"in_channels": 2, "out_channels": 1, "base_channels": 1}
        },
        "sampler": {
            "alpha": 0.7,
            "alpha_candidates": [0.25],
            "weight_epsilon": 1e-6,
            "max_iter": 30,
            "tolerance": 1e-4,
        },
    }
    monkeypatch.setattr(alpha_validation_module, "CONFIG", config)
    monkeypatch.setattr(alpha_validation_module, "ROOT", tmp_path)

    for role, run_name in (
        ("coarse_reconstructor", "coarse_reconstructor"),
        ("reconstructor", "reconstructor"),
    ):
        checkpoint_path = tmp_path / "run" / run_name / "best.pt"
        checkpoint_path.parent.mkdir(parents=True)
        model = ResUnet(**config[role]["model"])
        save({"model_state_dict": model.state_dict()}, checkpoint_path)

    alpha_validation_module.tune_alpha([])

    assert (tmp_path / "run" / "sampler" / "alpha_validation_metrics.csv").is_file()
    assert (tmp_path / "run" / "sampler" / "rmse_vs_alpha.png").is_file()
    assert config["sampler"]["alpha"] == 0.7


def test_default_sampler_config_separates_final_and_candidate_alpha_values():
    sampler_config = alpha_validation_module.CONFIG["sampler"]

    assert sampler_config["alpha"] == 0.5
    assert sampler_config["alpha_candidates"] == [0.0, 0.25, 0.5, 0.75, 1.0]
