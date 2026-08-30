import csv
from pathlib import Path
import tomllib

import pytest
from pytest import approx
from torch import Tensor, are_deterministic_algorithms_enabled, linspace, tensor
from torch.nn import Module, Parameter
from torch.utils.data import Dataset

from radio_map_reconstruction.artifacts import (
    EVALUATION_FIGURE_CASES,
    SAMPLING_DIAGNOSTIC_CASES,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.eval import (
    CONFIG,
    StrategyRmse,
    run_evaluation,
)
import radio_map_reconstruction.eval as eval_module


class FixedEvaluationDataset(Dataset):
    EVALUATION_SAMPLE_COUNTS = RadioDataset.EVALUATION_SAMPLE_COUNTS

    def __init__(self, example_count: int = 8):
        self.example_count = example_count

    def __len__(self) -> int:
        return self.example_count * len(RadioDataset.EVALUATION_SAMPLE_COUNTS)

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
        self.input_batches: list[Tensor] = []

    def forward(self, inputs: Tensor) -> Tensor:
        assert are_deterministic_algorithms_enabled()
        self.input_batches.append(inputs.detach().clone())
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


class MaskSensitiveReconstructor(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        coordinate_values = linspace(
            0,
            1,
            inputs.shape[-1],
            dtype=inputs.dtype,
            device=inputs.device,
        ).reshape(1, 1, 1, -1)
        return inputs[:, 3:4] * coordinate_values


def read_png_bytes(evaluation_dir):
    return {
        path.name: path.read_bytes()
        for path in sorted(evaluation_dir.glob("*.png"))
    }


def test_unified_evaluation_writes_reproducible_final_artifacts_without_npz(
    monkeypatch, tmp_path
):
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    assert EVALUATION_FIGURE_CASES == (
        (0, 10),
        (1, 50),
        (2, 100),
        (3, 200),
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
        return model, rmse_by_sample_count, rows, read_png_bytes(evaluation_dir)

    first_model, first_rmse, first_rows, first_pngs = run_once()
    second_model, second_rmse, second_rows, second_pngs = run_once()

    assert list(first_rows[0]) == [
        "sample_count",
        "random_mean_per_sample_normalized_rmse",
        "guided_mean_per_sample_normalized_rmse",
        "uniform_mean_per_sample_normalized_rmse",
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
    assert [
        float(row["uniform_mean_per_sample_normalized_rmse"])
        for row in first_rows
    ] == approx([0.55] * 10)
    assert list(first_rmse) == list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
    for comparison in first_rmse.values():
        assert comparison.random_rmse == approx(0.55)
        assert comparison.guided_rmse == approx(0.55)
        assert comparison.regular_grid_rmse == approx(0.55)
    assert second_rmse == first_rmse
    assert second_rows == first_rows

    for call_index, (mask_sums, sparse_values) in enumerate(
        first_model.calls
    ):
        sample_index = call_index // 3
        ground_truth_value = (sample_index + 1) / 10
        assert mask_sums == approx(
            list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        assert sparse_values == approx([ground_truth_value] * 10)
    assert len(first_model.calls) == 24
    assert first_model.calls == second_model.calls
    assert len(first_model.input_batches) == 24
    assert all(
        first_batch.equal(second_batch)
        for first_batch, second_batch in zip(
            first_model.input_batches,
            second_model.input_batches,
            strict=True,
        )
    )

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
    assert not list(evaluation_dir.glob("*.npz"))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (
        are_deterministic_algorithms_enabled()
        == deterministic_algorithms_were_enabled
    )


def test_random_and_guided_rmse_remain_numerically_stable_at_evaluation_seam(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(CONFIG, "device", "cpu")
    monkeypatch.setattr(eval_module, "_write_test_metrics", lambda *args: None)
    monkeypatch.setattr(
        eval_module, "_write_evaluation_artifacts", lambda *args: None
    )

    rmse_by_sample_count = run_evaluation(
        model=MaskSensitiveReconstructor(),
        coarse_model=FlatCoarseModel(),
        test_dataset=FixedEvaluationDataset(),
        sampler_config={
            "alpha": 0.5,
            "weight_epsilon": 1e-6,
            "max_iter": 30,
            "tolerance": 1e-4,
        },
        global_seed=42,
        evaluation_dir=tmp_path,
    )

    assert [
        value.random_rmse for value in rmse_by_sample_count.values()
    ] == approx([
        0.4484856817871332,
        0.4449373548850417,
        0.439734204672277,
        0.42589924298226833,
        0.4058195613324642,
        0.38635675236582756,
        0.37036666832864285,
        0.36023686826229095,
        0.35799524188041687,
        0.3653622455894947,
    ])
    assert [
        value.guided_rmse for value in rmse_by_sample_count.values()
    ] == approx([
        0.4535260535776615,
        0.45395165868103504,
        0.45640975795686245,
        0.45016542077064514,
        0.44552214071154594,
        0.44552696123719215,
        0.4348500221967697,
        0.4069547988474369,
        0.37894369289278984,
        0.3653622455894947,
    ])


def test_eval_loads_both_frozen_role_checkpoints_and_prints_three_strategies(
    monkeypatch, tmp_path, capsys
):
    loaded_paths = []
    model = RecordingReconstructor()

    class TinyDataset:
        EVALUATION_SAMPLE_COUNTS = RadioDataset.EVALUATION_SAMPLE_COUNTS

        def __init__(self, split):
            self.split = split

        def __len__(self):
            return 8 * len(self.EVALUATION_SAMPLE_COUNTS)

    run_kwargs = {}

    def fake_run_evaluation(**kwargs):
        run_kwargs.update(kwargs)
        return {
            count: StrategyRmse(
                random_rmse=0.3,
                guided_rmse=0.4,
                regular_grid_rmse=0.35,
            )
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

    eval_module.eval([])

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
    assert run_kwargs["requested_examples"] is None
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.endswith(
        "Random Sampling=0.300000 "
        "Regular-Grid Sampling=0.350000 "
        "Guided Sampling=0.400000"
    )


@pytest.mark.parametrize(
    ("arguments", "expected_limit"),
    [([], None), (["--examples", "8"], 8)],
)
def test_eval_command_routes_optional_example_limit_without_changing_outputs(
    monkeypatch,
    tmp_path,
    arguments,
    expected_limit,
):
    captured = {}
    dataset = FixedEvaluationDataset(example_count=9)
    config = {"device": "cpu", "seed": 17, "sampler": {"alpha": 0.5}}
    monkeypatch.setattr(eval_module, "ROOT", tmp_path)
    monkeypatch.setattr(eval_module, "CONFIG", config)
    monkeypatch.setattr(eval_module, "RadioDataset", lambda split: dataset)
    monkeypatch.setattr(eval_module, "load_checkpoint", lambda path: object())
    monkeypatch.setattr(eval_module, "load_coarse_checkpoint", lambda path: object())

    def capture_run(**kwargs):
        captured.update(kwargs)
        return {
            10: StrategyRmse(
                random_rmse=0.3,
                guided_rmse=0.4,
                regular_grid_rmse=0.35,
            )
        }

    monkeypatch.setattr(eval_module, "run_evaluation", capture_run)

    eval_module.eval(arguments)

    assert captured["test_dataset"] is dataset
    assert captured["requested_examples"] == expected_limit
    assert captured["global_seed"] == 17
    assert captured["evaluation_dir"] == tmp_path / "run" / "evaluation"


@pytest.mark.parametrize("argument", ["0", "-2", "not-a-number"])
def test_eval_command_rejects_invalid_example_arguments(argument):
    with pytest.raises(SystemExit):
        eval_module.eval(["--examples", argument])


@pytest.mark.parametrize(
    ("argument", "message"),
    [("7", "at least 8"), ("10", "requested 10.*available 9")],
)
def test_eval_command_rejects_example_boundaries_before_loading_models(
    monkeypatch,
    argument,
    message,
):
    monkeypatch.setattr(
        eval_module,
        "RadioDataset",
        lambda split: FixedEvaluationDataset(example_count=9),
    )
    monkeypatch.setattr(
        eval_module,
        "load_checkpoint",
        lambda path: pytest.fail("models must not load"),
    )
    monkeypatch.setattr(
        eval_module,
        "load_coarse_checkpoint",
        lambda path: pytest.fail("models must not load"),
    )

    with pytest.raises(ValueError, match=message):
        eval_module.eval(["--examples", argument])


def test_limited_evaluation_selects_a_deterministic_prefix_with_all_sample_counts(
    monkeypatch,
    tmp_path,
):
    dataset = FixedEvaluationDataset(example_count=10)
    model = RecordingReconstructor()
    monkeypatch.setattr(
        eval_module,
        "CONFIG",
        {
            "device": "cpu",
            "seed": 17,
            "sampler": {
                "alpha": 0.5,
                "weight_epsilon": 1e-6,
                "max_iter": 30,
                "tolerance": 1e-4,
            },
        },
    )
    monkeypatch.setattr(eval_module, "ROOT", tmp_path)
    monkeypatch.setattr(eval_module, "RadioDataset", lambda split: dataset)
    monkeypatch.setattr(eval_module, "load_checkpoint", lambda path: model)
    monkeypatch.setattr(
        eval_module,
        "load_coarse_checkpoint",
        lambda path: FlatCoarseModel(),
    )
    monkeypatch.setattr(eval_module, "_write_test_metrics", lambda *args: None)
    written_artifacts = {}

    def capture_artifacts(
        evaluation_dir,
        bundles,
        sampling_diagnostics,
        rmse_by_sample_count,
    ):
        written_artifacts.update(
            evaluation_dir=evaluation_dir,
            bundle_counts=[bundle.sample_count for bundle in bundles],
            diagnostic_counts=[
                diagnostic.sample_count for diagnostic in sampling_diagnostics
            ],
            rmse_by_sample_count=rmse_by_sample_count,
        )

    monkeypatch.setattr(
        eval_module,
        "_write_evaluation_artifacts",
        capture_artifacts,
    )
    progress_state = {}

    class RecordingProgress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            progress_state["total"] = kwargs["total"]
            progress_state["postfixes"] = []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            progress_state["postfixes"].append(kwargs)

    monkeypatch.setattr(eval_module, "tqdm", RecordingProgress)

    eval_module.eval(["--examples", "8"])

    selected = (9, 7, 0, 5, 6, 2, 4, 8)
    assert len(model.calls) == 24
    for selected_position, example_index in enumerate(selected):
        random_call = model.calls[selected_position * 3]
        regular_grid_call = model.calls[selected_position * 3 + 1]
        guided_call = model.calls[selected_position * 3 + 2]
        assert random_call[0] == approx(
            list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        assert guided_call[0] == approx(
            list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        assert regular_grid_call[0] == approx(
            list(RadioDataset.EVALUATION_SAMPLE_COUNTS)
        )
        assert random_call[1] == approx([(example_index + 1) / 10] * 10)
        assert regular_grid_call[1] == approx(
            [(example_index + 1) / 10] * 10
        )
        assert guided_call[1] == approx([(example_index + 1) / 10] * 10)
    assert written_artifacts["evaluation_dir"] == tmp_path / "run" / "evaluation"
    assert written_artifacts["bundle_counts"] == [10, 50, 100, 200]
    assert written_artifacts["diagnostic_counts"] == [10, 50, 100, 200]
    assert progress_state["total"] == 8
    assert len(progress_state["postfixes"]) == 8


def test_project_exposes_only_the_four_server_commands():
    project_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with project_path.open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["scripts"] == {
        "train": "radio_map_reconstruction.train:train",
        "train-coarse": "radio_map_reconstruction.train:train_coarse",
        "tune-alpha": "radio_map_reconstruction.alpha_validation:tune_alpha",
        "eval": "radio_map_reconstruction.eval:eval",
    }
