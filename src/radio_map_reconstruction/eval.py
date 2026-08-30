import csv
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from torch import (
    Tensor,
    are_deterministic_algorithms_enabled,
    backends,
    cat,
    inference_mode,
    is_deterministic_algorithms_warn_only_enabled,
    load,
    stack,
    use_deterministic_algorithms,
    zeros_like,
)
from torch.nn import Module
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from radio_map_reconstruction.artifacts import (
    COARSE_RECONSTRUCTOR_RUN_PATH,
    EVALUATION_FIGURE_CASES,
    EVALUATION_RUN_PATH,
    RECONSTRUCTOR_RUN_PATH,
    SAMPLING_DIAGNOSTIC_CASES,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import normalized_mse_per_sample
from radio_map_reconstruction.model import ResUnet
from radio_map_reconstruction.plot import (
    EvaluationBundle,
    SamplingDiagnosticData,
    evaluation_error_upper_limit,
    save_evaluation_bundle_figure,
    save_rmse_comparison_curve,
    save_sampling_diagnostic_figure,
)
from radio_map_reconstruction.sampling import (
    gradient_distance_weighted_clustering_sample,
)
from radio_map_reconstruction.train import CONFIG
from radio_map_reconstruction.util import (
    positive_int_argument,
    sample_identity,
    select_example_indices,
)

ROOT = Path(__file__).resolve().parents[2]

class StrategyRmse(NamedTuple):
    random_rmse: float
    guided_rmse: float


def _selected_test_example_indices(
    test_dataset: Dataset,
    *,
    requested_examples: int | None,
    global_seed: int,
) -> tuple[int, ...]:
    sample_counts = tuple(test_dataset.EVALUATION_SAMPLE_COUNTS)
    sample_total = len(test_dataset)
    if sample_total % len(sample_counts) != 0:
        raise ValueError(
            f"test dataset must hold {len(sample_counts)} cases per Example, "
            f"one per Valid Sampling Point count; found {sample_total} cases"
        )
    available_examples = sample_total // len(sample_counts)
    example_indices = select_example_indices(
        available_examples=available_examples,
        requested_examples=requested_examples,
        global_seed=global_seed,
        partition_name="test",
    )
    minimum_examples = 1 + max(
        index
        for index, _ in EVALUATION_FIGURE_CASES + SAMPLING_DIAGNOSTIC_CASES
    )
    if len(example_indices) < minimum_examples:
        raise ValueError(
            "evaluation requires at least "
            f"{minimum_examples} Examples to render all fixed Evaluation "
            "Bundle Figures and Sampling Diagnostics"
        )
    return example_indices


def load_checkpoint(path: str | Path) -> Module:
    checkpoint = load(path, map_location=CONFIG["device"])
    model = ResUnet(**CONFIG["reconstructor"]["model"]).to(CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_coarse_checkpoint(path: str | Path) -> Module:
    checkpoint = load(path, map_location=CONFIG["device"])
    model = ResUnet(**CONFIG["coarse_reconstructor"]["model"]).to(
        CONFIG["device"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _guided_sampling_inputs(
    *,
    coarse_model: Module,
    gain: Tensor,
    inputs: Tensor,
    sample_counts: tuple[int, ...],
    sampler_config: dict,
    global_seed: int,
    sample_id: str,
    example_position: int,
) -> tuple[Tensor, list[SamplingDiagnosticData]]:
    tx_map = inputs[0, 1:2]
    building_map = inputs[0, 2:3]
    coarse_map = coarse_model(cat((tx_map, building_map)).unsqueeze(0))[0]
    guided_inputs = []
    diagnostic_figures = []
    for sample_count_index, sample_count in enumerate(sample_counts):
        sampling_mask, diagnostics = gradient_distance_weighted_clustering_sample(
            coarse_map,
            inputs[sample_count_index, 1:2],
            inputs[sample_count_index, 2:3],
            sample_count,
            alpha=sampler_config["alpha"],
            global_seed=global_seed,
            sample_id=sample_id,
            weight_epsilon=sampler_config["weight_epsilon"],
            max_iter=sampler_config["max_iter"],
            tolerance=sampler_config["tolerance"],
        )
        guided_inputs.append(
            cat(
                (
                    gain[sample_count_index] * sampling_mask,
                    inputs[sample_count_index, 1:2],
                    inputs[sample_count_index, 2:3],
                    sampling_mask,
                )
            )
        )
        if (example_position, sample_count) in SAMPLING_DIAGNOSTIC_CASES:
            diagnostic_figures.append(
                SamplingDiagnosticData(
                    sample_count=sample_count,
                    coarse_map=diagnostics.coarse_map[0].cpu().numpy(),
                    normalized_gradient=(
                        diagnostics.normalized_gradient[0].cpu().numpy()
                    ),
                    normalized_distance=(
                        diagnostics.normalized_distance[0].cpu().numpy()
                    ),
                    score=diagnostics.score[0].cpu().numpy(),
                    cluster_labels=diagnostics.cluster_labels[0].cpu().numpy(),
                    sampling_mask=sampling_mask[0].bool().cpu().numpy(),
                    valid_receiving_area=(
                        diagnostics.cluster_labels[0] >= 0
                    ).cpu().numpy(),
                    transmitter_point=(
                        diagnostics.transmitter_point.cpu().numpy()
                    ),
                )
            )
    return stack(guided_inputs), diagnostic_figures


def _accumulate_rmse(
    outputs: Tensor,
    targets: dict[str, Tensor],
    rmse_sums: dict[int, float],
    sample_counts: tuple[int, ...],
) -> None:
    rmse_per_sample = normalized_mse_per_sample(
        outputs.clamp(0, 1), targets
    ).sqrt()
    for count_index, sample_count in enumerate(sample_counts):
        rmse_sums[sample_count] += rmse_per_sample[count_index].item()


def _evaluation_bundle(
    *,
    inputs: Tensor,
    output: Tensor,
    ground_truth: Tensor,
    valid_receiving_area: Tensor,
    sample_count: int,
) -> EvaluationBundle:
    building = inputs[2] > 0.5
    transmitter = inputs[1] > 0.5
    sampling_mask = (inputs[3] > 0.5) & valid_receiving_area
    sparse_map = inputs[0].where(sampling_mask, zeros_like(inputs[0]))
    reconstruction = output.clone()
    reconstruction[building] = 0
    reconstruction[transmitter] = 1
    absolute_error = (reconstruction - ground_truth).abs()
    absolute_error[~valid_receiving_area] = float("nan")
    return EvaluationBundle(
        sample_count=sample_count,
        ground_truth=ground_truth.cpu().numpy(),
        sampling_mask=sampling_mask.cpu().numpy(),
        sparse_map=sparse_map.cpu().numpy(),
        reconstruction=reconstruction.cpu().numpy(),
        absolute_error=absolute_error.cpu().numpy(),
    )


def _random_evaluation_bundles(
    *,
    example_position: int,
    inputs: Tensor,
    outputs: Tensor,
    targets: dict[str, Tensor],
    sample_counts: tuple[int, ...],
) -> list[EvaluationBundle]:
    bundles = []
    for bundle_sample_index, bundle_count in EVALUATION_FIGURE_CASES:
        if example_position != bundle_sample_index:
            continue
        count_index = sample_counts.index(bundle_count)
        bundles.append(
            _evaluation_bundle(
                inputs=inputs[count_index],
                output=outputs[count_index, 0].clamp(0, 1),
                ground_truth=targets["gain"][count_index, 0],
                valid_receiving_area=targets["mask"][count_index, 0],
                sample_count=bundle_count,
            )
        )
    return bundles


def _write_evaluation_artifacts(
    evaluation_dir: Path,
    bundles: list[EvaluationBundle],
    sampling_diagnostics: list[SamplingDiagnosticData],
    rmse_by_sample_count: dict[int, StrategyRmse],
) -> None:
    error_upper_limit = evaluation_error_upper_limit(bundles)
    for bundle in bundles:
        save_evaluation_bundle_figure(
            bundle,
            error_upper_limit=error_upper_limit,
            output_path=(
                evaluation_dir
                / f"evaluation_bundle_{bundle.sample_count}_samples.png"
            ),
        )
    for diagnostic in sampling_diagnostics:
        save_sampling_diagnostic_figure(
            diagnostic,
            output_path=(
                evaluation_dir
                / "gradient_distance_weighted_clustering_sampling_strategy_"
                "diagnostics_"
                f"{diagnostic.sample_count}_samples.png"
            ),
        )
    save_rmse_comparison_curve(
        sample_counts=list(rmse_by_sample_count),
        random_rmse=[
            comparison.random_rmse
            for comparison in rmse_by_sample_count.values()
        ],
        guided_rmse=[
            comparison.guided_rmse
            for comparison in rmse_by_sample_count.values()
        ],
        output_path=evaluation_dir / "rmse_vs_sample_count.png",
    )


def _write_test_metrics(
    evaluation_dir: Path,
    rmse_by_sample_count: dict[int, StrategyRmse],
) -> None:
    metrics_path = evaluation_dir / "test_metrics.csv"
    fieldnames = (
        "sample_count",
        "random_mean_per_sample_normalized_rmse",
        "guided_mean_per_sample_normalized_rmse",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_count, comparison in rmse_by_sample_count.items():
            writer.writerow({
                "sample_count": sample_count,
                "random_mean_per_sample_normalized_rmse": (
                    comparison.random_rmse
                ),
                "guided_mean_per_sample_normalized_rmse": (
                    comparison.guided_rmse
                ),
            })


def run_evaluation(
    *,
    model: Module,
    coarse_model: Module,
    test_dataset: Dataset,
    sampler_config: dict,
    global_seed: int,
    evaluation_dir: str | Path,
    requested_examples: int | None = None,
) -> dict[int, StrategyRmse]:
    """Compare fixed-seed random and configured-alpha guided sampling on one frozen reconstructor."""
    sample_counts = tuple(test_dataset.EVALUATION_SAMPLE_COUNTS)
    example_indices = _selected_test_example_indices(
        test_dataset,
        requested_examples=requested_examples,
        global_seed=global_seed,
    )
    example_count = len(example_indices)

    model_was_training = model.training
    coarse_model_was_training = coarse_model.training
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only_was_enabled = (
        is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_deterministic_was_enabled = backends.cudnn.deterministic
    cudnn_benchmark_was_enabled = backends.cudnn.benchmark
    model.eval()
    coarse_model.eval()
    use_deterministic_algorithms(True)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False

    random_rmse_sums = {sample_count: 0.0 for sample_count in sample_counts}
    guided_rmse_sums = {sample_count: 0.0 for sample_count in sample_counts}
    bundles: list[EvaluationBundle] = []
    sampling_diagnostics: list[SamplingDiagnosticData] = []

    try:
        with inference_mode():
            progress = tqdm(
                example_indices,
                desc="eval 1/1",
                unit="Example",
                total=example_count,
                leave=True,
            )
            for sample_position, sample_index in enumerate(progress):
                cases = [
                    test_dataset[
                        sample_index * len(sample_counts) + count_index
                    ]
                    for count_index in range(len(sample_counts))
                ]
                inputs = stack(
                    [case[0] for case in cases]
                ).to(CONFIG["device"])
                targets = {
                    name: stack(
                        [case[1][name] for case in cases]
                    ).to(CONFIG["device"])
                    for name in ("gain", "mask")
                }

                random_outputs = model(inputs)
                _accumulate_rmse(
                    random_outputs,
                    targets,
                    random_rmse_sums,
                    sample_counts,
                )
                bundles.extend(
                    _random_evaluation_bundles(
                        example_position=sample_position,
                        inputs=inputs,
                        outputs=random_outputs,
                        targets=targets,
                        sample_counts=sample_counts,
                    )
                )

                guided_inputs, case_diagnostics = _guided_sampling_inputs(
                    coarse_model=coarse_model,
                    gain=targets["gain"],
                    inputs=inputs,
                    sample_counts=sample_counts,
                    sampler_config=sampler_config,
                    global_seed=global_seed,
                    sample_id=sample_identity(
                        test_dataset,
                        sample_index,
                        fallback_prefix="test",
                    ),
                    example_position=sample_position,
                )
                sampling_diagnostics.extend(case_diagnostics)
                guided_outputs = model(guided_inputs)
                _accumulate_rmse(
                    guided_outputs,
                    targets,
                    guided_rmse_sums,
                    sample_counts,
                )

                evaluated_cases = (sample_position + 1) * 2 * len(sample_counts)
                running_rmse = (
                    sum(random_rmse_sums.values())
                    + sum(guided_rmse_sums.values())
                ) / evaluated_cases
                progress.set_postfix(rmse=f"{running_rmse:.6f}")
    finally:
        model.train(model_was_training)
        coarse_model.train(coarse_model_was_training)
        use_deterministic_algorithms(
            deterministic_algorithms_were_enabled,
            warn_only=deterministic_warn_only_was_enabled,
        )
        backends.cudnn.deterministic = cudnn_deterministic_was_enabled
        backends.cudnn.benchmark = cudnn_benchmark_was_enabled

    rmse_by_sample_count = {
        sample_count: StrategyRmse(
            random_rmse=random_rmse_sums[sample_count] / example_count,
            guided_rmse=guided_rmse_sums[sample_count] / example_count,
        )
        for sample_count in sample_counts
    }

    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_test_metrics(evaluation_dir, rmse_by_sample_count)
    _write_evaluation_artifacts(
        evaluation_dir,
        bundles,
        sampling_diagnostics,
        rmse_by_sample_count,
    )
    return rmse_by_sample_count


def eval(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Evaluate reconstruction strategies.")
    parser.add_argument(
        "--examples",
        type=positive_int_argument,
        help="deterministically select exactly N test Examples (minimum: 8)",
    )
    arguments = parser.parse_args(argv)
    test_dataset = RadioDataset("test")
    _selected_test_example_indices(
        test_dataset,
        requested_examples=arguments.examples,
        global_seed=CONFIG["seed"],
    )
    model = load_checkpoint(ROOT / RECONSTRUCTOR_RUN_PATH / "best.pt")
    coarse_model = load_coarse_checkpoint(
        ROOT / COARSE_RECONSTRUCTOR_RUN_PATH / "best.pt"
    )
    rmse_by_sample_count = run_evaluation(
        model=model,
        coarse_model=coarse_model,
        test_dataset=test_dataset,
        sampler_config=CONFIG["sampler"],
        global_seed=CONFIG["seed"],
        evaluation_dir=ROOT / EVALUATION_RUN_PATH,
        requested_examples=arguments.examples,
    )

    for sample_count, comparison in rmse_by_sample_count.items():
        print(
            f"test rmse ({sample_count} samples): "
            f"random={comparison.random_rmse:.6f} "
            f"guided={comparison.guided_rmse:.6f}"
        )
