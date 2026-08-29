import csv
from pathlib import Path

import numpy as np
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
    EVALUATION_BUNDLE_CASES,
    EVALUATION_RUN_PATH,
    RECONSTRUCTOR_RUN_PATH,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss, normalized_mse_per_sample
from radio_map_reconstruction.model import ResUnet
from radio_map_reconstruction.plot import (
    EvaluationBundle,
    evaluation_error_upper_limit,
    save_evaluation_bundle_figure,
    save_rmse_comparison_curve,
)
from radio_map_reconstruction.sampling import (
    gradient_distance_weighted_clustering_sample,
)
from radio_map_reconstruction.train import CONFIG
from radio_map_reconstruction.util import sample_identity

ROOT = Path(__file__).resolve().parents[2]

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
) -> Tensor:
    tx_map = inputs[0, 1:2]
    building_map = inputs[0, 2:3]
    coarse_map = coarse_model(cat((tx_map, building_map)).unsqueeze(0))[0]
    guided_inputs = []
    for sample_count_index, sample_count in enumerate(sample_counts):
        sampling_mask, _ = gradient_distance_weighted_clustering_sample(
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
    return stack(guided_inputs)


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


def _write_evaluation_artifacts(
    evaluation_dir: Path,
    bundles: list[EvaluationBundle],
    rmse_by_sample_count: dict[int, tuple[float, float]],
) -> None:
    for bundle_index, bundle in enumerate(bundles):
        np.savez_compressed(
            evaluation_dir / f"evaluation_bundle_{bundle_index}.npz",
            sample_count=np.asarray(bundle.sample_count),
            ground_truth=bundle.ground_truth,
            sampling_mask=bundle.sampling_mask,
            sparse_map=bundle.sparse_map,
            reconstruction=bundle.reconstruction,
            absolute_error=bundle.absolute_error,
        )
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
    save_rmse_comparison_curve(
        sample_counts=list(rmse_by_sample_count),
        random_rmse=[
            rmse[0] for rmse in rmse_by_sample_count.values()
        ],
        guided_rmse=[
            rmse[1] for rmse in rmse_by_sample_count.values()
        ],
        output_path=evaluation_dir / "rmse_vs_sample_count.png",
    )


def _write_test_metrics(
    evaluation_dir: Path,
    rmse_by_sample_count: dict[int, tuple[float, float]],
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
        for sample_count, (random_rmse, guided_rmse) in (
            rmse_by_sample_count.items()
        ):
            writer.writerow({
                "sample_count": sample_count,
                "random_mean_per_sample_normalized_rmse": random_rmse,
                "guided_mean_per_sample_normalized_rmse": guided_rmse,
            })


def run_evaluation(
    *,
    model: Module,
    coarse_model: Module,
    test_dataset: Dataset,
    criterion: Module,
    sampler_config: dict,
    global_seed: int,
    evaluation_dir: str | Path,
) -> tuple[float, float, dict[int, tuple[float, float]]]:
    """Compare fixed-seed random and configured-alpha guided sampling on one frozen reconstructor."""
    sample_counts = RadioDataset.EVALUATION_SAMPLE_COUNTS
    sample_total = len(test_dataset)
    if sample_total % len(sample_counts) != 0:
        raise ValueError(
            f"test dataset must hold {len(sample_counts)} cases per sample, "
            f"one per Valid Sampling Point count; found {sample_total} cases"
        )
    sample_num = sample_total // len(sample_counts)
    if sample_num < max(index for index, _ in EVALUATION_BUNDLE_CASES) + 1:
        raise ValueError(
            "test dataset must contain at least "
            f"{max(index for index, _ in EVALUATION_BUNDLE_CASES) + 1} samples "
            "to render the fixed Evaluation Bundle cases"
        )

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

    rmse_sums: dict[str, dict[int, float]] = {
        "random": {sample_count: 0.0 for sample_count in sample_counts},
        "guided": {sample_count: 0.0 for sample_count in sample_counts},
    }
    case_counts: dict[str, dict[int, int]] = {
        "random": {sample_count: 0 for sample_count in sample_counts},
        "guided": {sample_count: 0 for sample_count in sample_counts},
    }
    total_loss = 0.0
    total_rmse = 0.0
    total_cases = 0
    bundles: list[EvaluationBundle] = []

    try:
        with inference_mode():
            progress = tqdm(
                range(sample_num),
                desc="eval 1/1",
                unit="sample",
                leave=True,
            )
            for sample_index in progress:
                cases = [
                    test_dataset[sample_index * len(sample_counts) + count_index]
                    for count_index in range(len(sample_counts))
                ]
                inputs = stack([case[0] for case in cases]).to(
                    CONFIG["device"]
                )
                targets = {
                    name: stack([case[1][name] for case in cases]).to(
                        CONFIG["device"]
                    )
                    for name in ("gain", "mask")
                }
                gain = targets["gain"]

                for strategy, strategy_inputs in (
                    ("random", inputs),
                    (
                        "guided",
                        _guided_sampling_inputs(
                            coarse_model=coarse_model,
                            gain=gain,
                            inputs=inputs,
                            sample_counts=sample_counts,
                            sampler_config=sampler_config,
                            global_seed=global_seed,
                            sample_id=sample_identity(
                                test_dataset,
                                sample_index,
                                fallback_prefix="test",
                            ),
                        ),
                    ),
                ):
                    outputs = model(strategy_inputs)
                    total_loss += (
                        criterion(outputs, targets).item() * len(sample_counts)
                    )
                    rmse_per_sample = normalized_mse_per_sample(
                        outputs.clamp(0, 1), targets
                    ).sqrt()
                    total_rmse += rmse_per_sample.sum().item()
                    total_cases += len(sample_counts)
                    for count_index, sample_count in enumerate(sample_counts):
                        rmse_sums[strategy][sample_count] += (
                            rmse_per_sample[count_index].item()
                        )
                        case_counts[strategy][sample_count] += 1

                    if strategy == "random":
                        for bundle_sample_index, bundle_count in (
                            EVALUATION_BUNDLE_CASES
                        ):
                            if sample_index != bundle_sample_index:
                                continue
                            count_index = sample_counts.index(bundle_count)
                            bundles.append(
                                _evaluation_bundle(
                                    inputs=inputs[count_index],
                                    output=outputs[count_index, 0].clamp(0, 1),
                                    ground_truth=gain[count_index, 0],
                                    valid_receiving_area=targets["mask"][
                                        count_index, 0
                                    ],
                                    sample_count=bundle_count,
                                )
                            )
                progress.set_postfix(
                    loss=f"{(total_loss / total_cases):.6f}",
                    rmse=f"{(total_rmse / total_cases):.6f}",
                )
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
        sample_count: (
            rmse_sums["random"][sample_count] / case_counts["random"][sample_count],
            rmse_sums["guided"][sample_count] / case_counts["guided"][sample_count],
        )
        for sample_count in sample_counts
    }

    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    _write_test_metrics(evaluation_dir, rmse_by_sample_count)
    _write_evaluation_artifacts(
        evaluation_dir, bundles, rmse_by_sample_count
    )
    return total_loss / total_cases, total_rmse / total_cases, rmse_by_sample_count


def eval() -> None:
    test_dataset = RadioDataset("test")
    model = load_checkpoint(ROOT / RECONSTRUCTOR_RUN_PATH / "best.pt")
    coarse_model = load_coarse_checkpoint(
        ROOT / COARSE_RECONSTRUCTOR_RUN_PATH / "best.pt"
    )
    test_loss, test_rmse, rmse_by_sample_count = run_evaluation(
        model=model,
        coarse_model=coarse_model,
        test_dataset=test_dataset,
        criterion=RadioMapLoss(),
        sampler_config=CONFIG["sampler"],
        global_seed=CONFIG["seed"],
        evaluation_dir=ROOT / EVALUATION_RUN_PATH,
    )

    print(f"test loss: {test_loss:.6f}")
    print(f"test rmse: {test_rmse:.6f}")
    for sample_count, (random_rmse, guided_rmse) in (
        rmse_by_sample_count.items()
    ):
        print(
            f"test rmse ({sample_count} samples): "
            f"random={random_rmse:.6f} guided={guided_rmse:.6f}"
        )
