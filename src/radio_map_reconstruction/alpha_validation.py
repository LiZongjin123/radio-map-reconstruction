import csv
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
)
from torch.nn import Module
from torch.utils.data import Dataset
from yaml import safe_load

from radio_map_reconstruction.artifacts import (
    COARSE_RECONSTRUCTOR_RUN_PATH,
    RECONSTRUCTOR_RUN_PATH,
    SAMPLER_RUN_PATH,
)
from radio_map_reconstruction.data import CoarseRadioDataset, RadioDataset
from radio_map_reconstruction.loss import normalized_mse_per_sample
from radio_map_reconstruction.model import ResUnet
from radio_map_reconstruction.sampling import (
    gradient_distance_weighted_clustering_sample,
)
from radio_map_reconstruction.util import sample_identity, select_example_indices


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = safe_load(config_file)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise ArgumentTypeError("must be a positive integer")
    return parsed


@dataclass(frozen=True)
class AlphaValidationProgress:
    example_index: int
    example_position: int
    total_examples: int
    example_id: str
    alpha: float
    completed_pairs: int
    total_pairs: int
    eta_seconds: float


def report_alpha_validation_progress(progress: AlphaValidationProgress) -> None:
    print(
        "alpha validation: "
        f"Example {progress.example_position}/{progress.total_examples} "
        f"({progress.example_id}) alpha={progress.alpha:g} "
        f"completed={progress.completed_pairs}/{progress.total_pairs} "
        f"ETA={progress.eta_seconds:.1f}s"
    )


def _validated_candidates(sampler_config: dict) -> tuple[float, ...]:
    candidates = tuple(sampler_config["alpha_candidates"])
    if not candidates:
        raise ValueError("sampler.alpha_candidates must not be empty")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, (int, float))
        or not 0 <= candidate <= 1
        for candidate in candidates
    ):
        raise ValueError("sampler.alpha_candidates must contain numbers from 0 to 1")
    numeric_candidates = tuple(float(candidate) for candidate in candidates)
    if len(set(numeric_candidates)) != len(numeric_candidates):
        raise ValueError("sampler.alpha_candidates must not contain duplicates")
    return tuple(sorted(numeric_candidates))


def _save_rmse_vs_alpha(metrics: dict[float, float], output_path: Path) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(list(metrics), list(metrics.values()), marker="o")
    axes.set_title("Validation RMSE vs. Alpha")
    axes.set_xlabel("Alpha")
    axes.set_ylabel("Mean Per-Sample Normalized RMSE")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def run_alpha_validation(
    *,
    coarse_model: Module,
    reconstructor: Module,
    validation_dataset: Dataset,
    sampler_config: dict,
    global_seed: int,
    sample_counts: Sequence[int],
    output_dir: str | Path,
    device: str,
    requested_examples: int | None = None,
    progress_reporter: Callable[[AlphaValidationProgress], None] | None = (
        report_alpha_validation_progress
    ),
) -> dict[float, float]:
    """Evaluate configured alpha candidates on selected validation Examples."""
    candidates = _validated_candidates(sampler_config)
    counts = tuple(sample_counts)
    if not counts:
        raise ValueError("sample_counts must not be empty")
    example_indices = select_example_indices(
        available_examples=len(validation_dataset),
        requested_examples=requested_examples,
        global_seed=global_seed,
    )

    coarse_was_training = coarse_model.training
    reconstructor_was_training = reconstructor.training
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only_was_enabled = (
        is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_deterministic_was_enabled = backends.cudnn.deterministic
    cudnn_benchmark_was_enabled = backends.cudnn.benchmark
    coarse_model.eval()
    reconstructor.eval()
    use_deterministic_algorithms(True)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    rmse_sums = {alpha: 0.0 for alpha in candidates}
    case_counts = {alpha: 0 for alpha in candidates}
    total_pairs = len(example_indices) * len(candidates)
    completed_pairs = 0
    started_at = monotonic()

    try:
        with inference_mode():
            for example_position, sample_index in enumerate(example_indices, start=1):
                coarse_inputs, targets = validation_dataset[sample_index]
                coarse_inputs = coarse_inputs.to(device)
                gain = targets["gain"].to(device)
                valid_receiving_area = targets["mask"].to(device)
                tx_map = coarse_inputs[0:1]
                building_map = coarse_inputs[1:2]
                coarse_map = coarse_model(coarse_inputs.unsqueeze(0))[0]
                sample_id = sample_identity(
                    validation_dataset, sample_index, fallback_prefix="validation"
                )

                for alpha in candidates:
                    main_inputs: list[Tensor] = []
                    for sample_count in counts:
                        sampling_mask, _ = (
                            gradient_distance_weighted_clustering_sample(
                                coarse_map,
                                tx_map,
                                building_map,
                                sample_count,
                                alpha=alpha,
                                global_seed=global_seed,
                                sample_id=sample_id,
                                weight_epsilon=sampler_config["weight_epsilon"],
                                max_iter=sampler_config["max_iter"],
                                tolerance=sampler_config["tolerance"],
                            )
                        )
                        main_inputs.append(
                            cat(
                                (
                                    gain * sampling_mask,
                                    tx_map,
                                    building_map,
                                    sampling_mask,
                                )
                            )
                        )

                    batched_inputs = stack(main_inputs)
                    outputs = reconstructor(batched_inputs).clamp(0, 1)
                    batch_size = len(counts)
                    rmse = normalized_mse_per_sample(
                        outputs,
                        {
                            "gain": gain.unsqueeze(0).expand(
                                batch_size, -1, -1, -1
                            ),
                            "mask": valid_receiving_area.unsqueeze(0).expand(
                                batch_size, -1, -1, -1
                            ),
                        },
                    ).sqrt()
                    rmse_sums[alpha] += rmse.sum().item()
                    case_counts[alpha] += batch_size
                    completed_pairs += 1
                    elapsed_seconds = monotonic() - started_at
                    eta_seconds = (
                        elapsed_seconds
                        / completed_pairs
                        * (total_pairs - completed_pairs)
                    )
                    if progress_reporter is not None:
                        progress_reporter(
                            AlphaValidationProgress(
                                example_index=sample_index,
                                example_position=example_position,
                                total_examples=len(example_indices),
                                example_id=sample_id,
                                alpha=alpha,
                                completed_pairs=completed_pairs,
                                total_pairs=total_pairs,
                                eta_seconds=eta_seconds,
                            )
                        )
    finally:
        coarse_model.train(coarse_was_training)
        reconstructor.train(reconstructor_was_training)
        use_deterministic_algorithms(
            deterministic_algorithms_were_enabled,
            warn_only=deterministic_warn_only_was_enabled,
        )
        backends.cudnn.deterministic = cudnn_deterministic_was_enabled
        backends.cudnn.benchmark = cudnn_benchmark_was_enabled

    metrics = {
        alpha: rmse_sums[alpha] / case_counts[alpha] for alpha in candidates
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "alpha_validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=(
                "alpha",
                "validation_mean_per_sample_normalized_rmse",
            ),
        )
        writer.writeheader()
        for alpha, rmse in metrics.items():
            writer.writerow(
                {
                    "alpha": alpha,
                    "validation_mean_per_sample_normalized_rmse": rmse,
                }
            )
    _save_rmse_vs_alpha(metrics, output_dir / "rmse_vs_alpha.png")
    return metrics


def _load_role_model(role_config: dict, checkpoint_path: Path) -> Module:
    model = ResUnet(**role_config["model"]).to(CONFIG["device"])
    checkpoint = load(checkpoint_path, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def tune_alpha(argv: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(description="Tune the guided sampler alpha value.")
    parser.add_argument(
        "--examples",
        type=_positive_int,
        help="deterministically select exactly N validation Examples",
    )
    arguments = parser.parse_args(argv)
    dataset_config = {
        "dataset_path": CONFIG["dataset_path"],
        "partition": CONFIG["partition"],
        "seed": CONFIG["seed"],
    }
    validation_dataset = CoarseRadioDataset("val", **dataset_config)
    select_example_indices(
        available_examples=len(validation_dataset),
        requested_examples=arguments.examples,
        global_seed=CONFIG["seed"],
    )
    metrics = run_alpha_validation(
        coarse_model=_load_role_model(
            CONFIG["coarse_reconstructor"],
            ROOT / COARSE_RECONSTRUCTOR_RUN_PATH / "best.pt",
        ),
        reconstructor=_load_role_model(
            CONFIG["reconstructor"],
            ROOT / RECONSTRUCTOR_RUN_PATH / "best.pt",
        ),
        validation_dataset=validation_dataset,
        sampler_config=CONFIG["sampler"],
        global_seed=CONFIG["seed"],
        sample_counts=RadioDataset.EVALUATION_SAMPLE_COUNTS,
        output_dir=ROOT / SAMPLER_RUN_PATH,
        device=CONFIG["device"],
        requested_examples=arguments.examples,
    )
    for alpha, rmse in metrics.items():
        print(f"alpha={alpha:g}: validation rmse={rmse:.6f}")
