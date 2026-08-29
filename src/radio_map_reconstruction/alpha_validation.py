import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor, cat, inference_mode, load, stack
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


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    CONFIG = safe_load(config_file)


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


def _sample_identity(validation_dataset: Dataset, index: int) -> str:
    samples = getattr(validation_dataset, "samples", None)
    if samples is None:
        return f"validation-{index}"
    gain_path = samples[index][0]
    return Path(gain_path).stem


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
) -> dict[float, float]:
    """Evaluate configured alpha candidates on a full validation partition."""
    candidates = _validated_candidates(sampler_config)
    counts = tuple(sample_counts)
    if not counts:
        raise ValueError("sample_counts must not be empty")

    coarse_was_training = coarse_model.training
    reconstructor_was_training = reconstructor.training
    coarse_model.eval()
    reconstructor.eval()
    rmse_sums = {alpha: 0.0 for alpha in candidates}
    case_counts = {alpha: 0 for alpha in candidates}

    try:
        with inference_mode():
            for sample_index in range(len(validation_dataset)):
                coarse_inputs, targets = validation_dataset[sample_index]
                coarse_inputs = coarse_inputs.to(device)
                gain = targets["gain"].to(device)
                valid_receiving_area = targets["mask"].to(device)
                tx_map = coarse_inputs[0:1]
                building_map = coarse_inputs[1:2]
                coarse_map = coarse_model(coarse_inputs.unsqueeze(0))[0]
                sample_id = _sample_identity(validation_dataset, sample_index)

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
    finally:
        coarse_model.train(coarse_was_training)
        reconstructor.train(reconstructor_was_training)

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


def tune_alpha() -> None:
    dataset_config = {
        "dataset_path": CONFIG["dataset_path"],
        "partition": CONFIG["partition"],
        "seed": CONFIG["seed"],
    }
    metrics = run_alpha_validation(
        coarse_model=_load_role_model(
            CONFIG["coarse_reconstructor"],
            ROOT / COARSE_RECONSTRUCTOR_RUN_PATH / "best.pt",
        ),
        reconstructor=_load_role_model(
            CONFIG["reconstructor"],
            ROOT / RECONSTRUCTOR_RUN_PATH / "best.pt",
        ),
        validation_dataset=CoarseRadioDataset("val", **dataset_config),
        sampler_config=CONFIG["sampler"],
        global_seed=CONFIG["seed"],
        sample_counts=RadioDataset.EVALUATION_SAMPLE_COUNTS,
        output_dir=ROOT / SAMPLER_RUN_PATH,
        device=CONFIG["device"],
    )
    for alpha, rmse in metrics.items():
        print(f"alpha={alpha:g}: validation rmse={rmse:.6f}")
