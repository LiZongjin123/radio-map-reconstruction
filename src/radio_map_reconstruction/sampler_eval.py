from pathlib import Path
from random import Random

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch import inference_mode, load, tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from radio_map_reconstruction.config import (
    CONFIG,
    reconstructor_run_dir,
    sampler_run_dir,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.eval import (
    deterministic_evaluation_runtime,
    load_checkpoint as load_reconstructor_checkpoint,
    write_test_metrics,
)
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.sampler import Sampler, straight_through_top_k
from radio_map_reconstruction.sampler_train import eval_sampler_one_epoch


SAMPLING_DECISION_SAMPLE_COUNTS = (10, 50, 100, 200)


def load_sampler_checkpoint(path: str | Path) -> Module:
    checkpoint = load(path, map_location=CONFIG["runtime"]["device"])
    sampler = Sampler(**checkpoint["sampler_model_config"]).to(
        CONFIG["runtime"]["device"]
    )
    sampler.load_state_dict(checkpoint["model_state_dict"])
    return sampler


def _save_sampling_decision_figure(
    *,
    sampling_score_map,
    learned_sampling_mask,
    valid_receiving_area,
    building_map,
    transmitter_map,
    sample_count: int,
    output_path: Path,
) -> None:
    selected_count = int(learned_sampling_mask.sum().item())
    if selected_count != sample_count:
        raise RuntimeError(
            f"Learned Sampling Mask must contain exactly {sample_count} "
            f"Valid Sampling Points; found {selected_count}"
        )

    valid_receiving_area_array = valid_receiving_area.cpu().numpy().astype(bool)
    score_values = np.ma.masked_where(
        ~valid_receiving_area_array,
        sampling_score_map.cpu().numpy(),
    )
    building_values = building_map.cpu().numpy()
    transmitter_rows, transmitter_columns = np.nonzero(
        transmitter_map.cpu().numpy() > 0.5
    )
    sampling_rows, sampling_columns = np.nonzero(
        learned_sampling_mask.cpu().numpy() > 0.5
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        constrained_layout=True,
    )
    score_colormap = matplotlib.colormaps["viridis"].with_extremes(
        bad="lightgray"
    )
    score_image = axes[0].imshow(score_values, cmap=score_colormap)
    axes[0].set_title("Sampling Score Map")
    axes[0].set_axis_off()
    figure.colorbar(score_image, ax=axes[0], label="Sampling Score")

    axes[1].imshow(building_values, cmap="gray_r", vmin=0, vmax=1)
    axes[1].scatter(
        transmitter_columns,
        transmitter_rows,
        marker="*",
        s=90,
        c="red",
        edgecolors="white",
        linewidths=0.7,
        label="Transmitter",
    )
    axes[1].scatter(
        sampling_columns,
        sampling_rows,
        marker="o",
        s=18,
        c="deepskyblue",
        edgecolors="black",
        linewidths=0.35,
        label="Valid Sampling Points",
    )
    axes[1].set_title(
        f"Learned Sampling Mask — {sample_count} Valid Sampling Points"
    )
    axes[1].set_axis_off()
    axes[1].legend(loc="upper right")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _write_sampling_decision_figures(
    *,
    evaluation_dir: Path,
    sampler: Module,
    test_dataset: Dataset,
    evaluation_seed: int,
    sampler_training_config: dict,
) -> None:
    evaluation_sample_counts = RadioDataset.EVALUATION_SAMPLE_COUNTS
    base_sample_count, remainder = divmod(
        len(test_dataset), len(evaluation_sample_counts)
    )
    if remainder or base_sample_count < len(SAMPLING_DECISION_SAMPLE_COUNTS):
        raise ValueError(
            "Sampler evaluation requires at least four complete base test samples"
        )

    selected_base_samples = Random(evaluation_seed).sample(
        range(base_sample_count), len(SAMPLING_DECISION_SAMPLE_COUNTS)
    )
    sampler.eval()
    with inference_mode():
        for base_sample_index, sample_count in zip(
            selected_base_samples,
            SAMPLING_DECISION_SAMPLE_COUNTS,
            strict=True,
        ):
            sample_count_index = evaluation_sample_counts.index(sample_count)
            dataset_index = (
                base_sample_index * len(evaluation_sample_counts)
                + sample_count_index
            )
            inputs, targets = test_dataset[dataset_index]
            device_inputs = inputs.unsqueeze(0).to(CONFIG["runtime"]["device"])
            building_map = device_inputs[:, 2:3]
            transmitter_map = device_inputs[:, 1:2]
            valid_receiving_area = targets["mask"].unsqueeze(0).to(
                CONFIG["runtime"]["device"]
            ).bool()
            sampling_score_map = sampler(building_map, transmitter_map)
            learned_sampling_mask = straight_through_top_k(
                sampling_score_map,
                valid_receiving_area,
                tensor([sample_count], device=sampling_score_map.device),
                temperature=sampler_training_config["temperature"],
                tolerance=sampler_training_config["bisection_tolerance"],
                max_iterations=sampler_training_config[
                    "bisection_max_iterations"
                ],
            )
            _save_sampling_decision_figure(
                sampling_score_map=sampling_score_map[0, 0],
                learned_sampling_mask=learned_sampling_mask[0, 0],
                valid_receiving_area=valid_receiving_area[0, 0],
                building_map=building_map[0, 0],
                transmitter_map=transmitter_map[0, 0],
                sample_count=sample_count,
                output_path=(
                    evaluation_dir
                    / f"sampling_decision_{sample_count}_samples.png"
                ),
            )


def run_sampler_evaluation(
    *,
    sampler: Module,
    reconstructor: Module,
    test_dataset: Dataset,
    evaluation_dir: str | Path,
    batch_size: int,
    evaluation_seed: int,
    sampler_training_config: dict,
) -> dict[int, float]:
    with deterministic_evaluation_runtime():
        evaluation_dir = Path(evaluation_dir)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        _, rmse_by_sample_count = eval_sampler_one_epoch(
            sampler=sampler,
            reconstructor=reconstructor,
            dataloader=test_loader,
            criterion=RadioMapLoss(),
            sampler_training_config=sampler_training_config,
            epoch=0,
            epoch_num=1,
        )
        write_test_metrics(evaluation_dir, rmse_by_sample_count)
        _write_sampling_decision_figures(
            evaluation_dir=evaluation_dir,
            sampler=sampler,
            test_dataset=test_dataset,
            evaluation_seed=evaluation_seed,
            sampler_training_config=sampler_training_config,
        )
        return rmse_by_sample_count


def eval_sampler() -> None:
    sampler_dir = sampler_run_dir()
    reconstructor = load_reconstructor_checkpoint(
        reconstructor_run_dir() / "best.pt"
    )
    sampler = load_sampler_checkpoint(sampler_dir / "best.pt")
    rmse_by_sample_count = run_sampler_evaluation(
        sampler=sampler,
        reconstructor=reconstructor,
        test_dataset=RadioDataset("test"),
        evaluation_dir=sampler_dir / "evaluation",
        batch_size=CONFIG["evaluation"]["batch_size"],
        evaluation_seed=CONFIG["runtime"]["seed"],
        sampler_training_config=CONFIG["sampler"]["training"],
    )

    for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS:
        print(
            f"test rmse ({sample_count} samples): "
            f"{rmse_by_sample_count[sample_count]:.6f}"
        )
