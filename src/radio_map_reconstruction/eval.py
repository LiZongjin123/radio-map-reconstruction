import csv
from pathlib import Path

import numpy as np
from torch import (
    are_deterministic_algorithms_enabled,
    backends,
    inference_mode,
    is_deterministic_algorithms_warn_only_enabled,
    load,
    use_deterministic_algorithms,
    zeros_like,
)
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.model import ResUnet
from radio_map_reconstruction.train import CONFIG, eval_one_epoch

ROOT = Path(__file__).resolve().parents[2]

EVALUATION_BUNDLE_CASES = (
    (0, 10),
    (1, 50),
    (2, 100),
    (3, 200),
)


def load_checkpoint(path: str | Path) -> Module:
    checkpoint = load(path, map_location=CONFIG["device"])
    model = ResUnet().to(CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_evaluation(
    *,
    model: Module,
    test_dataset: Dataset,
    criterion: Module,
    evaluation_dir: str | Path,
    batch_size: int,
) -> tuple[float, float, dict[int, float]]:
    deterministic_algorithms_were_enabled = (
        are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only_was_enabled = (
        is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_deterministic_was_enabled = backends.cudnn.deterministic
    cudnn_benchmark_was_enabled = backends.cudnn.benchmark
    use_deterministic_algorithms(True)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    try:
        evaluation_dir = Path(evaluation_dir)
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        test_loss, test_rmse, rmse_by_sample_count = eval_one_epoch(
            model, test_loader, criterion, epoch=0, epoch_num=1
        )
        _write_test_metrics(evaluation_dir, rmse_by_sample_count)
        _write_evaluation_bundles(evaluation_dir, model, test_dataset)
        return test_loss, test_rmse, rmse_by_sample_count
    finally:
        use_deterministic_algorithms(
            deterministic_algorithms_were_enabled,
            warn_only=deterministic_warn_only_was_enabled,
        )
        backends.cudnn.deterministic = cudnn_deterministic_was_enabled
        backends.cudnn.benchmark = cudnn_benchmark_was_enabled


def _write_test_metrics(
    evaluation_dir: Path,
    rmse_by_sample_count: dict[int, float],
) -> None:
    metrics_path = evaluation_dir / "test_metrics.csv"
    fieldnames = (
        "sample_count",
        "mean_per_sample_normalized_rmse",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS:
            writer.writerow({
                "sample_count": sample_count,
                "mean_per_sample_normalized_rmse": rmse_by_sample_count[
                    sample_count
                ],
            })


def _write_evaluation_bundles(
    evaluation_dir: Path,
    model: Module,
    test_dataset: Dataset,
) -> None:
    model.eval()
    sample_counts = RadioDataset.EVALUATION_SAMPLE_COUNTS
    with inference_mode():
        for bundle_index, (sample_index, sample_count) in enumerate(
            EVALUATION_BUNDLE_CASES
        ):
            sample_count_index = sample_counts.index(sample_count)
            dataset_index = sample_index * len(sample_counts) + sample_count_index
            inputs, targets = test_dataset[dataset_index]
            device_inputs = inputs.unsqueeze(0).to(CONFIG["device"])
            output = model(device_inputs)[0, 0].clamp(0, 1)

            ground_truth = targets["gain"][0].to(output.device)
            valid_receiving_area = targets["mask"][0].to(output.device).bool()
            transmitter = device_inputs[0, 1] > 0.5
            building = device_inputs[0, 2] > 0.5
            sampling_mask = (
                (device_inputs[0, 3] > 0.5) & valid_receiving_area
            )
            sparse_map = device_inputs[0, 0].where(
                sampling_mask, zeros_like(device_inputs[0, 0])
            )

            reconstruction = output.clone()
            reconstruction[building] = 0
            reconstruction[transmitter] = 1
            absolute_error = (reconstruction - ground_truth).abs()
            absolute_error[~valid_receiving_area] = float("nan")

            np.savez_compressed(
                evaluation_dir / f"evaluation_bundle_{bundle_index}.npz",
                sample_count=np.asarray(sample_count),
                ground_truth=ground_truth.cpu().numpy(),
                sampling_mask=sampling_mask.cpu().numpy(),
                sparse_map=sparse_map.cpu().numpy(),
                reconstruction=reconstruction.cpu().numpy(),
                absolute_error=absolute_error.cpu().numpy(),
            )


def eval() -> None:
    test_dataset = RadioDataset("test")
    model = load_checkpoint(ROOT / "run" / "best.pt")
    test_loss, test_rmse, rmse_by_sample_count = run_evaluation(
        model=model,
        test_dataset=test_dataset,
        criterion=RadioMapLoss(),
        evaluation_dir=ROOT / "run" / "evaluation",
        batch_size=CONFIG["data_loader"]["batch_size"],
    )

    print(f"test loss: {test_loss:.6f}")
    print(f"test rmse: {test_rmse:.6f}")
    for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS:
        print(
            f"test rmse ({sample_count} samples): "
            f"{rmse_by_sample_count[sample_count]:.6f}"
        )
