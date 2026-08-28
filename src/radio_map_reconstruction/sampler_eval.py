from pathlib import Path

from torch import load
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
from radio_map_reconstruction.sampler import Sampler
from radio_map_reconstruction.sampler_train import eval_sampler_one_epoch


def load_sampler_checkpoint(path: str | Path) -> Module:
    checkpoint = load(path, map_location=CONFIG["runtime"]["device"])
    sampler = Sampler(**checkpoint["sampler_model_config"]).to(
        CONFIG["runtime"]["device"]
    )
    sampler.load_state_dict(checkpoint["model_state_dict"])
    return sampler


def run_sampler_evaluation(
    *,
    sampler: Module,
    reconstructor: Module,
    test_dataset: Dataset,
    evaluation_dir: str | Path,
    batch_size: int,
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
        sampler_training_config=CONFIG["sampler"]["training"],
    )

    for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS:
        print(
            f"test rmse ({sample_count} samples): "
            f"{rmse_by_sample_count[sample_count]:.6f}"
        )
