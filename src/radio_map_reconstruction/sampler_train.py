import csv
import swanlab
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, cat, no_grad, save
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from radio_map_reconstruction.config import (
    CONFIG,
    reconstructor_run_dir,
    sampler_run_dir,
)
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.eval import load_checkpoint as load_reconstructor_checkpoint
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.loss import normalized_mse_per_sample
from radio_map_reconstruction.sampler import Sampler, straight_through_top_k
from radio_map_reconstruction.util import delete_sampler_run


def _sampler_inputs_and_targets(
    inputs: Tensor,
    targets: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = CONFIG["runtime"]["device"]
    device_inputs = inputs.to(device)
    device_targets = {
        name: value.to(device) for name, value in targets.items()
    }
    transmitter_map = device_inputs[:, 1:2]
    building_map = device_inputs[:, 2:3]
    return (
        building_map,
        transmitter_map,
        device_targets["gain"],
        device_targets["mask"],
    )


def _learned_reconstructor_inputs(
    *,
    sampler: Module,
    building_map: Tensor,
    transmitter_map: Tensor,
    dense_target: Tensor,
    valid_receiving_area: Tensor,
    sample_counts: Tensor,
    sampler_training_config: dict,
) -> Tensor:
    sampling_score_map = sampler(building_map, transmitter_map)
    learned_sampling_mask = straight_through_top_k(
        sampling_score_map,
        valid_receiving_area,
        sample_counts,
        temperature=sampler_training_config["temperature"],
        tolerance=sampler_training_config["bisection_tolerance"],
        max_iterations=sampler_training_config["bisection_max_iterations"],
    )
    sparse_channel_map = dense_target * learned_sampling_mask
    return cat(
        (
            sparse_channel_map,
            transmitter_map,
            building_map,
            learned_sampling_mask,
        ),
        dim=1,
    )


def train_sampler_one_epoch(
    *,
    sampler: Module,
    reconstructor: Module,
    dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    sampler_training_config: dict,
    epoch: int,
    epoch_num: int,
) -> float:
    sampler.train()
    reconstructor.eval()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        dataloader,
        desc=f"sampler train {epoch + 1}/{epoch_num}",
        unit="batch",
        leave=True,
    )

    for inputs, targets in progress:
        building_map, transmitter_map, dense_target, valid_receiving_area = (
            _sampler_inputs_and_targets(inputs, targets)
        )
        sample_counts = torch.randint(
            sampler_training_config["minimum_sample_count"],
            sampler_training_config["maximum_sample_count"] + 1,
            (inputs.shape[0],),
            device=dense_target.device,
        )
        optimizer.zero_grad()
        reconstructor_inputs = _learned_reconstructor_inputs(
            sampler=sampler,
            building_map=building_map,
            transmitter_map=transmitter_map,
            dense_target=dense_target,
            valid_receiving_area=valid_receiving_area,
            sample_counts=sample_counts,
            sampler_training_config=sampler_training_config,
        )
        outputs = reconstructor(reconstructor_inputs)
        loss = criterion(outputs, {
            "gain": dense_target,
            "mask": valid_receiving_area,
        })
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.6f}")

    return total_loss / total_samples


def eval_sampler_one_epoch(
    *,
    sampler: Module,
    reconstructor: Module,
    dataloader: DataLoader,
    criterion: Module,
    sampler_training_config: dict,
    epoch: int,
    epoch_num: int,
) -> tuple[float, dict[int, float]]:
    sampler.eval()
    reconstructor.eval()
    total_loss = 0.0
    total_samples = 0
    rmse_sums_by_sample_count: dict[int, float] = {}
    case_counts_by_sample_count: dict[int, int] = {}
    progress = tqdm(
        dataloader,
        desc=f"sampler eval {epoch + 1}/{epoch_num}",
        unit="batch",
        leave=True,
    )

    with no_grad():
        for inputs, targets in progress:
            building_map, transmitter_map, dense_target, valid_receiving_area = (
                _sampler_inputs_and_targets(inputs, targets)
            )
            sample_counts = inputs[:, 3].flatten(start_dim=1).sum(dim=1).to(
                device=dense_target.device,
                dtype=torch.int64,
            )
            reconstructor_inputs = _learned_reconstructor_inputs(
                sampler=sampler,
                building_map=building_map,
                transmitter_map=transmitter_map,
                dense_target=dense_target,
                valid_receiving_area=valid_receiving_area,
                sample_counts=sample_counts,
                sampler_training_config=sampler_training_config,
            )
            outputs = reconstructor(reconstructor_inputs)
            target_dict = {
                "gain": dense_target,
                "mask": valid_receiving_area,
            }
            loss = criterion(outputs, target_dict)
            rmse_per_sample = normalized_mse_per_sample(
                outputs.clamp(0, 1), target_dict
            ).sqrt()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            for sample_count, rmse in zip(
                sample_counts.tolist(), rmse_per_sample.tolist(), strict=True
            ):
                rmse_sums_by_sample_count[sample_count] = (
                    rmse_sums_by_sample_count.get(sample_count, 0.0) + rmse
                )
                case_counts_by_sample_count[sample_count] = (
                    case_counts_by_sample_count.get(sample_count, 0) + 1
                )

    return total_loss / total_samples, {
        sample_count: rmse_sum / case_counts_by_sample_count[sample_count]
        for sample_count, rmse_sum in rmse_sums_by_sample_count.items()
    }


def save_sampler_checkpoint(
    path: str | Path,
    *,
    sampler: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    sampler_model_config: dict,
    sampler_training_config: dict,
    training_state: dict,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save(
        {
            "sampler_model_config": sampler_model_config,
            "sampler_training_config": sampler_training_config,
            "model_state_dict": sampler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "training_state": training_state,
        },
        path,
    )


def run_sampler_training(
    *,
    sampler: Module,
    reconstructor: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch_num: int,
    run_dir: str | Path,
    sampler_model_config: dict,
    sampler_training_config: dict,
    metric_logger: Callable[[dict[str, int | float]], None] | None = None,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for parameter in reconstructor.parameters():
        parameter.requires_grad_(False)
    reconstructor.eval()

    fieldnames = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_mean_per_sample_normalized_rmse",
        "learning_rate",
    )
    best_val_rmse = float("inf")
    with (run_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epoch_num):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss = train_sampler_one_epoch(
                sampler=sampler,
                reconstructor=reconstructor,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                sampler_training_config=sampler_training_config,
                epoch=epoch,
                epoch_num=epoch_num,
            )
            val_loss, val_rmse_by_sample_count = eval_sampler_one_epoch(
                sampler=sampler,
                reconstructor=reconstructor,
                dataloader=val_loader,
                criterion=criterion,
                sampler_training_config=sampler_training_config,
                epoch=epoch,
                epoch_num=epoch_num,
            )
            val_mean_rmse = sum(
                val_rmse_by_sample_count[sample_count]
                for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS
            ) / len(RadioDataset.EVALUATION_SAMPLE_COUNTS)
            metrics: dict[str, int | float] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_per_sample_normalized_rmse": val_mean_rmse,
                "learning_rate": learning_rate,
            }
            writer.writerow(metrics)
            history_file.flush()

            is_best = val_mean_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_mean_rmse
            scheduler.step()
            training_state = {
                "best_val_mean_per_sample_normalized_rmse": best_val_rmse,
            }
            save_sampler_checkpoint(
                run_dir / "latest.pt",
                sampler=sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                sampler_model_config=sampler_model_config,
                sampler_training_config=sampler_training_config,
                training_state=training_state,
            )
            if is_best:
                save_sampler_checkpoint(
                    run_dir / "best.pt",
                    sampler=sampler,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    sampler_model_config=sampler_model_config,
                    sampler_training_config=sampler_training_config,
                    training_state=training_state,
                )

            if metric_logger is not None:
                metric_logger({
                    **metrics,
                    **{
                        f"val_mean_per_sample_normalized_rmse_{count}_samples": rmse
                        for count, rmse in val_rmse_by_sample_count.items()
                    },
                })


def train_sampler() -> None:
    data_loader_config = CONFIG["dataset"]["data_loader"]
    model_config = CONFIG["sampler"]["model"]
    training_config = CONFIG["sampler"]["training"]
    run_dir = sampler_run_dir()
    delete_sampler_run()
    run_dir.mkdir(parents=True)

    train_loader = DataLoader(
        dataset=RadioDataset("train"),
        num_workers=data_loader_config["num_workers"],
        batch_size=data_loader_config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=RadioDataset("val"),
        num_workers=data_loader_config["num_workers"],
        batch_size=data_loader_config["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    swanlab.init(
        project=CONFIG["experiment_logging"]["project"],
        workspace=CONFIG["experiment_logging"]["workspace"],
        config={
            **model_config,
            **training_config,
        },
    )
    reconstructor = load_reconstructor_checkpoint(
        reconstructor_run_dir() / "best.pt"
    )
    sampler = Sampler(**model_config).to(CONFIG["runtime"]["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(
        sampler.parameters(), lr=training_config["optimizer"]["init_lr"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config["scheduler"]["T_max"],
        eta_min=training_config["scheduler"]["eta_min"],
    )

    try:
        run_sampler_training(
            sampler=sampler,
            reconstructor=reconstructor,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=training_config["epochs"],
            run_dir=run_dir,
            sampler_model_config=model_config,
            sampler_training_config=training_config,
            metric_logger=swanlab.log,
        )
    finally:
        swanlab.finish()
