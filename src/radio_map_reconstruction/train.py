import csv
import swanlab
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from torch import Tensor, inference_mode, save
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from os.path import join
from torch.nn import Module
from torch.optim import Adam, Optimizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from yaml import safe_load
from radio_map_reconstruction.artifacts import (
    COARSE_RECONSTRUCTOR_RUN_PATH,
    RECONSTRUCTOR_RUN_PATH,
)
from radio_map_reconstruction.data import CoarseRadioDataset, RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss, normalized_mse_per_sample
from radio_map_reconstruction.model import ResUnet

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

SAMPLING_MASK_CHANNEL = 3


class ValidationRmseMode(Enum):
    BY_SAMPLE_COUNT = auto()
    PER_SAMPLE = auto()

def train_one_epoch(
        model: Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: Module,
        epoch: int,
        epoch_num: int
) -> float:
    model.train()
    progress = tqdm(dataloader, desc=f"train {epoch + 1}/{epoch_num}", unit="batch", leave=True)
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in progress:
        optimizer.zero_grad()
        inputs = inputs.to(CONFIG["device"])
        targets = {name: value.to(CONFIG["device"]) for name, value in targets.items()}
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss = f"{(total_loss / total_samples):.6f}"
        )

    return total_loss / total_samples

def eval_one_epoch(
        model: Module,
        dataloader: DataLoader,
        criterion: Module,
        epoch: int,
        epoch_num: int,
        validation_rmse_mode: ValidationRmseMode = ValidationRmseMode.BY_SAMPLE_COUNT,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    progress = tqdm(dataloader, desc=f"eval {epoch + 1}/{epoch_num}", unit="batch", leave=True)
    total_loss = 0.0
    total_rmse = 0.0
    total_samples = 0
    rmse_sums_by_sample_count: dict[int, float] = {}
    case_counts_by_sample_count: dict[int, int] = {}
    with inference_mode():
        for inputs, targets in progress:
            inputs = inputs.to(CONFIG["device"])
            targets = {name: value.to(CONFIG["device"]) for name, value in targets.items()}
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size

            rmse_per_sample = normalized_mse_per_sample(
                outputs.clamp(0, 1), targets
            ).sqrt()
            total_rmse += rmse_per_sample.sum().item()
            total_samples += batch_size

            if validation_rmse_mode is ValidationRmseMode.BY_SAMPLE_COUNT:
                sample_counts = inputs[:, SAMPLING_MASK_CHANNEL].flatten(
                    start_dim=1
                ).sum(dim=1)
                for sample_count, rmse in zip(
                    sample_counts.tolist(), rmse_per_sample.tolist(), strict=True
                ):
                    count = int(sample_count)
                    rmse_sums_by_sample_count[count] = (
                        rmse_sums_by_sample_count.get(count, 0.0) + rmse
                    )
                    case_counts_by_sample_count[count] = (
                        case_counts_by_sample_count.get(count, 0) + 1
                    )

            progress.set_postfix(
                loss = f"{(total_loss / total_samples):.6f}",
                rmse = f"{(total_rmse / total_samples):.6f}"
            )

    rmse_by_sample_count = {
        sample_count: rmse_sum / case_counts_by_sample_count[sample_count]
        for sample_count, rmse_sum in rmse_sums_by_sample_count.items()
    }
    return (
        total_loss / total_samples,
        total_rmse / total_samples,
        rmse_by_sample_count,
    )


def run_training(
        *,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        epoch_num: int,
        run_dir: str | Path,
        metric_logger: Callable[[dict[str, int | float]], None] | None = None,
        validation_rmse_mode: ValidationRmseMode = ValidationRmseMode.BY_SAMPLE_COUNT,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    fieldnames = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_mean_per_sample_normalized_rmse",
        "learning_rate",
    )
    best_val_rmse = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epoch_num):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, epoch, epoch_num
            )
            val_loss, val_rmse, val_rmse_by_sample_count = eval_one_epoch(
                model,
                val_loader,
                criterion,
                epoch,
                epoch_num,
                validation_rmse_mode=validation_rmse_mode,
            )
            if validation_rmse_mode is ValidationRmseMode.PER_SAMPLE:
                val_mean_per_sample_normalized_rmse = val_rmse
            else:
                val_mean_per_sample_normalized_rmse = sum(
                    val_rmse_by_sample_count[sample_count]
                    for sample_count in RadioDataset.EVALUATION_SAMPLE_COUNTS
                ) / len(RadioDataset.EVALUATION_SAMPLE_COUNTS)

            metrics: dict[str, int | float] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_per_sample_normalized_rmse": (
                    val_mean_per_sample_normalized_rmse
                ),
                "learning_rate": learning_rate,
            }
            writer.writerow(metrics)
            history_file.flush()

            save_checkpoint(run_dir / "latest.pt", model)
            if val_mean_per_sample_normalized_rmse < best_val_rmse:
                best_val_rmse = val_mean_per_sample_normalized_rmse
                save_checkpoint(run_dir / "best.pt", model)

            if metric_logger is not None:
                metric_logger({
                    **metrics,
                    **{
                        (
                            "val_mean_per_sample_normalized_rmse_"
                            f"{sample_count}_samples"
                        ): rmse
                        for sample_count, rmse in val_rmse_by_sample_count.items()
                    },
                })
            scheduler.step()

def _train_model_role(
    *,
    role_config: dict,
    train_data: RadioDataset | CoarseRadioDataset,
    val_data: RadioDataset | CoarseRadioDataset,
    run_path: Path,
    validation_rmse_mode: ValidationRmseMode,
) -> None:
    train_loader = DataLoader(
        dataset=train_data,
        num_workers=role_config["data_loader"]["num_workers"],
        batch_size=role_config["data_loader"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        num_workers=role_config["data_loader"]["num_workers"],
        batch_size=role_config["data_loader"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    swanlab.init(
        project=CONFIG["swanlab"]["project"],
        workspace=CONFIG["swanlab"]["workspace"],
        config={
            "learning_rate": role_config["optimizer"]["learning_rate"],
            "epochs": role_config["training"]["epochs"],
            "t_max": role_config["scheduler"]["t_max"],
            "eta_min": role_config["scheduler"]["eta_min"],
        },
    )

    res_unet = ResUnet(**role_config["model"]).to(CONFIG["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(
        res_unet.parameters(),
        lr=role_config["optimizer"]["learning_rate"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=role_config["scheduler"]["t_max"],
        eta_min=role_config["scheduler"]["eta_min"],
    )

    try:
        run_training(
            model=res_unet,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=role_config["training"]["epochs"],
            run_dir=ROOT / run_path,
            metric_logger=swanlab.log,
            validation_rmse_mode=validation_rmse_mode,
        )
    finally:
        swanlab.finish()


def train() -> None:
    _train_model_role(
        role_config=CONFIG["reconstructor"],
        train_data=RadioDataset("train"),
        val_data=RadioDataset("val"),
        run_path=RECONSTRUCTOR_RUN_PATH,
        validation_rmse_mode=ValidationRmseMode.BY_SAMPLE_COUNT,
    )


def train_coarse() -> None:
    coarse_config = CONFIG["coarse_reconstructor"]
    dataset_config = {
        "dataset_path": CONFIG["dataset_path"],
        "partition": CONFIG["partition"],
        "seed": CONFIG["seed"],
    }
    train_data = CoarseRadioDataset("train", **dataset_config)
    val_data = CoarseRadioDataset("val", **dataset_config)
    _train_model_role(
        role_config=coarse_config,
        train_data=train_data,
        val_data=val_data,
        run_path=COARSE_RECONSTRUCTOR_RUN_PATH,
        validation_rmse_mode=ValidationRmseMode.PER_SAMPLE,
    )

def save_checkpoint(
        path: str | Path,
        model: Module
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save(
        {
            "model_state_dict": model.state_dict()
        },
        path
    )
