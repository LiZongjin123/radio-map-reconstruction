import swanlab
from pathlib import Path
from torch import Tensor, inference_mode, save
from torch.optim.lr_scheduler import CosineAnnealingLR
from os.path import join
from torch.nn import Module
from torch.optim import Adam, Optimizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from yaml import safe_load
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.model import ResUnet

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

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
        epoch_num: int
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

            squared_error = (outputs.clamp(0, 1) - targets["gain"]).square()
            error_per_sample = (
                squared_error * targets["mask"]
            ).flatten(start_dim=1).sum(dim=1)
            valid_pixels_per_sample = targets["mask"].flatten(start_dim=1).sum(dim=1)
            rmse_per_sample = (error_per_sample / valid_pixels_per_sample).sqrt()
            total_rmse += rmse_per_sample.sum().item()
            total_samples += batch_size

            sample_counts = inputs[:, 3].flatten(start_dim=1).sum(dim=1)
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

def train() -> None:
    train_data = RadioDataset("train")
    val_data = RadioDataset("val")

    train_loader = DataLoader(
        dataset=train_data,
        num_workers=CONFIG["data_loader"]["num_workers"],
        batch_size=CONFIG["data_loader"]["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        num_workers=CONFIG["data_loader"]["num_workers"],
        batch_size=CONFIG["data_loader"]["batch_size"],
        shuffle=True,
        pin_memory=True
    )

    swanlab.init(
        project=CONFIG["swanlab"]["project"],
        workspace=CONFIG["swanlab"]["workspace"],
        config={
            "learning_rate": CONFIG["optimizer"]["init_lr"],
            "epoch": CONFIG["epoch"],
            "T_max": CONFIG["scheduler"]["T_max"],
            "eta_min": CONFIG["scheduler"]["eta_min"]
        }
    )

    res_unet = ResUnet().to(CONFIG["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(res_unet.parameters(), lr=CONFIG["optimizer"]["init_lr"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["scheduler"]["T_max"],
        eta_min=CONFIG["scheduler"]["eta_min"]
    )

    lowest_val_loss = float("inf")
    for epoch in range(CONFIG["epoch"]):
        train_loss = train_one_epoch(res_unet, train_loader, optimizer, criterion, epoch, CONFIG["epoch"])
        val_loss, val_rmse, val_rmse_by_sample_count = eval_one_epoch(
            res_unet, val_loader, criterion, epoch, CONFIG["epoch"]
        )
        scheduler.step()
        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            **{
                f"val_rmse_{sample_count}_samples": rmse
                for sample_count, rmse in val_rmse_by_sample_count.items()
            },
        })

        last_checkpoint_path = join(ROOT, "run", "last.pt")
        save_checkpoint(last_checkpoint_path, res_unet)
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_checkpoint_path = join(ROOT, "run", "best.pt")
            save_checkpoint(best_checkpoint_path, res_unet)
        
    swanlab.finish()

def save_checkpoint(
        path: str,
        model: Module
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save(
        {
            "model_state_dict": model.state_dict()
        },
        path
    )
