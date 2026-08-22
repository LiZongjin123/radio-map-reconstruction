from pathlib import Path
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
):
    model.train()
    progress = tqdm(dataloader, desc=f"train {epoch + 1}/{epoch_num}", unit="batch", leave=True)
    total_loss = 0
    for batch_index, (inputs, targets) in enumerate(progress):
        optimizer.zero_grad()
        inputs = inputs.to(CONFIG["device"])
        targets = {name: value.to(CONFIG["device"]) for name, value in targets.items()}
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(
            loss = f"{(total_loss / (batch_index + 1)):.6f}"
        )

    return total_loss / len(dataloader)

def train():
    train_data = RadioDataset(CONFIG["dataset_path"], CONFIG["partition"])
    train_loader = DataLoader(
        dataset=train_data,
        num_workers=1,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    res_unet = ResUnet().to(CONFIG["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(res_unet.parameters(), lr=CONFIG["optimizer"]["init_lr"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["scheduler"]["T_max"],
        eta_min=CONFIG["scheduler"]["eta_min"]
    )

    for epoch in range(CONFIG["epoch"]):
        train_one_epoch(res_unet, train_loader, optimizer, criterion, epoch, CONFIG["epoch"])
        scheduler.step()