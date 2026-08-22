from os.path import join
from pathlib import Path
from torch.optim import Adam
from yaml import safe_load
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.model import ResUnet
from torch.utils.data import DataLoader, Subset
from radio_map_reconstruction.train import train_one_epoch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

def test_train_one_epoch():
    train_data = RadioDataset(CONFIG["dataset_path"], "DPM")
    indices = [i for i in range(8)]
    train_subset = Subset(train_data, indices)
    train_loader = DataLoader(
        dataset=train_subset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    res_unet = ResUnet().to(CONFIG["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(res_unet.parameters(), lr=3e-4)

    train_one_epoch(res_unet, train_loader, optimizer, criterion, 0, 1)