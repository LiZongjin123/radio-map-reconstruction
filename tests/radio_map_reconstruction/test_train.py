import swanlab
from os.path import join
from pathlib import Path
from torch.optim import Adam
from yaml import safe_load
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.model import ResUnet
from torch.utils.data import DataLoader, Subset
from radio_map_reconstruction.train import eval_one_epoch, save_checkpoint, train_one_epoch
from radio_map_reconstruction.util import delete_run
from radio_map_reconstruction.split import split
from radio_map_reconstruction.eval import eval

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

def test_train_one_epoch():
    delete_run()
    split()
    train_data = RadioDataset("train")
    val_data = RadioDataset("val")
    indices = [i for i in range(8)]
    train_subset = Subset(train_data, indices)
    val_subset = Subset(val_data, indices)

    train_loader = DataLoader(
        dataset=train_subset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_subset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )

    swanlab.init(
        project=CONFIG["swanlab"]["project"],
        workspace=CONFIG["swanlab"]["workspace"],
        tags=["unit test"],
        config={
            "learning_rate": 3e-4,
            "epoch": 1
        }
    )

    res_unet = ResUnet().to(CONFIG["device"])
    criterion = RadioMapLoss()
    optimizer = Adam(res_unet.parameters(), lr=3e-4)

    lowest_val_loss = float("inf")
    train_loss = train_one_epoch(res_unet, train_loader, optimizer, criterion, 0, 1)
    val_loss, val_rmse = eval_one_epoch(res_unet, val_loader, criterion, 0, 1)

    last_checkpoint_path = join(ROOT, "run", "last.pt")
    save_checkpoint(last_checkpoint_path, res_unet)
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        best_checkpoint_path = join(ROOT, "run", "best.pt")
        save_checkpoint(best_checkpoint_path, res_unet)

    swanlab.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_rmse": val_rmse
    })
    swanlab.finish()
    
    delete_run()
