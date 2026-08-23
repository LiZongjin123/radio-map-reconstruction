from pathlib import Path
from os.path import join
from torch import Tensor, load
from torch.nn import Module
from yaml import safe_load
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.loss import RadioMapLoss
from radio_map_reconstruction.model import ResUnet
from torch.utils.data import DataLoader

from radio_map_reconstruction.train import eval_one_epoch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

def load_checkpoint(path: str) -> Module:
    checkpoint = load(path, map_location=CONFIG["device"])
    model = ResUnet().to(CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model



def eval():
    test_dataset = RadioDataset("test")
    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )

    best_checkpoint_path = join(ROOT, "run", "best.pt")
    model = load_checkpoint(best_checkpoint_path)
    criterion = RadioMapLoss()
    
    test_loss, test_rmse = eval_one_epoch(model, test_loader, criterion, 0, 1)

    print(f"test loss: {test_loss:.6f}")
    print(f"test rmse: {test_rmse:.6f}")


    