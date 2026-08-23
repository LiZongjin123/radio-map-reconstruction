from pathlib import Path
from os.path import join
from matplotlib.pyplot import show, subplots
from torch import Tensor, inference_mode, load
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
        shuffle=False,
        pin_memory=True
    )

    best_checkpoint_path = join(ROOT, "run", "best.pt")
    model = load_checkpoint(best_checkpoint_path)

    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(CONFIG["device"])
    with inference_mode():
        outputs = model(inputs)
    ground_truth = targets["gain"][0, 0].cpu().numpy()
    prediction = outputs[0, 0].detach().cpu().clamp(0, 1).numpy()

    building_map = inputs[0, 2].cpu().numpy() > 0.5
    tx_map = inputs[0, 1].cpu().numpy() > 0.5

    prediction[building_map] = 0
    prediction[tx_map] = 1
    
    fig, axes = subplots(1, 2)
    axes[0].imshow(
        ground_truth,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(
        prediction,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    fig.tight_layout()
    show()

    # criterion = RadioMapLoss()
    
    # test_loss, test_rmse = eval_one_epoch(model, test_loader, criterion, 0, 1)

    # print(f"test loss: {test_loss:.6f}")
    # print(f"test rmse: {test_rmse:.6f}")


    
