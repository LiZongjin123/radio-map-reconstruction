from pathlib import Path
from pytest import fixture
from yaml import safe_load
from radio_map_reconstruction.data import RadioDataset

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"

@fixture(scope="module")
def dataset() -> RadioDataset:
    with open(CONFIG_PATH, encoding="utf-8") as file:
        CONFIG = safe_load(file)

    return RadioDataset(
        root=CONFIG["dataset_path"],
        partition="DPM"
    )

def test_get_item(dataset):
    input, label = dataset[0]
    assert isinstance(label, dict)
    assert input.shape == (4, 256, 256)
    assert label["gain"].shape == label["mask"].shape == (1, 256, 256)
        