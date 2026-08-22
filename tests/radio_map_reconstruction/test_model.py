from pathlib import Path
from os.path import join
from yaml import safe_load
from radio_map_reconstruction.model import ResUnet

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

def test_create_res_unet():
    model = ResUnet()
    assert model is not None