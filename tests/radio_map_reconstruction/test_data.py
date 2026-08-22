from pathlib import Path
from pytest import fixture
from radio_map_reconstruction.data import RadioDataset
from radio_map_reconstruction.util import delete_run
from radio_map_reconstruction.split import split

ROOT = Path(__file__).resolve().parents[2]

@fixture(scope="module")
def dataset() -> RadioDataset:
    delete_run()
    split()
    dataset = RadioDataset("test")
    delete_run()
    return dataset

def test_get_item(dataset):
    input, label = dataset[0]
    assert isinstance(label, dict)
    assert input.shape == (4, 256, 256)
    assert label["gain"].shape == label["mask"].shape == (1, 256, 256)