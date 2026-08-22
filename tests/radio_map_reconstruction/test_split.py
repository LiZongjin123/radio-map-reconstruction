from pathlib import Path
from json import load
from radio_map_reconstruction.split import split
from os.path import join
from radio_map_reconstruction.util import delete_run

ROOT = Path(__file__).resolve().parents[2]

def test_split():
    delete_run()
    split()
    run_dir = Path(join(ROOT, "run"))
    assert run_dir.exists()
    test_json = join(run_dir, "test.json")
    train_json = join(run_dir, "train.json")
    val_json = join(run_dir, "val.json")
    with open(test_json, encoding="utf-8") as file:
        data = load(file)
        assert isinstance(data, list)
    with open(train_json, encoding="utf-8") as file:
        data = load(file)
        assert isinstance(data, list)
    with open(val_json, encoding="utf-8") as file:
        data = load(file)
        assert isinstance(data, list)
    delete_run()