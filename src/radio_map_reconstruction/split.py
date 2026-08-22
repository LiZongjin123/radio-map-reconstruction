import json
from os import listdir
from os.path import join
from pathlib import Path
from random import shuffle
from yaml import safe_load

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = join(ROOT, "config.yml")
with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = safe_load(file)

def split():
    root = CONFIG["dataset_path"]
    partition = CONFIG["partition"]
    gain_dir = join(root, "gain", partition)
    tx_dir = join(root, "png", "antennas")
    building_map_dir = join(root, "png", "buildings_complete")

    gain_names = set(listdir(gain_dir))
    tx_names = set(listdir(tx_dir))
    building_map_names = set(listdir(building_map_dir))

    if len(gain_names) != len(tx_names):
        raise RuntimeError("Reference Radio Map的数量和Tx Map不一致。")
    
    samples = []
    for gain_name in gain_names:
        if gain_name not in tx_names:
            raise RuntimeError("存在不匹配的Reference Radio Map和Tx Map。")
        building_map_index = gain_name.split("_")[0]
        building_map_name = f"{building_map_index}.png"
        if building_map_name not in building_map_names:
            raise RuntimeError("存在不匹配的Reference Radio Map和Building Map")
        gain_path = join(gain_dir, gain_name)
        tx_path = join(tx_dir, gain_name)
        building_path = join(building_map_dir, building_map_name)
        samples.append((gain_path, tx_path, building_path))

    shuffle(samples)
    train_dataset_size = int(len(samples) * 0.8)
    val_dataset_size = int(len(samples) * 0.1)
    train_dataset = samples[:train_dataset_size]
    val_dataset = samples[train_dataset_size:train_dataset_size + val_dataset_size]
    test_dataset = samples[train_dataset_size + val_dataset_size:]

    run_dir = Path(join(ROOT, "run"))
    if run_dir.exists():
        raise RuntimeError("run目录已经存在")
    else:
        run_dir.mkdir()

    train_json = join(run_dir, "train.json")
    val_json = join(run_dir, "val.json")
    test_json = join(run_dir, "test.json")
    
    with open(train_json, "w", encoding="utf-8") as file:
        json.dump(train_dataset, file, ensure_ascii=False, indent=2)
    with open(val_json, "w", encoding="utf-8") as file:
        json.dump(val_dataset, file, ensure_ascii=False, indent=2)
    with open(test_json, "w", encoding="utf-8") as file:
        json.dump(test_dataset, file, ensure_ascii=False, indent=2)