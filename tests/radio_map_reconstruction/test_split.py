from pathlib import Path

from radio_map_reconstruction.data import RadioDataset


def create_dataset(root: Path, city_map_count: int = 10) -> None:
    gain_dir = root / "gain" / "DPM"
    antenna_dir = root / "png" / "antennas"
    building_dir = root / "png" / "buildings_complete"
    gain_dir.mkdir(parents=True)
    antenna_dir.mkdir(parents=True)
    building_dir.mkdir(parents=True)

    # Reverse creation order so discovery cannot accidentally rely on directory order.
    for city_map_id in reversed(range(city_map_count)):
        (building_dir / f"{city_map_id}.png").touch()
        for transmitter_id in (1, 0):
            name = f"{city_map_id}_{transmitter_id}.png"
            (gain_dir / name).touch()
            (antenna_dir / name).touch()


def city_map_ids(dataset: RadioDataset) -> list[int]:
    return [int(Path(sample[0]).stem.split("_", 1)[0]) for sample in dataset.samples]


def test_city_map_split_is_deterministic_and_keeps_groups_together(tmp_path: Path):
    create_dataset(tmp_path)

    splits = {
        part: RadioDataset(part, dataset_path=tmp_path, partition="DPM", seed=17)
        for part in ("train", "val", "test")
    }
    repeated_train = RadioDataset(
        "train", dataset_path=tmp_path, partition="DPM", seed=17
    )

    assert repeated_train.samples == splits["train"].samples
    assert [
        len(set(city_map_ids(splits[part])))
        for part in ("train", "val", "test")
    ] == [8, 1, 1]

    city_maps = {part: set(city_map_ids(dataset)) for part, dataset in splits.items()}
    assert city_maps["train"].isdisjoint(city_maps["val"])
    assert city_maps["train"].isdisjoint(city_maps["test"])
    assert city_maps["val"].isdisjoint(city_maps["test"])

    for dataset in splits.values():
        ids = city_map_ids(dataset)
        assert all(ids.count(city_map_id) == 2 for city_map_id in set(ids))

    assert not list(tmp_path.rglob("*.json"))


def test_city_maps_are_stably_sorted_before_seeded_shuffle(tmp_path: Path):
    create_dataset(tmp_path, city_map_count=12)

    split_from_first_discovery = RadioDataset(
        "test", dataset_path=tmp_path, partition="DPM", seed=123
    )

    # Recreating a file changes directory enumeration order on some file systems.
    recreated_file = tmp_path / "gain" / "DPM" / "2_0.png"
    recreated_file.unlink()
    recreated_file.touch()
    split_from_second_discovery = RadioDataset(
        "test", dataset_path=tmp_path, partition="DPM", seed=123
    )

    assert split_from_second_discovery.samples == split_from_first_discovery.samples
    assert list(dict.fromkeys(city_map_ids(split_from_first_discovery))) == [4, 0]
