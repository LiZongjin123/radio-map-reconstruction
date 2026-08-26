from pathlib import Path

from pytest import fixture
from radio_map_reconstruction.data import RadioDataset
from torch import Tensor, full, uint8, zeros
from torchvision.io import write_png

ROOT = Path(__file__).resolve().parents[2]


def create_image_dataset(
    root: Path, city_map_count: int = 10, transmitter_count: int = 1
) -> None:
    gain_dir = root / "gain" / "DPM"
    antenna_dir = root / "png" / "antennas"
    building_dir = root / "png" / "buildings_complete"
    gain_dir.mkdir(parents=True)
    antenna_dir.mkdir(parents=True)
    building_dir.mkdir(parents=True)

    for city_map_id in range(city_map_count):
        gain = full((1, 16, 16), 128, dtype=uint8)
        transmitter = zeros((1, 16, 16), dtype=gain.dtype)
        building = zeros((1, 16, 16), dtype=gain.dtype)
        transmitter[0, 0, 1] = 255
        building[0, 0, 0] = 255

        write_png(building, str(building_dir / f"{city_map_id}.png"))
        for transmitter_id in range(transmitter_count):
            name = f"{city_map_id}_{transmitter_id}.png"
            write_png(gain, str(gain_dir / name))
            write_png(transmitter, str(antenna_dir / name))


@fixture(scope="module")
def dataset() -> RadioDataset:
    return RadioDataset("test")

def test_get_item(dataset):
    input, label = dataset[0]
    assert isinstance(label, dict)
    assert input.shape == (4, 256, 256)
    assert label["gain"].shape == label["mask"].shape == (1, 256, 256)


def test_training_sampling_is_dynamic_and_stays_in_valid_receiving_area(tmp_path: Path):
    create_image_dataset(tmp_path)
    dataset = RadioDataset(
        "train", dataset_path=tmp_path, partition="DPM", seed=17
    )
    repeated_dataset = RadioDataset(
        "train", dataset_path=tmp_path, partition="DPM", seed=17
    )

    sampling_masks = [dataset[0][0][3] for _ in range(12)]
    repeated_sampling_masks = [repeated_dataset[0][0][3] for _ in range(12)]
    sample_counts = [int(mask.sum().item()) for mask in sampling_masks]

    assert all(10 <= sample_count <= 200 for sample_count in sample_counts)
    assert all(mask[0, 0].item() == 0 for mask in sampling_masks)
    assert all(mask[0, 1].item() == 0 for mask in sampling_masks)
    assert all(set(mask.unique().tolist()) <= {0.0, 1.0} for mask in sampling_masks)
    assert len(set(sample_counts)) > 1
    assert any(not sampling_masks[0].equal(mask) for mask in sampling_masks[1:])
    assert any(
        not first_mask.equal(repeated_mask)
        for first_mask, repeated_mask in zip(
            sampling_masks, repeated_sampling_masks
        )
    )


def test_validation_and_test_cover_reproducible_fixed_sampling_masks(tmp_path: Path):
    create_image_dataset(tmp_path, transmitter_count=2)
    expected_sample_counts = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]

    masks_by_part: dict[str, list[Tensor]] = {}
    for part in ("val", "test"):
        first = RadioDataset(part, dataset_path=tmp_path, partition="DPM", seed=17)
        repeated = RadioDataset(part, dataset_path=tmp_path, partition="DPM", seed=17)

        first_masks = [first[index][0][3] for index in range(len(first))]
        repeated_masks = [repeated[index][0][3] for index in range(len(repeated))]

        assert [int(mask.sum().item()) for mask in first_masks] == (
            expected_sample_counts * 2
        )
        assert all(
            first_mask.equal(repeated_mask)
            for first_mask, repeated_mask in zip(first_masks, repeated_masks)
        )
        assert all(mask[0, 0].item() == 0 for mask in first_masks)
        assert all(mask[0, 1].item() == 0 for mask in first_masks)
        assert any(
            not (smaller.bool() <= larger.bool()).all().item()
            for smaller, larger in zip(first_masks[:9], first_masks[1:10])
        )
        assert any(
            not first_masks[index].equal(first_masks[index + 10])
            for index in range(10)
        )
        masks_by_part[part] = first_masks

    assert any(
        not val_mask.equal(test_mask)
        for val_mask, test_mask in zip(masks_by_part["val"], masks_by_part["test"])
    )

    single_city_root = tmp_path / "single-city"
    create_image_dataset(single_city_root, city_map_count=1)
    first_seed = RadioDataset(
        "test", dataset_path=single_city_root, partition="DPM", seed=17
    )
    second_seed = RadioDataset(
        "test", dataset_path=single_city_root, partition="DPM", seed=18
    )
    assert any(
        not first_seed[index][0][3].equal(second_seed[index][0][3])
        for index in range(10)
    )
