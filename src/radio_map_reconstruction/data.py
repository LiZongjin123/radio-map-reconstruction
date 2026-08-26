from hashlib import blake2b
from pathlib import Path
from random import Random

from torch import Generator, Tensor, cat, float32, randint, randperm, zeros_like
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2.functional import to_dtype
from yaml import safe_load

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"
with CONFIG_PATH.open(encoding="utf-8") as file:
    CONFIG = safe_load(file)


class RadioDataset(Dataset):

    EVALUATION_SAMPLE_COUNTS = (10, 20, 30, 50, 75, 100, 125, 150, 175, 200)

    def __init__(
        self,
        part: str,
        *,
        dataset_path: str | Path | None = None,
        partition: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if part not in {"train", "val", "test"}:
            raise ValueError("part must be 'train', 'val', or 'test'")

        root = Path(dataset_path or CONFIG["dataset_path"])
        selected_partition = partition or CONFIG["partition"]
        base_seed = CONFIG["seed"] if seed is None else seed
        self.dataset_part = part
        self.base_seed = base_seed
        self.samples = self._discover_split(root, selected_partition, base_seed, part)

    @staticmethod
    def _discover_split(
        root: Path,
        partition: str,
        seed: int,
        part: str,
    ) -> list[tuple[str, str, str]]:
        gain_dir = root / "gain" / partition
        tx_dir = root / "png" / "antennas"
        building_map_dir = root / "png" / "buildings_complete"

        gain_names = {path.name for path in gain_dir.iterdir() if path.is_file()}
        tx_names = {path.name for path in tx_dir.iterdir() if path.is_file()}
        building_map_names = {
            path.name for path in building_map_dir.iterdir() if path.is_file()
        }

        if gain_names != tx_names:
            raise RuntimeError("存在不匹配的Reference Radio Map和Tx Map。")

        samples_by_city_map: dict[int, list[tuple[str, str, str]]] = {}
        for gain_name in sorted(gain_names):
            city_map_name = Path(gain_name).stem.split("_", 1)[0]
            try:
                city_map_id = int(city_map_name)
            except ValueError as error:
                raise RuntimeError(f"无法从文件名识别城市地图编号: {gain_name}") from error

            building_map_name = f"{city_map_id}.png"
            if building_map_name not in building_map_names:
                raise RuntimeError("存在不匹配的Reference Radio Map和Building Map")

            samples_by_city_map.setdefault(city_map_id, []).append(
                (
                    str(gain_dir / gain_name),
                    str(tx_dir / gain_name),
                    str(building_map_dir / building_map_name),
                )
            )

        city_maps = sorted(samples_by_city_map.items())
        Random(seed).shuffle(city_maps)

        train_count = int(len(city_maps) * 0.8)
        val_count = int(len(city_maps) * 0.1)
        boundaries = {
            "train": (0, train_count),
            "val": (train_count, train_count + val_count),
            "test": (train_count + val_count, len(city_maps)),
        }
        start, end = boundaries[part]
        return [sample for _, samples in city_maps[start:end] for sample in samples]

    def __len__(self):
        if self.dataset_part != "train":
            return len(self.samples) * len(self.EVALUATION_SAMPLE_COUNTS)
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        if self.dataset_part == "train":
            sample_index = index
            sample_count = int(randint(10, 201, ()).item())
            generator = None
        else:
            sample_index, sample_count_index = divmod(
                index, len(self.EVALUATION_SAMPLE_COUNTS)
            )
            sample_count = self.EVALUATION_SAMPLE_COUNTS[sample_count_index]

        gain_path, tx_path, building_map_path = self.samples[sample_index]

        if self.dataset_part != "train":
            seed_material = (
                f"{self.base_seed}:{self.dataset_part}:"
                f"{Path(gain_path).name}:{sample_count}"
            ).encode()
            derived_seed = int.from_bytes(
                blake2b(seed_material, digest_size=8).digest(), "big"
            )
            generator = Generator().manual_seed(derived_seed)

        gain = self.__read_image(gain_path)
        tx = self.__read_image(tx_path)
        building_map = self.__read_image(building_map_path)
        gain_mask = self.__random_sample(
            gain, tx, building_map, sample_count, generator
        )

        label = {
            "gain": gain,
            "mask": (tx < 0.5) & (building_map < 0.5)
        }

        return cat((gain * gain_mask, tx, building_map, gain_mask)), label

    def __random_sample(
            self,
            gain: Tensor,
            tx: Tensor,
            building_map: Tensor,
            sample_count: int,
            generator: Generator | None,
    ) -> Tensor:
        valid_area = (tx < 0.5) & (building_map < 0.5)
        valid_indices = valid_area.flatten().nonzero().squeeze(1)

        if valid_indices.numel() < sample_count:
            raise RuntimeError("有效采样点不够")

        selected = valid_indices[
            randperm(valid_indices.numel(), generator=generator)[:sample_count]
        ]

        gain_mask = zeros_like(gain, dtype=float32)
        gain_mask.flatten()[selected] = 1

        return gain_mask

    def __read_image(self, path: str) -> Tensor:
        image = decode_image(
            path, 
            mode=ImageReadMode.GRAY
        )
        return to_dtype(
            image,
            dtype=float32,
            scale=True
        )

